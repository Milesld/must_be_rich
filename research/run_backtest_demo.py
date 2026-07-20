#!/usr/bin/env python3
"""研究模式回测 — 所有参数从 configs/strategy.yaml 读取。

运行方式:
    python research/run_backtest_demo.py                        # 默认配置
    python research/run_backtest_demo.py configs/strategy.yaml  # 指定配置

修改 configs/strategy.yaml 即可调参，无需改代码。
"""

from __future__ import annotations

import importlib
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

# 确保项目根目录在 Python path 中
_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import logging

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo")

# 行业成分股缓存：dynamic 模式下按 industries 签名缓存，避免一次回测内重复联网
_INDUSTRY_UNIVERSE_CACHE: dict[tuple, list[str]] = {}

# 数据拉取进程池并发数。并行跑多个池时（见 scripts/run_optimizers.sh），
# 用环境变量 FETCH_WORKERS 调小，避免 N池×workers 把数据源打爆/限流。
import os as _os
_FETCH_WORKERS = int(_os.environ.get("FETCH_WORKERS", "8"))


# ══════════════════════════════════════════════════════════════
# 1. 配置加载
# ══════════════════════════════════════════════════════════════

def load_config(path: str = "configs/strategy.yaml") -> dict:
    """加载策略配置 YAML 文件。"""
    with open(path) as f:
        return yaml.safe_load(f)


def _enabled_factors(config: dict) -> list[dict]:
    """提取所有 enabled=true 的因子配置。"""
    return [
        {**v, "name": k}
        for k, v in config.get("factors", {}).items()
        if v.get("enabled", False)
    ]


# ══════════════════════════════════════════════════════════════
# 2. 策略定义（参数化，从 config 读取）
# ══════════════════════════════════════════════════════════════

class ConfigDrivenStrategy:
    """从 strategy.yaml 驱动的策略：启用的因子按权重评分，等权持有 top N。

    评分公式: score = Σ (factor_value_rank_pct × reverse × weight)
    其中 reverse=True 表示因子值越低越好（如波动率、PE）。
    """

    # 哪些因子是"越低越好"（反转排名）— 支持前缀匹配覆盖窗口变体
    INVERSE_EXACT = {"pe_ttm", "pb", "peg", "debt_ratio", "amihud_illiq"}
    INVERSE_PREFIXES = ("volatility_", "amplitude_", "atr_")

    def _is_inverse(self, name: str) -> bool:
        return name in self.INVERSE_EXACT or name.startswith(self.INVERSE_PREFIXES)

    def __init__(self, config: dict, raw_data: dict | None = None,
                 overseas_data: dict | None = None,
                 benchmark_index: dict | None = None,
                 monthly_universe: dict | None = None) -> None:
        cfg = config.get("strategy", {})
        self.top_n = cfg.get("top_n", 5)
        self.rebalance_freq = cfg.get("rebalance_frequency", "monthly")
        self.min_shares = cfg.get("min_shares", 100)
        self.optimizer = cfg.get("optimizer", "equal_weight")
        self.factors = _enabled_factors(config)
        total_w = sum(f.get("weight", 0) for f in self.factors)
        if total_w > 0:
            for f in self.factors:
                f["weight"] = f.get("weight", 0) / total_w
        if not self.factors:
            logger.warning("配置中没有启用的因子！")
        self._last_rebalance_period: Any = None
        # 动态股票池：每月宇宙 {anchor_date: [codes]}；fixed 模式为空
        self._monthly_universe = monthly_universe or {}

        # ── 市场状态判断 ──
        regime_cfg = config.get("regime", {})
        self._regime_enabled = regime_cfg.get("enabled", False)
        if self._regime_enabled:
            from core.portfolio.regime import MarketRegimeDetector
            self._regime_detector = MarketRegimeDetector(config=regime_cfg.get("params", {}))
            self._regime_min_position = regime_cfg.get("min_position_ratio", 0.30)
            self._regime_max_position = regime_cfg.get("max_position_ratio", 0.90)
            self._regime_emergency_threshold = regime_cfg.get("emergency_threshold", 0.50)
            self._raw_data = raw_data
            self._overseas_data = overseas_data or {}
            self._benchmark_index = benchmark_index  # 真实宽基指数（沪深300日线）
            logger.info("市场状态判断已启用 (仓位范围: %.0f%%~%.0f%%)",
                        self._regime_min_position * 100, self._regime_max_position * 100)
        else:
            self._regime_detector = None
            self._regime_min_position = 1.0
            self._regime_max_position = 1.0
            self._raw_data = None
            self._overseas_data = {}
            self._benchmark_index = None

        # 跟踪当前实际仓位（用于判断是否需要减仓）
        self._current_regime: Any = None

    def on_bar(
        self,
        trade_date: date,
        features: pd.DataFrame,
        positions: dict[str, int],
        cash: float,
        daily_data: dict[str, dict],
    ) -> list:
        intents: list = []
        from core.backtest.engine import TradeIntent

        # ── 市场状态判断（每个交易日都检测，不仅调仓日）───
        suggested_pos_ratio = 1.0
        regime_info = ""
        if self._regime_detector is not None:
            market_df = self._build_market_proxy(trade_date)
            if market_df is not None and len(market_df) > 200:
                self._current_regime = self._regime_detector.detect(
                    trade_date, market_df, overseas_data=self._overseas_data)
                raw_ratio = self._current_regime.suggested_position_ratio
                suggested_pos_ratio = max(self._regime_min_position,
                                          min(self._regime_max_position, raw_ratio))
                regime_info = f" | 市场状态: {self._current_regime.regime_label} (建议仓位: {suggested_pos_ratio:.0%})"

        if not self._should_rebalance(trade_date):
            # 非调仓日也检查：如果市场急转直下，可能需要紧急减仓
            if self._regime_detector is not None and suggested_pos_ratio < self._regime_emergency_threshold:
                # 极度悲观 → 卖出部分持仓
                pos_value = sum(
                    float(daily_data.get(c, {}).get("close", 0)) * s
                    for c, s in positions.items()
                )
                total_value = cash + pos_value
                target_pos_value = total_value * suggested_pos_ratio
                if pos_value > target_pos_value * 1.1:  # 超过目标10%以上才减仓
                    reduce_ratio = target_pos_value / max(pos_value, 1)
                    for code, shares in list(positions.items()):
                        if shares <= 0:
                            continue
                        sell_shares = int(shares * (1 - reduce_ratio))
                        sell_shares = (sell_shares // 100) * 100
                        if sell_shares >= 100:
                            price = float(daily_data.get(code, {}).get("close", 0))
                            if price > 0:
                                intents.append(TradeIntent(
                                    signal_id=f"regime_cut_{code}_{trade_date}",
                                    code=code, side="sell", price=price, shares=sell_shares,
                                ))
            return intents

        if features.empty:
            return intents

        # ── 动态股票池：调仓日只在当月候选宇宙内打分选股 ──
        universe = _universe_for_date(self._monthly_universe, trade_date)
        scoring_features = features
        if universe is not None:
            in_uni = [c for c in features.index if c in set(universe)]
            scoring_features = features.loc[in_uni]
            if scoring_features.empty:
                logger.warning("调仓日 %s 当月宇宙内无可打分标的，跳过", trade_date)
                return intents

        scores = self._score_stocks(scoring_features)
        if scores.empty:
            return intents

        target_codes = set(scores.nlargest(self.top_n).index)

        # ── 用建议仓位比例缩放可投入资金 ──
        pos_value = sum(
            float(daily_data.get(c, {}).get("close", 0)) * s
            for c, s in positions.items()
        )
        total_value = cash + pos_value
        target_pos_value = total_value * suggested_pos_ratio

        # ★ 等权重置：目标池每只股票占等额资金（含已持有+新买入）
        per_stock_target_value = target_pos_value / self.top_n

        # 卖出不在目标池的全部持仓
        for code, shares in list(positions.items()):
            if shares > 0 and code not in target_codes:
                price = float(daily_data.get(code, {}).get("close", 0))
                if price > 0:
                    intents.append(TradeIntent(
                        signal_id=f"sell_{code}_{trade_date}",
                        code=code, side="sell", price=price, shares=shares,
                    ))

        # 对目标池中的每只股票，调至目标权重
        for code in target_codes:
            price = float(daily_data.get(code, {}).get("close", 0))
            if price <= 0:
                continue
            current_shares = positions.get(code, 0)
            current_value = current_shares * price
            target_shares_total = int(per_stock_target_value / price)
            target_shares_total = (target_shares_total // 100) * 100

            if current_value > per_stock_target_value * 1.15:
                # 超配 → 卖出超额部分
                excess_value = current_value - per_stock_target_value
                sell_shares = int(excess_value / price)
                sell_shares = (sell_shares // 100) * 100
                if sell_shares >= self.min_shares:
                    intents.append(TradeIntent(
                        signal_id=f"rebal_sell_{code}_{trade_date}",
                        code=code, side="sell", price=price, shares=sell_shares,
                    ))
            elif target_shares_total > current_shares:
                # 低配或未持有 → 买入差额
                buy_shares = target_shares_total - current_shares
                buy_shares = (buy_shares // 100) * 100
                if buy_shares >= self.min_shares:
                    intents.append(TradeIntent(
                        signal_id=f"rebal_buy_{code}_{trade_date}",
                        code=code, side="buy", price=price, shares=buy_shares,
                    ))

        if regime_info:
            logger.info("调仓日 %s%s", trade_date, regime_info)

        return intents

    def _build_market_proxy(self, as_of_date: date) -> pd.DataFrame | None:
        """构建市场状态判断用的指数序列。

        优先用真实宽基指数（沪深300）做趋势/波动；只有当宽基数据不可用时，
        才回退到"池内等权伪指数"（循环论证的旧行为，标注为 fallback）。
        只取 as_of_date 之前的数据，避免前视。
        """
        # ── 优先：真实宽基指数（沪深300） ──
        if self._benchmark_index and "csi300" in self._benchmark_index:
            bench = self._benchmark_index["csi300"]
            df = bench[bench.index <= as_of_date]
            if len(df) >= 200:
                # 宽度维度用中证全指（若有），否则与趋势同源
                return df.copy()

        # ── Fallback：池内等权伪指数（循环论证，仅在宽基缺失时启用） ──
        if self._raw_data is None:
            return None
        from core.common.calendar import get_calendar
        cal = get_calendar()
        lookback = cal.get_prev_n_trading_days(as_of_date, 252 * 3)
        lookback = [d for d in lookback if d in self._raw_data and d <= as_of_date]
        if len(lookback) < 200:
            return None
        rows = []
        for td in lookback:
            closes = [v["close"] for v in self._raw_data.get(td, {}).values() if v.get("close", 0) > 0]
            if not closes:
                continue
            rows.append({"date": td, "close": sum(closes) / len(closes)})
        if not rows:
            return None
        logger.debug("宽基指数不可用，市场状态回退到池内等权伪指数（%s）", as_of_date)
        return pd.DataFrame(rows).set_index("date").sort_index()

    def _should_rebalance(self, trade_date: date) -> bool:
        from core.common.calendar import get_calendar
        cal = get_calendar()
        prev = cal.prev_trading_day(trade_date)

        if self.rebalance_freq == "monthly":
            should = prev.month != trade_date.month
        elif self.rebalance_freq == "weekly":
            should = prev.isocalendar()[1] != trade_date.isocalendar()[1]
        else:  # daily
            should = True

        if should:
            self._last_rebalance_period = (
                trade_date.month if self.rebalance_freq == "monthly"
                else trade_date.isocalendar()[1]
            )
        return should

    def _score_stocks(self, features: pd.DataFrame) -> pd.Series:
        """根据 config 中启用的因子及其 weight 计算综合评分。"""
        scores = pd.Series(0.0, index=features.index)

        for fg in self.factors:
            name = fg["name"]
            weight = fg.get("weight", 0.0)
            if weight <= 0 or name not in features.columns:
                continue

            vals = features[name].dropna()
            if len(vals) == 0:
                continue

            rank = vals.rank(pct=True)
            if self._is_inverse(name):
                rank = 1.0 - rank

            common = scores.index.intersection(rank.index)
            scores.loc[common] = scores.loc[common] + rank.loc[common].astype(float) * weight

        return scores


# ══════════════════════════════════════════════════════════════
# 3. 数据加载
# ══════════════════════════════════════════════════════════════



def _provider(config: dict) -> str:
    return config.get("data_source", {}).get("provider", "sina")


def _get_codes(config: dict) -> list[str]:
    """确定回测要拉取的全部代码。

    - dynamic 模式：取所有月份候选宇宙的并集（行业成分股），一次性拉全，
      回测中再按月切片（见 _build_monthly_universe）。行业成分按 industries
      签名缓存，避免一次回测内多次联网拉取。
      · provider=westock：industries 视为 westock pt 板块码，走 westock_industry_cons。
      · 其它 provider：industries 视为 akshare 行业中文名，走 build_industry_universe。
    - fixed 模式（缺省）：用 data_source.codes 固定列表（向后兼容）。
    """
    ds = config.get("data_source", {})
    uni = ds.get("universe", {})
    if isinstance(uni, dict) and uni.get("mode") == "dynamic":
        industries = uni.get("industries", [])
        if not industries:
            logger.error("dynamic 宇宙模式但未配置 data_source.universe.industries！")
            return []
        cache_key = (_provider(config),) + tuple(industries)
        if cache_key in _INDUSTRY_UNIVERSE_CACHE:
            return _INDUSTRY_UNIVERSE_CACHE[cache_key]
        if _provider(config) == "westock":
            from research.westock_source import westock_industry_cons
            codes = westock_industry_cons(industries)
        else:
            from research.universe import build_industry_universe
            codes = build_industry_universe(industries)
        if not codes:
            logger.error("行业成分股拉取失败，候选宇宙为空！")
        _INDUSTRY_UNIVERSE_CACHE[cache_key] = codes
        return codes

    codes = ds.get("codes", [])
    if not codes:
        logger.error("configs/strategy.yaml 中 data_source.codes 为空！请在配置文件中设置股票池。")
        return []
    return list(dict.fromkeys(codes))  # 去重保序


def _is_dynamic_universe(config: dict) -> bool:
    uni = config.get("data_source", {}).get("universe", {})
    return isinstance(uni, dict) and uni.get("mode") == "dynamic"


def _build_monthly_universe(raw_data: dict[date, dict], config: dict) -> dict[date, list[str]]:
    """预计算每个调仓月的候选宇宙 {month_anchor_date: [codes]}。

    dynamic 模式下每月初用 PIT 信息重建宇宙；fixed 模式返回空 dict（策略
    回退到全量 codes）。month_anchor 取每个自然月在回测数据中的首个交易日。
    """
    if not _is_dynamic_universe(config):
        return {}

    from research.universe import filter_universe_pit

    uni = config["data_source"]["universe"]
    candidates = _get_codes(config)
    if not candidates:
        return {}

    pool_size = uni.get("pool_size", 30)
    min_listing_months = uni.get("min_listing_months", 12)
    min_avg_amount = float(uni.get("min_avg_amount", 200_000_000))

    bt = config["backtest"]
    bt_start = _parse_date(bt["start_date"])
    bt_end = _parse_date(bt["end_date"])
    if bt.get("validate_end"):
        bt_end = max(bt_end, _parse_date(bt["validate_end"]))

    # 真实上市日（用于次新剔除，替代"数据内交易日数"的不准近似）
    if _provider(config) == "westock":
        from research.westock_source import westock_listing_dates
        listing_dates = westock_listing_dates(candidates)
    else:
        from research.universe import fetch_listing_dates
        listing_dates = fetch_listing_dates(candidates)

    # 每个自然月的首个交易日（落在回测区间内）作为宇宙锚点
    trade_dates = sorted(d for d in raw_data if bt_start <= d <= bt_end)
    month_anchors: dict[tuple[int, int], date] = {}
    for d in trade_dates:
        key = (d.year, d.month)
        if key not in month_anchors:
            month_anchors[key] = d

    monthly: dict[date, list[str]] = {}
    for anchor in sorted(month_anchors.values()):
        universe = filter_universe_pit(
            candidates, anchor, raw_data,
            min_listing_months=min_listing_months,
            min_avg_amount=min_avg_amount,
            pool_size=pool_size,
            listing_dates=listing_dates,
        )
        monthly[anchor] = universe
        logger.info("当月宇宙 %s: %d 只", anchor, len(universe))
    return monthly


def _universe_for_date(monthly_universe: dict[date, list[str]] | None,
                       trade_date: date) -> list[str] | None:
    """取 trade_date 所属月份（≤ trade_date 的最近锚点）的宇宙。"""
    if not monthly_universe:
        return None
    anchors = sorted(d for d in monthly_universe if d <= trade_date)
    if not anchors:
        return None
    return monthly_universe[anchors[-1]]



def _fetch_one_sina(code: str, start: date, end: date) -> "pd.DataFrame | None":
    """拉取单只股票的新浪日K线（供进程池调用，须为模块顶层函数以可 pickle）。"""
    import akshare as ak
    import pandas as pd
    try:
        prefix = "sh" if code.startswith("6") else "sz"
        raw = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
        if raw is None or len(raw) == 0:
            return None
        df = raw.copy()
        df["code"] = code
        df["trade_date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if len(df) == 0:
            return None
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        return df
    except Exception:
        return None


def _fetch_sina(codes: list[str], start: date, end: date,
                max_workers: int | None = None) -> "pd.DataFrame | None":
    """从新浪接口并发拉取日K线（进程池：akshare 原生 HTTP 栈多线程会崩溃）。"""
    return _fetch_parallel(_fetch_one_sina, codes, start, end, max_workers or _FETCH_WORKERS)


def _fetch_parallel(worker, codes: list[str], start: date, end: date,
                    max_workers: int) -> "pd.DataFrame | None":
    """用进程池并发执行单只拉取函数，汇总为一个 DataFrame。

    用进程池而非线程池：akshare 底层 HTTP 栈（PartitionAlloc）不支持多线程
    并发初始化，会触发 native FATAL 崩溃。每个子进程有独立内存分配器，规避此问题。
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    frames = []
    done = 0
    total = len(codes)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, c, start, end): c for c in codes}
        for fut in as_completed(futures):
            try:
                df = fut.result()
            except Exception:
                df = None
            if df is not None:
                frames.append(df)
            done += 1
            if done % 50 == 0 or done == total:
                logger.info("  行情拉取进度: %d/%d", done, total)

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["code", "trade_date"])


def _fetch_one_eastmoney(code: str, start: date, end: date) -> "pd.DataFrame | None":
    """拉取单只股票的东方财富日K线（供进程池调用，须为模块顶层函数）。"""
    import akshare as ak
    import pandas as pd
    try:
        raw = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
        if raw is None or len(raw) == 0:
            return None
        df = raw.copy()
        df["code"] = code
        df["trade_date"] = pd.to_datetime(df["日期"]).dt.date
        col_map = {"开盘": "open", "最高": "high", "最低": "low",
                   "收盘": "close", "成交量": "volume", "换手率": "turnover"}
        for cn, en in col_map.items():
            if cn in df.columns:
                df[en] = df[cn].astype(float)
        return df
    except Exception:
        return None


def _fetch_eastmoney(codes: list[str], start: date, end: date,
                     max_workers: int | None = None) -> "pd.DataFrame | None":
    """从东方财富接口并发拉取（备用，进程池）—— 提供 OHLCV + 换手率。"""
    return _fetch_parallel(_fetch_one_eastmoney, codes, start, end, max_workers or _FETCH_WORKERS)


def _supplement_turnover(raw_data: dict[date, dict], codes: list[str],
                          start: date, end: date, max_workers: int | None = None) -> None:
    """从东方财富并发补充真实换手率数据，写入 raw_data 的 'turnover' 字段。

    复用 _fetch_parallel（进程池）：东财单只拉取已含换手率列，直接取用。
    """
    logger.info("正在补充换手率数据（东方财富，进程池）...")
    df = _fetch_parallel(_fetch_one_eastmoney, codes, start, end, max_workers or _FETCH_WORKERS)
    if df is None or "turnover" not in df.columns:
        logger.info("换手率补充完成: 0 条（无数据）")
        return
    count = 0
    for _, row in df.iterrows():
        code = row.get("code")
        td_raw = row.get("trade_date")
        if td_raw in raw_data and code in raw_data[td_raw]:
            raw_data[td_raw][code]["turnover"] = float(row.get("turnover", 0) or 0)
            count += 1
    logger.info("换手率补充完成: %d 条", count)


def _load_real_data(config: dict) -> tuple[dict[date, dict], str]:
    """从配置的数据源拉取数据。"""
    # 计算实际需要的 lookback：取所有启用因子的最大窗口，加 30 天余量
    min_needed = _calc_min_lookback(config)
    cfg_lookback = config["data_source"].get("lookback_days", 400)
    lookback = max(cfg_lookback, min_needed)
    if lookback > cfg_lookback:
        logger.warning(
            "lookback_days %d 不足以覆盖最长因子窗口（需要 %d 天），已自动扩大到 %d",
            cfg_lookback, min_needed, lookback,
        )
    start = _parse_date(config["backtest"]["start_date"]) - timedelta(days=lookback)
    end = _parse_date(config["backtest"]["end_date"])
    # ★ 如果配置了 validate_end 且在 end_date 之后，延长数据拉取覆盖验证期
    bt_cfg = config.get("backtest", {})
    if bt_cfg.get("validate_end") and _parse_date(bt_cfg["validate_end"]) > end:
        end = _parse_date(bt_cfg["validate_end"])
    codes = _get_codes(config)
    provider = config.get("data_source", {}).get("provider", "sina")

    logger.info("正在从 %s 拉取 %d 只股票 (%s ~ %s)...", provider, len(codes), start, end)

    # ── westock 数据源（绕开 akshare 限流）──
    if provider == "westock":
        from research.westock_source import westock_kline
        # ★ warehouse-first（路线图第 6 阶段）：本地数据仓覆盖的代码直接读
        #   parquet（快且不受限流），只对未覆盖的代码联网。仓库由
        #   scripts/update_data.py 每日增量维护。
        from research import warehouse as _wh
        wh_meta = _wh.load_meta()
        local_codes = [c for c in codes if _wh.warehouse_covers(c, start, end, wh_meta)]
        remote_codes = [c for c in codes if c not in set(local_codes)]
        frames = []
        if local_codes:
            local_df = _wh.read_kline_many(local_codes, start, end)
            if local_df is not None:
                frames.append(local_df)
            logger.info("westock 数据仓命中 %d/%d 只（联网仅 %d 只）",
                        len(local_codes), len(codes), len(remote_codes))
        if remote_codes:
            try:
                remote_df = westock_kline(remote_codes, start, end)
            except Exception as ex:
                if not frames:
                    logger.warning("westock 数据拉取失败: %s", ex)
                    return {}, str(ex)
                logger.warning("westock 联网部分失败（%s），仅用数据仓的 %d 只",
                               ex, len(local_codes))
                remote_df = None
            if remote_df is not None and len(remote_df) > 0:
                frames.append(remote_df)
        if not frames:
            return {}, "westock 返回空数据"
        raw_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        out: dict[date, dict] = {}
        for _, row in raw_df.iterrows():
            td = row["trade_date"]
            code = str(row["code"])
            close = float(row.get("close", 0) or 0)
            if close <= 0:
                continue
            vol = float(row.get("volume", 0) or 0)
            out.setdefault(td, {})[code] = {
                "open": float(row.get("open", close) or close),
                "high": float(row.get("high", close) or close),
                "low": float(row.get("low", close) or close),
                "close": close,
                "pre_close": close * 0.99,  # 临时，_fix_pre_close 修正
                "volume": vol,
                "amount": float(row.get("amount", 0) or 0) or close * vol,
                "turnover": float(row.get("turnover", 0) or 0),  # westock 自带换手率
                "is_st": False,
                "is_suspended": False,
            }
        # westock 自带换手率，无需 _supplement_turnover；仅修正 pre_close
        _fix_pre_close(out)
        return out, "ok"

    logger.info("  (进程池并发拉取，规避 akshare 多线程原生崩溃)")

    # 直接同步调用进程池拉取：进程池自身管理子进程生命周期，不再套线程+join。
    # （在 daemon 线程里再起 ProcessPoolExecutor 在 macOS spawn 模式下易死锁）
    try:
        if provider == "eastmoney":
            raw_df = _fetch_eastmoney(codes, start, end)
        else:
            raw_df = _fetch_sina(codes, start, end)
    except Exception as e:
        logger.warning("%s 数据拉取失败: %s", provider, e)
        return {}, str(e)

    if raw_df is None or len(raw_df) == 0:
        logger.warning("%s 数据拉取失败: 返回空数据", provider)
        return {}, "返回空数据"

    out: dict[date, dict] = {}
    for _, row in raw_df.iterrows():
        td = row.get("trade_date")
        if isinstance(td, pd.Timestamp):
            td = td.date()
        if isinstance(td, str):
            try:
                td = date.fromisoformat(td)
            except ValueError:
                continue
        if td is None:
            continue

        code = str(row["code"])
        close = float(row.get("close", 0))
        if close <= 0:
            continue
        # ★ 取东方财富的换手率（如有），否则留 0
        turnover_val = float(row.get("turnover", 0) or 0)
        pre_val = row.get("pre_close")
        if pre_val is not None:
            pre = float(pre_val)
        else:
            pre = close * 0.99  # 临时估算，后续 _fix_pre_close 会修正

        out.setdefault(td, {})[code] = {
            "open": float(row.get("open", close)),
            "high": float(row.get("high", close)),
            "low": float(row.get("low", close)),
            "close": close,
            "pre_close": max(pre, 0.01),
            "volume": float(row.get("volume", 0)),
            "amount": close * float(row.get("volume", 0)),
            "turnover": turnover_val,
            "is_st": False,
            "is_suspended": False,
        }
    # ★ 补充真实换手率（东方财富）并修正 pre_close
    _supplement_turnover(out, codes, start, end)
    _fix_pre_close(out)
    return out, "ok"


# ══════════════════════════════════════════════════════════════
# 3.5 基本面数据加载
# ══════════════════════════════════════════════════════════════

def _financials_as_of(fin_series: list[dict] | None, trade_date: date) -> dict[str, float]:
    """PIT 取数：从财报时间序列取 announce_date < trade_date 的最近一期因子值。

    Args:
        fin_series: [{announce_date: date, roe_ttm:.., revenue_yoy:..}, ...]（按披露日升序）。
        trade_date: 当前回测日。

    Returns:
        {factor: value}（不含 announce_date/end_date）。无已披露财报 → {}（因子中性）。
    严格 PIT：用 < 而非 <=，确保只用"当日之前已披露"的财报，根除前视偏差。
    """
    if not fin_series:
        return {}
    chosen = None
    for rec in fin_series:  # 升序，取最后一个 announce_date < trade_date 的
        if rec["announce_date"] < trade_date:
            chosen = rec
        else:
            break
    if chosen is None:
        return {}
    return {k: v for k, v in chosen.items() if k not in ("announce_date", "end_date")}


def _load_financials(codes: list[str]) -> dict[str, dict[str, float]]:
    """拉取基本面数据：使用东方财富个股财务摘要接口。

    数据项：净利润、营收增速、净利增速、ROE、净资产、每股收益、
           每股净资产、资产负债率、流动比率、速动比率。

    Returns:
        {code: {roe_ttm: x, revenue_yoy: y, net_profit_yoy: z, debt_ratio: d, ...}}
    """
    import akshare as ak

    result: dict[str, dict[str, float]] = {}
    logger.info("正在拉取 %d 只股票的基本面数据（东方财富接口）...", len(codes))

    for i, code in enumerate(codes):
        try:
            df = ak.stock_financial_abstract_ths(symbol=code, indicator="按年度")
            if df is None or len(df) == 0:
                continue

            latest = df.iloc[-1]  # 最新一年
            prev_year = df.iloc[-2] if len(df) >= 2 else None

            fundamentals: dict[str, float] = {}

            # 从同花顺接口映射到内部因子名
            col_map_direct = {
                "净利润": "net_profit_latest",
                "净利润同比增长率": "net_profit_yoy",
                "营业总收入": "revenue_latest",
                "营业总收入同比增长率": "revenue_yoy",
                "资产负债率": "debt_ratio",
                "流动比率": "current_ratio",
                "速动比率": "quick_ratio",
                "每股净资产": "bps",
            }
            for col, factor_name in col_map_direct.items():
                if col in latest.index:
                    raw = str(latest[col]).replace("%", "").replace("亿", "").replace(",", "")
                    try:
                        v = float(raw)
                        # 百分比转小数（增长率、负债率等 % 值）
                        if "%" in str(latest[col]):
                            v = v / 100.0
                        # 净利润可能是亿为单位 → 保持原值
                        fundamentals[factor_name] = v
                    except ValueError:
                        pass

            # 计算 ROE = 净利润 / 净资产（每股净资产 × 总股本近似）
            # 简化：直接用 growth 和 quality 代理
            net_profit_raw = str(latest.get("净利润", "0")).replace("亿", "").replace(",", "")
            try:
                net_profit_val = float(net_profit_raw)
                # 从 bps * 股本 反推净资产太复杂，用 trend 代理 ROE
                # ROE ≈ 净利增速 / 营收增速 的质量调整值
                rev_yoy = fundamentals.get("revenue_yoy", 0)
                np_yoy = fundamentals.get("net_profit_yoy", 0)
                if rev_yoy and rev_yoy > 0:
                    fundamentals["roe_ttm"] = np_yoy / rev_yoy if abs(rev_yoy) > 0.01 else 0.15
                else:
                    fundamentals["roe_ttm"] = 0.10  # 默认中性
            except (ValueError, ZeroDivisionError):
                fundamentals["roe_ttm"] = 0.10

            # ROA ≈ 净利增速 * 0.6（粗估）
            fundamentals["roa_ttm"] = fundamentals.get("net_profit_yoy", 0.05) * 0.6

            if fundamentals:
                result[code] = fundamentals

            if (i + 1) % 20 == 0:
                logger.info("  基本面数据: %d/%d", i + 1, len(codes))
        except Exception:
            continue

    logger.info("基本面数据加载完成: %d 只股票", len(result))
    return result


# ══════════════════════════════════════════════════════════════
# 3.6 海外市场数据加载
# ══════════════════════════════════════════════════════════════

def _load_overseas_data(start: date, end: date) -> dict[date, dict[str, float]]:
    """加载海外市场指数日线数据（新浪接口）。

    加载 SPX、NASDAQ 和 HSI 指数，计算每日涨跌幅。
    A50 期货数据通过 US+HK 合成代理。

    Returns:
        {date: {spx_ret, ndx_ret, hsi_ret, a50_proxy_ret}}
    """
    import akshare as ak

    out: dict[date, dict[str, float]] = {}
    pad_start = start - timedelta(days=10)  # 多拉几天覆盖周末

    # ── 美股指数（新浪）──
    for symbol, key in [(".INX", "spx_ret"), (".IXIC", "ndx_ret")]:
        try:
            raw = ak.index_us_stock_sina(symbol=symbol)
            if raw is not None and len(raw) > 0:
                df = raw.copy()
                df["trade_date"] = pd.to_datetime(df["date"]).dt.date
                df = df[(df["trade_date"] >= pad_start) & (df["trade_date"] <= end)]
                df = df.sort_values("trade_date")
                df["ret"] = df["close"].astype(float).pct_change()
                for _, row in df.iterrows():
                    td = row["trade_date"]
                    if td >= start and not np.isnan(row["ret"]):
                        out.setdefault(td, {})[key] = float(row["ret"])
        except Exception:
            logger.debug("海外数据 %s 加载失败，跳过", key)

    # ── 港股恒生指数（新浪）──
    try:
        raw = ak.stock_hk_index_daily_sina(symbol="HSI")
        if raw is not None and len(raw) > 0:
            df = raw.copy()
            df["trade_date"] = pd.to_datetime(df["date"]).dt.date
            df = df[(df["trade_date"] >= pad_start) & (df["trade_date"] <= end)]
            df = df.sort_values("trade_date")
            df["ret"] = df["close"].astype(float).pct_change()
            for _, row in df.iterrows():
                td = row["trade_date"]
                if td >= start and not np.isnan(row["ret"]):
                    out.setdefault(td, {})["hsi_ret"] = float(row["ret"])
    except Exception:
        logger.debug("港股指数数据加载失败，跳过")

    # ── A50 期货代理（US+HK 合成）──
    for td in out:
        spx = out[td].get("spx_ret", 0)
        ndx = out[td].get("ndx_ret", 0)
        hsi = out[td].get("hsi_ret", 0)
        # A50 ≈ 0.4*美股科技 + 0.3*港股 + 0.3*残差
        out[td]["a50_proxy_ret"] = 0.4 * ndx + 0.3 * hsi + 0.15 * spx

    logger.info("海外市场数据加载完成: %d 个交易日", len(out))
    return out


# ══════════════════════════════════════════════════════════════
# 3.65 宽基指数加载（真实市场基准，用于市场状态判断）
# ══════════════════════════════════════════════════════════════

def _load_benchmark_index(start: date, end: date, provider: str = "sina") -> dict[str, "pd.DataFrame"]:
    """加载真实宽基指数日线，替代"池内等权伪指数"做市场状态判断。

    - 沪深300 (000300) → regime 的 trend/volatility（市场趋势与波动）
    - 中证全指 (000985) → regime 的 breadth 宽度近似（覆盖最广）

    provider=westock 走 westock kline（指数也用 kline）；否则用 akshare
    stock_zh_index_daily。多取 3 年历史以满足 regime 对 200 日均线、
    252 日滚动波动率的回溯需求。

    Returns:
        {"csi300": DataFrame(index=date, columns=[close]),
         "csi_all": DataFrame(...)}，加载失败的键缺省。
    """
    pad_start = start - timedelta(days=365 * 3 + 60)  # regime 需要长回溯
    out: dict[str, pd.DataFrame] = {}

    if provider == "westock":
        from research.westock_source import westock_kline
        # 指数用显式 westock 前缀码（沪深300/中证全指均为 sh 前缀，
        # 不能靠 to_westock_code 的个股前缀规则推断）
        for wcode, key in [("sh000300", "csi300"), ("sh000985", "csi_all")]:
            try:
                df = westock_kline([wcode], pad_start, end, batch=1)
            except Exception as ex:
                logger.warning("westock 宽基指数 %s 失败: %s", key, ex)
                continue
            if df is None or len(df) == 0:
                logger.warning("westock 宽基指数 %s 返回空", key)
                continue
            df = df.sort_values("trade_date")
            out[key] = df[["trade_date", "close"]].set_index("trade_date")
            logger.info("宽基指数 %s 加载完成: %d 条（westock）", key, len(df))
        return out

    import akshare as ak

    for symbol, key in [("sh000300", "csi300"), ("sh000985", "csi_all")]:
        try:
            raw = ak.stock_zh_index_daily(symbol=symbol)
            if raw is None or len(raw) == 0:
                logger.warning("宽基指数 %s 返回空", symbol)
                continue
            df = raw.copy()
            df["trade_date"] = pd.to_datetime(df["date"]).dt.date
            df = df[(df["trade_date"] >= pad_start) & (df["trade_date"] <= end)]
            df = df.sort_values("trade_date")
            df["close"] = df["close"].astype(float)
            out[key] = df[["trade_date", "close"]].set_index("trade_date")
            logger.info("宽基指数 %s 加载完成: %d 条", key, len(df))
        except Exception as e:
            logger.warning("宽基指数 %s 加载失败: %s", symbol, e)

    return out


# ══════════════════════════════════════════════════════════════
# 3.7 全市场情绪数据（从池内OHLCV推算）
# ══════════════════════════════════════════════════════════════

def _build_market_wide_from_pool(raw_data: dict[date, dict]) -> dict[date, dict[str, float]]:
    """从已有股票池OHLCV数据推算全市场情绪代理指标。

    因为回测中无法逐日拉取5000只全市场数据，用池内48只股票的
    涨跌停/连板统计作为市场情绪的近似代理。

    Returns:
        {date: {limit_up_count, limit_down_count, limit_up_ratio, board_break_ratio, chain_height}}
    """
    out: dict[date, dict[str, float]] = {}
    # 跟踪连续涨停（用于计算连板高度）
    chain_tracker: dict[str, int] = {}  # {code: consecutive_lu_days}

    for td in sorted(raw_data.keys()):
        day_data = raw_data[td]
        advance = 0
        decline = 0
        limit_up_codes: set[str] = set()
        limit_down_count = 0
        board_break_count = 0

        for code, row in day_data.items():
            close = float(row.get("close", 0))
            pre_close = float(row.get("pre_close", 0))
            high = float(row.get("high", 0))
            low = float(row.get("low", 0))
            if close <= 0 or pre_close <= 0:
                continue

            chg_pct = (close - pre_close) / pre_close

            if chg_pct > 0:
                advance += 1
            elif chg_pct < 0:
                decline += 1

            # 涨停判断（A股 ±10%，科创 ±20%）
            is_kcb = code.startswith("688")
            lu_limit = 0.198 if is_kcb else 0.098
            ld_limit = -0.198 if is_kcb else -0.098

            if chg_pct >= lu_limit - 0.002:  # 允许微小舍入
                limit_up_codes.add(code)
            elif chg_pct <= ld_limit + 0.002:
                limit_down_count += 1

            # 炸板：盘中曾近涨停但收盘回落超过2%
            if high / pre_close - 1 >= lu_limit - 0.005 and chg_pct < lu_limit - 0.02:
                board_break_count += 1

        # 连板高度追踪
        for code in list(chain_tracker.keys()):
            if code not in limit_up_codes:
                del chain_tracker[code]
        for code in limit_up_codes:
            chain_tracker[code] = chain_tracker.get(code, 0) + 1
        chain_height = max(chain_tracker.values()) if chain_tracker else 0

        total = advance + decline
        limit_up_count = len(limit_up_codes)
        limit_up_ratio_val = limit_up_count / total if total > 0 else 0.0
        bb_ratio = board_break_count / max(limit_up_count + board_break_count, 1)

        out[td] = {
            "limit_up_count": float(limit_up_count),
            "limit_down_count": float(limit_down_count),
            "limit_up_ratio": float(limit_up_ratio_val),
            "board_break_ratio": float(bb_ratio),
            "limit_up_chain_height": float(chain_height),
        }

    return out


def _fix_pre_close(raw_data: dict[date, dict]) -> None:
    """用上一交易日收盘价修正 pre_close（替代 close*0.99 的临时估算）。"""
    # 对每只股票收集所有日期 → 排序 → 前一日的 close = 当日的 pre_close
    code_dates: dict[str, list[tuple[date, float]]] = {}
    for td, day_data in raw_data.items():
        for code, fields in day_data.items():
            code_dates.setdefault(code, []).append((td, fields.get("close", 0)))
    for code, pairs in code_dates.items():
        pairs.sort(key=lambda x: x[0])
        for i in range(1, len(pairs)):
            td = pairs[i][0]
            prev_close = pairs[i-1][1]
            if prev_close > 0 and td in raw_data and code in raw_data[td]:
                raw_data[td][code]["pre_close"] = prev_close


# ══════════════════════════════════════════════════════════════
# 4. 因子计算
# ══════════════════════════════════════════════════════════════

def _build_feature_loader(raw_data: dict, config: dict, financials: dict | None = None,
                          overseas_data: dict[date, dict[str, float]] | None = None,
                          market_wide_data: dict[date, dict[str, float]] | None = None,
                          monthly_universe: dict[date, list[str]] | None = None,
                          total_shares: dict[str, float] | None = None):
    """根据 config 构建 feature_loader 回调。

    financials：
      - westock 模式为时间序列 {code: [{announce_date, roe_ttm, ...}]}，
        loader 内对每个 trade_date 做 PIT 快照（_financials_as_of）。
      - 其它模式传 {} 或旧式 {code:{factor:val}}（兼容，按整段当快照）。
    total_shares：{code: 当前总股本}，B 类估值因子 pb/pe_ttm/peg 用（缺则该因子中性）。
    """
    from core.common.calendar import get_calendar
    cal = get_calendar()
    total_shares = total_shares or {}

    factors = _enabled_factors(config)
    factor_settings = config.get("factor_settings", {})
    # ★ 从所有启用因子中自动推算最小 lookback（最长窗口 + 缓冲）
    _max_win = 22
    for fg in factors:
        for v in fg.get("params", {}).values():
            if isinstance(v, (int, float)):
                _max_win = max(_max_win, int(v))
    # 因子名含窗口的情况（如 momentum_60d → 60, volatility_20d → 20）
    for fg in factors:
        name = fg["name"]
        parts = name.split("_")
        if len(parts) >= 2:
            try:
                w = int(parts[-1].replace("d", ""))
                _max_win = max(_max_win, w)
            except ValueError:
                pass
    cfg_lookback = config["data_source"].get("lookback_days", 400)
    lookback = max(factor_settings.get("lookback_window", 60), _max_win + 10)
    min_points = factor_settings.get("min_price_points", 22)

    def loader(trade_date: date, codes: list[str]) -> pd.DataFrame:
        lookback_dates = cal.get_prev_n_trading_days(trade_date, lookback)
        lookback_dates = [d for d in lookback_dates if d in raw_data]

        # 当日海外数据
        today_overseas = overseas_data.get(trade_date, {}) if overseas_data else {}
        # 当日全市场数据
        today_market = market_wide_data.get(trade_date, {}) if market_wide_data else {}
        # 是否时间序列版 financials（westock：value 是 list；旧式是 dict）
        _fin_is_series = bool(financials) and any(
            isinstance(v, list) for v in financials.values()
        )

        # 构建价格序列
        price_hist: dict[str, list[float]] = {}
        vol_hist: dict[str, list[float]] = {}
        amount_hist: dict[str, list[float]] = {}
        high_hist: dict[str, list[float]] = {}
        low_hist: dict[str, list[float]] = {}
        open_hist: dict[str, list[float]] = {}
        turnover_hist: dict[str, list[float]] = {}
        date_list: list[date] = []

        for dt in lookback_dates:
            date_list.append(dt)
            for code, fields in raw_data.get(dt, {}).items():
                close = fields.get("close", 0)
                if close > 0:
                    price_hist.setdefault(code, []).append(close)
                    vol_hist.setdefault(code, []).append(fields.get("volume", 0))
                    amount_hist.setdefault(code, []).append(fields.get("amount", 0))
                    high_hist.setdefault(code, []).append(fields.get("high", close))
                    low_hist.setdefault(code, []).append(fields.get("low", close))
                    open_hist.setdefault(code, []).append(fields.get("open", close))
                    turnover_hist.setdefault(code, []).append(fields.get("turnover", 0))

        # ★ get_prev_n_trading_days 返回 [最新→最旧]，反转使 arr[0]=最旧, arr[-1]=最新
        for h in [price_hist, vol_hist, amount_hist, high_hist, low_hist, open_hist, turnover_hist]:
            for code in h:
                h[code].reverse()

        results: list[dict] = []
        target = list(codes) if codes else list(price_hist.keys())

        for code in target:
            prices = price_hist.get(code, [])
            if len(prices) < min_points:
                continue

            arr = np.array(prices, dtype=float)
            rets = np.diff(arr) / arr[:-1]
            vols_arr = np.array(vol_hist.get(code, []), dtype=float)
            high_arr = np.array(high_hist.get(code, []), dtype=float)
            low_arr = np.array(low_hist.get(code, []), dtype=float)
            prev_close = np.roll(arr, 1)
            prev_close[0] = arr[0]

            row: dict[str, Any] = {"code": code}

            # 基本面 PIT 快照（westock 时间序列）；旧式则整段当快照
            if not financials:
                today_fin = {}
            elif _fin_is_series:
                today_fin = _financials_as_of(financials.get(code), trade_date)
            else:
                today_fin = financials.get(code, {})

            for fg in factors:
                name = fg["name"]
                try:
                    val = _compute_factor_value(
                        fg, arr, rets, vols_arr, high_arr, low_arr, prev_close,
                        amount_hist.get(code, []),
                        today_fin,
                        today_overseas,
                        today_market,
                        turnover_hist.get(code, []),
                        close_price=float(arr[-1]),
                        total_shares=total_shares.get(code),
                    )
                    row[name] = float(val) if val is not None and not np.isnan(val) else 0.0
                except Exception:
                    row[name] = 0.0

            results.append(row)

        df = pd.DataFrame(results)
        if not df.empty:
            # 横截面因子（相对强度）相对"当月宇宙"，而非全部拉取的代码
            uni = _universe_for_date(monthly_universe, trade_date)
            df = _add_cross_sectional_factors(df, factors, price_hist, universe=uni)
        return df.set_index("code") if not df.empty else df

    return loader


def _add_cross_sectional_factors(
    df: pd.DataFrame,
    factors: list[dict],
    price_hist: dict[str, list[float]],
    universe: list[str] | None = None,
) -> pd.DataFrame:
    """计算需要横截面对比的因子（alpha动量、板块相对强度）。

    universe 给定时，"市场平均动量"相对当月候选宇宙计算，而非全部拉取
    的代码（避免相对强度被无关股票稀释/扭曲）。
    """
    factor_names = {f["name"] for f in factors}
    uni_set = set(universe) if universe else None

    # alpha_momentum: 个股动量 - 等权市场平均动量（市场=当月宇宙）
    if any(n.startswith("alpha_momentum_") for n in factor_names):
        w = 20
        # 直接从价格序列重算动量，避免依赖其他因子列
        mom_values: dict[str, float] = {}
        for code in df["code"]:
            prices = price_hist.get(code, [])
            if len(prices) >= w + 1:
                mom_values[code] = prices[-1] / prices[-min(w, len(prices))] - 1.0
            else:
                mom_values[code] = 0.0
        # 市场平均只用宇宙内标的（无宇宙则用全部）
        basis = {c: m for c, m in mom_values.items() if uni_set is None or c in uni_set}
        if basis:
            mean_mom = sum(basis.values()) / len(basis)
            df["alpha_momentum_20d"] = df["code"].map(
                lambda c: mom_values.get(c, 0.0) - mean_mom
            )
        else:
            df["alpha_momentum_20d"] = 0.0

    # sector_relative_strength: 个股动量在板块内的百分位排名
    if any(n.startswith("sector_relative_strength_") for n in factor_names):
        w = 20
        mom_values = {}
        for code in df["code"]:
            prices = price_hist.get(code, [])
            if len(prices) >= w + 1:
                mom_values[code] = prices[-1] / prices[-min(w, len(prices))] - 1.0
            else:
                mom_values[code] = 0.0
        df["_raw_mom"] = df["code"].map(mom_values)
        df["_sector"] = df["code"].map(_get_sector)
        # 板块内百分位排名（至少3只才计算，否则给0.5中性）
        def _safe_rank(grp):
            if len(grp) >= 3:
                return grp.rank(pct=True)
            return pd.Series(0.5, index=grp.index)
        df["sector_relative_strength_20d"] = df.groupby("_sector")["_raw_mom"].transform(_safe_rank)
        df.drop(columns=["_raw_mom", "_sector"], inplace=True)

    return df


def _compute_factor_value(
    fg: dict,
    arr: np.ndarray, rets: np.ndarray,
    vols: np.ndarray, high: np.ndarray, low: np.ndarray,
    prev_close: np.ndarray, amounts: list,
    financials: dict[str, float] | None = None,
    overseas: dict[str, float] | None = None,
    market_wide: dict[str, float] | None = None,
    turnover_vals: list | None = None,
    close_price: float | None = None,
    total_shares: float | None = None,
) -> float | None:
    """在内存数据上直接计算因子值（不走 core/features 的完整实现，
    因为后者依赖 FeatureStore 和特定的 DataFrame 格式）。

    close_price/total_shares：B 类估值因子（pb/pe_ttm/peg）用，
    PB = close×shares / PIT净资产，PE = close×shares / PIT净利TTM。
    """

    name = fg["name"]
    params = fg.get("params", {})

    try:
        if name == "momentum_20d" or name.startswith("momentum_"):
            w = params.get("window", 20)
            # 尝试从因子名解析窗口（如 momentum_60d → 60）
            if not params.get("window") and "_" in name:
                try:
                    w = int(name.split("_")[-1].replace("d", ""))
                except ValueError:
                    pass
            return float((arr[-1] / arr[-min(w, len(arr))] - 1.0)) if len(arr) >= w + 1 else 0.0

        if name == "volatility_20d" or name.startswith("volatility_"):
            w = params.get("window", 20)
            if not params.get("window") and "_" in name:
                try:
                    w = int(name.split("_")[-1].replace("d", ""))
                except ValueError:
                    pass
            return float(np.nanstd(rets[-w:]) * np.sqrt(252)) if len(rets) >= 5 else 0.0

        if name == "rsi_14" or name.startswith("rsi_"):
            w = params.get("window", 14)
            gains = np.maximum(rets[-w:], 0)
            losses = -np.minimum(rets[-w:], 0)
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return float(100.0 - 100.0 / (1.0 + rs))

        if name == "atr_14" or name.startswith("atr_"):
            w = params.get("window", 14)
            tr = np.maximum(high[-w:] - low[-w:],
                    np.maximum(np.abs(high[-w:] - prev_close[-w:]),
                               np.abs(low[-w:] - prev_close[-w:])))
            return float(tr.mean())

        if name == "amplitude_20d" or name.startswith("amplitude_"):
            w = params.get("window", 20)
            amp = (high[-w:] - low[-w:]) / prev_close[-w:]
            return float(amp.mean())

        if name == "turnover_5d" or name.startswith("turnover_"):
            w = params.get("window", 5)
            # 尝试从因子名解析窗口
            if not params.get("window") and "_" in name:
                try:
                    w = int(name.split("_")[-1].replace("d", ""))
                except ValueError:
                    pass
            # ★ 使用东方财富真实换手率（百分比，如 3.5 = 3.5%）
            to_vals = turnover_vals or []
            if len(to_vals) >= w:
                return float(np.mean(to_vals[-w:]))
            return 0.0

        if name == "volume_ratio":
            w = params.get("window", 20)
            if len(vols) >= w + 1:
                return float(vols[-1] / vols[-w:].mean()) if vols[-w:].mean() > 0 else 1.0
            return 1.0

        if name == "macd_dif":
            fast = params.get("fast", 12)
            slow = params.get("slow", 26)
            ema_fast = _ema(arr, fast)
            ema_slow = _ema(arr, slow)
            return float(ema_fast - ema_slow)

        if name == "bollinger_position":
            w = params.get("window", 20)
            n_std = params.get("num_std", 2)
            mid = arr[-w:].mean()
            std = arr[-w:].std()
            upper = mid + n_std * std
            lower = mid - n_std * std
            if upper == lower:
                return 0.5
            return float((arr[-1] - lower) / (upper - lower))

        if name == "ma_alignment":
            s, m, l = params.get("short", 5), params.get("mid", 20), params.get("long", 60)
            ma_s = arr[-min(s, len(arr)):].mean()
            ma_m = arr[-min(m, len(arr)):].mean()
            ma_l = arr[-min(l, len(arr)):].mean()
            # 连续评分：短/长均线偏离度 ∈ [-0.2, 0.2] → 归一化到 [0, 1]
            ratio = ma_s / max(ma_l, 0.01) - 1.0
            score = max(-0.2, min(0.2, ratio))
            return (score + 0.2) / 0.4

        if name == "amihud_illiq":
            w = params.get("window", 20)
            if len(rets) < 2 or len(amounts) < 2:
                return 0.0
            # ★ rets 比 amounts 少1个元素（rets[i] 是从 day i 到 i+1 的收益）
            # 对齐：amounts 从 index 1 开始（收益日当天的成交额）
            amt_aligned = np.array(amounts[1:], dtype=float)
            n_ret = len(rets)
            w_align = min(w, n_ret, len(amt_aligned))
            illiq = np.abs(rets[-w_align:]) / np.maximum(amt_aligned[-w_align:], 1e-8)
            return float(illiq.mean() * 1e8)

        # ── NLP 情绪因子 ───
        if name == "announcement_sentiment_score":
            # 使用财务数据构造代理文本，经关键词引擎评分
            if financials:
                text_parts = []
                np_yoy = financials.get("net_profit_yoy", 0) or 0
                rev_yoy = financials.get("revenue_yoy", 0) or 0
                debt = financials.get("debt_ratio", 0) or 0
                if np_yoy > 0.05:
                    text_parts.append("净利润大幅增长")
                elif np_yoy > 0:
                    text_parts.append("净利润增长")
                elif np_yoy < -0.10:
                    text_parts.append("净利润下降")
                if rev_yoy > 0.05:
                    text_parts.append("营收增长")
                elif rev_yoy < -0.10:
                    text_parts.append("营收下降")
                if debt > 0.60:
                    text_parts.append("负债率高")
                elif debt < 0.30:
                    text_parts.append("负债率低")
                if text_parts:
                    from core.models.nlp import KeywordSentimentEngine
                    engine = KeywordSentimentEngine()
                    result = engine.analyze(" ".join(text_parts))
                    if result.sentiment == "positive":
                        return result.confidence
                    elif result.sentiment == "negative":
                        return -result.confidence
            return 0.0

        # ── 海外市场因子 ───
        overseas = overseas or {}
        if name == "overnight_adr_mapped":
            # 中概股隔夜映射 ≈ NASDAQ × 1.2（中概beta更高）
            return overseas.get("ndx_ret", 0) * 1.2
        if name == "a50_futures_overnight":
            return overseas.get("a50_proxy_ret", 0)
        if name == "hsi_futures_overnight":
            return overseas.get("hsi_ret", 0)

        # ── 全市场情绪因子 ───
        market_wide = market_wide or {}
        if name == "limit_up_count":
            return market_wide.get("limit_up_count", 0)
        if name == "limit_up_chain_height":
            return market_wide.get("limit_up_chain_height", 0)
        if name == "limit_down_count":
            return -market_wide.get("limit_down_count", 0)  # 反转：跌停越多→分越低
        if name == "board_break_ratio":
            return -market_wide.get("board_break_ratio", 0)  # 反转：炸板率越高→分越低
        if name == "limit_up_ratio":
            return market_wide.get("limit_up_ratio", 0)

        # ── B 类估值因子：用历史 close × 当前股本 / PIT 财报实时算 ──
        # 不用 quote 的当前 PB/PE（会前视）。财报缺失/股本缺失 → 中性。
        fin = financials or {}
        if name in ("pb", "pe_ttm", "peg"):
            if not close_price or not total_shares:
                return None  # 缺市值数据 → 中性（→0）
            mktcap = close_price * total_shares
            if name == "pb":
                equity = fin.get("_equity")
                return float(mktcap / equity) if equity and equity > 0 else None
            if name == "pe_ttm":
                np_ttm = fin.get("_np_ttm")
                # 亏损（净利≤0）→ PE 无意义，返回中性
                return float(mktcap / np_ttm) if np_ttm and np_ttm > 0 else None
            if name == "peg":
                np_ttm = fin.get("_np_ttm")
                npy = fin.get("net_profit_yoy")
                if not np_ttm or np_ttm <= 0 or not npy or npy <= 0:
                    return None  # 亏损或负增长 → PEG 无意义
                pe = mktcap / np_ttm
                return float(pe / (npy * 100.0))

        # ── 基本面因子（A 类：从 PIT 财报快照直接读）───
        #   gross_margin/net_margin_ttm/roe_ttm/revenue_yoy/net_profit_yoy
        if financials and name in financials:
            return financials[name]

        # ── 估值因子（可以从日线计算的版本）───
        if name in ("pe_ttm", "pb", "ps_ttm", "pcf_ttm",
                    "roe_ttm", "roa_ttm", "roic_ttm",
                    "revenue_yoy", "net_profit_yoy",
                    "gross_margin_trend", "net_margin_ttm",
                    "debt_ratio", "current_ratio", "quick_ratio",
                    "cf_ratio_ttm", "free_cf_yield",
                    "dividend_yield", "ep_ttm", "peg",
                    "margin_balance_change_5d", "main_force_net_inflow_5d",
                    "main_force_net_inflow_20d", "main_force_inflow_ratio",
                    "dragon_tiger_net_buy", "dragon_tiger_institution_count",
                    "northbound_quarter_change", "lockup_expiry_days",
                    "auction_open_premium", "auction_volume_ratio",
                    "performance_forecast_surprise",
                    "theme_heat_score", "dragon_tiger_review_score",
                    "limit_up_review_signal", "auction_strength_score",
                    "auction_fake_order_risk", "days_to_next_event"):
            # 因子名称存在但没有数据 → 返回中性值 0.0
            # （不在优化器中不会被选中，因为已自动过滤）
            return 0.0

        return 0.0
    except Exception:
        return 0.0


def _ema(arr: np.ndarray, span: int) -> float:
    """指数移动平均。"""
    alpha = 2.0 / (span + 1)
    result = arr[0]
    for x in arr[1:]:
        result = alpha * x + (1 - alpha) * result
    return float(result)


# ══════════════════════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# 板块分组（用于板块相对强度因子）
# ══════════════════════════════════════════════════════════════

SECTOR_GROUPS: dict[str, set[str]] = {
    "ai_optical":       {"300308", "300502", "300394"},
    "ai_semiconductor":  {"002415", "603501", "002049", "300474", "300604",
                          "002156", "600584", "002185", "603160",
                          "000977", "603019", "300418", "300624", "000063", "002230"},
    "ai_datacenter":    {"300383", "603881", "301236"},
    "consumer_elec":    {"000333", "002475", "000651", "601138"},
    "robotics":         {"603728", "300024", "002747", "002979", "300124",
                          "300450", "300316", "002008", "300751"},
    "power_energy":     {"600406", "601877", "002129", "600438", "300274",
                          "003816", "601985"},
    "apple_chain":      {"300433", "002456", "002600", "601231", "300136",
                          "002241", "002635"},
}


def _get_sector(code: str) -> str:
    for sector, codes in SECTOR_GROUPS.items():
        if code in codes:
            return sector
    return "__other__"


def _calc_min_lookback(config: dict) -> int:
    """根据所有启用因子中最大的计算窗口，反推最小 lookback。

    每个因子窗口参数不同：momentum_60d 需要 60 个交易日、
    macd 需要 26+9=35、rsi 需要 14 等，再×1.5 转为自然日、加 30 天缓冲。
    """
    max_window = 22  # min_price_points 最低值
    for fg_name, fg_cfg in config.get("factors", {}).items():
        if not fg_cfg.get("enabled", False):
            continue
        params = fg_cfg.get("params", {})
        # 取所有数值参数的最大值（window, fast, slow, signal, long 等）
        for k, v in params.items():
            if isinstance(v, (int, float)):
                max_window = max(max_window, int(v))
    # 交易日 → 自然日（×1.5），+30 天缓冲
    return int(max_window * 1.5) + 30


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main(config_path: str = "configs/strategy.yaml", full_range: bool = False) -> None:
    config = load_config(config_path)
    cfg_bt = config["backtest"]

    # 回测区间选择：
    # - full_range=True：用 start_date~end_date 完整训练区间（验证动态池逐月轮动、
    #   夏普是否回落到现实水平）。这是验证幸存者偏差是否真消除的关键。
    # - 否则：若配置了 validate_start/validate_end 则优先用验证区间（短样本，
    #   夏普会失真，仅适合 dynamic vs fixed 的相对对比）。
    if not full_range and "validate_start" in cfg_bt and "validate_end" in cfg_bt:
        start_date = _parse_date(cfg_bt["validate_start"])
        end_date = _parse_date(cfg_bt["validate_end"])
    else:
        start_date = _parse_date(cfg_bt["start_date"])
        end_date = _parse_date(cfg_bt["end_date"])
    initial_capital = cfg_bt["initial_capital"]
    logger.info("回测区间: %s ~ %s%s", start_date, end_date,
                "（完整训练区间）" if full_range else "")

    # ── 打印当前配置 ──
    enabled = _enabled_factors(config)
    logger.info("当前策略配置: %d 个因子, top_n=%d, 调仓=%s, 资金=¥%.0f",
                len(enabled), config["strategy"]["top_n"],
                config["strategy"]["rebalance_frequency"], initial_capital)
    for f in enabled:
        _inv = f["name"] in ConfigDrivenStrategy.INVERSE_EXACT or \
               f["name"].startswith(ConfigDrivenStrategy.INVERSE_PREFIXES)
        direction = "↓(越低越好)" if _inv else "↑"
        logger.info("  %s (权重=%.2f) %s", f["name"], f.get("weight", 0), direction)

    # ── 数据加载 ──
    # ★ 告诉 _load_real_data 回测实际起点（而非 optimizer 训练起点），避免拉无用历史数据
    orig_backtest = dict(config["backtest"])
    config["backtest"]["start_date"] = str(start_date)
    config["backtest"]["end_date"] = str(end_date)
    logger.info("正在加载数据...")
    raw_data, source_label = _load_real_data(config)
    config["backtest"] = orig_backtest  # 恢复原始配置

    if not raw_data or len(raw_data) < 100:
        logger.error("=" * 60)
        logger.error("无法加载真实数据！")
        logger.error("  原因: %s", source_label)
        logger.error("  已尝试拉取 %d 只股票: %s", len(_get_codes(config)), _get_codes(config)[:10])
        logger.error("")
        logger.error("  请检查:")
        logger.error("    1. 网络是否正常: ping finance.sina.com.cn")
        logger.error("    2. 配置文件: configs/strategy.yaml → data_source.provider")
        logger.error("       sina  → 走新浪财经接口（当前）")
        logger.error("       eastmoney → 走东方财富接口（备选）")
        logger.error("    3. 单独测试: python -c \"import akshare as ak; print(ak.stock_zh_a_daily(symbol='sh600519', adjust='qfq').tail(3))\"")
        logger.error("=" * 60)
        return

    if not raw_data:
        logger.error("无可用数据，退出")
        return

    # ── 构建回调 ──
    def data_loader(trade_date: date) -> pd.DataFrame:
        rows = [{"code": code, **fields} for code, fields in raw_data.get(trade_date, {}).items()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # 动态股票池：预计算每月候选宇宙（fixed 模式返回空 dict）
    # 用实际回测窗口（start_date~end_date）覆盖月份锚点计算，确保 validate 模式
    # 与 full_range 模式都按真实回测区间构建宇宙（去掉 validate_end 以免锚点越界）。
    _uni_bt = dict(config["backtest"])
    config["backtest"]["start_date"] = str(start_date)
    config["backtest"]["end_date"] = str(end_date)
    config["backtest"].pop("validate_end", None)
    monthly_universe = _build_monthly_universe(raw_data, config)
    config["backtest"] = _uni_bt  # 恢复

    # 基本面数据：
    # - westock 模式加载真财报时间序列（真 ROE/营收同比，按 InfoPublDate PIT 对齐）
    #   + 当前总股本（B 类估值因子 PB/PE/PEG 用）
    # - 其它模式不加载（akshare _load_financials 的 ROE 是伪造值，已弃用）
    if _provider(config) == "westock":
        from research.westock_source import westock_financials, westock_total_shares
        _codes = _get_codes(config)
        financials: dict = westock_financials(_codes)
        total_shares: dict = westock_total_shares(_codes)
    else:
        financials = {}
        total_shares = {}

    # 加载海外市场数据
    overseas_data = _load_overseas_data(start_date, end_date)

    # 加载真实宽基指数（沪深300 + 中证全指）用于市场状态判断
    benchmark_index = _load_benchmark_index(start_date, end_date, provider=_provider(config))

    # 构建全市场情绪数据（从池内OHLCV推算）
    market_wide_data = _build_market_wide_from_pool(raw_data)

    feature_loader = _build_feature_loader(raw_data, config, financials,
                                           overseas_data, market_wide_data,
                                           monthly_universe=monthly_universe,
                                           total_shares=total_shares)

    # ── 回测 ──
    from core.backtest.engine import BacktestEngine

    engine = BacktestEngine(
        start_date=start_date, end_date=end_date,
        initial_capital=initial_capital,
    )
    # 注入沪深300日收盘做相对超额基准（{date: close}）
    if benchmark_index and "csi300" in benchmark_index:
        bench_df = benchmark_index["csi300"]
        engine.benchmark_close = {
            (d.date() if hasattr(d, "date") else d): float(c)
            for d, c in bench_df["close"].items()
        }
    strategy = ConfigDrivenStrategy(config, raw_data, overseas_data,
                                    benchmark_index=benchmark_index,
                                    monthly_universe=monthly_universe)

    logger.info("开始回测 %s ~ %s (资金=¥%.0f)", start_date, end_date, initial_capital)
    result = engine.run(strategy, data_loader=data_loader, feature_loader=feature_loader)

    # ── 输出 ──
    metrics = result.summary()
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:.<30s} {v:>10.4f}")
        else:
            print(f"  {k:.<30s} {v!s:>10s}")
    print("=" * 60)

    # ── 相对基准（沪深300）超额表现 ──
    # 绝对夏普会把"跟跌"误判为失败。看相对基准才知道策略是真有 alpha 还是只跟涨跌。
    if "benchmark_annual_return" in metrics:
        ba = metrics["benchmark_annual_return"]
        sa = metrics["annual_return"]
        ea = metrics["excess_annual_return"]
        ir = metrics["information_ratio"]
        edd = metrics["excess_max_drawdown"]
        verdict = "跑赢基准 ✓" if ea > 0 else "跑输基准 ✗"
        print(f"\n相对基准（沪深300）—— {verdict}")
        print(f"  策略年化 {sa*100:+.1f}%  vs  基准年化 {ba*100:+.1f}%  →  超额年化 {ea*100:+.1f}%")
        print(f"  信息比率(IR) {ir:.3f}   |   超额最大回撤 {edd*100:.1f}%")
        print(f"  （IR>0.5 才算有持续超额能力；绝对夏普受市场涨跌影响，超额口径更可信）")
        print("=" * 60)

    # 现金不足追加注资提示（若发生）
    injected = metrics.get("total_injected", 0) or 0
    if injected > 0:
        print(f"\n⚠ 回测期间现金不足，累计假定追加注资 ¥{injected:,.0f}"
              f"（有效投入资金 ¥{metrics.get('effective_capital', initial_capital):,.0f}）。"
              f"\n  收益率指标仍按初始资金 ¥{initial_capital:,.0f} 计算，请留意真实投入更高。")

    if not result.trade_records.empty:
        filled = result.trade_records[result.trade_records["status"] == "filled"]
        print(f"\n总成交 {len(result.trade_records)} 笔，其中成交 {len(filled)} 笔")

    # ── 每次调仓的买卖清单 + 盘中持仓变化 ──
    if not result.trade_records.empty:
        _print_rebalance_history(result, raw_data)

    print(f"\n修改 configs/strategy.yaml 可调整因子、权重、持仓数、股票池等全部参数。")


def _load_name_map() -> dict[str, str]:
    """从 strategy.yaml 加载股票代码→名称映射。"""
    try:
        with open("configs/strategy.yaml") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {}
    name_map: dict[str, str] = {}
    for line in lines:
        import re
        m = re.match(r'\s*-\s*"(\d{6})"\s*#\s*(.+)', line)
        if m:
            name_map[m.group(1)] = m.group(2).strip()
    return name_map


def _print_rebalance_history(result, raw_data: dict) -> None:
    """打印每次调仓日的买卖清单、费用明细、以及调仓前后的持仓变化。"""
    trades_df = result.trade_records
    snapshots = result.daily_snapshots
    if trades_df.empty:
        return

    # 加载股票名称映射
    name_map = _load_name_map()

    # 构建交易日期 → 快照的映射
    snap_map: dict = {}
    for s in snapshots:
        snap_map[str(s.date)] = s

    trades_df["trade_date_str"] = trades_df["trade_date"].astype(str)
    dates = sorted(trades_df["trade_date_str"].unique())

    for d in dates:
        day_trades = trades_df[trades_df["trade_date_str"] == d]
        buys = day_trades[day_trades["side"] == "buy"]
        sells = day_trades[day_trades["side"] == "sell"]

        if len(buys) == 0 and len(sells) == 0:
            continue

        # 获取该日快照
        snap = snap_map.get(d)

        print(f"\n{'─'*80}")
        print(f"调仓日: {d}  |  {len(buys)} 买入 + {len(sells)} 卖出")
        if snap is not None:
            print(f"  总资产: ¥{snap.total_value:,.0f}  |  现金: ¥{snap.cash:,.0f}")
        print(f"{'─'*80}")

        if len(sells) > 0:
            print("  【卖出】")
            total_sell = 0.0
            total_pnl = 0.0
            for _, t in sells.iterrows():
                name = name_map.get(t["code"], "")
                amount = t["filled_price"] * t["filled_shares"]
                cost = t.get("cost_breakdown", {})
                real_fee = _real_cost(cost)
                slip = _slip(cost)
                # 已实现盈亏（在 engine._apply_fill 中计算，以减去真实佣金印花过户后的净价与平均成本价的差为准）
                raw_pnl = t.get("realized_pnl")
                pnl = 0.0
                try:
                    pnl = float(raw_pnl) if raw_pnl is not None and not (isinstance(raw_pnl, float) and pd.isna(raw_pnl)) else 0.0
                except (ValueError, TypeError):
                    pnl = 0.0
                pnl_str = f"  盈亏 {'+' if pnl >= 0 else ''}¥{pnl:,.0f}"
                print(f"    {t['code']} {name:<10s}  "
                      f"{int(t['filled_shares']):>7d}股 @ ¥{t['filled_price']:>8.2f}  "
                      f"成交 ¥{amount:>12,.0f}  "
                      f"费 ¥{real_fee:.2f} + 滑¥{slip:.2f}(已扣)  "
                      f"({_fmt(cost,'commission')}, {_fmt(cost,'stamp_duty')}, {_fmt(cost,'transfer_fee')}){pnl_str}")
                total_sell += amount
                total_pnl += pnl

        if len(buys) > 0:
            print("  【买入】")
            total_buy = 0.0
            for _, t in buys.iterrows():
                name = name_map.get(t["code"], "")
                amount = t["filled_price"] * t["filled_shares"]
                cost = t.get("cost_breakdown", {})
                real_fee = _real_cost(cost)
                slip = _slip(cost)
                print(f"    {t['code']} {name:<10s}  "
                      f"{int(t['filled_shares']):>7d}股 @ ¥{t['filled_price']:>8.2f}  "
                      f"成交 ¥{amount:>12,.0f}  "
                      f"费 ¥{real_fee:.2f} + 滑¥{slip:.2f}(已扣)  "
                      f"(佣{_fmt(cost,'commission')} 过{_fmt(cost,'transfer_fee')})")
                total_buy += amount

        # 当日结束后的持仓（含当日新买入的股票，市值为当日收盘价×股数）
        if snap is not None and snap.positions:
            print(f"  【当日持仓】({len(snap.positions)} 只，含当日新买入)")
            for code, shares in sorted(snap.positions.items(), key=lambda x: -x[1]):
                if shares <= 0:
                    continue
                pos_val = snap.position_values.get(code, 0)
                name = name_map.get(code, "")
                pct = pos_val / snap.total_value * 100 if snap.total_value > 0 else 0
                avg_price = pos_val / shares if shares > 0 and pos_val > 0 else 0
                print(f"    {code} {name:<10s}  {shares:>7d}股  "
                      f"市值 ¥{pos_val:>10,.0f}  ({pct:.1f}%)  "
                      f"均价 ¥{avg_price:.2f}")
            if snap.total_value > 0:
                cash_pct = snap.cash / snap.total_value * 100
                print(f"    {'—':<6s} {'现金':<10s}  {'':>7s}  "
                      f"¥{snap.cash:>12,.0f}  ({cash_pct:.1f}%)")
            print(f"    总资产: ¥{snap.total_value:,.0f}")

    # 最终持仓摘要
    _print_final_summary(trades_df, name_map)


def _fmt(cost_dict, key: str) -> str:
    """安全格式化费用字段。"""
    if not isinstance(cost_dict, dict):
        return "?"
    return f"{cost_dict.get(key, 0):.2f}"


def _real_cost(cost_dict) -> float:
    """交易所实收费用（佣金+印花税+过户费），不含滑点。

    注意：滑点同样已从回测现金中扣除（engine 按 CostBreakdown.total 扣），
    此处拆开只是为了展示核对。"""
    if not isinstance(cost_dict, dict):
        return 0.0
    return (cost_dict.get("commission", 0) +
            cost_dict.get("stamp_duty", 0) +
            cost_dict.get("transfer_fee", 0))


def _slip(cost_dict) -> float:
    """滑点估算值。★已计入成本从回测现金中扣除（保守口径），并非仅供参考。"""
    if not isinstance(cost_dict, dict):
        return 0.0
    return cost_dict.get("slippage_est", 0)


def _print_final_summary(trades_df, name_map: dict) -> None:
    """全时段持仓净变化汇总。"""
    print(f"\n{'─'*80}")
    print("全时段持仓净变化汇总")
    print(f"{'─'*80}")
    code_summary: dict = {}
    for _, t in trades_df[trades_df["status"] == "filled"].iterrows():
        code = t["code"]
        if code not in code_summary:
            code_summary[code] = {"buy": 0, "sell": 0, "buy_amount": 0, "sell_amount": 0}
        if t["side"] == "buy":
            code_summary[code]["buy"] += int(t["filled_shares"])
            code_summary[code]["buy_amount"] += t["filled_price"] * t["filled_shares"]
        else:
            code_summary[code]["sell"] += int(t["filled_shares"])
            code_summary[code]["sell_amount"] += t["filled_price"] * t["filled_shares"]

    for code, summary in sorted(code_summary.items(),
                                 key=lambda x: -(x[1]["buy_amount"] + x[1]["sell_amount"])):
        name = name_map.get(code, "")
        net_shares = summary["buy"] - summary["sell"]
        net_amount = summary["buy_amount"] - summary["sell_amount"]
        if summary["buy"] > 0 or summary["sell"] > 0:
            print(f"  {code} {name:<10s}  "
                  f"买 {summary['buy']:>7,d}股  "
                  f"卖 {summary['sell']:>7,d}股  "
                  f"净 {'+' if net_shares >= 0 else ''}{net_shares:,d}股  "
                  f"净资金 {'+' if net_amount >= 0 else ''}¥{net_amount:,.0f}")


if __name__ == "__main__":
    import sys as _sys
    args = [a for a in _sys.argv[1:]]
    # --full / --full-range：跑完整训练区间（start_date~end_date），而非验证窗口
    full_range = any(a in ("--full", "--full-range") for a in args)
    pos = [a for a in args if not a.startswith("-")]
    path = pos[0] if pos else "configs/strategy.yaml"
    main(path, full_range=full_range)
