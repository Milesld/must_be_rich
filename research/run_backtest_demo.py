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
    INVERSE_EXACT = {"pe_ttm", "pb", "debt_ratio", "amihud_illiq"}
    INVERSE_PREFIXES = ("volatility_", "amplitude_", "atr_")

    def _is_inverse(self, name: str) -> bool:
        return name in self.INVERSE_EXACT or name.startswith(self.INVERSE_PREFIXES)

    def __init__(self, config: dict, raw_data: dict | None = None,
                 overseas_data: dict | None = None) -> None:
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
            logger.info("市场状态判断已启用 (仓位范围: %.0f%%~%.0f%%)",
                        self._regime_min_position * 100, self._regime_max_position * 100)
        else:
            self._regime_detector = None
            self._regime_min_position = 1.0
            self._regime_max_position = 1.0
            self._raw_data = None
            self._overseas_data = {}

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

        scores = self._score_stocks(features)
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
        """从 raw_data 构建等权市场指数（用于市场状态判断）。"""
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



def _get_codes(config: dict) -> list[str]:
    codes = config.get("data_source", {}).get("codes", [])
    if not codes:
        logger.error("configs/strategy.yaml 中 data_source.codes 为空！请在配置文件中设置股票池。")
        return []
    return list(dict.fromkeys(codes))  # 去重保序


def _fetch_sina(codes: list[str], start: date, end: date) -> pd.DataFrame | None:
    """从新浪接口拉取日K线。"""
    import akshare as ak

    frames = []
    for code in codes:
        try:
            prefix = "sh" if code.startswith("6") else "sz"
            raw = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
            if raw is None or len(raw) == 0:
                continue
            df = raw.copy()
            df["code"] = code
            df["trade_date"] = pd.to_datetime(df["date"]).dt.date
            df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
            if len(df) == 0:
                continue
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["code", "trade_date"])


def _fetch_eastmoney(codes: list[str], start: date, end: date) -> pd.DataFrame | None:
    """从东方财富接口拉取（备用）—— 提供 OHLCV + 换手率。"""
    import akshare as ak

    frames = []
    for code in codes:
        try:
            raw = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="qfq",
            )
            if raw is None or len(raw) == 0:
                continue
            df = raw.copy()
            df["code"] = code
            df["trade_date"] = pd.to_datetime(df["日期"]).dt.date
            col_map = {"开盘": "open", "最高": "high", "最低": "low",
                       "收盘": "close", "成交量": "volume", "换手率": "turnover"}
            for cn, en in col_map.items():
                if cn in df.columns:
                    df[en] = df[cn].astype(float)
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["code", "trade_date"])


def _supplement_turnover(raw_data: dict[date, dict], codes: list[str],
                          start: date, end: date) -> None:
    """从东方财富补充真实换手率数据，写入 raw_data 的 'turnover' 字段。"""
    import akshare as ak
    logger.info("正在补充换手率数据（东方财富）...")
    count = 0
    for code in codes:
        try:
            raw = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="qfq",
            )
            if raw is None or len(raw) == 0:
                continue
            for _, row in raw.iterrows():
                try:
                    td_raw = pd.to_datetime(row["日期"]).date()
                except Exception:
                    continue
                if td_raw in raw_data and code in raw_data[td_raw]:
                    raw_data[td_raw][code]["turnover"] = float(row.get("换手率", 0) or 0)
                    count += 1
        except Exception:
            continue
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
    codes = _get_codes(config)
    provider = config.get("data_source", {}).get("provider", "sina")

    logger.info("正在从 %s 拉取 %d 只股票 (%s ~ %s)...", provider, len(codes), start, end)

    import threading
    result: list = []
    error_msg: list = []

    def _fetch():
        try:
            if provider == "eastmoney":
                df = _fetch_eastmoney(codes, start, end)
            else:
                df = _fetch_sina(codes, start, end)
            if df is not None and len(df) > 0:
                result.append(df)
            else:
                error_msg.append("返回空数据")
        except Exception as e:
            error_msg.append(str(e))

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout=120.0)

    if not result:
        reason = error_msg[0] if error_msg else "超时 (120s)"
        logger.warning("%s 数据拉取失败: %s", provider, reason)
        return {}, reason

    raw_df = result[0]
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

def _build_feature_loader(raw_data: dict, config: dict, financials: dict[str, dict[str, float]] | None = None,
                          overseas_data: dict[date, dict[str, float]] | None = None,
                          market_wide_data: dict[date, dict[str, float]] | None = None):
    """根据 config 构建 feature_loader 回调。"""
    from core.common.calendar import get_calendar
    cal = get_calendar()

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

            for fg in factors:
                name = fg["name"]
                try:
                    val = _compute_factor_value(
                        fg, arr, rets, vols_arr, high_arr, low_arr, prev_close,
                        amount_hist.get(code, []),
                        financials.get(code, {}) if financials else {},
                        today_overseas,
                        today_market,
                        turnover_hist.get(code, []),
                    )
                    row[name] = float(val) if val is not None and not np.isnan(val) else 0.0
                except Exception:
                    row[name] = 0.0

            results.append(row)

        df = pd.DataFrame(results)
        if not df.empty:
            df = _add_cross_sectional_factors(df, factors, price_hist)
        return df.set_index("code") if not df.empty else df

    return loader


def _add_cross_sectional_factors(
    df: pd.DataFrame,
    factors: list[dict],
    price_hist: dict[str, list[float]],
) -> pd.DataFrame:
    """计算需要横截面对比的因子（alpha动量、板块相对强度）。"""
    factor_names = {f["name"] for f in factors}

    # alpha_momentum: 个股动量 - 等权市场平均动量
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
        if mom_values:
            mean_mom = sum(mom_values.values()) / len(mom_values)
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
) -> float | None:
    """在内存数据上直接计算因子值（不走 core/features 的完整实现，
    因为后者依赖 FeatureStore 和特定的 DataFrame 格式）。"""

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

        # ── 基本面因子（从已加载的 financials 字典读取）───
        # 待拉取的数据: pe_ttm/pb/ps_ttm/pcf_ttm/roe_ttm/roa_ttm/roic_ttm/
        #              gross_margin/net_margin_ttm/revenue_yoy/net_profit_yoy/
        #              debt_ratio/current_ratio/quick_ratio/cf_ratio_ttm
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


def main(config_path: str = "configs/strategy.yaml") -> None:
    config = load_config(config_path)
    cfg_bt = config["backtest"]

    start_date = _parse_date(cfg_bt["start_date"])
    end_date = _parse_date(cfg_bt["end_date"])
    initial_capital = cfg_bt["initial_capital"]

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
    logger.info("正在加载数据...")
    raw_data, source_label = _load_real_data(config)

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
        logger.error("    3. 单独测试: python -c \"import akshare as ak; print(ak.stock_zh_a_daily('sh600519','qfq').tail(3))\"")
        logger.error("=" * 60)
        return

    if not raw_data:
        logger.error("无可用数据，退出")
        return

    # ── 构建回调 ──
    def data_loader(trade_date: date) -> pd.DataFrame:
        rows = [{"code": code, **fields} for code, fields in raw_data.get(trade_date, {}).items()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # 加载基本面数据（如有）
    codes = _get_codes(config)
    financials = _load_financials(codes)

    # 加载海外市场数据
    overseas_data = _load_overseas_data(start_date, end_date)

    # 构建全市场情绪数据（从池内OHLCV推算）
    market_wide_data = _build_market_wide_from_pool(raw_data)

    feature_loader = _build_feature_loader(raw_data, config, financials,
                                           overseas_data, market_wide_data)

    # ── 回测 ──
    from core.backtest.engine import BacktestEngine

    engine = BacktestEngine(
        start_date=start_date, end_date=end_date,
        initial_capital=initial_capital,
    )
    strategy = ConfigDrivenStrategy(config, raw_data, overseas_data)

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
                      f"费 ¥{real_fee:.2f} + 滑(估)¥{slip:.2f}  "
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
                      f"费 ¥{real_fee:.2f} + 滑(估)¥{slip:.2f}  "
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
    """实际交易费用（佣金+印花税+过户费），不含滑点估算。"""
    if not isinstance(cost_dict, dict):
        return 0.0
    return (cost_dict.get("commission", 0) +
            cost_dict.get("stamp_duty", 0) +
            cost_dict.get("transfer_fee", 0))


def _slip(cost_dict) -> float:
    """滑点估算值（非实际费用，仅用于评估策略可行性）。"""
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
    path = _sys.argv[1] if len(_sys.argv) > 1 else "configs/strategy.yaml"
    main(path)
