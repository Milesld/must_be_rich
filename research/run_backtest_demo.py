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

    # 哪些因子是"越低越好"（反转排名）
    INVERSE_FACTORS = {
        "volatility_20d", "pe_ttm", "pb", "debt_ratio",
        "amihud_illiq", "amplitude_20d",
    }

    def __init__(self, config: dict) -> None:
        cfg = config.get("strategy", {})
        self.top_n = cfg.get("top_n", 5)
        self.rebalance_freq = cfg.get("rebalance_frequency", "monthly")
        self.min_shares = cfg.get("min_shares", 100)
        self.optimizer = cfg.get("optimizer", "equal_weight")
        self.factors = _enabled_factors(config)
        # 自动归一化权重（用户不需要手动凑到1.0）
        total_w = sum(f.get("weight", 0) for f in self.factors)
        if total_w > 0:
            for f in self.factors:
                f["weight"] = f.get("weight", 0) / total_w
        if not self.factors:
            logger.warning("配置中没有启用的因子！")
        self._last_rebalance_period: Any = None

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

        if not self._should_rebalance(trade_date):
            return intents

        if features.empty:
            return intents

        scores = self._score_stocks(features)
        if scores.empty:
            return intents

        target_codes = set(scores.nlargest(self.top_n).index)

        # 卖出不在目标池的持仓
        for code, shares in list(positions.items()):
            if shares > 0 and code not in target_codes:
                price = float(daily_data.get(code, {}).get("close", 0))
                if price > 0:
                    intents.append(TradeIntent(
                        signal_id=f"sell_{code}_{trade_date}",
                        code=code, side="sell", price=price, shares=shares,
                    ))

        # 等权买入目标池
        if target_codes:
            per_stock_cash = cash / len(target_codes)
            for code in target_codes:
                if positions.get(code, 0) > 0:
                    continue
                price = float(daily_data.get(code, {}).get("close", 0))
                if price <= 0:
                    continue
                shares = int(per_stock_cash / price)
                shares = (shares // 100) * 100
                if shares >= self.min_shares:
                    intents.append(TradeIntent(
                        signal_id=f"buy_{code}_{trade_date}",
                        code=code, side="buy", price=price, shares=shares,
                    ))
        return intents

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
            if name in self.INVERSE_FACTORS:
                rank = 1.0 - rank

            common = scores.index.intersection(rank.index)
            scores.loc[common] = scores.loc[common] + rank.loc[common].astype(float) * weight

        return scores


# ══════════════════════════════════════════════════════════════
# 3. 数据加载
# ══════════════════════════════════════════════════════════════

_DEFAULT_CODES = [
    # 大盘蓝筹
    "600519", "000858", "000568",  # 白酒
    "601318", "601628",            # 保险
    "600036", "601398", "000001",  # 银行
    "601166",                      # 兴业
    # 新能源
    "300750", "002594",            # 宁德、比亚迪
    "601012", "688223",            # 隆基、晶科
    # 医药
    "600276", "300760",            # 恒瑞、迈瑞
    "300122",                      # 智飞
    # 科技
    "002415", "688981",            # 海康、中芯
    "603501",                      # 韦尔
    # 消费
    "000333", "600887",            # 美的、伊利
    "002714",                      # 牧原
    # 地产/基建
    "000002", "601668",            # 万科、中建
    # 交通运输
    "600009", "601111",            # 上海机场、国航
    # 化工/有色
    "600309", "603799",            # 万华、华友
    # 中小市值
    "002230", "300124",            # 科大讯飞、汇川
    "002475", "300274",            # 立讯、阳光电源
]


def _get_codes(config: dict) -> list[str]:
    cfg_codes = config.get("data_source", {}).get("codes", [])
    if cfg_codes:
        return list(dict.fromkeys(cfg_codes))  # 去重保序
    return list(dict.fromkeys(_DEFAULT_CODES))


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
    """从东方财富接口拉取（备用）。"""
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
            for col in ["开盘", "最高", "最低", "收盘", "成交量"]:
                if col in df.columns:
                    df[df.columns[df.columns.get_loc(col)]] = df[col].astype(float)
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["code", "trade_date"])


def _load_real_data(config: dict) -> tuple[dict[date, dict], str]:
    """从配置的数据源拉取数据。"""
    start = _parse_date(config["backtest"]["start_date"]) - timedelta(days=config["data_source"].get("lookback_days", 400))
    end = _parse_date(config["backtest"]["end_date"]) + timedelta(days=30)
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
        pre = float(row.get("pre_close", close * 0.99)) if "pre_close" in row else close * 0.99

        out.setdefault(td, {})[code] = {
            "open": float(row.get("open", close)),
            "high": float(row.get("high", close)),
            "low": float(row.get("low", close)),
            "close": close,
            "pre_close": max(pre, 0.01),
            "volume": float(row.get("volume", 0)),
            "amount": close * float(row.get("volume", 0)),
            "is_st": False,
            "is_suspended": False,
        }
    return out, "ok"


def _gen_simulated_data(config: dict) -> dict[date, dict]:
    """生成模拟数据（网络不可用时的降级方案）。"""
    from core.common.calendar import get_calendar

    sim = config.get("simulation", {})
    num_stocks = sim.get("num_stocks", 100)
    good_count = sim.get("good_stocks", 30)
    drift_range = sim.get("good_drift_range", [0.0003, 0.0015])
    vol_range = sim.get("vol_range", [0.015, 0.04])

    cal = get_calendar()
    start = _parse_date(config["backtest"]["start_date"])
    end = _parse_date(config["backtest"]["end_date"])
    trading_days = cal.get_trading_days(start, end)

    logger.info(
        "akshare 不可用，使用模拟数据 (%d只, %d个交易日, %d只是好股票)",
        num_stocks, len(trading_days), good_count,
    )

    rng = np.random.default_rng(42)
    result: dict[date, dict] = {}

    good_codes = [f"600{i:03d}" for i in range(1, good_count + 1)]
    normal_codes = [f"000{i:03d}" for i in range(1, num_stocks - good_count + 1)]

    for codes, is_good in [(good_codes, True), (normal_codes, False)]:
        for code in codes:
            base = rng.uniform(8, 200)
            drift = rng.uniform(*drift_range) if is_good else rng.uniform(-0.0005, 0.001)
            vol = rng.uniform(*vol_range)
            prices = base * np.cumprod(1 + rng.normal(drift, vol, len(trading_days)))
            for i, td in enumerate(trading_days):
                result.setdefault(td, {})[code] = {
                    "open": prices[i] * 0.99, "high": prices[i] * 1.02,
                    "low": prices[i] * 0.98, "close": prices[i],
                    "pre_close": prices[i] * 0.99,
                    "volume": 10_000_000, "amount": prices[i] * 10_000_000,
                    "is_st": False, "is_suspended": False,
                }

    logger.info("模拟数据生成完成: %d 个交易日", len(result))
    return result


# ══════════════════════════════════════════════════════════════
# 4. 因子计算
# ══════════════════════════════════════════════════════════════

def _build_feature_loader(raw_data: dict, config: dict):
    """根据 config 构建 feature_loader 回调。"""
    from core.common.calendar import get_calendar
    cal = get_calendar()

    factors = _enabled_factors(config)
    factor_settings = config.get("factor_settings", {})
    lookback = factor_settings.get("lookback_window", 60)
    min_points = factor_settings.get("min_price_points", 22)

    def loader(trade_date: date, codes: list[str]) -> pd.DataFrame:
        lookback_dates = cal.get_prev_n_trading_days(trade_date, lookback)
        lookback_dates = [d for d in lookback_dates if d in raw_data]

        # 构建价格序列
        price_hist: dict[str, list[float]] = {}
        vol_hist: dict[str, list[float]] = {}
        amount_hist: dict[str, list[float]] = {}
        high_hist: dict[str, list[float]] = {}
        low_hist: dict[str, list[float]] = {}
        open_hist: dict[str, list[float]] = {}
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
                    )
                    row[name] = float(val) if val is not None and not np.isnan(val) else 0.0
                except Exception:
                    row[name] = 0.0

            results.append(row)

        df = pd.DataFrame(results)
        return df.set_index("code") if not df.empty else df

    return loader


def _compute_factor_value(
    fg: dict,
    arr: np.ndarray, rets: np.ndarray,
    vols: np.ndarray, high: np.ndarray, low: np.ndarray,
    prev_close: np.ndarray, amounts: list,
) -> float | None:
    """在内存数据上直接计算因子值（不走 core/features 的完整实现，
    因为后者依赖 FeatureStore 和特定的 DataFrame 格式）。"""

    name = fg["name"]
    params = fg.get("params", {})

    try:
        if name == "momentum_20d":
            w = params.get("window", 20)
            return float((arr[-1] / arr[-min(w, len(arr))] - 1.0)) if len(arr) >= w + 1 else 0.0

        if name == "volatility_20d":
            w = params.get("window", 20)
            return float(np.nanstd(rets[-w:]) * np.sqrt(252)) if len(rets) >= 5 else 0.0

        if name == "rsi_14":
            w = params.get("window", 14)
            gains = np.maximum(rets[-w:], 0)
            losses = -np.minimum(rets[-w:], 0)
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return float(100.0 - 100.0 / (1.0 + rs))

        if name == "atr_14":
            w = params.get("window", 14)
            tr = np.maximum(high[-w:] - low[-w:],
                    np.maximum(np.abs(high[-w:] - prev_close[-w:]),
                               np.abs(low[-w:] - prev_close[-w:])))
            return float(tr.mean())

        if name == "amplitude_20d":
            w = params.get("window", 20)
            amp = (high[-w:] - low[-w:]) / prev_close[-w:]
            return float(amp.mean())

        if name == "turnover_5d":
            w = params.get("window", 5)
            if len(vols) >= w:
                denom = arr[-w:] if len(arr) >= w else arr
                return float(vols[-w:].mean() / denom.mean()) if denom.mean() > 0 else 0.0
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
            if ma_s > ma_m > ma_l:
                return 1.0
            elif ma_s < ma_m < ma_l:
                return -1.0
            return 0.0

        if name == "amihud_illiq":
            w = params.get("window", 20)
            if len(rets) < 2 or len(amounts) < 2:
                return 0.0
            illiq = np.abs(rets[-w:]) / np.maximum(np.array(amounts[-w:], dtype=float), 1e-8)
            return float(illiq.mean() * 1e8)

        # 基本面/资金面因子 — demo 中无财报/资金流数据，返回中性值
        if name in ("roe_ttm", "revenue_yoy"):
            return 0.0  # 需要 financials 数据，demo 默认返回中性

        if name in ("margin_balance_change_5d",):
            return 0.0  # 需要两融数据

        if name == "limit_up_ratio":
            return 0.0  # 需要全市场数据

        if name in ("pe_ttm", "pb"):
            return 0.0  # 需要 financials（PIT 模式）

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
        direction = "↓(越低越好)" if f["name"] in ConfigDrivenStrategy.INVERSE_FACTORS else "↑"
        logger.info("  %s (权重=%.2f) %s", f["name"], f.get("weight", 0), direction)

    # ── 数据加载 ──
    logger.info("正在加载数据...")
    raw_data, source_label = _load_real_data(config)

    if not raw_data or len(raw_data) < 100:
        logger.info("真实数据不可用 (%s)，切换到模拟数据", source_label)
        raw_data = _gen_simulated_data(config)

    if not raw_data:
        logger.error("无可用数据，退出")
        return

    # ── 构建回调 ──
    def data_loader(trade_date: date) -> pd.DataFrame:
        rows = [{"code": code, **fields} for code, fields in raw_data.get(trade_date, {}).items()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    feature_loader = _build_feature_loader(raw_data, config)

    # ── 回测 ──
    from core.backtest.engine import BacktestEngine

    engine = BacktestEngine(
        start_date=start_date, end_date=end_date,
        initial_capital=initial_capital,
    )
    strategy = ConfigDrivenStrategy(config)

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

    print(f"\n修改 configs/strategy.yaml 可调整因子、权重、持仓数、股票池等全部参数。")


if __name__ == "__main__":
    import sys as _sys
    path = _sys.argv[1] if len(_sys.argv) > 1 else "configs/strategy.yaml"
    main(path)
