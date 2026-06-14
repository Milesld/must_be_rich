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

    def __init__(self, config: dict, raw_data: dict | None = None) -> None:
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
            self._raw_data = raw_data
            logger.info("市场状态判断已启用 (仓位范围: %.0f%%~%.0f%%)",
                        self._regime_min_position * 100, self._regime_max_position * 100)
        else:
            self._regime_detector = None
            self._regime_min_position = 1.0
            self._regime_max_position = 1.0
            self._raw_data = None

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
                self._current_regime = self._regime_detector.detect(trade_date, market_df)
                raw_ratio = self._current_regime.suggested_position_ratio
                suggested_pos_ratio = max(self._regime_min_position,
                                          min(self._regime_max_position, raw_ratio))
                regime_info = f" | 市场状态: {self._current_regime.regime_label} (建议仓位: {suggested_pos_ratio:.0%})"

        if not self._should_rebalance(trade_date):
            # 非调仓日也检查：如果市场急转直下，可能需要紧急减仓
            if self._regime_detector is not None and suggested_pos_ratio < 0.50:
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

        # 卖出不在目标池的持仓
        for code, shares in list(positions.items()):
            if shares > 0 and code not in target_codes:
                price = float(daily_data.get(code, {}).get("close", 0))
                if price > 0:
                    intents.append(TradeIntent(
                        signal_id=f"sell_{code}_{trade_date}",
                        code=code, side="sell", price=price, shares=shares,
                    ))

        # 如果当前仓位已经超过目标，不再买入
        if pos_value >= target_pos_value * 0.95:
            return intents

        # 可用买入资金 = min(cash, target_pos_value - current_pos_value)
        buy_budget = min(cash, max(0, target_pos_value - pos_value))

        # 等权买入目标池中的新股
        new_codes = [c for c in target_codes if positions.get(c, 0) == 0]
        if new_codes and buy_budget > 0:
            per_stock_cash = buy_budget / len(new_codes)
            for code in new_codes:
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
            if name in self.INVERSE_FACTORS:
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
# 4. 因子计算
# ══════════════════════════════════════════════════════════════

def _build_feature_loader(raw_data: dict, config: dict, financials: dict[str, dict[str, float]] | None = None):
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
                        financials.get(code, {}) if financials else {},
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
    financials: dict[str, float] | None = None,
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
                    "limit_up_count", "limit_up_chain_height", "limit_down_count",
                    "board_break_ratio", "limit_up_ratio",
                    "auction_open_premium", "auction_volume_ratio",
                    "performance_forecast_surprise",
                    "overnight_adr_mapped", "a50_futures_overnight",
                    "hsi_futures_overnight", "announcement_sentiment_score",
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

    feature_loader = _build_feature_loader(raw_data, config, financials)

    # ── 回测 ──
    from core.backtest.engine import BacktestEngine

    engine = BacktestEngine(
        start_date=start_date, end_date=end_date,
        initial_capital=initial_capital,
    )
    strategy = ConfigDrivenStrategy(config, raw_data)

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
