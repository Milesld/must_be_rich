"""技术面因子计算。

所有因子函数遵循统一签名：f(data: pd.DataFrame, params: dict) -> pd.Series

data 必须包含列：code, trade_date, open, high, low, close, volume, amount, adj_factor
返回 index 与 data 对齐的 pd.Series。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# ── 动量因子 ──────────────────────────────────────

def momentum(data: pd.DataFrame, params: dict) -> pd.Series:
    """N日累计收益率。

    params: {window: 20}
    """
    window = params.get("window", 20)
    close_adj = _safe_col(data, "close")
    adj = _safe_col(data, "adj_factor", default=1.0)
    price_adj = close_adj * adj
    result = price_adj.groupby(data["code"]).pct_change(periods=window)
    return result


def alpha_momentum(data: pd.DataFrame, params: dict) -> pd.Series:
    """剔除市场beta后的纯alpha动量。

    简化实现：个股N日收益率 - 等权市场平均N日收益率。
    完整实现需要滚动beta回归。

    params: {window: 20}
    """
    window = params.get("window", 20)
    close_adj = _safe_col(data, "close") * _safe_col(data, "adj_factor", default=1.0)
    mom = close_adj.groupby(data["code"]).pct_change(periods=window)
    # 市场平均动量（等权）
    market_mom = mom.groupby(data["trade_date"]).transform("mean")
    return mom - market_mom


# ── 波动率因子 ──────────────────────────────────

def volatility(data: pd.DataFrame, params: dict) -> pd.Series:
    """N日年化波动率。

    params: {window: 20}
    """
    window = params.get("window", 20)
    returns = _daily_returns(data)
    vol = returns.groupby(data["code"]).rolling(window).std().droplevel(0)
    return vol * np.sqrt(252)  # 年化


def amplitude(data: pd.DataFrame, params: dict) -> pd.Series:
    """N日平均振幅 = AVG((high-low)/pre_close)。

    params: {window: 20}
    """
    window = params.get("window", 20)
    amp = (data["high"] - data["low"]) / data["pre_close"]
    return amp.groupby(data["code"]).rolling(window).mean().droplevel(0)


def atr(data: pd.DataFrame, params: dict) -> pd.Series:
    """平均真实波幅 (ATR)。

    TR = max(high-low, |high-prev_close|, |low-prev_close|)

    params: {window: 14}
    """
    window = params.get("window", 14)
    high, low, pre_close = data["high"], data["low"], data["pre_close"]
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - pre_close), np.abs(low - pre_close),
    ))
    tr_series = pd.Series(tr, index=data.index)
    return tr_series.groupby(data["code"]).rolling(window).mean().droplevel(0)


# ── 成交量/流动性因子 ─────────────────────────────

def turnover(data: pd.DataFrame, params: dict) -> pd.Series:
    """N日平均换手率。

    params: {window: 5}
    """
    window = params.get("window", 5)
    turnover_val = _safe_col(data, "turnover", default=0.0)
    result = turnover_val.groupby(data["code"]).rolling(window).mean().droplevel(0)
    return result


def volume_ratio(data: pd.DataFrame, params: dict) -> pd.Series:
    """量比 = 当日成交量 / N日均量。

    params: {window: 20}
    """
    window = params.get("window", 20)
    volume = data["volume"]
    avg_vol = volume.groupby(data["code"]).rolling(window).mean().droplevel(0)
    return volume / avg_vol.replace(0, float("nan"))


def amihud_illiq(data: pd.DataFrame, params: dict) -> pd.Series:
    """Amihud非流动性指标 = |return| / amount（取对数）。

    值越大 → 流动性越差。

    params: {window: 20}  — 取N日均值
    """
    window = params.get("window", 20)
    returns = _daily_returns(data).abs()
    amount = data["amount"].replace(0, float("nan"))
    illiq = returns / amount
    return illiq.groupby(data["code"]).rolling(window).mean().droplevel(0) * 1e8


# ── 技术指标 ────────────────────────────────────

def rsi(data: pd.DataFrame, params: dict) -> pd.Series:
    """相对强弱指标 (RSI)。

    params: {window: 14}
    """
    window = params.get("window", 14)
    returns = _daily_returns(data)
    gain = returns.clip(lower=0)
    loss = (-returns).clip(lower=0)

    avg_gain = gain.groupby(data["code"]).rolling(window).mean().droplevel(0)
    avg_loss = loss.groupby(data["code"]).rolling(window).mean().droplevel(0)
    avg_loss = avg_loss.replace(0, float("nan"))
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(data: pd.DataFrame, params: dict, component: str = "dif") -> pd.Series:
    """MACD 指标。

    params: {fast: 12, slow: 26, signal: 9}
    component: 'dif' | 'dea' | 'hist'
    """
    fast_w = params.get("fast", 12)
    slow_w = params.get("slow", 26)
    signal_w = params.get("signal", 9)
    close_adj = _safe_col(data, "close") * _safe_col(data, "adj_factor", default=1.0)

    ema_fast = close_adj.groupby(data["code"]).transform(
        lambda x: x.ewm(span=fast_w, adjust=False).mean()
    )
    ema_slow = close_adj.groupby(data["code"]).transform(
        lambda x: x.ewm(span=slow_w, adjust=False).mean()
    )
    dif = ema_fast - ema_slow
    dea = dif.groupby(data["code"]).transform(
        lambda x: x.ewm(span=signal_w, adjust=False).mean()
    )

    if component == "dif":
        return dif
    elif component == "dea":
        return dea
    else:  # hist
        return (dif - dea) * 2  # 柱状图通常×2


def macd_dif(data: pd.DataFrame, params: dict) -> pd.Series:
    return macd(data, params, "dif")


def macd_dea(data: pd.DataFrame, params: dict) -> pd.Series:
    return macd(data, params, "dea")


def macd_hist(data: pd.DataFrame, params: dict) -> pd.Series:
    return macd(data, params, "hist")


def bollinger_position(data: pd.DataFrame, params: dict) -> pd.Series:
    """布林带位置 = (close - lower_band) / (upper_band - lower_band)。

    0 = 在下轨，0.5 = 中线，1 = 在上轨。

    params: {window: 20, num_std: 2}
    """
    window = params.get("window", 20)
    num_std = params.get("num_std", 2)
    close_adj = _safe_col(data, "close") * _safe_col(data, "adj_factor", default=1.0)

    mid = close_adj.groupby(data["code"]).rolling(window).mean().droplevel(0)
    std = close_adj.groupby(data["code"]).rolling(window).std().droplevel(0)
    upper = mid + num_std * std
    lower = mid - num_std * std

    denom = (upper - lower).replace(0, float("nan"))
    return (close_adj - lower) / denom


def ma_alignment(data: pd.DataFrame, params: dict) -> pd.Series:
    """均线多头/空头排列评分。

    MA5 > MA20 > MA60 → 多头排列 → +1
    MA5 < MA20 < MA60 → 空头排列 → -1
    其他 → 0 (交叉/震荡)

    params: {short: 5, mid: 20, long: 60}
    """
    short_w = params.get("short", 5)
    mid_w = params.get("mid", 20)
    long_w = params.get("long", 60)
    close_adj = _safe_col(data, "close") * _safe_col(data, "adj_factor", default=1.0)

    def _ma(window: int) -> pd.Series:
        return close_adj.groupby(data["code"]).rolling(window).mean().droplevel(0)

    ma_short = _ma(short_w)
    ma_mid = _ma(mid_w)
    ma_long = _ma(long_w)

    # 多头=1, 空头=-1, 其他=0
    score = pd.Series(0.0, index=data.index)
    score[(ma_short > ma_mid) & (ma_mid > ma_long)] = 1.0
    score[(ma_short < ma_mid) & (ma_mid < ma_long)] = -1.0
    return score


# ── 盘口因子（需L2，此处提供基础版本）─────────────

def order_imbalance(data: pd.DataFrame, params: dict) -> pd.Series:
    """订单不平衡 = (买量 - 卖量) / (买量 + 卖量)。

    L2 数据不可用时返回 NaN（静默降级）。

    params: {}
    """
    if "bid_volume" not in data.columns or "ask_volume" not in data.columns:
        return pd.Series(np.nan, index=data.index)
    buy = data["bid_volume"]
    sell = data["ask_volume"]
    total = buy + sell
    total = total.replace(0, float("nan"))
    return (buy - sell) / total


# ── 辅助 ──────────────────────────────────────

def _daily_returns(data: pd.DataFrame) -> pd.Series:
    """计算日收益率（基于收盘价×复权因子）。"""
    close_adj = data["close"] * _safe_col(data, "adj_factor", default=1.0)
    return close_adj.groupby(data["code"]).pct_change()


def _safe_col(data: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """安全获取列；列不存在时返回默认值。"""
    if col in data.columns:
        return data[col]
    return pd.Series(default, index=data.index)
