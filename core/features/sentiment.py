"""情绪/事件因子计算。

统一签名：f(data: pd.DataFrame, params: dict) -> pd.Series

许多情绪因子是市场级指标（全市场统计），返回结果按 data 的 index 对齐。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# ── 市场宽度/情绪温度计 ────────────────────────────

def limit_up_count(data: pd.DataFrame, params: dict) -> pd.Series:
    """全市场涨停家数（按日期对齐）。"""
    # 涨停判断：当日涨跌幅 >= 对应板块的涨停限制
    if "pct_change" in data.columns:
        pct = data["pct_change"]
    else:
        pct = (data["close"] - data["pre_close"]) / data["pre_close"].replace(0, float("nan"))

    # 默认涨跌停判断：>= 9.9%（主板）/ >=19.9%（科创创业板）
    # 简化版：使用10%和20%两个阈值
    limit_up_mask = (
        (pct >= 0.098) & (pct <= 0.105)   # 主板 ±10%
    ) | (
        (pct >= 0.198) & (pct <= 0.205)   # 科创/创业板 ±20%
    ) | (
        (pct >= 0.298) & (pct <= 0.305)   # 北交所 ±30%
    )

    counts = limit_up_mask.groupby(data["trade_date"]).transform("sum")
    return counts


def limit_up_chain_height(data: pd.DataFrame, params: dict) -> pd.Series:
    """最高连板高度（全市场当日值，按日期对齐）。"""
    # 需要 consecutive_limit_up 列（前一交易日是否涨停）
    if "consecutive_limit_up" not in data.columns:
        # 近似：当天涨停且昨天涨停 = 连板
        pct = (data["close"] - data["pre_close"]) / data["pre_close"].replace(0, float("nan"))
        today_limit_up = (pct >= 0.098)
        # 简化：按 code 分组检查连续涨停天数
        chain = today_limit_up.groupby(data["code"]).transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
        )
        max_chain = chain.groupby(data["trade_date"]).transform("max")
        return max_chain

    chain = data["consecutive_limit_up"]
    return chain.groupby(data["trade_date"]).transform("max")


def limit_down_count(data: pd.DataFrame, params: dict) -> pd.Series:
    """全市场跌停家数。"""
    if "pct_change" in data.columns:
        pct = data["pct_change"]
    else:
        pct = (data["close"] - data["pre_close"]) / data["pre_close"].replace(0, float("nan"))

    limit_down_mask = (
        (pct <= -0.098) & (pct >= -0.105)
    ) | (
        (pct <= -0.198) & (pct >= -0.205)
    )

    counts = limit_down_mask.groupby(data["trade_date"]).transform("sum")
    return counts


def board_break_ratio(data: pd.DataFrame, params: dict) -> pd.Series:
    """炸板率 = 曾涨停后开板数 / 曾涨停数。

    依赖列：touched_limit_up（是否盘中曾触及涨停）、limit_up（是否最终封涨停）。
    如果列不可用，返回 NaN。
    """
    if "touched_limit_up" not in data.columns or "limit_up" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    touched = data["touched_limit_up"].astype(bool)
    sealed = data["limit_up"].astype(bool)

    total_touched = touched.groupby(data["trade_date"]).transform("sum").replace(0, float("nan"))
    total_sealed = sealed.groupby(data["trade_date"]).transform("sum")

    broke = (total_touched - total_sealed)
    return broke / total_touched


def limit_up_ratio(data: pd.DataFrame, params: dict) -> pd.Series:
    """涨停家数/上涨家数——赚钱效应代理指标。"""
    if "pct_change" in data.columns:
        pct = data["pct_change"]
    else:
        pct = (data["close"] - data["pre_close"]) / data["pre_close"].replace(0, float("nan"))

    limit_up_mask = (pct >= 0.098) | (pct >= 0.198)
    up_mask = pct > 0

    limit_up_count = limit_up_mask.groupby(data["trade_date"]).transform("sum")
    up_count = up_mask.groupby(data["trade_date"]).transform("sum").replace(0, float("nan"))

    return limit_up_count / up_count


# ── 集合竞价 ─────────────────────────────────────

def auction_open_premium(data: pd.DataFrame, params: dict) -> pd.Series:
    """集合竞价虚拟开盘价相对昨收的涨跌幅。

    依赖列：auction_price（虚拟开盘价）。

    params: {}
    """
    if "auction_price" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    return (data["auction_price"] - data["pre_close"]) / data["pre_close"].replace(0, float("nan"))


def auction_volume_ratio(data: pd.DataFrame, params: dict) -> pd.Series:
    """竞价匹配量/近20日均量。

    依赖列：auction_volume（竞价匹配量）。

    params: {avg_window: 20}
    """
    if "auction_volume" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    window = params.get("avg_window", 20)
    vol = data["auction_volume"]
    avg_vol = data["volume"].groupby(data["code"]).rolling(window).mean().droplevel(0)
    avg_vol = avg_vol.replace(0, float("nan"))
    return vol / avg_vol


# ── 业绩预告 ─────────────────────────────────────

def performance_forecast_surprise(data: pd.DataFrame, params: dict) -> pd.Series:
    """业绩预告超预期程度的量化映射。

    基于业绩预告类型：大增=+2, 略增=+1, 续盈=+0.5, 略减=-1, 大减=-2,
    首亏=-3, 扭亏=+3。

    依赖列：forecast_type（业绩预告类型字符串）。

    params: {}
    """
    if "forecast_type" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    score_map = {
        "大增": 2.0, "预增": 2.0, "大幅上升": 2.0,
        "扭亏": 3.0, "扭亏为盈": 3.0,
        "略增": 1.0, "预盈": 1.0, "续盈": 0.5,
        "略减": -1.0, "预减": -1.0,
        "大减": -2.0, "大幅下降": -2.0,
        "首亏": -3.0, "续亏": -2.0,
    }

    result = data["forecast_type"].map(score_map).fillna(0.0)
    return result
