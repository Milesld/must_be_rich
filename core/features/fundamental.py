"""基本面因子计算。

★所有因子默认启用 Point-in-Time 模式★：

在回测的每个时间点，只使用 announce_date <= 该时间点的财报数据。
这意味着不能使用"提前知道"的未来财报信息——防止前视偏差。

PIT 模式通过 market_daily 数据推断回测日期，并检查 financials 数据中
的 announce_date 列（如果存在）。如果 financials 中无 announce_date 列，
且 point_in_time=True，则记录警告并使用 report_date 作为近似（不够严格）。

统一签名：f(data: pd.DataFrame, params: dict) -> pd.Series
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _pit_filter(
    data: pd.DataFrame,
    financials: pd.DataFrame,
    point_in_time: bool,
) -> pd.DataFrame:
    """如果启用PIT且有announce_date列，过滤掉尚未披露的财报。"""
    if not point_in_time:
        return financials

    if "announce_date" not in financials.columns:
        logger.warning(
            "PIT模式已启用但 financials 缺少 announce_date 列，"
            "使用 report_date 作为近似（不够严格，可能存在前视偏差风险）"
        )
        return financials

    # ★ 数据通常包含单一 trade_date（因子按天调用），用 max 等价于该日
    # 如有跨日数据也不误用：每行的 trade_date 是它自身的 PIT 截止点
    # 统一转换为 date 类型
    if "announce_date" not in financials.columns:
        return financials
    fin = financials.copy()
    fin["_ann_dt"] = pd.to_datetime(fin["announce_date"]).apply(
        lambda x: x.date() if hasattr(x, "date") else x
    )
    data_dates = pd.to_datetime(data["trade_date"]).apply(
        lambda x: x.date() if hasattr(x, "date") else x
    )
    pit_date = data_dates.max()

    filtered = fin[fin["_ann_dt"] <= pit_date].drop(columns=["_ann_dt"])
    skipped = len(financials) - len(filtered)
    if skipped > 0:
        logger.debug(
            "PIT过滤: 跳过 %d 条 announce_date > %s 的财报", skipped, pit_date,
        )
    return filtered


def _merge_financials(
    data: pd.DataFrame,
    financials: pd.DataFrame,
    point_in_time: bool = True,
) -> pd.DataFrame:
    """将财报数据按 code 合并到行情数据上，应用 PIT 过滤。"""
    fin = _pit_filter(data, financials, point_in_time)
    if fin.empty:
        # 无可合并的财报 — 返回 NaN 列
        for col in financials.columns:
            if col not in ("code", "announce_date", "report_date"):
                data[col] = np.nan
        return data
    # ★ 用 merge_asof 对齐：对每行行情取最近一次已公告的财报，避免多对多行爆炸
    # 前提：fin 需按 (code, announce_date) 排序
    data = data.sort_values(["code", "trade_date"])
    fin = fin.sort_values(["code", "announce_date"]) if "announce_date" in fin.columns else fin
    merged = pd.merge_asof(
        data, fin,
        by="code",
        left_on="trade_date",
        right_on="announce_date" if "announce_date" in fin.columns else "report_date",
        direction="backward",
        suffixes=("", "_fin"),
    )
    return merged


# ── 估值因子 ──────────────────────────────────────

def pe_ttm(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """市盈率 = 股价 / 每股收益(TTM)。

    优先从 financials 中获取 pe_ttm；否则使用 market_cap / net_profit 近似。
    """
    pt = params.get("point_in_time", True)
    if financials is not None and "pe_ttm" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["pe_ttm"]
    # 近似：市值/净利润
    if financials is not None and "net_profit" in financials.columns and "market_cap" in data.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["market_cap"] / merged["net_profit"].replace(0, float("nan"))
    return pd.Series(np.nan, index=data.index)


def pb(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """市净率 = 股价 / 每股净资产。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "pb" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["pb"]
    if financials is not None and "total_equity" in financials.columns and "market_cap" in data.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["market_cap"] / merged["total_equity"].replace(0, float("nan"))
    return pd.Series(np.nan, index=data.index)


def ps_ttm(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """市销率。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "revenue" in financials.columns and "market_cap" in data.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["market_cap"] / merged["revenue"].replace(0, float("nan"))
    return pd.Series(np.nan, index=data.index)


def pcf_ttm(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """市现率 = 市值 / 经营现金流。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "operating_cf" in financials.columns and "market_cap" in data.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["market_cap"] / merged["operating_cf"].replace(0, float("nan"))
    return pd.Series(np.nan, index=data.index)


def peg(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """PEG = PE / 盈利增速。"""
    pe = pe_ttm(data, params, financials)
    if financials is not None and "net_profit_yoy" in financials.columns:
        pt = params.get("point_in_time", True)
        merged = _merge_financials(data, financials, pt)
        growth = merged["net_profit_yoy"].replace(0, float("nan"))
        return pe / growth
    return pd.Series(np.nan, index=data.index)


def dividend_yield(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """股息率 = 每股股息 / 股价。"""
    if financials is not None and "dividend_per_share" in financials.columns:
        pt = params.get("point_in_time", True)
        merged = _merge_financials(data, financials, pt)
        return merged["dividend_per_share"] / merged["close"]
    return pd.Series(np.nan, index=data.index)


def ep_ttm(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """盈利收益率 = 1/PE。PE越低，EP越高 → 估值越低。"""
    pe = pe_ttm(data, params, financials)
    return 1.0 / pe.replace(0, float("nan"))


# ── 盈利质量 ──────────────────────────────────────

def roe_ttm(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """ROE(TTM) = 净利润 / 净资产。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "roe" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["roe"]
    if financials is not None and "net_profit" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["net_profit"] / merged["total_equity"].replace(0, float("nan"))
    return pd.Series(np.nan, index=data.index)


def roa_ttm(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """ROA = 净利润 / 总资产。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "roa" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["roa"]
    if financials is not None:
        merged = _merge_financials(data, financials, pt)
        return merged["net_profit"] / merged["total_assets"].replace(0, float("nan"))
    return pd.Series(np.nan, index=data.index)


def roic_ttm(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """ROIC = 息前税后利润 / 投入资本。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "roic" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["roic"]
    # 近似：NOPAT / (总资产 - 无息负债)
    if financials is not None:
        merged = _merge_financials(data, financials, pt)
        nopat = merged["net_profit"] * 1.25  # 粗估税后经营利润
        ic = merged["total_assets"] - (merged.get("accounts_payable", 0))
        return nopat / ic.replace(0, float("nan"))
    return pd.Series(np.nan, index=data.index)


def gross_margin_trend(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """毛利率近4季度趋势（线性回归斜率）。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "gross_margin" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        # 按 code 分组，取最近4季度毛利率的斜率
        def _slope(series: pd.Series) -> float:
            if len(series) < 2:
                return 0.0
            x = np.arange(len(series))
            return np.polyfit(x, series.values, 1)[0] if len(x) > 1 else 0.0

        result = merged.groupby("code")["gross_margin"].rolling(
            4, min_periods=2
        ).apply(_slope).droplevel(0)
        return result
    return pd.Series(np.nan, index=data.index)


def net_margin_ttm(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """净利率。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "net_margin" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["net_margin"]
    return pd.Series(np.nan, index=data.index)


# ── 成长性 ────────────────────────────────────────

def revenue_yoy(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """营收同比增速。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "revenue_yoy" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["revenue_yoy"]
    return pd.Series(np.nan, index=data.index)


def net_profit_yoy(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """净利润同比增速。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "net_profit_yoy" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["net_profit_yoy"]
    return pd.Series(np.nan, index=data.index)


# ── 财务健康 ──────────────────────────────────────

def debt_ratio(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """资产负债率。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "debt_ratio" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["debt_ratio"]
    return pd.Series(np.nan, index=data.index)


def current_ratio(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """流动比率。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "current_ratio" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["current_ratio"]
    return pd.Series(np.nan, index=data.index)


def quick_ratio(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """速动比率。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "quick_ratio" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["quick_ratio"]
    return pd.Series(np.nan, index=data.index)


# ── 现金流 ──────────────────────────────────────

def cf_ratio_ttm(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """经营现金流/净利润——盈利质量检查。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "cf_ratio" in financials.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["cf_ratio"]
    if financials is not None:
        merged = _merge_financials(data, financials, pt)
        return merged["operating_cf"] / merged["net_profit"].replace(0, float("nan"))
    return pd.Series(np.nan, index=data.index)


def free_cf_yield(data: pd.DataFrame, params: dict, financials: Optional[pd.DataFrame] = None) -> pd.Series:
    """自由现金流/市值。"""
    pt = params.get("point_in_time", True)
    if financials is not None and "free_cf" in financials.columns and "market_cap" in data.columns:
        merged = _merge_financials(data, financials, pt)
        return merged["free_cf"] / merged["market_cap"].replace(0, float("nan"))
    return pd.Series(np.nan, index=data.index)
