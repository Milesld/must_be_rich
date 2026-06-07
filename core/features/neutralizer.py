"""因子中性化工具。

对因子值进行行业和市值中性化处理，剔除行业归属和规模效应后
的残差作为"纯因子暴露"——更干净、更稳健的Alpha来源。

支持：
- 行业中性化（申万一级/中信一级/GICS）
- 市值中性化（log(market_cap)）
- 行业+市值双重中性化
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# 行业分类方案名称
INDUSTRY_CLASSIFICATIONS = {
    "sw_l1": "申万一级",
    "citics_l1": "中信一级",
    "gics": "GICS",
}


def neutralize_by_industry(
    factor_values: pd.Series,
    industry_labels: pd.Series,
) -> pd.Series:
    """行业中性化。

    factor_values ~ industry_dummies → 取残差。
    残差表示"同行业内该股票的因子值偏离行业均值的程度"。

    Args:
        factor_values: 因子值序列（index 与 industry_labels 对齐）。
        industry_labels: 行业分类标签。

    Returns:
        行业中性化后的残差序列。
    """
    if factor_values.empty or industry_labels.empty:
        return factor_values

    # 去除因子值和标签同时为空的样本
    valid_mask = factor_values.notna() & industry_labels.notna()
    if valid_mask.sum() < 10:
        logger.debug("有效样本不足(<10)，跳过行业中性化")
        return factor_values

    y = factor_values[valid_mask].values.reshape(-1, 1)
    dummies = pd.get_dummies(industry_labels[valid_mask], drop_first=True).astype(float)
    X = dummies.values

    if X.shape[1] == 0:
        return factor_values

    model = LinearRegression()
    model.fit(X, y)
    residuals = y.flatten() - model.predict(X)

    result = pd.Series(np.nan, index=factor_values.index)
    result[valid_mask[valid_mask].index] = residuals
    return result


def neutralize_by_market_cap(
    factor_values: pd.Series,
    market_caps: pd.Series,
) -> pd.Series:
    """市值中性化。

    factor_values ~ log(market_cap) → 取残差。
    残差表示"剔除规模效应后的纯因子暴露"。

    Args:
        factor_values: 因子值序列。
        market_caps: 市值序列（元）。

    Returns:
        市值中性化后的残差序列。
    """
    if factor_values.empty or market_caps.empty:
        return factor_values

    valid_mask = factor_values.notna() & market_caps.notna() & (market_caps > 0)
    if valid_mask.sum() < 10:
        logger.debug("有效样本不足(<10)，跳过市值中性化")
        return factor_values

    y = factor_values[valid_mask].values.reshape(-1, 1)
    log_mcap = np.log(market_caps[valid_mask].astype(float))
    X = log_mcap.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    residuals = y.flatten() - model.predict(X)

    result = pd.Series(np.nan, index=factor_values.index)
    result[valid_mask[valid_mask].index] = residuals
    return result


def neutralize_industry_mcap(
    factor_values: pd.Series,
    industry_labels: pd.Series,
    market_caps: pd.Series,
) -> pd.Series:
    """行业+市值双重中性化。

    factor_values ~ industry_dummies + log(market_cap) → 取残差。

    Args:
        factor_values: 因子值序列。
        industry_labels: 行业分类标签。
        market_caps: 市值序列（元）。

    Returns:
        双重中性化后的残差序列。
    """
    if factor_values.empty:
        return factor_values

    valid_mask = (
        factor_values.notna()
        & industry_labels.notna()
        & market_caps.notna()
        & (market_caps > 0)
    )
    if valid_mask.sum() < 10:
        logger.debug("有效样本不足(<10)，跳过行业+市值中性化")
        return factor_values

    y = factor_values[valid_mask].values.reshape(-1, 1)
    dummies = pd.get_dummies(industry_labels[valid_mask], drop_first=True).astype(float)
    log_mcap = np.log(market_caps[valid_mask].astype(float)).values.reshape(-1, 1)
    X = np.hstack([dummies.values, log_mcap])

    model = LinearRegression()
    model.fit(X, y)
    residuals = y.flatten() - model.predict(X)

    result = pd.Series(np.nan, index=factor_values.index)
    result[valid_mask[valid_mask].index] = residuals
    return result


def mad_winsorize(
    series: pd.Series,
    n_mad: float = 5.0,
) -> pd.Series:
    """MAD 缩尾处理：将超过 ±n_mad × MAD 的值裁剪到边界。

    中位数绝对偏差 (MAD) 比标准差更稳健，不受极端值影响。

    Args:
        series: 需要去极值的序列。
        n_mad: MAD 倍数阈值（默认5）。

    Returns:
        缩尾后的序列。
    """
    if series.empty:
        return series

    median = series.median()
    mad = (series - median).abs().median()

    if mad == 0:
        return series

    lower = median - n_mad * mad
    upper = median + n_mad * mad

    return series.clip(lower, upper)
