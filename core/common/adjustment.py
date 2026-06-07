"""复权处理。

提供A股前复权价格计算，解决回测中的股价跳空问题。
复权因子来源于日线数据的 adj_factor 字段。
"""

from __future__ import annotations

from typing import List

import pandas as pd


def forward_adjust(
    df: pd.DataFrame,
    price_cols: List[str] | None = None,
    adj_factor_col: str = "adj_factor",
) -> pd.DataFrame:
    """对价格列执行前复权处理。

    前复权公式：adjusted_price = raw_price × adj_factor

    Args:
        df: 包含价格列和复权因子的 DataFrame，必须含 {adj_factor_col} 列。
        price_cols: 需要复权的价格列名列表。默认覆盖
            open, high, low, close, pre_close。
        adj_factor_col: 复权因子列名。

    Returns:
        复权后的 DataFrame（原地修改的副本），新增/覆写对应的价格列。

    Raises:
        ValueError: 如果 adj_factor_col 不在 df 中。
    """
    result = df.copy()

    if adj_factor_col not in result.columns:
        raise ValueError(f"复权因子列 '{adj_factor_col}' 不在 DataFrame 中")

    if price_cols is None:
        # 默认复权所有OHLCV相关价格列
        candidates = {"open", "high", "low", "close", "pre_close"}
        price_cols = [c for c in candidates if c in result.columns]

    adj = result[adj_factor_col]
    for col in price_cols:
        if col in result.columns:
            result[col] = result[col] * adj

    return result


def reverse_forward_adjusted(
    df: pd.DataFrame,
    price_cols: List[str] | None = None,
    adj_factor_col: str = "adj_factor",
) -> pd.DataFrame:
    """从已前复权的价格反推原始价格。

    逆公式：raw_price = adjusted_price / adj_factor

    Args:
        df: 包含已复权价格列和复权因子的 DataFrame。
        price_cols: 需要反推的价格列名。
        adj_factor_col: 复权因子列名。

    Returns:
        恢复为原始价格的 DataFrame 副本。
    """
    result = df.copy()

    if price_cols is None:
        candidates = {"open", "high", "low", "close", "pre_close"}
        price_cols = [c for c in candidates if c in result.columns]

    adj = result[adj_factor_col].replace(0, float("nan"))
    for col in price_cols:
        if col in result.columns:
            result[col] = result[col] / adj

    return result


def compute_adj_factor_from_raw(
    df: pd.DataFrame,
    close_col: str = "close",
    adj_close_col: str = "close_adj",
) -> pd.Series:
    """从原始收盘价和复权收盘价推算复权因子。

    adj_factor = adj_close / close

    当原始数据源不提供 adj_factor 但提供了复权收盘价时使用。
    """
    return df[adj_close_col] / df[close_col]


def align_adjustment_factors(
    *dataframes: pd.DataFrame,
    code_col: str = "code",
    date_col: str = "trade_date",
    adj_col: str = "adj_factor",
) -> list[pd.DataFrame]:
    """对齐多个数据源的复权因子，确保全系统统一的复权基准。

    以第一个 DataFrame 的 adj_factor 为基准，将其余 DataFrame
    的价格按比例缩放以对齐。用于切换数据源时的一致性保证。
    """
    if len(dataframes) < 2:
        return list(dataframes)

    ref = dataframes[0]
    aligned = [ref]

    for df in dataframes[1:]:
        merged = df.merge(
            ref[[code_col, date_col, adj_col]],
            on=[code_col, date_col],
            how="left",
            suffixes=("", "_ref"),
        )
        scale = merged[adj_col + "_ref"] / merged[adj_col]
        for col in ["open", "high", "low", "close"]:
            if col in merged.columns:
                merged[col] = merged[col] * scale
        merged.drop(columns=[adj_col + "_ref"], inplace=True)
        aligned.append(merged)

    return aligned
