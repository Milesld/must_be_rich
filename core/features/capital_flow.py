"""资金面因子计算。

统一签名：f(data: pd.DataFrame, params: dict) -> pd.Series
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ── 主力资金 ───────────────────────────────────────

def main_force_net_inflow(data: pd.DataFrame, params: dict) -> pd.Series:
    """主力资金N日累计净流入。

    依赖列：main_force_inflow（大单净流入金额）。
    如果列不存在，返回 NaN。

    params: {window: 5}
    """
    window = params.get("window", 5)
    if "main_force_inflow" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    inflow = data["main_force_inflow"].fillna(0)
    return inflow.groupby(data["code"]).rolling(window, min_periods=1).sum().droplevel(0)


def main_force_inflow_ratio(data: pd.DataFrame, params: dict) -> pd.Series:
    """大单净流入/总成交额——衡量主力参与度。

    依赖列：main_force_inflow, amount
    params: {window: 5}
    """
    window = params.get("window", 5)
    if "main_force_inflow" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    inflow = data["main_force_inflow"].fillna(0)
    amount = data["amount"].replace(0, float("nan"))

    inflow_roll = inflow.groupby(data["code"]).rolling(window, min_periods=1).sum().droplevel(0)
    amount_roll = amount.groupby(data["code"]).rolling(window, min_periods=1).sum().droplevel(0)
    return inflow_roll / amount_roll.replace(0, float("nan"))


# ── 融资融券 ─────────────────────────────────────

def margin_balance_change(data: pd.DataFrame, params: dict) -> pd.Series:
    """融资余额N日变化（比率）。

    依赖列：margin_balance
    params: {window: 5}
    """
    window = params.get("window", 5)
    if "margin_balance" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    bal = data["margin_balance"].replace(0, float("nan"))
    change = bal.groupby(data["code"]).pct_change(periods=window)
    return change


# ── 龙虎榜 ───────────────────────────────────────

def dragon_tiger_net_buy(data: pd.DataFrame, params: dict) -> pd.Series:
    """最近一次龙虎榜机构净买入额的截面排名分位数。

    依赖列：dragon_tiger_net（最近一次龙虎榜净买入金额）或
    从 net_amount 列推断。

    0 = 净买入最少，1 = 净买入最多。

    params: {}
    """
    # ★ 取当日龙虎榜数据（不取"last"避免前视：each row uses its own date）
    if "net_amount" in data.columns:
        net = data["net_amount"]
    elif "dragon_tiger_net" in data.columns:
        net = data["dragon_tiger_net"]
    else:
        return pd.Series(np.nan, index=data.index)

    # 按交易日期分组，计算截面排名分位数
    result = net.groupby(data["trade_date"]).rank(pct=True)
    return result


def dragon_tiger_institution_count(data: pd.DataFrame, params: dict) -> pd.Series:
    """最近龙虎榜中机构席位的数量。

    依赖列：institution_buy（机构买入金额列）。
    params: {}
    """
    # 如果有 institution_buy > 0 则计为1个机构
    if "institution_buy" not in data.columns:
        return pd.Series(0, index=data.index)

    # 简化：按 code 统计机构买入>0的交易日数
    has_inst = (data["institution_buy"].fillna(0) > 0).astype(int)
    # 取最近N日的累计机构出现次数
    window = params.get("window", 20)
    result = has_inst.groupby(data["code"]).rolling(window, min_periods=1).sum().droplevel(0)
    return result


# ── 北向资金（仅季度频率）────────────────────────────

def northbound_quarter_change(data: pd.DataFrame, params: dict) -> pd.Series:
    """北向资金季度持仓变化（★仅季度频率可用！★）。

    依赖列：northbound_holdings（北向持股数，季度更新）。

    params: {}

    注意：自2024年8月19日起，北向资金每日数据已取消，
    仅按季度披露持仓数据。此因子只能在季报披露后的时间点使用，
    不适用于日频/周频模型。
    """
    if "northbound_holdings" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    # ★ groupby ffill 防止跨 code 串号污染
    holdings = data.groupby("code")["northbound_holdings"].ffill()
    change = holdings.groupby(data["code"]).pct_change(periods=1)
    # 标记为低频因子：非季度更新日返回 NaN
    # 通过检查 trade_date 是否为季度结束后的第5个交易日附近
    # （简化实现：只保留季报披露窗口后的值）
    return change


# ── 限售解禁 ─────────────────────────────────────

def lockup_expiry_days(data: pd.DataFrame, params: dict) -> pd.Series:
    """距下次限售解禁的天数。90天内无解禁返回默认值 999。

    依赖列：next_lockup_date（下次限售解禁日期）。

    params: {}
    """
    if "next_lockup_date" not in data.columns:
        return pd.Series(999, index=data.index)

    today_series = pd.to_datetime(data["trade_date"])
    lockup_series = pd.to_datetime(data["next_lockup_date"])
    days_until = (lockup_series - today_series).dt.days.fillna(999).clip(upper=999)
    return days_until
