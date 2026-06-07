"""凯利公式 — 股票连续收益版。

f* = μ / σ²

实际使用半凯利（f = 0.5 × f*）或 1/4 凯利（f = 0.25 × f*）。
全凯利理论最优但对输入参数极度敏感——估计误差会导致灾难性超配。
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ANNUAL_FACTOR = 252


class KellyCalculator:
    """凯利公式计算器（连续收益版）。

    使用方式：
        kc = KellyCalculator()
        f_star = kc.calculate_f_star(returns)
        safe_bet = kc.quarter_kelly(returns)  # 推荐新手使用
    """

    def __init__(self, annual_factor: int = _ANNUAL_FACTOR) -> None:
        self._annual_factor = annual_factor

    # ── 核心计算 ──────────────────────────────

    def calculate_f_star(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
    ) -> float:
        """计算最优仓位比例 f*。

        公式：f* = (μ - r_f) / σ²

        Args:
            returns: 日收益率序列。
            risk_free_rate: 年化无风险利率（默认2%，约为中国10年期国债）。

        Returns:
            f*（最优仓位比例）。0 = 不投, 0.5 = 半仓, 1.0 = 满仓。
            负值表示看空（不应做多）。
        """
        clean = returns.dropna()
        if len(clean) < 60:
            logger.warning("收益率序列过短（%d < 60），凯利估计不靠谱，返回0", len(clean))
            return 0.0

        # 年化超额收益
        mu_daily = clean.mean()
        mu_annual = mu_daily * self._annual_factor
        excess_return = mu_annual - risk_free_rate

        # 年化方差
        var_daily = clean.var()
        var_annual = var_daily * self._annual_factor

        if var_annual <= 0:
            return 0.0

        f_star = excess_return / var_annual
        return float(f_star)

    # ── 保守版本 ────────────────────────────

    def half_kelly(
        self, returns: pd.Series, risk_free_rate: float = 0.02,
    ) -> float:
        """半凯利：f = 0.5 × f*。

        Returns:
            半凯利仓位比例。非负（不做空）。
        """
        f = self.calculate_f_star(returns, risk_free_rate)
        return max(0.0, f * 0.5)

    def quarter_kelly(
        self, returns: pd.Series, risk_free_rate: float = 0.02,
    ) -> float:
        """1/4 凯利：f = 0.25 × f*。

        最为保守，推荐新手使用。
        鲁棒性最强——即使收益率估计有偏，也不至于仓位过高。

        Returns:
            1/4凯利仓位比例。非负。
        """
        f = self.calculate_f_star(returns, risk_free_rate)
        return max(0.0, f * 0.25)

    # ── 滚动凯利 ────────────────────────────

    def rolling_kelly(
        self,
        returns: pd.Series,
        window: int = 252,
        method: str = "half",
    ) -> pd.Series:
        """计算滚动窗口的凯利仓位比例。

        Args:
            returns: 日收益率序列。
            window: 滚动窗口（交易日）。
            method: 'full' | 'half' | 'quarter'。

        Returns:
            滚动凯利序列（与 returns 对齐）。
        """
        clean = returns.dropna()
        if len(clean) < window:
            return pd.Series(0.0, index=returns.index)

        roll_mu = clean.rolling(window).mean() * self._annual_factor
        roll_var = clean.rolling(window).var() * self._annual_factor
        rf_daily = 0.02 / self._annual_factor
        roll_excess = roll_mu - 0.02
        roll_f_star = roll_excess / roll_var.replace(0, float("nan"))

        if method == "half":
            result = roll_f_star * 0.5
        elif method == "quarter":
            result = roll_f_star * 0.25
        else:
            result = roll_f_star

        result = result.clip(lower=0.0, upper=2.0)  # 上限200%（杠杆）
        return result.reindex(returns.index).fillna(0.0)

    # ── 单笔下注（离散版，二元赌局）─────────────

    @staticmethod
    def discrete_kelly(
        win_prob: float,
        win_loss_ratio: float,  # b = 盈利/亏损（盈利时的净收益倍数）
    ) -> float:
        """离散凯利（二元赌局版）：f* = (p·b - q) / b。

        用于"一次下注多少"的场景（不适用于连续股票收益）。

        Args:
            win_prob: 胜率 p (0~1)。
            win_loss_ratio: 盈亏比 b（如 b=2 表示赢了赚2元，输了亏1元）。

        Returns:
            f*。
        """
        q = 1.0 - win_prob
        if win_loss_ratio <= 0:
            return 0.0
        return max(0.0, (win_prob * win_loss_ratio - q) / win_loss_ratio)
