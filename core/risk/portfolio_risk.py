"""组合级风控。

实时监控组合层面风险：回撤熔断、VaR/CVaR、波动率目标。
触发熔断时按优先级执行减仓（交易仓优先→核心仓）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ANNUAL_FACTOR = 252


@dataclass
class RiskAction:
    """风控动作。"""
    action_type: str
    target_positions: dict[str, int] = field(default_factory=dict)
    reason: str = ""


class PortfolioRiskManager:
    """组合级实时风控。

    使用方式：
        mgr = PortfolioRiskManager(config)
        action = mgr.check_drawdown(current_nav, peak_nav)
        if action:
            # 执行减仓或熔断
            ...
        target_ratio = mgr.vol_targeting(current_vol, target_vol, pos_ratio)
    """

    def __init__(self, config: dict | None = None) -> None:
        """初始化。

        Args:
            config: 配置（来自 risk/thresholds.yaml 的 portfolio 节）。
        """
        cfg = config or {}
        self._dd_daily_circuit = cfg.get("drawdown_daily_circuit", 0.15)
        self._dd_rolling_circuit = cfg.get("drawdown_rolling_circuit", 0.25)
        self._rolling_window = cfg.get("rolling_window_days", 5)
        self._var_limit_95 = cfg.get("var_limit_95", 0.05)
        self._cvar_limit_95 = cfg.get("cvar_limit_95", 0.08)
        self._vol_target = cfg.get("vol_target", 0.15)
        self._max_leverage = cfg.get("max_leverage", 1.0)

        # 历史净值（用于滚动回撤）
        self._nav_history: list[float] = []
        # 熔断状态
        self._circuit_breaker_triggered = False

    # ── 回撤熔断 ──────────────────────────────

    def check_drawdown(
        self,
        current_nav: float,
        peak_nav: float,
        circuit_breaker_mode: str = "auto",
    ) -> RiskAction | None:
        """回撤熔断检查。

        - 当日回撤 > 15% → 强制减短期仓至0
        - 5日滚动回撤 > 25% → 强制减全仓至30%

        Args:
            current_nav: 当前净值。
            peak_nav: 历史最高净值。
            circuit_breaker_mode: 'auto' 自动执行 / 'alert' 仅告警。

        Returns:
            RiskAction（None=未触发）。
        """
        if peak_nav <= 0:
            return None

        daily_dd = (peak_nav - current_nav) / peak_nav
        self._nav_history.append(current_nav)

        # 当日回撤
        if daily_dd > self._dd_daily_circuit:
            self._circuit_breaker_triggered = True
            reason = (
                f"当日回撤{daily_dd:.1%} > {self._dd_daily_circuit:.0%}——"
                f"强制减交易仓至0"
            )
            logger.warning(reason)
            if circuit_breaker_mode == "auto":
                return RiskAction(
                    action_type="reduce_trading",
                    reason=reason,
                )

        # 滚动回撤
        rolling_dd = self._calc_rolling_drawdown()
        if rolling_dd is not None and rolling_dd > self._dd_rolling_circuit:
            self._circuit_breaker_triggered = True
            reason = (
                f"{self._rolling_window}日滚动回撤{rolling_dd:.1%} > "
                f"{self._dd_rolling_circuit:.0%}——强制减全仓至30%"
            )
            logger.warning(reason)
            if circuit_breaker_mode == "auto":
                return RiskAction(
                    action_type="reduce_all",
                    reason=reason,
                )

        return None

    def _calc_rolling_drawdown(self) -> float | None:
        """计算滚动窗口最大回撤。"""
        if len(self._nav_history) < max(self._rolling_window, 3):
            return None
        window = self._nav_history[-self._rolling_window:]
        peak = max(window)
        if peak <= 0:
            return 0.0
        return (peak - window[-1]) / peak

    # ── VaR / CVaR ────────────────────────────

    def calc_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """历史模拟法 VaR。

        Args:
            returns: 日收益率序列。
            confidence: 置信水平（0.95 = 95% VaR）。

        Returns:
            正数 VaR 值（如 0.03 = 3% 日 VaR）。
        """
        clean = returns.dropna()
        if len(clean) < 60:
            return float("inf")

        var = -np.percentile(clean, 100 * (1 - confidence))
        return float(var)

    def calc_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """CVaR（Expected Shortfall）：超过 VaR 的平均损失。

        Args:
            returns: 日收益率序列。
            confidence: 置信水平。

        Returns:
            正数 CVaR 值。
        """
        clean = returns.dropna()
        if len(clean) < 60:
            return float("inf")

        var_threshold = -np.percentile(clean, 100 * (1 - confidence))
        tail_losses = clean[clean <= -var_threshold]
        if len(tail_losses) == 0:
            return var_threshold
        return float(-tail_losses.mean())

    def check_var_limits(
        self, returns: pd.Series,
    ) -> tuple[bool, float, float]:
        """检查 VaR/CVaR 是否超限。

        Returns:
            (超限?, VaR值, CVaR值)
        """
        var = self.calc_var(returns)
        cvar = self.calc_cvar(returns)
        over_limit = (var > self._var_limit_95) or (cvar > self._cvar_limit_95)
        return over_limit, var, cvar

    # ── 波动率目标 ────────────────────────────

    def vol_targeting(
        self,
        current_vol: float,
        target_vol: float | None = None,
        current_position_ratio: float = 1.0,
    ) -> float:
        """波动率目标（Vol Targeting）。

        adjustment = target_vol / current_vol

        实现方式：
        - 当前波动率 > 目标 → 降仓至 target/current 比例
        - 当前波动率 < 目标 → 可适当加仓（不超过 1.0）

        Args:
            current_vol: 当前组合年化波动率。
            target_vol: 目标波动率（None 则用配置值）。
            current_position_ratio: 当前仓位比例。

        Returns:
            建议的目标仓位比例（0.0 ~ 1.0）。
        """
        target = target_vol or self._vol_target
        if current_vol <= 0:
            return current_position_ratio

        adjustment = target / current_vol
        # 降仓时快速执行，加仓时设上限
        suggested = current_position_ratio * min(adjustment, 1.0 / current_position_ratio)
        return max(0.0, min(1.0, suggested))

    def calc_portfolio_volatility(
        self,
        returns: pd.Series,
    ) -> float:
        """计算年化波动率。"""
        clean = returns.dropna()
        if len(clean) < 5:
            return 0.0
        return float(clean.std() * np.sqrt(_ANNUAL_FACTOR))

    # ── 状态查询 ────────────────────────────

    def reset(self) -> None:
        """重置熔断状态和历史。"""
        self._nav_history.clear()
        self._circuit_breaker_triggered = False

    @property
    def is_circuit_breaker_triggered(self) -> bool:
        return self._circuit_breaker_triggered

    def feed_nav(self, nav: float) -> None:
        """记录净值（在每日回测/实盘收盘后调用）。"""
        self._nav_history.append(nav)
        # 仅保留最近60日用于滚动回撤计算
        if len(self._nav_history) > 60:
            self._nav_history = self._nav_history[-60:]
