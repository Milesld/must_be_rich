"""退出监控。

监控持仓的退出条件，分为长期仓和短期仓两套逻辑：
- 短期仓：盯"价格和纪律"，机械执行（止损/止盈/时间止损）。
- 长期仓：盯"逻辑"，不盯"价格"（逻辑破坏/估值极端/基本面踩雷）。

核心原则：退出规则在买入时定好，触发时检验预设逻辑是否仍成立。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 数据类 ────────────────────────────────────────

@dataclass
class ExitSignal:
    """退出信号（短期仓，机械执行）。"""
    code: str
    exit_type: str         # 'stop_loss' / 'take_profit' / 'time_stop'
    reason: str
    urgency: str = "immediate"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "exit_type": self.exit_type,
            "reason": self.reason,
            "urgency": self.urgency,
        }


@dataclass
class ExitAlert:
    """退出预警（长期仓，需人工确认）。"""
    code: str
    alert_type: str        # 'logic_breakdown' / 'valuation_extreme' / 'fundamental_crash'
    detail: str
    severity: str = "warn"
    requires_human_confirm: bool = True

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "alert_type": self.alert_type,
            "detail": self.detail,
            "severity": self.severity,
            "requires_human_confirm": self.requires_human_confirm,
        }


# ── 退出监控器 ─────────────────────────────────────

class ExitMonitor:
    """持仓退出条件监控。

    使用方式：
        monitor = ExitMonitor(config)
        # 短期仓
        sig = monitor.check_stop_loss("600519", 1680, 1650, atr=25.0)
        if sig:
            # 生成卖出 TradeIntent
            ...
        # 长期仓
        alert = monitor.check_logic_breakdown("600519", {"roe": 0.08, "prev_roe": 0.25})
        if alert:
            # 推送给人工确认
            ...
    """

    def __init__(self, config: dict | None = None) -> None:
        """初始化。

        Args:
            config: 配置字典，包含止损止盈参数。
        """
        cfg = config or {}
        self._stop_loss_atr_mult = cfg.get("stop_loss_atr_mult", 2.0)
        self._stop_loss_pct = cfg.get("stop_loss_pct", 0.08)        # 固定比例止损 8%
        self._take_profit_target = cfg.get("take_profit_target", 0.20)  # 目标止盈 20%
        self._trailing_stop_pct = cfg.get("trailing_stop_pct", 0.10)    # 从高点回撤10%止盈
        self._max_hold_days = cfg.get("max_hold_days", 60)              # 时间止损 60天
        self._roe_decline_threshold = cfg.get("roe_decline_threshold", 0.30)  # ROE下降>30%
        self._cf_ratio_threshold = cfg.get("cf_ratio_threshold", 0.5)        # CF/净利<0.5
        self._valuation_sigma = cfg.get("valuation_sigma", 2.0)              # 估值>历史2σ

        # 止损位追踪：{code: (entry_price, highest_price, stop_price)}
        self._trailing_stops: dict[str, tuple[float, float, float]] = {}

    # ── 短期仓：止损 ──────────────────────────────

    def check_stop_loss(
        self,
        code: str,
        entry_price: float,
        current_price: float,
        atr: float | None = None,
    ) -> ExitSignal | None:
        """ATR 倍数止损 / 固定比例止损。

        止损位只能往盈利方向移动，永不下移或取消。
        """
        reasons: list[str] = []

        # ATR 止损
        if atr is not None and atr > 0:
            stop_atr = entry_price - self._stop_loss_atr_mult * atr
            if current_price <= stop_atr:
                reasons.append(
                    f"ATR止损触发: 当前价{current_price:.2f} ≤ 止损{stop_atr:.2f}"
                    f"(ATR={atr:.2f}, multiplier={self._stop_loss_atr_mult})"
                )

        # 固定比例止损
        stop_pct = entry_price * (1 - self._stop_loss_pct)
        if current_price <= stop_pct:
            reasons.append(
                f"固定比例止损触发: 跌幅{(1 - current_price/entry_price)*100:.1f}%"
                f" > {self._stop_loss_pct*100:.0f}%"
            )

        if reasons:
            return ExitSignal(
                code=code,
                exit_type="stop_loss",
                reason="; ".join(reasons),
                urgency="immediate",
            )

        return None

    # ── 短期仓：止盈 ──────────────────────────────

    def check_take_profit(
        self,
        code: str,
        entry_price: float,
        current_price: float,
        highest_price: float | None = None,
    ) -> ExitSignal | None:
        """固定目标止盈 / Trailing Stop。

        - 固定目标：达到 target% 收益就卖。
        - Trailing Stop：从最高点回撤 X% 就卖（锁定利润）。
        """
        hp = highest_price or current_price
        reasons: list[str] = []

        # 固定目标止盈
        target_price = entry_price * (1 + self._take_profit_target)
        if current_price >= target_price:
            reasons.append(
                f"目标止盈触发: 涨幅{(current_price/entry_price - 1)*100:.1f}%"
                f" ≥ {self._take_profit_target*100:.0f}%"
            )

        # Trailing Stop
        trailing_stop = hp * (1 - self._trailing_stop_pct)
        if current_price <= trailing_stop:
            pct_from_high = (1 - current_price / hp) * 100
            reasons.append(
                f"移动止盈触发: 从最高点{hp:.2f}回撤{pct_from_high:.1f}%"
                f" > {self._trailing_stop_pct*100:.0f}%"
            )

        if reasons:
            return ExitSignal(
                code=code,
                exit_type="take_profit",
                reason="; ".join(reasons),
                urgency="immediate",
            )
        return None

    # ── 短期仓：时间止损 ──────────────────────────

    def check_time_stop(
        self,
        code: str,
        entry_date: date,
        current_date: date,
        current_price: float,
        entry_price: float,
        max_hold_days: int | None = None,
    ) -> ExitSignal | None:
        """时间止损：持有超过 N 天仍未盈利或未达预期→退出。

        避免资金被"僵尸"持仓长期占用。
        """
        days_held = (current_date - entry_date).days
        max_days = max_hold_days or self._max_hold_days

        if days_held >= max_days:
            pnl_pct = (current_price / entry_price - 1) * 100
            return ExitSignal(
                code=code,
                exit_type="time_stop",
                reason=(
                    f"持有{days_held}天 ≥ {max_days}天，当前盈亏{pnl_pct:+.1f}%"
                ),
                urgency="immediate",
            )
        return None

    # ── 长期仓：逻辑破坏检查 ────────────────────────

    def check_logic_breakdown(
        self,
        code: str,
        fundamental_factors: dict[str, float],
    ) -> ExitAlert | None:
        """逻辑检查清单（不自动卖出，生成 ExitAlert 供人工确认）。

        检查项：
        - ROE 连续两季下降 > 30%
        - 经营现金流/净利润 < 0.5 且持续恶化
        - 负债率超过行业均值 2σ
        - 出现财务造假/违规公告标志
        """
        alerts: list[str] = []

        # ROE 持续恶化
        roe = fundamental_factors.get("roe")
        prev_roe = fundamental_factors.get("prev_roe")
        prev2_roe = fundamental_factors.get("prev2_roe")

        if roe is not None and roe > 0 and prev_roe is not None:
            decline = (prev_roe - roe) / prev_roe
            if decline > self._roe_decline_threshold:
                # 检查是否连续两季
                if prev2_roe is not None and prev2_roe > prev_roe:
                    alerts.append(
                        f"ROE连续两季恶化: {prev2_roe:.2f}→{prev_roe:.2f}→{roe:.2f}"
                        f"(累计下降{decline*100:.0f}%)"
                    )
                else:
                    alerts.append(
                        f"ROE单季下降{decline*100:.0f}%: {prev_roe:.2f}→{roe:.2f}"
                    )

        # 经营现金流/净利润
        cf_ratio = fundamental_factors.get("cf_ratio")
        prev_cf_ratio = fundamental_factors.get("prev_cf_ratio")
        if cf_ratio is not None and cf_ratio < self._cf_ratio_threshold:
            if prev_cf_ratio is not None and cf_ratio < prev_cf_ratio:
                alerts.append(
                    f"经营现金流/净利润={cf_ratio:.2f} < {self._cf_ratio_threshold} 且持续恶化"
                )
            else:
                alerts.append(
                    f"经营现金流/净利润={cf_ratio:.2f} < {self._cf_ratio_threshold}"
                )

        # 负债率
        debt_ratio = fundamental_factors.get("debt_ratio")
        industry_avg_debt = fundamental_factors.get("industry_avg_debt")
        industry_std_debt = fundamental_factors.get("industry_std_debt")
        if (
            debt_ratio is not None
            and industry_avg_debt is not None
            and industry_std_debt is not None
            and industry_std_debt > 0
        ):
            z_score = (debt_ratio - industry_avg_debt) / industry_std_debt
            if z_score > 2.0:
                alerts.append(
                    f"负债率{debt_ratio:.1%}超过行业均值{industry_avg_debt:.1%} + "
                    f"{z_score:.1f}σ"
                )

        # 财务违规标志
        if fundamental_factors.get("fraud_flag", 0) == 1:
            alerts.append("检测到财务造假/违规公告标志——无条件退出预警")

        if alerts:
            is_critical = any("fraud_flag" in a or "违规" in a for a in alerts)
            return ExitAlert(
                code=code,
                alert_type="logic_breakdown",
                detail="; ".join(alerts),
                severity="critical" if is_critical else "warn",
                requires_human_confirm=True,
            )
        return None

    # ── 长期仓：估值极端检查 ───────────────────────

    def check_valuation_extreme(
        self,
        code: str,
        current_pe: float,
        pe_history: pd.Series | None = None,
        current_pb: float | None = None,
        pb_history: pd.Series | None = None,
    ) -> ExitAlert | None:
        """估值严重高估（超出历史 2σ 或过分位数）→ 预警。

        不自动卖出，提示人工复审。
        """
        alerts: list[str] = []

        if pe_history is not None and len(pe_history) > 30:
            pe_mean = pe_history.mean()
            pe_std = pe_history.std()
            if pe_std > 0:
                z_pe = (current_pe - pe_mean) / pe_std
                if z_pe > self._valuation_sigma:
                    alerts.append(
                        f"PE={current_pe:.1f}超出历史均值{pe_mean:.1f} + "
                        f"{z_pe:.1f}σ (历史2σ={pe_mean + 2*pe_std:.1f})"
                    )
                # 分位检查
                rank = (pe_history < current_pe).mean()
                if rank > 0.95:
                    alerts.append(f"PE处于历史{rank*100:.0f}%分位——极度高估")

        if pb_history is not None and len(pb_history) > 30 and current_pb is not None:
            pb_mean = pb_history.mean()
            pb_std = pb_history.std()
            if pb_std > 0:
                z_pb = (current_pb - pb_mean) / pb_std
                if z_pb > self._valuation_sigma:
                    alerts.append(
                        f"PB={current_pb:.1f}超出历史均值{pb_mean:.1f} + {z_pb:.1f}σ"
                    )

        if alerts:
            return ExitAlert(
                code=code,
                alert_type="valuation_extreme",
                detail="; ".join(alerts),
                severity="warn",
                requires_human_confirm=True,
            )
        return None

    # ── 止损位追踪 ────────────────────────────

    def register_position(
        self, code: str, entry_price: float,
    ) -> None:
        """注册新持仓的止损追踪。

        初始止损位 = 入场价 × (1 - 固定止损比例)。
        """
        stop_price = entry_price * (1 - self._stop_loss_pct)
        self._trailing_stops[code] = (entry_price, entry_price, stop_price)

    def update_trailing_stop(
        self, code: str, current_price: float,
    ) -> float | None:
        """更新移动止盈位。

        止损位只能上移（盈利保护），永不下移。

        Returns:
            当前止损价位。
        """
        if code not in self._trailing_stops:
            return None

        entry, prev_high, stop = self._trailing_stops[code]
        new_high = max(prev_high, current_price)

        # 新的 trailing stop = 最高价 × (1 - 回撤比例)
        new_stop = new_high * (1 - self._trailing_stop_pct)
        # 只能上移
        if new_stop > stop:
            stop = new_stop

        self._trailing_stops[code] = (entry, new_high, stop)
        return stop

    def remove_position(self, code: str) -> None:
        """持仓退出后清除追踪。"""
        self._trailing_stops.pop(code, None)

    def get_stop_price(self, code: str) -> float | None:
        """获取当前止损价位。"""
        info = self._trailing_stops.get(code)
        return info[2] if info else None
