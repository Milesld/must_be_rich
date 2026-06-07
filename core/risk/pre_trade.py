"""事前风控检查器。

下单前同步调用（延迟预算 < 10ms），检查所有预定义风控规则。
任何一项不通过即拒绝下单，记录到 risk_log 供合规审计。

Python 版先跑通逻辑，后续第8批用 C++ 重写热路径。
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """风控检查结果。"""
    passed: bool
    reject_reason: Optional[str] = None
    checks_detail: dict[str, bool] = field(default_factory=dict)
    warn_only: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reject_reason": self.reject_reason,
            "checks_detail": self.checks_detail,
            "warn_only": self.warn_only,
        }


@dataclass
class RiskAction:
    """风控动作。"""
    action_type: str          # 'reduce_trading' / 'reduce_all' / 'hedge_with_futures' / 'halt'
    target_positions: dict[str, int] = field(default_factory=dict)  # 目标持仓调整
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "target_positions": self.target_positions,
            "reason": self.reason,
        }


class PreTradeRiskChecker:
    """事前风控检查器。

    在每笔订单发出前执行，所有检查项全部通过才放行。

    使用方式：
        checker = PreTradeRiskChecker(config)
        result = checker.check("600519", "buy", 1680.0, 100, account_id,
                               positions, cash, industry_map)
        if not result.passed:
            log.error("风控拒绝: %s", result.reject_reason)
    """

    def __init__(self, config: dict | None = None) -> None:
        """初始化。

        Args:
            config: 风控阈值配置（来自 risk/thresholds.yaml 的 pre_trade 节）。
        """
        cfg = config or {}
        self._max_single_ratio = cfg.get("single_stock_max_ratio", 0.20)
        self._max_single_hard = cfg.get("single_stock_hard_max", 0.25)
        self._max_industry = cfg.get("industry_max_ratio", 0.40)
        self._daily_max_orders = cfg.get("daily_max_orders", 20000)
        self._second_max_orders = cfg.get("second_max_orders", 300)
        self._cancel_rate_max = cfg.get("cancel_rate_max", 0.70)
        self._single_order_max_amount = cfg.get("single_order_max_amount", 500_000)
        self._blacklist_reasons: set[str] = set(cfg.get("blacklist_reasons", []))

        # 滑动窗口计数器
        self._order_count_day = 0
        self._order_timestamps: list[float] = []       # 秒级订单时间戳
        self._cancel_timestamps: list[float] = []       # 秒级撤单时间戳
        self._blacklist_codes: set[str] = set()

    # ── 主检查方法 ──────────────────────────────

    def check(
        self,
        code: str,
        side: str,
        price: float,
        shares: int,
        account_id: int,
        current_positions: dict[str, int],
        current_cash: float,
        total_asset: float,
        current_prices: dict[str, float],
        industry_map: dict[str, str] | None = None,
        is_st: bool = False,
        is_suspended: bool = False,
        blacklist_flags: list[str] | None = None,
    ) -> RiskCheckResult:
        """执行全部事前风控检查。

        Args:
            code: 股票代码。
            side: 'buy' / 'sell'。
            price: 申报价格。
            shares: 申报股数。
            account_id: 账户ID。
            current_positions: {code: shares} 当前持仓。
            current_cash: 当前可用现金。
            total_asset: 总资产（现金+持仓市值）。
            current_prices: {code: latest_price} 最新市价。
            industry_map: {code: industry} 行业分类。
            is_st: 是否ST股。
            is_suspended: 是否停牌。
            blacklist_flags: 黑名单标记列表。

        Returns:
            RiskCheckResult。
        """
        details: dict[str, bool] = {}
        warns: list[str] = []
        reject: Optional[str] = None

        amount = price * shares
        # 预估成本（买：佣金+过户费；卖：佣金+印花税+过户费）
        est_cost = self._estimate_cost(side, amount)

        checks = [
            ("blacklist", self._check_blacklist(code, is_st, blacklist_flags)),
            ("suspension", self._check_suspension(is_suspended)),
            ("single_stock_limit", self._check_single_stock(code, shares, side,
                                                             current_positions, current_prices, total_asset)),
            ("industry_limit", self._check_industry(code, industry_map, total_asset,
                                                    current_positions, current_prices,
                                                    shares, side, price)),
            ("daily_order_limit", self._check_daily_order_count()),
            ("second_order_limit", self._check_second_order_count()),
            ("cancel_rate", self._check_cancel_rate()),
            ("single_order_amount", self._check_single_order_amount(amount)),
            ("cash_sufficient", self._check_cash(side, amount, est_cost, current_cash)),
        ]

        for name, (passed, reason) in checks:
            details[name] = passed
            if not passed and reject is None:
                reject = reason
            if not passed and name in ("industry_limit", "single_stock_limit"):
                # 这些也可以降级为 warn（取决于配置）
                warns.append(reason)

        return RiskCheckResult(
            passed=reject is None,
            reject_reason=reject,
            checks_detail=details,
            warn_only=warns,
        )

    # ── 各项检查 ──────────────────────────────

    def _check_blacklist(
        self, code: str, is_st: bool, flags: list[str] | None,
    ) -> tuple[bool, str]:
        if code in self._blacklist_codes:
            return False, f"{code} 在黑名单中"
        if is_st:
            return False, f"{code} 为ST/*ST股票，禁止交易"
        if flags:
            for f in flags:
                if f in self._blacklist_reasons:
                    return False, f"{code}: {f}"
        return True, "OK"

    def _check_suspension(self, is_suspended: bool) -> tuple[bool, str]:
        if is_suspended:
            return False, "股票处于停牌状态"
        return True, "OK"

    def _check_single_stock(
        self, code: str, shares: int, side: str,
        positions: dict[str, int], prices: dict[str, float],
        total_asset: float,
    ) -> tuple[bool, str]:
        """单票上限检查（含硬上限）。"""
        current_shares = positions.get(code, 0)
        price = prices.get(code, 0.0)
        if total_asset <= 0:
            return True, "OK"

        if side == "buy":
            new_shares = current_shares + shares
        else:
            new_shares = max(0, current_shares - shares)

        new_ratio = (new_shares * price) / total_asset if total_asset > 0 else 0.0

        if new_ratio > self._max_single_hard:
            return False, (
                f"单票仓位{new_ratio:.1%} > 硬上限{self._max_single_hard:.0%} "
                f"({code}, 当前{current_shares}股, 拟{side}{shares}股)"
            )
        if new_ratio > self._max_single_ratio:
            return False, (
                f"单票仓位{new_ratio:.1%} > 上限{self._max_single_ratio:.0%}"
            )
        return True, "OK"

    def _check_industry(
        self, code: str, industry_map: dict[str, str] | None,
        total_asset: float, positions: dict[str, int],
        prices: dict[str, float], shares: int, side: str, price: float,
    ) -> tuple[bool, str]:
        """行业集中度检查。"""
        if industry_map is None or total_asset <= 0:
            return True, "OK"

        ind = industry_map.get(code, "other")
        # 计算该行业当前市值
        ind_value = 0.0
        for c, s in positions.items():
            if industry_map.get(c) == ind:
                ind_value += s * prices.get(c, 0.0)

        trade_value = shares * price
        if side == "buy":
            new_ind_value = ind_value + trade_value
        else:
            new_ind_value = max(0.0, ind_value - trade_value)

        new_ratio = new_ind_value / total_asset
        if new_ratio > self._max_industry:
            return False, (
                f"行业'{ind}'仓位{new_ratio:.1%} > 上限{self._max_industry:.0%}"
            )
        return True, "OK"

    def _check_daily_order_count(self) -> tuple[bool, str]:
        if self._order_count_day >= self._daily_max_orders:
            return False, f"日申报笔数已达上限{self._daily_max_orders}"
        return True, "OK"

    def _check_second_order_count(self) -> tuple[bool, str]:
        now = time.time()
        # 清理 1 秒前的时间戳
        cutoff = now - 1.0
        self._order_timestamps = [t for t in self._order_timestamps if t > cutoff]
        if len(self._order_timestamps) >= self._second_max_orders:
            return False, f"秒申报笔数已达上限{self._second_max_orders}"
        return True, "OK"

    def _check_cancel_rate(self) -> tuple[bool, str]:
        now = time.time()
        cutoff = now - 1.0
        self._order_timestamps = [t for t in self._order_timestamps if t > cutoff]
        self._cancel_timestamps = [t for t in self._cancel_timestamps if t > cutoff]
        total = len(self._order_timestamps) + len(self._cancel_timestamps)
        if total >= 10 and len(self._cancel_timestamps) / total > self._cancel_rate_max:
            return False, (
                f"撤单率{len(self._cancel_timestamps)}/{total}={len(self._cancel_timestamps)/total:.0%}"
                f" > {self._cancel_rate_max:.0%}"
            )
        return True, "OK"

    def _check_single_order_amount(self, amount: float) -> tuple[bool, str]:
        if amount > self._single_order_max_amount:
            return False, (
                f"单笔金额{amount:,.0f} > 上限{self._single_order_max_amount:,}"
            )
        return True, "OK"

    def _check_cash(
        self, side: str, amount: float, est_cost: float, cash: float,
    ) -> tuple[bool, str]:
        if side != "buy":
            return True, "OK"
        needed = amount + est_cost
        if cash < needed:
            return False, (
                f"可用资金不足: 需要{needed:,.2f}(含费{est_cost:,.2f}), "
                f"可用{cash:,.2f}"
            )
        return True, "OK"

    @staticmethod
    def _estimate_cost(side: str, amount: float) -> float:
        """快速估算成本（不含精确滑点）。"""
        comm = max(amount * 0.00015, 5.0)
        stamp = amount * 0.0005 if side == "sell" else 0.0
        transfer = amount * 0.00001
        return comm + stamp + transfer

    # ── 状态管理 ─────────────────────────────

    def record_order(self) -> None:
        """记录一笔订单（用于申报笔数和撤单率统计）。"""
        self._order_count_day += 1
        self._order_timestamps.append(time.time())

    def record_cancel(self) -> None:
        """记录一笔撤单。"""
        self._cancel_timestamps.append(time.time())

    def add_blacklist(self, code: str, reason: str = "") -> None:
        """动态加入黑名单。"""
        self._blacklist_codes.add(code)
        logger.info("黑名单新增: %s (%s)", code, reason)

    def remove_blacklist(self, code: str) -> None:
        """从黑名单移除。"""
        self._blacklist_codes.discard(code)

    def reset_daily(self) -> None:
        """每日重置（开盘前调用）。"""
        self._order_count_day = 0
        self._order_timestamps.clear()
        self._cancel_timestamps.clear()
        logger.debug("事前风控日计数器已重置")
