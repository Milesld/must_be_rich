"""交易成本精确建模。

严格按照方案2.4和2.5节定义，精确计算A股每笔交易的所有成本：
- 印花税：仅卖方 0.05%，ETF免征
- 佣金：万1.5，买卖双向，最低5元/笔
- 过户费：0.01‰，买卖双向
- 滑点估算：按流动性和板块分层

金额运算使用 Decimal 保证精度，避免浮点误差累积。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from typing import Literal

import pandas as pd

# ── 费率常量（Decimal 精度） ──────────────────────

STAMP_DUTY_RATE = Decimal("0.0005")        # 印花税 0.05%（仅卖方）
COMMISSION_RATE = Decimal("0.00015")        # 佣金 万1.5
MIN_COMMISSION = Decimal("5.00")            # 最低5元/笔
TRANSFER_FEE_RATE = Decimal("0.00001")      # 过户费 十万分之一（按成交额）
TRANSFER_FEE_ALT = Decimal("0.00002")       # 过户费 十万分之二（中登公司标准 2022年起按成交额0.001‰，即十万分之一  √ 确认）

# 滑点分层
SLIPPAGE_CSI800 = Decimal("0.001")          # 中证800成分股 0.1%
SLIPPAGE_MAIN = Decimal("0.002")            # 主板其他 0.2%
SLIPPAGE_GEM_STAR = Decimal("0.003")        # 创业板/科创板 0.3%
SLIPPAGE_SMALL_CAP = Decimal("0.005")       # 小盘股（<50亿市值） 0.5%

# 板块识别
_ETF_PREFIX = "51"


@dataclass(frozen=True)
class CostBreakdown:
    """单笔交易的成本拆分。"""
    commission: Decimal = Decimal("0")
    stamp_duty: Decimal = Decimal("0")
    transfer_fee: Decimal = Decimal("0")
    slippage_est: Decimal = Decimal("0")
    total: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        return {
            "commission": float(self.commission),
            "stamp_duty": float(self.stamp_duty),
            "transfer_fee": float(self.transfer_fee),
            "slippage_est": float(self.slippage_est),
            "total": float(self.total),
        }


class TransactionCostModel:
    """A股交易成本精确建模。

    所有金额计算使用 Decimal 保证精度。

    使用方式：
        model = TransactionCostModel()
        cost = model.calculate("600519", "buy", Decimal("1680"), 100)
        # cost.commission = 25.20, cost.total = 26.88
    """

    def __init__(self, csi800_codes: set[str] | None = None) -> None:
        """初始化。

        Args:
            csi800_codes: 中证800成分股代码集合，用于滑点分层。
                          为 None 时所有股票按主板滑点率处理。
        """
        self._csi800 = csi800_codes or set()

    # ── 主计算方法 ──────────────────────────────

    def calculate(
        self,
        code: str,
        side: Literal["buy", "sell"],
        price: Decimal | float,
        shares: int,
        is_etf: bool | None = None,
        market_cap: Decimal | float | None = None,
    ) -> CostBreakdown:
        """计算单笔交易的全部成本。

        Args:
            code: 股票代码 ('600519')。
            side: 买('buy')/卖('sell')。
            price: 成交价。
            shares: 成交股数。
            is_etf: 是否ETF（None 时根据 code 前缀自动判断）。
            market_cap: 市值（可选，用于小盘股滑点识别）。

        Returns:
            CostBreakdown 包含各费用项的拆分。
        """
        p = _to_d(price)
        amount = p * shares  # 成交金额

        is_etf_val = self._is_etf(code) if is_etf is None else is_etf
        commission = self._calc_commission(amount)
        stamp_duty = self._calc_stamp_duty(amount, side, is_etf_val)
        transfer_fee = self._calc_transfer_fee(amount)
        slippage = self._calc_slippage(code, amount, market_cap)

        total = commission + stamp_duty + transfer_fee + slippage

        return CostBreakdown(
            commission=commission,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            slippage_est=slippage,
            total=total,
        )

    # ── 各项费用计算 ────────────────────────────────

    def _calc_commission(self, amount: Decimal) -> Decimal:
        """佣金：万1.5 × 成交金额，最低5元。

        Examples:
            3000元 × 万1.5 = 4.5 → 实收5元
            50000元 × 万1.5 = 7.5 → 实收7.5元
        """
        raw = amount * COMMISSION_RATE
        return max(raw, MIN_COMMISSION)

    def _calc_stamp_duty(
        self, amount: Decimal, side: str, is_etf: bool,
    ) -> Decimal:
        """印花税：仅卖方 0.05%，ETF免征。"""
        if side != "sell" or is_etf:
            return Decimal("0")
        return (amount * STAMP_DUTY_RATE).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )

    def _calc_transfer_fee(self, amount: Decimal) -> Decimal:
        """过户费：0.01‰（万分之0.1），买卖双向。"""
        return (amount * TRANSFER_FEE_RATE).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )

    def _calc_slippage(
        self, code: str, amount: Decimal, market_cap: Decimal | float | None,
    ) -> Decimal:
        """滑点估算：按流动性分层。

        ★ 口径说明：滑点计入 CostBreakdown.total，回测引擎按 total 扣现金，
        即回测收益已包含滑点成本（保守口径）。它与佣金/印花税的区别仅在于
        是估算值而非交易所实收，报告中单列展示以便核对。
        """
        if market_cap is not None:
            cap = _to_d(market_cap)
            if cap < Decimal("5_000_000_000"):   # < 50亿 → 小盘
                rate = SLIPPAGE_SMALL_CAP
            elif self._is_csi800(code):
                rate = SLIPPAGE_CSI800
            elif self._is_gem_or_star(code):
                rate = SLIPPAGE_GEM_STAR
            else:
                rate = SLIPPAGE_MAIN
        else:
            # 无市值信息 → 仅按板块分层
            if self._is_csi800(code):
                rate = SLIPPAGE_CSI800
            elif self._is_gem_or_star(code):
                rate = SLIPPAGE_GEM_STAR
            else:
                rate = SLIPPAGE_MAIN

        return (amount * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # ── 板块判断 ──────────────────────────────

    def _is_etf(self, code: str) -> bool:
        """根据代码前缀判断是否为ETF。51xxxx开头的为ETF。"""
        return code.startswith(_ETF_PREFIX)

    def _is_gem_or_star(self, code: str) -> bool:
        """判断是否为创业板/科创板。"""
        return code.startswith("300") or code.startswith("301") or code.startswith("688")

    def _is_csi800(self, code: str) -> bool:
        """判断是否中证800成分股。"""
        return code in self._csi800

    # ── 成本计算（便捷函数，float 接口，兼容旧代码） ──

    def calculate_f(
        self, code: str, side: str, price: float, shares: int,
    ) -> dict[str, float]:
        """float 接口版本，返回字典。"""
        return self.calculate(code, side, Decimal(str(price)), shares).to_dict()


# ── Decimal 工具函数 ──────────────────────────────

def _to_d(value: Decimal | float | int) -> Decimal:
    """安全转换为 Decimal。"""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def to_decimal(value: float | int | str) -> Decimal:
    """公开的 Decimal 转换函数。"""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def from_decimal(value: Decimal) -> float:
    """Decimal → float（仅用于最终报告展示，已损失精度）。"""
    return float(value)
