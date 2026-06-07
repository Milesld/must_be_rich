"""交易成本模型测试 — 覆盖边界条件。"""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.backtest.cost_model import (
    CostBreakdown,
    TransactionCostModel,
    from_decimal,
    to_decimal,
)


@pytest.fixture
def model() -> TransactionCostModel:
    return TransactionCostModel(csi800_codes={"600519", "000001"})


class TestCostBreakdown:
    def test_creation(self) -> None:
        cb = CostBreakdown(
            commission=Decimal("5.00"),
            stamp_duty=Decimal("2.50"),
            transfer_fee=Decimal("0.05"),
            total=Decimal("7.55"),
        )
        assert cb.commission == Decimal("5.00")
        assert cb.total == Decimal("7.55")

    def test_to_dict(self) -> None:
        cb = CostBreakdown(commission=Decimal("5.00"), total=Decimal("5.00"))
        d = cb.to_dict()
        assert d["commission"] == 5.0
        assert d["stamp_duty"] == 0.0


class TestCommission:
    """佣金计算：万1.5，最低5元。"""

    def test_min_commission_rule_3000_yuan(self, model: TransactionCostModel) -> None:
        """3000元 → 万1.5=4.5元 → 实收5元"""
        cost = model.calculate("000002", "buy", 10.0, 300)
        assert cost.commission == Decimal("5.00"), f"预期5.00，实际{cost.commission}"

    def test_min_commission_rule_5000_yuan(self, model: TransactionCostModel) -> None:
        """5000元 → 万1.5=0.75元 → <5元 → 实收5元（最低5元规则）"""
        cost = model.calculate("000002", "buy", 50.0, 100)
        assert cost.commission == Decimal("5.00"), f"预期5.00(最低)，实际{cost.commission}"

    def test_commission_50000_yuan(self, model: TransactionCostModel) -> None:
        """50000元 → 万1.5=7.5元 → 超过5元，实收7.5元"""
        cost = model.calculate("000002", "buy", 50.0, 1000)
        assert cost.commission == Decimal("7.50")

    def test_commission_1m_yuan(self, model: TransactionCostModel) -> None:
        """100万元 → 万1.5=150元"""
        cost = model.calculate("000002", "buy", 100.0, 10000)
        assert cost.commission == Decimal("150.00")


class TestStampDuty:
    """印花税：仅卖方 0.05%，ETF免征。"""

    def test_buy_no_stamp(self, model: TransactionCostModel) -> None:
        cost = model.calculate("600519", "buy", 1680.0, 100)
        assert cost.stamp_duty == Decimal("0")

    def test_sell_with_stamp(self, model: TransactionCostModel) -> None:
        cost = model.calculate("600519", "sell", 1680.0, 100)
        amount = Decimal("168000")
        expected = amount * Decimal("0.0005")  # 84
        assert cost.stamp_duty == expected.quantize(Decimal("0.01"))

    def test_etf_sell_no_stamp(self, model: TransactionCostModel) -> None:
        """ETF (51xxxx) 卖出也不交印花税"""
        cost = model.calculate("510050", "sell", 3.0, 10000)
        assert cost.stamp_duty == Decimal("0")

    def test_etf_buy_no_stamp(self, model: TransactionCostModel) -> None:
        cost = model.calculate("510050", "buy", 3.0, 10000)
        assert cost.stamp_duty == Decimal("0")


class TestTransferFee:
    """过户费：0.01‰，买卖双向。"""

    def test_buy_with_transfer(self, model: TransactionCostModel) -> None:
        cost = model.calculate("600519", "buy", 1680.0, 100)
        assert cost.transfer_fee > Decimal("0"), "买入应收过户费"

    def test_sell_with_transfer(self, model: TransactionCostModel) -> None:
        cost = model.calculate("600519", "sell", 1680.0, 100)
        assert cost.transfer_fee > Decimal("0"), "卖出应收过户费"


class TestSlippage:
    """滑点分层。"""

    def test_csi800_slippage(self, model: TransactionCostModel) -> None:
        cost = model.calculate("600519", "buy", 1680.0, 100)
        amount = Decimal("168000")
        expected = amount * Decimal("0.001")  # 0.1%
        assert cost.slippage_est == expected.quantize(Decimal("0.01"))

    def test_small_cap_slippage(self, model: TransactionCostModel) -> None:
        """小盘股（<50亿市值）滑点 0.5%"""
        cost = model.calculate("002999", "buy", 10.0, 1000, market_cap=Decimal("3_000_000_000"))
        amount = Decimal("10000")
        expected = amount * Decimal("0.005")
        assert cost.slippage_est == expected.quantize(Decimal("0.01"))

    def test_gem_slippage_default(self, model: TransactionCostModel) -> None:
        """创业板股票无市值信息时按0.3%"""
        cost = model.calculate("300750", "buy", 210.0, 500)
        amount = Decimal("105000")
        expected = amount * Decimal("0.003")
        assert cost.slippage_est == expected.quantize(Decimal("0.01"))


class TestTotalCost:
    """总成本 = 佣金 + 印花税(仅卖) + 过户费 + 滑点。"""

    def test_buy_total(self, model: TransactionCostModel) -> None:
        cost = model.calculate("600519", "buy", 1680.0, 100)
        expected_total = (
            cost.commission + cost.stamp_duty + cost.transfer_fee + cost.slippage_est
        )
        assert cost.total == expected_total

    def test_sell_total(self, model: TransactionCostModel) -> None:
        cost = model.calculate("600519", "sell", 1680.0, 100)
        expected_total = (
            cost.commission + cost.stamp_duty + cost.transfer_fee + cost.slippage_est
        )
        assert cost.total == expected_total


class TestDecimalUtils:
    def test_to_decimal_float(self) -> None:
        assert to_decimal(3.14) == Decimal("3.14")

    def test_to_decimal_int(self) -> None:
        assert to_decimal(100) == Decimal("100")

    def test_to_decimal_pass_through(self) -> None:
        d = Decimal("5.00")
        assert to_decimal(d) is d

    def test_from_decimal(self) -> None:
        assert from_decimal(Decimal("5.00")) == 5.0
