"""交易成本边界回归测试。"""

import pytest
from decimal import Decimal
from core.backtest.cost_model import TransactionCostModel


@pytest.fixture
def cm() -> TransactionCostModel:
    return TransactionCostModel(csi800_codes={"600519"})


class TestCommissionBoundaries:
    def test_1000_yuan_buy_min_commission_5r(self, cm: TransactionCostModel) -> None:
        """1000元买单 → 佣金=5元（万1.5=0.15 < 5→最低5元）"""
        cost = cm.calculate("000002", "buy", 10.0, 100)
        assert cost.commission == Decimal("5.00"), f"1000×0.00015=0.15<5→应5元, 实际{cost.commission}"

    def test_50000_yuan_buy_5r_commission(self, cm: TransactionCostModel) -> None:
        """50000元买单 → 佣金=7.5元（万1.5×5万=7.5>5→7.5元）"""
        cost = cm.calculate("000002", "buy", 50.0, 1000)
        assert cost.commission == Decimal("7.50"), f"50000×0.00015=7.5, 实际{cost.commission}"

    def test_1m_yuan_buy_commission(self, cm: TransactionCostModel) -> None:
        """100万元买单 → 佣金=150元"""
        cost = cm.calculate("000002", "buy", 100.0, 10000)
        assert cost.commission == Decimal("150.00")


class TestStampDutyBoundaries:
    def test_buy_never_has_stamp(self, cm: TransactionCostModel) -> None:
        for code in ["600519", "000001", "300750", "688981", "000002"]:
            cost = cm.calculate(code, "buy", 100.0, 1000)
            assert cost.stamp_duty == Decimal("0"), f"{code} 买方不应收印花税"

    def test_sell_always_has_stamp(self, cm: TransactionCostModel) -> None:
        for code in ["600519", "000001", "300750"]:
            cost = cm.calculate(code, "sell", 100.0, 1000)
            assert cost.stamp_duty > Decimal("0"), f"{code} 卖方应收印花税"

    def test_etf_sell_no_stamp(self, cm: TransactionCostModel) -> None:
        cost = cm.calculate("510050", "sell", 3.0, 10000)
        assert cost.stamp_duty == Decimal("0"), "ETF卖方免征印花税"

    def test_etf_buy_no_stamp(self, cm: TransactionCostModel) -> None:
        cost = cm.calculate("510050", "buy", 3.0, 10000)
        assert cost.stamp_duty == Decimal("0")


class TestSlippageBoundaries:
    def test_gem_slippage_higher_than_main(self, cm: TransactionCostModel) -> None:
        main_cost = cm.calculate("000002", "buy", 100.0, 1000)
        gem_cost = cm.calculate("300750", "buy", 100.0, 1000)
        assert gem_cost.slippage_est > main_cost.slippage_est, "创业板滑点应高于主板"
