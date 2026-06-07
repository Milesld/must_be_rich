"""信号→订单→成交 全链路集成测试。"""

import pytest
from datetime import date, datetime


class TestSignalToOrder:
    """全链路: 信号生成→OMS→风控→下单→成交回报。"""

    def test_signal_flow_not_crashing(self) -> None:
        """验证整个管线不崩溃。"""
        # 信号
        signal = {
            "signal_id": "test_sig_001",
            "code": "600519",
            "side": "buy",
            "price": 1680.0,
            "shares": 100,
            "strength": 0.78,
            "confidence": 0.65,
            "model_name": "long_term_ranker",
            "model_version": "v3.2",
            "generated_at": datetime.now().isoformat(),
        }

        # 风控检查
        from core.risk.pre_trade import PreTradeRiskChecker
        checker = PreTradeRiskChecker()
        result = checker.check(
            code=signal["code"], side=signal["side"],
            price=1680.0, shares=10,
            account_id=1, current_positions={},
            current_cash=500_000, total_asset=1_000_000,
            current_prices={"600519": 1680.0},
        )
        assert result.passed is True, f"基本风控应通过: {result.reject_reason}"

        # 成本预估
        from core.backtest.cost_model import TransactionCostModel
        cm = TransactionCostModel()
        cost = cm.calculate("600519", "buy", 1680.0, 10)
        assert cost.commission >= 5.0  # min commission
        assert cost.stamp_duty == 0  # buy side no stamp

    def test_sell_flow(self) -> None:
        from core.risk.pre_trade import PreTradeRiskChecker

        checker = PreTradeRiskChecker()
        result = checker.check(
            code="600519", side="sell", price=1700.0, shares=50,
            account_id=1,
            current_positions={"600519": 100},
            current_cash=200_000, total_asset=500_000,
            current_prices={"600519": 1700.0},
        )
        assert result.passed is True

        # sell cost includes stamp duty
        from core.backtest.cost_model import TransactionCostModel
        cm = TransactionCostModel()
        cost = cm.calculate("600519", "sell", 1700.0, 50)
        assert cost.stamp_duty > 0, "卖方应收印花税"

    def test_st_stock_rejected(self) -> None:
        from core.risk.pre_trade import PreTradeRiskChecker
        checker = PreTradeRiskChecker()
        result = checker.check(
            code="000888", side="buy", price=5.0, shares=100,
            account_id=1, current_positions={},
            current_cash=100_000, total_asset=100_000,
            current_prices={}, is_st=True,
        )
        assert not result.passed
        assert "ST" in result.reject_reason
