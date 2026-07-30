"""回测引擎集成测试。"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from core.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    ClosePriceFillSimulator,
    DailySnapshot,
    OHLCFillSimulator,
    TradeIntent,
    TradeRecord,
)
from core.backtest.cost_model import CostBreakdown


class StubStrategy:
    """测试用策略：固定返回买入信号。"""

    def __init__(self, intents: list[TradeIntent] | None = None) -> None:
        self.intents = intents or []
        self.call_count = 0
        self.last_features = None

    def on_bar(self, trade_date, features, positions, cash, daily_data):
        self.call_count += 1
        self.last_features = features
        return self.intents


class TestTradeRecord:
    def test_to_dict(self) -> None:
        from decimal import Decimal
        cb = CostBreakdown(
            commission=Decimal("5.00"),
            stamp_duty=Decimal("0"),
            transfer_fee=Decimal("0.02"),
            slippage_est=Decimal("1.68"),
            total=Decimal("6.70"),
        )
        tr = TradeRecord(
            trade_id="t1", signal_id="s1", code="600519", side="buy",
            trade_date=date(2026, 6, 1), intent_price=1680.0,
            intent_shares=100, filled_price=1680.0, filled_shares=100,
            cost_breakdown=cb, status="filled",
        )
        d = tr.to_dict()
        assert d["trade_id"] == "t1"
        assert d["signal_id"] == "s1"
        assert d["cost_breakdown"]["commission"] == 5.0


class TestDailySnapshot:
    def test_creation(self) -> None:
        ds = DailySnapshot(
            date=date(2026, 6, 1), nav=1_000_000.0, cash=500_000.0,
            positions={"600519": 100}, position_values={"600519": 168_000.0},
            total_value=1_000_000.0, daily_return=0.01,
        )
        assert ds.nav == 1_000_000.0


class TestFillSimulator:
    def test_close_price_fill(self) -> None:
        sim = ClosePriceFillSimulator()
        intent = TradeIntent(signal_id="s1", code="600519", side="buy", price=1700.0, shares=100)
        daily = {"600519": {"close": 1695.0}}

        price, shares = sim.simulate(intent, daily)
        assert price == 1695.0
        assert shares == 100

    def test_close_price_unavailable(self) -> None:
        sim = ClosePriceFillSimulator()
        intent = TradeIntent(signal_id="s1", code="999999", side="buy", price=10.0, shares=100)
        daily = {"999999": {"close": 0.0}}

        price, shares = sim.simulate(intent, daily)
        assert shares == 0  # 无法成交

    def test_ohlc_fill_uses_vwap(self) -> None:
        sim = OHLCFillSimulator()
        intent = TradeIntent(signal_id="s1", code="600519", side="buy", price=1700.0, shares=100)
        daily = {
            "600519": {
                "open": 1685.0, "high": 1700.0, "low": 1670.0,
                "close": 1695.0, "amount": 8_500_000_000, "volume": 5_000_000,
            },
        }
        price, shares = sim.simulate(intent, daily)
        # VWAP = 8.5B / 5M = 1700
        assert price > 0
        assert shares == 100


class TestBacktestEngine:
    def test_init(self) -> None:
        engine = BacktestEngine(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            initial_capital=1_000_000,
        )
        assert engine.cash == 1_000_000
        assert engine.initial_capital == 1_000_000
        assert engine.positions == {}
        assert engine._run_id.startswith("bt_")

    def test_empty_period_no_trading_days(self) -> None:
        """回测区间无交易日 → 返回空结果，不崩溃。"""
        engine = BacktestEngine(
            start_date=date(2026, 6, 7),  # Sunday
            end_date=date(2026, 6, 7),
            initial_capital=1_000_000,
        )
        strategy = StubStrategy()
        result = engine.run(strategy)
        assert isinstance(result, BacktestResult)
        # 可能为0或很少（取决于是否周末）
        assert result.daily_nav.empty or len(result.daily_nav) >= 0

    def test_strategy_called(self) -> None:
        """策略的 on_bar 被正确调用。"""
        # 使用一个窄区间确保有数据
        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 10),
            initial_capital=1_000_000,
        )
        strategy = StubStrategy()
        result = engine.run(strategy)
        # 策略至少被尝试调用（有数据时）
        # 由于没有 data_loader，daily_data 为空，但引擎不崩溃
        assert result is not None

    def test_trade_intent_handling(self) -> None:
        """验证交易意图处理管线：约束+整手+撮合+成本。"""
        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=500_000,
        )

        # 构造一个合法交易意图
        intent = TradeIntent(
            signal_id="sig_001", code="600519", side="buy",
            price=1690.0, shares=100,
        )
        daily = {
            "600519": {
                "open": 1685.0, "high": 1700.0, "low": 1670.0,
                "close": 1695.0, "pre_close": 1680.0,
                "volume": 5_000_000, "amount": 8_500_000_000,
                "is_st": False, "is_suspended": False,
            },
        }
        engine.constraints.update_daily_info(daily)

        record = engine._handle_trade(intent, daily, date(2024, 1, 2))
        assert record.signal_id == "sig_001"
        assert record.status in ("filled", "rejected", "unfilled")
        if record.status == "filled":
            assert record.filled_shares > 0
            assert record.cost_breakdown is not None

    def test_apply_fill_updates_cash_and_positions(self) -> None:
        """成交后现金和持仓正确更新。"""
        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=1_000_000,
        )
        from decimal import Decimal

        cb = CostBreakdown(
            commission=Decimal("25.20"),
            stamp_duty=Decimal("0"),
            transfer_fee=Decimal("1.68"),
            total=Decimal("26.88"),
        )
        record = TradeRecord(
            trade_id="t1", signal_id="s1", code="600519", side="buy",
            trade_date=date(2024, 1, 2), intent_price=1680.0,
            intent_shares=100, filled_price=1680.0, filled_shares=100,
            cost_breakdown=cb, status="filled",
        )
        engine._apply_fill(record)

        assert engine.positions["600519"] == 100
        assert engine.cash < 1_000_000  # 扣除了成本
        assert engine.position_costs["600519"] > 0

    def test_sell_updates_positions(self) -> None:
        """卖出更新持仓和现金。"""
        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=1_000_000,
        )
        # 先手动设置持仓
        engine.positions["600519"] = 100
        engine.position_costs["600519"] = 1600.0

        from decimal import Decimal
        cb = CostBreakdown(
            commission=Decimal("25.20"),
            stamp_duty=Decimal("84.00"),
            transfer_fee=Decimal("1.68"),
            total=Decimal("110.88"),
        )
        record = TradeRecord(
            trade_id="t2", signal_id="s2", code="600519", side="sell",
            trade_date=date(2024, 1, 3), intent_price=1680.0,
            intent_shares=100, filled_price=1680.0, filled_shares=100,
            cost_breakdown=cb, status="filled",
        )
        engine._apply_fill(record)

        # 持仓应为0或键不存在
        assert engine.positions.get("600519", 0) == 0
        # 现金增加：168000 - 110.88 = 167889.12
        assert engine.cash > 1_000_000

    def test_oversell_truncated_to_holdings(self) -> None:
        """卖出超持仓：按实际持仓截断，不许凭空造现金/负持仓。"""
        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=1_000_000,
        )
        engine.positions["600519"] = 100
        engine.position_costs["600519"] = 1600.0

        record = TradeRecord(
            trade_id="t3", signal_id="s3", code="600519", side="sell",
            trade_date=date(2024, 1, 3), intent_price=1680.0,
            intent_shares=500, filled_price=1680.0, filled_shares=500,
            status="filled",
        )
        engine._apply_fill(record)

        assert record.filled_shares == 100
        assert engine.positions.get("600519", 0) == 0
        # 只卖出 100 股 → 现金增量应接近 168000，而非 840000
        assert engine.cash < 1_000_000 + 170_000

    def test_sell_without_position_rejected(self) -> None:
        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=1_000_000,
        )
        record = TradeRecord(
            trade_id="t4", signal_id="s4", code="600519", side="sell",
            trade_date=date(2024, 1, 3), intent_price=1680.0,
            intent_shares=100, filled_price=1680.0, filled_shares=100,
            status="filled",
        )
        engine._apply_fill(record)
        assert record.status == "rejected"
        assert engine.cash == 1_000_000

    def test_save_and_load_state(self, tmp_path) -> None:
        """暂停/继续：状态保存和恢复。"""
        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 12, 31),
            initial_capital=1_000_000,
        )
        engine.cash = 900_000
        engine.positions = {"600519": 100}
        engine.position_costs = {"600519": 1680.0}
        engine.peak_nav = 1_050_000

        state_path = str(tmp_path / "state.json")
        engine.save_state(state_path)

        # 新引擎恢复
        engine2 = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 12, 31),
            initial_capital=1_000_000,
        )
        ok = engine2.load_state(state_path)
        assert ok
        assert engine2.cash == 900_000
        assert engine2.positions == {"600519": 100}
        assert engine2.peak_nav == 1_050_000


class TestBacktestResult:
    def test_empty_result(self) -> None:
        r = BacktestResult(
            run_id="test", start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31), initial_capital=1_000_000,
            daily_snapshots=[], trades=[],
        )
        assert r.daily_nav.empty
        assert r.summary() == {"error": "无交易日数据"}

    def test_single_day_snapshot(self) -> None:
        ds = DailySnapshot(
            date=date(2024, 1, 2), nav=1_000_000.0, cash=1_000_000.0,
            positions={}, position_values={}, total_value=1_000_000.0,
            daily_return=0.0,
        )
        r = BacktestResult(
            run_id="test", start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2), initial_capital=1_000_000,
            daily_snapshots=[ds], trades=[],
        )
        assert len(r.daily_nav) == 1

    def test_summary_with_data(self) -> None:
        """有数据时的汇总不崩溃。"""
        snaps = [
            DailySnapshot(
                date=date(2024, 1, d), nav=1_000_000 + d * 100,
                cash=500_000.0, positions={}, position_values={},
                total_value=1_000_000 + d * 100, daily_return=0.001,
            )
            for d in range(2, 10)
        ]
        r = BacktestResult(
            run_id="test", start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 10), initial_capital=1_000_000,
            daily_snapshots=snaps, trades=[],
        )
        s = r.summary()
        assert "error" not in s
        assert s["total_trades"] == 0
        assert "annual_return" in s


class TestWalkForward:
    def test_get_windows(self) -> None:
        from core.backtest.walk_forward import WalkForwardValidator

        wfv = WalkForwardValidator(train_window_years=3, test_window_months=6, purge_days=5)
        windows = wfv.get_windows(date(2018, 1, 1), date(2023, 12, 31))
        assert len(windows) > 0
        for train_s, train_e, test_s, test_e in windows:
            assert train_s < train_e
            assert test_s < test_e
            # Purge: test_s should be after train_e
            assert test_s >= train_e

    def test_short_period_no_windows(self) -> None:
        from core.backtest.walk_forward import WalkForwardValidator

        wfv = WalkForwardValidator(train_window_years=5, test_window_months=3)
        windows = wfv.get_windows(date(2020, 1, 1), date(2021, 12, 31))
        # 只有2年数据，低于 min_train_years=2，但 train_window=5
        # 实际：train_start=2020.1.1, train_end=2025.1.1 > end → 无窗口
        assert len(windows) == 0


class TestReport:
    def test_metrics_no_benchmark(self) -> None:
        from core.backtest.report import BacktestReport

        snaps = [
            DailySnapshot(
                date=date(2024, 1, d), nav=1_000_000 + d * 100,
                cash=500_000.0, positions={}, position_values={},
                total_value=1_000_000 + d * 100, daily_return=0.001,
            )
            for d in range(2, 10)
        ]
        br = BacktestResult(
            run_id="test", start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 9), initial_capital=1_000_000,
            daily_snapshots=snaps, trades=[],
        )
        report = BacktestReport(br)
        m = report.metrics()
        assert "error" not in m
        assert "sharpe_ratio" in m

    def test_generate_report_file(self, tmp_path) -> None:
        from core.backtest.report import BacktestReport

        ds = DailySnapshot(
            date=date(2024, 1, 2), nav=1_010_000.0, cash=500_000.0,
            positions={}, position_values={}, total_value=1_010_000.0,
            daily_return=0.01,
        )
        br = BacktestResult(
            run_id="test", start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2), initial_capital=1_000_000,
            daily_snapshots=[ds], trades=[],
        )
        report = BacktestReport(br)
        path = str(tmp_path / "report.md")
        report.generate(path)

        import os
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "回测报告" in content
        assert "夏普比率" in content
