"""Paper trading 闭环单元测试（路线图第 4 阶段）。

覆盖：订单生成（等权 top-N 调仓）、次日开盘撮合（含涨跌停拒单/停牌顺延/
撤单）、双账本（真实 vs 影子）、净值推进、执行偏差记录。
撮合与净值均为纯函数，行情用手工构造的 bars 注入，不依赖网络。
"""

from __future__ import annotations

import pytest

from research.paper_trading import (
    MAX_WAIT_TRADING_DAYS,
    build_rebalance_orders,
    new_ledger,
    settle_orders,
    update_nav,
)


def _ledger(cash: float = 100_000.0) -> dict:
    return new_ledger("testpool", "configs/fake.yaml", cash)


def _bars(code: str, rows: list[tuple[str, float, float]]) -> dict:
    """rows: [(date, open, close)]"""
    return {code: [{"date": d, "open": o, "close": c} for d, o, c in rows]}


class TestBuildOrders:
    def test_initial_buy_equal_weight(self):
        led = _ledger(100_000)
        orders = build_rebalance_orders(
            led, ["600000", "000001"], {"600000": 10.0, "000001": 20.0}, "2026-07-15")
        assert len(orders) == 2
        by_code = {o["code"]: o for o in orders}
        # 每只 5 万：600000@10 → 5000 股；000001@20 → 2500→2500//100*100=2500
        assert by_code["600000"]["side"] == "buy"
        assert by_code["600000"]["shares"] == 5000
        assert by_code["000001"]["shares"] == 2500

    def test_sell_dropped_position(self):
        led = _ledger(0)
        led["positions"] = {"600519": {"shares": 100, "avg_cost": 1500.0}}
        orders = build_rebalance_orders(
            led, ["600000"], {"600000": 10.0, "600519": 1500.0}, "2026-07-15")
        sides = {(o["code"], o["side"]) for o in orders}
        assert ("600519", "sell") in sides  # 移出目标池 → 清仓
        assert ("600000", "buy") in sides

    def test_no_order_when_already_at_target(self):
        led = _ledger(100.0)  # 现金可忽略
        led["positions"] = {"600000": {"shares": 5000, "avg_cost": 10.0}}
        orders = build_rebalance_orders(led, ["600000"], {"600000": 10.0}, "2026-07-15")
        assert orders == []

    def test_overweight_trimmed_only_beyond_15pct(self):
        led = _ledger(0)
        # nav = 60000，目标每只 60000；持仓市值 60000 → 不超配
        led["positions"] = {"600000": {"shares": 6000, "avg_cost": 10.0}}
        orders = build_rebalance_orders(led, ["600000"], {"600000": 10.0}, "2026-07-15")
        assert orders == []
        # 涨到 13 → nav=78000，持仓 78000 = 目标的 1.0 倍（单只时永不超配）
        # 用两只验证：持仓 A 占比过高时应卖出超额
        led2 = _ledger(0)
        led2["positions"] = {"600000": {"shares": 8000, "avg_cost": 10.0}}
        # nav=80000，两只 top → 每只 40000；A 市值 80000 > 40000×1.15
        orders2 = build_rebalance_orders(
            led2, ["600000", "000001"], {"600000": 10.0, "000001": 20.0}, "2026-07-15")
        by = {(o["code"], o["side"]): o for o in orders2}
        assert ("600000", "sell") in by
        assert by[("600000", "sell")]["shares"] == 4000  # (80000-40000)/10


class TestSettle:
    def test_fill_next_day_open(self):
        led = _ledger(100_000)
        led["pending_orders"] = build_rebalance_orders(
            led, ["600000"], {"600000": 10.0}, "2026-07-15")
        bars = _bars("600000", [("2026-07-15", 9.9, 10.0), ("2026-07-16", 10.2, 10.5)])
        processed = settle_orders(led, bars)
        assert len(processed) == 1
        o = processed[0]
        assert o["status"] == "filled"
        assert o["fill_date"] == "2026-07-16"
        assert o["fill_price"] == 10.2  # 次日开盘价，非信号日收盘
        assert o["exec_gap_pct"] == pytest.approx(2.0, abs=0.01)
        assert led["positions"]["600000"]["shares"] == o["filled_shares"]
        # 现金 = 初始 - 成交额 - 费用
        amount = 10.2 * o["filled_shares"]
        assert led["cash"] == pytest.approx(100_000 - amount - o["cost"], abs=0.01)
        assert led["pending_orders"] == []

    def test_limit_up_open_rejects_buy(self):
        led = _ledger(100_000)
        led["pending_orders"] = build_rebalance_orders(
            led, ["600000"], {"600000": 10.0}, "2026-07-15")
        # 主板 ±10%：昨收 10.0 → 涨停 11.0，开盘 11.0 = 一字涨停
        bars = _bars("600000", [("2026-07-15", 9.9, 10.0), ("2026-07-16", 11.0, 11.0)])
        processed = settle_orders(led, bars)
        assert processed[0]["status"] == "rejected"
        assert "涨停" in processed[0]["reject_reason"]
        assert led["positions"] == {}
        assert led["cash"] == 100_000
        # 影子账本按回测假设成交了 → 这正是要暴露的执行偏差
        assert led["shadow_positions"]["600000"]["shares"] > 0

    def test_limit_down_open_rejects_sell(self):
        led = _ledger(0)
        led["positions"] = {"600000": {"shares": 1000, "avg_cost": 10.0}}
        led["shadow_positions"] = {"600000": {"shares": 1000, "avg_cost": 10.0}}
        led["pending_orders"] = [{
            "id": "x", "signal_date": "2026-07-15", "code": "600000", "side": "sell",
            "shares": 1000, "signal_close": 10.0, "score": 0, "status": "pending",
            "wait_days": 0,
        }]
        bars = _bars("600000", [("2026-07-15", 10.0, 10.0), ("2026-07-16", 9.0, 9.0)])
        processed = settle_orders(led, bars)
        assert processed[0]["status"] == "rejected"
        assert "跌停" in processed[0]["reject_reason"]
        assert led["positions"]["600000"]["shares"] == 1000  # 没卖掉
        assert "600000" not in led["shadow_positions"]  # 影子卖出成功

    def test_gem_20pct_open_not_rejected_at_15pct(self):
        """创业板 ±20%：+15% 高开不算涨停，正常成交。"""
        led = _ledger(200_000)
        led["pending_orders"] = build_rebalance_orders(
            led, ["300750"], {"300750": 100.0}, "2026-07-15")
        bars = _bars("300750", [("2026-07-15", 99.0, 100.0), ("2026-07-16", 115.0, 118.0)])
        processed = settle_orders(led, bars)
        assert processed[0]["status"] == "filled"
        assert processed[0]["fill_price"] == 115.0

    def test_suspension_defers_then_cancels(self):
        led = _ledger(100_000)
        led["pending_orders"] = build_rebalance_orders(
            led, ["600000"], {"600000": 10.0}, "2026-07-15")
        # 600000 停牌（15 日之后无行情），市场日历用另一只股撑起来
        bars = _bars("600000", [("2026-07-15", 10.0, 10.0)])
        market_days = [(f"2026-07-{16+i:02d}", 5.0, 5.0) for i in range(3)]
        bars.update(_bars("000001", [("2026-07-15", 5.0, 5.0)] + market_days))
        processed = settle_orders(led, bars)
        assert processed == []  # 顺延中
        assert len(led["pending_orders"]) == 1
        assert led["pending_orders"][0]["wait_days"] == 3

        # 超过 MAX_WAIT_TRADING_DAYS 个市场交易日 → 撤单
        many_days = [(f"2026-08-{i+1:02d}", 5.0, 5.0)
                     for i in range(MAX_WAIT_TRADING_DAYS + 1)]
        bars2 = _bars("600000", [("2026-07-15", 10.0, 10.0)])
        bars2.update(_bars("000001", many_days))
        processed2 = settle_orders(led, bars2)
        assert processed2[0]["status"] == "cancelled"
        assert led["pending_orders"] == []

    def test_sell_capped_at_held_shares(self):
        led = _ledger(0)
        led["positions"] = {"600000": {"shares": 300, "avg_cost": 10.0}}
        led["pending_orders"] = [{
            "id": "x", "signal_date": "2026-07-15", "code": "600000", "side": "sell",
            "shares": 1000, "signal_close": 10.0, "score": 0, "status": "pending",
            "wait_days": 0,
        }]
        bars = _bars("600000", [("2026-07-15", 10.0, 10.0), ("2026-07-16", 10.0, 10.0)])
        processed = settle_orders(led, bars)
        assert processed[0]["filled_shares"] == 300
        assert "600000" not in led["positions"]

    def test_realized_pnl_recorded(self):
        led = _ledger(0)
        led["positions"] = {"600000": {"shares": 1000, "avg_cost": 10.0}}
        led["pending_orders"] = [{
            "id": "x", "signal_date": "2026-07-15", "code": "600000", "side": "sell",
            "shares": 1000, "signal_close": 12.0, "score": 0, "status": "pending",
            "wait_days": 0,
        }]
        bars = _bars("600000", [("2026-07-15", 12.0, 12.0), ("2026-07-16", 12.0, 12.0)])
        processed = settle_orders(led, bars)
        o = processed[0]
        # 毛利 2000，减费用后略低
        assert 1900 < o["realized_pnl"] < 2000


class TestNav:
    def test_nav_tracks_close_and_shadow_divergence(self):
        led = _ledger(100_000)
        led["pending_orders"] = build_rebalance_orders(
            led, ["600000"], {"600000": 10.0}, "2026-07-15")
        # 次日高开 5% 成交（真实买价 10.5，影子按 10.0 买）
        bars = _bars("600000", [
            ("2026-07-15", 10.0, 10.0),
            ("2026-07-16", 10.5, 10.8),
            ("2026-07-17", 10.8, 11.0),
        ])
        settle_orders(led, bars)
        added = update_nav(led, bars)
        assert added == 3
        latest = led["nav_history"][-1]
        # 同样按 11.0 收盘估值，影子买得便宜 → shadow_nav 更高
        assert latest["shadow_nav"] > latest["nav"]
        assert latest["n_positions"] == 1

    def test_nav_idempotent(self):
        led = _ledger(100_000)
        bars = _bars("600000", [("2026-07-15", 10.0, 10.0)])
        assert update_nav(led, bars) == 1
        assert update_nav(led, bars) == 0  # 重复调用不追加
        assert len(led["nav_history"]) == 1

    def test_nav_uses_last_close_for_suspended(self):
        led = _ledger(0)
        led["positions"] = {"600000": {"shares": 1000, "avg_cost": 10.0}}
        # 600000 只有 15 日行情，市场推进到 17 日（另一只股）
        bars = _bars("600000", [("2026-07-15", 10.0, 10.0)])
        bars.update(_bars("000001", [
            ("2026-07-15", 5.0, 5.0), ("2026-07-16", 5.0, 5.0), ("2026-07-17", 5.0, 5.0)]))
        update_nav(led, bars)
        # 三天净值都按 600000 最近收盘 10.0 估值
        assert [h["nav"] for h in led["nav_history"]] == [10_000.0] * 3
