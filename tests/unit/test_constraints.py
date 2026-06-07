"""交易约束检查测试 — 覆盖T+1/涨跌停/停牌/封板/价格笼子。"""

from __future__ import annotations

from datetime import date

import pytest

from core.backtest.constraints import ConstraintCheckResult, TradingConstraints


@pytest.fixture
def tc() -> TradingConstraints:
    """使用默认板块规则初始化。"""
    return TradingConstraints()


@pytest.fixture
def daily_data() -> dict[str, dict]:
    return {
        "600519": {
            "pre_close": 1680.0,
            "close": 1695.0,
            "open": 1685.0,
            "high": 1700.0,
            "low": 1670.0,
            "volume": 5_000_000,
            "is_st": False,
            "is_suspended": False,
        },
        "688981": {
            "pre_close": 55.0,
            "close": 55.5,
            "open": 55.0,
            "high": 56.0,
            "low": 54.5,
            "volume": 10_000_000,
            "is_st": False,
            "is_suspended": False,
        },
        "300750": {
            "pre_close": 210.0,
            "close": 211.0,
            "open": 209.0,
            "high": 213.0,
            "low": 208.0,
            "volume": 20_000_000,
            "is_st": False,
            "is_suspended": False,
        },
    }


class TestPriceLimit:
    def test_price_within_limit(self, tc: TradingConstraints, daily_data: dict) -> None:
        tc.update_daily_info(daily_data)

        # 600519: pre_close=1680, limit±10%=1512~1848
        passed, reason = tc.check_price_limit("600519", 1700.0)
        assert passed, f"应通过: {reason}"

    def test_price_above_limit_up(self, tc: TradingConstraints, daily_data: dict) -> None:
        tc.update_daily_info(daily_data)

        # 涨停价 = 1680 * 1.10 = 1848
        passed, reason = tc.check_price_limit("600519", 1900.0)
        assert not passed, "超过涨停价应拒绝"

    def test_price_below_limit_down(self, tc: TradingConstraints, daily_data: dict) -> None:
        tc.update_daily_info(daily_data)

        # 跌停价 = 1680 * 0.90 = 1512
        passed, reason = tc.check_price_limit("600519", 1400.0)
        assert not passed, "低于跌停价应拒绝"


class TestTPlus1:
    def test_buy_always_allowed(self, tc: TradingConstraints) -> None:
        tc.mark_bought("600519")
        passed, reason = tc.check_t_plus_1("600519", "buy")
        assert passed, "买入不限T+1"

    def test_sell_today_bought_rejected(self, tc: TradingConstraints) -> None:
        tc.mark_bought("600519")
        passed, reason = tc.check_t_plus_1("600519", "sell")
        assert not passed, "当日买入不能卖出"
        assert "T+1" in reason

    def test_sell_not_bought_allowed(self, tc: TradingConstraints) -> None:
        passed, reason = tc.check_t_plus_1("000001", "sell")
        assert passed, "非当日买入可卖出"


class TestSuspension:
    def test_suspended_rejected(self, tc: TradingConstraints) -> None:
        data = {"000888": {"is_suspended": True, "pre_close": 10.0}}
        tc.update_daily_info(data)

        passed, reason = tc.check_suspension("000888")
        assert not passed
        assert "停牌" in reason

    def test_normal_allowed(self, tc: TradingConstraints) -> None:
        data = {"600519": {"is_suspended": False, "pre_close": 1680.0}}
        tc.update_daily_info(data)

        passed, _ = tc.check_suspension("600519")
        assert passed


class TestLimitSealed:
    def test_limit_up_sealed_rejects_buy(self, tc: TradingConstraints) -> None:
        """涨停封板（vol=0且close≈涨停价）→ 买单被拒"""
        data = {
            "000888": {
                "pre_close": 10.0,
                "close": 11.0,   # = 涨停价 10×1.1
                "volume": 0,
                "is_st": False,
                "is_suspended": False,
            },
        }
        tc.update_daily_info(data)

        passed, reason = tc.check_limit_sealed("000888", "buy")
        assert not passed, "涨停封板应拒绝买单"
        assert "涨停封板" in reason

    def test_normal_no_sealed(self, tc: TradingConstraints, daily_data: dict) -> None:
        tc.update_daily_info(daily_data)

        passed, _ = tc.check_limit_sealed("600519", "buy")
        assert passed


class TestCheckAll:
    def test_all_pass(self, tc: TradingConstraints, daily_data: dict) -> None:
        tc.update_daily_info(daily_data)
        result = tc.check_all("600519", "buy", 1690.0)
        assert result.passed is True
        assert result.reject_reason is None
        assert len(result.checks_detail) >= 3

    def test_one_fail_rejects_all(self, tc: TradingConstraints, daily_data: dict) -> None:
        """任何一项不通过即整体拒绝。"""
        tc.update_daily_info(daily_data)
        # 跌停价以下的卖出
        result = tc.check_all("600519", "sell", 1000.0)  # <1512
        assert result.passed is False
        assert result.reject_reason is not None


class TestResetDay:
    def test_reset_clears_today_bought(self, tc: TradingConstraints) -> None:
        tc.mark_bought("600519")
        tc.mark_bought("000001")
        assert len(tc._today_bought) == 2

        tc.reset_day()
        assert len(tc._today_bought) == 0

    def test_reset_clears_limits(self, tc: TradingConstraints, daily_data: dict) -> None:
        tc.update_daily_info(daily_data)
        assert len(tc._price_limits) >= 3

        tc.reset_day()
        assert len(tc._price_limits) == 0


class TestConstraintCheckResult:
    def test_passed_result(self) -> None:
        r = ConstraintCheckResult(passed=True, checks_detail={"a": True, "b": True})
        assert r.passed

    def test_failed_result(self) -> None:
        r = ConstraintCheckResult(
            passed=False,
            reject_reason="涨停封板",
            checks_detail={"price_limit": True, "limit_sealed": False},
        )
        assert not r.passed
        assert "涨停" in r.reject_reason
