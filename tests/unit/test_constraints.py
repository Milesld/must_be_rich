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

    def test_limit_up_sealed_with_volume_rejects_buy(self, tc: TradingConstraints) -> None:
        """回归：涨停日成交量>0（日线常态），收盘贴板仍应视为封板拒买。

        旧逻辑要求 volume<=0 才认定封板，而日线数据上涨停日也有集合竞价
        成交量，导致封板检查形同虚设、回测可在一字板买入。
        """
        data = {
            "000888": {
                "pre_close": 10.0,
                "close": 11.0,        # = 涨停价
                "volume": 3_000_000,  # 有成交量（现实情况）
                "is_st": False,
                "is_suspended": False,
            },
        }
        tc.update_daily_info(data)

        passed, reason = tc.check_limit_sealed("000888", "buy")
        assert not passed, "有成交量但收盘贴涨停价，仍应认定封板拒买"
        # 卖出方向不受涨停封板限制
        passed_sell, _ = tc.check_limit_sealed("000888", "sell")
        assert passed_sell

    def test_limit_down_sealed_rejects_sell(self, tc: TradingConstraints) -> None:
        """跌停封板（收盘≈跌停价）→ 卖单被拒。"""
        data = {
            "000888": {
                "pre_close": 10.0,
                "close": 9.0,         # = 跌停价 10×0.9
                "volume": 2_000_000,
                "is_st": False,
                "is_suspended": False,
            },
        }
        tc.update_daily_info(data)

        passed, reason = tc.check_limit_sealed("000888", "sell")
        assert not passed, "跌停封板应拒绝卖单"
        assert "跌停封板" in reason

    def test_normal_no_sealed(self, tc: TradingConstraints, daily_data: dict) -> None:
        tc.update_daily_info(daily_data)

        passed, _ = tc.check_limit_sealed("600519", "buy")
        assert passed


class TestBoardDefaultPriceLimits:
    """回归：无板块配置时，创业板/科创板/北交所应按各自涨跌幅检查。

    历史 bug：TradingConstraints(None) 时所有板块统一按 ±10%，
    创业板/科创板合法的 10~20% 波动被误拒单，扭曲全部回测结果。
    """

    def test_gem_15pct_move_allowed(self, tc: TradingConstraints) -> None:
        """创业板（300）±20%：+15% 的价格应通过。"""
        data = {"300750": {"pre_close": 100.0, "close": 115.0, "volume": 1_000_000,
                            "is_st": False, "is_suspended": False}}
        tc.update_daily_info(data)
        passed, reason = tc.check_price_limit("300750", 115.0)
        assert passed, f"创业板 +15% 合法波动被误拒: {reason}"

    def test_gem_21pct_rejected(self, tc: TradingConstraints) -> None:
        """创业板 +21% 超过 ±20% 上限应拒绝。"""
        data = {"300750": {"pre_close": 100.0, "close": 119.0, "volume": 1_000_000,
                            "is_st": False, "is_suspended": False}}
        tc.update_daily_info(data)
        passed, _ = tc.check_price_limit("300750", 121.0)
        assert not passed

    def test_star_19pct_move_allowed(self, tc: TradingConstraints) -> None:
        """科创板（688）±20%：+19% 的价格应通过。"""
        data = {"688981": {"pre_close": 50.0, "close": 59.5, "volume": 1_000_000,
                            "is_st": False, "is_suspended": False}}
        tc.update_daily_info(data)
        passed, reason = tc.check_price_limit("688981", 59.5)
        assert passed, f"科创板 +19% 合法波动被误拒: {reason}"

    def test_bse_25pct_move_allowed(self, tc: TradingConstraints) -> None:
        """北交所（920）±30%：+25% 的价格应通过。"""
        data = {"920139": {"pre_close": 10.0, "close": 12.5, "volume": 500_000,
                            "is_st": False, "is_suspended": False}}
        tc.update_daily_info(data)
        passed, reason = tc.check_price_limit("920139", 12.5)
        assert passed, f"北交所 +25% 合法波动被误拒: {reason}"

    def test_main_board_still_10pct(self, tc: TradingConstraints) -> None:
        """主板仍是 ±10%：+12% 应拒绝。"""
        data = {"600519": {"pre_close": 1000.0, "close": 1090.0, "volume": 1_000_000,
                            "is_st": False, "is_suspended": False}}
        tc.update_daily_info(data)
        passed, _ = tc.check_price_limit("600519", 1120.0)
        assert not passed


class TestEngineLoadsBoardsConfig:
    """回归：BacktestEngine 不传 config 时也应加载 configs/boards 板块规则。"""

    def test_engine_auto_loads_boards(self) -> None:
        from datetime import date as _date
        from core.backtest.engine import BacktestEngine

        engine = BacktestEngine(
            start_date=_date(2024, 1, 2),
            end_date=_date(2024, 1, 10),
            initial_capital=1_000_000,
        )
        boards = engine.constraints._boards
        assert boards, "引擎应自动加载 configs/boards/*.yaml"
        assert boards.get("szse_gem", {}).get("price_limit") == 0.20
        assert boards.get("sse_star", {}).get("price_limit") == 0.20
        assert boards.get("bse", {}).get("price_limit") == 0.30


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
