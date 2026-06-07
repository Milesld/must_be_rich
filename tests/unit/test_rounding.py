"""整手化算法测试 — 覆盖各板块规则和边界条件。"""

from __future__ import annotations

import pytest

from core.backtest.rounding import ShareRounder


@pytest.fixture
def rounder() -> ShareRounder:
    return ShareRounder()


class TestRoundDown:
    """单只股票整手化。"""

    def test_main_board_100_lot(self, rounder: ShareRounder) -> None:
        """主板：必须是100整数倍"""
        assert rounder.round_down("600519", 150) == 100
        assert rounder.round_down("600519", 299) == 200
        assert rounder.round_down("600519", 99) == 0     # 不到一手→买不起
        assert rounder.round_down("600519", 1050) == 1000

    def test_szse_main_100_lot(self, rounder: ShareRounder) -> None:
        """深主板：也是100整数倍"""
        assert rounder.round_down("000001", 150) == 100
        assert rounder.round_down("000001", 50) == 0

    def test_gem_100_lot(self, rounder: ShareRounder) -> None:
        """创业板：100整数倍"""
        assert rounder.round_down("300750", 250) == 200
        assert rounder.round_down("300750", 80) == 0

    def test_star_market_200_min(self, rounder: ShareRounder) -> None:
        """科创板：≥200股，超过后1股递增"""
        assert rounder.round_down("688981", 300) == 300  # 直接保留
        assert rounder.round_down("688981", 150) == 0    # <200→买不起
        assert rounder.round_down("688981", 201) == 201  # 1股递增
        assert rounder.round_down("688981", 199) == 0    # <200

    def test_bse_100_lot(self, rounder: ShareRounder) -> None:
        """北交所"""
        assert rounder.round_down("920001", 150) == 100
        assert rounder.round_down("920001", 50) == 0


class TestRoundUp:
    def test_main_board_round_up(self, rounder: ShareRounder) -> None:
        assert rounder.round_up("600519", 150) == 200
        assert rounder.round_up("600519", 50) == 100    # 至少一手

    def test_star_market_round_up(self, rounder: ShareRounder) -> None:
        assert rounder.round_up("688981", 150) == 200   # 至少200
        assert rounder.round_up("688981", 250) == 250


class TestRoundPortfolio:
    """组合级整手化 — 迭代收敛核心算法。"""

    def test_normal_portfolio(self, rounder: ShareRounder) -> None:
        """正常组合：资金足够覆盖所有股票。"""
        weights = {"600519": 0.3, "000001": 0.4, "300750": 0.3}
        prices = {"600519": 1680.0, "000001": 12.0, "300750": 210.0}
        shares, actual_w, n = rounder.round_portfolio(
            weights, total_capital=1_000_000, current_prices=prices,
        )
        # 每只都应该能买到
        assert n >= 2, f"至少保留2只，实际{n}"
        for code, s in shares.items():
            assert s > 0, f"{code} 股数为0"
            price = prices[code]
            # 检查整手化合规
            assert s % 100 == 0 if not code.startswith("688") else s >= 200

    def test_small_capital_drops_expensive(self, rounder: ShareRounder) -> None:
        """小资金账户(5万) + 茅台(1680) → 茅台应被砍掉"""
        weights = {"600519": 0.3, "000001": 0.4, "300750": 0.3}
        prices = {"600519": 1680.0, "000001": 12.0, "300750": 210.0}
        shares, actual_w, n = rounder.round_portfolio(
            weights, total_capital=50_000, current_prices=prices,
        )
        # 茅台 50000×0.3=15000元 → ~9股 → 不足一手(100股=168000元) → 应被砍掉
        assert "600519" not in shares or shares.get("600519", 0) == 0, (
            f"小资金不应买得起茅台，但shares={shares}"
        )

    def test_all_dropped_empty(self, rounder: ShareRounder) -> None:
        """资金太小，全部砍掉——不崩溃。"""
        weights = {"600519": 1.0}
        prices = {"600519": 1680.0}
        shares, actual_w, n = rounder.round_portfolio(
            weights, total_capital=500, current_prices=prices,
        )
        assert n == 0
        assert shares == {}

    def test_star_market_in_portfolio(self, rounder: ShareRounder) -> None:
        """科创板股票在组合中，按200股起规则。"""
        weights = {"688981": 1.0}
        prices = {"688981": 55.0}
        shares, _, n = rounder.round_portfolio(
            weights, total_capital=20_000, current_prices=prices,
        )
        if n > 0:
            assert shares["688981"] >= 200

    def test_can_afford(self, rounder: ShareRounder) -> None:
        can, shares = rounder.can_afford("600519", cash=200_000, price=1680.0)
        assert can
        assert shares >= 100  # 至少能买一手

        can2, shares2 = rounder.can_afford("600519", cash=50_000, price=1680.0)
        # 50000/1680 ≈ 29股 < 100 → 买不起
        assert not can2
