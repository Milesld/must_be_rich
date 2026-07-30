"""组合管理与风控引擎单元测试。"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════
# Market Regime
# ═══════════════════════════════════════════════

class TestMarketRegime:
    def test_detect_no_data_returns_neutral(self) -> None:
        from core.portfolio.regime import MarketRegimeDetector

        d = MarketRegimeDetector()
        r = d.detect(date(2026, 6, 7))
        assert r.regime_label == "震荡市"
        assert r.composite_score == 0.5

    def test_detect_with_bullish_data(self) -> None:
        from core.portfolio.regime import MarketRegimeDetector

        dates = pd.date_range("2020-01-01", "2026-06-07", freq="B")
        # 模拟牛市：持续上涨 + 低波动 + 低PE
        close = pd.Series(100 * (1.0003 ** np.arange(len(dates))), index=dates)
        pe = pd.Series(12 + 3 * np.sin(np.arange(len(dates)) / 500), index=dates)
        df = pd.DataFrame({
            "close": close,
            "pe_ttm": pe,
        }, index=dates)

        d = MarketRegimeDetector()
        r = d.detect(date(2026, 6, 7), df)
        assert r.composite_score > 0.5
        assert "牛市" in r.regime_label or "震荡" in r.regime_label

    def test_detect_with_bearish_data(self) -> None:
        from core.portfolio.regime import MarketRegimeDetector

        dates = pd.date_range("2020-01-01", "2026-06-07", freq="B")
        # 模拟下跌 + 高波动 + 高PE
        close = pd.Series(100 * (0.9997 ** np.arange(len(dates))), index=dates)
        pe = pd.Series(30 + 5 * np.random.randn(len(dates)).cumsum() / 10, index=dates)
        df = pd.DataFrame({
            "close": close,
            "pe_ttm": pe,
        }, index=dates)

        d = MarketRegimeDetector()
        r = d.detect(date(2026, 6, 7), df)
        # 熊市时应给低评分
        assert r.composite_score < 0.7

    def test_four_dimensions_all_computed(self) -> None:
        from core.portfolio.regime import MarketRegimeDetector

        dates = pd.date_range("2020-01-01", "2026-06-07", freq="B")
        df = pd.DataFrame({
            "close": 100 * (1.0002 ** np.arange(len(dates))),
            "pe_ttm": 15 + 2 * np.random.randn(len(dates)),
        }, index=dates)

        d = MarketRegimeDetector()
        r = d.detect(date(2026, 6, 7), df)
        # regime 含 5 个维度（trend/volatility/valuation/breadth/overseas）
        assert set(r.dimension_scores.keys()) == {
            "trend", "volatility", "valuation", "breadth", "overseas",
        }
        for v in r.dimension_scores.values():
            assert 0.0 <= v <= 1.0

    def test_missing_dimensions_dont_dilute_score(self) -> None:
        """宽基路径只有 close：breadth/valuation/overseas 无数据，
        应剔除后按剩余权重归一化，而不是按 0.5 计权稀释趋势/波动。"""
        from core.portfolio.regime import MarketRegimeDetector

        dates = pd.date_range("2020-01-01", "2026-06-07", freq="B")
        close = pd.Series(100 * (1.0005 ** np.arange(len(dates))), index=dates)
        df = pd.DataFrame({"close": close}, index=dates)

        d = MarketRegimeDetector({"dimension_weights": {
            "trend": 0.30, "volatility": 0.25, "breadth": 0.15,
            "overseas": 0.30, "valuation": 0.0,
        }})
        r = d.detect(date(2026, 6, 7), df)
        expected = (r.dimension_scores["trend"] * 0.30
                    + r.dimension_scores["volatility"] * 0.25) / 0.55
        assert abs(r.composite_score - expected) < 1e-9
        # 报告口径仍列出全部 5 个维度，无数据的显示中性 0.5
        assert r.dimension_scores["breadth"] == 0.5

    def test_regime_result_repr(self) -> None:
        from core.portfolio.regime import RegimeResult
        r = RegimeResult(regime_label="牛市低波动", composite_score=0.85)
        assert "牛市低波动" in repr(r)


# ═══════════════════════════════════════════════
# Portfolio Optimizer
# ═══════════════════════════════════════════════

class TestPortfolioOptimizer:
    def test_equal_weight(self) -> None:
        from core.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        scores = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
        w = opt.equal_weight(scores, top_n=3)
        assert len(w) == 3
        assert all(abs(v - 1 / 3) < 0.01 for v in w.values())

    def test_equal_weight_empty(self) -> None:
        from core.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        w = opt.equal_weight({})
        assert w == {}

    def test_signal_weighted(self) -> None:
        from core.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        scores = {"A": 0.9, "B": 0.3, "C": 0.3}
        w = opt.signal_weighted(scores)
        assert w["A"] > w["B"]
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_signal_weighted_cap(self) -> None:
        from core.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        scores = {"A": 0.8, "B": 0.2}
        w = opt.signal_weighted(scores, max_single=0.50)
        # A gets 0.8, B gets 0.2, both below 0.5 cap
        assert w["A"] <= 0.50 + 0.01
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_inverse_vol_weight(self) -> None:
        from core.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        scores = {"A": 0.9, "B": 0.8, "C": 0.7}
        vols = {"A": 0.3, "B": 0.15, "C": 0.10}
        w = opt.inverse_vol_weight(scores, vols, top_n=3)
        # C 波动最低 → 权重最高
        assert w["C"] > w["A"]

    def test_inverse_vol_weight_missing_vol(self) -> None:
        from core.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        scores = {"A": 0.9, "B": 0.8}
        # 没有波动率 → fallback 等权
        w = opt.inverse_vol_weight(scores, {}, top_n=2)
        assert len(w) == 2

    def test_risk_parity(self) -> None:
        from core.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        scores = {"A": 0.9, "B": 0.8, "C": 0.7}
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        returns = pd.DataFrame({
            "A": np.random.randn(120) * 0.02,
            "B": np.random.randn(120) * 0.01,
            "C": np.random.randn(120) * 0.03,
        }, index=dates)
        w = opt.risk_parity(scores, returns, top_n=3)
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_min_cvar(self) -> None:
        from core.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        scores = {"A": 0.9, "B": 0.8}
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        returns = pd.DataFrame({
            "A": np.random.randn(120) * 0.02,
            "B": np.random.randn(120) * 0.01,
        }, index=dates)
        w = opt.min_cvar(scores, returns, top_n=2)
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_apply_constraints(self) -> None:
        from core.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        industry = {"A": "白酒", "B": "白酒", "C": "银行"}
        original_white = weights["A"] + weights["B"]  # 0.8
        w = opt.apply_constraints(weights, industry, max_industry=0.40)
        # 白酒应被显著缩减
        white_spirit = w.get("A", 0) + w.get("B", 0)
        assert white_spirit < original_white, f"白酒{white_spirit} 应 < 原始{original_white}"
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_apply_constraints_no_violation(self) -> None:
        from core.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        weights = {"A": 0.3, "B": 0.3, "C": 0.4}
        industry = {"A": "白酒", "B": "银行", "C": "芯片"}
        w = opt.apply_constraints(weights, industry, max_industry=0.40)
        assert abs(sum(w.values()) - 1.0) < 0.01


# ═══════════════════════════════════════════════
# Kelly
# ═══════════════════════════════════════════════

class TestKelly:
    def test_f_star_positive(self) -> None:
        from core.portfolio.kelly import KellyCalculator
        kc = KellyCalculator()
        # 构造确定性的正收益序列（避免随机生成导致的偶然失败）
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.002, 0.015, 252))
        f = kc.calculate_f_star(returns)
        assert f > 0, f"f*={f} 应该为正"

    def test_short_series_returns_zero(self) -> None:
        from core.portfolio.kelly import KellyCalculator
        kc = KellyCalculator()
        f = kc.calculate_f_star(pd.Series(np.random.randn(10)))
        assert f == 0.0

    def test_half_kelly(self) -> None:
        from core.portfolio.kelly import KellyCalculator
        kc = KellyCalculator()
        # Use strong positive returns to ensure f* > 0
        returns = pd.Series(np.random.normal(0.003, 0.015, 252))
        f_half = kc.half_kelly(returns)
        f_full = kc.calculate_f_star(returns)
        if f_full > 0:
            assert abs(f_half - f_full * 0.5) < 0.01
        else:
            assert f_half == 0.0

    def test_quarter_kelly(self) -> None:
        from core.portfolio.kelly import KellyCalculator
        kc = KellyCalculator()
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        f_q = kc.quarter_kelly(returns)
        assert f_q >= 0

    def test_discrete_kelly(self) -> None:
        from core.portfolio.kelly import KellyCalculator
        f = KellyCalculator.discrete_kelly(win_prob=0.6, win_loss_ratio=2.0)
        # f* = (0.6*2 - 0.4) / 2 = 0.4
        assert abs(f - 0.4) < 0.01

    def test_discrete_kelly_losing(self) -> None:
        from core.portfolio.kelly import KellyCalculator
        f = KellyCalculator.discrete_kelly(win_prob=0.3, win_loss_ratio=1.0)
        assert f == 0.0

    def test_rolling_kelly(self) -> None:
        from core.portfolio.kelly import KellyCalculator
        kc = KellyCalculator()
        returns = pd.Series(np.random.normal(0.0005, 0.015, 500))
        roll = kc.rolling_kelly(returns, window=252, method="half")
        assert len(roll) == 500
        assert (roll >= 0).all()


# ═══════════════════════════════════════════════
# Exit Monitor
# ═══════════════════════════════════════════════

class TestExitMonitor:
    def test_stop_loss_atr(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor()
        # entry=100, atr=5, multiplier=2 → stop=90
        sig = m.check_stop_loss("600519", entry_price=100.0, current_price=89.0, atr=5.0)
        assert sig is not None
        assert sig.exit_type == "stop_loss"

    def test_stop_loss_not_triggered(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor()
        sig = m.check_stop_loss("600519", entry_price=100.0, current_price=95.0, atr=5.0)
        assert sig is None

    def test_stop_loss_pct(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor({"stop_loss_pct": 0.08})
        # entry=100, 8%止损→92, current=91→触发
        sig = m.check_stop_loss("600519", 100.0, 91.0)
        assert sig is not None

    def test_take_profit_target(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor({"take_profit_target": 0.20})
        sig = m.check_take_profit("600519", entry_price=100.0, current_price=121.0)
        assert sig is not None
        assert sig.exit_type == "take_profit"

    def test_trailing_stop(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor({"trailing_stop_pct": 0.10})
        # 最高132, 回撤10%→118.8, current=115→触发
        sig = m.check_take_profit("600519", 100.0, 115.0, highest_price=132.0)
        assert sig is not None

    def test_time_stop(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor({"max_hold_days": 30})
        entry = date(2026, 5, 1)
        current = date(2026, 6, 7)  # 37 days
        sig = m.check_time_stop("600519", entry, current, 100.0, 100.0)
        assert sig is not None
        assert sig.exit_type == "time_stop"

    def test_time_stop_not_triggered(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor()
        entry = date(2026, 6, 1)
        current = date(2026, 6, 7)  # 6 days
        sig = m.check_time_stop("600519", entry, current, 100.0, 100.0)
        assert sig is None

    def test_logic_breakdown_roe(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor({"roe_decline_threshold": 0.30})
        alert = m.check_logic_breakdown("600519", {
            "roe": 0.08, "prev_roe": 0.25, "prev2_roe": 0.28,
        })
        assert alert is not None
        assert alert.alert_type == "logic_breakdown"
        assert "ROE" in alert.detail

    def test_logic_breakdown_cf(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor({"cf_ratio_threshold": 0.5})
        alert = m.check_logic_breakdown("000001", {
            "cf_ratio": 0.3, "prev_cf_ratio": 0.6,
        })
        assert alert is not None
        assert "现金流" in alert.detail

    def test_logic_breakdown_fraud(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor()
        alert = m.check_logic_breakdown("000888", {"fraud_flag": 1})
        assert alert is not None
        assert alert.severity == "critical"

    def test_logic_breakdown_none(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor()
        alert = m.check_logic_breakdown("600519", {"roe": 0.25, "prev_roe": 0.24})
        assert alert is None

    def test_valuation_extreme(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor({"valuation_sigma": 2.0})
        pe_history = pd.Series(np.random.normal(20, 5, 200))
        alert = m.check_valuation_extreme("600519", current_pe=50.0, pe_history=pe_history)
        assert alert is not None
        assert alert.alert_type == "valuation_extreme"

    def test_valuation_normal(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor()
        pe_history = pd.Series(np.random.normal(20, 5, 200))
        alert = m.check_valuation_extreme("600519", current_pe=22.0, pe_history=pe_history)
        assert alert is None

    def test_trailing_stop_tracking(self) -> None:
        from core.portfolio.exit_monitor import ExitMonitor
        m = ExitMonitor()
        m.register_position("600519", 100.0)
        # 初始止损=92
        assert m.get_stop_price("600519") == pytest.approx(92.0)
        # 价格上涨到120，止损应上移
        new_stop = m.update_trailing_stop("600519", 120.0)
        # new stop = 120 * (1 - 0.10) = 108
        assert new_stop == pytest.approx(108.0)
        # 价格回落到110，止损不应下移
        new_stop2 = m.update_trailing_stop("600519", 110.0)
        assert new_stop2 == pytest.approx(108.0)

        m.remove_position("600519")
        assert m.get_stop_price("600519") is None


# ═══════════════════════════════════════════════
# Pre-Trade Risk
# ═══════════════════════════════════════════════

class TestPreTradeRisk:
    def test_all_pass(self) -> None:
        from core.risk.pre_trade import PreTradeRiskChecker
        checker = PreTradeRiskChecker()
        result = checker.check(
            code="600519", side="buy", price=1680.0, shares=10,  # 16,800 远低于总资产
            account_id=1, current_positions={}, current_cash=500_000,
            total_asset=1_000_000, current_prices={"600519": 1680.0},
        )
        assert result.passed is True

    def test_blacklist_st(self) -> None:
        from core.risk.pre_trade import PreTradeRiskChecker
        checker = PreTradeRiskChecker()
        result = checker.check(
            code="000888", side="buy", price=10.0, shares=100,
            account_id=1, current_positions={}, current_cash=100_000,
            total_asset=100_000, current_prices={}, is_st=True,
        )
        assert result.passed is False
        assert "ST" in result.reject_reason

    def test_suspension(self) -> None:
        from core.risk.pre_trade import PreTradeRiskChecker
        checker = PreTradeRiskChecker()
        result = checker.check(
            code="000888", side="buy", price=10.0, shares=100,
            account_id=1, current_positions={}, current_cash=100_000,
            total_asset=100_000, current_prices={}, is_suspended=True,
        )
        assert not result.passed
        assert "停牌" in result.reject_reason

    def test_single_stock_limit_buy(self) -> None:
        from core.risk.pre_trade import PreTradeRiskChecker
        checker = PreTradeRiskChecker()
        # 已经持有 600519 占 25%（超过20%上限）
        result = checker.check(
            code="600519", side="buy", price=1680.0, shares=50,
            account_id=1,
            current_positions={"600519": 100},  # 100*1680=168000
            current_cash=400_000,
            total_asset=400_000,
            current_prices={"600519": 1680.0},
        )
        # 现有 168000/400000=42% > 25% 硬上限 → 拒绝
        assert not result.passed

    def test_industry_limit(self) -> None:
        from core.risk.pre_trade import PreTradeRiskChecker
        checker = PreTradeRiskChecker()
        result = checker.check(
            code="600519", side="buy", price=1680.0, shares=100,
            account_id=1,
            current_positions={"000858": 2000},  # 五粮液，同行业
            current_cash=1_000_000,
            total_asset=1_000_000,
            current_prices={"600519": 1680.0, "000858": 150.0},
            industry_map={"600519": "白酒", "000858": "白酒"},
        )
        # 白酒已有 300000, 加 168000 = 468000/1M = 46.8% > 40%
        assert not result.passed
        assert "白酒" in result.reject_reason

    def test_cash_insufficient(self) -> None:
        from core.risk.pre_trade import PreTradeRiskChecker
        checker = PreTradeRiskChecker()
        result = checker.check(
            code="600519", side="buy", price=1680.0, shares=10,  # 16,800
            account_id=1, current_positions={}, current_cash=500,  # 只有500现金
            total_asset=500_000, current_prices={"600519": 1680.0},
        )
        assert not result.passed
        assert "资金" in result.reject_reason

    def test_single_order_amount(self) -> None:
        from core.risk.pre_trade import PreTradeRiskChecker
        checker = PreTradeRiskChecker({"single_order_max_amount": 100_000})
        result = checker.check(
            code="600519", side="buy", price=1680.0, shares=100,
            account_id=1, current_positions={}, current_cash=500_000,
            total_asset=500_000, current_prices={"600519": 1680.0},
        )
        # 168,000 > 100,000 → 拒绝
        assert not result.passed

    def test_daily_order_counter(self) -> None:
        from core.risk.pre_trade import PreTradeRiskChecker
        checker = PreTradeRiskChecker({"daily_max_orders": 5})
        checker._order_count_day = 5
        result = checker.check(
            code="000001", side="buy", price=10.0, shares=100,
            account_id=1, current_positions={}, current_cash=10_000,
            total_asset=10_000, current_prices={},
        )
        assert not result.passed

    def test_reset_daily(self) -> None:
        from core.risk.pre_trade import PreTradeRiskChecker
        checker = PreTradeRiskChecker()
        checker._order_count_day = 100
        checker.reset_daily()
        assert checker._order_count_day == 0

    def test_blacklist_manage(self) -> None:
        from core.risk.pre_trade import PreTradeRiskChecker
        checker = PreTradeRiskChecker()
        checker.add_blacklist("000888", "手工加入")
        result = checker.check(
            code="000888", side="buy", price=10.0, shares=100,
            account_id=1, current_positions={}, current_cash=100_000,
            total_asset=100_000, current_prices={},
        )
        assert not result.passed
        checker.remove_blacklist("000888")
        result2 = checker.check(
            code="000888", side="buy", price=10.0, shares=100,
            account_id=1, current_positions={}, current_cash=100_000,
            total_asset=100_000, current_prices={},
        )
        assert result2.passed


# ═══════════════════════════════════════════════
# Portfolio Risk
# ═══════════════════════════════════════════════

class TestPortfolioRisk:
    def test_no_drawdown(self) -> None:
        from core.risk.portfolio_risk import PortfolioRiskManager
        mgr = PortfolioRiskManager()
        action = mgr.check_drawdown(current_nav=1_100_000, peak_nav=1_000_000)
        assert action is None

    def test_daily_drawdown_triggers(self) -> None:
        from core.risk.portfolio_risk import PortfolioRiskManager
        mgr = PortfolioRiskManager({"drawdown_daily_circuit": 0.10})
        # peak=1M, current=850K → drawdown=15%
        action = mgr.check_drawdown(current_nav=850_000, peak_nav=1_000_000)
        assert action is not None
        assert action.action_type == "reduce_trading"

    def test_rolling_drawdown_triggers(self) -> None:
        from core.risk.portfolio_risk import PortfolioRiskManager
        mgr = PortfolioRiskManager({
            "drawdown_rolling_circuit": 0.05,
            "rolling_window_days": 5,
        })
        # Feed a sequence that creates a rolling drawdown
        nvs = [1_000_000, 990_000, 980_000, 970_000, 960_000, 940_000]
        action = None
        for nv in nvs:
            action = mgr.check_drawdown(nv, peak_nav=1_000_000)
        assert action is not None
        assert action.action_type == "reduce_all"

    def test_var_calculation(self) -> None:
        from core.risk.portfolio_risk import PortfolioRiskManager
        mgr = PortfolioRiskManager()
        returns = pd.Series(np.random.normal(0, 0.02, 500))
        var = mgr.calc_var(returns)
        assert var > 0

    def test_cvar_calculation(self) -> None:
        from core.risk.portfolio_risk import PortfolioRiskManager
        mgr = PortfolioRiskManager()
        returns = pd.Series(np.random.normal(0, 0.02, 500))
        cvar = mgr.calc_cvar(returns)
        assert cvar > 0
        # CVaR >= VaR（tail average ≥ threshold）
        assert cvar >= mgr.calc_var(returns) - 0.01

    def test_vol_targeting_reduce(self) -> None:
        from core.risk.portfolio_risk import PortfolioRiskManager
        mgr = PortfolioRiskManager({"vol_target": 0.15})
        # 当前波动 30% → 应降仓至 15/30 = 0.5
        target = mgr.vol_targeting(current_vol=0.30, current_position_ratio=1.0)
        assert target < 0.6
        assert target > 0.4

    def test_vol_targeting_no_change(self) -> None:
        from core.risk.portfolio_risk import PortfolioRiskManager
        mgr = PortfolioRiskManager({"vol_target": 0.15})
        target = mgr.vol_targeting(current_vol=0.15, current_position_ratio=1.0)
        assert abs(target - 1.0) < 0.05

    def test_vol_calculation(self) -> None:
        from core.risk.portfolio_risk import PortfolioRiskManager
        mgr = PortfolioRiskManager()
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        vol = mgr.calc_portfolio_volatility(returns)
        assert 0.10 < vol < 0.50

    def test_reset(self) -> None:
        from core.risk.portfolio_risk import PortfolioRiskManager
        mgr = PortfolioRiskManager()
        mgr.feed_nav(1_000_000)
        mgr.feed_nav(900_000)
        assert len(mgr._nav_history) == 2
        mgr.reset()
        assert len(mgr._nav_history) == 0
        assert not mgr.is_circuit_breaker_triggered


# ═══════════════════════════════════════════════
# Extreme Scenario
# ═══════════════════════════════════════════════

class TestExtremeScenario:
    def test_circuit_breaker_generates_sells(self) -> None:
        from core.risk.extreme_scenario import ExtremeScenarioHandler
        handler = ExtremeScenarioHandler()
        positions = {"600519": 100, "000001": 500, "300750": 200}
        prices = {"600519": 1680.0, "000001": 12.0, "300750": 210.0}
        intents = handler.on_circuit_breaker(positions, prices)
        assert len(intents) == 3
        assert all(i.side == "sell" for i in intents)

    def test_circuit_breaker_empty(self) -> None:
        from core.risk.extreme_scenario import ExtremeScenarioHandler
        handler = ExtremeScenarioHandler()
        intents = handler.on_circuit_breaker({}, {})
        assert intents == []

    def test_liquidity_crisis_detects_limit_down(self) -> None:
        from core.risk.extreme_scenario import ExtremeScenarioHandler
        handler = ExtremeScenarioHandler()
        positions = {"000888": 100}
        prices = {"000888": 9.0}
        daily = {
            "000888": {
                "close": 9.0,   # 10*(1-0.1) = 9.0 → 跌停
                "pre_close": 10.0,
                "volume": 0,     # 无量封板
                "is_suspended": False,
            },
        }
        result = handler.on_liquidity_crisis(positions, prices, daily)
        assert len(result["executable"]) == 0
        assert len(result["unexecutable"]) == 1
        assert result["unexecutable"][0]["reason"] == "跌停封板"

    def test_liquidity_crisis_hedge_advice(self) -> None:
        from core.risk.extreme_scenario import ExtremeScenarioHandler
        handler = ExtremeScenarioHandler()
        positions = {"600519": 200, "000001": 10000}
        prices = {"600519": 1680.0, "000001": 12.0}
        daily = {
            "600519": {"close": 1512.0, "pre_close": 1680.0, "volume": 0, "is_suspended": False},
            "000001": {"close": 10.8, "pre_close": 12.0, "volume": 0, "is_suspended": False},
        }
        result = handler.on_liquidity_crisis(positions, prices, daily)
        hedge = result.get("hedge_advice", {})
        assert len(hedge) > 0
        assert "lots" in list(hedge.values())[0]

    def test_liquidity_crisis_suspended(self) -> None:
        from core.risk.extreme_scenario import ExtremeScenarioHandler
        handler = ExtremeScenarioHandler()
        positions = {"600519": 100}
        prices = {"600519": 1680.0}
        daily = {"600519": {"is_suspended": True, "close": 1680.0, "volume": 0}}
        result = handler.on_liquidity_crisis(positions, prices, daily)
        assert len(result["unexecutable"]) == 1
        assert result["unexecutable"][0]["reason"] == "停牌"

    def test_emergency_buffer(self) -> None:
        from core.risk.extreme_scenario import ExtremeScenarioHandler
        handler = ExtremeScenarioHandler({"emergency_cash_buffer_ratio": 0.05})
        ready, shortfall = handler.check_emergency_ready(cash=30_000, total_asset=1_000_000)
        assert not ready
        assert shortfall > 0

        ready2, _ = handler.check_emergency_ready(cash=60_000, total_asset=1_000_000)
        assert ready2
