"""factor_ic.py 纯函数层测试（路线图第 5 阶段）。

不联网、不加载真实行情：因子面板与前瞻收益面板全部人工构造。
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research.factor_ic import (
    compute_ic_series, ic_stats, quintile_backtest, evaluate_factor,
    filter_candidates_by_report, month_anchor_dates, build_forward_returns,
    factor_direction,
)


def _codes(n: int) -> list[str]:
    return [f"{600000 + i}" for i in range(n)]


def _panels(n_dates: int = 12, n_codes: int = 20, rho: float = 1.0, seed: int = 7):
    """构造因子/前瞻收益面板：rho=1 完美正相关，rho=0 纯噪声。"""
    rng = np.random.default_rng(seed)
    codes = _codes(n_codes)
    fac_panel, fwd_panel = {}, {}
    for i in range(n_dates):
        d = date(2024, 1 + i % 12, 1 + i // 12)
        fac = pd.Series(rng.normal(size=n_codes), index=codes)
        noise = pd.Series(rng.normal(size=n_codes), index=codes)
        fwd = rho * fac.rank() / n_codes + (1 - rho) * noise
        fac_panel[d] = fac
        fwd_panel[d] = fwd
    return fac_panel, fwd_panel


class TestComputeICSeries:
    def test_perfect_positive_correlation(self):
        fac, fwd = _panels(rho=1.0)
        ic = compute_ic_series(fac, fwd)
        assert len(ic) == 12
        assert (ic > 0.99).all()

    def test_noise_factor_ic_near_zero(self):
        fac, fwd = _panels(rho=0.0, n_dates=24, n_codes=50)
        ic = compute_ic_series(fac, fwd)
        assert abs(ic.mean()) < 0.15

    def test_min_obs_skips_small_cross_section(self):
        fac, fwd = _panels(n_codes=5)  # < MIN_CROSS_SECTION=8
        ic = compute_ic_series(fac, fwd)
        assert len(ic) == 0

    def test_constant_factor_skipped(self):
        codes = _codes(20)
        d = date(2024, 1, 2)
        fac = {d: pd.Series(0.0, index=codes)}  # 无数据因子恒 0
        fwd = {d: pd.Series(np.arange(20, dtype=float), index=codes)}
        ic = compute_ic_series(fac, fwd)
        assert len(ic) == 0

    def test_index_intersection(self):
        d = date(2024, 1, 2)
        fac = {d: pd.Series(np.arange(10, dtype=float), index=_codes(10))}
        # 收益面板只覆盖后 8 只 + 2 只额外的
        fwd_idx = _codes(12)[2:]
        fwd = {d: pd.Series(np.arange(10, dtype=float), index=fwd_idx)}
        ic = compute_ic_series(fac, fwd)
        assert len(ic) == 1  # 共同 8 只，刚好达 min_obs


class TestICStats:
    def test_positive_factor(self):
        ic = pd.Series([0.1, 0.15, 0.05, 0.2, 0.1])
        s = ic_stats(ic, direction=1)
        assert s["ic_mean"] == pytest.approx(0.12, abs=1e-9)
        assert s["icir"] > 1.0
        assert s["positive_rate"] == 1.0
        assert s["n_periods"] == 5

    def test_direction_adjustment_flips_sign(self):
        """反向因子（如波动率）原始 IC 为负，方向调整后应为正。"""
        ic = pd.Series([-0.1, -0.15, -0.05, -0.2])
        s = ic_stats(ic, direction=-1)
        assert s["ic_mean"] > 0
        assert s["icir"] > 0
        assert s["positive_rate"] == 1.0

    def test_empty_series(self):
        s = ic_stats(pd.Series(dtype=float))
        assert s["n_periods"] == 0
        assert s["icir"] == 0.0

    def test_t_stat_scales_with_periods(self):
        ic = pd.Series([0.1, 0.12, 0.08, 0.11] * 4)
        s = ic_stats(ic)
        assert s["t_stat"] == pytest.approx(s["icir"] * np.sqrt(16), rel=1e-6)


class TestQuintileBacktest:
    def test_monotonic_factor(self):
        fac, fwd = _panels(rho=1.0, n_codes=25)
        q = quintile_backtest(fac, fwd, direction=1)
        assert q["n_periods"] == 12
        assert q["top_bottom_spread"] > 0
        assert q["monotonicity"] == pytest.approx(1.0)
        rets = q["quantile_mean_returns"]
        assert len(rets) == 5
        assert rets == sorted(rets)

    def test_inverse_factor_direction(self):
        """反向因子：低因子值组收益高，direction=-1 调整后 spread 应为正。"""
        codes = _codes(25)
        fac_panel, fwd_panel = {}, {}
        for m in range(1, 7):
            d = date(2024, m, 1)
            vals = np.arange(25, dtype=float)
            fac_panel[d] = pd.Series(vals, index=codes)
            fwd_panel[d] = pd.Series(-vals / 100, index=codes)  # 因子越高收益越低
        q = quintile_backtest(fac_panel, fwd_panel, direction=-1)
        assert q["top_bottom_spread"] > 0
        assert q["monotonicity"] == pytest.approx(1.0)

    def test_ties_do_not_crash(self):
        """大量重复因子值（qcut 会炸的场景）应正常分层。"""
        codes = _codes(20)
        d = date(2024, 1, 2)
        fac = {d: pd.Series([1.0] * 10 + [2.0] * 10, index=codes)}
        fwd = {d: pd.Series(np.arange(20, dtype=float), index=codes)}
        q = quintile_backtest(fac, fwd)
        assert q["n_periods"] == 1
        assert len(q["quantile_mean_returns"]) == 5


class TestEvaluateFactor:
    def test_good_factor_passes(self):
        fac, fwd1 = _panels(rho=1.0)
        rep = evaluate_factor("momentum_20d", fac, {1: fwd1}, icir_threshold=0.3)
        assert rep["verdict"] == "pass"
        assert rep["direction"] == 1
        assert "1" in rep["ic_decay"]

    def test_noise_factor_fails(self):
        fac, fwd1 = _panels(rho=0.0, n_dates=24, n_codes=50)
        rep = evaluate_factor("momentum_20d", fac, {1: fwd1}, icir_threshold=0.3)
        assert rep["verdict"] == "fail"

    def test_inverse_factor_direction_applied(self):
        """volatility_* 是反向因子：负 raw IC 调整后 pass。"""
        assert factor_direction("volatility_20d") == -1
        codes = _codes(20)
        fac_panel, fwd_panel = {}, {}
        rng = np.random.default_rng(3)
        for m in range(1, 13):
            d = date(2024, m, 1)
            fac = pd.Series(rng.normal(size=20), index=codes)
            fac_panel[d] = fac
            fwd_panel[d] = -fac  # 因子越高收益越低（反向因子的预期行为）
        rep = evaluate_factor("volatility_20d", fac_panel, {1: fwd_panel},
                              icir_threshold=0.3)
        assert rep["verdict"] == "pass"
        assert rep["icir"] > 0

    def test_ic_decay_horizons(self):
        fac, fwd1 = _panels(rho=1.0)
        _, fwd3 = _panels(rho=0.0, seed=11)  # 3 期后信息消失
        rep = evaluate_factor("momentum_20d", fac, {1: fwd1, 3: fwd3})
        assert rep["ic_decay"]["1"]["icir"] > rep["ic_decay"]["3"]["icir"]


class TestFilterCandidates:
    REPORT = {
        "icir_threshold": 0.3,
        "factors": {
            "momentum_20d": {"icir": 0.55, "n_periods": 30},
            "rsi_14": {"icir": 0.10, "n_periods": 30},
            "volatility_20d": {"icir": 0.42, "n_periods": 30},
            "empty_factor": {"icir": 0.0, "n_periods": 0},
        },
    }

    def test_filtering(self):
        kept, dropped, missing = filter_candidates_by_report(
            ["momentum_20d", "rsi_14", "volatility_20d", "empty_factor", "pb"],
            self.REPORT, min_icir=0.3)
        assert kept == ["momentum_20d", "volatility_20d", "pb"]
        assert dropped == ["rsi_14", "empty_factor"]  # empty: n_periods=0 也剔除
        assert missing == ["pb"]  # 未检验的保留但标注

    def test_higher_threshold(self):
        kept, dropped, _ = filter_candidates_by_report(
            ["momentum_20d", "volatility_20d"], self.REPORT, min_icir=0.5)
        assert kept == ["momentum_20d"]
        assert dropped == ["volatility_20d"]


class TestDataAssembly:
    def test_month_anchor_dates(self):
        days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 1),
                date(2024, 2, 15), date(2024, 3, 4)]
        assert month_anchor_dates(days) == [
            date(2024, 1, 2), date(2024, 2, 1), date(2024, 3, 4)]

    def test_build_forward_returns(self):
        anchors = [date(2024, m, 1) for m in range(1, 5)]
        raw = {d: {"600000": {"close": 10.0 * (1.1 ** i)},
                   "600001": {"close": 20.0}}
               for i, d in enumerate(anchors)}
        fwd = build_forward_returns(raw, anchors, horizons=(1, 3))
        # h=1: 3 个截面有下期；h=3: 只有第一个截面
        assert len(fwd[1]) == 3
        assert len(fwd[3]) == 1
        assert fwd[1][anchors[0]]["600000"] == pytest.approx(0.1)
        assert fwd[1][anchors[0]]["600001"] == pytest.approx(0.0)
        assert fwd[3][anchors[0]]["600000"] == pytest.approx(1.1 ** 3 - 1)

    def test_forward_returns_skip_suspended(self):
        """未来截面缺行情（停牌）的股票应从该截面剔除。"""
        anchors = [date(2024, 1, 1), date(2024, 2, 1)]
        raw = {
            anchors[0]: {"600000": {"close": 10.0}, "600001": {"close": 20.0}},
            anchors[1]: {"600000": {"close": 11.0}},  # 600001 停牌
        }
        fwd = build_forward_returns(raw, anchors, horizons=(1,))
        assert list(fwd[1][anchors[0]].index) == ["600000"]
