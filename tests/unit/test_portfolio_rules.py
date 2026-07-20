"""portfolio_rules.py + 中性化打分接入测试（路线图第 7 阶段）。"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research.portfolio_rules import (
    select_target_portfolio, per_stock_weight, portfolio_rules_from_config,
)


def _scores(pairs: list[tuple[str, float]]) -> pd.Series:
    return pd.Series(dict(pairs))


class TestSelectTargetPortfolio:
    def test_plain_top_n_without_rules(self):
        s = _scores([("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)])
        assert select_target_portfolio(s, 2) == ["A", "B"]

    def test_empty_and_zero_n(self):
        assert select_target_portfolio(pd.Series(dtype=float), 3) == []
        assert select_target_portfolio(_scores([("A", 1.0)]), 0) == []

    def test_rank_buffer_keeps_held_in_buffer_zone(self):
        """已持有票排名在 (top_n, buffer] 区间 → 保留，不被新票替换。"""
        s = _scores([("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)])
        # 持有 C（排名3），top_n=2, buffer=3 → C 保留，A 补足名额，B 挤出
        out = select_target_portfolio(s, 2, held={"C"}, rank_buffer=3)
        assert out == ["A", "C"]

    def test_rank_buffer_drops_held_beyond_buffer(self):
        s = _scores([("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)])
        # 持有 D（排名4）> buffer=3 → 跌出，正常 top_n
        out = select_target_portfolio(s, 2, held={"D"}, rank_buffer=3)
        assert out == ["A", "B"]

    def test_rank_buffer_none_is_pure_topn(self):
        s = _scores([("A", 0.9), ("B", 0.8), ("C", 0.7)])
        out = select_target_portfolio(s, 2, held={"C"}, rank_buffer=None)
        assert out == ["A", "B"]

    def test_buffer_reduces_turnover_vs_pure_topn(self):
        """排名小幅抖动时 buffer 组合不动，纯 top_n 换手。"""
        held = {"A", "B"}
        # B 从第2 滑到第3（噪声级抖动）
        s = _scores([("A", 0.9), ("C", 0.85), ("B", 0.84), ("D", 0.5)])
        pure = select_target_portfolio(s, 2, held=held, rank_buffer=None)
        buffered = select_target_portfolio(s, 2, held=held, rank_buffer=3)
        assert pure == ["A", "C"]        # 卖 B 买 C（换手）
        assert buffered == ["A", "B"]    # B 在 buffer 内保留（零换手）

    def test_max_per_sector(self):
        s = _scores([("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)])
        sectors = {"A": "bank", "B": "bank", "C": "bank", "D": "semi"}
        out = select_target_portfolio(s, 3, sector_map=sectors, max_per_sector=2)
        assert out == ["A", "B", "D"]  # C 被行业配额跳过，D 补位

    def test_unmapped_codes_not_sector_limited(self):
        s = _scores([("A", 0.9), ("B", 0.8), ("C", 0.7)])
        out = select_target_portfolio(s, 3, sector_map={"A": "bank"}, max_per_sector=1)
        assert out == ["A", "B", "C"]  # B/C 无标签不受限

    def test_held_counts_against_sector_quota(self):
        """buffer 保留的持仓也占用行业配额。"""
        s = _scores([("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)])
        sectors = {c: "bank" for c in "ABC"} | {"D": "semi"}
        # 持有 C（排名3, buffer 内）保留后 bank 已满1 → A 还能进（满2），B 被跳过
        out = select_target_portfolio(s, 3, held={"C"}, rank_buffer=4,
                                      sector_map=sectors, max_per_sector=2)
        assert out == ["A", "C", "D"]

    def test_deterministic_order_by_rank(self):
        s = _scores([("B", 0.8), ("A", 0.9), ("C", 0.7)])
        assert select_target_portfolio(s, 3) == ["A", "B", "C"]


class TestPerStockWeight:
    def test_equal_weight_default(self):
        assert per_stock_weight(4) == pytest.approx(0.25)

    def test_cap_binds_small_portfolio(self):
        assert per_stock_weight(3, max_weight=0.10) == pytest.approx(0.10)

    def test_cap_not_binding_large_portfolio(self):
        assert per_stock_weight(20, max_weight=0.10) == pytest.approx(0.05)

    def test_zero_n(self):
        assert per_stock_weight(0) == 0.0


class TestRulesFromConfig:
    def test_defaults_all_off(self):
        rules = portfolio_rules_from_config({"strategy": {"top_n": 3}})
        assert rules == {"rank_buffer": None, "max_per_sector": None, "max_weight": None}

    def test_configured(self):
        rules = portfolio_rules_from_config({"strategy": {
            "rank_buffer": 40, "max_per_sector": 6, "max_weight_per_stock": 0.1}})
        assert rules == {"rank_buffer": 40, "max_per_sector": 6, "max_weight": 0.1}


class TestNeutralizedScoring:
    """_score_stocks 的中性化路径（构造小型 features 面板直接调用）。"""

    def _strategy(self, neutralize: dict | None = None, sector_map: dict | None = None):
        from research.run_backtest_demo import ConfigDrivenStrategy
        config = {
            "strategy": {"top_n": 2, **({"neutralize": neutralize} if neutralize else {})},
            "factors": {"momentum_20d": {"enabled": True, "weight": 1.0}},
        }
        st = ConfigDrivenStrategy(config)
        if sector_map is not None:
            st._sector_map = sector_map
            st._neut_industry = bool(neutralize and neutralize.get("industry"))
        return st

    def test_no_neutralize_matches_old_behavior(self):
        st = self._strategy()
        feats = pd.DataFrame({"momentum_20d": [0.3, 0.2, 0.1]}, index=["A", "B", "C"])
        scores = st._score_stocks(feats)
        assert scores.idxmax() == "A"

    def test_industry_neutralization_changes_ranking(self):
        """行业间水平差被中和：行业内相对强者胜出。

        bank 行业整体动量高（0.50/0.40），semi 整体低（0.10/0.02）。
        原始排名 top2 = 两只 bank；行业中性化后按「行业内偏离」排名，
        每个行业各出一只强者。
        """
        sector_map = {"B1": "bank", "B2": "bank", "S1": "semi", "S2": "semi"}
        st = self._strategy(neutralize={"industry": True}, sector_map=sector_map)
        # 样本 ≥10 才回归：扩到 12 只（两行业各 6）
        codes = [f"B{i}" for i in range(6)] + [f"S{i}" for i in range(6)]
        st._sector_map = {c: ("bank" if c.startswith("B") else "semi") for c in codes}
        vals = [0.50, 0.40, 0.41, 0.42, 0.43, 0.44,   # bank：B0 内部最强
                0.10, 0.02, 0.03, 0.04, 0.05, 0.06]   # semi：S0 内部最强
        feats = pd.DataFrame({"momentum_20d": vals}, index=codes)
        scores = st._score_stocks(feats)
        top2 = set(scores.nlargest(2).index)
        assert top2 == {"B0", "S0"}  # 未中性化时会是 {B0, B5}

    def test_mktcap_neutralization_uses_helper_column(self):
        st = self._strategy(neutralize={"mktcap": True})
        st._neut_mktcap = True
        rng = np.random.default_rng(5)
        n = 20
        codes = [f"C{i}" for i in range(n)]
        mktcap = np.exp(rng.normal(23, 1, n))  # 1e10 量级
        # 因子值 = 纯规模效应 + 微噪声 → 中性化后应基本无信息
        factor = np.log(mktcap) * 0.1 + rng.normal(0, 1e-4, n)
        feats = pd.DataFrame({"momentum_20d": factor, "_mktcap": mktcap}, index=codes)
        scores = st._score_stocks(feats)
        # 中性化后的排名与市值排名不再一致
        mc_rank = pd.Series(mktcap, index=codes).rank()
        sc_rank = scores.rank()
        assert abs(mc_rank.corr(sc_rank, method="spearman")) < 0.9

    def test_underscore_columns_never_scored(self):
        """_mktcap 等辅助列不参与打分（不在 factors 配置中即天然跳过）。"""
        st = self._strategy()
        feats = pd.DataFrame({"momentum_20d": [0.3, 0.1], "_mktcap": [1e10, 9e12]},
                             index=["A", "B"])
        scores = st._score_stocks(feats)
        assert scores.idxmax() == "A"  # 巨大 _mktcap 不影响
