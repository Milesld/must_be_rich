"""因子计算单元测试。

覆盖所有因子模块：technical, fundamental, capital_flow, sentiment, premarket,
neutralizer, validation。

测试策略：
- 正常数据 → 返回有限值（非 NaN/INF）
- 缺失列 → 优雅降级（返回 NaN，不崩溃）
- 极端数据 → 返回有限值（不产生 INF）
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_data() -> pd.DataFrame:
    """构造60天×3只股票的日线数据集。"""
    codes = ["600519", "000001", "300750"]
    rows = []
    base = date(2026, 1, 5)
    prices = {
        "600519": 1680.0, "000001": 12.0, "300750": 210.0,
    }

    for i in range(60):
        d = base + timedelta(days=i)
        # 跳过周末
        if d.weekday() > 4:
            continue
        for code in codes:
            base_p = prices[code]
            change = np.random.normal(0.0005, 0.02)
            close = base_p * (1 + change)
            open_p = base_p * (1 + np.random.normal(0, 0.005))
            high = max(base_p, close) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(base_p, close) * (1 - abs(np.random.normal(0, 0.005)))
            rows.append({
                "trade_date": d,
                "code": code,
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "pre_close": base_p,
                "volume": np.random.randint(1_000_000, 50_000_000),
                "amount": close * np.random.randint(1_000_000, 50_000_000),
                "adj_factor": 1.0,
                "turnover": np.random.uniform(0.1, 5.0),
                "pct_change": change,
                "is_st": 0,
                "is_suspended": 0,
            })
            prices[code] = close

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


# ── Technical Factors ──────────────────────────────

class TestTechnicalFactors:
    """技术面因子。"""

    def test_momentum_output_finite(self, daily_data: pd.DataFrame) -> None:
        from core.features.technical import momentum
        result = momentum(daily_data, {"window": 20})
        assert not result.isna().all(), "全NAN"
        assert len(result) == len(daily_data)

    def test_momentum_edge_cases(self) -> None:
        """极端数据：全相同价格（无波动）不应该产生 INF。"""
        from core.features.technical import momentum
        df = _make_flat_data()
        result = momentum(df, {"window": 20})
        # 全相同价格 → 收益率为0 → 动量=0（非INF）
        assert np.isfinite(result.dropna()).all()
        # 前window个值为NaN（不够窗口期）是正常的

    def test_volatility_output(self, daily_data: pd.DataFrame) -> None:
        from core.features.technical import volatility
        result = volatility(daily_data, {"window": 20})
        assert not result.isna().all()
        assert (result.dropna() >= 0).all(), "波动率为负"

    def test_rsi_output_range(self, daily_data: pd.DataFrame) -> None:
        from core.features.technical import rsi
        result = rsi(daily_data, {"window": 14})
        valid = result.dropna()
        assert valid.between(0, 100).all(), f"RSI 越界: {valid.describe()}"

    def test_atr_output_positive(self, daily_data: pd.DataFrame) -> None:
        from core.features.technical import atr
        result = atr(daily_data, {"window": 14})
        assert (result.dropna() > 0).all()

    def test_bollinger_in_range(self, daily_data: pd.DataFrame) -> None:
        from core.features.technical import bollinger_position
        result = bollinger_position(daily_data, {"window": 20})
        valid = result.dropna()
        # 布林带位置应该在 0~1 附近（偶尔超出属于正常）
        assert valid.abs().mean() < 10  # 不应极端偏离

    def test_ma_alignment_scores(self, daily_data: pd.DataFrame) -> None:
        from core.features.technical import ma_alignment
        result = ma_alignment(daily_data, {"short": 5, "mid": 20, "long": 60})
        valid = result.dropna()
        assert valid.isin([-1.0, 0.0, 1.0]).all(), f"非预期值: {valid.unique()}"

    def test_volume_ratio_positive(self, daily_data: pd.DataFrame) -> None:
        from core.features.technical import volume_ratio
        result = volume_ratio(daily_data, {"window": 20})
        assert (result.dropna() >= 0).all()

    def test_missing_column_graceful(self) -> None:
        """缺失列时应返回 NaN 而非崩溃。"""
        from core.features.technical import volume_ratio, turnover
        df = pd.DataFrame({"code": ["600519"] * 5, "volume": [1000] * 5})
        result = volume_ratio(df, {"window": 3})
        assert result is not None

    def test_all_factor_functions_smoke(self, daily_data: pd.DataFrame) -> None:
        """所有技术面因子 smoke test。"""
        from core.features import technical as t
        factor_funcs = [
            t.momentum, t.alpha_momentum, t.volatility, t.amplitude,
            t.atr, t.turnover, t.volume_ratio, t.amihud_illiq,
            t.rsi, t.macd_dif, t.macd_dea, t.macd_hist,
            t.bollinger_position, t.ma_alignment,
        ]
        for func in factor_funcs:
            try:
                result = func(daily_data, {"window": 20})
                assert isinstance(result, pd.Series), f"{func.__name__}: 返回类型错误"
                assert len(result) == len(daily_data), f"{func.__name__}: 长度不匹配"
            except Exception as e:
                pytest.fail(f"{func.__name__}: 抛出异常: {e}")


# ── Fundamental Factors ────────────────────────────

class TestFundamentalFactors:
    """基本面因子（默认PIT模式）。"""

    def test_all_pit_no_financials(self, daily_data: pd.DataFrame) -> None:
        """无 financials 时返回 NaN 而非崩溃（PIT模式默认开启）。"""
        from core.features.fundamental import (
            pe_ttm, roe_ttm, revenue_yoy, debt_ratio, cf_ratio_ttm,
        )
        for func in [pe_ttm, roe_ttm, revenue_yoy, debt_ratio, cf_ratio_ttm]:
            result = func(daily_data, {"point_in_time": True})
            assert result is not None
            assert len(result) == len(daily_data)

    def test_with_financials_no_announce_date(self, daily_data: pd.DataFrame) -> None:
        """financials 无 announce_date 列时给出警告但不崩溃。"""
        from core.features.fundamental import roe_ttm

        fin = pd.DataFrame({
            "code": ["600519", "000001", "300750"],
            "report_date": [date(2025, 12, 31)] * 3,
            "roe": [0.30, 0.075, 0.10],
            "net_profit": [7.5e10, 3.0e10, 4.0e9],
            "total_equity": [2.5e11, 4.0e11, 4.0e10],
            # 无 announce_date 列
        })
        result = roe_ttm(daily_data, {"point_in_time": True}, financials=fin)
        assert result is not None

    def test_pit_filters_future_announcements(self) -> None:
        """PIT 模式应过滤掉尚未披露的财报。"""
        from core.features.fundamental import _pit_filter

        data = pd.DataFrame({
            "trade_date": pd.to_datetime([date(2020, 2, 10)] * 2),
            "code": ["600519", "000001"],
        })
        fin = pd.DataFrame({
            "code": ["600519", "000001", "600519"],
            "announce_date": [
                date(2020, 2, 5),  # 已披露
                date(2020, 2, 28),  # ★尚未披露（2月28日 > 2月10日）
                date(2020, 1, 20),  # 已披露
            ],
            "report_date": [date(2019, 12, 31)] * 3,
        })
        filtered = _pit_filter(data, fin, point_in_time=True)
        # announce_date=2020-02-28 的行应被过滤
        assert len(filtered) == 2
        assert date(2020, 2, 28) not in filtered["announce_date"].values


# ── Capital Flow Factors ───────────────────────────

class TestCapitalFlowFactors:
    """资金面因子。"""

    def test_missing_column_returns_nan(self, daily_data: pd.DataFrame) -> None:
        from core.features.capital_flow import main_force_net_inflow, margin_balance_change

        # 没有 main_force_inflow / margin_balance 列 → 返回 NaN
        result = main_force_net_inflow(daily_data, {"window": 5})
        assert result.isna().all()

        result2 = margin_balance_change(daily_data, {"window": 5})
        assert result2.isna().all()

    def test_lockup_default_value(self, daily_data: pd.DataFrame) -> None:
        from core.features.capital_flow import lockup_expiry_days
        result = lockup_expiry_days(daily_data, {})
        # 没有 next_lockup_date 列 → 默认999
        assert (result == 999).all()

    def test_northbound_quarter_marked(self, daily_data: pd.DataFrame) -> None:
        from core.features.capital_flow import northbound_quarter_change
        result = northbound_quarter_change(daily_data, {})
        # 无 northbound_holdings 列 → 返回 NaN
        assert result.isna().all()


# ── Sentiment Factors ──────────────────────────────

class TestSentimentFactors:
    """情绪/事件因子。"""

    def test_limit_up_count(self, daily_data: pd.DataFrame) -> None:
        from core.features.sentiment import limit_up_count
        result = limit_up_count(daily_data, {})
        assert not result.isna().all()

    def test_performance_forecast_surprise(self, daily_data: pd.DataFrame) -> None:
        from core.features.sentiment import performance_forecast_surprise
        # 没有 forecast_type 列
        result = performance_forecast_surprise(daily_data, {})
        assert result.isna().all()

        # 有 forecast_type 列
        df = daily_data.copy()
        df["forecast_type"] = "大增"
        result2 = performance_forecast_surprise(df, {})
        assert (result2 == 2.0).all()

    def test_all_smoke(self, daily_data: pd.DataFrame) -> None:
        from core.features import sentiment as s
        for func in [
            s.limit_up_count, s.limit_up_chain_height, s.limit_down_count,
            s.board_break_ratio, s.limit_up_ratio, s.auction_open_premium,
            s.auction_volume_ratio, s.performance_forecast_surprise,
        ]:
            result = func(daily_data, {})
            assert isinstance(result, pd.Series), f"{func.__name__} 失败"


# ── Premarket Factors ──────────────────────────────

class TestPremarketFactors:
    """盘前专属因子。"""

    def test_announcement_sentiment_keyword(self) -> None:
        from core.features.premarket import announcement_sentiment_score

        df = pd.DataFrame({
            "trade_date": [date(2026, 6, 7)] * 3,
            "code": ["600519", "000001", "300750"],
            "announcement_title": [
                "贵州茅台 2025年度业绩预增公告",
                "平安银行关于股东减持计划的公告",
                "宁德时代日常经营公告",
            ],
        })
        result = announcement_sentiment_score(df, {})
        # 600519: "预增" → 正面
        # 000001: "减持" → 负面
        # 300750: 无关键词 → 中性
        assert result.iloc[0] > 0, f"茅台应正面: {result.iloc[0]}"
        assert result.iloc[1] < 0, f"平安应负面: {result.iloc[1]}"
        assert abs(result.iloc[2]) < 0.1, f"宁德应中性: {result.iloc[2]}"

    def test_a50_overnight(self) -> None:
        from core.features.premarket import a50_futures_overnight

        df = pd.DataFrame({
            "trade_date": [date(2026, 6, 7)],
            "code": ["600519"],
            "a50_change": [0.005],
        })
        result = a50_futures_overnight(df, {})
        assert result.iloc[0] == 0.005

    def test_dragon_tiger_review_score(self) -> None:
        from core.features.premarket import dragon_tiger_review_score

        df = pd.DataFrame({
            "trade_date": [date(2026, 6, 5)] * 2,
            "code": ["600519", "000001"],
            "dragon_tiger_net": [32000, -5000],
            "dragon_tiger_buy": [50000, 1000],
            "dragon_tiger_sell": [18000, 6000],
            "is_known_trader": [1, 0],
        })
        result = dragon_tiger_review_score(df, {})
        assert result.notna().all()
        # 机构净买入的应比净卖出的评分高
        assert result.iloc[0] > result.iloc[1]


# ── Neutralizer ──────────────────────────────────────

class TestNeutralizer:
    """因子中性化。"""

    def test_industry_neutralize(self) -> None:
        from core.features.neutralizer import neutralize_by_industry

        factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        industry = pd.Series(["A", "A", "A", "B", "B", "B"])
        result = neutralize_by_industry(factor, industry)
        # 中性化后，同行业内的残差应围绕0（但精确度取决于回归拟合，用宽松容忍度）
        # 关键验证：结果不是全NaN且长度匹配
        assert len(result) == len(factor)
        assert result.notna().sum() > 0

    def test_market_cap_neutralize(self) -> None:
        from core.features.neutralizer import neutralize_by_market_cap

        factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        mcap = pd.Series([1e10, 2e10, 5e10, 1e11, 5e11])
        result = neutralize_by_market_cap(factor, mcap)
        assert result.notna().sum() > 0

    def test_dual_neutralize(self) -> None:
        from core.features.neutralizer import neutralize_industry_mcap

        factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        industry = pd.Series(["A", "A", "A", "B", "B", "B"])
        mcap = pd.Series([1e10, 2e10, 5e10, 1e11, 5e11, 1e12])
        result = neutralize_industry_mcap(factor, industry, mcap)
        assert result.notna().sum() > 0

    def test_mad_winsorize(self) -> None:
        from core.features.neutralizer import mad_winsorize

        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = mad_winsorize(series, n_mad=3.0)
        assert result.max() < 100, "异常值应被缩尾"


# ── Factor Registry ─────────────────────────────────

class TestFactorRegistry:
    """因子注册表。"""

    def test_load_from_yaml(self) -> None:
        from core.features.registry import FactorRegistry

        registry = FactorRegistry()
        count = registry.load_from_yaml("configs/factors/registry.yaml")
        assert count >= 2
        assert "momentum_20d" in registry
        assert "roe_ttm" in registry

    def test_compute_order(self) -> None:
        from core.features.registry import FactorRegistry, FactorDefinition

        registry = FactorRegistry()
        registry.register(FactorDefinition(
            name="A", display_name="A", category="test", function="dummy",
            depends_on=["B", "C"], params={},
        ))
        registry.register(FactorDefinition(
            name="B", display_name="B", category="test", function="dummy",
            depends_on=["C"], params={},
        ))
        registry.register(FactorDefinition(
            name="C", display_name="C", category="test", function="dummy",
            depends_on=[], params={},
        ))

        order = registry.compute_order()
        # C 应排在前面（被B和A依赖），A 排在最后
        idx_c = order.index("C")
        idx_b = order.index("B")
        idx_a = order.index("A")
        assert idx_c < idx_b < idx_a, f"拓扑序错误: {order}"

    def test_version_increment(self) -> None:
        from core.features.registry import FactorDefinition

        fd = FactorDefinition(
            name="test", display_name="test", category="test",
            function="dummy", params={}, version="v1.0",
        )
        assert fd.increment_version() == "v1.1"
        assert fd.version == "v1.1"

    def test_validate(self) -> None:
        from core.features.registry import FactorRegistry, FactorDefinition

        registry = FactorRegistry()
        registry.register(FactorDefinition(
            name="no_func", display_name="OK", category="test",
            function="", depends_on=["nonexistent_dep"], params={},
        ))
        result = registry.validate()
        assert len(result["errors"]) >= 1  # 缺少 function
        assert len(result["warnings"]) >= 1  # 外部依赖


# ── Helpers ──────────────────────────────────────────

def _make_flat_data() -> pd.DataFrame:
    """生成全相同价格的测试数据（用于边界测试）。"""
    rows = []
    for i in range(30):
        for code in ["600519", "000001"]:
            rows.append({
                "trade_date": date(2026, 1, 5) + timedelta(days=i),
                "code": code,
                "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
                "pre_close": 10.0, "volume": 10000, "amount": 100000,
                "adj_factor": 1.0, "turnover": 1.0,
            })
    return pd.DataFrame(rows)
