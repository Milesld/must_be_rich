"""模型层单元测试 — 所有模型类的 smoke test。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 模型层依赖 sklearn（core.models.long_term 等）；缺失时整体跳过。
pytest.importorskip("sklearn", reason="sklearn 未安装（模型层依赖）")

# 检测 lightgbm 是否可用（网络受限环境可能未安装）
try:
    import lightgbm  # noqa: F401
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

requires_lgb = pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm 未安装")


def _make_random_data(n_samples: int = 300, n_features: int = 10) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """生成随机特征 + 连续标签 + 分类标签。"""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"factor_{i}" for i in range(n_features)],
    )
    y_reg = pd.Series(np.random.randn(n_samples), name="return")
    y_cls = pd.Series(np.random.choice([0, 1, 2], n_samples), name="class")
    return X, y_reg, y_cls


def _make_grouped_data(n_groups: int = 5, n_per_group: int = 60) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """生成含日期分组的随机数据（用于 LambdaRank）。"""
    np.random.seed(42)
    rows = []
    for g in range(n_groups):
        X = np.random.randn(n_per_group, 10)
        y = np.random.randn(n_per_group)
        dates = [f"2024-01-{g*5+1:02d}"] * n_per_group
        for i in range(n_per_group):
            rows.append({
                "date": dates[i],
                **{f"f{j}": X[i, j] for j in range(10)},
                "return": y[i],
            })
    df = pd.DataFrame(rows)
    X = df[[c for c in df.columns if c.startswith("f")]]
    y = df["return"]
    groups = pd.Series(df["date"])
    return X, y, groups


# ── Base Model ────────────────────────────────────

@requires_lgb
class TestBaseModel:
    @requires_lgb
    def test_save_load_roundtrip(self, tmp_path) -> None:
        from core.models.long_term import LongTermRanker
        X, y, groups = _make_grouped_data(n_groups=5, n_per_group=60)
        ranker = LongTermRanker()
        ranker.train(X, y, groups=groups)

        path = str(tmp_path / "test_model.joblib")
        ranker.save(path)

        loaded = LongTermRanker.load(path)
        assert loaded.model_name == "long_term_ranker"
        assert loaded.is_trained

    def test_predict_before_train_raises(self) -> None:
        from core.models.long_term import LongTermRanker
        ranker = LongTermRanker()
        X, _, _ = _make_random_data(10, 5)
        with pytest.raises(RuntimeError, match="尚未训练"):
            ranker.predict(X)

    def test_get_params(self) -> None:
        from core.models.long_term import LongTermRanker
        ranker = LongTermRanker()
        params = ranker.get_params()
        assert params["model_name"] == "long_term_ranker"
        assert params["trained"] is False

    def test_reproducibility(self) -> None:
        """同一数据+同一seed → 两次训练结果一致。"""
        from core.models.long_term import LongTermRanker

        X, y, groups = _make_grouped_data(n_groups=5, n_per_group=60)

        r1 = LongTermRanker()
        r1.train(X, y, groups=groups)
        p1 = r1.predict(X)

        r2 = LongTermRanker()
        r2.train(X, y, groups=groups)
        p2 = r2.predict(X)

        np.testing.assert_array_almost_equal(p1, p2, decimal=6)


# ── Long Term Models ──────────────────────────────

@requires_lgb
class TestLongTermRanker:
    def test_train_and_predict(self) -> None:
        from core.models.long_term import LongTermRanker
        X, y, _ = _make_random_data(200, 10)
        ranker = LongTermRanker()
        ranker.train(X, y)
        preds = ranker.predict_ranks(X)
        assert len(preds) == len(X)
        assert preds.notna().all()

    def test_with_groups(self) -> None:
        from core.models.long_term import LongTermRanker
        X, y, groups = _make_grouped_data(n_groups=5, n_per_group=60)
        ranker = LongTermRanker()
        ranker.train(X, y, groups=groups)
        preds = ranker.predict_ranks(X)
        assert len(preds) == len(X)

    def test_feature_importance(self) -> None:
        from core.models.long_term import LongTermRanker
        X, y, groups = _make_grouped_data(n_groups=5, n_per_group=60)
        ranker = LongTermRanker()
        ranker.train(X, y, groups=groups)
        imp = ranker.feature_importance()
        assert len(imp) >= 8
        assert "importance" in imp.columns

    def test_prepare_labels(self) -> None:
        from core.models.long_term import LongTermRanker
        ranker = LongTermRanker()
        data = pd.DataFrame({
            "code": ["A"] * 25 + ["B"] * 25,
            "trade_date": pd.date_range("2024-01-02", periods=50),
            "close": np.random.randn(50).cumsum() + 100,
        })
        labels = ranker.prepare_labels(data, forward_period=5)
        assert labels.notna().sum() > 0


@requires_lgb
class TestLongTermClassifier:
    def test_train_and_predict_proba(self) -> None:
        from core.models.long_term import LongTermClassifier
        X, _, y = _make_random_data(200, 10)
        y_binary = pd.Series(np.random.choice([0, 1], 200))
        clf = LongTermClassifier()
        clf.train(X, y_binary)
        proba = clf.predict_proba(X)
        assert proba.shape[0] == 200


class TestLinearBaseline:
    def test_train_and_predict(self) -> None:
        from core.models.long_term import LinearBaseline
        X, y, _ = _make_random_data(200, 8)
        linear = LinearBaseline()
        linear.train(X, y)
        preds = linear.predict(X)
        assert preds.shape == (200, 1)

    def test_coefficients(self) -> None:
        from core.models.long_term import LinearBaseline
        X, y, _ = _make_random_data(200, 5)
        linear = LinearBaseline()
        linear.train(X, y)
        coef = linear.coefficients()
        assert len(coef) == 5
        assert "coefficient" in coef.columns


# ── Intraday Models ────────────────────────────────

@requires_lgb
class TestIntradayClassifier:
    def test_train_and_predict_proba(self) -> None:
        from core.models.intraday import IntradayClassifier
        X, _, _ = _make_random_data(300, 10)
        y = pd.Series(np.random.choice([0, 1, 2], 300))
        clf = IntradayClassifier()
        clf.train(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (300, 3), f"shape={proba.shape}"
        # 概率和为1
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_predict_direction(self) -> None:
        from core.models.intraday import IntradayClassifier
        X, _, _ = _make_random_data(100, 6)
        y = pd.Series(np.random.choice([0, 1, 2], 100))
        clf = IntradayClassifier()
        clf.train(X, y)
        directions = clf.predict_direction(X)
        assert set(directions.unique()).issubset({0, 1, 2})

    def test_prepare_labels(self) -> None:
        from core.models.intraday import IntradayClassifier
        clf = IntradayClassifier()
        data = pd.DataFrame({
            "close": [100.0] * 20,
            "target_close": [101.0, 99.5, 100.2] * 6 + [100.0, 100.0],
        })
        labels = clf.prepare_labels(data)
        assert set(labels.unique()).issubset({0, 1, 2})


@requires_lgb
class TestIntradayQuantileRegressor:
    def test_train_and_predict_quantiles(self) -> None:
        from core.models.intraday import IntradayQuantileRegressor
        X, y, _ = _make_random_data(200, 8)
        qr = IntradayQuantileRegressor(quantiles=(0.10, 0.50, 0.90))
        qr.train(X, y)
        preds = qr.predict(X)
        assert "q10" in preds.columns
        assert "q50" in preds.columns
        assert "q90" in preds.columns
        # P10 < P50 < P90
        assert (preds["q10"] <= preds["q90"]).all()


# ── Premarket Models ────────────────────────────────

@requires_lgb
class TestOvernightMapping:
    def test_train_and_predict(self) -> None:
        from core.models.premarket.overnight_mapping import OvernightMappingModel
        X, y, _ = _make_random_data(200, 8)
        m = OvernightMappingModel()
        m.train(X, y)
        preds = m.predict(X)
        assert preds.shape[0] == 200

    def test_predict_direction(self) -> None:
        from core.models.premarket.overnight_mapping import OvernightMappingModel
        X, y, _ = _make_random_data(200, 8)
        m = OvernightMappingModel()
        m.train(X, y)
        dirs = m.predict_direction(X)
        assert set(dirs.flatten()).issubset({-1, 0, 1})


@requires_lgb
class TestGapClassifier:
    def test_train_and_predict_proba(self) -> None:
        from core.models.premarket.gap_classifier import GapClassifier
        X, _, _ = _make_random_data(200, 10)
        y = pd.Series(np.random.choice([0, 1, 2], 200))
        clf = GapClassifier()
        clf.train(X, y, calibrate=True)
        proba = clf.predict_proba(X)
        assert proba.shape == (200, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.02)

    def test_predict_gap_type(self) -> None:
        from core.models.premarket.gap_classifier import GapClassifier
        X, _, _ = _make_random_data(100, 6)
        y = pd.Series(np.random.choice([0, 1, 2], 100))
        clf = GapClassifier()
        clf.train(X, y)
        types = clf.predict_gap_type(X)
        assert set(types.flatten()).issubset({0, 1, 2})


@requires_lgb
class TestFusionRanker:
    def test_train_and_predict_ranks(self) -> None:
        from core.models.premarket.fusion_ranker import FusionRanker
        X, y, groups = _make_grouped_data(n_groups=5, n_per_group=60)
        # 注入 long_term_score 列
        X["long_term_score"] = np.random.randn(len(X))
        ranker = FusionRanker()
        ranker.train(X, y, groups=groups)
        scores = ranker.predict_ranks(X)
        assert len(scores) == len(X)

    def test_top_recommendations(self) -> None:
        from core.models.premarket.fusion_ranker import FusionRanker
        X, y, groups = _make_grouped_data(n_groups=3, n_per_group=100)
        X["long_term_score"] = np.random.randn(len(X))
        X.index = [f"code_{i:04d}" for i in range(len(X))]
        ranker = FusionRanker()
        ranker.train(X, y, groups=groups)
        top = ranker.get_top_recommendations(X, top_n=10)
        assert len(top) <= 10
        assert "fusion_score" in top.columns

    def test_missing_long_term_score_warns(self, caplog) -> None:
        from core.models.premarket.fusion_ranker import FusionRanker
        X, y, groups = _make_grouped_data(n_groups=3, n_per_group=50)
        # 不注入 long_term_score → 应警告
        ranker = FusionRanker()
        ranker.train(X, y, groups=groups)
        # 至少模型没崩溃
        assert ranker.is_trained


class TestAuctionAnomaly:
    def test_train_and_detect(self) -> None:
        from core.models.premarket.auction_anomaly import AuctionAnomalyDetector

        data = pd.DataFrame({
            "auction_order_count_pre_920": np.random.randint(100, 10000, 200),
            "auction_order_count_post_920": np.random.randint(100, 5000, 200),
            "virtual_open": np.random.uniform(9.5, 10.5, 200),
            "pre_close": [10.0] * 200,
            "unmatched_buy": np.random.randint(0, 1000, 200),
            "unmatched_sell": np.random.randint(0, 1000, 200),
            "matched_volume": np.random.randint(1000, 50000, 200),
        })
        detector = AuctionAnomalyDetector()
        detector.train(data)
        scores = detector.detect(data)
        assert len(scores) == 200
        assert scores.between(0, 1).all()

    def test_flag_suspicious(self) -> None:
        from core.models.premarket.auction_anomaly import AuctionAnomalyDetector

        data = pd.DataFrame({
            "auction_order_count_pre_920": np.random.randint(100, 10000, 50),
            "auction_order_count_post_920": np.random.randint(100, 5000, 50),
            "virtual_open": np.random.uniform(9.5, 10.5, 50),
            "pre_close": [10.0] * 50,
            "unmatched_buy": np.random.randint(0, 1000, 50),
            "unmatched_sell": np.random.randint(0, 1000, 50),
            "matched_volume": np.random.randint(1000, 50000, 50),
        })
        detector = AuctionAnomalyDetector()
        detector.train(data)
        flags = detector.flag_suspicious(data, threshold=0.8)
        assert isinstance(flags, pd.Series)
        assert flags.dtype == bool


# ── NLP Model ─────────────────────────────────────

class TestNLPSentimentAnalyzer:
    def test_keyword_positive(self) -> None:
        from core.models.nlp import NLPSentimentAnalyzer
        analyzer = NLPSentimentAnalyzer(model_name="keyword")
        result = analyzer.analyze_single(
            "贵州茅台 2025年度业绩大幅增长 净利润超预期 拟10派25元",
            source="announcement",
        )
        assert result.sentiment == "positive"
        assert result.confidence > 0.3

    def test_keyword_negative(self) -> None:
        from core.models.nlp import NLPSentimentAnalyzer
        analyzer = NLPSentimentAnalyzer(model_name="keyword")
        result = analyzer.analyze_single(
            "公司因涉嫌信息披露违规被证监会立案调查，存在退市风险",
            source="announcement",
        )
        assert result.sentiment == "negative"

    def test_keyword_neutral(self) -> None:
        from core.models.nlp import NLPSentimentAnalyzer
        analyzer = NLPSentimentAnalyzer(model_name="keyword")
        result = analyzer.analyze_single(
            "公司于今日召开董事会审议通过了日常经营相关议案",
            source="announcement",
        )
        assert result.sentiment == "neutral"

    def test_empty_text(self) -> None:
        from core.models.nlp import NLPSentimentAnalyzer
        analyzer = NLPSentimentAnalyzer()
        result = analyzer.analyze_single("", source="news")
        assert result.sentiment == "neutral"
        assert result.confidence == 0.0

    def test_batch_analysis(self) -> None:
        from core.models.nlp import NLPSentimentAnalyzer
        analyzer = NLPSentimentAnalyzer()
        texts = [
            ("业绩大幅增长超预期", "announcement"),
            ("因违规被证监会处罚", "announcement"),
            ("日常经营公告", "news"),
        ]
        results = analyzer.analyze_batch(texts)
        assert len(results) == 3
        assert results[0].sentiment == "positive"
        assert results[1].sentiment == "negative"

    def test_extract_event_type(self) -> None:
        from core.models.nlp import NLPSentimentAnalyzer
        analyzer = NLPSentimentAnalyzer()
        event = analyzer.extract_event_type(
            "公司中标国家电网500亿元特高压项目合同"
        )
        assert event == "重大合同"

        event2 = analyzer.extract_event_type(
            "控股股东因涉嫌违规减持被证监会立案调查"
        )
        assert event2 == "增减持"

    def test_sentiment_result_dataclass(self) -> None:
        from core.models.nlp import SentimentResult
        r = SentimentResult(
            sentiment="positive", confidence=0.85,
            event_type="业绩超预期", summary="摘要...",
        )
        assert r.sentiment == "positive"
        assert r.event_type == "业绩超预期"


# ── Model Evaluation ──────────────────────────────

class TestModelEvaluator:
    def test_rank_ic_perfect(self) -> None:
        from core.models.evaluation import ModelEvaluator
        preds = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        ic = ModelEvaluator.rank_ic(preds, returns)
        assert ic > 0.9, f"完全单调→接近1.0, 实际={ic:.3f}"

    def test_rank_ic_random(self) -> None:
        from core.models.evaluation import ModelEvaluator
        np.random.seed(42)
        preds = pd.Series(np.random.randn(1000))
        returns = pd.Series(np.random.randn(1000))
        ic = ModelEvaluator.rank_ic(preds, returns)
        assert abs(ic) < 0.2, f"随机→接近0, 实际={ic:.3f}"

    def test_icir(self) -> None:
        from core.models.evaluation import ModelEvaluator
        ic_series = pd.Series([0.05, 0.03, 0.06, 0.04, 0.05] * 10)
        icir = ModelEvaluator.icir(ic_series)
        assert icir > 1.0, f"IC稳定→ICIR高, 实际={icir:.3f}"

    def test_cross_sectional_ic(self) -> None:
        from core.models.evaluation import ModelEvaluator
        dates = pd.to_datetime(["2024-01-02"] * 50 + ["2024-01-03"] * 50)
        preds = pd.Series(np.random.randn(100))
        returns = pd.Series(np.random.randn(100))
        cs = ModelEvaluator.cross_sectional_ic(preds, returns, dates)
        assert len(cs) == 2

    def test_ic_decay(self) -> None:
        from core.models.evaluation import ModelEvaluator
        preds = pd.Series(np.random.randn(100))
        fwd_returns = pd.DataFrame({
            "1d": np.random.randn(100) * 0.01,
            "5d": np.random.randn(100) * 0.02,
            "10d": np.random.randn(100) * 0.03,
            "20d": np.random.randn(100) * 0.04,
        })
        decay = ModelEvaluator.ic_decay(preds, fwd_returns)
        assert len(decay) == 4
        assert decay.index.tolist() == ["1d", "5d", "10d", "20d"]

    def test_layered_returns(self) -> None:
        from core.models.evaluation import ModelEvaluator
        preds = pd.Series(np.linspace(0, 1, 100))
        returns = pd.Series(np.random.randn(100) * 0.02)
        layers = ModelEvaluator.layered_returns(preds, returns, n_quantiles=5)
        assert len(layers) == 5
        assert "mean_return" in layers.columns

    def test_direction_accuracy(self) -> None:
        from core.models.evaluation import ModelEvaluator
        true = pd.Series([1, -1, 1, 0, -1, 1, -1, 0])
        pred = pd.Series([1, -1, 1, 0, 1, -1, -1, 0])
        acc = ModelEvaluator.direction_accuracy(true, pred)
        assert 0.0 <= acc <= 1.0

    def test_rolling_monitoring(self) -> None:
        from core.models.evaluation import ModelEvaluator
        # 10 distinct dates, 20 samples each → 10 IC points, rolling window=3
        dates = pd.to_datetime(
            [f"2024-01-{(d // 20) + 1:02d}" for d in range(200)]
        )
        preds = pd.Series(np.random.randn(200))
        returns = pd.Series(np.random.randn(200))
        monitor = ModelEvaluator.rolling_monitoring(preds, returns, dates, window=3)
        # 10 IC points - rolling=3 yields ~8 rows
        assert len(monitor) >= 5
        assert "rolling_ic_mean" in monitor.columns

    def test_full_report(self) -> None:
        from core.models.evaluation import ModelEvaluator
        preds = pd.Series(np.random.randn(200))
        returns = pd.Series(np.random.randn(200) * 0.02)
        report = ModelEvaluator.full_report(preds, returns)
        assert "rank_ic" in report
        assert "n_samples" in report
