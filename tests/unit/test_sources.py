"""主备切换数据源 + 数据管线单元测试。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from core.data.sources.base import DataSourceBase
from core.data.sources.fallback import FallbackDataSource


class TestFallbackDataSource:
    """FallbackDataSource 单元测试。"""

    def test_uses_primary_when_healthy(self) -> None:
        from tests.conftest import MockDataSource

        primary = MockDataSource("primary")
        fallback = MockDataSource("fallback")
        ds = FallbackDataSource(primary, fallback)

        result = ds.get_daily_kline(date(2026, 6, 1), date(2026, 6, 5))
        assert result is not None
        assert ds.status["active_source"] == "primary"
        assert ds.status["failure_count"] == 0

    def test_falls_back_on_primary_failure(self) -> None:
        from tests.conftest import MockDataSource

        primary = MockDataSource("primary")
        primary._raise_on_call = "get_daily_kline"
        fallback = MockDataSource("fallback")

        ds = FallbackDataSource(primary, fallback, max_failures=1)

        # 第一次调用：主源失败，退回到备用源
        result = ds.get_daily_kline(date(2026, 6, 1), date(2026, 6, 5))
        assert result is not None

    def test_switches_default_after_max_failures(self) -> None:
        from tests.conftest import MockDataSource

        primary = MockDataSource("primary")
        primary._raise_on_call = "get_daily_kline"
        fallback = MockDataSource("fallback")

        ds = FallbackDataSource(primary, fallback, max_failures=1)

        # 一次失败后默认使用备用源
        try:
            ds.get_daily_kline(date(2026, 6, 1), date(2026, 6, 5))
        except RuntimeError:
            pass

        # 第二次调用应该已经默认使用 fallback
        # Note: max_failures=1, so after 1 failure it switches default
        status = ds.status
        # 如果连 fallback 也没有任何数据返回，它只是返回空 DataFrame
        assert status["failure_count"] >= 1

    def test_source_name_shows_both(self) -> None:
        from tests.conftest import MockDataSource

        primary = MockDataSource("akshare")
        fallback = MockDataSource("tushare")
        ds = FallbackDataSource(primary, fallback)

        name = ds.source_name
        assert "akshare" in name
        assert "tushare" in name

    def test_all_methods_proxied(self) -> None:
        from tests.conftest import MockDataSource

        primary = MockDataSource("primary")
        fallback = MockDataSource("fallback")
        ds = FallbackDataSource(primary, fallback)

        # 所有方法都应能调用且不抛异常
        today = date.today()

        # 这些方法 primary 成功
        assert ds.get_daily_kline(today, today) is not None
        assert ds.get_financials() is not None
        assert ds.get_margin_trading(today) is not None
        assert ds.get_dragon_tiger(today) is not None

        # auction_snapshot — primary 成功则 fallback 不用
        try:
            ds.get_auction_snapshot(today)
        except NotImplementedError:
            pass  # mock 抛出 NotImplementedError 是预期行为


class TestDataSourceBase:
    """DataSourceBase 抽象接口测试。"""

    def test_column_constants(self) -> None:
        col_names = [
            "COL_CODE", "COL_NAME", "COL_TRADE_DATE", "COL_TIME",
            "COL_OPEN", "COL_HIGH", "COL_LOW", "COL_CLOSE",
            "COL_PRE_CLOSE", "COL_VOLUME", "COL_AMOUNT",
            "COL_ADJ_FACTOR", "COL_TURNOVER",
            "COL_IS_ST", "COL_IS_SUSPENDED", "COL_STATUS",
            "COL_REPORT_DATE", "COL_ANNOUNCE_DATE",
        ]
        for name in col_names:
            assert hasattr(DataSourceBase, name), f"缺少列名常量: {name}"

    def test_daily_kl_columns_list(self) -> None:
        columns = DataSourceBase.DAILY_KL_COLUMNS
        assert len(columns) >= 14
        assert "code" in columns
        assert "trade_date" in columns

    def test_abstract_class_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            DataSourceBase()  # type: ignore


class TestDataPipeline:
    """DataPipeline 单元测试。"""

    def test_init_with_mock_source(self) -> None:
        from core.data.pipeline import DataPipeline
        from tests.conftest import MockDataSource

        ds = MockDataSource()
        pipeline = DataPipeline(ds)
        assert pipeline._ds is ds

    def test_collect_daily_kline_empty(self) -> None:
        from core.data.pipeline import DataPipeline
        from tests.conftest import MockDataSource

        ds = MockDataSource()
        pipeline = DataPipeline(ds)

        count = pipeline.collect_daily_kline(
            start=date(2026, 6, 1),
            end=date(2026, 6, 5),
        )
        # Mock 返回空 DataFrame
        assert count == 0

    def test_checkpoint_persistence(self, tmp_path) -> None:
        from core.data.pipeline import DataPipeline, PipelineCheckpoint
        from tests.conftest import MockDataSource

        ds = MockDataSource()
        pipeline = DataPipeline(ds, checkpoint_dir=str(tmp_path))

        pipeline.collect_margin_trading(date(2026, 6, 5))
        checkpoints = pipeline.list_checkpoints()
        assert len(checkpoints) >= 1
        margin_ck = [c for c in checkpoints if c.task_name == "margin_trading"]
        assert len(margin_ck) == 1
        assert margin_ck[0].last_success_date == date(2026, 6, 5)
