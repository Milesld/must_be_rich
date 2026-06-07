"""pytest fixtures — 测试数据库、模拟数据源等共用组件。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
import pytest

from core.data.sources.base import DataSourceBase


class MockDataSource(DataSourceBase):
    """模拟数据源，返回固定格式的空 DataFrame 用于单元测试。

    可注入自定义返回值来模拟成功/失败/异常场景。
    """

    def __init__(self, name: str = "mock") -> None:
        self._name = name
        self.daily_kline_result: Optional[pd.DataFrame] = None
        self.minute_kline_result: Optional[pd.DataFrame] = None
        self.financials_result: Optional[pd.DataFrame] = None
        self.margin_trading_result: Optional[pd.DataFrame] = None
        self.dragon_tiger_result: Optional[pd.DataFrame] = None
        self._raise_on_call: Optional[str] = None  # 方法名，模拟异常

    @property
    def source_name(self) -> str:
        return self._name

    def get_daily_kline(self, start, end, codes=None):
        if self._raise_on_call == "get_daily_kline":
            raise RuntimeError("模拟异常")
        if self.daily_kline_result is not None:
            return self.daily_kline_result
        return pd.DataFrame(columns=self.DAILY_KL_COLUMNS)

    def get_minute_kline(self, trade_date, codes=None):
        if self._raise_on_call == "get_minute_kline":
            raise RuntimeError("模拟异常")
        if self.minute_kline_result is not None:
            return self.minute_kline_result
        return pd.DataFrame()

    def get_financials(self, codes=None, report_dates=None):
        if self._raise_on_call == "get_financials":
            raise RuntimeError("模拟异常")
        if self.financials_result is not None:
            return self.financials_result
        return pd.DataFrame()

    def get_margin_trading(self, trade_date):
        if self._raise_on_call == "get_margin_trading":
            raise RuntimeError("模拟异常")
        if self.margin_trading_result is not None:
            return self.margin_trading_result
        return pd.DataFrame()

    def get_dragon_tiger(self, trade_date):
        if self._raise_on_call == "get_dragon_tiger":
            raise RuntimeError("模拟异常")
        if self.dragon_tiger_result is not None:
            return self.dragon_tiger_result
        return pd.DataFrame()

    def get_auction_snapshot(self, trade_date):
        raise NotImplementedError("mock 不支持集合竞价快照")


@pytest.fixture
def mock_ds() -> MockDataSource:
    """返回 MockDataSource 实例。"""
    return MockDataSource()


@pytest.fixture
def sample_daily_kline_df() -> pd.DataFrame:
    """生成合法的日K线样本数据，用于质量检查测试。"""
    today = date.today() - timedelta(days=1)
    return pd.DataFrame({
        "trade_date": [today] * 5,
        "code": ["600519", "000001", "300750", "688981", "920001"],
        "name": ["茅台", "平安银行", "宁德时代", "中芯国际", "北证测试"],
        "open": [1680.0, 12.0, 210.0, 55.0, 20.0],
        "high": [1700.0, 12.5, 215.0, 56.0, 21.0],
        "low": [1670.0, 11.8, 208.0, 54.0, 19.5],
        "close": [1695.0, 12.3, 213.0, 55.5, 20.8],
        "pre_close": [1685.0, 12.0, 212.0, 54.8, 20.0],
        "volume": [5_000_000, 80_000_000, 20_000_000, 10_000_000, 1_000_000],
        "amount": [8.5e9, 9.8e8, 4.2e9, 5.5e8, 2.0e7],
        "adj_factor": [1.0, 1.0, 1.0, 1.0, 1.0],
        "turnover": [0.5, 2.0, 1.5, 3.0, 1.0],
        "is_st": [0, 0, 0, 0, 0],
        "is_suspended": [0, 0, 0, 0, 0],
        "status": ["正常"] * 5,
    })


@pytest.fixture
def sample_invalid_kline_df() -> pd.DataFrame:
    """生成含数据质量问题的日K线样本。"""
    today = date.today() - timedelta(days=1)
    return pd.DataFrame({
        "trade_date": [today] * 3,
        "code": ["600001", "600002", "600003"],
        "name": ["问题股1", "问题股2", "问题股3"],
        "open": [10.0, -5.0, 8.0],          # 600002: open 为负
        "high": [12.0, 6.0, 6.0],           # 600003: high < low
        "low": [9.0, 4.0, 7.0],             # 600003: low > high
        "close": [11.0, 5.0, 7.5],
        "pre_close": [10.0, 5.0, 8.0],
        "volume": [1_000_000, 500_000, 0],
        "amount": [1.1e7, 2.5e6, 0],
        "adj_factor": [1.0, 1.0, 1.0],
        "turnover": [1.0, 0.5, 0.0],
        "is_st": [0, 0, 0],
        "is_suspended": [0, 0, 1],
        "status": ["正常", "正常", "停牌"],
    })


@pytest.fixture
def sample_financials_df() -> pd.DataFrame:
    """生成合法的财务报表样本数据。"""
    return pd.DataFrame({
        "code": ["600519", "000001", "300750"],
        "report_date": [date(2025, 12, 31)] * 3,
        "announce_date": [date(2026, 3, 15), date(2026, 3, 20), date(2026, 4, 10)],
        "statement_type": ["年报"] * 3,
        "revenue": [1.5e11, 5.0e10, 3.0e10],
        "revenue_yoy": [15.0, 8.0, 25.0],
        "net_profit": [7.5e10, 3.0e10, 4.0e9],
        "net_profit_yoy": [12.0, 6.0, 30.0],
        "gross_margin": [0.92, 0.45, 0.30],
        "net_margin": [0.50, 0.60, 0.13],
        "total_assets": [3.0e11, 5.0e12, 1.0e11],
        "total_liabilities": [5.0e10, 4.6e12, 6.0e10],
        "total_equity": [2.5e11, 4.0e11, 4.0e10],
        "debt_ratio": [0.17, 0.92, 0.60],
        "current_ratio": [3.5, 1.0, 1.5],
        "quick_ratio": [3.0, 0.8, 1.2],
        "operating_cf": [7.0e10, -5.0e10, 5.0e9],
        "free_cf": [5.0e10, -8.0e10, 2.0e9],
        "cf_ratio": [0.93, -1.67, 1.25],
        "eps": [59.0, 1.5, 8.5],
        "bps": [198.0, 20.0, 85.0],
        "roe": [0.30, 0.075, 0.10],
        "roa": [0.25, 0.006, 0.04],
        "roic": [0.28, 0.01, 0.06],
        "data_source": ["mock"] * 3,
        "version": ["1.0"] * 3,
    })
