"""交易日历缓存投毒防护回归测试（路线图第 2 阶段）。

场景：离线运行时系统落一份「估算日历」缓存（周一至五全算交易日，
含节假日误差）。修复前该缓存会被后续所有运行当真实日历直接采用；
修复后估算缓存带 estimated 标记，下次初始化优先联网替换。
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import core.common.calendar as cal_mod
from core.common.calendar import TradingCalendar


@pytest.fixture()
def tmp_cache(tmp_path, monkeypatch):
    """把日历缓存重定向到临时目录，避免污染真实缓存。"""
    cache_dir = tmp_path / "calendar"
    cache_file = cache_dir / "trading_calendar.parquet"
    monkeypatch.setattr(cal_mod, "_CACHE_DIR", cache_dir)
    monkeypatch.setattr(cal_mod, "_CACHE_FILE", cache_file)
    return cache_file


def _block_network(monkeypatch):
    """让 akshare 拉取必然失败，模拟离线。"""
    def _fail(self):
        raise RuntimeError("simulated offline")
    monkeypatch.setattr(TradingCalendar, "_fetch_from_akshare", _fail)


def _fake_real_calendar(monkeypatch, days: set[date]):
    """让 akshare 拉取返回指定真实日历。"""
    def _ok(self):
        self._trading_days = set(days)
        self._trading_days_sorted = sorted(days)
    monkeypatch.setattr(TradingCalendar, "_fetch_from_akshare", _ok)


class TestEstimatedCacheMarking:
    def test_offline_saves_estimated_flag(self, tmp_cache, monkeypatch):
        """离线兜底生成的缓存必须带 estimated=True 标记。"""
        _block_network(monkeypatch)
        TradingCalendar()
        assert tmp_cache.exists()
        df = pd.read_parquet(tmp_cache)
        assert "estimated" in df.columns
        assert bool(df["estimated"].iloc[0]) is True

    def test_online_saves_real_flag(self, tmp_cache, monkeypatch):
        """联网拉取的缓存 estimated=False。"""
        real_days = {date(2026, 7, 13), date(2026, 7, 14), date(2026, 7, 15)}
        _fake_real_calendar(monkeypatch, real_days)
        TradingCalendar()
        df = pd.read_parquet(tmp_cache)
        assert bool(df["estimated"].iloc[0]) is False


class TestEstimatedCacheReplacement:
    def test_estimated_cache_replaced_when_online(self, tmp_cache, monkeypatch):
        """核心回归：估算缓存存在时，下次联网初始化必须替换为真实日历。

        修复前行为：缓存存在 → 直接采用，估算日历永久生效（投毒）。
        """
        # 第一步：离线运行落估算缓存
        _block_network(monkeypatch)
        cal_offline = TradingCalendar()
        # 估算日历把春节也当交易日（2024-02-12 是春节假期的周一）
        assert cal_offline.is_trading_day(date(2024, 2, 12))

        # 第二步：恢复联网，重新初始化
        real_days = {date(2024, 2, 8), date(2024, 2, 19), date(2024, 2, 20)}
        _fake_real_calendar(monkeypatch, real_days)
        cal_online = TradingCalendar()

        # 真实日历生效：春节不再是交易日，缓存已更新为 estimated=False
        assert not cal_online.is_trading_day(date(2024, 2, 12))
        assert cal_online.is_trading_day(date(2024, 2, 19))
        df = pd.read_parquet(tmp_cache)
        assert bool(df["estimated"].iloc[0]) is False

    def test_estimated_cache_still_usable_offline(self, tmp_cache, monkeypatch):
        """持续离线时估算缓存仍可用（有误差的日历 > 没有日历）。"""
        _block_network(monkeypatch)
        TradingCalendar()  # 落估算缓存
        cal2 = TradingCalendar()  # 仍离线，应回退到估算缓存而非报错
        assert len(cal2.all_trading_days) > 0

    def test_real_cache_short_circuits_network(self, tmp_cache, monkeypatch):
        """真实日历缓存直接采用，不再联网（保持原有快路径）。"""
        real_days = {date(2026, 7, 13), date(2026, 7, 14)}
        _fake_real_calendar(monkeypatch, real_days)
        TradingCalendar()  # 落真实缓存

        calls = []
        def _count(self):
            calls.append(1)
            raise AssertionError("should not fetch")
        monkeypatch.setattr(TradingCalendar, "_fetch_from_akshare", _count)
        cal2 = TradingCalendar()
        assert not calls
        assert cal2.is_trading_day(date(2026, 7, 14))


class TestLegacyCacheHeuristic:
    def test_legacy_cache_without_flag_detected_by_spring_festival(
        self, tmp_cache, monkeypatch
    ):
        """旧版缓存（无 estimated 列）含春节交易日 → 判定为估算日历。"""
        # 手工构造旧格式估算缓存：含 2024-02-12（春节假期）
        tmp_cache.parent.mkdir(parents=True, exist_ok=True)
        days = [date(2024, 2, 8), date(2024, 2, 12), date(2024, 2, 19)]
        pd.DataFrame({"trade_date": pd.to_datetime(days)}).to_parquet(
            tmp_cache, index=False
        )
        assert TradingCalendar._cache_is_estimated() is True

        # 联网初始化应替换它
        real_days = {date(2024, 2, 8), date(2024, 2, 19)}
        _fake_real_calendar(monkeypatch, real_days)
        cal = TradingCalendar()
        assert not cal.is_trading_day(date(2024, 2, 12))

    def test_legacy_real_cache_not_flagged(self, tmp_cache):
        """旧版真实日历缓存（不含春节）不被误判为估算。"""
        tmp_cache.parent.mkdir(parents=True, exist_ok=True)
        days = [date(2024, 2, 8), date(2024, 2, 19), date(2024, 2, 20)]
        pd.DataFrame({"trade_date": pd.to_datetime(days)}).to_parquet(
            tmp_cache, index=False
        )
        assert TradingCalendar._cache_is_estimated() is False
