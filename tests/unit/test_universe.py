"""动态股票池 PIT 筛选单测（离线，用构造数据，不依赖网络）。"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.universe import filter_universe_pit


def _make_raw_data(specs: dict[str, dict], start: date, n_days: int) -> dict:
    """构造 {date: {code: {close, amount, is_st}}}。

    specs: {code: {"start_offset": 上市偏移天数, "amount": 日成交额, "is_st": bool}}
    """
    raw: dict[date, dict] = {}
    for i in range(n_days):
        d = start + timedelta(days=i)
        day: dict[str, dict] = {}
        for code, spec in specs.items():
            if i < spec.get("start_offset", 0):
                continue  # 尚未上市
            day[code] = {
                "close": 10.0,
                "amount": spec.get("amount", 3e8),
                "is_st": spec.get("is_st", False),
            }
        raw[d] = day
    return raw


def test_filter_excludes_recent_ipo():
    """次新股（上市不足门槛）应被剔除。"""
    start = date(2023, 1, 1)
    # 老股从第0天上市，次新股从第 290 天才上市（< 12月*21≈252交易日门槛）
    raw = _make_raw_data(
        {"000001": {"start_offset": 0}, "000002": {"start_offset": 290}},
        start, 320,
    )
    as_of = start + timedelta(days=315)
    result = filter_universe_pit(
        ["000001", "000002"], as_of, raw,
        min_listing_months=12, min_avg_amount=1e8,
    )
    assert "000001" in result
    assert "000002" not in result  # 次新被剔除


def test_filter_excludes_low_liquidity():
    """成交额低于门槛的股票应被剔除。"""
    start = date(2023, 1, 1)
    raw = _make_raw_data(
        {"000001": {"amount": 5e8}, "000003": {"amount": 1e7}},
        start, 300,
    )
    as_of = start + timedelta(days=295)
    result = filter_universe_pit(
        ["000001", "000003"], as_of, raw,
        min_listing_months=1, min_avg_amount=2e8,
    )
    assert "000001" in result
    assert "000003" not in result  # 流动性不足被剔除


def test_filter_excludes_st():
    """ST 股应被剔除。"""
    start = date(2023, 1, 1)
    raw = _make_raw_data(
        {"000001": {"is_st": False}, "000004": {"is_st": True}},
        start, 300,
    )
    as_of = start + timedelta(days=295)
    result = filter_universe_pit(
        ["000001", "000004"], as_of, raw,
        min_listing_months=1, min_avg_amount=1e8, exclude_st=True,
    )
    assert "000001" in result
    assert "000004" not in result


def test_pool_size_cap():
    """pool_size 限制宇宙容量，按成交额降序取 top。"""
    start = date(2023, 1, 1)
    raw = _make_raw_data(
        {
            "000001": {"amount": 9e8},
            "000002": {"amount": 5e8},
            "000003": {"amount": 3e8},
        },
        start, 300,
    )
    as_of = start + timedelta(days=295)
    result = filter_universe_pit(
        ["000001", "000002", "000003"], as_of, raw,
        min_listing_months=1, min_avg_amount=1e8, pool_size=2,
    )
    assert result == ["000001", "000002"]  # 取成交额最高的2只


def test_universe_changes_over_time():
    """同一候选集在不同 as_of 日宇宙不同（次新随时间变可投）。"""
    start = date(2023, 1, 1)
    raw = _make_raw_data(
        {"000001": {"start_offset": 0}, "000002": {"start_offset": 100}},
        start, 400,
    )
    # 早期：次新还不满上市门槛
    early = filter_universe_pit(
        ["000001", "000002"], start + timedelta(days=150), raw,
        min_listing_months=4, min_avg_amount=1e8,  # 4月≈84交易日
    )
    # 晚期：次新已满门槛
    late = filter_universe_pit(
        ["000001", "000002"], start + timedelta(days=395), raw,
        min_listing_months=4, min_avg_amount=1e8,
    )
    assert "000002" not in early
    assert "000002" in late


def test_no_future_data_used():
    """PIT：只用 as_of_date 之前的数据，未来交易不影响筛选。"""
    start = date(2023, 1, 1)
    raw = _make_raw_data({"000001": {"amount": 5e8}}, start, 300)
    as_of = start + timedelta(days=100)
    # as_of 之后的日期数据存在，但筛选只看之前
    result = filter_universe_pit(
        ["000001"], as_of, raw,
        min_listing_months=1, min_avg_amount=2e8,
    )
    assert "000001" in result


def test_real_listing_date_excludes_recent_ipo():
    """给定真实上市日时，按上市日判断次新（不再依赖数据天数）。"""
    start = date(2024, 1, 1)
    # 数据只有 80 天（< 252），但真实上市日很早 → 不应被当次新剔除
    raw = _make_raw_data({"000001": {"amount": 5e8}}, start, 80)
    as_of = start + timedelta(days=75)
    listing = {"000001": date(2015, 1, 1)}  # 上市 9 年
    result = filter_universe_pit(
        ["000001"], as_of, raw,
        min_listing_months=12, min_avg_amount=1e8,
        listing_dates=listing,
    )
    assert "000001" in result  # 真实上市日早，数据天数少也不剔除


def test_real_listing_date_filters_genuine_recent_ipo():
    """真实上市日不足 12 个月 → 剔除。"""
    start = date(2024, 1, 1)
    raw = _make_raw_data({"000002": {"amount": 5e8}}, start, 200)
    as_of = date(2024, 6, 1)
    listing = {"000002": date(2024, 3, 1)}  # 上市仅 3 个月
    result = filter_universe_pit(
        ["000002"], as_of, raw,
        min_listing_months=12, min_avg_amount=1e8,
        listing_dates=listing,
    )
    assert "000002" not in result
