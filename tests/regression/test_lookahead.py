"""前视偏差回归测试 — 构造含未来数据的场景验证 TimeTravelChecker 捕获能力。"""

import pytest
from datetime import date

# TimeTravelChecker 经 core.features.__init__ 间接依赖 sklearn；缺失时整体跳过。
pytest.importorskip("sklearn", reason="sklearn 未安装（core.features.neutralizer 依赖）")


def test_catches_future_announce_date() -> None:
    """财报 announce_date 晚于回测日期 → 应被检查器捕获。"""
    from core.features.time_travel_checker import TimeTravelChecker, TimeTravelError

    checker = TimeTravelChecker(strict_mode=True)
    as_of = date(2020, 2, 10)

    with pytest.raises(TimeTravelError) as exc:
        checker.check(as_of, {
            "financial_statements": {
                "max_announce_date": date(2020, 2, 28),  # 2月28日才披露 → 违规！
                "codes_with_violations": ["600519", "000001"],
            },
        })

    assert "announce_date" in str(exc.value)
    assert "2020-02-28" in str(exc.value)


def test_catches_future_market_data() -> None:
    """行情数据日期晚于回测日期 → 应被捕获。"""
    from core.features.time_travel_checker import TimeTravelChecker, TimeTravelError

    checker = TimeTravelChecker(strict_mode=True)
    as_of = date(2019, 6, 15)

    with pytest.raises(TimeTravelError):
        checker.check(as_of, {
            "market_daily": {"max_trade_date": date(2019, 6, 20)},
        })


def test_passes_when_all_data_before_as_of() -> None:
    """全部合法数据应通过。"""
    from core.features.time_travel_checker import TimeTravelChecker

    checker = TimeTravelChecker(strict_mode=True)
    assert checker.check(date(2020, 3, 10), {
        "market_daily": {"max_trade_date": date(2020, 3, 10)},
        "financial_statements": {"max_announce_date": date(2020, 3, 5)},
        "factor_values": {"max_calc_date": date(2020, 3, 9)},
    })


def test_pit_filter_excludes_future() -> None:
    """PIT模式应过滤尚未披露的财报。"""
    import pandas as pd
    from core.features.fundamental import _pit_filter

    data = pd.DataFrame({
        "trade_date": pd.to_datetime([date(2020, 2, 10)] * 3),
        "code": ["600519", "000001", "300750"],
    })
    fin = pd.DataFrame({
        "code": ["600519", "000001", "300750"],
        "announce_date": [
            date(2020, 2, 5),   # 已披露 ✓
            date(2020, 3, 15),  # 未披露 ✗
            date(2020, 1, 20),  # 已披露 ✓
        ],
        "report_date": [date(2019, 12, 31)] * 3,
    })
    filtered = _pit_filter(data, fin, point_in_time=True)
    # 000001 (announce_date=2020-03-15) → 应被过滤
    assert "000001" not in filtered["code"].values
    assert len(filtered) == 2


def test_timetravel_config_loaded() -> None:
    """验证 TimeTravelChecker 可在默认模式下使用。"""
    from core.features.time_travel_checker import TimeTravelChecker, validate_pit_financials

    # 非严格模式：只告警不抛异常
    checker = TimeTravelChecker(strict_mode=False)
    result = checker.check(date(2020, 5, 1), {
        "market_daily": {"max_trade_date": date(2020, 5, 10)},  # 未来
    })
    assert result is False  # 未通过但不抛异常

    # PIT快速验证
    violations = validate_pit_financials(
        as_of_date=date(2020, 5, 1),
        announce_dates=[date(2020, 5, 15), date(2020, 4, 20)],
        codes=["A", "B"],
    )
    assert len(violations) == 1
    assert "A" in violations[0]
