"""TimeTravelChecker 测试 — ★系统最重要的安全测试★。

构造各种"含前视偏差的数据"来验证检查器能正确捕获违规。
"""

from __future__ import annotations

from datetime import date

import pytest

from core.features.time_travel_checker import (
    TimeTravelChecker,
    TimeTravelError,
    TimeTravelViolation,
    validate_pit_financials,
)


class TestTimeTravelChecker:
    """TimeTravelChecker 核心检查逻辑。"""

    def test_pass_when_all_data_before_as_of(self) -> None:
        checker = TimeTravelChecker()
        as_of = date(2020, 2, 10)
        result = checker.check(as_of, {
            "market_daily": {"max_trade_date": date(2020, 2, 10)},
            "financial_statements": {"max_announce_date": date(2020, 2, 5)},
            "factor_values": {"max_calc_date": date(2020, 2, 9)},
        })
        assert result is True

    def test_detect_market_data_future_leak(self) -> None:
        """行情数据日期超过了回测日期 → 违规"""
        checker = TimeTravelChecker()
        as_of = date(2020, 2, 10)
        with pytest.raises(TimeTravelError) as exc:
            checker.check(as_of, {
                "market_daily": {"max_trade_date": date(2020, 2, 15)},  # 未来！
            })
        assert len(exc.value.violations) >= 1
        v = exc.value.violations[0]
        assert v.data_source == "market_daily"
        assert v.as_of_date == as_of

    def test_detect_financial_announce_date_leak(self) -> None:
        """财报的 announce_date 超过了回测日期 → 这是最经典的前视偏差"""
        checker = TimeTravelChecker()
        as_of = date(2020, 2, 10)
        # 模拟：今天的回测用到了2月15日才披露的财报
        with pytest.raises(TimeTravelError) as exc:
            checker.check(as_of, {
                "financial_statements": {
                    "max_announce_date": date(2020, 2, 15),  # ★违规：尚未披露
                    "codes_with_violations": ["600519", "000001"],
                },
            })
        assert len(exc.value.violations) >= 1
        assert "600519" in str(exc.value)

    def test_non_strict_mode_logs_but_passes(self) -> None:
        """非严格模式：违规时记录告警但不抛异常。"""
        checker = TimeTravelChecker(strict_mode=False)
        as_of = date(2020, 2, 10)
        # 不应该抛异常
        result = checker.check(as_of, {
            "market_daily": {"max_trade_date": date(2020, 2, 15)},
        })
        assert result is False  # 未通过，但不抛异常
        assert checker.stats["total_violations"] > 0

    def test_stats_accumulation(self) -> None:
        checker = TimeTravelChecker(strict_mode=False)
        as_of = date(2020, 2, 10)

        # 第一次检查：违规
        checker.check(as_of, {"market_daily": {"max_trade_date": date(2020, 2, 15)}})
        # 第二次检查：通过
        checker.check(as_of, {"market_daily": {"max_trade_date": date(2020, 2, 10)}})

        assert checker.stats["total_checks"] == 2
        assert checker.stats["total_violations"] == 1

        checker.reset_stats()
        assert checker.stats["total_checks"] == 0
        assert checker.stats["total_violations"] == 0

    def test_list_of_dates_detection(self) -> None:
        """列表中的单个未来日期应被检测到。"""
        checker = TimeTravelChecker()
        as_of = date(2020, 2, 10)
        with pytest.raises(TimeTravelError) as exc:
            checker.check(as_of, {
                "factor_values": {
                    "calc_dates": [
                        date(2020, 2, 8),
                        date(2020, 2, 9),
                        date(2020, 2, 12),  # ← 这条是未来的
                    ],
                },
            })
        assert len(exc.value.violations) >= 1

    def test_max_date_field_generic(self) -> None:
        """通用的 max_date 字段也支持。"""
        checker = TimeTravelChecker()
        as_of = date(2020, 2, 10)
        with pytest.raises(TimeTravelError):
            checker.check(as_of, {
                "unknown_source": {"max_date": date(2020, 3, 1)},
            })


class TestCheckFinancials:
    """财务报表专项PIT检查。"""

    def test_announce_after_as_of_violation(self) -> None:
        checker = TimeTravelChecker()
        as_of = date(2020, 2, 10)
        violations = checker.check_financials(as_of, {
            "max_announce_date": date(2020, 2, 15),
            "codes_with_violations": ["600519"],
        })
        assert len(violations) >= 1
        assert violations[0].field_name == "announce_date"

    def test_report_date_later_but_announce_ok(self) -> None:
        """report_date 晚于 as_of 但 announce_date 合法 → 正常（不是前视偏差）。

        这是常见场景：2020Q4年报的报告期是2020-12-31，
        但实际披露日是2021-03-15。2021-03-16的回测使用该数据是合法的。
        """
        checker = TimeTravelChecker()
        as_of = date(2021, 3, 16)
        violations = checker.check_financials(as_of, {
            "max_report_date": date(2020, 12, 31),  # 报告期早于回测日
            "max_announce_date": date(2021, 3, 15),  # 已披露
        })
        assert len(violations) == 0


class TestTimeTravelViolation:
    """TimeTravelViolation 数据类。"""

    def test_violation_repr(self) -> None:
        v = TimeTravelViolation(
            data_source="test_source",
            field_name="announce_date",
            observed_max_value=date(2020, 3, 1),
            as_of_date=date(2020, 2, 10),
            detail="测试违规",
        )
        s = str(v)
        assert "test_source" in s
        assert "announce_date" in s


class TestValidatePITFinancials:
    """PIT 快速验证函数。"""

    def test_all_announce_before_as_of(self) -> None:
        violations = validate_pit_financials(
            as_of_date=date(2020, 2, 10),
            announce_dates=[date(2020, 2, 5), date(2020, 2, 8)],
            codes=["600519", "000001"],
        )
        assert len(violations) == 0

    def test_announce_after_as_of(self) -> None:
        violations = validate_pit_financials(
            as_of_date=date(2020, 2, 10),
            announce_dates=[date(2020, 2, 15)],
            codes=["600519"],
        )
        assert len(violations) == 1
        assert "600519" in violations[0]
        assert "2020-02-15" in violations[0]
