"""数据质量检查单元测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from core.data.quality import (
    DataQualityChecker,
    DataQualityError,
    QualityIssue,
    QualityReport,
    Severity,
)


class TestQualityReport:
    """QualityReport 数据类测试。"""

    def test_passed_report(self) -> None:
        report = QualityReport(check_name="测试", passed=True)
        assert report.passed is True
        assert report.error_count == 0
        assert report.warn_count == 0

    def test_with_errors(self) -> None:
        report = QualityReport(
            check_name="测试",
            passed=False,
            issues=[
                QualityIssue(field="price", expected=">0", actual="-5", severity=Severity.ERROR),
                QualityIssue(field="volume", expected=">0", actual="0", severity=Severity.WARN),
            ],
        )
        assert report.error_count == 1
        assert report.warn_count == 1
        assert report.passed is False

    def test_str_format(self) -> None:
        report = QualityReport(
            check_name="到达率检查",
            passed=False,
            issues=[
                QualityIssue(field="count", expected="5000", actual="4500", severity=Severity.ERROR),
            ],
        )
        s = str(report)
        assert "到达率检查" in s
        assert "ERROR" in s


class TestArrivalRate:
    """行情到达率检查。"""

    def test_all_arrived(self, sample_daily_kline_df: pd.DataFrame) -> None:
        checker = DataQualityChecker()
        checker._prev_day_stock_count = 5
        report = checker.check_arrival_rate(sample_daily_kline_df, expected_stock_count=5)
        assert report.passed is True
        assert report.stats["missing_rate"] == 0.0

    def test_missing_above_threshold(self, sample_daily_kline_df: pd.DataFrame) -> None:
        checker = DataQualityChecker()
        # 预期5000只，实际只有5只 → 缺失率99.9% → ERROR
        report = checker.check_arrival_rate(sample_daily_kline_df, expected_stock_count=5000)
        assert report.passed is False
        assert len(report.issues) > 0

    def test_no_baseline_skip(self, sample_daily_kline_df: pd.DataFrame) -> None:
        checker = DataQualityChecker()
        report = checker.check_arrival_rate(sample_daily_kline_df)  # 无 expected_count 也无 prev_day
        assert report.passed is True  # 跳过


class TestPriceValidity:
    """价格合理性检查。"""

    def test_valid_data_passes(self, sample_daily_kline_df: pd.DataFrame) -> None:
        checker = DataQualityChecker()
        report = checker.check_price_validity(sample_daily_kline_df)
        assert report.passed is True, f"合法数据应通过检查: {report}"

    def test_invalid_data_detected(self, sample_invalid_kline_df: pd.DataFrame) -> None:
        checker = DataQualityChecker()
        report = checker.check_price_validity(sample_invalid_kline_df)
        # 600002: open=-5 (负价), 600003: low > high
        assert report.passed is False, f"应检测到价格异常"
        assert report.error_count > 0


class TestAccountingIdentity:
    """财务勾稽检查。"""

    def test_valid_financials_passes(self, sample_financials_df: pd.DataFrame) -> None:
        checker = DataQualityChecker()
        report = checker.check_accounting_identity(sample_financials_df)
        assert report.passed is True, f"合法财务数据应通过: {report}"

    def test_unbalanced_detected(self) -> None:
        """构造 资产 ≠ 负债+权益 的异常数据。"""
        import pandas as pd
        from datetime import date

        df = pd.DataFrame({
            "code": ["600888"],
            "report_date": [date(2025, 12, 31)],
            "total_assets": [100.0],        # 资产100
            "total_liabilities": [30.0],     # 负债30
            "total_equity": [30.0],          # 权益30 → 30+30=60 ≠ 100（误差40%）
        })
        checker = DataQualityChecker()
        report = checker.check_accounting_identity(df, tolerance=0.01)
        assert report.passed is False, "应检测到勾稽不平"


class TestDataQualityError:
    """DataQualityError 异常测试。"""

    def test_raises_with_report(self) -> None:
        report = QualityReport(
            check_name="测试",
            passed=False,
            issues=[QualityIssue(field="x", expected="1", actual="2", severity=Severity.ERROR)],
        )
        with pytest.raises(DataQualityError) as exc:
            raise DataQualityError(report)
        assert exc.value.report is report
        assert "测试" in str(exc.value)
