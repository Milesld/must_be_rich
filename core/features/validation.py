"""特征工程规范性工具。

提供因子质量监控：
- 缺失率检查（单因子 + 全局热力图）
- 异常值检测（MAD 方法）
- 分布漂移检测（KS 检验）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 缺失率检查 ──────────────────────────────────

@dataclass
class MissingRateReport:
    """单因子缺失率报告。"""
    factor_name: str
    total_rows: int
    missing_count: int
    missing_rate: float
    threshold: float = 0.05
    passed: bool = True

    def __str__(self) -> str:
        status = "✓" if self.passed else "✗"
        return (
            f"{status} {self.factor_name}: "
            f"{self.missing_count}/{self.total_rows} ({self.missing_rate:.1%}) "
            f"[threshold={self.threshold:.1%}]"
        )


class FactorMissingRateChecker:
    """因子缺失率检查器。

    在因子入库前和定期监控中检查每个因子的缺失率，
    超过注册时的 missing_threshold 则告警。
    """

    def __init__(self, thresholds: Optional[dict[str, float]] = None) -> None:
        """初始化。

        Args:
            thresholds: {factor_name: threshold} 字典，覆盖因子注册表中的默认值。
        """
        self._thresholds = thresholds or {}

    def check_factor(
        self,
        factor_name: str,
        values: pd.Series,
        threshold: Optional[float] = None,
    ) -> MissingRateReport:
        """检查单因子缺失率。

        Args:
            factor_name: 因子名。
            values: 因子值序列。
            threshold: 告警阈值（覆盖注册表中的值）。

        Returns:
            MissingRateReport。
        """
        total = len(values)
        missing = values.isna().sum()
        rate = missing / total if total > 0 else 0.0
        thresh = threshold or self._thresholds.get(factor_name, 0.05)

        report = MissingRateReport(
            factor_name=factor_name,
            total_rows=total,
            missing_count=missing,
            missing_rate=rate,
            threshold=thresh,
            passed=rate <= thresh,
        )

        if not report.passed:
            logger.warning("因子缺失率超标: %s", report)

        return report

    def check_all(
        self,
        factors: dict[str, pd.Series],
    ) -> list[MissingRateReport]:
        """批量检查所有因子的缺失率。"""
        return [
            self.check_factor(name, values)
            for name, values in factors.items()
        ]

    def heatmap_data(
        self,
        factors: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """生成因子缺失热力图数据（按日期分组）。"""
        # 将所有因子值合并到一个 DataFrame
        dfs = []
        for name, values in factors.items():
            dfs.append(values.rename(name).to_frame())
        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, axis=1)
        # 按日期分组计算缺失率
        missing_by_date = combined.isna().groupby(
            combined.index.get_level_values("trade_date")
            if "trade_date" in combined.index.names
            else combined.index
        ).mean()
        return missing_by_date


# ── 异常值检测 ──────────────────────────────────

@dataclass
class OutlierReport:
    """异常值检测报告。"""
    factor_name: str
    total_rows: int
    outlier_count: int
    outlier_ratio: float
    lower_bound: float
    upper_bound: float
    method: str = "MAD"


def detect_outliers_mad(
    series: pd.Series,
    n_mad: float = 5.0,
) -> Tuple[pd.Series, OutlierReport]:
    """使用 MAD 方法检测异常值。

    Args:
        series: 因子值序列。
        n_mad: MAD 倍数阈值。

    Returns:
        (异常值mask, OutlierReport)
    """
    valid = series.dropna()
    if len(valid) == 0:
        return pd.Series(False, index=series.index), OutlierReport(
            factor_name="unknown",
            total_rows=len(series),
            outlier_count=0,
            outlier_ratio=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
        )

    median = valid.median()
    mad = (valid - median).abs().median()
    if mad == 0:
        mad = valid.std()

    lower = median - n_mad * mad
    upper = median + n_mad * mad

    is_outlier = (series < lower) | (series > upper)

    return is_outlier, OutlierReport(
        factor_name="unknown",
        total_rows=len(series),
        outlier_count=is_outlier.sum(),
        outlier_ratio=is_outlier.sum() / len(series) if len(series) > 0 else 0.0,
        lower_bound=lower,
        upper_bound=upper,
        method=f"MAD({n_mad}×)",
    )


# ── 分布漂移检测 ──────────────────────────────────

@dataclass
class DriftReport:
    """分布漂移检测报告。"""
    factor_name: str
    ks_statistic: float
    p_value: float
    drifted: bool        # True = 当前分布与历史分布显著不同
    threshold_p: float = 0.01


def detect_distribution_drift(
    current_values: pd.Series,
    reference_values: pd.Series,
    threshold_p: float = 0.01,
    factor_name: str = "unknown",
) -> DriftReport:
    """KS 检验检测因子分布漂移。

    比较当前窗口和参考窗口的因子值分布。
    如果 p-value < threshold_p，说明分布发生显著漂移——模型可能需要重训练。

    Args:
        current_values: 当前窗口的因子值。
        reference_values: 参考窗口的因子值（通常为训练集窗口）。
        threshold_p: p-value 阈值（默认 0.01）。
        factor_name: 因子名（用于报告）。

    Returns:
        DriftReport
    """
    from scipy.stats import ks_2samp

    cur = current_values.dropna()
    ref = reference_values.dropna()

    if len(cur) < 30 or len(ref) < 30:
        return DriftReport(
            factor_name=factor_name,
            ks_statistic=0.0,
            p_value=1.0,
            drifted=False,
            threshold_p=threshold_p,
        )

    stat, pval = ks_2samp(cur, ref)

    report = DriftReport(
        factor_name=factor_name,
        ks_statistic=stat,
        p_value=pval,
        drifted=pval < threshold_p,
        threshold_p=threshold_p,
    )

    if report.drifted:
        logger.warning(
            "因子分布漂移: %s (KS=%.4f, p=%.6f)", factor_name, stat, pval,
        )

    return report


def detect_all_drifts(
    current: dict[str, pd.Series],
    reference: dict[str, pd.Series],
    threshold_p: float = 0.01,
) -> list[DriftReport]:
    """批量检测所有因子的分布漂移。"""
    reports: list[DriftReport] = []
    for name in current:
        ref = reference.get(name, pd.Series())
        reports.append(
            detect_distribution_drift(
                current[name], ref, threshold_p, name,
            )
        )
    return reports
