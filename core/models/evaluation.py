"""模型评估工具箱。

量化模型评估核心指标：
- Rank IC / ICIR：排序模型的截面预测能力
- IC Decay：IC在不同前向期的衰减曲线
- 分层收益：按预测分位数分组的单调性检验
- 方向准确率：日内三分类模型评估
- 概率校准曲线：分类概率可靠性
- Rolling 监控：模型衰减自动化检测
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估静态方法合集。

    所有方法均为纯函数，输入 Series/DataFrame，输出浮点或 DataFrame。
    """

    # ── Rank IC ──────────────────────────────

    @staticmethod
    def rank_ic(predictions: pd.Series, returns: pd.Series) -> float:
        """Rank IC = Spearman 相关系数。

        截面：同一交易日内的排名相关性。
        时序：跨日期的所有样本。

        Args:
            predictions: 模型预测分数（越高期望收益越高）。
            returns: 实际收益。

        Returns:
            float: -1.0 ~ 1.0
        """
        common = predictions.dropna().index.intersection(returns.dropna().index)
        if len(common) < 5:
            return 0.0
        return float(predictions.loc[common].corr(returns.loc[common], method="spearman"))

    @staticmethod
    def cross_sectional_ic(
        predictions: pd.Series,
        returns: pd.Series,
        date_index: pd.Index,
    ) -> pd.Series:
        """逐日截面 IC 序列（按 trade_date 分组计算）。"""
        df = pd.DataFrame({
            "pred": predictions.values,
            "ret": returns.values,
            "_date": date_index.values,
        }).dropna()
        ic_values: dict[str, float] = {}
        for date_val, group in df.groupby("_date"):
            if len(group) >= 5:
                ic_values[str(date_val)] = float(
                    group["pred"].corr(group["ret"], method="spearman")
                )
        return pd.Series(ic_values)

    # ── ICIR ────────────────────────────────

    @staticmethod
    def icir(ic_series: pd.Series) -> float:
        """ICIR = IC 均值 / IC 标准差。

        ICIR > 0.5 表示因子/模型的预测能力在统计上显著。
        """
        ic = ic_series.dropna()
        if len(ic) < 10 or ic.std() == 0:
            return 0.0
        return float(ic.mean() / ic.std())

    # ── IC 衰减 ──────────────────────────────

    @staticmethod
    def ic_decay(
        predictions: pd.Series,
        forward_returns: pd.DataFrame,  # columns = 1d, 5d, 10d, 20d, ...
    ) -> pd.Series:
        """IC 在不同前向期的衰减曲线。

        Args:
            predictions: 当日预测分数。
            forward_returns: 各前向期的收益 DataFrame，columns 为期数。

        Returns:
            IC 衰减序列，index = 期数。
        """
        ics: dict[str, float] = {}
        for period in forward_returns.columns:
            ic = ModelEvaluator.rank_ic(predictions, forward_returns[period])
            ics[str(period)] = ic
        return pd.Series(ics)

    # ── 分层收益 ──────────────────────────────

    @staticmethod
    def layered_returns(
        predictions: pd.Series,
        returns: pd.Series,
        n_quantiles: int = 10,
    ) -> pd.DataFrame:
        """分层收益分析（Monotonicity 检验）。

        如果收益随着预测分位组单调递增 → 模型排序能力好。

        Args:
            predictions: 模型预测分数。
            returns: 实际收益。
            n_quantiles: 分位组数（默认10）。

        Returns:
            DataFrame: columns=[quantile, mean_return, count]
        """
        common = predictions.dropna().index.intersection(returns.dropna().index)
        df = pd.DataFrame({
            "pred": predictions.loc[common],
            "ret": returns.loc[common],
        })

        df["quantile"] = pd.qcut(
            df["pred"], n_quantiles, labels=False, duplicates="drop",
        )

        result = df.groupby("quantile").agg(
            mean_return=("ret", "mean"),
            median_return=("ret", "median"),
            std_return=("ret", "std"),
            count=("ret", "count"),
        ).reset_index()

        return result

    # ── 方向准确率 ──────────────────────────

    @staticmethod
    def direction_accuracy(
        y_true_direction: pd.Series,  # 实际涨跌方向: +1/0/-1
        y_pred_direction: pd.Series,  # 预测方向: +1/0/-1
    ) -> float:
        """方向准确率——日内模型关键评估指标。

        Returns:
            float: 0.0 ~ 1.0（>0.55 视为及格）。
        """
        mask = y_true_direction.notna() & y_pred_direction.notna()
        if mask.sum() == 0:
            return 0.0
        correct = (y_true_direction.loc[mask] == y_pred_direction.loc[mask]).sum()
        return float(correct / mask.sum())

    # ── 概率校准 ──────────────────────────────

    @staticmethod
    def calibration_curve(
        y_true: pd.Series,
        y_prob: np.ndarray,  # (n_samples,)
        n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """概率校准曲线（Reliability Diagram）。

        Args:
            y_true: 真实标签 (0/1)。
            y_prob: 预测概率。
            n_bins: 分桶数。

        Returns:
            (bin_means, bin_probs) — 用于画校准曲线。
        """
        from sklearn.calibration import calibration_curve

        mask = ~np.isnan(y_prob) & y_true.notna()
        return calibration_curve(
            y_true.loc[mask].values.astype(int),
            y_prob[mask],
            n_bins=n_bins,
            strategy="uniform",
        )

    # ── Rolling 监控 ──────────────────────────

    @staticmethod
    def rolling_monitoring(
        predictions: pd.Series,
        returns: pd.Series,
        date_index: pd.Index,
        window: int = 20,
    ) -> pd.DataFrame:
        """Rolling IC 监控——用于模型衰减检测。

        当 Rolling IC 连续N期低于阈值 → 触发重训练告警。

        Args:
            predictions: 预测分数。
            returns: 实际收益。
            date_index: 日期索引（按交易日）。
            window: 滚动窗口大小（交易日数）。

        Returns:
            DataFrame: columns=[date, rolling_ic_mean, rolling_ic_std, icir]
        """
        cs_ic = ModelEvaluator.cross_sectional_ic(predictions, returns, date_index)
        if len(cs_ic) < window:
            return pd.DataFrame()

        ic_vals = pd.Series(cs_ic.values)
        roll_mean = ic_vals.rolling(window, min_periods=window // 2).mean()
        roll_std = ic_vals.rolling(window, min_periods=window // 2).std()

        result = pd.DataFrame({
            "date": cs_ic.index,
            "ic": cs_ic.values,
            "rolling_ic_mean": roll_mean.values,
            "rolling_ic_std": roll_std.values,
        })
        result["icir"] = result["rolling_ic_mean"] / result["rolling_ic_std"].replace(0, float("nan"))
        return result

    # ── 综合报告 ────────────────────────────

    @staticmethod
    def full_report(
        predictions: pd.Series,
        returns: pd.Series,
        date_index: Optional[pd.Index] = None,
    ) -> dict[str, float]:
        """生成综合评估报告。"""
        ic_val = ModelEvaluator.rank_ic(predictions, returns)
        icir_val = 0.0

        if date_index is not None:
            cs_ic = ModelEvaluator.cross_sectional_ic(predictions, returns, date_index)
            icir_val = ModelEvaluator.icir(cs_ic)

        # 方向准确率（简化）
        pred_dir = pd.Series(np.sign(predictions), index=predictions.index)
        true_dir = pd.Series(np.sign(returns), index=returns.index)
        dir_acc = ModelEvaluator.direction_accuracy(true_dir, pred_dir)

        report = {
            "rank_ic": float(ic_val),
            "icir": float(icir_val),
            "direction_accuracy": float(dir_acc),
            "n_samples": int(predictions.dropna().index.intersection(returns.dropna().index).shape[0]),
        }

        if abs(ic_val) < 0.03:
            logger.warning("Rank IC = %.4f < 0.03，模型排序能力偏弱", ic_val)

        return report
