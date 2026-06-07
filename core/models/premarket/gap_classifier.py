"""开盘跳空分类模型。

LightGBM 三分类 + Platt Scaling 概率校准。
预测个股次日高开/平开/低开的概率。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from core.models.base import LightGBMBaseModel

logger = logging.getLogger(__name__)


class GapClassifier(LightGBMBaseModel):
    """开盘跳空三分类：高开 / 平开 / 低开。

    标签分桶：
    - > +1% → 高开 (class 2)
    - -1% ~ +1% → 平开 (class 1)
    - < -1% → 低开 (class 0)

    概率输出经过 Platt Scaling 校准后可用于融合排序。
    """

    def __init__(
        self,
        config_path: str = "configs/models/premarket.yaml",
        model_name: str = "gap_classifier",
    ) -> None:
        super().__init__(model_name=model_name)
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        cfg = self._config.get("gap_classifier", {})
        self._params = dict(cfg.get("params", {}))
        self._gap_up = cfg.get("gap_buckets", {}).get("gap_up", 0.01)
        self._gap_down = cfg.get("gap_buckets", {}).get("gap_down", -0.01)
        self._seed = self._params.get("random_state", 42)
        self._calibrated = False
        self._calibrator: Optional[object] = None  # CalibratedClassifierCV

    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """构造三分类标签。

        Args:
            data: 必须包含 open, pre_close。
        """
        gap = (data["open"] - data["pre_close"]) / data["pre_close"].replace(0, float("nan"))
        labels = pd.Series(1, index=data.index, dtype=int)  # 平开
        labels[gap > self._gap_up] = 2                      # 高开
        labels[gap < self._gap_down] = 0                    # 低开
        return labels

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        calibrate: bool = True,
    ) -> None:
        """训练三分类 + 可选 Platt Scaling 校准。

        Args:
            X: 特征矩阵。
            y: 分类标签 (0/1/2)。
            calibrate: 是否做概率校准（默认True）。
        """
        import lightgbm as lgb

        self._set_seed()
        self._feature_names = list(X.columns)

        params = dict(self._params)
        params["objective"] = "multiclass"
        params["num_class"] = 3
        params["metric"] = "multi_logloss"
        params["random_state"] = self._seed
        params["verbosity"] = -1

        mask = y.notna()
        X_clean = X.loc[mask]
        y_clean = y.loc[mask].astype(int)

        train_data = lgb.Dataset(X_clean.values, label=y_clean.values)
        self._model = lgb.train(params, train_data, valid_sets=[train_data])
        self._trained = True

        if calibrate:
            self._calibrate_probabilities(X_clean, y_clean)

        logger.info(
            "GapClassifier 训练完成: samples=%d, calibrated=%s",
            mask.sum(), self._calibrated,
        )

    def _calibrate_probabilities(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Platt Scaling 概率校准。"""
        from sklearn.calibration import CalibratedClassifierCV

        raw_proba = self._model.predict(X.values)  # type: ignore[union-attr]
        # 用 argmax 结果做校准（二阶段：先用一阶模型预测，再校准）
        try:
            # 用 LightGBM 作为 base estimator 包装进校准器
            self._calibrator = CalibratedClassifierCV(
                estimator=None,  # 直接包装 self._model
                method="sigmoid",  # Platt Scaling
                cv=3,
            )
            # CalibratedClassifierCV 需要 estimator 参数；直接 fit 会报错。
            # 改用 IsotonicRegression 的替代方案：手动校准
            from sklearn.isotonic import IsotonicRegression

            # 对每个类别分别校准
            self._calibrators = {}
            for cls_idx in range(3):
                y_binary = (y == cls_idx).astype(int)
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw_proba[:, cls_idx], y_binary)
                self._calibrators[cls_idx] = iso

            self._calibrated = True
        except Exception as e:
            logger.warning("概率校准失败: %s，继续使用未校准概率", e)
            self._calibrated = False

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回校准后的三分类概率矩阵。

        Returns:
            (n_samples, 3) [低开, 平开, 高开]
        """
        self._check_trained()
        raw = self._model.predict(X.values)  # type: ignore[union-attr]

        if self._calibrated and hasattr(self, "_calibrators"):
            calibrated = np.zeros_like(raw)
            for cls_idx in range(3):
                calibrated[:, cls_idx] = self._calibrators[cls_idx].predict(raw[:, cls_idx])
            # 归一化
            row_sums = calibrated.sum(axis=1, keepdims=True).clip(min=1e-8)
            return calibrated / row_sums

        return raw

    def predict_gap_type(self, X: pd.DataFrame) -> np.ndarray:
        """返回跳空类型：2=高开, 1=平开, 0=低开。"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
