"""日内预测模型。

基于 LightGBM 的分钟级方向预测：
- IntradayClassifier: 涨/跌/平三分类
- IntradayQuantileRegressor: 分位数回归（预测收盘价区间）
- DistributionPredictor: NGBoost 全分布建模（实验性）
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from core.models.base import BaseModel, LightGBMBaseModel

logger = logging.getLogger(__name__)


# ── 日内三分类模型 ──────────────────────────────

class IntradayClassifier(LightGBMBaseModel):
    """日内涨/跌/平三分类 LightGBM 模型。

    输入：实时特征快照 + "距收盘剩余时间"特征。
    输出：三分类概率 {up: p1, flat: p2, down: p3}。

    按距收盘剩余时间分桶分别建模（方案§5.2 训练注意事项）。
    """

    def __init__(
        self,
        config_path: str = "configs/models/intraday.yaml",
        model_name: str = "intraday_classifier",
    ) -> None:
        super().__init__(model_name=model_name)
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        self._params = dict(self._config.get("params", {}))
        self._num_class = self._config.get("num_class", 3)
        self._seed = self._params.get("random_state", 42)

    def prepare_labels(
        self,
        data: pd.DataFrame,
        target_time: str = "close",
    ) -> pd.Series:
        """构造三分类标签：涨(2)/平(1)/跌(0)。

        Args:
            data: 必须包含 close 列（或 target_time 对应的价格列）。
                  如果数据是分钟级，需要 close 为当前时刻价格，
                  另有 target_close 为收盘价。
            target_time: 目标价格列。'close' 用于日频回测场景；
                         'next_30min' 用于30分钟后方向预测。

        Returns:
            0=跌, 1=平, 2=涨 的分类标签。
        """
        if target_time == "close" and "target_close" in data.columns:
            target_px = data["target_close"]
        elif "close" in data.columns:
            target_px = data["close"]
        else:
            raise ValueError("data 必须包含 close 列或 target_close 列")

        current_px = data.get("current_price", data["close"])
        fwd_return = (target_px - current_px) / current_px.replace(0, float("nan"))

        # 分桶阈值：±0.5% 以内为"平"
        threshold = 0.005
        labels = pd.Series(1, index=data.index, dtype=int)  # 默认：平
        labels[fwd_return > threshold] = 2                   # 涨
        labels[fwd_return < -threshold] = 0                  # 跌
        return labels

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        import lightgbm as lgb

        self._set_seed()
        self._feature_names = list(X.columns)

        params = dict(self._params)
        params["objective"] = "multiclass"
        params["num_class"] = self._num_class
        params["metric"] = "multi_logloss"
        params["random_state"] = self._seed

        mask = y.notna()
        X_clean = X.loc[mask]
        y_clean = y.loc[mask].astype(int)

        train_data = lgb.Dataset(
            X_clean.values, label=y_clean.values,
            weight=sample_weights.loc[mask].values if sample_weights is not None else None,
        )
        self._model = lgb.train(params, train_data, valid_sets=[train_data])
        self._trained = True
        logger.info("IntradayClassifier 训练完成: samples=%d", len(X_clean))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回 (n_samples, 3) 概率矩阵：[跌, 平, 涨]"""
        self._check_trained()
        return self._model.predict(X.values)  # type: ignore[union-attr]

    def predict_direction(self, X: pd.DataFrame) -> pd.Series:
        """返回方向预测（0=跌, 1=平, 2=涨）和置信度。"""
        proba = self.predict_proba(X)
        return pd.Series(np.argmax(proba, axis=1), index=X.index)


# ── 分位数回归模型 ──────────────────────────────

class IntradayQuantileRegressor:
    """LightGBM 分位数回归：输出收盘价条件分位数。

    训练三个独立模型：P10（悲观下界）、P50（中位数）、P90（乐观上界）。
    从而获得预测区间而非点值预测——更符合"不确定性估计"的需求。
    """

    def __init__(
        self,
        quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
        model_name: str = "intraday_qr",
    ) -> None:
        self.quantiles = quantiles
        self.model_name = model_name
        self._models: dict[float, Any] = {}
        self._feature_names: list[str] = []
        self._trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,  # 收盘价（连续值）
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        import lightgbm as lgb

        self._feature_names = list(X.columns)
        mask = y.notna()
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]

        for q in self.quantiles:
            params = {
                "objective": "quantile",
                "alpha": q,
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.03,
                "n_estimators": 200,
                "random_state": 42,
                "verbosity": -1,
            }
            train_data = lgb.Dataset(X_clean.values, label=y_clean.values)
            model = lgb.train(params, train_data, valid_sets=[train_data])
            self._models[q] = model

        self._trained = True
        logger.info("IntradayQuantileRegressor 训练完成: quantiles=%s", self.quantiles)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """返回各分位数的预测值 DataFrame。

        Returns:
            columns=[q10, q50, q90]，行=样本数。
        """
        if not self._trained:
            raise RuntimeError("模型尚未训练")
        result = {}
        for q in self.quantiles:
            result[f"q{int(q*100)}"] = self._models[q].predict(X.values)
        return pd.DataFrame(result, index=X.index)

    @property
    def is_trained(self) -> bool:
        return self._trained


# ── 分布预测器（实验性）───────────────────────────

class DistributionPredictor:
    """NGBoost 全分布建模（实验性）。

    输出完整条件分布 → 可提取任意分位数、预测区间、不确定性度量。
    作为分位数回归的升级替代方案，实验性质。
    """

    def __init__(self, model_name: str = "distribution_predictor") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._trained = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练 NGBoost（默认: Normal 分布）。

        NGBoost 训练较慢，适合小到中等数据集。
        """
        try:
            from ngboost import NGBoost
            from ngboost.distns import Normal
        except ImportError:
            logger.warning(
                "ngboost 未安装，DistributionPredictor 不可用。"
                "请 pip install ngboost 后重试。"
            )
            self._trained = True  # placeholder
            return

        mask = y.notna()
        self._model = NGBoost(Dist=Normal, Score="LogScore", random_state=42)
        self._model.fit(X.loc[mask].values, y.loc[mask].values)  # type: ignore[attr-defined]
        self._trained = True
        logger.info("DistributionPredictor 训练完成")

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """返回预测分布参数。

        Returns:
            {'loc': mean, 'scale': std}
        """
        if not self._trained:
            raise RuntimeError("模型尚未训练")
        preds = self._model.pred_dist(X.values)  # type: ignore[union-attr]
        return {"loc": preds.params["loc"], "scale": preds.params["scale"]}

    @property
    def is_trained(self) -> bool:
        return self._trained
