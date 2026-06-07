"""模型抽象基类。

所有模型（LightGBM、CatBoost、线性、NLP）继承此基类，
统一 save/load 接口和版本管理。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """所有模型的抽象基类。

    子类必须实现 train() 和 predict()。
    save/load 使用 joblib，版本号纳入文件名以保证可追溯。
    """

    def __init__(self, model_name: str = "base", seed: int = 42) -> None:
        self.model_name = model_name
        self.model_version = "v0.0.0"
        self._seed = seed
        self._trained = False
        # 子类在 train() 中设置
        self._feature_names: list[str] = []
        self._model: object = None  # type: ignore[assignment]

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        """训练模型。

        Args:
            X: 特征矩阵。
            y: 目标变量（回归/分类/排序标签）。
            sample_weights: 样本权重（可选）。
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """推理（返回原始预测值——分数或类别索引）。"""
        ...

    # ── 持久化 ────────────────────────────────

    def save(self, path: str) -> None:
        """保存模型到文件。

        文件名格式: {model_name}_{model_version}.joblib
        """
        import joblib

        _path = Path(path)
        _path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self._model,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_names": self._feature_names,
            "seed": self._seed,
            "trained": self._trained,
        }
        joblib.dump(data, _path)
        logger.info("模型已保存: %s (%s)", _path, self.model_version)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """从文件加载模型。

        注意：返回的是基类类型，调用方需要用具体子类重新包装。
        子类可以覆盖此方法返回正确的类型。
        """
        import joblib

        data = joblib.load(Path(path))
        instance = cls.__new__(cls)
        instance.model_name = data["model_name"]
        instance.model_version = data["model_version"]
        instance._feature_names = data.get("feature_names", [])
        instance._seed = data.get("seed", 42)
        instance._trained = data.get("trained", True)
        instance._model = data["model"]
        return instance

    # ── 元信息 ────────────────────────────────

    def get_params(self) -> dict:
        """获取模型参数（子类覆盖以提供具体参数）。"""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "seed": self._seed,
            "trained": self._trained,
        }

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def _set_seed(self) -> None:
        """固定随机种子（训练前调用）。"""
        np.random.seed(self._seed)


class LightGBMBaseModel(BaseModel):
    """LightGBM 专用基类，封装通用 save/load/predict。"""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_trained()
        return self._model.predict(X.values)  # type: ignore[union-attr]

    def _check_trained(self) -> None:
        if not self._trained or self._model is None:
            raise RuntimeError(f"模型 {self.model_name} 尚未训练，请先调用 train()")

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """获取特征重要性。

        Args:
            importance_type: 'gain' 或 'split'。

        Returns:
            DataFrame: columns=[feature, importance]
        """
        self._check_trained()
        # lightgbm 4.x: train() returns booster directly
        booster = getattr(self._model, "booster_", self._model)
        imp = booster.feature_importance(importance_type=importance_type)
        names = booster.feature_name()
        return pd.DataFrame({
            "feature": names,
            "importance": imp,
        }).sort_values("importance", ascending=False)
