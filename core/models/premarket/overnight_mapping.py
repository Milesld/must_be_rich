"""隔夜海外市场→A股开盘方向映射模型。

输入：隔夜海外市场/期货涨跌幅 + 个股特征
输出：个股次日开盘涨跌方向及幅度
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from core.models.base import LightGBMBaseModel

logger = logging.getLogger(__name__)


class OvernightMappingModel(LightGBMBaseModel):
    """隔夜海外市场→A股开盘方向映射。

    标签：次日开盘价相对前日收盘价的涨跌幅（连续回归）。
    可切换为分类模式（涨/跌/平）。
    """

    def __init__(
        self,
        config_path: str = "configs/models/premarket.yaml",
        model_name: str = "overnight_mapping",
    ) -> None:
        super().__init__(model_name=model_name)
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        cfg = self._config.get("overnight_mapping", {})
        self._params = dict(cfg.get("params", {}))
        self._seed = self._params.get("random_state", 42)

    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """构造次日开盘涨跌幅标签。

        Args:
            data: 必须包含 open（当日开盘）, pre_close（昨收）。
        """
        return (data["open"] - data["pre_close"]) / data["pre_close"].replace(0, float("nan"))

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
        params["objective"] = "regression"
        params["metric"] = "rmse"
        params["random_state"] = self._seed
        params["verbosity"] = -1

        mask = y.notna()
        train_data = lgb.Dataset(
            X.loc[mask].values, label=y.loc[mask].values,
        )
        self._model = lgb.train(params, train_data, valid_sets=[train_data])
        self._trained = True
        logger.info("OvernightMappingModel 训练完成: samples=%d", mask.sum())

    def predict_direction(self, X: pd.DataFrame) -> np.ndarray:
        """返回预期开盘方向：+1(涨), 0(平), -1(跌)。"""
        preds = self.predict(X).flatten()
        result = np.zeros(len(preds), dtype=int)
        result[preds > 0.01] = 1
        result[preds < -0.01] = -1
        return result
