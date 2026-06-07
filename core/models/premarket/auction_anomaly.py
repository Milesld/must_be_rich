"""集合竞价异常检测。

Isolation Forest + 规则引擎双层检测：
- 规则层：捕捉明显的虚假申报模式（如9:15-9:20大单后撤单）
- 模型层：Isolation Forest 捕捉复杂异常模式
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class AuctionAnomalyDetector:
    """集合竞价异常检测器。

    双层架构：
    1. 规则引擎（快速、可解释）：捕捉已知模式
    2. Isolation Forest（统计）：捕捉未知复杂异常

    输出：异常评分 0（正常）~ 1（高度可疑）。
    """

    def __init__(
        self,
        config_path: str = "configs/models/premarket.yaml",
        model_name: str = "auction_anomaly",
    ) -> None:
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        cfg = self._config.get("auction_anomaly", {})
        self._model_params = cfg.get("params", {})
        self._rules = cfg.get("rules", {})
        self._contamination = self._model_params.get("contamination", 0.05)
        self._seed = self._model_params.get("random_state", 42)

        self._fake_order_threshold = self._rules.get("fake_order_ratio_threshold", 3.0)
        self._price_dev_threshold = self._rules.get("price_deviation_threshold", 0.05)

        self._model: Any = None  # IsolationForest
        self._trained = False
        self._feature_names: list[str] = []
        self.model_name = model_name

    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """从竞价快照数据中提取用于异常检测的特征。"""
        features = pd.DataFrame(index=data.index)

        # 申报量与匹配量比值（虚假申报核心指标）
        if "auction_order_count_pre_920" in data.columns and "auction_order_count_post_920" in data.columns:
            pre = data["auction_order_count_pre_920"].fillna(0)
            post = data["auction_order_count_post_920"].fillna(1).replace(0, 1)
            features["fake_order_ratio"] = pre / post

        # 虚拟开盘价偏离度
        if "virtual_open" in data.columns and "pre_close" in data.columns:
            features["price_deviation"] = (
                (data["virtual_open"] - data["pre_close"])
                / data["pre_close"].replace(0, float("nan"))
            ).abs()

        # 买卖不平衡度
        if "unmatched_buy" in data.columns and "unmatched_sell" in data.columns:
            total = (data["unmatched_buy"] + data["unmatched_sell"]).replace(0, float("nan"))
            features["imbalance"] = (
                (data["unmatched_buy"] - data["unmatched_sell"]) / total
            ).abs()

        # 成交量异常
        if "matched_volume" in data.columns:
            vol = data["matched_volume"].fillna(0)
            features["volume_zscore"] = (vol - vol.mean()) / (vol.std() + 1e-8)

        self._feature_names = list(features.columns)
        return features.fillna(0)

    def train(self, X: pd.DataFrame) -> None:
        """训练 Isolation Forest 异常检测模型。"""
        from sklearn.ensemble import IsolationForest

        features = self._extract_features(X)
        self._model = IsolationForest(
            contamination=self._contamination,
            random_state=self._seed,
        )
        self._model.fit(features.values)
        self._trained = True
        logger.info("AuctionAnomalyDetector 训练完成: samples=%d", len(features))

    def detect(self, data: pd.DataFrame) -> pd.Series:
        """检测异常并返回综合评分 0（正常）~ 1（可疑）。

        两层检测融合：规则评分 × 0.4 + 模型评分 × 0.6。
        """
        features = self._extract_features(data)

        # 1. 规则层评分
        rule_score = pd.Series(0.0, index=features.index)
        if "fake_order_ratio" in features.columns:
            rule_score += (features["fake_order_ratio"] / self._fake_order_threshold).clip(upper=1.0) * 0.6
        if "price_deviation" in features.columns:
            rule_score += (features["price_deviation"] / self._price_dev_threshold).clip(upper=1.0) * 0.4
        rule_score = rule_score.clip(0.0, 1.0)

        # 2. 模型层评分
        model_score = pd.Series(0.0, index=features.index)
        if self._model is not None:
            # IsolationForest: -1=异常, 1=正常
            raw = self._model.decision_function(features.values)
            # 归一化到 0~1（越低越异常 → 越高越可疑）
            raw_scaled = (1.0 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-8))
            model_score = pd.Series(raw_scaled, index=features.index)

        # 融合
        combined = rule_score * 0.4 + model_score * 0.6
        return combined.clip(0.0, 1.0)

    def flag_suspicious(self, data: pd.DataFrame, threshold: float = 0.7) -> pd.Series:
        """标记高度可疑的竞价行为（bool mask）。"""
        scores = self.detect(data)
        return scores > threshold

    @property
    def is_trained(self) -> bool:
        return self._trained
