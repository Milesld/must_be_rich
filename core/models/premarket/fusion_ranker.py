"""盘前融合排序模型。

融合长期底池评分 + 隔夜信号 + 公告NLP信号 → 今日推荐综合排序。

关键设计：长期评分是软特征输入而非硬约束——强隔夜信号
（如重大利好公告）可以突破低长期评分，获得高综合排序。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from core.models.base import LightGBMBaseModel

logger = logging.getLogger(__name__)


class FusionRanker(LightGBMBaseModel):
    """盘前融合排序：LightGBM LambdaRank。

    输入特征结构（约50%复用长期特征 + 50%盘前专属特征）：
    - long_term_score: 长期底池排序分数（来自 LongTermRanker.predict_ranks）
    - overnight_adr_mapped, a50_futures, hsi_futures: 隔夜映射
    - announcement_sentiment, announcement_event_type: 公告NLP
    - dragon_tiger_net_buying_rank, limit_up_review_signal: 复盘信号
    - theme_heat_score: 题材热度
    - auction_price_deviation, auction_volume_ratio, ...: 竞价

    输出：今日推荐的综合排序分数（越高越值得关注）。
    """

    def __init__(
        self,
        config_path: str = "configs/models/premarket.yaml",
        model_name: str = "fusion_ranker",
    ) -> None:
        super().__init__(model_name=model_name)
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        cfg = self._config.get("fusion_ranker", {})
        self._params = dict(cfg.get("params", {}))
        self._seed = self._params.get("random_state", 42)

    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """标签：当日日内最大涨幅（作为"值得关注"的代理标签）。

        使用日内 high / open 的最大涨幅（截止14:30）。
        """
        if "high" in data.columns and "open" in data.columns:
            return (data["high"] - data["open"]) / data["open"].replace(0, float("nan"))
        return pd.Series(np.nan, index=data.index)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> None:
        """训练 LambdaRank 排序模型。

        Args:
            X: 特征矩阵（必须包含 long_term_score 列——由调用方注入）。
            y: 标签（日内最大涨幅）。
            groups: 日期分组。
        """
        import lightgbm as lgb

        self._set_seed()
        self._feature_names = list(X.columns)

        # 验证 long_term_score 在输入中
        if "long_term_score" not in self._feature_names:
            logger.warning(
                "融合排序器输入中缺少 'long_term_score' 特征。"
                "盘前融合模型应接收长期排序分数作为基线输入——"
                "请确认调用方在 X 中包含了 long_term_score 列。"
            )

        params = dict(self._params)
        params["random_state"] = self._seed
        params["verbosity"] = -1

        mask = y.notna()
        X_clean = X.loc[mask.values]
        y_clean = y.loc[mask.values]

        y_int_ok = y_clean.apply(lambda v: v == int(v)).all() if len(y_clean) > 0 else False
        use_lambdarank = (groups is not None) and y_int_ok

        if use_lambdarank:
            params["objective"] = "lambdarank"
            params["metric"] = "ndcg"
            y_vals = y_clean.values.astype(int)
            train_data = lgb.Dataset(
                X_clean.values, label=y_vals,
                group=groups.loc[mask.values].value_counts(sort=False).values,
            )
        else:
            logger.info("FusionRanker: 使用回归模式")
            params["objective"] = "regression"
            params["metric"] = "rmse"
            train_data = lgb.Dataset(X_clean.values, label=y_clean.values.astype(float))

        self._model = lgb.train(params, train_data, valid_sets=[train_data])
        self._trained = True
        logger.info("FusionRanker 训练完成: samples=%d, features=%d", mask.sum(), len(self._feature_names))

    def predict_ranks(self, X: pd.DataFrame) -> pd.Series:
        """返回综合推荐排序分数（越高越值得关注）。"""
        preds = self.predict(X)
        return pd.Series(preds.flatten(), index=X.index, name="fusion_score")

    def get_top_recommendations(
        self,
        X: pd.DataFrame,
        top_n: int = 30,
        min_score: float = 0.3,
    ) -> pd.DataFrame:
        """返回 Top-N 推荐清单。

        Args:
            X: 特征矩阵（index 为股票代码）。
            top_n: 最多推荐数量。
            min_score: 综合评分最低门限。

        Returns:
            DataFrame: columns=[code, fusion_score, ...]
        """
        scores = self.predict_ranks(X)
        qualified = scores[scores >= min_score].nlargest(top_n)
        result = qualified.reset_index()
        result.columns = ["code", "fusion_score"]
        return result
