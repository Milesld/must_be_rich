"""长期选股模型。

基于 LightGBM 的多因子横截面排序与分类模型。
- LongTermRanker: LambdaRank 排序（月度全市场扫描）
- LongTermClassifier: 二分类（跑赢/跑输基准）
- LinearBaseline: OLS多因子线性基线
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from core.models.base import BaseModel, LightGBMBaseModel

logger = logging.getLogger(__name__)


# ── LightGBM LambdaRank 排序模型 ──────────────────

class LongTermRanker(LightGBMBaseModel):
    """LightGBM LambdaRank 多因子横截面排序模型。

    用途：月度全市场扫描，输出排序分数（越高期望收益越高）。

    标签为分组排序标签（LambdaRank 所需格式）：同一交易日的所有股票
    按未来收益排序，排名越高的标签值越大。
    """

    def __init__(
        self,
        config_path: str = "configs/models/long_term.yaml",
        model_name: str = "long_term_ranker",
    ) -> None:
        super().__init__(model_name=model_name)
        self._config = self._load_config(config_path)
        self._params = self._config.get("params", {})
        self._seed = self._params.get("random_state", 42)

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def prepare_labels(
        self,
        data: pd.DataFrame,
        forward_period: int = 20,
    ) -> pd.Series:
        """构造前向收益标签（用于排序→LambdaRank）。

        Args:
            data: 包含 code, trade_date, close 的行情 DataFrame。
            forward_period: 前向窗口（交易日）。

        Returns:
            未来N日累计收益率序列（前复权）。
        """
        close = data["close"]
        # 按股票分组计算未来N日收益
        fwd_close = close.groupby(data["code"]).shift(-forward_period)
        returns = (fwd_close - close) / close
        return returns

    def prepare_group_labels(
        self,
        data: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> pd.Series:
        """构造 LambdaRank 所需的分组排序标签。

        同一交易日内，按前向收益排名 → 标签值 = 排名（越大越好）。
        """
        df = pd.DataFrame({
            "trade_date": data["trade_date"],
            "fwd_return": forward_returns,
        })
        # 按日期分组，组内排名
        df["rank_label"] = df.groupby("trade_date")["fwd_return"].rank(
            ascending=True, method="first",
        )
        return df["rank_label"]

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> None:
        """训练 LambdaRank 排序模型。

        Args:
            X: 特征矩阵。
            y: 标签（前向收益或排名标签）。
            sample_weights: 样本权重。
            groups: 日期分组（必须提供——同一天的数据在同一query组）。
        """
        import lightgbm as lgb

        self._set_seed()
        self._feature_names = list(X.columns)

        params = dict(self._params)
        params["random_state"] = self._seed
        params["verbosity"] = -1

        # LightGBM LambdaRank requires integer labels. Auto-detect:
        y_int_ok = y.dropna().apply(lambda v: v == int(v)).all() if len(y.dropna()) > 0 else False

        use_lambdarank = (groups is not None) and y_int_ok
        if use_lambdarank:
            params["objective"] = "lambdarank"
            params["metric"] = "ndcg"
            X_vals = X.values
            y_vals = y.values.astype(int)
            train_data = lgb.Dataset(
                X_vals, label=y_vals,
                group=groups.value_counts(sort=False).values,
                weight=sample_weights.values if sample_weights is not None else None,
            )
        else:
            if groups is not None and not y_int_ok:
                logger.info("LongTermRanker: labels非整数，使用回归模式")
            else:
                logger.info("LongTermRanker: 使用回归模式")
            params["objective"] = "regression"
            params["metric"] = "rmse"
            train_data = lgb.Dataset(
                X.values, label=y.values.astype(float),
                weight=sample_weights.values if sample_weights is not None else None,
            )

        self._model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data],
        )
        self._trained = True
        logger.info("LongTermRanker 训练完成: samples=%d, features=%d", len(X), len(self._feature_names))

    def predict_ranks(self, X: pd.DataFrame) -> pd.Series:
        """返回排序分数（越高越好）。

        Returns:
            与输入 index 对齐的 Series。
        """
        preds = self.predict(X)
        return pd.Series(preds.flatten(), index=X.index, name="score")

    def get_params(self) -> dict:
        base = super().get_params()
        base["lgbm_params"] = self._params
        return base


# ── LightGBM 二分类模型 ──────────────────────────

class LongTermClassifier(LightGBMBaseModel):
    """LightGBM 二分类：预测未来N日是否跑赢基准（中证800）。"""

    def __init__(
        self,
        config_path: str = "configs/models/long_term.yaml",
        model_name: str = "long_term_classifier",
    ) -> None:
        super().__init__(model_name=model_name)
        self._config = LongTermRanker._load_config(config_path)
        self._params = self._config.get("params", {})
        self._seed = self._params.get("random_state", 42)

    def prepare_labels(
        self,
        data: pd.DataFrame,
        forward_period: int = 20,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """构造二分类标签：跑赢基准=1，跑输=0。

        Args:
            data: 行情 DataFrame（含 code, close）。
            forward_period: 前向窗口。
            benchmark_returns: 基准（中证800）同期收益率，None 则用全市场中位数。

        Returns:
            0/1 分类标签。
        """
        close = data["close"]
        fwd_close = close.groupby(data["code"]).shift(-forward_period)
        stock_return = (fwd_close - close) / close

        if benchmark_returns is not None:
            median_return = benchmark_returns
        else:
            median_return = stock_return.groupby(data["trade_date"]).transform("median")

        return (stock_return > median_return).astype(int)

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
        params["objective"] = "binary"
        params["metric"] = "auc"
        params["random_state"] = self._seed

        train_data = lgb.Dataset(
            X.values, label=y.values,
            weight=sample_weights.values if sample_weights is not None else None,
        )
        self._model = lgb.train(params, train_data, valid_sets=[train_data])
        self._trained = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回跑赢基准的概率 (n_samples, 1)。"""
        return self.predict(X)


# ── 多因子线性基线 ──────────────────────────────

class LinearBaseline(BaseModel):
    """OLS多因子线性基线——作为非线性模型的最小可解释基线。

    用法：先跑通线性，再上 LightGBM。如果线性模型已经能给出
    不错的 IC，说明因子本身有信号；LightGBM 的增量来自非线性组合。
    """

    def __init__(self, model_name: str = "linear_baseline") -> None:
        super().__init__(model_name=model_name)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        from sklearn.linear_model import LinearRegression

        self._set_seed()
        self._feature_names = list(X.columns)

        # 去NaN
        mask = y.notna() & X.notna().all(axis=1)
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]

        weights = sample_weights.loc[mask].values if sample_weights is not None else None

        self._model = LinearRegression()
        self._model.fit(X_clean, y_clean, sample_weight=weights)  # type: ignore[call-arg]
        self._trained = True
        logger.info("LinearBaseline 训练完成: samples=%d", len(X_clean))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_trained()
        return self._model.predict(X.values).reshape(-1, 1)  # type: ignore[union-attr]

    def _check_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("线性模型尚未训练")

    def coefficients(self) -> pd.DataFrame:
        """返回因子权重（系数）。"""
        self._check_trained()
        return pd.DataFrame({
            "factor": self._feature_names,
            "coefficient": self._model.coef_.flatten(),  # type: ignore[union-attr]
        }).sort_values("coefficient", ascending=False)
