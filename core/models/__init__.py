"""模型训练与推理 — LightGBM/CatBoost排序与分类、NLP情绪分析。

子模块：
- base: 模型抽象基类 (BaseModel, LightGBMBaseModel)
- long_term: 长期选股模型 (LongTermRanker, LongTermClassifier, LinearBaseline)
- intraday: 日内预测模型 (IntradayClassifier, IntradayQuantileRegressor)
- premarket/: 盘前推荐模型组
  · overnight_mapping: 隔夜海外→A股开盘方向映射
  · gap_classifier: 开盘跳空三分类 + Platt校准
  · fusion_ranker: 长期+隔夜融合排序
  · auction_anomaly: 竞价异常检测 (IsolationForest+规则)
- nlp: NLP情绪分析 (关键词规则 + 可选Transformer升级)
- evaluation: 模型评估工具 (IC/ICIR/分层收益/校准/衰减监控)
"""

from core.models.base import BaseModel, LightGBMBaseModel
from core.models.long_term import LongTermRanker, LongTermClassifier, LinearBaseline
from core.models.intraday import IntradayClassifier, IntradayQuantileRegressor
from core.models.nlp import NLPSentimentAnalyzer, SentimentResult
from core.models.evaluation import ModelEvaluator

__all__ = [
    "BaseModel",
    "LightGBMBaseModel",
    "LongTermRanker",
    "LongTermClassifier",
    "LinearBaseline",
    "IntradayClassifier",
    "IntradayQuantileRegressor",
    "NLPSentimentAnalyzer",
    "SentimentResult",
    "ModelEvaluator",
]
