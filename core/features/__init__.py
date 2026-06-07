"""特征工程 — 因子计算、注册、验证、存储、中性化。

子模块：
- registry: 因子注册与版本管理
- technical: 技术面因子 (14个)
- fundamental: 基本面因子 (17个, 全PIT模式)
- capital_flow: 资金面因子 (7个)
- sentiment: 情绪/事件因子 (8个)
- premarket: 盘前专属因子 (10个)
- neutralizer: 行业+市值中性化 + MAD缩尾
- validation: 缺失率/异常值/分布漂移检测
- time_travel_checker: ★前视偏差安全检查★
- store: 特征存储 (ClickHouse + Redis)
"""

from core.features.registry import FactorRegistry, FactorDefinition
from core.features.store import FeatureStore
from core.features.time_travel_checker import TimeTravelChecker, TimeTravelError, TimeTravelViolation
from core.features.neutralizer import (
    neutralize_by_industry,
    neutralize_by_market_cap,
    neutralize_industry_mcap,
    mad_winsorize,
)
from core.features.validation import (
    FactorMissingRateChecker,
    detect_outliers_mad,
    detect_distribution_drift,
    MissingRateReport,
    OutlierReport,
    DriftReport,
)

__all__ = [
    "FactorRegistry",
    "FactorDefinition",
    "FeatureStore",
    "TimeTravelChecker",
    "TimeTravelError",
    "TimeTravelViolation",
    "neutralize_by_industry",
    "neutralize_by_market_cap",
    "neutralize_industry_mcap",
    "mad_winsorize",
    "FactorMissingRateChecker",
    "detect_outliers_mad",
    "detect_distribution_drift",
    "MissingRateReport",
    "OutlierReport",
    "DriftReport",
]
