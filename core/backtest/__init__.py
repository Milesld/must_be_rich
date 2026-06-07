"""回测引擎 — 事件驱动回测、交易成本建模、约束检查、Walk-Forward验证。

核心模块：
- engine: BacktestEngine (事件驱动主循环), BacktestResult (结果分析)
- cost_model: TransactionCostModel (Decimal精度成本计算)
- constraints: TradingConstraints (A股交易规则约束检查)
- rounding: ShareRounder (迭代收敛整手化算法)
- walk_forward: WalkForwardValidator (滚动验证防过拟合)
- report: BacktestReport (指标+图表+Markdown报告)
"""

from core.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    DailySnapshot,
    Strategy,
    TradeIntent,
    TradeRecord,
    ClosePriceFillSimulator,
    OHLCFillSimulator,
)
from core.backtest.cost_model import (
    CostBreakdown,
    TransactionCostModel,
    to_decimal,
    from_decimal,
)
from core.backtest.constraints import (
    ConstraintCheckResult,
    TradingConstraints,
)
from core.backtest.rounding import ShareRounder
from core.backtest.walk_forward import WalkForwardValidator
from core.backtest.report import BacktestReport

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestResult",
    "DailySnapshot",
    "Strategy",
    "TradeIntent",
    "TradeRecord",
    "ClosePriceFillSimulator",
    "OHLCFillSimulator",
    # Cost
    "CostBreakdown",
    "TransactionCostModel",
    "to_decimal",
    "from_decimal",
    # Constraints
    "ConstraintCheckResult",
    "TradingConstraints",
    # Rounding
    "ShareRounder",
    # Walk-Forward
    "WalkForwardValidator",
    # Report
    "BacktestReport",
]
