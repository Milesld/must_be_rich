"""组合管理 — 市场状态识别、组合优化、凯利公式、退出监控。

子模块：
- regime: MarketRegimeDetector, RegimeResult
- optimizer: PortfolioOptimizer（等权/信号加权/波动率倒数/风险平价/最小CVaR）
- kelly: KellyCalculator（连续收益版 + 离散版）
- exit_monitor: ExitMonitor, ExitSignal, ExitAlert
"""

from core.portfolio.regime import MarketRegimeDetector, RegimeResult
from core.portfolio.optimizer import PortfolioOptimizer
from core.portfolio.kelly import KellyCalculator
from core.portfolio.exit_monitor import ExitMonitor, ExitSignal, ExitAlert

__all__ = [
    "MarketRegimeDetector",
    "RegimeResult",
    "PortfolioOptimizer",
    "KellyCalculator",
    "ExitMonitor",
    "ExitSignal",
    "ExitAlert",
]
