"""事件驱动回测引擎。

按A股交易日历逐日推进，每天按固定事件顺序处理：
开盘前加载数据 → 盘中处理策略信号 → 收盘前T+1检查 → 收盘后记录快照。

支持：
- 注入自定义撮合模拟器（默认日线收盘价撮合）
- 暂停/继续（通过 JSON 状态文件）
- signal_id → trade_id 完整链路追溯
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from core.backtest.cost_model import CostBreakdown, TransactionCostModel, _to_d, to_decimal
from core.backtest.constraints import ConstraintCheckResult, TradingConstraints
from core.backtest.rounding import ShareRounder
from core.common.calendar import get_calendar

logger = logging.getLogger(__name__)


# ── 数据类 ─────────────────────────────────────

@dataclass
class TradeIntent:
    """策略产生的交易意图。"""
    signal_id: str
    code: str
    side: Literal["buy", "sell"]
    price: float                    # 意图价格
    shares: int                     # 意图股数（已整手化）
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class TradeRecord:
    """一笔实际成交记录。"""
    trade_id: str
    signal_id: str
    code: str
    side: str
    trade_date: date
    intent_price: float
    intent_shares: int
    filled_price: float
    filled_shares: int
    cost_breakdown: CostBreakdown | None = None
    status: str = "filled"    # filled / partial / rejected / unfilled
    reject_reason: Optional[str] = None
    realized_pnl: float = 0.0  # 卖出时的已实现盈亏（买入时为0）

    def to_dict(self) -> dict:
        cb = self.cost_breakdown
        return {
            "trade_id": self.trade_id,
            "signal_id": self.signal_id,
            "code": self.code,
            "side": self.side,
            "trade_date": self.trade_date.isoformat(),
            "intent_price": self.intent_price,
            "intent_shares": self.intent_shares,
            "filled_price": self.filled_price,
            "filled_shares": self.filled_shares,
            "cost_breakdown": cb.to_dict() if cb else None,
            "status": self.status,
            "reject_reason": self.reject_reason,
        }


@dataclass
class DailySnapshot:
    """每日快照。"""
    date: date
    nav: float                        # 当日净值
    cash: float                       # 可用现金
    positions: dict[str, int]         # {code: shares}
    position_values: dict[str, float] # {code: market_value}
    total_value: float                # 总资产
    daily_return: float               # 日收益率
    trades_today: list[TradeRecord] = field(default_factory=list)


# ── 策略协议 ──────────────────────────────────

class Strategy(Protocol):
    """策略回调协议。"""

    def on_bar(
        self,
        trade_date: date,
        features: pd.DataFrame,
        positions: dict[str, int],
        cash: float,
        daily_data: dict[str, dict],
    ) -> list[TradeIntent]:
        """每个交易日调用，返回交易意图列表。"""
        ...


# ── 撮合模拟器协议 ────────────────────────────

class FillSimulator(Protocol):
    """撮合模拟器：根据行情数据决定成交价和成交量。"""

    def simulate(
        self,
        intent: TradeIntent,
        daily_data: dict[str, dict],
    ) -> tuple[float, int]:   # (成交价, 成交股数)
        ...


class ClosePriceFillSimulator:
    """默认撮合器：以当日收盘价成交（无法成交则返回 0 股）。"""

    def simulate(
        self,
        intent: TradeIntent,
        daily_data: dict[str, dict],
    ) -> tuple[float, int]:
        row = daily_data.get(intent.code, {})
        close = float(row.get("close", 0))
        if close <= 0:
            return 0.0, 0
        # 收盘价撮合：全部以收盘价成交
        return close, intent.shares


class OHLCFillSimulator:
    """以当日均价成交（更接近实际），可选日内最高/最低限制。"""

    def simulate(
        self,
        intent: TradeIntent,
        daily_data: dict[str, dict],
    ) -> tuple[float, int]:
        row = daily_data.get(intent.code, {})
        open_p = float(row.get("open", 0))
        high = float(row.get("high", 0))
        low = float(row.get("low", 0))
        close = float(row.get("close", 0))
        vwap = float(row.get("amount", 0)) / float(row.get("volume", 1)) if row.get("volume", 0) > 0 else close

        if close <= 0:
            return 0.0, 0

        # 用 VWAP 作为成交价近似，受 high/low 约束
        fill_price = vwap if vwap > 0 else close
        fill_price = max(low, min(fill_price, high))
        return fill_price, intent.shares


# ── 回测引擎 ────────────────────────────────────

class BacktestEngine:
    """A股事件驱动回测引擎。

    使用示例：
        engine = BacktestEngine(
            start_date=date(2020,1,1),
            end_date=date(2025,12,31),
            initial_capital=1_000_000,
            config=config,
        )
        result = engine.run(strategy)
    """

    def __init__(
        self,
        start_date: date,
        end_date: date,
        initial_capital: float,
        config: Any = None,
        fill_simulator: FillSimulator | None = None,
        commission_rate: float = 0.00015,
        stamp_duty_rate: float = 0.0005,
    ) -> None:
        """初始化回测引擎。

        Args:
            start_date: 回测起始日。
            end_date: 回测截止日。
            initial_capital: 初始资金。
            config: ConfigLoader 实例（用于加载板块/风控规则）。
            fill_simulator: 自定义撮合模拟器（默认 ClosePriceFillSimulator）。
            commission_rate: 佣金率（覆盖系统默认值）。
            stamp_duty_rate: 印花税率（覆盖系统默认值）。
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.config = config

        # 子系统
        board_cfg = getattr(config, "boards", None) if config else None
        self.cost_model = TransactionCostModel()
        self.constraints = TradingConstraints(board_cfg)
        self.rounder = ShareRounder(board_cfg)
        self.fill_simulator = fill_simulator or ClosePriceFillSimulator()

        # 费率覆盖
        self._commission_rate = commission_rate
        self._stamp_duty_rate = stamp_duty_rate

        # 运行时状态
        self.cash: float = initial_capital
        self.positions: dict[str, int] = {}
        self.position_costs: dict[str, float] = {}  # {code: avg_cost}
        self.trades: list[TradeRecord] = []
        self.daily_snapshots: list[DailySnapshot] = []
        self.peak_nav: float = initial_capital

        # 回测可复现性
        self._run_id: str = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # ── 主循环 ──────────────────────────────

    def run(
        self,
        strategy: Strategy,
        data_loader: Callable[[date], pd.DataFrame] | None = None,
        feature_loader: Callable[[date, list[str]], pd.DataFrame] | None = None,
        progress_callback: Callable[[date, int, int], None] | None = None,
    ) -> "BacktestResult":
        """执行回测主循环。

        Args:
            strategy: 实现了 on_bar() 协议的策略对象。
            data_loader: 数据加载回调 f(date) → 当日行情 DataFrame。
                         为 None 时使用 engine 内置的数据获取（需配合 data_provider）。
            feature_loader: 因子加载回调 f(date, codes) → 因子 DataFrame。
            progress_callback: 进度回调 f(current_date, day_index, total_days)。

        Returns:
            BacktestResult 包含完整回测记录和分析方法。
        """
        cal = get_calendar()
        trading_days = cal.get_trading_days(self.start_date, self.end_date)
        total = len(trading_days)

        if total == 0:
            logger.warning("回测区间内无交易日: %s ~ %s", self.start_date, self.end_date)
            return BacktestResult(
                run_id=self._run_id,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.initial_capital,
                daily_snapshots=[],
                trades=[],
            )

        for i, td in enumerate(trading_days):
            # ── 开盘前 ──────────────────────
            daily_data = self._load_daily_data(td, data_loader)
            if not daily_data:
                continue

            self.constraints.update_daily_info(daily_data, trade_date=td)

            # mtm 持仓
            positions_mtm = self._mtm_positions(daily_data)

            # 加载因子
            features = pd.DataFrame()
            if feature_loader is not None:
                # 传递全市场股票代码（不仅是持仓的），使策略能在调仓日评估整个股票池
                universe_codes = list(self.positions.keys()) + [c for c in daily_data.keys() if c not in self.positions]
                features = feature_loader(td, universe_codes)

            # ── 盘中 ────────────────────────
            trade_intents = strategy.on_bar(
                trade_date=td,
                features=features,
                positions=dict(self.positions),
                cash=self.cash,
                daily_data=daily_data,
            )

            trades_today: list[TradeRecord] = []
            for intent in trade_intents:
                record = self._handle_trade(intent, daily_data, td)
                trades_today.append(record)

                if record.status == "filled" and record.filled_shares > 0:
                    self._apply_fill(record)
                    if record.side == "buy":
                        self.constraints.mark_bought(record.code)

            # ── 收盘前 T+1 检查 ──────────────
            # (T+1约束已在 check_t_plus_1 中处理，买入记录在 _apply_fill 后 mark_bought)

            # ── 收盘后 ──────────────────────
            nav = self._calc_nav(daily_data)
            day_return = 0.0
            if self.daily_snapshots:
                prev_nav = self.daily_snapshots[-1].nav
                day_return = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0

            snapshot = DailySnapshot(
                date=td,
                nav=nav,
                cash=self.cash,
                positions=dict(self.positions),
                position_values=positions_mtm,
                total_value=nav,
                daily_return=day_return,
                trades_today=trades_today,
            )
            self.daily_snapshots.append(snapshot)

            # 回撤熔断检查
            if nav > self.peak_nav:
                self.peak_nav = nav
            drawdown = (self.peak_nav - nav) / self.peak_nav if self.peak_nav > 0 else 0.0
            self._check_circuit_breaker(td, drawdown, strategy)

            # 新交易日
            self.constraints.reset_day()

            if progress_callback:
                progress_callback(td, i + 1, total)

        return BacktestResult(
            run_id=self._run_id,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            daily_snapshots=self.daily_snapshots,
            trades=self.trades,
        )

    # ── 单笔交易处理 ────────────────────────

    def _handle_trade(
        self,
        intent: TradeIntent,
        daily_data: dict[str, dict],
        trade_date: date,
    ) -> TradeRecord:
        """处理单笔交易意图：约束→整手→撮合→成本→记录。"""
        trade_id = f"trade_{trade_date.isoformat()}_{uuid.uuid4().hex[:8]}"

        # 1. 约束检查
        constraint_result = self.constraints.check_all(
            intent.code, intent.side, intent.price,
        )
        if not constraint_result.passed:
            return TradeRecord(
                trade_id=trade_id,
                signal_id=intent.signal_id,
                code=intent.code,
                side=intent.side,
                trade_date=trade_date,
                intent_price=intent.price,
                intent_shares=intent.shares,
                filled_price=0.0,
                filled_shares=0,
                status="rejected",
                reject_reason=constraint_result.reject_reason,
            )

        # 2. 整手化
        rounded_shares = self.rounder.round_down(intent.code, intent.shares)
        if rounded_shares <= 0:
            return TradeRecord(
                trade_id=trade_id,
                signal_id=intent.signal_id,
                code=intent.code,
                side=intent.side,
                trade_date=trade_date,
                intent_price=intent.price,
                intent_shares=intent.shares,
                filled_price=0.0,
                filled_shares=0,
                status="rejected",
                reject_reason=f"整手化后股数为0（理想{intent.shares}股）",
            )

        # 3. 撮合
        fill_intent = TradeIntent(
            signal_id=intent.signal_id,
            code=intent.code,
            side=intent.side,
            price=intent.price,
            shares=rounded_shares,
            confidence=intent.confidence,
        )
        fill_price, fill_shares = self.fill_simulator.simulate(fill_intent, daily_data)

        if fill_shares <= 0 or fill_price <= 0:
            return TradeRecord(
                trade_id=trade_id,
                signal_id=intent.signal_id,
                code=intent.code,
                side=intent.side,
                trade_date=trade_date,
                intent_price=intent.price,
                intent_shares=rounded_shares,
                filled_price=0.0,
                filled_shares=0,
                status="unfilled",
                reject_reason="撮合未成交（无流动性或停牌）",
            )

        # 4. 成本计算
        cost = self.cost_model.calculate(intent.code, intent.side, fill_price, fill_shares)

        status = "filled" if fill_shares == rounded_shares else "partial"

        return TradeRecord(
            trade_id=trade_id,
            signal_id=intent.signal_id,
            code=intent.code,
            side=intent.side,
            trade_date=trade_date,
            intent_price=intent.price,
            intent_shares=rounded_shares,
            filled_price=fill_price,
            filled_shares=fill_shares,
            cost_breakdown=cost,
            status=status,
        )

    # ── 成交执行 ────────────────────────────

    def _apply_fill(self, record: TradeRecord) -> None:
        """将成交记录应用到持仓和现金。"""
        cb = record.cost_breakdown
        total_cost = float(cb.total) if cb else 0.0
        amount = record.filled_price * record.filled_shares

        if record.side == "buy":
            # 扣除现金（成交金额 + 佣金 + 过户费，印花税仅在卖方）
            self.cash -= (amount + total_cost)
            # 更新持仓
            old_shares = self.positions.get(record.code, 0)
            old_cost = self.position_costs.get(record.code, 0.0)
            new_shares = old_shares + record.filled_shares
            # 移动平均成本
            if new_shares > 0:
                new_avg_cost = (
                    (old_cost * old_shares + amount + total_cost) / new_shares
                    if old_shares > 0 else (amount + total_cost) / new_shares
                )
            else:
                new_avg_cost = 0.0
            self.positions[record.code] = new_shares
            self.position_costs[record.code] = new_avg_cost
        else:
            # 卖出：增加现金（成交金额 - 佣金 - 印花税 - 过户费）
            self.cash += (amount - total_cost)
            old_shares = self.positions.get(record.code, 0)
            avg_cost = self.position_costs.get(record.code, 0.0)
            # 计算已实现盈亏 = 卖出净收入 - 卖出股数对应的成本
            record.realized_pnl = (amount - total_cost) - (avg_cost * record.filled_shares)
            new_shares = old_shares - record.filled_shares
            if new_shares <= 0:
                self.positions.pop(record.code, None)
                self.position_costs.pop(record.code, None)
            else:
                self.positions[record.code] = new_shares

        self.trades.append(record)

    # ── 辅助方法 ──────────────────────────────

    def _load_daily_data(
        self,
        trade_date: date,
        loader: Callable | None,
    ) -> dict[str, dict]:
        """加载当日行情数据，转换为 {code: {field: value}} 格式。"""
        if loader is None:
            return {}

        try:
            df = loader(trade_date)
            if df is None or len(df) == 0:
                return {}

            result: dict[str, dict] = {}
            for _, row in df.iterrows():
                code = str(row.get("code", ""))
                if not code:
                    continue
                result[code] = {
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "pre_close": float(row.get("pre_close", 0)),
                    "volume": float(row.get("volume", 0)),
                    "amount": float(row.get("amount", 0)),
                    "is_st": bool(row.get("is_st", False)),
                    "is_suspended": bool(row.get("is_suspended", False)),
                }
            return result
        except Exception as e:
            logger.warning("加载 %s 数据失败: %s", trade_date, e)
            return {}

    def _mtm_positions(self, daily_data: dict[str, dict]) -> dict[str, float]:
        """按市价估值持仓。"""
        values: dict[str, float] = {}
        for code, shares in self.positions.items():
            row = daily_data.get(code, {})
            close = float(row.get("close", 0))
            if close <= 0:
                close = self.position_costs.get(code, 0.0)
            values[code] = close * shares
        return values

    def _calc_nav(self, daily_data: dict[str, dict]) -> float:
        """计算当日净值 = 现金 + 持仓市值。"""
        mtm = self._mtm_positions(daily_data)
        return self.cash + sum(mtm.values())

    def _check_circuit_breaker(
        self, td: date, drawdown: float, strategy: Strategy,
    ) -> None:
        """回撤熔断检查。超过阈值时记录告警但不中断回测。"""
        # 从配置读取阈值
        threshold = 0.25  # 默认25%回撤熔断
        if self.config:
            try:
                threshold = float(getattr(
                    getattr(getattr(self.config, "risk", None), "thresholds", None),
                    "portfolio", {},
                ).get("drawdown_rolling_circuit", 0.25))
            except Exception:
                pass

        if drawdown > threshold:
            logger.warning(
                "[%s] 回撤 %.2f%% 超过熔断线 %.1f%%。实盘中应触发强制减仓。",
                td, drawdown * 100, threshold * 100,
            )

    # ── 暂停/继续 ────────────────────────────

    def save_state(self, path: str) -> None:
        """保存回测状态到 JSON 文件（支持暂停/继续）。"""
        state = {
            "run_id": self._run_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": self.positions,
            "position_costs": self.position_costs,
            "peak_nav": self.peak_nav,
            "trades_count": len(self.trades),
            "snapshots_count": len(self.daily_snapshots),
            "last_date": (
                self.daily_snapshots[-1].date.isoformat()
                if self.daily_snapshots else None
            ),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info("回测状态已保存到 %s", path)

    def load_state(self, path: str) -> bool:
        """从 JSON 文件恢复回测状态。"""
        try:
            with open(path) as f:
                state = json.load(f)
            self._run_id = state["run_id"]
            self.cash = state["cash"]
            self.positions = state["positions"]
            self.position_costs = state["position_costs"]
            self.peak_nav = state["peak_nav"]
            logger.info("回测状态已从 %s 恢复", path)
            return True
        except Exception as e:
            logger.error("恢复回测状态失败: %s", e)
            return False


# ── 回测结果 ──────────────────────────────────

class BacktestResult:
    """回测结果对象，包含完整记录和分析方法。"""

    def __init__(
        self,
        run_id: str,
        start_date: date,
        end_date: date,
        initial_capital: float,
        daily_snapshots: list[DailySnapshot],
        trades: list[TradeRecord],
    ) -> None:
        self.run_id = run_id
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.daily_snapshots = daily_snapshots
        self.trades = trades

    # ── DataFrame 导出 ──────────────────────

    @property
    def daily_nav(self) -> pd.DataFrame:
        """每日净值 DataFrame。"""
        if not self.daily_snapshots:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "date": s.date,
                "nav": s.nav,
                "cash": s.cash,
                "total_value": s.total_value,
                "daily_return": s.daily_return,
            }
            for s in self.daily_snapshots
        ]).set_index("date")

    @property
    def trade_records(self) -> pd.DataFrame:
        """所有成交记录的 DataFrame。"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    @property
    def signals_history(self) -> pd.DataFrame:
        """所有信号历史的 DataFrame。"""
        return self.trade_records[["signal_id", "code", "side", "trade_date", "status"]]

    # ── 汇总统计 ────────────────────────────

    def summary(self) -> dict:
        """计算回测核心指标。"""
        df = self.daily_nav
        if df.empty:
            return {"error": "无交易日数据"}

        nav_series = df["nav"]
        return_series = df["daily_return"].dropna()

        n_days = len(nav_series)
        total_return = (nav_series.iloc[-1] / self.initial_capital) - 1
        years = n_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        annual_vol = return_series.std() * np.sqrt(252) if len(return_series) > 1 else 0.0
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0.0

        # 最大回撤
        peak = nav_series.expanding().max()
        drawdown = (nav_series - peak) / peak
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin() if max_dd < 0 else None

        # 交易统计
        filled_trades = [t for t in self.trades if t.status == "filled"]
        buys = [t for t in filled_trades if t.side == "buy"]
        sells = [t for t in filled_trades if t.side == "sell"]

        # 盈亏比：卖出时计算已实现盈亏
        winning_sells = [t for t in sells if t.realized_pnl > 0]
        losing_sells = [t for t in sells if t.realized_pnl <= 0]
        win_count = len(winning_sells)
        total_closed = len(sells)
        win_rate = win_count / total_closed if total_closed > 0 else (None if total_closed == 0 else 0.0)

        total_realized_pnl = sum(t.realized_pnl for t in sells)

        total_trades = len(filled_trades)

        # 成本统计
        total_cost = sum(
            float(t.cost_breakdown.total) for t in filled_trades if t.cost_breakdown
        )

        return {
            "run_id": self.run_id,
            "period": f"{self.start_date} ~ {self.end_date}",
            "n_trading_days": n_days,
            "initial_capital": self.initial_capital,
            "final_nav": float(nav_series.iloc[-1]),
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "max_drawdown_date": str(max_dd_date) if max_dd_date else None,
            "calmar_ratio": float(annual_return / abs(max_dd)) if max_dd < 0 else 0.0,
            "win_rate": float(win_rate) if win_rate is not None else "N/A (无已平仓交易)",
            "total_trades": total_trades,
            "n_buys": len(buys),
            "n_sells": len(sells),
            "winning_sells": win_count,
            "losing_sells": len(losing_sells),
            "total_realized_pnl": float(total_realized_pnl),
            "total_cost": float(total_cost),
            "total_cost_pct": float(total_cost / self.initial_capital) if self.initial_capital > 0 else 0.0,
        }
