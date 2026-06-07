"""极端行情预案。

"卖不出去"是必然场景而非例外。当千股跌停/停牌潮/流动性枯竭
发生时，系统必须有清晰的降级路径：
1. 优先减可成交、流动性好的持仓
2. 无法离场时转用股指期货对冲
3. 预设最坏情形资金/保证金缓冲
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeIntent:
    """极端行情下的交易意图（简化版，与 backtest.engine.TradeIntent 兼容）。"""
    signal_id: str
    code: str
    side: str       # 'buy' / 'sell'
    price: float
    shares: int
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


class ExtremeScenarioHandler:
    """极端行情预案处理器。

    使用方式：
        handler = ExtremeScenarioHandler(config)
        # 熔断时
        intents = handler.on_circuit_breaker(positions, market_status)
        # 千股跌停时
        intents = handler.on_liquidity_crisis(positions, prices, liquidity)
    """

    # 股指期货合约规格
    _FUTURES_SPEC = {
        "IF": {"multiplier": 300, "margin_rate": 0.14, "notional_per_lot": 0},  # 动态计算
        "IC": {"multiplier": 200, "margin_rate": 0.14},
        "IM": {"multiplier": 200, "margin_rate": 0.15},
    }

    def __init__(self, config: dict | None = None) -> None:
        """初始化。

        Args:
            config: 配置（来自 risk/thresholds.yaml 的 extreme_scenario 节）。
        """
        cfg = config or {}
        self._max_single_day_loss = cfg.get("max_single_day_loss", 0.10)
        self._cb_action = cfg.get("circuit_breaker_action", "reduce_trading_first")
        self._hedge_priority: list[str] = cfg.get("hedge_instrument_priority", ["IF", "IC", "IM"])
        self._emergency_cash_ratio = cfg.get("emergency_cash_buffer_ratio", 0.05)

    # ── 熔断触发处理 ────────────────────────────

    def on_circuit_breaker(
        self,
        positions: dict[str, int],
        prices: dict[str, float],
        liquidity_scores: dict[str, float] | None = None,
        market_status: str = "",
    ) -> list[TradeIntent]:
        """回撤熔断触发时的处置。

        减仓优先级：
        1. 交易仓全部清仓（最优先）
        2. 核心仓减至30%（如果交易仓清完后仍然超回撤线）

        Args:
            positions: {code: shares} 当前持仓。
            prices: {code: latest_price}。
            liquidity_scores: {code: liquidity_score} 流动性评分
                              （Amihud 非流动性指标 × -1，越高越容易卖出）。
                              None 时按成交额排序。
            market_status: 市场状态描述。

        Returns:
            卖出 TradeIntent 列表（按优先级排序）。
        """
        if not positions:
            return []

        # 按流动性排序
        sorted_codes = self._rank_by_liquidity(positions, prices, liquidity_scores)

        intents: list[TradeIntent] = []
        for code in sorted_codes:
            shares = positions.get(code, 0)
            if shares <= 0:
                continue
            price = prices.get(code, 0.0)
            if price <= 0:
                continue

            intents.append(TradeIntent(
                signal_id=f"cb_{code}",
                code=code,
                side="sell",
                price=price,
                shares=shares,
                metadata={"scenario": "circuit_breaker"},
            ))

        logger.warning(
            "熔断处置: %d 只持仓 → %d 笔卖出意图 (市场状态: %s)",
            len(positions), len(intents), market_status or "未知",
        )
        return intents

    # ── 流动性危机处理 ──────────────────────────

    def on_liquidity_crisis(
        self,
        positions: dict[str, int],
        prices: dict[str, float],
        daily_data: dict[str, dict] | None = None,
    ) -> dict[str, Any]:
        """千股跌停/停牌潮处理。

        返回两个清单：
        - executable: 可卖出清单（能成交的优先卖）
        - unexecutable: 无法卖出的持仓 + 建议的股指期货对冲手数
        """
        if not positions:
            return {"executable": [], "unexecutable": [], "hedge_advice": {}}

        executable: list[TradeIntent] = []
        unexecutable: dict[str, tuple[int, str]] = {}  # {code: (shares, reason)}

        for code, shares in positions.items():
            if shares <= 0:
                continue
            price = prices.get(code, 0.0)

            # 检查能否成交
            row = daily_data.get(code, {}) if daily_data else {}
            is_limit_down = self._is_limit_down(row)
            is_suspended = row.get("is_suspended", False)
            has_volume = row.get("volume", 0) > 0

            if is_limit_down:
                unexecutable[code] = (shares, "跌停封板")
            elif is_suspended:
                unexecutable[code] = (shares, "停牌")
            elif not has_volume:
                unexecutable[code] = (shares, "零成交量")
            elif price <= 0:
                unexecutable[code] = (shares, "无有效价格")
            else:
                executable.append(TradeIntent(
                    signal_id=f"lc_{code}",
                    code=code,
                    side="sell",
                    price=price,
                    shares=shares,
                    metadata={"scenario": "liquidity_crisis"},
                ))

        # 对无法卖出的持仓计算对冲手数
        hedge_advice = self._calc_hedge_advice(unexecutable, prices)

        logger.warning(
            "流动性危机: 可卖出%d只, 无法卖出%d只 → 建议对冲: %s",
            len(executable), len(unexecutable), hedge_advice,
        )
        return {
            "executable": executable,
            "unexecutable": [
                {"code": c, "shares": s, "reason": r}
                for c, (s, r) in unexecutable.items()
            ],
            "hedge_advice": hedge_advice,
        }

    # ── 对冲计算 ──────────────────────────────

    def _calc_hedge_advice(
        self,
        unexecutable: dict[str, tuple[int, str]],
        prices: dict[str, float],
    ) -> dict[str, dict]:
        """计算股指期货对冲手数。

        简化算法：无法卖出的总市值 / 期货合约名义价值 = 所需手数。
        """
        total_unexec_value = 0.0
        for code, (shares, _) in unexecutable.items():
            total_unexec_value += shares * prices.get(code, 0.0)

        if total_unexec_value <= 0:
            return {}

        advice: dict[str, dict] = {}
        for fut in self._hedge_priority:
            spec = self._FUTURES_SPEC.get(fut, {})
            multiplier = spec.get("multiplier", 300)
            margin_rate = spec.get("margin_rate", 0.14)

            # 获取期货当前价格（简化：用指数点估算）
            notional_per_lot = multiplier * self._estimate_index_level(fut)
            if notional_per_lot <= 0:
                continue

            lots_float = total_unexec_value / notional_per_lot
            lots = max(1, int(lots_float))
            margin_required = lots * notional_per_lot * margin_rate

            advice[fut] = {
                "lots": lots,
                "notional_per_lot": notional_per_lot,
                "hedge_value": lots * notional_per_lot,
                "margin_required": margin_required,
                "unexec_value": total_unexec_value,
            }

            # 只推荐第一个可用的对冲工具
            break

        return advice

    @staticmethod
    def _estimate_index_level(future_code: str) -> float:
        """估算指数点位（简化版，实际应从行情获取）。"""
        estimates = {
            "IF": 4000.0,   # 沪深300
            "IC": 6000.0,   # 中证500
            "IM": 6500.0,   # 中证1000
        }
        return estimates.get(future_code, 0.0)

    @staticmethod
    def _is_limit_down(row: dict) -> bool:
        """判断是否跌停封板。"""
        close = row.get("close", 0.0)
        pre_close = row.get("pre_close", 0.0)
        volume = row.get("volume", 0)
        if pre_close <= 0:
            return False
        pct_change = (close - pre_close) / pre_close
        # 跌停且无量 → 封板
        return pct_change <= -0.098 and volume <= 0

    # ── 流动性排序 ────────────────────────────

    @staticmethod
    def _rank_by_liquidity(
        positions: dict[str, int],
        prices: dict[str, float],
        liquidity_scores: dict[str, float] | None,
    ) -> list[str]:
        """按流动性排序持仓：容易卖的排前面。"""
        if liquidity_scores:
            # liquidity_score 越高越好卖（Amihud负值→正值←→流动性好）
            return sorted(
                positions.keys(),
                key=lambda c: liquidity_scores.get(c, 0.0),
                reverse=True,
            )
        # fallback: 市值越大越容易卖
        return sorted(
            positions.keys(),
            key=lambda c: positions.get(c, 0) * prices.get(c, 0),
            reverse=True,
        )

    # ── 紧急现金缓冲 ──────────────────────────

    def calc_emergency_buffer(self, total_asset: float) -> float:
        """计算所需的紧急现金缓冲。"""
        return total_asset * self._emergency_cash_ratio

    def check_emergency_ready(self, cash: float, total_asset: float) -> tuple[bool, float]:
        """检查是否有足够的紧急现金缓冲。"""
        required = self.calc_emergency_buffer(total_asset)
        return cash >= required, required - cash if cash < required else 0.0
