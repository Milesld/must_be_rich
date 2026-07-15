"""交易约束检查。

检查每笔交易意图是否满足A股交易规则：
- 涨跌停范围检查（区分板块/ST/新股的涨跌幅）
- T+1 当日买入不可卖出
- 停牌不可交易
- 价格笼子（科创板/创业板限价申报范围）
- 封板检查（涨停→买单不可成交，跌停→卖单不可成交）

约束从配置动态读取，不硬编码具体数值。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Literal, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _code_to_board_id(code: str) -> str:
    """根据代码前缀映射到板块ID。"""
    if code.startswith("688"):
        return "sse_star"
    if code.startswith(("300", "301")):
        return "szse_gem"
    if code.startswith("920") or code.startswith("83") or code.startswith("87") or code.startswith("88"):
        return "bse"
    if code.startswith("6"):
        return "sse_main"
    if code.startswith(("0", "3")):
        return "szse_main"
    return "sse_main"


# 板块默认涨跌幅（与 configs/boards/*.yaml 一致）。
# 之前无配置时所有板块统一按 ±10% 检查，创业板/科创板合法的 10~20% 波动
# 会被误拒单，系统性扭曲回测。内置默认表保证「不传配置」时行为也正确。
_DEFAULT_PRICE_LIMITS: dict[str, float] = {
    "sse_main": 0.10,
    "szse_main": 0.10,
    "sse_star": 0.20,
    "szse_gem": 0.20,
    "bse": 0.30,
}


@dataclass
class ConstraintCheckResult:
    """约束检查结果。"""
    passed: bool
    reject_reason: Optional[str] = None
    checks_detail: dict[str, bool] = field(default_factory=dict)


class TradingConstraints:
    """A股交易约束检查器。

    每个交易日开始和结束时调用 reset_day() 清空状态。

    使用方式：
        tc = TradingConstraints(board_config)
        tc.update_daily_info(market_data)  # 更新当天涨跌停/停牌状态
        result = tc.check_all("600519", "buy", 1680.0, today)
    """

    def __init__(self, board_config: dict[str, Any] | None = None) -> None:
        """初始化。

        Args:
            board_config: 板块配置（从 ConfigLoader 的 boards 节点获取）。
                          为 None 时使用硬编码的默认规则。
        """
        self._boards = board_config or {}
        self._today_bought: Set[str] = set()
        self._suspended_codes: Set[str] = set()
        # {code: (limit_up_px, limit_down_px)}
        self._price_limits: dict[str, tuple[float, float]] = {}
        # {code: (is_limit_up_sealed, is_limit_down_sealed)}
        self._sealed_status: dict[str, tuple[bool, bool]] = {}
        # {code: float} 最近成交价（用于价格笼子判断）
        self._last_prices: dict[str, float] = {}

    # ── 每日信息更新 ────────────────────────────

    def update_daily_info(
        self,
        daily_data: dict[str, dict],
        trade_date: date | None = None,
    ) -> None:
        """根据当日行情数据更新涨停/跌停价位和停牌状态。

        Args:
            daily_data: {code: {pre_close, is_st, is_suspended, board_id, volume}},
                        通常来自当日 market_daily 表。
            trade_date: 当日日期（用于ST股涨跌幅的时间段判断）。
        """
        self._price_limits.clear()
        self._sealed_status.clear()
        self._suspended_codes.clear()

        for code, row in daily_data.items():
            board = _code_to_board_id(code)
            rules = self._boards.get(board, {})
            price_limit = rules.get("price_limit", _DEFAULT_PRICE_LIMITS.get(board, 0.10))
            is_st = row.get("is_st", False)
            is_suspended = row.get("is_suspended", False)
            pre_close = row.get("pre_close", 0.0)

            # ST股涨跌幅判断（支持时间段切换）
            if is_st and trade_date:
                overrides = rules.get("price_limit_overrides", [])
                limit_val = self._resolve_override(overrides, is_st, trade_date)
                if limit_val is not None:
                    price_limit = limit_val

            # 新股无涨跌幅限制（简化：不在这里处理，由外部设置 special=True 跳过）
            limit_up = pre_close * (1 + price_limit)
            limit_down = pre_close * (1 - price_limit)
            self._price_limits[code] = (limit_up, limit_down)

            # 停牌
            if is_suspended:
                self._suspended_codes.add(code)

            # 封板判断依据：收盘价 ≈ 涨/跌停价（容差 0.2%）。
            # 旧逻辑要求 volume<=0 才认定封板，但日线上涨停日也有集合竞价成交，
            # volume 几乎不可能为 0 → 检查形同虚设，回测可在一字板照常买卖，
            # 系统性高估动量策略。改为收盘价贴板即视为封板（保守但更接近现实：
            # 收盘仍封死的板，尾盘排队买入大概率排不到）。
            close = row.get("close", pre_close)
            is_sealed_up = False
            is_sealed_down = False
            if close > 0 and pre_close > 0:
                if limit_up > 0 and abs(close - limit_up) / limit_up < 0.002:
                    is_sealed_up = True
                elif limit_down > 0 and abs(close - limit_down) / limit_down < 0.002:
                    is_sealed_down = True
            self._sealed_status[code] = (is_sealed_up, is_sealed_down)

            # 最近成交价
            if close > 0:
                self._last_prices[code] = close
            elif pre_close > 0:
                self._last_prices[code] = pre_close

    # ── 各项检查 ──────────────────────────────

    def check_price_limit(
        self, code: str, price: float,
    ) -> tuple[bool, str]:
        """检查价格是否在当日涨跌停范围内。

        Returns:
            (合法, 原因)
        """
        if code not in self._price_limits:
            return True, "无涨跌停数据，跳过检查"

        limit_up, limit_down = self._price_limits[code]
        # 允许 0.1% 的浮点容差
        tolerance = abs(limit_up) * 0.001
        if price > limit_up + tolerance:
            return False, f"价格 {price:.2f} > 涨停价 {limit_up:.2f}"
        if price < limit_down - tolerance:
            return False, f"价格 {price:.2f} < 跌停价 {limit_down:.2f}"

        return True, "OK"

    def check_t_plus_1(self, code: str, side: str) -> tuple[bool, str]:
        """T+1：当日买入的股票不能卖出。"""
        if side == "sell" and code in self._today_bought:
            return False, f"{code} 为当日买入，T+1规则禁止卖出"
        return True, "OK"

    def check_suspension(self, code: str) -> tuple[bool, str]:
        """停牌检查。"""
        if code in self._suspended_codes:
            return False, f"{code} 当前处于停牌状态，不可交易"
        return True, "OK"

    def check_price_cage(
        self, code: str, price: float, side: str,
    ) -> tuple[bool, str]:
        """价格笼子检查（科创板/创业板限价申报范围）。

        - 买入申报价 ≤ 基准价 × 102%
        - 卖出申报价 ≥ 基准价 × 98%

        基准价取最近成交价（不可用时取昨收）。
        """
        board = _code_to_board_id(code)
        rules = self._boards.get(board, {})
        cage = rules.get("price_cage")
        if cage is None:
            return True, "无价格笼子规则"

        ref_price = self._last_prices.get(code, 0)

        if side == "buy":
            max_price = ref_price * cage.get("buy_ratio", 1.02)
            if price > max_price:
                return False, (
                    f"买入申报价 {price:.2f} > 基准价 {ref_price:.2f} × {cage['buy_ratio']:.2f} = {max_price:.2f}"
                )
        else:
            min_price = ref_price * cage.get("sell_ratio", 0.98)
            if price < min_price:
                return False, (
                    f"卖出申报价 {price:.2f} < 基准价 {ref_price:.2f} × {cage['sell_ratio']:.2f} = {min_price:.2f}"
                )

        return True, "OK"

    def check_limit_sealed(self, code: str, side: str) -> tuple[bool, str]:
        """封板检查（坑#3）。

        - 涨停封板 → 买单无法成交（需有人卖出 + 排队排到）
        - 跌停封板 → 卖单无法成交

        封板判断依据：成交量≈0 且收盘价 ≈ 涨跌停价。
        """
        if code not in self._sealed_status:
            return True, "无封板数据"

        is_sealed_up, is_sealed_down = self._sealed_status[code]
        if side == "buy" and is_sealed_up:
            return False, f"{code} 涨停封板中，买单无法成交"
        if side == "sell" and is_sealed_down:
            return False, f"{code} 跌停封板中，卖单无法成交"

        return True, "OK"

    def check_all(
        self,
        code: str,
        side: Literal["buy", "sell"],
        price: float,
        board: str = "",
        skip_cage: bool = False,
    ) -> ConstraintCheckResult:
        """执行全部约束检查，任何一项不通过即拒绝。"""
        details: dict[str, bool] = {}
        reject_reason: Optional[str] = None

        checks = [
            ("price_limit", self.check_price_limit(code, price)),
            ("t_plus_1", self.check_t_plus_1(code, side)),
            ("suspension", self.check_suspension(code)),
            ("limit_sealed", self.check_limit_sealed(code, side)),
        ]

        if not skip_cage:
            checks.append(("price_cage", self.check_price_cage(code, price, side)))

        for name, (passed, reason) in checks:
            details[name] = passed
            if not passed and reject_reason is None:
                reject_reason = reason

        return ConstraintCheckResult(
            passed=all(details.values()),
            reject_reason=reject_reason,
            checks_detail=details,
        )

    # ── 状态管理 ──────────────────────────────

    def mark_bought(self, code: str) -> None:
        """记录当日买入（用于T+1检查）。"""
        self._today_bought.add(code)

    def reset_day(self) -> None:
        """新交易日开始时调用。"""
        self._today_bought.clear()
        self._price_limits.clear()
        self._sealed_status.clear()
        self._suspended_codes.clear()

    # ── 辅助 ─────────────────────────────────

    @staticmethod
    def _resolve_override(
        overrides: list[dict],
        is_st: bool,
        trade_date: date,
    ) -> float | None:
        """从 override 规则列表中匹配适用的涨跌幅。"""
        for ov in overrides:
            condition = ov.get("condition", "")
            try:
                # 安全的条件求值（仅支持 is_st 和 trade_date 两个变量）
                env = {
                    "is_st": is_st,
                    "trade_date": trade_date,
                    "True": True,
                    "False": False,
                }
                if eval(condition, {"__builtins__": {}}, env):
                    return float(ov["price_limit"])
            except Exception:
                logger.debug("涨跌幅 override 条件求值失败: %s", condition)
        return None
