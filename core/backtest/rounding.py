"""整手化算法。

A股不同板块有不同的最小交易单位和递增方式，且小资金下
某些高价股可能买不起。整手化是迭代收敛过程而非一次性取整。

板块规则（从 configs/boards/*.yaml 读取）：
- 主板(min_shares=100, increment=100)：必须是100的整数倍
- 创业板(min_shares=100, increment=100)：100的整数倍
- 科创板(min_shares=200, increment=1)：≥200股，超200后1股递增
- 北交所(min_shares=100, increment=100)：100的整数倍

坑#4：整手化是迭代收敛过程。5万元账户买茅台(1700元/股)：
理想权重3.7% → 1850元 → ~1股 → 1手=17万 → 买不起 → 砍掉
→ 资金重分配 → 重新计算剩余股票的股数 → 迭代至收敛。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 默认板块规则（无配置时使用）
_DEFAULT_BOARD_RULES: dict[str, dict] = {
    "sse_main":    {"min_shares": 100, "increment": 100},
    "szse_main":   {"min_shares": 100, "increment": 100},
    "sse_star":    {"min_shares": 200, "increment": 1},
    "szse_gem":    {"min_shares": 100, "increment": 100},
    "bse":         {"min_shares": 100, "increment": 100},
}

# 代码 → 板块映射
def _code_to_board(code: str) -> str:
    """根据代码前缀映射到板块ID。"""
    if code.startswith("688"):
        return "sse_star"
    if code.startswith("300") or code.startswith("301"):
        return "szse_gem"
    if code.startswith(("8", "4", "9")):
        # 北交所：8开头(原精选层)、920开头(新代码段)
        if code.startswith("920") or code.startswith("83") or code.startswith("87") or code.startswith("88"):
            return "bse"
    if code.startswith("6"):
        return "sse_main"
    if code.startswith(("0", "3")):
        return "szse_main"
    # 默认主板
    return "sse_main"


class ShareRounder:
    """A股整手化算法。

    使用方式：
        rounder = ShareRounder(board_config)
        shares = rounder.round_down("600519", 150)  # → 100
    """

    def __init__(self, board_config: dict | None = None) -> None:
        """初始化。

        Args:
            board_config: 板块规则配置（从 ConfigLoader 的 boards 节点获取）。
                          为 None 时使用默认规则。
        """
        self._rules: dict[str, dict] = {}

        if board_config is not None:
            for board_id, cfg in board_config.items():
                self._rules[board_id] = {
                    "min_shares": cfg.get("min_shares", 100),
                    "increment": cfg.get("increment", 100),
                }
        else:
            self._rules = _DEFAULT_BOARD_RULES.copy()

    # ── 单只整手化 ──────────────────────────

    def round_down(self, code: str, ideal_shares: float) -> int:
        """单只股票的整手化，按板块规则向下取整。

        Args:
            code: 股票代码。
            ideal_shares: 理想股数（浮点，如 1.5 股）。

        Returns:
            可执行的整数股数。买不起时返回 0。
        """
        board = _code_to_board(code)
        rules = self._rules.get(board, {"min_shares": 100, "increment": 100})
        min_sh = rules["min_shares"]
        inc = rules["increment"]

        shares = int(ideal_shares)

        # 达不到最小股数 → 买不起
        if shares < min_sh:
            return 0

        # 科创板：≥200股，超200后1股递增
        if inc == 1:
            return shares  # 直接保留，不需要取整

        # 主板/创业板/北交所：必须是 increment 的整数倍，且 ≥ min_shares
        # 向下取整到 increment 的倍数
        rounded = (shares // inc) * inc
        if rounded < min_sh:
            return 0
        return rounded

    def round_up(self, code: str, ideal_shares: float) -> int:
        """单只股票的整手化，向上取整（用于买入时确保足额）。"""
        board = _code_to_board(code)
        rules = self._rules.get(board, {"min_shares": 100, "increment": 100})
        min_sh = rules["min_shares"]
        inc = rules["increment"]

        shares = int(ideal_shares)
        if shares < min_sh:
            return min_sh  # 至少买一手

        if inc == 1:
            return shares

        rounded = ((shares + inc - 1) // inc) * inc
        return max(rounded, min_sh)

    # ── 组合级整手化（核心算法）─────────────────

    def round_portfolio(
        self,
        ideal_weights: dict[str, float],
        total_capital: float,
        current_prices: dict[str, float],
        min_single_amount: float = 5000.0,
        max_fee_ratio: float = 0.001,
        max_iterations: int = 10,
    ) -> tuple[dict[str, int], dict[str, float], int]:
        """组合级整手化——迭代收敛算法。

        流程：
        1. 理想金额 = 权重 × 总资金
        2. 理想股数 = 理想金额 / 价格
        3. round_down 取整
        4. 检查每只是否达标（最小配置额 + 佣金占比）
        5. 不达标 → 砍掉 → 资金重新分配给剩余股票 → 回到步骤1
        6. 迭代至收敛或达到最大迭代次数

        Args:
            ideal_weights: {code: weight} 理想权重（sum ≈ 1.0）。
            total_capital: 总可用资金。
            current_prices: {code: price} 当前市价。
            min_single_amount: 单只最小配置金额（默认5000元）。
            max_fee_ratio: 单笔佣金/金额的最大允许比例（默认0.1%）。
            max_iterations: 最大迭代次数。

        Returns:
            (整手持仓股数, 实际权重, 保留股票数量)
        """
        if not ideal_weights or total_capital <= 0:
            return {}, {}, 0

        active_codes = list(ideal_weights.keys())
        remaining_capital = total_capital
        final_shares: dict[str, int] = {}
        actual_weights: dict[str, float] = {}

        for iteration in range(max_iterations):
            if not active_codes:
                break

            # 归一化剩余股票的权重
            current_weights = {
                c: ideal_weights[c] / sum(ideal_weights[c] for c in active_codes)
                for c in active_codes
            }

            # 本轮计算
            round_shares: dict[str, int] = {}
            round_costs: dict[str, float] = {}
            kept: list[str] = []
            dropped: list[str] = []

            for code in active_codes:
                weight = current_weights[code]
                alloc = weight * total_capital
                price = current_prices.get(code, 1_000_000)

                ideal_s = alloc / price
                shares = self.round_down(code, ideal_s)

                if shares <= 0:
                    dropped.append(code)
                    continue

                # 检查最小配置额
                actual_amount = shares * price
                if actual_amount < min_single_amount:
                    dropped.append(code)
                    continue

                # 检查佣金占比（估算）
                fee_ratio = self._estimate_fee_ratio(code, shares, price)
                if fee_ratio > max_fee_ratio:
                    dropped.append(code)
                    continue

                kept.append(code)
                round_shares[code] = shares
                round_costs[code] = actual_amount

            # 如果没砍掉任何股票 → 收敛
            if not dropped:
                final_shares = round_shares
                break

            # 有砍掉的 → 重分配
            active_codes = kept
            final_shares = round_shares

            if iteration == max_iterations - 1:
                logger.debug(
                    "整手化迭代达最大次数 %d，%d 只保留",
                    max_iterations, len(final_shares),
                )

        # 计算实际权重
        total_allocated = sum(
            final_shares.get(c, 0) * current_prices.get(c, 0)
            for c in final_shares
        )
        if total_allocated > 0:
            actual_weights = {
                c: (final_shares[c] * current_prices.get(c, 0)) / total_capital
                for c in final_shares
            }

        return final_shares, actual_weights, len(final_shares)

    def _estimate_fee_ratio(self, code: str, shares: int, price: float) -> float:
        """估算单笔交易的佣金占比（简化算法，不实例化完整CostModel）。"""
        amount = shares * price
        if amount <= 0:
            return 1.0
        # 佣金 ≈ max(amount * 0.00015, 5.00)
        commission = max(amount * 0.00015, 5.0)
        return commission / amount

    def check_min_config(
        self,
        code: str,
        shares: int,
        price: float,
        min_amount: float = 5000.0,
    ) -> tuple[bool, str]:
        """检查是否满足最小配置额要求。

        Returns:
            (达标, 不达标原因)
        """
        amount = shares * price
        if amount < min_amount:
            return False, f"配置金额 {amount:.0f} < 最小 {min_amount:.0f}"
        return True, "OK"

    # ── 余额检查 ──────────────────────────

    def can_afford(self, code: str, cash: float, price: float) -> tuple[bool, int]:
        """给定现金，最多能买多少股。

        Returns:
            (是否买得起至少一手, 可买股数)
        """
        board = _code_to_board(code)
        rules = self._rules.get(board, {"min_shares": 100, "increment": 100})
        min_sh = rules["min_shares"]
        inc = rules["increment"]

        max_shares = int(cash / price) if price > 0 else 0
        if max_shares < min_sh:
            return False, 0

        if inc == 1:
            return True, max_shares

        rounded = (max_shares // inc) * inc
        if rounded < min_sh:
            return False, 0
        return True, rounded
