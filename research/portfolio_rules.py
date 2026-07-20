#!/usr/bin/env python3
"""组合构建规则（路线图第 7 阶段）：换手控制 + 行业分散 + 权重上限。

纯函数，被回测（ConfigDrivenStrategy.on_bar）与实盘选股（pick_stocks →
paper_trading signal）共用，保证「回测的组合规则 = 实盘的组合规则」。

三个约束（均配置驱动，不配置 = 旧行为）：

1. rank buffer 换手控制（strategy.rank_buffer）
   经典双阈值：新买入须进 top_n，已持有跌出 rank_buffer（> top_n）才卖。
   排名在 (top_n, rank_buffer] 之间的持仓保留——避免月度调仓在排名噪声上
   反复买卖。路线图原文「月换手率上限（如单边 50%）」用此机制实现：
   buffer 越大换手越低（硬换手率上限需要拒单顺序决策，等权 top-N 场景
   下 rank buffer 是业界更常用且行为更稳定的等效物）。

2. 单行业只数上限（strategy.max_per_sector）
   按分数从高到低贪心填充，某行业已满则跳过（被跳过的位置由后续分数
   填补）。等权组合下「只数上限 k / top_n」≈ 行业敞口上限。
   路线图「单行业敞口 ≤ 30%」→ top_n=20 时配 max_per_sector: 6。

3. 单票权重上限（strategy.max_weight_per_stock）
   等权 top-N 的每只权重 = 1/n；上限对 n < 1/cap 的小组合才生效
   （top_n=3 时 33% → 压到 10%，余下留现金）。保守处理：超额部分
   不重分配（重分配会突破其它票的上限或放大集中度）。
"""

from __future__ import annotations

import math

import pandas as pd


def select_target_portfolio(
    scores: pd.Series,
    top_n: int,
    held: set[str] | None = None,
    rank_buffer: int | None = None,
    sector_map: dict[str, str] | None = None,
    max_per_sector: int | None = None,
) -> list[str]:
    """从打分结果选目标组合（含换手缓冲与行业分散约束）。

    Args:
        scores: {code: score}，越高越好。
        top_n: 目标持仓只数。
        held: 当前持仓代码（rank buffer 用；None/空 = 无缓冲效果）。
        rank_buffer: 已持有票排名 ≤ rank_buffer 即保留。None = 关闭
                     （行为退化为纯 top_n，与旧版一致）。须 ≥ top_n。
        sector_map: {code: 行业标签}。缺标签的票视为各自独立行业（不受限）。
        max_per_sector: 单行业最多只数。None = 关闭。

    Returns:
        目标代码列表（按分数降序，长度 ≤ top_n）。

    选择顺序（保证确定性）：
    1. 全部候选按分数降序排名；
    2. 若开启 buffer：先保留「已持有且排名 ≤ rank_buffer」的票（按排名序）；
    3. 剩余名额按排名从未持有（或跌出 buffer）的票中填充；
    4. 全程执行行业只数上限（保留的持仓也计入行业配额）。
    """
    if scores.empty or top_n <= 0:
        return []
    held = held or set()
    ranked = scores.sort_values(ascending=False)
    order = list(ranked.index)
    rank_of = {c: i + 1 for i, c in enumerate(order)}
    buffer = max(rank_buffer, top_n) if rank_buffer else None

    sector_map = sector_map or {}
    sector_count: dict[str, int] = {}

    def _sector_ok(code: str) -> bool:
        if not max_per_sector:
            return True
        sec = sector_map.get(code)
        if sec is None:
            return True
        return sector_count.get(sec, 0) < max_per_sector

    def _take(code: str) -> None:
        sec = sector_map.get(code)
        if sec is not None:
            sector_count[sec] = sector_count.get(sec, 0) + 1

    target: list[str] = []
    taken: set[str] = set()

    # 1. buffer 保留：已持有且排名未跌出 buffer（按排名序，行业配额同样约束）
    if buffer:
        for code in order:
            if len(target) >= top_n:
                break
            if code in held and rank_of[code] <= buffer and _sector_ok(code):
                target.append(code)
                taken.add(code)
                _take(code)

    # 2. 名额不足部分按排名填充
    for code in order:
        if len(target) >= top_n:
            break
        if code in taken:
            continue
        if _sector_ok(code):
            target.append(code)
            taken.add(code)
            _take(code)

    return sorted(target, key=lambda c: rank_of[c])


def per_stock_weight(top_n: int, max_weight: float | None = None) -> float:
    """等权 top-N 的单票目标权重，受单票上限约束（超额留现金）。"""
    if top_n <= 0:
        return 0.0
    w = 1.0 / top_n
    if max_weight and max_weight > 0:
        w = min(w, max_weight)
    return w


def portfolio_rules_from_config(config: dict) -> dict:
    """从 strategy 配置节提取组合规则参数（统一入口，回测/实盘共用）。"""
    cfg = config.get("strategy", {})
    rank_buffer = cfg.get("rank_buffer")
    max_per_sector = cfg.get("max_per_sector")
    max_weight = cfg.get("max_weight_per_stock")
    return {
        "rank_buffer": int(rank_buffer) if rank_buffer else None,
        "max_per_sector": int(max_per_sector) if max_per_sector else None,
        "max_weight": float(max_weight) if max_weight else None,
    }
