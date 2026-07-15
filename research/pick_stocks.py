#!/usr/bin/env python3
"""快速选股脚本：加载策略配置 → 拉数据 → 算因子 → 输出当天应持有的股票。

用法:
    python research/pick_stocks.py
"""

import sys
from datetime import date
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("picker")

from research.run_backtest_demo import (
    load_config, _enabled_factors, _get_codes, _provider,
    _load_real_data, _load_overseas_data,
    _build_market_wide_from_pool, _build_feature_loader,
    _build_monthly_universe, ConfigDrivenStrategy,
)


def pick_stocks(config_path: str, show_n: int, buy_n: int, budget: float) -> list[dict]:
    """加载策略配置，计算今天应持有的 top-N 股票。

    Args:
        config_path: 策略配置文件路径
        show_n: 显示前几名
        buy_n: 实际买入几只（用于算每只预算 = budget / buy_n）
        budget: 该池总预算
    """
    config = load_config(config_path)
    config["strategy"]["top_n"] = show_n
    original_end = config["backtest"]["end_date"]
    config["backtest"]["end_date"] = str(date.today())

    # ── 拉数据 ──
    raw_data, label = _load_real_data(config)
    if not raw_data or len(raw_data) < 100:
        print(f"  ✗ 数据加载失败: {label}")
        return []

    codes = _get_codes(config)
    # ★ 与回测同一套打分数据（路线图 1.6）：westock 模式加载真财报
    # （PIT 对齐的 ROE/营收同比）+ 总股本（PB/PE/PEG 用），保证
    # 「回测的策略 = 实盘选股的策略」。其它 provider 无真财报，保持为空。
    if _provider(config) == "westock":
        from research.westock_source import westock_financials, westock_total_shares
        financials: dict = westock_financials(codes)
        total_shares: dict = westock_total_shares(codes)
    else:
        financials = {}
        total_shares = {}
    start_date = date.fromisoformat(config["backtest"]["start_date"])
    overseas_data = _load_overseas_data(start_date, date.today())
    market_wide_data = _build_market_wide_from_pool(raw_data)
    # 动态股票池（fixed 模式返回空 dict）
    monthly_universe = _build_monthly_universe(raw_data, config)
    feature_loader = _build_feature_loader(raw_data, config, financials,
                                           overseas_data, market_wide_data,
                                           monthly_universe=monthly_universe,
                                           total_shares=total_shares)

    # ── 找最近有数据的交易日 ──
    from core.common.calendar import get_calendar
    cal = get_calendar()
    trading_days = cal.get_trading_days(start_date, date.today())
    last_td = None
    for td in reversed(trading_days):
        if td in raw_data:
            last_td = td
            break
    if last_td is None:
        print("  ✗ 无可用交易日")
        return []

    # ── 算因子 + 评分 ──
    features = feature_loader(last_td, codes)
    if features.empty:
        print("  ✗ 因子计算为空")
        return []

    strategy = ConfigDrivenStrategy(config, raw_data, monthly_universe=monthly_universe)
    # 选股与回测一致：dynamic 模式下只在当月候选宇宙内打分
    from research.run_backtest_demo import _universe_for_date
    universe = _universe_for_date(monthly_universe, last_td)
    if universe is not None:
        in_uni = [c for c in features.index if c in set(universe)]
        features = features.loc[in_uni]
        if features.empty:
            print("  ✗ 当月宇宙内无可打分标的")
            return []
    scores = strategy._score_stocks(features)
    top = scores.nlargest(show_n)

    per_stock = budget / buy_n
    results = []
    for code, score in top.items():
        close_price = None
        if last_td in raw_data and code in raw_data[last_td]:
            close_price = raw_data[last_td][code]["close"]
        shares = int(per_stock / close_price) // 100 * 100 if close_price else 0
        results.append({
            "code": code,
            "score": round(float(score), 4),
            "close": close_price,
            "shares": shares,
            "amount": round(shares * close_price, 2) if close_price and shares else 0,
        })

    # ── 恢复配置 ──
    config["backtest"]["end_date"] = original_end
    config["strategy"]["top_n"] = 3  # 恢复默认

    return results


def _load_name_map(config_path: str) -> dict[str, str]:
    try:
        with open(config_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {}
    import re
    name_map = {}
    for line in lines:
        m = re.match(r'\s*-\s*"(\d{6})"\s*#\s*([^（(]+)', line)
        if m:
            name_map[m.group(1)] = m.group(2).strip()
    return name_map


def main():
    tasks = [
        ("半导体", "configs/strategy_semiconductor.yaml", 3, 2, 100_000),
        ("机器人", "configs/strategy_robotics.yaml", 3, 2, 100_000),
    ]

    total_amount = 0
    for label, config_path, show_n, buy_n, budget in tasks:
        buy_label = f"买{buy_n}只" if buy_n != show_n else ""
        print(f"\n{'='*70}")
        print(f"  {label}池 — top_{show_n} | {buy_label} | 资金 ¥{budget:,} (每只 ¥{budget//buy_n:,})")
        print(f"{'='*70}")

        name_map = _load_name_map(config_path)
        results = pick_stocks(config_path, show_n, buy_n, budget)
        if not results:
            continue

        print(f"  {'代码':<8s} {'名称':<16s} {'评分':>8s} {'收盘价':>10s} {'建议股数':>10s} {'预估金额':>14s} {'  备注':<s}")
        print(f"  {'─'*8} {'─'*16} {'─'*8} {'─'*10} {'─'*10} {'─'*14} {'  ─────'}")
        for i, r in enumerate(results):
            name = name_map.get(r["code"], "")
            note = "← 建议买入" if i < buy_n else "← 备选"
            code_display = f"  {r['code']:<8s}"
            if i < buy_n:
                code_display = f" ★{r['code']:<7s}"
            print(f"{code_display} {name:<16s} {r['score']:>8.4f} "
                  f"¥{r['close']:>9.2f} {r['shares']:>10,d} ¥{r['amount']:>13,.0f}  {note}")
            if i < buy_n:
                total_amount += r["amount"]

    print(f"\n{'='*70}")
    print(f"  预估投入: ¥{total_amount:,.0f} (★ 标记为建议买入项)")
    print(f"\n  ⚠ 基于最近交易日收盘价估算，实盘以实际成交价为准。")
    print(f"  ⚠ ★ 标记的股票如果股价过高导致 0 股，可从「备选」中替换。")


if __name__ == "__main__":
    main()
