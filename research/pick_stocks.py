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
    load_config, _enabled_factors, _get_codes,
    _load_real_data, _load_financials, _load_overseas_data,
    _build_market_wide_from_pool, _build_feature_loader,
    ConfigDrivenStrategy,
)


def pick_stocks(config_path: str, top_n: int, budget: float) -> list[dict]:
    """加载策略配置，计算今天应持有的 top-N 股票。"""
    config = load_config(config_path)
    config["strategy"]["top_n"] = top_n
    original_end = config["backtest"]["end_date"]
    config["backtest"]["end_date"] = str(date.today())

    # ── 拉数据 ──
    raw_data, label = _load_real_data(config)
    if not raw_data or len(raw_data) < 100:
        print(f"  ✗ 数据加载失败: {label}")
        return []

    codes = _get_codes(config)
    financials = _load_financials(codes)
    start_date = date.fromisoformat(config["backtest"]["start_date"])
    overseas_data = _load_overseas_data(start_date, date.today())
    market_wide_data = _build_market_wide_from_pool(raw_data)
    feature_loader = _build_feature_loader(raw_data, config, financials,
                                           overseas_data, market_wide_data)

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

    strategy = ConfigDrivenStrategy(config, raw_data)
    scores = strategy._score_stocks(features)
    top = scores.nlargest(top_n)

    per_stock = budget / top_n
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
        m = re.match(r'\s*-\s*"(\d{6})"\s*#\s*(.+)', line)
        if m:
            name_map[m.group(1)] = m.group(2).strip()
    return name_map


def main():
    tasks = [
        ("半导体", "configs/strategy_semiconductor.yaml", 2, 100_000),
        ("机器人", "configs/strategy_robotics.yaml", 2, 100_000),
    ]

    total_amount = 0
    for label, config_path, top_n, budget in tasks:
        print(f"\n{'='*70}")
        print(f"  {label}池 — top_{top_n} | 资金 ¥{budget:,}")
        print(f"{'='*70}")

        name_map = _load_name_map(config_path)
        results = pick_stocks(config_path, top_n, budget)
        if not results:
            continue

        print(f"  {'代码':<8s} {'名称':<20s} {'评分':>8s} {'收盘价':>10s} {'建议股数':>10s} {'预估金额':>14s}")
        print(f"  {'─'*8} {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*14}")
        for r in results:
            name = name_map.get(r["code"], "")
            print(f"  {r['code']:<8s} {name:<20s} {r['score']:>8.4f} "
                  f"¥{r['close']:>9.2f} {r['shares']:>10,d} ¥{r['amount']:>13,.0f}")
            total_amount += r["amount"]

    print(f"\n{'='*70}")
    print(f"  合计预估投入: ¥{total_amount:,.0f}")
    print(f"\n  ⚠ 以上基于最近交易日收盘价估算，实盘以明天开盘价为准。")
    print(f"  ⚠ 因子计算不含今日数据（T+1前视偏差保护）。")
    print(f"  ⚠ 刚修复了 momentum_60d bug，建议重跑 optimizer 后再正式执行。")


if __name__ == "__main__":
    main()
