#!/usr/bin/env python3
"""Paper trading 闭环（路线图第 4 阶段）：实盘投钱前验证「回测 ≈ 现实」。

工作流（建议每个调仓日收盘后跑 signal，之后任意一天跑 settle）：

    # 1. 收盘后生成信号（与回测/pick_stocks 同一套打分），落虚拟订单
    python research/paper_trading.py signal \
        --config configs/strategy_semiconductor_westock.yaml \
        --pool semiconductor --budget 100000

    # 2. 次日（或之后任意时间）用真实行情撮合 + 更新每日净值
    python research/paper_trading.py settle --pool semiconductor

    # 3. 对账报告：模拟盘净值 vs 回测预期（影子净值）
    python research/paper_trading.py report --pool semiconductor

★ 双账本设计：
- 真实账本（paper）：订单按信号日之后第一个交易日的【开盘价】成交，
  开盘即涨停买单拒绝 / 跌停卖单拒绝（保守），停牌自动顺延，费用走
  TransactionCostModel（含滑点，与回测同口径）。
- 影子账本（shadow）：同样的订单按【信号日收盘价】无条件成交——
  这正是回测引擎的乐观假设（收盘看到信号且按收盘价成交）。
  两条净值曲线的差 = 回测里隐藏的执行偏差（隔夜跳空、封板、停牌）。

账本为 JSON 单文件（data/paper_trading/ledger_{pool}.json），可手工检查。
简化项（文档化，不影响对账目标）：不含 regime 仓位缩放（满仓等权 top-N）、
订单全部或全不成交（无部分成交）。
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paper")

LEDGER_DIR = _PROJECT_DIR / "data" / "paper_trading"

# 停牌顺延上限：信号日后超过这么多个「市场有行情的交易日」仍无该股行情 → 撤单
MAX_WAIT_TRADING_DAYS = 10
# 涨跌停判定容差（与回测 constraints 封板口径一致）
LIMIT_TOLERANCE = 0.002


# ── 账本持久化 ──────────────────────────────────────

def ledger_path(pool: str) -> Path:
    return LEDGER_DIR / f"ledger_{pool}.json"


def new_ledger(pool: str, config_path: str, initial_capital: float) -> dict:
    return {
        "pool": pool,
        "config_path": config_path,
        "initial_capital": float(initial_capital),
        "created": str(date.today()),
        "cash": float(initial_capital),
        "shadow_cash": float(initial_capital),
        # {code: {"shares": int, "avg_cost": float}}
        "positions": {},
        "shadow_positions": {},
        "pending_orders": [],
        "trades": [],
        "nav_history": [],  # [{date, nav, shadow_nav, cash, shadow_cash, n_positions}]
    }


def load_ledger(pool: str) -> dict | None:
    p = ledger_path(pool)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_ledger(ledger: dict) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    p = ledger_path(ledger["pool"])
    p.write_text(json.dumps(ledger, ensure_ascii=False, indent=2))


# ── 订单生成（纯函数） ──────────────────────────────

def build_rebalance_orders(
    ledger: dict,
    target_codes: list[str],
    prices: dict[str, float],
    signal_date: str,
    scores: dict[str, float] | None = None,
) -> list[dict]:
    """按等权 top-N 目标生成调仓订单（与回测 ConfigDrivenStrategy 同规则）。

    - 卖出不在目标池的全部持仓；
    - 目标池内每只调至 nav/top_n 等权：低配买入差额，超配 >15% 卖出超额；
    - 股数取整到 100 股（简化：不分板块，与 pick_stocks 一致）。

    幂等：调用方应先清掉同 signal_date 的旧 pending 订单再调用。
    """
    scores = scores or {}
    nav = ledger["cash"] + sum(
        pos["shares"] * prices.get(code, pos["avg_cost"])
        for code, pos in ledger["positions"].items()
    )
    top_n = max(len(target_codes), 1)
    per_stock = nav / top_n
    target_set = set(target_codes)
    orders: list[dict] = []

    def _mk(code: str, side: str, shares: int) -> dict:
        return {
            "id": f"{signal_date}_{side}_{code}",
            "signal_date": signal_date,
            "code": code,
            "side": side,
            "shares": int(shares),
            "signal_close": float(prices[code]),
            "score": round(float(scores.get(code, 0.0)), 4),
            "status": "pending",
            "wait_days": 0,
        }

    for code, pos in ledger["positions"].items():
        if pos["shares"] > 0 and code not in target_set and code in prices:
            orders.append(_mk(code, "sell", pos["shares"]))

    for code in target_codes:
        price = prices.get(code, 0.0)
        if price <= 0:
            continue
        cur_shares = ledger["positions"].get(code, {}).get("shares", 0)
        cur_value = cur_shares * price
        target_shares = int(per_stock / price) // 100 * 100
        if cur_value > per_stock * 1.15:
            sell_shares = int((cur_value - per_stock) / price) // 100 * 100
            if sell_shares >= 100:
                orders.append(_mk(code, "sell", sell_shares))
        elif target_shares > cur_shares:
            buy_shares = (target_shares - cur_shares) // 100 * 100
            if buy_shares >= 100:
                orders.append(_mk(code, "buy", buy_shares))
    return orders


# ── 撮合（纯函数，行情由调用方传入） ────────────────

def _board_limit(code: str) -> float:
    from research.westock_source import _board_price_limit
    return _board_price_limit(code)


def _cost_total(code: str, side: str, price: float, shares: int) -> float:
    from core.backtest.cost_model import TransactionCostModel
    from decimal import Decimal
    cb = TransactionCostModel().calculate(code, side, Decimal(str(price)), shares)
    return float(cb.total)


def _apply_fill_to_book(
    cash: float, positions: dict, code: str, side: str,
    price: float, shares: int, cost: float,
) -> tuple[float, float]:
    """把成交落到 (cash, positions)，返回 (新 cash, 卖出已实现盈亏)。"""
    amount = price * shares
    realized = 0.0
    if side == "buy":
        cash -= amount + cost
        pos = positions.setdefault(code, {"shares": 0, "avg_cost": 0.0})
        total_cost_basis = pos["avg_cost"] * pos["shares"] + amount + cost
        pos["shares"] += shares
        pos["avg_cost"] = total_cost_basis / pos["shares"]
    else:
        pos = positions.get(code, {"shares": 0, "avg_cost": 0.0})
        sell_shares = min(shares, pos["shares"])
        net = price * sell_shares - cost
        realized = net - pos["avg_cost"] * sell_shares
        cash += net
        pos["shares"] -= sell_shares
        if pos["shares"] <= 0:
            positions.pop(code, None)
    return cash, realized


def settle_orders(ledger: dict, bars: dict[str, list[dict]]) -> list[dict]:
    """撮合 pending 订单。

    Args:
        ledger: 账本（就地修改）。
        bars: {code: [{"date": "YYYY-MM-DD", "open": f, "close": f}, ...]}
              按日期升序，需覆盖 signal_date 前一日到最新。

    撮合规则（真实账本）：
    - 成交日 = signal_date 之后该股第一个有行情的交易日，成交价 = 当日开盘价；
    - 开盘价贴涨停（≥ pre_close×(1+板块限制)×(1-容差)）→ 买单拒绝；
      贴跌停 → 卖单拒绝（宁可保守，与回测封板口径一致）；
    - 该股无行情但市场有行情 = 停牌顺延，超过 MAX_WAIT_TRADING_DAYS 撤单；
    - 卖单股数超过持仓时按持仓截断。
    影子账本：同订单按 signal_close 无条件成交（= 回测乐观假设）。

    Returns:
        本次处理（成交/拒绝/撤销）的订单列表。
    """
    market_dates = sorted({b["date"] for blist in bars.values() for b in blist})
    processed: list[dict] = []
    still_pending: list[dict] = []

    for order in ledger["pending_orders"]:
        code, side = order["code"], order["side"]
        code_bars = bars.get(code, [])
        fill_bar = None
        prev_close = order["signal_close"]  # 信号日收盘即成交日前收（无更早行情时的兜底）
        for b in code_bars:
            if b["date"] > order["signal_date"]:
                fill_bar = b
                break
            prev_close = b["close"]

        if fill_bar is None:
            # 该股在 signal_date 后无行情：市场有新交易日则视为停牌顺延
            waited = len([d for d in market_dates if d > order["signal_date"]])
            order["wait_days"] = waited
            if waited > MAX_WAIT_TRADING_DAYS:
                order["status"] = "cancelled"
                order["reject_reason"] = f"停牌超过 {MAX_WAIT_TRADING_DAYS} 个交易日，撤单"
                processed.append(order)
                _shadow_fill(ledger, order)  # 影子账本仍按回测假设成交
            else:
                still_pending.append(order)
            continue

        open_px = float(fill_bar["open"])
        limit = _board_limit(code)
        limit_up = prev_close * (1 + limit)
        limit_down = prev_close * (1 - limit)
        rejected = None
        if side == "buy" and open_px >= limit_up * (1 - LIMIT_TOLERANCE):
            rejected = f"开盘即涨停({open_px:.2f}≥{limit_up:.2f})，买单不成交"
        elif side == "sell" and open_px <= limit_down * (1 + LIMIT_TOLERANCE):
            rejected = f"开盘即跌停({open_px:.2f}≤{limit_down:.2f})，卖单不成交"

        if rejected:
            order["status"] = "rejected"
            order["reject_reason"] = rejected
            order["fill_date"] = fill_bar["date"]
            processed.append(order)
            _shadow_fill(ledger, order)
            continue

        shares = order["shares"]
        if side == "sell":
            held = ledger["positions"].get(code, {}).get("shares", 0)
            if held <= 0:
                order["status"] = "cancelled"
                order["reject_reason"] = "无持仓可卖"
                processed.append(order)
                _shadow_fill(ledger, order)
                continue
            shares = min(shares, held)

        cost = _cost_total(code, side, open_px, shares)
        ledger["cash"], realized = _apply_fill_to_book(
            ledger["cash"], ledger["positions"], code, side, open_px, shares, cost)
        order["status"] = "filled"
        order["fill_date"] = fill_bar["date"]
        order["fill_price"] = open_px
        order["filled_shares"] = shares
        order["cost"] = round(cost, 2)
        order["realized_pnl"] = round(realized, 2)
        # 执行偏差：真实成交价 vs 回测假设价（信号日收盘）
        order["exec_gap_pct"] = round(
            (open_px / order["signal_close"] - 1) * 100, 3)
        processed.append(order)
        _shadow_fill(ledger, order)

    ledger["pending_orders"] = still_pending
    ledger["trades"].extend(processed)
    return processed


def _shadow_fill(ledger: dict, order: dict) -> None:
    """影子账本：按信号日收盘价无条件成交（回测的乐观执行假设）。"""
    code, side = order["code"], order["side"]
    px = order["signal_close"]
    shares = order["shares"]
    if side == "sell":
        held = ledger["shadow_positions"].get(code, {}).get("shares", 0)
        if held <= 0:
            return
        shares = min(shares, held)
    cost = _cost_total(code, side, px, shares)
    ledger["shadow_cash"], _ = _apply_fill_to_book(
        ledger["shadow_cash"], ledger["shadow_positions"], code, side, px, shares, cost)


# ── 净值更新（纯函数） ──────────────────────────────

def update_nav(ledger: dict, bars: dict[str, list[dict]]) -> int:
    """把 nav_history 推进到行情覆盖的最新交易日，返回新增条数。

    每个交易日 nav = cash + Σ 持仓股数 × 当日收盘（停牌用最近已知收盘）。
    注意：nav 只在订单结算之后调用才准确（settle 流程保证顺序）。
    """
    last_date = ledger["nav_history"][-1]["date"] if ledger["nav_history"] else ""
    market_dates = sorted({b["date"] for blist in bars.values() for b in blist})
    new_dates = [d for d in market_dates if d > last_date]
    if not new_dates:
        return 0

    # {code: {date: close}} 快速查询
    close_map: dict[str, dict[str, float]] = {
        code: {b["date"]: float(b["close"]) for b in blist}
        for code, blist in bars.items()
    }

    def _last_close(code: str, d: str, fallback: float) -> float:
        cm = close_map.get(code, {})
        if d in cm:
            return cm[d]
        earlier = [k for k in cm if k <= d]
        return cm[max(earlier)] if earlier else fallback

    added = 0
    for d in new_dates:
        nav = ledger["cash"] + sum(
            pos["shares"] * _last_close(code, d, pos["avg_cost"])
            for code, pos in ledger["positions"].items())
        shadow_nav = ledger["shadow_cash"] + sum(
            pos["shares"] * _last_close(code, d, pos["avg_cost"])
            for code, pos in ledger["shadow_positions"].items())
        ledger["nav_history"].append({
            "date": d,
            "nav": round(nav, 2),
            "shadow_nav": round(shadow_nav, 2),
            "cash": round(ledger["cash"], 2),
            "shadow_cash": round(ledger["shadow_cash"], 2),
            "n_positions": sum(1 for p in ledger["positions"].values() if p["shares"] > 0),
        })
        added += 1
    return added


# ── CLI 子命令 ──────────────────────────────────────

def cmd_signal(config_path: str, pool: str, budget: float, top_n: int) -> None:
    """收盘后生成调仓信号，写入账本 pending 订单（幂等：同日重跑覆盖）。"""
    from research.pick_stocks import pick_stocks

    ledger = load_ledger(pool)
    if ledger is None:
        ledger = new_ledger(pool, config_path, budget)
        print(f"  新建账本: {ledger_path(pool)}（初始资金 ¥{budget:,.0f}）")

    results = pick_stocks(config_path, show_n=top_n, buy_n=top_n, budget=budget)
    if not results:
        print("  ✗ 信号生成失败（数据/宇宙为空），账本未变更")
        return

    signal_date = str(date.today())
    # 已有成交记录的信号日不允许重发（会造成重复调仓）
    if any(t.get("signal_date") == signal_date for t in ledger["trades"]):
        print(f"  ✗ {signal_date} 的订单已有成交记录，拒绝重发信号")
        return

    target_codes = [r["code"] for r in results]
    prices = {r["code"]: float(r["close"]) for r in results if r["close"]}
    scores = {r["code"]: r["score"] for r in results}
    # 持仓里不在目标池的股票也需要现价（生成卖单用信号日收盘）——
    # pick_stocks 只返回 top-N，持仓价用最近成本价兜底，settle 时按真实开盘成交
    for code, pos in ledger["positions"].items():
        prices.setdefault(code, pos["avg_cost"])

    # 幂等：清掉同日旧 pending 再生成
    ledger["pending_orders"] = [
        o for o in ledger["pending_orders"] if o["signal_date"] != signal_date]
    orders = build_rebalance_orders(ledger, target_codes, prices, signal_date, scores)
    ledger["pending_orders"].extend(orders)
    save_ledger(ledger)

    print(f"  信号日 {signal_date} | 目标池: {', '.join(target_codes)}")
    if not orders:
        print("  持仓已与目标一致，无需调仓")
    for o in orders:
        print(f"    {o['side']:<4s} {o['code']} {o['shares']:>7d}股 "
              f"@信号价 ¥{o['signal_close']:.2f} (score {o['score']})")
    print(f"  → 共 {len(orders)} 笔订单待次日开盘撮合（跑 settle 结算）")


def cmd_settle(pool: str) -> None:
    """拉真实行情撮合 pending 订单 + 推进每日净值。"""
    from research.westock_source import westock_kline

    ledger = load_ledger(pool)
    if ledger is None:
        print(f"  ✗ 账本不存在: {ledger_path(pool)}（先跑 signal）")
        return

    codes = sorted(set(
        [o["code"] for o in ledger["pending_orders"]]
        + list(ledger["positions"].keys())
        + list(ledger["shadow_positions"].keys())))
    if not codes:
        print("  账本无持仓且无挂单，跳过")
        return

    # 行情窗口：最早挂单信号日/最后净值日 往前几天 → 今天
    anchors = [o["signal_date"] for o in ledger["pending_orders"]]
    if ledger["nav_history"]:
        anchors.append(ledger["nav_history"][-1]["date"])
    if not anchors:
        anchors.append(ledger["created"])
    start = date.fromisoformat(min(anchors)) - timedelta(days=7)
    df = westock_kline(codes, start, date.today())
    if df is None or df.empty:
        print("  ✗ 行情拉取失败，稍后重试")
        return

    bars: dict[str, list[dict]] = {}
    for code, g in df.groupby("code"):
        g = g.sort_values("trade_date")
        bars[str(code)] = [
            {"date": str(r.trade_date), "open": float(r.open), "close": float(r.close)}
            for r in g.itertuples()]

    processed = settle_orders(ledger, bars)
    added = update_nav(ledger, bars)
    save_ledger(ledger)

    for o in processed:
        if o["status"] == "filled":
            print(f"    ✓ {o['side']:<4s} {o['code']} {o['filled_shares']:>7d}股 "
                  f"@ ¥{o['fill_price']:.2f} ({o['fill_date']}) "
                  f"费¥{o['cost']:.2f} 执行偏差 {o['exec_gap_pct']:+.2f}%")
        else:
            print(f"    ✗ {o['side']:<4s} {o['code']} {o['status']}: {o.get('reject_reason','')}")
    print(f"  结算 {len(processed)} 笔 | 剩余挂单 {len(ledger['pending_orders'])} | "
          f"净值推进 {added} 天")
    if ledger["nav_history"]:
        latest = ledger["nav_history"][-1]
        print(f"  最新净值 ({latest['date']}): 模拟盘 ¥{latest['nav']:,.0f} | "
              f"回测预期 ¥{latest['shadow_nav']:,.0f}")


def cmd_report(pool: str) -> None:
    """对账报告：模拟盘 vs 回测预期（影子净值）。"""
    ledger = load_ledger(pool)
    if ledger is None:
        print(f"  ✗ 账本不存在: {ledger_path(pool)}")
        return

    init = ledger["initial_capital"]
    hist = ledger["nav_history"]
    print(f"\n{'='*72}")
    print(f"  Paper Trading 对账报告 — {pool} 池")
    print(f"  账本: {ledger_path(pool)}")
    print(f"  初始资金 ¥{init:,.0f} | 开始日期 {ledger['created']} | "
          f"净值天数 {len(hist)}")
    print(f"{'='*72}")

    if hist:
        latest = hist[-1]
        n_days = len(hist)
        paper_ret = latest["nav"] / init - 1
        shadow_ret = latest["shadow_nav"] / init - 1
        gap = paper_ret - shadow_ret
        peak, max_dd = init, 0.0
        for h in hist:
            peak = max(peak, h["nav"])
            max_dd = max(max_dd, 1 - h["nav"] / peak)
        print(f"  模拟盘净值:   ¥{latest['nav']:>12,.0f}  ({paper_ret:+.2%})")
        print(f"  回测预期净值: ¥{latest['shadow_nav']:>12,.0f}  ({shadow_ret:+.2%})"
              f"  ← 信号日收盘全成交的理想执行")
        print(f"  执行偏差:     {gap:+.2%}（负值 = 回测比现实乐观）")
        print(f"  最大回撤:     {max_dd:.2%} | 跟踪天数 {n_days}")

    trades = ledger["trades"]
    filled = [t for t in trades if t["status"] == "filled"]
    rejected = [t for t in trades if t["status"] in ("rejected", "cancelled")]
    if filled:
        gaps = [t["exec_gap_pct"] for t in filled]
        buy_gaps = [t["exec_gap_pct"] for t in filled if t["side"] == "buy"]
        print(f"\n  成交 {len(filled)} 笔 | 拒/撤 {len(rejected)} 笔")
        print(f"  开盘 vs 信号收盘价偏差: 均值 {sum(gaps)/len(gaps):+.2f}% "
              f"| 买单均值 {sum(buy_gaps)/len(buy_gaps):+.2f}%" if buy_gaps else "")
    if rejected:
        print("  被拒/撤订单（回测中它们可能被假设成交）:")
        for t in rejected:
            print(f"    {t['signal_date']} {t['side']} {t['code']}: {t.get('reject_reason','')}")

    if ledger["pending_orders"]:
        print(f"\n  待结算挂单 {len(ledger['pending_orders'])} 笔（跑 settle 处理）")
    if hist and len(hist) >= 5:
        print(f"\n  近 5 日净值:")
        for h in hist[-5:]:
            print(f"    {h['date']}  模拟 ¥{h['nav']:>11,.0f}  预期 ¥{h['shadow_nav']:>11,.0f}"
                  f"  持仓 {h['n_positions']} 只")
    print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Paper trading 闭环")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sig = sub.add_parser("signal", help="收盘后生成调仓信号（落虚拟订单）")
    p_sig.add_argument("--config", required=True, help="策略配置 yaml")
    p_sig.add_argument("--pool", required=True, help="账本名（如 semiconductor）")
    p_sig.add_argument("--budget", type=float, default=100_000, help="初始资金（仅建账本时生效）")
    p_sig.add_argument("--top-n", type=int, default=3, help="持仓只数")

    p_set = sub.add_parser("settle", help="用真实行情撮合订单 + 更新净值")
    p_set.add_argument("--pool", required=True)

    p_rep = sub.add_parser("report", help="对账报告：模拟盘 vs 回测预期")
    p_rep.add_argument("--pool", required=True)

    args = parser.parse_args()
    if args.cmd == "signal":
        cmd_signal(args.config, args.pool, args.budget, args.top_n)
    elif args.cmd == "settle":
        cmd_settle(args.pool)
    elif args.cmd == "report":
        cmd_report(args.pool)


if __name__ == "__main__":
    main()
