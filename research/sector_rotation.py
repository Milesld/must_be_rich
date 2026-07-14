#!/usr/bin/env python3
"""行业轮动回测 — 多因子行业打分 + 月度选 top-K 重仓。

与个股选股链路（run_backtest_demo.py）完全独立：行业指数无涨跌停/ST/T+1/
整手，故不复用 BacktestEngine 的个股撮合，改为轻量净值推进。

数据源：westock 行业指数日线（复用 research/westock_source.westock_kline）。
核心判据：轮动组合 vs 沪深300 的超额年化 / 信息比率(IR)。

运行:
    python research/sector_rotation.py                          # 默认配置
    python research/sector_rotation.py configs/sector_rotation.yaml
    python research/sector_rotation.py configs/sector_rotation.yaml --full  # 完整区间
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rotation")


def _parse_date(s: str) -> date:
    return date.fromisoformat(str(s))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════
# 1. 数据加载（行业指数 + 基准，复用 westock_kline）
# ══════════════════════════════════════════════════════════════

def _load_sector_prices(config: dict, start: date, end: date) -> dict[str, pd.Series]:
    """拉行业指数 + 基准的收盘价序列。

    多拉 lookback（最长因子窗口）以支持起点当天打分。
    Returns: {code: close_series(index=date, 升序)}，含基准 code。
    """
    from research.westock_source import westock_kline

    # 最长因子窗口（交易日）→ 自然日预留（×1.6 + 缓冲）
    max_win = max((f.get("window", 0) for f in config.get("factors", {}).values()), default=120)
    pad = int(max_win * 1.6) + 30
    pull_start = start - timedelta(days=pad)

    codes = [s["code"] for s in config["sectors"]]
    bench = config.get("benchmark", "sh000300")
    all_codes = list(dict.fromkeys(codes + [bench]))

    out: dict[str, pd.Series] = {}
    for code in all_codes:
        # 行业指数只有十几条，用更强重试(4次)确保拉全——数据缺口会让轮动排名失真
        df = westock_kline([code], pull_start, end, batch=1, max_retries=4)
        if df is None or df.empty:
            logger.warning("行业指数 %s 拉取为空，跳过", code)
            continue
        s = df.set_index("trade_date")["close"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        out[code] = s
        logger.info("  %s: %d 条 (%s ~ %s)", code, len(s), s.index[0], s.index[-1])
    return out


# ══════════════════════════════════════════════════════════════
# 2. 行业因子打分（PIT：只用 as_of 之前的数据）
# ══════════════════════════════════════════════════════════════

def _factor_raw(prices: pd.Series, as_of: date, fname: str, window: int) -> float | None:
    """算单个行业单个因子的原始值（只用 as_of 之前的 close）。"""
    hist = prices[prices.index < as_of]
    if len(hist) < window + 1:
        return None
    arr = hist.values.astype(float)
    if fname.startswith("momentum") or fname == "rel_strength":
        # 动量：close[t]/close[t-window] − 1
        return float(arr[-1] / arr[-1 - window] - 1.0)
    if fname.startswith("volatility"):
        rets = np.diff(arr[-window - 1:]) / arr[-window - 1:-1]
        return float(np.nanstd(rets) * np.sqrt(252))
    return None


def _score_sectors(prices: dict[str, pd.Series], as_of: date,
                   factors: dict, sector_codes: list[str]) -> dict[str, float]:
    """多因子横截面打分：各因子 rank 百分位 × 权重（inverse 反向）求和。

    rel_strength 特殊：先算各行业动量，再取"本行业动量 − 全池均值"作为原始值。
    Returns: {code: 综合分}（仅含数据足够的行业）。
    """
    # 1) 收集各因子原始值 {fname: {code: raw}}
    raw: dict[str, dict[str, float]] = {}
    for fname, fcfg in factors.items():
        window = fcfg.get("window", 120)
        vals: dict[str, float] = {}
        for code in sector_codes:
            if code not in prices:
                continue
            if fname == "rel_strength":
                v = _factor_raw(prices[code], as_of, "momentum", window)
            else:
                v = _factor_raw(prices[code], as_of, fname, window)
            if v is not None:
                vals[code] = v
        # rel_strength：减全池均值
        if fname == "rel_strength" and vals:
            mean_v = sum(vals.values()) / len(vals)
            vals = {c: v - mean_v for c, v in vals.items()}
        raw[fname] = vals

    # 2) 各因子横截面 rank 百分位 → 加权
    scores: dict[str, float] = {c: 0.0 for c in sector_codes if c in prices}
    total_w = sum(f.get("weight", 0) for f in factors.values()) or 1.0
    for fname, fcfg in factors.items():
        vals = raw.get(fname, {})
        if len(vals) < 2:
            continue
        w = fcfg.get("weight", 0) / total_w
        inverse = fcfg.get("inverse", False)
        s = pd.Series(vals)
        rank = s.rank(pct=True)
        if inverse:
            rank = 1.0 - rank
        for code, r in rank.items():
            scores[code] = scores.get(code, 0.0) + float(r) * w

    # 只保留有打分数据的行业（至少有一个因子算出来）
    scored_codes = {c for f in raw.values() for c in f}
    return {c: v for c, v in scores.items() if c in scored_codes}


# ══════════════════════════════════════════════════════════════
# 3. 轮动回测（自建净值推进，不用 BacktestEngine）
# ══════════════════════════════════════════════════════════════

def run_rotation(config: dict, start: date, end: date) -> dict:
    """月度行业轮动回测，返回 {nav_series, weights_log, picks_log}。

    每月首交易日按多因子打分选 top_k 行业等权重仓；每日按持仓行业指数
    日收益推进组合净值。月度换手扣固定双边成本 cost_bps。
    """
    from core.common.calendar import get_calendar
    cal = get_calendar()

    prices = _load_sector_prices(config, start, end)
    sector_codes = [s["code"] for s in config["sectors"] if s["code"] in prices]
    if len(sector_codes) < config["rotation"].get("top_k", 3):
        raise RuntimeError(f"可用行业数 {len(sector_codes)} < top_k，无法轮动")

    factors = config.get("factors", {})
    top_k = config["rotation"].get("top_k", 3)
    cost_bps = config["rotation"].get("cost_bps", 0) / 1e4

    # 各行业日收益序列（对齐到交易日）
    rets = {c: prices[c].pct_change() for c in sector_codes}

    trading_days = cal.get_trading_days(start, end)
    nav = 1.0
    cur_weights: dict[str, float] = {}   # 当前持仓 {code: weight}
    nav_rows: list[dict] = []
    picks_log: list[dict] = []

    def _is_rebalance(td: date) -> bool:
        prev = cal.prev_trading_day(td)
        return prev.month != td.month  # 每月首个交易日

    for td in trading_days:
        # ── 先按当日行业收益推进净值（用昨日持仓）──
        if cur_weights:
            day_ret = 0.0
            for code, w in cur_weights.items():
                r = rets[code].get(td, 0.0)
                if pd.isna(r):
                    r = 0.0
                day_ret += w * float(r)
            nav *= (1.0 + day_ret)

        # ── 调仓日：重新打分选 top_k ──
        if _is_rebalance(td):
            scores = _score_sectors(prices, td, factors, sector_codes)
            if scores:
                ranked = sorted(scores.items(), key=lambda x: -x[1])
                picks = [c for c, _ in ranked[:top_k]]
                new_weights = {c: 1.0 / len(picks) for c in picks}
                # 换手成本：与旧持仓的权重变动之和 × cost_bps
                turnover = sum(
                    abs(new_weights.get(c, 0) - cur_weights.get(c, 0))
                    for c in set(new_weights) | set(cur_weights)
                )
                nav *= (1.0 - turnover * cost_bps)
                cur_weights = new_weights
                picks_log.append({
                    "date": td, "picks": picks,
                    "names": [_name_of(config, c) for c in picks],
                })

        nav_rows.append({"date": td, "nav": nav})

    nav_series = pd.DataFrame(nav_rows).set_index("date")["nav"]
    return {"nav": nav_series, "picks": picks_log, "prices": prices}


def _name_of(config: dict, code: str) -> str:
    for s in config["sectors"]:
        if s["code"] == code:
            return s.get("name", code)
    return code


# ══════════════════════════════════════════════════════════════
# 4. 指标（含相对基准超额）
# ══════════════════════════════════════════════════════════════

def _metrics(nav: pd.Series, bench_close: pd.Series | None) -> dict:
    """从净值序列算年化/夏普/回撤 + 相对基准超额年化/信息比率/超额回撤。"""
    nav = nav.dropna()
    if len(nav) < 20:
        return {"error": "净值样本不足"}
    rets = nav.pct_change().dropna()
    n = len(nav)
    years = n / 252
    total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1)
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    ann_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.0
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0.0
    peak = nav.expanding().max()
    max_dd = float(((nav - peak) / peak).min())

    m = {
        "n_days": n, "total_return": total_ret, "annual_return": float(ann_ret),
        "annual_vol": ann_vol, "sharpe": float(sharpe), "max_drawdown": max_dd,
        "calmar": float(ann_ret / abs(max_dd)) if max_dd < 0 else 0.0,
    }

    # 相对基准超额
    if bench_close is not None and not bench_close.empty:
        b = bench_close.reindex(nav.index).ffill().bfill()
        b_total = float(b.iloc[-1] / b.iloc[0] - 1)
        b_ann = (1 + b_total) ** (1 / years) - 1 if years > 0 else 0.0
        b_rets = b.pct_change().dropna()
        common = rets.index.intersection(b_rets.index)
        excess = (rets.loc[common] - b_rets.loc[common]).dropna()
        ir = float(excess.mean() * 252 / (excess.std() * np.sqrt(252))) if len(excess) > 1 and excess.std() > 0 else 0.0
        rel = (nav / nav.iloc[0]) / (b / b.iloc[0])
        rel_dd = float(((rel - rel.expanding().max()) / rel.expanding().max()).min())
        m.update({
            "benchmark_annual": float(b_ann),
            "excess_annual": float(ann_ret - b_ann),
            "information_ratio": ir,
            "excess_max_drawdown": rel_dd,
        })
    return m


# ══════════════════════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════════════════════

def main(config_path: str = "configs/sector_rotation.yaml", full_range: bool = False) -> None:
    config = load_config(config_path)
    bt = config["backtest"]
    if not full_range and "validate_start" in bt and "validate_end" in bt:
        start, end = _parse_date(bt["validate_start"]), _parse_date(bt["validate_end"])
    else:
        start, end = _parse_date(bt["start_date"]), _parse_date(bt["end_date"])

    logger.info("行业轮动回测 %s ~ %s | %d 个行业 | top_k=%d",
                start, end, len(config["sectors"]), config["rotation"].get("top_k", 3))

    result = run_rotation(config, start, end)
    nav = result["nav"]
    bench = config.get("benchmark", "sh000300")
    bench_close = result["prices"].get(bench)

    m = _metrics(nav, bench_close)

    print("\n" + "=" * 60)
    print("行业轮动回测结果")
    print("=" * 60)
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:.<28s} {v:>10.4f}")
        else:
            print(f"  {k:.<28s} {v!s:>10s}")
    print("=" * 60)

    if "excess_annual" in m:
        verdict = "跑赢基准 ✓" if m["excess_annual"] > 0 else "跑输基准 ✗"
        print(f"\n相对基准（{bench}）—— {verdict}")
        print(f"  策略年化 {m['annual_return']*100:+.1f}%  vs  基准年化 {m['benchmark_annual']*100:+.1f}%"
              f"  →  超额年化 {m['excess_annual']*100:+.1f}%")
        print(f"  信息比率(IR) {m['information_ratio']:.3f}   |   超额最大回撤 {m['excess_max_drawdown']*100:.1f}%")
        print(f"  （IR>0.5 才算有持续超额；轮动的意义在于跑赢'躺着买沪深300'）")
        print("=" * 60)

    # 各月选中行业
    print("\n各月轮动选中行业:")
    for p in result["picks"]:
        print(f"  {p['date']}: {' / '.join(p['names'])}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    full = any(a in ("--full", "--full-range") for a in args)
    pos = [a for a in args if not a.startswith("-")]
    path = pos[0] if pos else "configs/sector_rotation.yaml"
    main(path, full_range=full)
