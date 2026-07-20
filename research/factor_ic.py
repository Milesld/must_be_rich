#!/usr/bin/env python3
"""因子 IC 检验层（路线图第 5 阶段）：先证明因子有信息量，再让优化器搜权重。

方法论（alphalens 风格）：
- 对每个候选因子，在每个月度调仓截面上计算 RankIC（因子值与下期收益的
  Spearman 秩相关），得到 IC 时间序列 → IC 均值 / ICIR / t 统计量 / 正率；
- IC 衰减：同一因子对 1/3/6 个月后收益分别算 ICIR，看信息保持多久；
- 分层回测：每个截面按因子值分五层（quintile），看 top-bottom spread
  是否为正、各层收益是否单调；
- ICIR（方向调整后）≥ 阈值（默认 0.3）判 pass，报告落 JSON。

与优化器的闭环：
    # 1. 生成 IC 报告（与回测同一套因子计算/PIT 财报/动态宇宙）
    python research/factor_ic.py --config configs/strategy_semiconductor_westock.yaml

    # 2. 优化器用报告过滤候选池（ICIR 不达标的因子直接不进搜索空间）
    python research/factor_optimizer.py --task long_term \
        --config configs/strategy_semiconductor_westock.yaml \
        --ic-report data/reports/factor_ic/ic_semiconductor_westock.json

方向约定：反向因子（波动率/PE 等，见 ConfigDrivenStrategy.INVERSE_*）
预期 IC 为负，统计时按方向调整（adj_ic = direction × raw_ic），
使「好因子」的 adj ICIR 恒为正，阈值判断统一。

前视说明：forward return 用未来截面的收盘价，这是研究评估的目标变量，
不是交易信号；因子值本身仍按 PIT 计算（与回测同一 feature_loader）。
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import date
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("factor_ic")

REPORT_DIR = _PROJECT_DIR / "data" / "reports" / "factor_ic"

# 默认 ICIR 门槛（业界常用 0.3~0.5；月频截面本就少，取下限）
DEFAULT_ICIR_THRESHOLD = 0.3
# 单截面最少有效样本数（低于此不算 IC，横截面太小没有统计意义）
MIN_CROSS_SECTION = 8
# IC 衰减的持有期（单位：调仓期数，月度调仓即 1/3/6 个月）
DECAY_HORIZONS = (1, 3, 6)


def factor_direction(name: str) -> int:
    """因子预期方向：+1 越高越好，-1 越低越好（与回测打分的反转口径一致）。"""
    from research.run_backtest_demo import ConfigDrivenStrategy
    if (name in ConfigDrivenStrategy.INVERSE_EXACT
            or name.startswith(ConfigDrivenStrategy.INVERSE_PREFIXES)):
        return -1
    return 1


# ══════════════════════════════════════════════════════════════
# 纯函数计算层（无 IO，可单测）
# ══════════════════════════════════════════════════════════════

def compute_ic_series(
    factor_panel: dict[date, pd.Series],
    fwd_panel: dict[date, pd.Series],
    min_obs: int = MIN_CROSS_SECTION,
) -> pd.Series:
    """逐截面 RankIC（Spearman）序列。

    Args:
        factor_panel: {截面日: Series(因子值, index=code)}。
        fwd_panel: {截面日: Series(下期收益, index=code)}。
        min_obs: 单截面最少共同样本数。

    Returns:
        Series(ic, index=截面日)，无效截面（样本不足/因子无区分度）跳过。
    """
    out: dict[date, float] = {}
    for d in sorted(factor_panel.keys()):
        if d not in fwd_panel:
            continue
        fac = factor_panel[d].dropna()
        fwd = fwd_panel[d].dropna()
        common = fac.index.intersection(fwd.index)
        if len(common) < min_obs:
            continue
        f, r = fac.loc[common], fwd.loc[common]
        if f.nunique() < 2 or r.nunique() < 2:
            continue  # 全同值（如无数据因子恒 0）→ 秩相关无定义
        ic = f.corr(r, method="spearman")
        if not np.isnan(ic):
            out[d] = float(ic)
    return pd.Series(out, dtype=float)


def ic_stats(ic_series: pd.Series, direction: int = 1) -> dict:
    """IC 序列 → 汇总统计（方向调整后）。

    Returns:
        {n_periods, ic_mean, ic_std, icir, t_stat, positive_rate}
        （均为方向调整后口径：好因子 ic_mean/icir 为正）。空序列 → 全 0。
    """
    if len(ic_series) == 0:
        return {"n_periods": 0, "ic_mean": 0.0, "ic_std": 0.0,
                "icir": 0.0, "t_stat": 0.0, "positive_rate": 0.0}
    adj = ic_series * direction
    mean = float(adj.mean())
    std = float(adj.std(ddof=1)) if len(adj) > 1 else 0.0
    if std > 1e-12:
        icir = mean / std
    else:
        # 退化情形：IC 恒定（如完美相关）。均值非 0 = 极强信号，封顶 99；
        # 均值也为 0 = 无信号。
        icir = float(np.sign(mean)) * 99.0 if abs(mean) > 1e-12 else 0.0
    t_stat = icir * np.sqrt(len(adj))
    return {
        "n_periods": int(len(adj)),
        "ic_mean": round(mean, 4),
        "ic_std": round(std, 4),
        "icir": round(icir, 4),
        "t_stat": round(float(t_stat), 4),
        "positive_rate": round(float((adj > 0).mean()), 4),
    }


def quintile_backtest(
    factor_panel: dict[date, pd.Series],
    fwd_panel: dict[date, pd.Series],
    direction: int = 1,
    n_quantiles: int = 5,
    min_obs: int = MIN_CROSS_SECTION,
) -> dict:
    """分层回测：每个截面按（方向调整后）因子值分 n 层，统计各层平均下期收益。

    Q1 = 因子最差层，Qn = 因子最好层（方向调整后，好因子应 Qn > Q1）。

    Returns:
        {n_periods, quantile_mean_returns: [Q1..Qn], top_bottom_spread,
         monotonicity}；monotonicity = 层序号与层收益的 Spearman 相关
        （1.0 = 完美单调递增）。无有效截面 → n_periods=0 其余 0/空。
    """
    per_date_rows: list[list[float]] = []
    for d in sorted(factor_panel.keys()):
        if d not in fwd_panel:
            continue
        fac = factor_panel[d].dropna() * direction
        fwd = fwd_panel[d].dropna()
        common = fac.index.intersection(fwd.index)
        if len(common) < max(min_obs, n_quantiles):
            continue
        f, r = fac.loc[common], fwd.loc[common]
        if f.nunique() < 2:
            continue
        # rank(method=first) 断平局，保证每层都有成员（qcut 对大量重复值会失败）
        pct = f.rank(method="first", pct=True)
        bins = np.minimum((pct * n_quantiles).apply(np.ceil).astype(int), n_quantiles)
        means = [float(r[bins == q].mean()) if (bins == q).any() else np.nan
                 for q in range(1, n_quantiles + 1)]
        if not any(np.isnan(m) for m in means):
            per_date_rows.append(means)

    if not per_date_rows:
        return {"n_periods": 0, "quantile_mean_returns": [],
                "top_bottom_spread": 0.0, "monotonicity": 0.0}

    mat = np.array(per_date_rows, dtype=float)
    q_means = mat.mean(axis=0)
    spread = float(q_means[-1] - q_means[0])
    mono = pd.Series(q_means).corr(
        pd.Series(range(1, n_quantiles + 1), dtype=float), method="spearman")
    return {
        "n_periods": int(len(per_date_rows)),
        "quantile_mean_returns": [round(float(x), 5) for x in q_means],
        "top_bottom_spread": round(spread, 5),
        "monotonicity": round(float(mono) if not np.isnan(mono) else 0.0, 4),
    }


def evaluate_factor(
    name: str,
    factor_panel: dict[date, pd.Series],
    fwd_panels: dict[int, dict[date, pd.Series]],
    icir_threshold: float = DEFAULT_ICIR_THRESHOLD,
) -> dict:
    """单因子完整评估：主 horizon(1 期) IC 统计 + IC 衰减 + 分层回测 + 判定。

    Args:
        fwd_panels: {持有期数: fwd_panel}，须含 1（主 horizon）。

    判定规则：主 horizon 的（方向调整后）ICIR ≥ 阈值 → pass。
    分层结果作为参考信息输出，不参与判定（截面小，spread 噪声大）。
    """
    direction = factor_direction(name)
    main_ic = compute_ic_series(factor_panel, fwd_panels[1])
    stats = ic_stats(main_ic, direction)

    decay = {}
    for h, panel in sorted(fwd_panels.items()):
        if h == 1:
            decay[str(h)] = {"ic_mean": stats["ic_mean"], "icir": stats["icir"]}
        else:
            s = ic_stats(compute_ic_series(factor_panel, panel), direction)
            decay[str(h)] = {"ic_mean": s["ic_mean"], "icir": s["icir"]}

    quintile = quintile_backtest(factor_panel, fwd_panels[1], direction)
    verdict = "pass" if (stats["n_periods"] > 0
                         and stats["icir"] >= icir_threshold) else "fail"
    return {
        "direction": direction,
        **stats,
        "ic_decay": decay,
        "quintile": quintile,
        "verdict": verdict,
    }


def filter_candidates_by_report(
    candidates: list[str],
    report: dict,
    min_icir: float,
) -> tuple[list[str], list[str], list[str]]:
    """用 IC 报告过滤候选因子池（优化器门禁）。

    Returns:
        (kept, dropped, missing)：
        - kept: ICIR 达标的因子；
        - dropped: 报告中 ICIR < min_icir 的因子（被剔除）；
        - missing: 报告未覆盖的因子（保留，附提示——未检验 ≠ 无效）。
    """
    factors = report.get("factors", {})
    kept, dropped, missing = [], [], []
    for c in candidates:
        rec = factors.get(c)
        if rec is None:
            missing.append(c)
            kept.append(c)
        elif rec.get("icir", 0.0) >= min_icir and rec.get("n_periods", 0) > 0:
            kept.append(c)
        else:
            dropped.append(c)
    return kept, dropped, missing


# ══════════════════════════════════════════════════════════════
# 数据组装层（复用回测链路：同一 feature_loader / PIT 财报 / 动态宇宙）
# ══════════════════════════════════════════════════════════════

def month_anchor_dates(trade_dates: list[date]) -> list[date]:
    """每个自然月的首个交易日（= 回测的月度调仓截面日）。"""
    anchors: dict[tuple[int, int], date] = {}
    for d in sorted(trade_dates):
        anchors.setdefault((d.year, d.month), d)
    return sorted(anchors.values())


def build_forward_returns(
    raw_data: dict[date, dict],
    anchors: list[date],
    horizons: tuple[int, ...] = DECAY_HORIZONS,
) -> dict[int, dict[date, pd.Series]]:
    """各截面日的 h 期后收益：close(anchor[i+h]) / close(anchor[i]) − 1。

    只对两端都有行情的股票计算（停牌缺行情 → 该截面剔除该股）。
    """
    out: dict[int, dict[date, pd.Series]] = {h: {} for h in horizons}
    for i, d in enumerate(anchors):
        day_now = raw_data.get(d, {})
        for h in horizons:
            if i + h >= len(anchors):
                continue
            d_fut = anchors[i + h]
            day_fut = raw_data.get(d_fut, {})
            rets = {
                code: day_fut[code]["close"] / fields["close"] - 1.0
                for code, fields in day_now.items()
                if fields.get("close", 0) > 0
                and code in day_fut and day_fut[code].get("close", 0) > 0
            }
            if rets:
                out[h][d] = pd.Series(rets, dtype=float)
    return out


def default_candidates(config: dict) -> list[str]:
    """默认候选池：long_term 因子池，按 provider 排除无数据源因子。

    与 factor_optimizer.optimize 的过滤口径完全一致（同一份集合定义）。
    """
    from research.factor_optimizer import (
        FACTOR_POOL, ALL_NO_DATA, WESTOCK_REAL_FUNDAMENTALS,
    )
    from research.run_backtest_demo import _provider

    all_factors = config.get("factors", {})
    raw = [c for c in FACTOR_POOL["long_term"]["candidates"] if c in all_factors]
    no_data = set(ALL_NO_DATA)
    if _provider(config) == "westock":
        no_data -= WESTOCK_REAL_FUNDAMENTALS
    return [c for c in raw if c not in no_data]


def build_ic_dataset(
    config_path: str,
    factor_names: list[str] | None = None,
) -> tuple[dict[str, dict[date, pd.Series]], dict[int, dict[date, pd.Series]], dict]:
    """加载数据并组装 IC 分析用的因子面板与前瞻收益面板。

    与回测/选股同一套数据链路（westock 真财报 PIT 对齐、动态月度宇宙、
    同一 feature_loader），保证「检验的因子 = 回测里用的因子」。

    Returns:
        (factor_panels, fwd_panels, meta)：
        - factor_panels: {因子名: {截面日: Series}}（已按当月宇宙过滤）；
        - fwd_panels: {持有期: {截面日: Series}}；
        - meta: {candidates, anchors, provider, config_path}。
    """
    from research.run_backtest_demo import (
        load_config, _load_real_data, _build_feature_loader,
        _build_monthly_universe, _universe_for_date, _get_codes,
        _build_market_wide_from_pool, _provider, _parse_date,
    )

    config = load_config(config_path)
    candidates = factor_names or default_candidates(config)
    if not candidates:
        raise ValueError("候选因子池为空（检查 config.factors 与 provider）")

    # 只启用候选因子（feature_loader 按 enabled 计算），权重与 IC 无关
    cfg = copy.deepcopy(config)
    for name in cfg.get("factors", {}):
        cfg["factors"][name]["enabled"] = name in candidates

    raw_data, label = _load_real_data(cfg)
    if not raw_data or len(raw_data) < 100:
        raise RuntimeError(f"数据加载失败: {label}")

    codes = _get_codes(cfg)
    if _provider(cfg) == "westock":
        from research.westock_source import westock_financials, westock_total_shares
        financials: dict = westock_financials(codes)
        total_shares: dict = westock_total_shares(codes)
    else:
        financials, total_shares = {}, {}

    monthly_universe = _build_monthly_universe(raw_data, cfg)
    market_wide = _build_market_wide_from_pool(raw_data)
    feature_loader = _build_feature_loader(
        raw_data, cfg, financials, None, market_wide,
        monthly_universe=monthly_universe, total_shares=total_shares)

    bt_start = _parse_date(cfg["backtest"]["start_date"])
    bt_end = _parse_date(cfg["backtest"]["end_date"])
    anchors = month_anchor_dates([d for d in raw_data if bt_start <= d <= bt_end])
    if len(anchors) < 8:
        raise RuntimeError(f"月度截面太少（{len(anchors)} 个），IC 无统计意义")

    fwd_panels = build_forward_returns(raw_data, anchors)

    factor_panels: dict[str, dict[date, pd.Series]] = {n: {} for n in candidates}
    for i, d in enumerate(anchors):
        features = feature_loader(d, codes)
        if features.empty:
            continue
        universe = _universe_for_date(monthly_universe, d)
        if universe is not None:
            in_uni = [c for c in features.index if c in set(universe)]
            features = features.loc[in_uni]
        if features.empty:
            continue
        for name in candidates:
            if name in features.columns:
                factor_panels[name][d] = features[name].astype(float)
        if (i + 1) % 6 == 0 or i == len(anchors) - 1:
            logger.warning("  因子截面计算: %d/%d (%s)", i + 1, len(anchors), d)

    meta = {"candidates": candidates, "anchors": [str(d) for d in anchors],
            "provider": _provider(cfg), "config_path": config_path}
    return factor_panels, fwd_panels, meta


# ══════════════════════════════════════════════════════════════
# 报告
# ══════════════════════════════════════════════════════════════

def run_ic_analysis(
    config_path: str,
    icir_threshold: float = DEFAULT_ICIR_THRESHOLD,
    factor_names: list[str] | None = None,
) -> dict:
    """完整 IC 分析：数据组装 → 逐因子评估 → 汇总报告 dict。"""
    factor_panels, fwd_panels, meta = build_ic_dataset(config_path, factor_names)

    factors_report: dict[str, dict] = {}
    for name, panel in factor_panels.items():
        factors_report[name] = evaluate_factor(name, panel, fwd_panels, icir_threshold)

    passed = sorted([n for n, r in factors_report.items() if r["verdict"] == "pass"],
                    key=lambda n: -factors_report[n]["icir"])
    failed = sorted([n for n, r in factors_report.items() if r["verdict"] == "fail"],
                    key=lambda n: -factors_report[n]["icir"])
    return {
        "generated": str(date.today()),
        "config_path": meta["config_path"],
        "provider": meta["provider"],
        "rebalance": "monthly",
        "n_anchors": len(meta["anchors"]),
        "period": f"{meta['anchors'][0]} ~ {meta['anchors'][-1]}",
        "icir_threshold": icir_threshold,
        "factors": factors_report,
        "passed": passed,
        "failed": failed,
    }


def print_report(report: dict) -> None:
    factors = report["factors"]
    print(f"\n{'='*100}")
    print(f"  因子 RankIC 检验报告 — {Path(report['config_path']).stem}")
    print(f"  区间 {report['period']} | {report['n_anchors']} 个月度截面 | "
          f"provider={report['provider']} | ICIR 阈值 {report['icir_threshold']}")
    print(f"  （IC 已按因子方向调整：反向因子 dir=-1，好因子 IC/ICIR 恒为正）")
    print(f"{'='*100}")
    print(f"  {'因子':<28s} {'dir':>4s} {'IC均值':>7s} {'ICIR':>7s} {'t值':>6s} "
          f"{'正率':>5s} {'截面':>4s} {'IC衰减(1/3/6月)':>18s} {'Q5-Q1':>8s} {'单调':>5s} {'判定':<s}")
    print(f"  {'─'*28} {'─'*4} {'─'*7} {'─'*7} {'─'*6} {'─'*5} {'─'*4} {'─'*18} {'─'*8} {'─'*5} {'─'*4}")

    ordered = sorted(factors.items(), key=lambda kv: -kv[1]["icir"])
    for name, r in ordered:
        decay = r["ic_decay"]
        decay_str = "/".join(
            f"{decay.get(str(h), {}).get('icir', 0):+.2f}" for h in DECAY_HORIZONS)
        q = r["quintile"]
        spread = q["top_bottom_spread"]
        mark = "✓ pass" if r["verdict"] == "pass" else "✗ fail"
        print(f"  {name:<28s} {r['direction']:>+4d} {r['ic_mean']:>+7.3f} "
              f"{r['icir']:>+7.3f} {r['t_stat']:>6.2f} {r['positive_rate']:>5.0%} "
              f"{r['n_periods']:>4d} {decay_str:>18s} {spread:>+8.3%} "
              f"{q['monotonicity']:>+5.2f} {mark}")

    print(f"\n  通过 {len(report['passed'])} 个: {', '.join(report['passed']) or '（无）'}")
    print(f"  剔除 {len(report['failed'])} 个: {', '.join(report['failed']) or '（无）'}")
    print(f"\n  优化器接入（只在通过检验的因子里搜权重）:")
    print(f"    python research/factor_optimizer.py --task long_term \\")
    print(f"        --config {report['config_path']} \\")
    print(f"        --ic-report <本报告 json 路径>")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="因子 RankIC/ICIR/分层检验")
    parser.add_argument("--config", required=True, help="策略配置 yaml（决定股票池/区间/provider）")
    parser.add_argument("--threshold", type=float, default=DEFAULT_ICIR_THRESHOLD,
                        help=f"ICIR 通过阈值（默认 {DEFAULT_ICIR_THRESHOLD}）")
    parser.add_argument("--factors", default=None,
                        help="逗号分隔的因子名列表（默认 long_term 全候选池）")
    parser.add_argument("--out", default=None, help="报告 JSON 输出路径（默认 data/reports/factor_ic/）")
    args = parser.parse_args()

    names = [s.strip() for s in args.factors.split(",")] if args.factors else None
    report = run_ic_analysis(args.config, args.threshold, names)
    print_report(report)

    out = Path(args.out) if args.out else (
        REPORT_DIR / f"ic_{Path(args.config).stem.replace('strategy_', '')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n  报告已保存: {out}")


if __name__ == "__main__":
    main()
