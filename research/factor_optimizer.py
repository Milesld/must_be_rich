#!/usr/bin/env python3
"""因子组合优化器 — Optuna TPE 自适应搜索因子子集和权重。

用法:
    python research/factor_optimizer.py                          # 默认 100 轮，1年搜索窗口
    python research/factor_optimizer.py --rounds 200              # 200 轮
    python research/factor_optimizer.py --task long_term          # 限定长期因子池
    python research/factor_optimizer.py --task intraday --rounds 150
    python research/factor_optimizer.py --min-factors 3 --max-factors 6

加速技巧:
    --search-years 1     仅用最近1年数据搜索（快2-4倍），找到后全区间验证
    --search-years 0.5   仅用最近半年搜索（最快），Top-3 自动全区间验证

引擎:
    - optuna 已安装 → TPE 自适应搜索
    - optuna 未安装 → 随机搜索（狄利克雷权重分配）

安装 optuna: pip install optuna
"""

from __future__ import annotations

import copy
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import logging

import numpy as np
import yaml

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("optimizer")

# ══════════════════════════════════════════════════════════════
# 数据依赖标注
# ══════════════════════════════════════════════════════════════

NEEDS_FINANCIAL_DATA = {
    "pe_ttm", "pb", "ps_ttm", "pcf_ttm", "peg",
    "dividend_yield", "ep_ttm",
    "roic_ttm", "gross_margin_trend", "net_margin_ttm",
    "cf_ratio_ttm", "free_cf_yield",
}
NEEDS_MARGIN_DATA = {
    "main_force_net_inflow_5d", "main_force_net_inflow_20d",
    "main_force_inflow_ratio", "margin_balance_change_5d",
}
NEEDS_DRAGON_TIGER = {"dragon_tiger_net_buy", "dragon_tiger_institution_count"}
NEEDS_OVERSEAS = {
    "overnight_adr_mapped", "a50_futures_overnight", "hsi_futures_overnight",
}
NEEDS_L2_AUCTION = {
    "auction_open_premium", "auction_volume_ratio",
    "auction_strength_score", "auction_fake_order_risk",
}
NEEDS_NLP = {"announcement_sentiment_score", "theme_heat_score"}
NEEDS_MARKET_WIDE = {
    "limit_up_count", "limit_up_chain_height", "limit_down_count",
    "board_break_ratio", "limit_up_ratio",
}
NEEDS_QUARTERLY = {"northbound_quarter_change"}

ALL_NO_DATA = (
    NEEDS_FINANCIAL_DATA | NEEDS_MARGIN_DATA | NEEDS_DRAGON_TIGER
    | NEEDS_OVERSEAS | NEEDS_L2_AUCTION | NEEDS_NLP
    | NEEDS_MARKET_WIDE | NEEDS_QUARTERLY
)

# ══════════════════════════════════════════════════════════════
# 因子池
# ══════════════════════════════════════════════════════════════

FACTOR_POOL = {
    "long_term": {
        "label": "长期选股（月度调仓）",
        "candidates": [
            "momentum_20d", "momentum_60d", "alpha_momentum_20d",
            "volatility_20d", "amplitude_20d", "atr_14",
            "turnover_5d", "turnover_20d", "volume_ratio", "amihud_illiq",
            "rsi_14", "macd_dif", "bollinger_position", "ma_alignment",
            "roe_ttm", "roa_ttm",
            "revenue_yoy", "net_profit_yoy",
            "debt_ratio", "current_ratio", "quick_ratio",
        ],
    },
    "premarket": {
        "label": "盘前推荐（日频）",
        "candidates": [
            "momentum_20d", "volatility_20d", "volume_ratio", "turnover_5d",
        ],
    },
    "intraday": {
        "label": "日内预测（分钟级）",
        "candidates": [
            "momentum_20d", "volatility_20d", "atr_14", "amplitude_20d",
            "turnover_5d", "volume_ratio", "amihud_illiq",
            "rsi_14", "macd_dif", "bollinger_position", "ma_alignment",
        ],
    },
}

# ══════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════

def load_config(path: str = "configs/strategy.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


_backtest_cache: dict[str, dict] = {}


def _cache_key(enabled: list[str], weights: tuple, config_sig: str) -> str:
    return config_sig + "|" + "|".join(sorted(enabled))


def run_single_backtest(config: dict, label: str = "") -> dict | None:
    """用给定配置跑一次回测，返回指标字典。结果基于 config 签名缓存。"""
    from research.run_backtest_demo import (
        ConfigDrivenStrategy, _build_feature_loader,
        _load_real_data, _load_financials, _get_codes,
    )
    from core.backtest.engine import BacktestEngine

    enabled = sorted([n for n, c in config["factors"].items() if c.get("enabled")])
    bt = config["backtest"]
    sig = f"{bt['start_date']}|{bt['end_date']}|{bt['initial_capital']}"
    ck = _cache_key(enabled, tuple(), sig)
    if ck in _backtest_cache:
        return _backtest_cache[ck]

    start_date = _parse_date(bt["start_date"])
    end_date = _parse_date(bt["end_date"])

    raw_data, _label = _load_real_data(config)
    if not raw_data or len(raw_data) < 100:
        return None

    import pandas as pd

    def data_loader(trade_date: date):
        rows = [{"code": code, **fields} for code, fields in raw_data.get(trade_date, {}).items()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    codes = _get_codes(config)
    financials = _load_financials(codes) if codes else {}
    feature_loader = _build_feature_loader(raw_data, config, financials)
    strategy = ConfigDrivenStrategy(config, raw_data)
    engine = BacktestEngine(
        start_date=start_date, end_date=end_date,
        initial_capital=bt["initial_capital"],
    )
    result = engine.run(strategy, data_loader=data_loader, feature_loader=feature_loader)
    metrics = result.summary()
    if metrics and "error" not in metrics:
        _backtest_cache[ck] = metrics
    return metrics


# ══════════════════════════════════════════════════════════════
# 引擎: Optuna TPE
# ══════════════════════════════════════════════════════════════

def optimize_optuna(
    base_config: dict,
    candidates: list[str],
    n_trials: int = 100,
    min_factors: int = 2,
    max_factors: int = 10,
) -> list[dict] | None:
    try:
        import optuna
    except ImportError:
        return None

    all_factors = base_config.get("factors", {})

    def objective(trial: optuna.Trial) -> float:
        selected: list[str] = []
        for name in candidates:
            if trial.suggest_categorical(name, [True, False]):
                selected.append(name)
        if len(selected) < min_factors:
            remaining = [c for c in candidates if c not in selected]
            if remaining:
                idx = trial.suggest_categorical("_fill", list(range(len(remaining))))
                selected.append(remaining[idx % len(remaining)])
        if len(selected) > max_factors:
            selected = selected[:max_factors]
        if len(selected) < 2:
            return -999.0

        raw_weights = np.array([
            trial.suggest_float(f"_w_{name}", 0.1, 10.0, log=True)
            for name in selected
        ])
        weights = raw_weights / raw_weights.sum()

        cfg = copy.deepcopy(base_config)
        for name in all_factors:
            cfg["factors"][name]["enabled"] = False
        for name, w in zip(selected, weights):
            cfg["factors"][name]["enabled"] = True
            cfg["factors"][name]["weight"] = float(w)

        metrics = run_single_backtest(cfg)
        if metrics is None or "error" in metrics:
            return -999.0

        trial.set_user_attr("annual_return", float(metrics.get("annual_return", 0)))
        trial.set_user_attr("max_drawdown", float(metrics.get("max_drawdown", -1)))
        trial.set_user_attr("win_rate", str(metrics.get("win_rate", "N/A")))
        trial.set_user_attr("total_trades", int(metrics.get("total_trades", 0)))
        trial.set_user_attr("factors", "|".join(selected))
        trial.set_user_attr(
            "weights",
            "|".join(f"{n}={w:.3f}" for n, w in zip(selected, weights)),
        )

        sharpe = float(metrics.get("sharpe_ratio", -99))
        mdd = abs(float(metrics.get("max_drawdown", 0)))
        penalty = 0.0
        if mdd > 0.30:
            penalty += (mdd - 0.30) * 5.0
        if sharpe > 3.0:
            penalty += (sharpe - 3.0) * 2.0
        return sharpe - penalty

    print(f"\n[Optuna TPE] {n_trials} 轮, 因子 {min_factors}~{max_factors} 个")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    def _callback(study: optuna.Study, trial: optuna.Trial):
        if trial.number % 20 == 19 or trial.number == 0:
            nf = len(trial.user_attrs.get("factors", "").split("|")) if trial.user_attrs.get("factors") else 0
            print(f"  [{trial.number+1:3d}/{n_trials}] "
                  f"最佳夏普={study.best_value:.3f} (本组用了{nf}个因子)")

    study.optimize(objective, n_trials=n_trials, callbacks=[_callback], show_progress_bar=False)

    results: list[dict] = []
    # 基线
    enabled_baseline = [n for n, c in all_factors.items() if c.get("enabled")]
    baseline = run_single_backtest(base_config, label="基线")
    if baseline:
        results.append({
            "label": f"基线 ({len(enabled_baseline)}因子)",
            "factors": enabled_baseline,
            **{k: v for k, v in baseline.items() if isinstance(v, (int, float))},
        })

    trials = sorted(
        [t for t in study.trials if t.value is not None and t.value > -900],
        key=lambda t: t.value, reverse=True,
    )
    for rank, t in enumerate(trials[:30]):
        factors = t.user_attrs.get("factors", "").split("|") if t.user_attrs.get("factors") else []
        weights_str = t.user_attrs.get("weights", "")
        weights = {}
        if weights_str:
            for pair in weights_str.split("|"):
                if "=" in pair:
                    n, w = pair.split("=")
                    weights[n] = float(w)
        results.append({
            "label": f"Optuna #{rank+1} ({len(factors)}因子)",
            "factors": factors,
            "weights": weights,
            "sharpe_ratio": float(t.value),
            "annual_return": float(t.user_attrs.get("annual_return", 0)),
            "max_drawdown": float(t.user_attrs.get("max_drawdown", 0)),
            "win_rate": t.user_attrs.get("win_rate", "N/A"),
            "total_trades": int(t.user_attrs.get("total_trades", 0)),
        })

    return results


# ══════════════════════════════════════════════════════════════
# 引擎: 随机搜索（无 optuna 时回退）
# ══════════════════════════════════════════════════════════════

def optimize_random(
    base_config: dict,
    candidates: list[str],
    n_rounds: int = 200,
    min_factors: int = 2,
    max_factors: int = 10,
) -> list[dict]:
    all_factors = base_config.get("factors", {})
    rng = np.random.default_rng(42)
    results: list[dict] = []

    enabled_baseline = [n for n, c in all_factors.items() if c.get("enabled")]
    baseline = run_single_backtest(base_config, label="基线")
    if baseline:
        results.append({
            "label": f"基线 ({len(enabled_baseline)}因子)",
            "factors": enabled_baseline,
            **{k: v for k, v in baseline.items() if isinstance(v, (int, float))},
        })

    print(f"\n[随机搜索] {n_rounds} 轮, 因子 {min_factors}~{max_factors} 个")

    for i in range(n_rounds):
        nf = rng.integers(min_factors, max_factors + 1)
        selected = list(rng.choice(candidates, size=min(nf, len(candidates)), replace=False))
        raw = rng.dirichlet(np.ones(len(selected)))
        weights = {s: float(w) for s, w in zip(selected, raw)}

        cfg = copy.deepcopy(base_config)
        for name in all_factors:
            cfg["factors"][name]["enabled"] = False
        for name, w in weights.items():
            cfg["factors"][name]["enabled"] = True
            cfg["factors"][name]["weight"] = float(w)

        metrics = run_single_backtest(cfg)
        if metrics is None:
            continue

        results.append({
            "label": f"随机 #{i+1} ({len(selected)}因子)",
            "factors": list(selected),
            "weights": {k: round(v, 4) for k, v in weights.items()},
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        })

        if i % 20 == 19 or i == 0:
            best = max((r.get("sharpe_ratio", -99) for r in results[1:] if "sharpe_ratio" in r), default=-99)
            print(f"  [{i+1:3d}/{n_rounds}] 当前最佳夏普={best:.3f}")

    results[1:] = sorted(results[1:], key=lambda r: r.get("sharpe_ratio", -999), reverse=True)
    return results


# ══════════════════════════════════════════════════════════════
# 全区间验证（搜索完成后对 Top-N 做全区间验证）
# ══════════════════════════════════════════════════════════════

def _verify_top_on_full_window(
    config_path: str,
    best_from_search: dict,
) -> dict | None:
    """对搜索结果在完整区间上重跑一遍，返回真实指标。"""
    full_config = load_config(config_path)
    all_factors = full_config.get("factors", {})

    # 应用最佳组合的因子和权重
    for name in all_factors:
        full_config["factors"][name]["enabled"] = False
    factors = best_from_search.get("factors", [])
    weights = best_from_search.get("weights", {})
    for name in factors:
        full_config["factors"][name]["enabled"] = True
        full_config["factors"][name]["weight"] = float(weights.get(name, 1.0))

    # 恢复完整回测区间
    original_bt = load_config(config_path)["backtest"]
    full_config["backtest"] = original_bt

    full_metrics = run_single_backtest(full_config, label="全区间验证")
    return full_metrics


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def optimize(
    config_path: str = "configs/strategy.yaml",
    task: str | None = None,
    rounds: int = 100,
    min_factors: int = 2,
    max_factors: int = 10,
    search_years: float | None = None,
) -> list[dict]:
    """因子组合优化入口：短窗口搜索 + 全区间验证 Top-3。

    Args:
        config_path: 策略配置 YAML。
        task: 限定因子池。
        rounds: 搜索轮数。
        min_factors / max_factors: 因子数范围。
        search_years: 搜索用的窗口长度（年）。None=全区间, 0.5=半年, 1=1年。
                      缩短搜索窗口可加速 2~4 倍，Top-3 自动用全区间验证。
    """
    base_config = load_config(config_path)
    all_factors = base_config.get("factors", {})

    # 确定候选池，排除无数据源的因子
    if task and task in FACTOR_POOL:
        raw_candidates = [c for c in FACTOR_POOL[task]["candidates"] if c in all_factors]
        label = FACTOR_POOL[task]["label"]
    else:
        seen: set[str] = set()
        raw_candidates = []
        for pn in FACTOR_POOL:
            for c in FACTOR_POOL[pn]["candidates"]:
                if c in all_factors and c not in seen:
                    if "technical" in all_factors[c].get("module", ""):
                        raw_candidates.append(c)
                        seen.add(c)
        label = "全部技术面因子"

    candidates = [c for c in raw_candidates if c not in ALL_NO_DATA]
    skipped = [c for c in raw_candidates if c in ALL_NO_DATA]

    if skipped:
        print(f"  ⚠ 已排除 {len(skipped)} 个无数据源的因子")
    if len(candidates) < 3:
        print(f"候选因子不足（{len(candidates)}个），至少需要 3 个")
        return []

    # 缩短搜索窗口
    full_end = _parse_date(base_config["backtest"]["end_date"])
    if search_years and search_years > 0:
        search_days = int(search_years * 252)
        base_config["backtest"]["start_date"] = str(
            full_end - timedelta(days=search_days + base_config["data_source"].get("lookback_days", 400))
        )
        search_label = f"{search_years}年窗口搜索"
    else:
        search_label = "全区间搜索"

    print(f"\n{'='*60}")
    print(f"因子组合优化器 — {search_label}")
    print(f"  任务: {label}")
    print(f"  候选池: {len(candidates)} 个（已排除无数据源因子）")
    print(f"  搜索: {rounds} 轮, 因子 {min_factors}~{max_factors} 个")
    print(f"  区间: {base_config['backtest']['start_date']} ~ {base_config['backtest']['end_date']}")
    print(f"{'='*60}")

    # 跑基线
    baseline = run_single_backtest(base_config, label="基线")
    if baseline is None:
        print("数据加载失败！请确认网络正常。")
        return []
    print(f"  基线夏普: {baseline.get('sharpe_ratio', 0):.3f}")

    # 搜索
    optuna_results = optimize_optuna(base_config, candidates, rounds, min_factors, max_factors)
    if optuna_results is not None:
        print("  ✓ 使用 Optuna TPE 引擎")
        results = optuna_results
    else:
        print("  ⚠ Optuna 未安装，回退到随机搜索")
        results = optimize_random(base_config, candidates, rounds, min_factors, max_factors)

    # 如果是短窗口搜索，对 Top-3 做全区间验证
    if search_years and search_years > 0 and len(results) > 1:
        print(f"\n{'─'*60}")
        print("短窗口搜索完成，对 Top-3 做全区间验证...")
        verified: list[dict] = [results[0]]  # 保留基线
        for i, r in enumerate(results[1:4]):
            fm = _verify_top_on_full_window(config_path, r)
            if fm:
                verified.append({
                    "label": f"✓ 全区间验证 #{i+1} ({len(r['factors'])}因子)",
                    "factors": r["factors"],
                    "weights": r["weights"],
                    **{k: v for k, v in fm.items() if isinstance(v, (int, float))},
                })
        if len(verified) > 1:
            print(f"  验证完成: {len(verified)-1} 个组合")
            return verified

    return results


# ══════════════════════════════════════════════════════════════
# 报告
# ══════════════════════════════════════════════════════════════

def print_report(results: list[dict]) -> None:
    if not results:
        return

    baseline = results[0] if results else {}
    best = results[1] if len(results) > 1 else baseline
    improved = [r for r in results[1:]
                if r.get("sharpe_ratio", -99) > baseline.get("sharpe_ratio", -99)]

    print(f"\n{'='*100}")
    print(f"优化结果")
    print(f"{'='*100}")
    print(f"  基线夏普: {baseline.get('sharpe_ratio',0):.3f}"
          f"  |  最佳夏普: {best.get('sharpe_ratio',0):.3f}"
          f"  |  优于基线: {len(improved)}/{len(results)-1 if len(results)>1 else 0}")
    print(f"{'='*100}")
    print(f"  {'排':<3s} {'标签':<28s} {'夏普':>7s} {'年化':>7s} {'回撤':>7s} {'胜率':>7s} {'交易':>5s} {'组合':<s}")
    print(f"  {'─'*3} {'─'*28} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*5} {'─'*30}")

    for rank, r in enumerate(results[:20]):
        sharpe = r.get("sharpe_ratio", 0)
        ann_ret = r.get("annual_return", 0)
        mdd = r.get("max_drawdown", 0)
        wr = r.get("win_rate", 0)
        trades = r.get("total_trades", 0)

        if isinstance(wr, (int, float)):
            wr_str = f"{wr*100:.0f}%"
        elif isinstance(wr, str) and wr not in ("N/A",):
            try:
                wr_str = f"{float(wr)*100:.0f}%"
            except ValueError:
                wr_str = wr
        else:
            wr_str = str(wr)

        factors = r.get("factors", [])
        weights = r.get("weights", {})
        factor_str = ", ".join(
            f"{f}" + (f"({weights[f]:.2f})" if weights and f in weights else "")
            for f in factors[:5]
        )
        if len(factors) > 5:
            factor_str += f" ...共{len(factors)}个"

        print(
            f"  {rank+1:<3d} {r['label']:<28s} "
            f"{sharpe:7.3f} {ann_ret*100:6.1f}% {mdd*100:6.1f}% {wr_str:>7s} {trades:>5d} "
            f"{factor_str}"
        )

    # 过拟合诊断
    if len(results) > 1:
        top = results[1]
        sharpe = top.get("sharpe_ratio", 0)
        n_factors = len(top.get("factors", []))
        n_days = top.get("n_trading_days", 0)
        print(f"\n{'─'*100}")
        print("过拟合诊断:")
        checks = []
        if sharpe > 2.0:
            checks.append(f"✗ 夏普 {sharpe:.2f} > 2.0 — 极高概率过拟合，实盘预计衰减 40-60%")
        elif sharpe > 1.5:
            checks.append(f"△ 夏普 {sharpe:.2f} > 1.5 — 可能过拟合，建议 Walk-Forward 验证")
        if n_factors >= 8 and n_days < 1000:
            checks.append(f"△ {n_factors} 个因子 / {n_days} 个交易日 — 因子偏多")
        if not checks:
            checks.append("✓ 未检测到明显过拟合信号")
        for c in checks:
            print(f"  {c}")

    # Top-3 权重
    top3 = [r for r in results[1:4] if r.get("weights")]
    if top3:
        print(f"\n{'─'*100}")
        print("Top-3 权重详解:")
        for rank, r in enumerate(top3):
            weights = r.get("weights", {})
            sorted_w = sorted(weights.items(), key=lambda x: -x[1])
            w_str = " + ".join(f"{n}={w:.1%}" for n, w in sorted_w)
            print(f"  #{rank+1}: {w_str}")

    # 配置建议
    print(f"\n{'─'*100}")
    print("将最佳组合复制到 configs/strategy.yaml:")
    print()
    if len(results) > 1:
        top = results[1]
        w = top.get("weights", {})
        if w:
            print("  将所有因子设为 enabled: false，然后以下设为 enabled: true:")
            print()
            for name, weight in sorted(w.items(), key=lambda x: -x[1]):
                print(f"    {name}:")
                print(f"      enabled: true")
                print(f"      weight: {weight:.4f}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="因子组合优化器 — Optuna TPE + 短窗口搜索 + 全区间验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python research/factor_optimizer.py                                     # 默认 100 轮\n"
            "  python research/factor_optimizer.py --task long_term --rounds 200        # 长期因子\n"
            "  python research/factor_optimizer.py --task long_term --search-years 1     # 1年窗口加速\n"
            "  python research/factor_optimizer.py --task long_term --search-years 0.5 --rounds 300  # 半年最快\n"
        ),
    )
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--task", choices=["long_term", "premarket", "intraday", None], default=None)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--min-factors", type=int, default=2, dest="min_factors")
    parser.add_argument("--max-factors", type=int, default=6, dest="max_factors")
    parser.add_argument("--search-years", type=float, default=None, dest="search_years",
                        help="搜索用窗口长度（年），例如 1=1年, 0.5=半年。缩短可加速 2-4 倍")
    args = parser.parse_args()

    results = optimize(
        args.config, args.task, args.rounds,
        args.min_factors, args.max_factors,
        args.search_years,
    )
    print_report(results)


if __name__ == "__main__":
    main()
