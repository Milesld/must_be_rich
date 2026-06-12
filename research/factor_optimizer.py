#!/usr/bin/env python3
"""因子组合优化器 — Optuna TPE 自适应搜索因子子集和权重。

用法:
    python research/factor_optimizer.py                          # 默认 100 轮
    python research/factor_optimizer.py --rounds 200              # 200 轮
    python research/factor_optimizer.py --task long_term          # 限定长期因子池
    python research/factor_optimizer.py --task intraday --rounds 150
    python research/factor_optimizer.py --min-factors 3 --max-factors 6  # 3~6个因子

引擎:
    - optuna 已安装 → TPE (Tree-structured Parzen Estimator) 自适应搜索
    - optuna 未安装 → 随机搜索（狄利克雷权重分配），功能等价但收敛慢

原理:
    TPE 采样器能学习"哪些因子组合 + 哪些权重区间效果好"，
    后续迭代概率集中到高分区域搜索 → 收敛比随机快 3~10 倍。

安装 optuna（推荐，一条命令）:
    pip install optuna
"""

from __future__ import annotations

import copy
import sys
from datetime import date
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

# ── 因子池：按任务分类 ──────────────────
FACTOR_POOL = {
    "long_term": {
        "label": "长期选股（月度调仓）",
        "candidates": [
            "momentum_20d", "momentum_60d", "alpha_momentum_20d",
            "volatility_20d", "amplitude_20d", "atr_14",
            "turnover_5d", "turnover_20d", "volume_ratio", "amihud_illiq",
            "rsi_14", "macd_dif", "bollinger_position", "ma_alignment",
            "pe_ttm", "pb", "ps_ttm", "pcf_ttm", "peg",
            "dividend_yield", "ep_ttm",
            "roe_ttm", "roa_ttm", "roic_ttm",
            "gross_margin_trend", "net_margin_ttm",
            "revenue_yoy", "net_profit_yoy",
            "debt_ratio", "current_ratio", "quick_ratio",
            "cf_ratio_ttm", "free_cf_yield",
            "northbound_quarter_change",
        ],
    },
    "premarket": {
        "label": "盘前推荐（日频）",
        "candidates": [
            "overnight_adr_mapped", "a50_futures_overnight", "hsi_futures_overnight",
            "announcement_sentiment_score", "dragon_tiger_review_score",
            "limit_up_review_signal", "theme_heat_score",
            "auction_strength_score", "auction_fake_order_risk",
            "days_to_next_event",
            "momentum_20d", "volatility_20d", "volume_ratio", "turnover_5d",
            "limit_up_ratio", "performance_forecast_surprise",
        ],
    },
    "intraday": {
        "label": "日内预测（分钟级）",
        "candidates": [
            "momentum_20d", "volatility_20d", "atr_14", "amplitude_20d",
            "turnover_5d", "volume_ratio", "amihud_illiq",
            "rsi_14", "macd_dif", "bollinger_position", "ma_alignment",
            "limit_up_count", "limit_up_chain_height", "limit_down_count",
            "board_break_ratio",
            "auction_open_premium", "auction_volume_ratio",
            "main_force_inflow_ratio", "margin_balance_change_5d",
        ],
    },
}

# ── 配置/回测工具 ──────────────────────

def load_config(path: str = "configs/strategy.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


# 回测结果缓存（避免同一组合跑多次）
_backtest_cache: dict[str, dict] = {}


def _cache_key(enabled: list[str], weights: tuple) -> str:
    return "|".join(sorted(enabled)) + "|" + "|".join(
        f"{w:.4f}" for w in sorted(weights)
    )


def run_single_backtest(config: dict) -> dict | None:
    """用给定配置跑一次回测，返回指标字典。"""
    from research.run_backtest_demo import (
        ConfigDrivenStrategy,
        _build_feature_loader,
        _load_real_data,
    )
    from core.backtest.engine import BacktestEngine

    # 检查缓存
    enabled = sorted([n for n, c in config["factors"].items() if c.get("enabled")])
    weights_tuple = tuple(
        config["factors"][n].get("weight", 0) for n in enabled
    )
    ck = _cache_key(enabled, weights_tuple)
    if ck in _backtest_cache:
        return _backtest_cache[ck]

    cfg_bt = config["backtest"]
    start_date = _parse_date(cfg_bt["start_date"])
    end_date = _parse_date(cfg_bt["end_date"])

    raw_data, _label = _load_real_data(config)
    if not raw_data or len(raw_data) < 100:
        return None

    import pandas as pd

    def data_loader(trade_date: date):
        rows = [{"code": code, **fields} for code, fields in raw_data.get(trade_date, {}).items()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    feature_loader = _build_feature_loader(raw_data, config)
    strategy = ConfigDrivenStrategy(config)
    engine = BacktestEngine(
        start_date=start_date, end_date=end_date,
        initial_capital=cfg_bt["initial_capital"],
    )
    result = engine.run(strategy, data_loader=data_loader, feature_loader=feature_loader)
    metrics = result.summary()
    if metrics and "error" not in metrics:
        _backtest_cache[ck] = metrics
    return metrics


# ══════════════════════════════════════════════════════════════
# 引擎 1: Optuna TPE 自适应搜索（主引擎）
# ══════════════════════════════════════════════════════════════

def _try_import_optuna():
    try:
        import optuna
        return optuna
    except ImportError:
        return None


def optimize_optuna(
    base_config: dict,
    candidates: list[str],
    n_trials: int = 100,
    min_factors: int = 2,
    max_factors: int = 10,
) -> list[dict]:
    """Optuna TPE 自适应搜索。"""
    optuna = _try_import_optuna()
    if optuna is None:
        return None  # 回退到随机搜索

    all_factors = base_config.get("factors", {})

    def objective(trial: optuna.Trial) -> float:
        # Step 1: 选 k 个因子（TPE 对每个因子学习"选它"的概率）
        selected: list[str] = []
        for name in candidates:
            if trial.suggest_categorical(name, [True, False]):
                selected.append(name)

        # 确保因子数在 [min, max] 范围内
        if len(selected) < min_factors:
            # 从剩余候选里补足
            remaining = [c for c in candidates if c not in selected]
            if remaining:
                picked = trial.suggest_categorical(
                    "_fill", list(range(len(remaining)))
                )
                selected.append(remaining[picked % len(remaining)])
        if len(selected) > max_factors:
            selected = selected[:max_factors]
        if len(selected) < 2:
            return -999.0  # 不够因子的直接拦截

        # Step 2: 分配权重（TPE 对每个因子学习"多少权重好"）
        raw_weights = np.array([
            trial.suggest_float(f"_w_{name}", 0.1, 10.0, log=True)
            for name in selected
        ])
        weights = raw_weights / raw_weights.sum()

        # Step 3: 构造配置
        cfg = copy.deepcopy(base_config)
        for name in all_factors:
            cfg["factors"][name]["enabled"] = False
        for name, w in zip(selected, weights):
            cfg["factors"][name]["enabled"] = True
            cfg["factors"][name]["weight"] = float(w)

        # Step 4: 回测
        metrics = run_single_backtest(cfg)
        if metrics is None or "error" in metrics:
            return -999.0

        # 记录额外指标
        trial.set_user_attr("annual_return", float(metrics.get("annual_return", 0)))
        trial.set_user_attr("max_drawdown", float(metrics.get("max_drawdown", -1)))
        trial.set_user_attr("win_rate", str(metrics.get("win_rate", "N/A")))
        trial.set_user_attr("factors", "|".join(selected))
        trial.set_user_attr(
            "weights",
            "|".join(f"{n}={w:.3f}" for n, w in zip(selected, weights)),
        )

        # 优化目标：最大化夏普（penalize 极端回撤）
        sharpe = float(metrics.get("sharpe_ratio", -99))
        mdd = abs(float(metrics.get("max_drawdown", 0)))
        # 惩罚项：回撤 > 30% 或夏普异常高（>3，几乎一定是过拟合）
        penalty = 0.0
        if mdd > 0.30:
            penalty += (mdd - 0.30) * 5.0
        if sharpe > 3.0:
            penalty += (sharpe - 3.0) * 2.0
        return sharpe - penalty

    print(f"\n[Optuna TPE] 开始搜索: {n_trials} 轮, 因子 {min_factors}~{max_factors} 个")
    print(f"[Optuna TPE] 前 20 轮探索空间, 后续收敛到高分区域")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    def _callback(study: optuna.Study, trial: optuna.Trial):
        if trial.number % 20 == 19 or trial.number == 0:
            best = study.best_value
            print(f"  [{trial.number+1:3d}/{n_trials}] 当前最佳夏普={best:.3f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[_callback], show_progress_bar=False)

    # 收集结果
    results: list[dict] = []
    # 基线
    enabled_baseline = [n for n, c in all_factors.items() if c.get("enabled")]
    baseline = run_single_backtest(base_config)
    if baseline:
        results.append({
            "label": f"基线 ({len(enabled_baseline)}因子)",
            "factors": enabled_baseline,
            **{k: v for k, v in baseline.items() if isinstance(v, (int, float))},
        })

    # Optuna 结果（按 value 降序）
    trials = sorted(
        [t for t in study.trials if t.value is not None and t.value > -900],
        key=lambda t: t.value,
        reverse=True,
    )
    for rank, t in enumerate(trials[:50]):
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
        })

    return results


# ══════════════════════════════════════════════════════════════
# 引擎 2: 随机搜索（optuna 未安装时的后备）
# ══════════════════════════════════════════════════════════════

def optimize_random(
    base_config: dict,
    candidates: list[str],
    n_rounds: int = 200,
    min_factors: int = 2,
    max_factors: int = 10,
) -> list[dict]:
    """随机因子子集 + Dirichlet 权重搜索（无 optuna 时的后备方案）。"""
    all_factors = base_config.get("factors", {})
    rng = np.random.default_rng(42)
    results: list[dict] = []

    # 基线
    enabled_baseline = [n for n, c in all_factors.items() if c.get("enabled")]
    baseline = run_single_backtest(base_config)
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
# 主入口
# ══════════════════════════════════════════════════════════════

def optimize(
    config_path: str = "configs/strategy.yaml",
    task: str | None = None,
    rounds: int = 100,
    min_factors: int = 2,
    max_factors: int = 10,
) -> list[dict]:
    """因子组合优化入口：优先 Optuna TPE，回退随机搜索。"""
    base_config = load_config(config_path)
    all_factors = base_config.get("factors", {})

    # 确定候选池
    if task and task in FACTOR_POOL:
        candidates = [c for c in FACTOR_POOL[task]["candidates"] if c in all_factors]
        label = FACTOR_POOL[task]["label"]
    else:
        # 所有技术面因子（避免需额外数据的因子）
        seen: set[str] = set()
        candidates = []
        for pool_name in FACTOR_POOL:
            for c in FACTOR_POOL[pool_name]["candidates"]:
                if c in all_factors and c not in seen:
                    module = all_factors[c].get("module", "")
                    if "technical" in module:
                        candidates.append(c)
                        seen.add(c)
        label = "全部技术面因子"

    if len(candidates) < 3:
        print(f"候选因子不足（{len(candidates)}个），至少需要 3 个")
        return []

    print(f"\n{'='*60}")
    print(f"因子组合优化器")
    print(f"  任务: {label}")
    print(f"  候选池: {len(candidates)} 个")
    print(f"  搜索轮数: {rounds}")
    print(f"  每次因子数: {min_factors}~{max_factors}")
    print(f"{'='*60}")

    print("  正在加载数据并跑基线回测...")
    baseline = run_single_backtest(base_config)
    if baseline is None:
        print("  数据加载失败！请确认网络正常、配置文件正确。")
        return []
    print(f"  基线夏普: {baseline.get('sharpe_ratio', 0):.3f}")

    # 尝试 Optuna
    optuna_results = optimize_optuna(base_config, candidates, rounds, min_factors, max_factors)
    if optuna_results is not None:
        print(f"  ✓ 使用 Optuna TPE 引擎")
        return optuna_results

    # 回退随机
    print(f"  ⚠ Optuna 未安装，回退到随机搜索（pip install optuna 可加速收敛）")
    return optimize_random(base_config, candidates, rounds, min_factors, max_factors)


# ══════════════════════════════════════════════════════════════
# 报告输出
# ══════════════════════════════════════════════════════════════

def print_report(results: list[dict]) -> None:
    if not results:
        return

    baseline = results[0] if results else {}
    best = results[1] if len(results) > 1 else baseline
    improved = [r for r in results[1:]
                if r.get("sharpe_ratio", -99) > baseline.get("sharpe_ratio", -99)]

    print(f"\n{'='*95}")
    print(f"优化结果")
    print(f"{'='*95}")
    print(f"  基线夏普: {baseline.get('sharpe_ratio',0):.3f}"
          f"  |  最佳夏普: {best.get('sharpe_ratio',0):.3f}"
          f"  |  优于基线: {len(improved)}/{len(results)-1 if len(results)>1 else 0}")
    print(f"{'='*95}")
    print(f"  {'排':<3s} {'标签':<25s} {'夏普':>7s} {'年化':>7s} {'回撤':>7s} {'胜率':>7s} {'组合':<s}")
    print(f"  {'─'*3} {'─'*25} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*30}")

    for rank, r in enumerate(results[:20]):
        sharpe = r.get("sharpe_ratio", 0)
        ann_ret = r.get("annual_return", 0)
        mdd = r.get("max_drawdown", 0)
        wr = r.get("win_rate", 0)
        wr_str = f"{wr*100:.0f}%" if isinstance(wr, (int, float)) else str(wr)
        factors = r.get("factors", [])
        weights = r.get("weights", {})
        factor_str = ", ".join(
            f"{f}" + (f"({weights[f]:.2f})" if weights and f in weights else "")
            for f in factors[:5]
        )
        if len(factors) > 5:
            factor_str += f" ...共{len(factors)}个"

        print(
            f"  {rank+1:<3d} {r['label']:<25s} "
            f"{sharpe:7.3f} {ann_ret*100:6.1f}% {mdd*100:6.1f}% {wr_str:>7s} "
            f"{factor_str}"
        )

    # Top-3 详细权重
    top3 = [r for r in results[1:4] if r.get("weights")]
    if top3:
        print(f"\n{'─'*95}")
        print("Top-3 组合权重详解:")
        for rank, r in enumerate(top3):
            weights = r.get("weights", {})
            sorted_w = sorted(weights.items(), key=lambda x: -x[1])
            w_str = " + ".join(f"{n}={w:.1%}" for n, w in sorted_w)
            print(f"  #{rank+1}: {w_str}")

    # 建议
    print(f"\n{'─'*95}")
    print("将最佳组合的因子和权重复制到 configs/strategy.yaml 即可固定使用:")
    print()
    if len(results) > 1:
        top = results[1]
        w = top.get("weights", {})
        if w:
            print("  第一步: 把所有因子设为 enabled: false")
            print("  第二步: 将以下因子设为 enabled: true 并填入 weight:")
            print()
            for name, weight in sorted(w.items(), key=lambda x: -x[1]):
                print(f"    {name}:")
                print(f"      enabled: true")
                print(f"      weight: {weight:.4f}")
        else:
            print(f"  将以下因子设为 enabled: true:")
            for name in top.get("factors", []):
                print(f"    - {name}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="因子组合优化器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python research/factor_optimizer.py                          # 默认 100 轮\n"
            "  python research/factor_optimizer.py --task long_term          # 仅长期因子\n"
            "  python research/factor_optimizer.py --task intraday --rounds 200\n"
            "  python research/factor_optimizer.py --min-factors 3 --max-factors 6  # 精简组合\n"
            "\n安装 optuna 可加速收敛 3-10 倍: pip install optuna"
        ),
    )
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--task", choices=["long_term", "premarket", "intraday", None], default=None)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--min-factors", type=int, default=2, dest="min_factors")
    parser.add_argument("--max-factors", type=int, default=10, dest="max_factors")
    args = parser.parse_args()

    results = optimize(args.config, args.task, args.rounds, args.min_factors, args.max_factors)
    print_report(results)


if __name__ == "__main__":
    main()
