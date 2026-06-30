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

时间衰减加权:
    在 strategy_xxx.yaml 中配置 optimizer.time_decay_halflife（年），
    如 1.0 表示一年前的交易日权重为今天的 50%。值越小越偏好近期。
    不配置则使用传统等权夏普（向后兼容）。

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
# 以下因子来自同花顺 ths 接口，数据精度不足（CONTEXT.md #92 记录），
# 在准确财报数据接入前必须排除，否则 optimizer 会被假信号误导。
NEEDS_THS_FUNDAMENTAL = {
    "roe_ttm", "roa_ttm",
    "revenue_yoy", "net_profit_yoy",
    "debt_ratio", "current_ratio", "quick_ratio",
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

# 注意：NEEDS_OVERSEAS 和 NEEDS_MARKET_WIDE 已实现数据源，
# 不再排除。NEEDS_NLP 中只保留 theme_heat_score，
# announcement_sentiment_score 也已实现（从财务数据构造代理文本）。
NLP_NO_DATA = {"theme_heat_score"}

ALL_NO_DATA = (
    NEEDS_FINANCIAL_DATA | NEEDS_THS_FUNDAMENTAL | NEEDS_MARGIN_DATA
    | NEEDS_DRAGON_TIGER | NEEDS_L2_AUCTION | NLP_NO_DATA
    | NEEDS_QUARTERLY
)

# ══════════════════════════════════════════════════════════════
# 因子池
# ══════════════════════════════════════════════════════════════

FACTOR_POOL = {
    "long_term": {
        "label": "长期选股（月度调仓）",
        "candidates": [
            # 纯技术面（14个）— 全部从OHLCV计算，数据可靠
            "momentum_20d", "momentum_60d", "alpha_momentum_20d",
            "volatility_20d", "amplitude_20d", "atr_14",
            "turnover_5d", "turnover_20d", "volume_ratio", "amihud_illiq",
            "rsi_14", "macd_dif", "bollinger_position", "ma_alignment",
            # 板块内横截面（1个）— 从OHLCV推算，数据可靠
            "sector_relative_strength_20d",
            # 基本面（2个）— 仅 westock provider 有真数据（真 ROE/营收同比，
            # 按 InfoPublDate PIT 对齐）；非 westock 时仍被 ALL_NO_DATA 过滤。
            "roe_ttm", "revenue_yoy",
        ],
    },
    "premarket": {
        "label": "盘前推荐（日频）",
        "candidates": [
            # 技术面（7个）
            "momentum_20d", "momentum_60d", "volatility_20d", "volume_ratio",
            "turnover_5d", "rsi_14", "bollinger_position",
            # 海外映射（3个）
            "overnight_adr_mapped", "a50_futures_overnight", "hsi_futures_overnight",
            # 全市场情绪（5个）
            "limit_up_count", "limit_up_chain_height", "limit_down_count",
            "board_break_ratio", "limit_up_ratio",
            # NLP情绪（1个）
            "announcement_sentiment_score",
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


def _cache_key(enabled: list[str], weights: dict[str, float], config_sig: str,
               halflife: float | None = None) -> str:
    # 缓存键包含因子名、权重和半衰期
    weight_str = "|".join(f"{n}={weights.get(n,0):.3f}" for n in sorted(enabled))
    hl_str = f"|hl={halflife:.2f}" if halflife else ""
    return config_sig + "|" + weight_str + hl_str


def _compute_time_weighted_sharpe(daily_returns: "pd.Series", end_date: date,
                                  halflife_years: float) -> float:
    """计算时间衰减加权夏普比率（EWMA 思想扩展到夏普）。

    近期的收益率赋更高权重，远期赋更低权重。
    权重 = exp(-ln(2) × days_from_end / half_life_days)

    Args:
        daily_returns: 日收益率 Series，index 为 date（从回测 daily_nav 获取）。
        end_date: 回测结束日期（权重从这一天往回衰减）。
        halflife_years: 半衰期（年）。1.0 表示 252 个交易日前的权重 = 今天的一半。

    Returns:
        时间加权年化夏普比率。
    """
    import math

    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 20:
        return 0.0

    half_life_days = halflife_years * 252

    # 计算每个交易日距结束日的自然日距离
    weights = []
    for d in daily_returns.index:
        td = d if isinstance(d, date) else (d.date() if hasattr(d, "date") else d)
        if isinstance(td, str):
            td = date.fromisoformat(td)
        days = (end_date - td).days
        # 权重衰减：半衰期内权重减半
        w = math.exp(-math.log(2) * max(days, 0) / half_life_days)
        weights.append(w)
    weights_arr = np.array(weights, dtype=float)

    if weights_arr.sum() < 1e-10:
        return 0.0

    rets = daily_returns.values.astype(float)

    # 加权日均收益
    weighted_mean_ret = np.average(rets, weights=weights_arr)
    # 加权日均波动率
    weighted_var = np.average((rets - weighted_mean_ret) ** 2, weights=weights_arr)

    ann_ret = weighted_mean_ret * 252
    ann_vol = np.sqrt(weighted_var * 252) if weighted_var > 0 else 0.0
    if ann_vol <= 0:
        return 0.0
    return float((ann_ret - 0.02) / ann_vol)


def _load_shared_data(config: dict) -> dict | None:
    """整轮优化只调用一次：加载与因子组合无关的所有数据。

    价格、海外指数、全市场情绪、宽基指数、动态宇宙都只取决于股票池和日期，
    与"选了哪些因子"无关。把它们提到优化循环外只加载一次，避免每个候选
    因子组合都重复联网拉取（之前 100 轮 = 最多 100 次全量网络加载）。

    注意：不加载基本面数据（_load_financials）——其输出（ROE/营收增速等）
    全部在 ALL_NO_DATA 中被过滤，且 ROE 是伪造值，加载纯属浪费。

    Returns:
        {raw_data, overseas_data, market_wide_data, benchmark_index, monthly_universe}
        或 None（数据加载失败）。
    """
    from research.run_backtest_demo import (
        _load_real_data, _load_overseas_data, _build_market_wide_from_pool,
        _load_benchmark_index, _build_monthly_universe, _provider,
    )

    bt = config["backtest"]
    start_date = _parse_date(bt["start_date"])
    end_date = _parse_date(bt["end_date"])

    raw_data, _label = _load_real_data(config)
    if not raw_data or len(raw_data) < 100:
        return None

    # westock 模式加载真财报时间序列（PIT 对齐用）；其它模式为空
    financials = {}
    if _provider(config) == "westock":
        from research.westock_source import westock_financials
        from research.run_backtest_demo import _get_codes
        financials = westock_financials(_get_codes(config))

    return {
        "raw_data": raw_data,
        "overseas_data": _load_overseas_data(start_date, end_date),
        "market_wide_data": _build_market_wide_from_pool(raw_data),
        "benchmark_index": _load_benchmark_index(start_date, end_date, provider=_provider(config)),
        "monthly_universe": _build_monthly_universe(raw_data, config),
        "financials": financials,
    }


def run_single_backtest(config: dict, label: str = "",
                       halflife_years: float | None = None,
                       shared_data: dict | None = None) -> dict | None:
    """用给定配置跑一次回测，返回指标字典。结果基于 config 签名缓存。

    Args:
        config: 策略配置字典。
        label: 标签（用于日志，可选）。
        halflife_years: 时间衰减半衰期（年）。None = 传统等权夏普。
        shared_data: 由 _load_shared_data 预加载的共享数据包。为 None 时
                     回退到自行加载（向后兼容，仅用于独立调用场景）。

    Returns:
        指标字典，其中 sharpe_ratio 已根据 halflife_years 计算（如果设置）。
    """
    from research.run_backtest_demo import (
        ConfigDrivenStrategy, _build_feature_loader,
    )
    from core.backtest.engine import BacktestEngine

    enabled = sorted([n for n, c in config["factors"].items() if c.get("enabled")])
    weights = {n: config["factors"][n].get("weight", 0) for n in enabled}
    bt = config["backtest"]
    sig = f"{bt['start_date']}|{bt['end_date']}|{bt['initial_capital']}"
    ck = _cache_key(enabled, weights, sig, halflife_years)
    if ck in _backtest_cache:
        return _backtest_cache[ck]

    start_date = _parse_date(bt["start_date"])
    end_date = _parse_date(bt["end_date"])

    # 共享数据缺省时回退到自行加载（独立调用场景）
    if shared_data is None:
        shared_data = _load_shared_data(config)
        if shared_data is None:
            return None

    raw_data = shared_data["raw_data"]
    overseas_data = shared_data["overseas_data"]
    market_wide_data = shared_data["market_wide_data"]
    benchmark_index = shared_data.get("benchmark_index")
    monthly_universe = shared_data.get("monthly_universe")
    financials = shared_data.get("financials") or {}

    import pandas as pd

    def data_loader(trade_date: date):
        rows = [{"code": code, **fields} for code, fields in raw_data.get(trade_date, {}).items()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # 基本面财报（westock 时间序列，PIT 对齐）；非 westock 为空
    feature_loader = _build_feature_loader(raw_data, config, financials,
                                           overseas_data, market_wide_data,
                                           monthly_universe=monthly_universe)
    strategy = ConfigDrivenStrategy(config, raw_data, overseas_data,
                                    benchmark_index=benchmark_index,
                                    monthly_universe=monthly_universe)
    engine = BacktestEngine(
        start_date=start_date, end_date=end_date,
        initial_capital=bt["initial_capital"],
    )
    # 注入沪深300日收盘做相对超额基准
    if benchmark_index and "csi300" in benchmark_index:
        _bdf = benchmark_index["csi300"]
        engine.benchmark_close = {
            (d.date() if hasattr(d, "date") else d): float(c)
            for d, c in _bdf["close"].items()
        }
    result = engine.run(strategy, data_loader=data_loader, feature_loader=feature_loader)
    metrics = result.summary()
    if metrics and "error" not in metrics:
        # ★ 时间衰减加权夏普：从 daily_nav 取日收益序列，按 halflife 衰减
        if halflife_years and halflife_years > 0:
            nav_df = result.daily_nav
            if not nav_df.empty and "daily_return" in nav_df.columns:
                tw_sharpe = _compute_time_weighted_sharpe(
                    nav_df["daily_return"], end_date, halflife_years,
                )
                metrics["sharpe_ratio"] = tw_sharpe
                metrics["time_weighted_sharpe"] = True
                metrics["halflife_years"] = halflife_years
        _backtest_cache[ck] = metrics
    return metrics


# ══════════════════════════════════════════════════════════════
# Walk-Forward 样本外评分
# ══════════════════════════════════════════════════════════════

def _walk_forward_score(
    cfg: dict,
    shared_data: dict,
    halflife_years: float | None,
    wf_lambda: float = 0.5,
) -> dict | None:
    """对一个因子组合做 Walk-Forward 样本外评分。

    用样本外稳定性替代"在训练集上调参再惩罚高夏普"的治标手段：
    复用 core.backtest.walk_forward 的滚动窗口，对每段 test 窗口跑回测，
    取各段样本外夏普 S_i，目标值 = mean(S_i) − λ·std(S_i)。

    因子组合+权重已固定（即"训练产物"），所以每段只需在 test 区间跑一次回测，
    无需再训练模型。std 项惩罚"只有某几期爆发、其余拉胯"的不稳定组合，
    天然防过拟合，不再需要人为压制高夏普。

    Returns:
        {"score": 目标值, "wf_sharpes": [S_i...], "wf_periods": ["test起~止"...],
         "wf_mean": mean, "wf_std": std}，无有效窗口时返回 None。
    """
    import copy as _copy

    from core.backtest.walk_forward import WalkForwardValidator

    bt = cfg["backtest"]
    full_start = _parse_date(bt["start_date"])
    full_end = _parse_date(bt["end_date"])

    wf_cfg = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer"), dict) else {}
    validator = WalkForwardValidator(
        train_window_years=wf_cfg.get("wf_train_years", 2),
        test_window_months=wf_cfg.get("wf_test_months", 3),
        purge_days=wf_cfg.get("wf_purge_days", 5),
        min_train_years=wf_cfg.get("wf_min_train_years", 1),
    )
    windows = validator.get_windows(full_start, full_end)
    if not windows:
        return None

    sharpes: list[float] = []
    periods: list[str] = []
    for (_train_s, _train_e, test_s, test_e) in windows:
        # 在 test 区间上跑回测：因子组合固定，只改回测窗口
        seg_cfg = _copy.deepcopy(cfg)
        seg_cfg["backtest"]["start_date"] = str(test_s)
        seg_cfg["backtest"]["end_date"] = str(test_e)
        m = run_single_backtest(seg_cfg, halflife_years=halflife_years,
                                shared_data=shared_data)
        if m is None or "error" in m:
            continue
        sharpes.append(float(m.get("sharpe_ratio", 0.0)))
        periods.append(f"{test_s}~{test_e}")

    if len(sharpes) < 2:
        return None

    arr = np.array(sharpes, dtype=float)
    mean_s = float(arr.mean())
    std_s = float(arr.std())
    return {
        "score": mean_s - wf_lambda * std_s,
        "wf_sharpes": sharpes,
        "wf_periods": periods,
        "wf_mean": mean_s,
        "wf_std": std_s,
    }


# ══════════════════════════════════════════════════════════════
# 引擎: Optuna TPE
# ══════════════════════════════════════════════════════════════

def optimize_optuna(
    base_config: dict,
    candidates: list[str],
    n_trials: int = 100,
    min_factors: int = 2,
    max_factors: int = 10,
    factor_max_weight: dict[str, float] | None = None,
    halflife_years: float | None = None,
    shared_data: dict | None = None,
) -> list[dict] | None:
    try:
        import optuna
    except ImportError:
        return None

    all_factors = base_config.get("factors", {})
    opt_cfg = base_config.get("optimizer", {}) if isinstance(base_config.get("optimizer"), dict) else {}
    use_wf = bool(opt_cfg.get("use_walk_forward", False))
    wf_lambda = float(opt_cfg.get("wf_lambda", 0.5))

    def objective(trial: optuna.Trial) -> float:
        selected: list[str] = []
        for name in candidates:
            if trial.suggest_categorical(name, [True, False]):
                selected.append(name)
        if len(selected) < min_factors:
            remaining = [c for c in candidates if c not in selected]
            if remaining:
                # suggest_int 范围固定为候选池总长度，避免 CategoricalDistribution 动态报错
                idx = trial.suggest_int("_fill", 0, len(candidates) - 1)
                # 优先取 remaining 中最接近 idx 位置的因子
                pick = remaining[min(idx, len(remaining) - 1)]
                selected.append(pick)
        if len(selected) > max_factors:
            selected = selected[:max_factors]
        if len(selected) < 2:
            return -999.0

        raw_weights = np.array([
            trial.suggest_float(f"_w_{name}", 0.1, 10.0, log=True)
            for name in selected
        ])
        weights = raw_weights / raw_weights.sum()
        weights_float = [float(w) for w in weights]

        # 单因子权重上限：防止 optimizer 把宝全押在一个因子上
        # 默认: amihud_illiq 0.20, 其他 0.40。可通过 factor_max_weight 按策略覆盖
        DEFAULT_MAX_WEIGHT = 0.40
        FACTOR_MAX_WEIGHT: dict[str, float] = {"amihud_illiq": 0.20}
        if factor_max_weight:
            FACTOR_MAX_WEIGHT.update(factor_max_weight)
        MAX_REDISTRIBUTE = 5  # 最多重分配 5 次以防死循环
        for _ in range(MAX_REDISTRIBUTE):
            over: list[int] = []
            excess_total = 0.0
            for i, name in enumerate(selected):
                cap = FACTOR_MAX_WEIGHT.get(name, DEFAULT_MAX_WEIGHT)
                if weights_float[i] > cap:
                    excess = weights_float[i] - cap
                    weights_float[i] = cap
                    excess_total += excess
                    over.append(i)
            if excess_total < 0.001:
                break
            # 将超出部分按比例分给未触及上限的因子
            eligible = [i for i in range(len(selected)) if i not in over]
            if not eligible:
                break
            eligible_total = sum(weights_float[i] for i in eligible)
            if eligible_total < 0.001:
                break
            for i in eligible:
                weights_float[i] += excess_total * (weights_float[i] / eligible_total)

        cfg = copy.deepcopy(base_config)
        for name in all_factors:
            cfg["factors"][name]["enabled"] = False
        for name, w in zip(selected, weights_float):
            cfg["factors"][name]["enabled"] = True
            cfg["factors"][name]["weight"] = float(w)

        metrics = run_single_backtest(cfg, halflife_years=halflife_years, shared_data=shared_data)
        if metrics is None or "error" in metrics:
            return -999.0

        trial.set_user_attr("annual_return", float(metrics.get("annual_return", 0)))
        trial.set_user_attr("max_drawdown", float(metrics.get("max_drawdown", -1)))
        trial.set_user_attr("win_rate", str(metrics.get("win_rate", "N/A")))
        trial.set_user_attr("total_trades", int(metrics.get("total_trades", 0)))
        trial.set_user_attr("n_trading_days", int(metrics.get("n_trading_days", 0)))
        trial.set_user_attr("factors", "|".join(selected))
        trial.set_user_attr(
            "weights",
            "|".join(f"{n}={w:.3f}" for n, w in zip(selected, weights_float)),
        )

        sharpe = float(metrics.get("sharpe_ratio", -99))
        mdd = abs(float(metrics.get("max_drawdown", 0)))
        total_trades = int(metrics.get("total_trades", 0))
        n_trading_days = int(metrics.get("n_trading_days", 0))
        # 全区间真实夏普（无论 WF 与否都记录，供报告显示，避免把"目标分"当夏普）
        trial.set_user_attr("full_sharpe", sharpe)

        # ── 目标信号 ──
        # use_walk_forward=True：用样本外夏普均值 − λ·标准差，天然防过拟合，
        #   不再需要人为压制高夏普（std 项已惩罚不稳定的爆发型组合）。
        # use_walk_forward=False：沿用旧的"单区间夏普 − 高夏普惩罚"（向后兼容）。
        if use_wf:
            wf = _walk_forward_score(cfg, shared_data, halflife_years, wf_lambda)
            if wf is None:
                return -999.0
            objective_signal = wf["score"]
            trial.set_user_attr("wf_sharpes", "|".join(f"{s:.2f}" for s in wf["wf_sharpes"]))
            trial.set_user_attr("wf_periods", "|".join(wf["wf_periods"]))
            trial.set_user_attr("wf_mean", float(wf["wf_mean"]))
            trial.set_user_attr("wf_std", float(wf["wf_std"]))
        else:
            objective_signal = sharpe

        penalty = 0.0
        if mdd > 0.25:
            penalty += (mdd - 0.25) * 5.0
        if not use_wf:
            # 仅旧模式需要人为压制高夏普；WF 模式由 std 项天然防过拟合
            if sharpe > 2.0:
                penalty += (sharpe - 2.0) * 3.0
            if sharpe > 3.0:
                penalty += (sharpe - 3.0) * 5.0
        # 低交易量惩罚：交易太少 ≈ buy-and-hold 运气的概率高
        if n_trading_days > 0:
            avg_trades_per_month = total_trades / (n_trading_days / 21)
            if avg_trades_per_month < 3.0:
                penalty += (3.0 - avg_trades_per_month) * 0.5
        # amihud_illiq 过重惩罚：非流动性因子是约束项，不是 Alpha 源
        amihud_w = weights_float[selected.index("amihud_illiq")] if "amihud_illiq" in selected else 0.0
        if amihud_w > 0.15:
            penalty += (amihud_w - 0.15) * 10.0
        return objective_signal - penalty

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
    baseline = run_single_backtest(base_config, label="基线", halflife_years=halflife_years,
                                   shared_data=shared_data)
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
    # 去重：TPE 常收敛到同一组合反复采样。按(因子集合, 权重四舍五入)去重，
    # 只保留每个独特组合得分最高的一个，避免结果表 30 行几乎相同。
    seen_combos: set[tuple] = set()
    rank = 0
    for t in trials:
        factors = t.user_attrs.get("factors", "").split("|") if t.user_attrs.get("factors") else []
        weights_str = t.user_attrs.get("weights", "")
        weights = {}
        if weights_str:
            for pair in weights_str.split("|"):
                if "=" in pair:
                    n, w = pair.split("=")
                    weights[n] = float(w)
        # 组合指纹：因子名 + 权重保留2位小数
        combo_key = tuple(sorted((n, round(w, 2)) for n, w in weights.items()))
        if combo_key in seen_combos:
            continue
        seen_combos.add(combo_key)
        rank += 1
        if rank > 30:
            break
        # 全区间真实夏普（与优化目标分区分开）。WF 模式下 t.value 是
        # mean(样本外夏普)−λ·std 的"目标分"，不是真夏普，单列展示。
        full_sharpe = float(t.user_attrs.get("full_sharpe", t.value))
        entry = {
            "label": f"Optuna #{rank} ({len(factors)}因子)",
            "factors": factors,
            "weights": weights,
            "objective_score": float(t.value),   # 优化目标分（WF: mean−λstd）
            "sharpe_ratio": full_sharpe,          # 全区间真实夏普
            "annual_return": float(t.user_attrs.get("annual_return", 0)),
            "max_drawdown": float(t.user_attrs.get("max_drawdown", 0)),
            "win_rate": t.user_attrs.get("win_rate", "N/A"),
            "total_trades": int(t.user_attrs.get("total_trades", 0)),
            "n_trading_days": int(t.user_attrs.get("n_trading_days", 0)),
        }
        # Walk-Forward 子窗口诊断明细（use_walk_forward 时可用）
        if t.user_attrs.get("wf_sharpes"):
            entry["wf_sharpes"] = t.user_attrs["wf_sharpes"]
            entry["wf_periods"] = t.user_attrs.get("wf_periods", "")
            entry["wf_mean"] = float(t.user_attrs.get("wf_mean", 0))
            entry["wf_std"] = float(t.user_attrs.get("wf_std", 0))
        results.append(entry)

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
    halflife_years: float | None = None,
    shared_data: dict | None = None,
) -> list[dict]:
    all_factors = base_config.get("factors", {})
    rng = np.random.default_rng(42)
    results: list[dict] = []

    enabled_baseline = [n for n, c in all_factors.items() if c.get("enabled")]
    baseline = run_single_backtest(base_config, label="基线", halflife_years=halflife_years,
                                    shared_data=shared_data)
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

        metrics = run_single_backtest(cfg, halflife_years=halflife_years, shared_data=shared_data)
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

    hl = full_config.get("optimizer", {}).get("time_decay_halflife", None)
    full_metrics = run_single_backtest(full_config, label="全区间验证", halflife_years=hl)
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

    # 从 YAML 读取时间衰减半衰期（可选，未配置则使用传统等权夏普）
    optimizer_cfg = base_config.get("optimizer", {}) if isinstance(base_config.get("optimizer"), dict) else {}
    halflife_years = optimizer_cfg.get("time_decay_halflife", None)

    # 从 YAML 读取单因子权重上限（可选，未配置则使用 optimizer 内置默认值）
    factor_max_weight = base_config.get("factor_max_weight", None)

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

    # 排除无数据源的因子。westock 模式下 roe_ttm/revenue_yoy 有真数据（PIT 对齐），
    # 从排除集移出；其它 provider 仍排除（akshare 无可靠基本面）。
    from research.run_backtest_demo import _provider
    no_data = set(ALL_NO_DATA)
    if _provider(base_config) == "westock":
        no_data -= {"roe_ttm", "revenue_yoy"}
    candidates = [c for c in raw_candidates if c not in no_data]
    skipped = [c for c in raw_candidates if c in no_data]

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

    hl_label = f" | 半衰期 {halflife_years}年" if halflife_years else ""
    print(f"\n{'='*60}")
    print(f"因子组合优化器 — {search_label}{hl_label}")
    print(f"  任务: {label}")
    print(f"  候选池: {len(candidates)} 个（已排除无数据源因子）")
    print(f"  搜索: {rounds} 轮, 因子 {min_factors}~{max_factors} 个")
    print(f"  区间: {base_config['backtest']['start_date']} ~ {base_config['backtest']['end_date']}")
    print(f"{'='*60}")

    # ★ 整轮优化只加载一次数据（价格/海外/宽基/宇宙与因子组合无关）
    #   注意：必须在上面调整 start_date（短窗口搜索）之后加载，确保区间一致
    shared_data = _load_shared_data(base_config)
    if shared_data is None:
        print("数据加载失败！请确认网络正常。")
        return []

    # 跑基线
    baseline = run_single_backtest(base_config, label="基线", halflife_years=halflife_years,
                                   shared_data=shared_data)
    if baseline is None:
        print("数据加载失败！请确认网络正常。")
        return []
    print(f"  基线夏普: {baseline.get('sharpe_ratio', 0):.3f}")

    # 搜索
    optuna_results = optimize_optuna(base_config, candidates, rounds, min_factors, max_factors,
                                     factor_max_weight, halflife_years, shared_data=shared_data)
    if optuna_results is not None:
        print("  ✓ 使用 Optuna TPE 引擎")
        results = optuna_results
    else:
        print("  ⚠ Optuna 未安装，回退到随机搜索")
        results = optimize_random(base_config, candidates, rounds, min_factors, max_factors,
                                  halflife_years, shared_data=shared_data)

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
    # 是否 WF 模式（结果带 wf_mean）→ 决定排序/展示口径
    is_wf = any("wf_mean" in r for r in results[1:])

    print(f"\n{'='*100}")
    print(f"优化结果")
    print(f"{'='*100}")
    if is_wf:
        print(f"  ⚠ Walk-Forward 模式：排序按'目标分(mean−λ·std)'，'夏普'列为全区间真实夏普。")
        print(f"  基线夏普: {baseline.get('sharpe_ratio',0):.3f}"
              f"  |  最佳全区间夏普: {best.get('sharpe_ratio',0):.3f}"
              f"  |  最佳目标分: {best.get('objective_score',0):.3f}"
              f"  |  全区间优于基线: {len(improved)}/{len(results)-1 if len(results)>1 else 0}")
    else:
        print(f"  基线夏普: {baseline.get('sharpe_ratio',0):.3f}"
              f"  |  最佳夏普: {best.get('sharpe_ratio',0):.3f}"
              f"  |  优于基线: {len(improved)}/{len(results)-1 if len(results)>1 else 0}")
    print(f"{'='*100}")
    if is_wf:
        print(f"  {'排':<3s} {'标签':<26s} {'目标分':>7s} {'WF均值':>7s} {'WFstd':>6s} {'真夏普':>7s} {'年化':>7s} {'回撤':>7s} {'组合':<s}")
        print(f"  {'─'*3} {'─'*26} {'─'*7} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*30}")
    else:
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

        if is_wf:
            print(
                f"  {rank+1:<3d} {r['label']:<26s} "
                f"{r.get('objective_score',0):7.3f} {r.get('wf_mean',0):7.2f} {r.get('wf_std',0):6.2f} "
                f"{sharpe:7.3f} {ann_ret*100:6.1f}% {mdd*100:6.1f}% "
                f"{factor_str}"
            )
        else:
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

    # ── Walk-Forward 子窗口夏普诊断（use_walk_forward 时可用）──
    wf_results = [r for r in results[1:6] if r.get("wf_sharpes")]
    if wf_results:
        print(f"\n{'─'*100}")
        print("Walk-Forward 子窗口夏普（各样本外段，std 高=不稳定 ⚑）:")
        for rank, r in enumerate(wf_results):
            sharpes = [float(x) for x in r["wf_sharpes"].split("|") if x]
            periods = r.get("wf_periods", "").split("|")
            mean_s = r.get("wf_mean", 0.0)
            std_s = r.get("wf_std", 0.0)
            flag = " ⚑红旗(波动大)" if std_s > 0.8 else ""
            seg_str = " | ".join(
                f"{p.split('~')[0]}={s:.2f}" for p, s in zip(periods, sharpes)
            )
            print(f"  #{rank+1}: mean={mean_s:.2f} std={std_s:.2f}{flag}")
            print(f"       {seg_str}")

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
