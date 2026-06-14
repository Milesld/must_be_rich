"""市场状态识别。

综合趋势、波动率、估值、市场宽度四个维度，
输出客观的市场状态标签和对应的建议总仓位比例。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    """市场状态判断结果。"""
    regime_label: str = "震荡市"
    composite_score: float = 0.5
    dimension_scores: dict[str, float] = field(default_factory=lambda: {
        "trend": 0.5, "volatility": 0.5, "valuation": 0.5, "breadth": 0.5,
    })
    suggested_position_ratio: float = 0.5

    def __repr__(self) -> str:
        dims = ", ".join(f"{k}={v:.2f}" for k, v in self.dimension_scores.items())
        return (
            f"RegimeResult(label={self.regime_label}, score={self.composite_score:.2f}, "
            f"position={self.suggested_position_ratio:.0%}, dims=({dims}))"
        )


class MarketRegimeDetector:
    """市场状态识别器。

    综合四个客观指标给出 0~1 的市场评分：
    1. 趋势信号：指数与均线关系、均线多空排列
    2. 波动率信号：已实现波动率在历史中的分位数
    3. 估值信号：全市场PE/PB历史分位
    4. 市场宽度信号：上涨家数、新高家数、涨停温度

    使用方式：
        detector = MarketRegimeDetector()
        result = detector.detect(today, market_data_df)
        # result.suggested_position_ratio → 建议总仓位
    """

    # 标签映射
    _LABEL_THRESHOLDS = [
        (0.80, "牛市低波动"),
        (0.65, "牛市高波动"),
        (0.45, "震荡偏多"),
        (0.35, "震荡市"),
        (0.20, "熊市高波动"),
        (0.00, "极端恐慌"),
    ]

    def __init__(self, config: dict | None = None) -> None:
        """初始化。

        Args:
            config: 配置字典或 ConfigLoader 节点。
                    包含 trend/volatility/valuation/breadth 各维度参数。
        """
        cfg = config or {}
        self._trend_ma_windows = cfg.get("trend_ma_windows", [5, 20, 60, 200])
        self._vol_lookback_years = cfg.get("vol_lookback_years", 3)
        self._val_lookback_years = cfg.get("val_lookback_years", 10)
        self._weights = cfg.get("dimension_weights", {
            "trend": 0.25, "volatility": 0.25, "valuation": 0.25, "breadth": 0.25,
        })
        # 评分→仓位映射
        self._position_map = cfg.get("position_ratio_map", {
            (0.70, 1.00): 0.85,
            (0.50, 0.70): 0.65,
            (0.30, 0.50): 0.50,
            (0.15, 0.30): 0.35,
            (0.00, 0.15): 0.20,
        })

    # ── 主入口 ──────────────────────────────

    def detect(self, as_of_date: date, market_data: pd.DataFrame | None = None) -> RegimeResult:
        """综合四个维度判断市场状态。

        Args:
            as_of_date: 判断日期。
            market_data: 市场行情 DataFrame，至少包含：
                close, volume（指数级数据）。
                可选列：pe_ttm, pb, advance_count, decline_count,
                new_high_20d, limit_up_count。
                None 时返回中性结果（0.5/震荡市）。

        Returns:
            RegimeResult。
        """
        if market_data is None or market_data.empty:
            return RegimeResult()

        # 确保数据截止于 as_of_date（统一转为 Timestamp 比较）
        ts_index = pd.to_datetime(market_data.index)
        cutoff = pd.Timestamp(as_of_date)
        df = market_data.loc[ts_index <= cutoff].copy()
        if df.empty:
            return RegimeResult()

        trend_score = self._calc_trend_signal(df)
        vol_score = self._calc_volatility_signal(df)
        val_score = self._calc_valuation_signal(df)
        breadth_score = self._calc_breadth_signal(df)

        scores = {
            "trend": trend_score,
            "volatility": vol_score,
            "valuation": val_score,
            "breadth": breadth_score,
        }

        composite = sum(
            scores[k] * self._weights.get(k, 0.25) for k in scores
        )
        composite = max(0.0, min(1.0, composite))

        regime = self._score_to_label(composite)
        position_ratio = self._score_to_position(composite)

        return RegimeResult(
            regime_label=regime,
            composite_score=composite,
            dimension_scores=scores,
            suggested_position_ratio=position_ratio,
        )

    # ── 维度1: 趋势信号 ───────────────────────

    def _calc_trend_signal(self, df: pd.DataFrame) -> float:
        """趋势信号：0=空头排列, 1=多头排列。"""
        close = df["close"]
        if len(close) < max(self._trend_ma_windows):
            return 0.5

        # 计算各周期均线
        mas = {w: close.rolling(w).mean().iloc[-1] for w in self._trend_ma_windows}

        latest = close.iloc[-1]
        ma_long = mas.get(200, latest)

        # 价格 vs 200日线
        price_vs_ma200 = 1.0 if latest > ma_long else 0.0

        # 均线排列：短>中>长 → 多头
        alignment_score = 0.0
        ordered = sorted(mas.items(), key=lambda x: x[0])  # 按窗口升序
        pairs = 0
        aligned = 0
        for i in range(len(ordered) - 1):
            pairs += 1
            if ordered[i][1] > ordered[i + 1][1]:
                aligned += 1
        if pairs > 0:
            alignment_score = aligned / pairs

        # 综合 = 价格vs均线 × 0.5 + 排列评分 × 0.5
        return price_vs_ma200 * 0.5 + alignment_score * 0.5

    # ── 维度2: 波动率信号 ───────────────────────

    def _calc_volatility_signal(self, df: pd.DataFrame) -> float:
        """波动率信号：低波动=高分, 高波动=低分。

        使用滚动年化波动率在历史窗口中的分位数。
        """
        close = df["close"]
        if len(close) < 252:
            return 0.5

        returns = close.pct_change().dropna()
        rolling_vol = returns.rolling(252).std().dropna() * np.sqrt(252)

        if len(rolling_vol) < 60:
            return 0.5

        current_vol = rolling_vol.iloc[-1]
        # 使用整个滚动波动率序列作为历史分布
        rank = (rolling_vol < current_vol).mean()  # 当前波动率的分位数

        # 低波动(<20分位) → 高分; 高波动(>80分位) → 低分
        if rank <= 0.20:
            return 0.8
        elif rank <= 0.40:
            return 0.6
        elif rank <= 0.60:
            return 0.5
        elif rank <= 0.80:
            return 0.3
        else:
            return 0.1

    # ── 维度3: 估值信号 ───────────────────────

    def _calc_valuation_signal(self, df: pd.DataFrame) -> float:
        """估值信号：低估值=高分, 高估值=低分。

        使用 PE/PB 在历史中的分位数。
        """
        if "pe_ttm" not in df.columns and "pb" not in df.columns:
            return 0.5

        # 优先PE，其次PB
        val_col = "pe_ttm" if "pe_ttm" in df.columns else "pb"
        vals = df[val_col].dropna()
        if len(vals) < 100:
            return 0.5

        current = vals.iloc[-1]
        rank = (vals < current).mean()  # 分位数

        # 低估值(<30分位) → 高分; 高估值(>70分位) → 低分
        # 反转：rank=0.1 → score=0.9（估值很低→高分）
        score = 1.0 - rank
        return max(0.0, min(1.0, score))

    # ── 维度4: 市场宽度信号 ───────────────────────

    def _calc_breadth_signal(self, df: pd.DataFrame) -> float:
        """市场宽度信号。

        综合：上涨家数占比 + 创20日新高家数占比 + 涨停温度
        """
        scores: list[float] = []

        # 上涨家数占比
        if "advance_count" in df.columns and "decline_count" in df.columns:
            total = df["advance_count"] + df["decline_count"]
            adv_ratio = (df["advance_count"] / total.replace(0, float("nan"))).iloc[-1]
            if not np.isnan(adv_ratio):
                # >60%上涨 → 高分, <40% → 低分
                scores.append(min(1.0, max(0.0, (adv_ratio - 0.3) / 0.4)))

        # 创新高家数占比（需有成分股总数）
        if "new_high_20d" in df.columns and len(df) > 0:
            avg_total = 5000  # 全市场约5000只
            nh_ratio = df["new_high_20d"].iloc[-1] / avg_total if len(df["new_high_20d"]) > 0 else 0.0
            scores.append(min(1.0, nh_ratio * 10))

        # 涨停温度
        if "limit_up_count" in df.columns and len(df["limit_up_count"]) > 0:
            lu = df["limit_up_count"].iloc[-1]
            # 50只以下→低, 50-100→中, 100+→高
            if lu >= 100:
                scores.append(0.8)
            elif lu >= 50:
                scores.append(0.5)
            else:
                scores.append(0.2)

        if not scores:
            return 0.5
        return sum(scores) / len(scores)

    # ── 评分→标签/仓位 ───────────────────────

    def _score_to_label(self, score: float) -> str:
        for threshold, label in self._LABEL_THRESHOLDS:
            if score >= threshold:
                return label
        return "极端恐慌"

    def _score_to_position(self, score: float) -> float:
        for (lo, hi), ratio in sorted(self._position_map.items(), key=lambda x: -x[0][0]):
            if lo <= score <= hi:
                return ratio
        return 0.30  # 默认低仓位
