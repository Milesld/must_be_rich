"""组合优化器。

将排序分数或预测信号转化为实际可执行的仓位权重方案。
提供五个优化策略（由简到繁），和行业集中度约束后处理。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 默认参数
_DEFAULT_MAX_SINGLE = 0.20
_DEFAULT_MAX_INDUSTRY = 0.40
_DEFAULT_TOP_N = 15


class PortfolioOptimizer:
    """组合权重优化器。

    使用方式：
        opt = PortfolioOptimizer(config)
        weights = opt.inverse_vol_weight(scores, volatilities)
        # 施加行业约束
        weights = opt.apply_constraints(weights, industry_map)
    """

    def __init__(self, config: dict | None = None) -> None:
        """初始化。

        Args:
            config: 配置字典（通常来自 risk/thresholds.yaml 的 position_sizing 节）。
        """
        cfg = config or {}
        self._max_single = cfg.get("single_stock_max_ratio", _DEFAULT_MAX_SINGLE)
        self._max_single_hard = cfg.get("single_stock_hard_max", self._max_single * 1.25)
        self._max_industry = cfg.get("industry_max_ratio", _DEFAULT_MAX_INDUSTRY)

    # ── 策略1: 等权 ──────────────────────────────

    def equal_weight(
        self, scores: dict[str, float], top_n: int = _DEFAULT_TOP_N,
    ) -> dict[str, float]:
        """等权：取 score 最高的 top_n 只股票，每只等权。"""
        if not scores:
            return {}

        sorted_codes = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type, return-value]
        selected = sorted_codes[:top_n]
        if not selected:
            return {}

        w = 1.0 / len(selected)
        return {c: w for c in selected}

    # ── 策略2: 信号强度加权 ─────────────────────────

    def signal_weighted(
        self,
        scores: dict[str, float],
        max_single: float | None = None,
    ) -> dict[str, float]:
        """按信号强度/置信度加权，超过上限的截断并重分配。"""
        if not scores:
            return {}

        cap = max_single or self._max_single
        total_raw = sum(max(0.0, v) for v in scores.values())
        if total_raw <= 0:
            return {}

        # 第一轮：按比例加权
        weights = {c: max(0.0, v) / total_raw for c, v in scores.items()}

        # 截断重分配
        return self._cap_and_redistribute(weights, cap)

    # ── 策略3: 波动率倒数加权 ────────────────────────

    def inverse_vol_weight(
        self,
        scores: dict[str, float],
        volatilities: dict[str, float],
        top_n: int = _DEFAULT_TOP_N,
    ) -> dict[str, float]:
        """波动率倒数加权：权重 ∝ 1/σ。

        波动越低的股票权重越高——这是降低组合波动最简单有效的方法。
        先按 score 选 top_n，再按 1/vol 分配权重。
        """
        if not scores or not volatilities:
            return self.equal_weight(scores, top_n)

        sorted_codes = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type, return-value]
        selected = sorted_codes[:top_n]

        inv_vols: dict[str, float] = {}
        for c in selected:
            vol = volatilities.get(c, 0.0)
            if vol > 0:
                inv_vols[c] = 1.0 / vol
            else:
                inv_vols[c] = 0.0

        total_inv = sum(inv_vols.values())
        if total_inv <= 0:
            return self.equal_weight(scores, top_n)

        weights = {c: iv / total_inv for c, iv in inv_vols.items()}
        return self._cap_and_redistribute(weights, self._max_single)

    # ── 策略4: 风险平价 ─────────────────────────────

    def risk_parity(
        self,
        scores: dict[str, float],
        returns: pd.DataFrame,  # columns=codes, index=dates
        top_n: int = _DEFAULT_TOP_N,
        max_iter: int = 50,
    ) -> dict[str, float]:
        """风险平价：使每只股票对组合的风险贡献相等。

        使用迭代算法（Naive Risk Parity）求解。
        输入 returns 为历史收益率矩阵（columns=股票代码, index=日期）。
        """
        sorted_codes = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type, return-value]
        selected = [c for c in sorted_codes[:top_n] if c in returns.columns]

        if len(selected) < 2:
            return self.equal_weight(scores, top_n)

        ret_mat = returns[selected].dropna()
        if ret_mat.empty or len(ret_mat) < len(selected):
            return self.inverse_vol_weight(
                scores,
                {c: ret_mat[c].std() for c in selected},
                top_n,
            )

        cov = ret_mat.cov().values
        n = len(selected)

        # Naive Risk Parity: w ∝ 1 / (Σ ρ_ij * σ_i * σ_j) 的简化版本
        # 实际使用: w_i = 1/σ_i / Σ(1/σ_j)
        vols = np.sqrt(np.diag(cov))
        inv_vols = 1.0 / np.maximum(vols, 1e-8)
        raw_w = inv_vols / inv_vols.sum()

        # 迭代修正
        for _ in range(max_iter):
            # 当前风险贡献
            port_vol = np.sqrt(raw_w @ cov @ raw_w)
            if port_vol < 1e-8:
                break
            mrc = cov @ raw_w / port_vol  # 边际风险贡献
            rc = raw_w * mrc                # 风险贡献
            target_rc = port_vol / n        # 目标：平均分配
            # 调整权重
            error = rc / target_rc
            raw_w = raw_w / np.maximum(error, 0.5)
            raw_w = raw_w / raw_w.sum()

        weights = {selected[i]: float(raw_w[i]) for i in range(n)}
        return self._cap_and_redistribute(weights, self._max_single)

    # ── 策略5: 最小CVaR ────────────────────────────

    def min_cvar(
        self,
        scores: dict[str, float],
        returns: pd.DataFrame,
        top_n: int = _DEFAULT_TOP_N,
        alpha: float = 0.05,
    ) -> dict[str, float]:
        """最小CVaR优化：最小化条件风险价值。

        使用 Ledoit-Wolf 收缩协方差估计。
        简化解法：对协方差矩阵做特征分解后取最小方差组合。
        """
        sorted_codes = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type, return-value]
        selected = [c for c in sorted_codes[:top_n] if c in returns.columns]

        if len(selected) < 2:
            return self.equal_weight(scores, top_n)

        ret_mat = returns[selected].dropna()
        if ret_mat.empty:
            return self.equal_weight(scores, top_n)

        # Ledoit-Wolf 收缩估计
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(ret_mat.values)
            cov = lw.covariance_
        except Exception:
            cov = ret_mat.cov().values

        n = len(selected)
        # 解析解：最小方差组合 w ∝ Σ⁻¹ 1
        try:
            inv_cov = np.linalg.inv(cov)
            ones = np.ones(n)
            raw_w = inv_cov @ ones
            raw_w = raw_w / raw_w.sum()
        except np.linalg.LinAlgError:
            vols = np.sqrt(np.maximum(np.diag(cov), 1e-10))
            raw_w = (1.0 / vols)
            raw_w = raw_w / raw_w.sum()

        weights = {selected[i]: float(max(raw_w[i], 0)) for i in range(n)}
        # 去负权重后重新归一化
        return self._cap_and_redistribute(weights, self._max_single)

    # ── 行业集中度约束 ────────────────────────────

    def apply_constraints(
        self,
        weights: dict[str, float],
        industry_map: dict[str, str],
        max_industry: float | None = None,
    ) -> dict[str, float]:
        """后处理：施加行业集中度约束。

        单行业权重 ≤ max_industry，超过的截断并重分配到其他行业。

        Args:
            weights: {code: weight}。
            industry_map: {code: industry}。
            max_industry: 行业最大权重。None 则使用配置值。

        Returns:
            调整后的权重。
        """
        if not weights:
            return {}

        cap = max_industry or self._max_industry
        result = dict(weights)

        for _ in range(10):  # 最多10轮重分配
            # 按行业聚合
            industry_weight: dict[str, float] = {}
            industry_codes: dict[str, list[str]] = {}
            for code, w in result.items():
                ind = industry_map.get(code, "other")
                industry_weight[ind] = industry_weight.get(ind, 0.0) + w
                industry_codes.setdefault(ind, []).append(code)

            # 找出超标行业
            over: list[str] = []
            for ind, w in industry_weight.items():
                if w > cap:
                    over.append(ind)

            if not over:
                break

            # 截断超标行业
            for ind in over:
                codes_in_ind = industry_codes[ind]
                if not codes_in_ind:
                    continue
                excess = industry_weight[ind] - cap
                # 按权重比例缩减该行业内各股票
                scale = cap / industry_weight[ind]
                for c in codes_in_ind:
                    result[c] *= scale

                # 将 excess 重分配到其他行业（按当前权重比例）
                other_codes = [c for c in result if c not in codes_in_ind]
                other_total = sum(result[c] for c in other_codes)
                if other_total > 0:
                    for c in other_codes:
                        result[c] += excess * (result[c] / other_total)

            # 重归一化
            total = sum(result.values())
            if total > 0:
                result = {c: w / total for c, w in result.items()}

        return result

    # ── 内部工具 ─────────────────────────────

    def _cap_and_redistribute(
        self, weights: dict[str, float], cap: float,
    ) -> dict[str, float]:
        """截断超标权重并重分配到未超标的。"""
        result = dict(weights)
        for _ in range(10):
            over_cap = {c: w for c, w in result.items() if w > cap}
            if not over_cap:
                break

            under_cap = {c: w for c, w in result.items() if w <= cap}
            excess = sum(w - cap for w in over_cap.values())

            for c in over_cap:
                result[c] = cap

            if not under_cap or excess <= 0:
                break

            # 按原权重比例分配到未超标的
            under_total = sum(under_cap.values())
            for c in under_cap:
                result[c] += excess * (under_cap[c] / under_total)

        # 归一化
        total = sum(result.values())
        if total > 0:
            result = {c: w / total for c, w in result.items()}
        return result
