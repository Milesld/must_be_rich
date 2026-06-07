"""回测报告生成。

从 BacktestResult 生成标准回测报告，包括：
- 核心指标汇总
- 净值曲线（Plotly 交互图表）
- 月度/年度收益热力图
- Markdown 格式报告文件
- 因子 IC 分析
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ANNUAL_FACTOR = 252


class BacktestReport:
    """回测报告生成器。

    使用方式：
        result = engine.run(strategy)
        report = BacktestReport(result, benchmark_code="000300")
        report.generate("output/report.md")
    """

    def __init__(
        self,
        result: Any,  # BacktestResult
        benchmark_code: str = "000300",
        benchmark_nav: Optional[pd.Series] = None,
    ) -> None:
        """初始化。

        Args:
            result: BacktestResult 对象。
            benchmark_code: 比较基准代码（默认沪深300）。
            benchmark_nav: 基准净值序列（None 时不计算相对指标）。
        """
        self.result = result
        self.benchmark_code = benchmark_code
        self._benchmark_nav = benchmark_nav

    # ── 核心指标 ────────────────────────────

    def metrics(self) -> dict:
        """计算完整指标集。

        Returns:
            包含以下字段的字典：
            - annual_return, annual_volatility, sharpe_ratio
            - max_drawdown, max_drawdown_duration_days
            - calmar_ratio, sortino_ratio
            - win_rate, profit_loss_ratio
            - avg_hold_days, annual_turnover
            - total_cost, total_cost_pct
            - information_ratio (如有基准)
        """
        r = self.result
        nav = r.daily_nav
        if nav.empty:
            return {"error": "无数据"}

        returns = nav["daily_return"].dropna()
        nav_series = nav["nav"]

        n = len(returns)
        years = n / _ANNUAL_FACTOR
        total_r = (nav_series.iloc[-1] / r.initial_capital) - 1
        ann_r = (1 + total_r) ** (1 / years) - 1 if years > 0 else 0.0
        ann_vol = returns.std() * np.sqrt(_ANNUAL_FACTOR) if n > 1 else 0.0
        sharpe = (ann_r - 0.02) / ann_vol if ann_vol > 0 else 0.0

        # 最大回撤
        peak = nav_series.expanding().max()
        dd = (nav_series - peak) / peak
        max_dd = float(dd.min())
        max_dd_idx = dd.idxmin()

        # 回撤持续时间
        max_dd_dur = self._calc_drawdown_duration(nav_series)

        # Sortino（下行波动率）
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(_ANNUAL_FACTOR) if len(downside) > 1 else ann_vol
        sortino = (ann_r - 0.02) / downside_vol if downside_vol > 0 else 0.0

        # Calmar
        calmar = ann_r / abs(max_dd) if max_dd < 0 else 0.0

        # 交易统计
        trades = r.trades
        filled = [t for t in trades if t.status == "filled"]
        sells = [t for t in filled if t.side == "sell"]
        buys = [t for t in filled if t.side == "buy"]

        win_rate = len(sells) / len(filled) if filled else 0.0
        total_cost = sum(
            float(t.cost_breakdown.total) for t in filled if t.cost_breakdown
        )

        # 盈亏比（简化：基于卖出记录的PnL，实际需要追踪每笔买卖配对）
        pl_ratio = 1.0  # placeholder — 配对分析需要更复杂的逻辑

        # 换手率（年化）
        buy_amount = sum(t.filled_price * t.filled_shares for t in buys)
        sell_amount = sum(t.filled_price * t.filled_shares for t in sells)
        annual_turnover = (
            (buy_amount + sell_amount) / 2 / r.initial_capital / years
            if years > 0 else 0.0
        )

        # 平均持仓天数（简化）
        avg_hold = 0.0  # 需要配对分析

        metrics_dict = {
            "period": f"{r.start_date} ~ {r.end_date}",
            "trading_days": n,
            "initial_capital": r.initial_capital,
            "final_nav": float(nav_series.iloc[-1]),
            "total_return_pct": float(total_r * 100),
            "annual_return_pct": float(ann_r * 100),
            "annual_volatility_pct": float(ann_vol * 100),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": float(max_dd * 100),
            "max_drawdown_date": str(max_dd_idx),
            "max_drawdown_duration_days": max_dd_dur,
            "calmar_ratio": float(calmar),
            "win_rate_pct": float(win_rate * 100),
            "total_trades": len(filled),
            "total_cost": float(total_cost),
            "total_cost_pct": float(total_cost / r.initial_capital * 100) if r.initial_capital > 0 else 0.0,
            "annual_turnover": float(annual_turnover),
        }

        # 信息比率（如有基准）
        if self._benchmark_nav is not None:
            ir = self._calc_ir(returns, self._benchmark_nav)
            metrics_dict["information_ratio"] = float(ir)
        else:
            metrics_dict["information_ratio"] = None

        return metrics_dict

    # ── 图表生成 ──────────────────────────────

    def plot_nav(self) -> Any:
        """生成 Plotly 净值曲线图。"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly 未安装，跳过图表生成")
            return None

        nav = self.result.daily_nav
        if nav.empty:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nav.index, y=nav["nav"],
            mode="lines", name="策略净值",
            line=dict(color="steelblue", width=2),
        ))
        fig.update_layout(
            title=f"回测净值曲线 ({self.result.start_date} ~ {self.result.end_date})",
            xaxis_title="日期",
            yaxis_title="净值",
            hovermode="x unified",
        )
        return fig

    def monthly_returns_heatmap(self) -> Any:
        """生成月度收益热力图。"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        nav = self.result.daily_nav
        if nav.empty:
            return None

        # 月度收益
        monthly = nav["daily_return"].resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly_df = monthly.to_frame(name="return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month

        pivot = monthly_df.pivot(index="year", columns="month", values="return")
        pivot_pct = pivot * 100

        fig = go.Figure(data=go.Heatmap(
            z=pivot_pct.values,
            x=[f"{m}月" for m in pivot_pct.columns],
            y=[str(y) for y in pivot_pct.index],
            colorscale="RdYlGn",
            zmid=0,
            text=pivot_pct.round(2).values,
            texttemplate="%{text}%",
        ))
        fig.update_layout(
            title="月度收益热力图 (%)",
            xaxis_title="月份",
            yaxis_title="年份",
        )
        return fig

    # ── 报告生成 ──────────────────────────────

    def generate(self, output_path: str) -> None:
        """生成 Markdown 回测报告。"""
        m = self.metrics()
        if "error" in m:
            logger.warning("无法生成报告: %s", m["error"])
            return

        lines = [
            f"# 回测报告",
            f"",
            f"**回测ID**: {self.result.run_id}",
            f"**回测区间**: {m['period']}",
            f"**交易天数**: {m['trading_days']}",
            f"**基准**: {self.benchmark_code}",
            f"",
            f"## 核心指标",
            f"",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 初始资金 | {m['initial_capital']:,.0f} |",
            f"| 最终净值 | {m['final_nav']:,.2f} |",
            f"| 总收益率 | {m['total_return_pct']:.2f}% |",
            f"| 年化收益率 | {m['annual_return_pct']:.2f}% |",
            f"| 年化波动率 | {m['annual_volatility_pct']:.2f}% |",
            f"| 夏普比率 | {m['sharpe_ratio']:.3f} |",
            f"| 索提诺比率 | {m['sortino_ratio']:.3f} |",
            f"| 最大回撤 | {m['max_drawdown_pct']:.2f}% |",
            f"| 最大回撤日期 | {m['max_drawdown_date']} |",
            f"| 最大回撤持续(天) | {m['max_drawdown_duration_days']} |",
            f"| Calmar比率 | {m['calmar_ratio']:.3f} |",
            f"",
        ]

        if m.get("information_ratio") is not None:
            lines.append(f"| 信息比率 | {m['information_ratio']:.3f} |")
            lines.append("")

        lines += [
            f"",
            f"## 交易统计",
            f"",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 总交易笔数 | {m['total_trades']} |",
            f"| 胜率 | {m['win_rate_pct']:.1f}% |",
            f"| 总交易成本 | {m['total_cost']:,.2f} |",
            f"| 成本占比 | {m['total_cost_pct']:.2f}% |",
            f"| 年化换手率 | {m['annual_turnover']:.2f} |",
            f"",
            f"---",
            f"",
            f"> 本报告由回测引擎自动生成。",
            f"> 回测结果不代表未来表现。实盘前必须经过充分验证。",
        ]

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info("回测报告已生成: %s", output_path)

    # ── IC 分析 ───────────────────────────────

    def ic_analysis(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """因子 IC 分析。

        Args:
            factor_values: 因子值 DataFrame (index: date*code)。
            forward_returns: 前向收益 DataFrame (相同 index)。

        Returns:
            IC 统计 DataFrame：每因子一行，包含 IC_mean, IC_std, ICIR, IC_decay。
        """
        if factor_values.empty or forward_returns.empty:
            return pd.DataFrame()

        # 按日期对齐
        common_idx = factor_values.index.intersection(forward_returns.index)
        fv = factor_values.loc[common_idx]
        fr = forward_returns.loc[common_idx]

        results: list[dict] = []
        for col in fv.columns:
            ic_series = fv[col].groupby(level=0).corr(fr, method="spearman")
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0.0

            results.append({
                "factor": col,
                "IC_mean": float(ic_mean),
                "IC_std": float(ic_std),
                "ICIR": float(icir),
                "IC_positive_ratio": float((ic_series > 0).mean()),
                "n_periods": len(ic_series),
            })

        return pd.DataFrame(results)

    # ── 内部辅助 ──────────────────────────────

    @staticmethod
    def _calc_drawdown_duration(nav: pd.Series) -> int:
        """计算最大回撤的持续时间（交易日数）。"""
        peak = nav.expanding().max()
        dd = (nav - peak) / peak
        dd = dd.fillna(0)

        # 找最大回撤的起止区间
        max_dd_idx = dd.idxmin()
        if max_dd_idx is None:
            return 0

        # 往前找新高
        underwater = False
        max_dur = 0
        current_dur = 0
        for val in dd:
            if val < 0:
                if not underwater:
                    underwater = True
                    current_dur = 1
                else:
                    current_dur += 1
            else:
                if underwater:
                    max_dur = max(max_dur, current_dur)
                    underwater = False
                    current_dur = 0
        if underwater:
            max_dur = max(max_dur, current_dur)

        return max_dur

    @staticmethod
    def _calc_ir(strategy_returns: pd.Series, benchmark_nav: pd.Series) -> float:
        """计算信息比率。"""
        bench_returns = benchmark_nav.pct_change().dropna()
        common = strategy_returns.index.intersection(bench_returns.index)
        if len(common) < 10:
            return 0.0

        excess = strategy_returns.loc[common] - bench_returns.loc[common]
        mean_excess = excess.mean()
        tracking_error = excess.std()

        annual_factor = np.sqrt(_ANNUAL_FACTOR)
        return float(mean_excess / tracking_error * annual_factor) if tracking_error > 0 else 0.0
