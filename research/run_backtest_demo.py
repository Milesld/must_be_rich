#!/usr/bin/env python3
"""研究模式完整示例：动量+低波动选股月度调仓策略的回测闭环。

运行方式：
    python research/run_backtest_demo.py

前置条件：
    1. lightgbm 已安装（如未安装，请先在项目目录执行:
       cd $TMPDIR && unzip -o ~/Downloads/lightgbm-4.6.0-*.whl
       cp -r $TMPDIR/lightgbm $TMPDIR/lightgbm-* <项目目录>/)
    2. 首次运行会从 akshare 拉取交易日历，以后使用缓存
    3. 数据从 akshare 在线拉取（需要网络）

数据存储位置（见脚本末尾注释或运行 scripts/clean_data.sh）：
    ~/.quant_system/calendar/trading_calendar.parquet  — 交易日历缓存
    data/checkpoints/  — 数据管线断点续传记录
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. 配置日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo")


# ---------------------------------------------------------------------------
# 1. 策略定义
# ---------------------------------------------------------------------------
class MomentumLowVolStrategy:
    """动量+低波动策略：每月初等权持有综合评分最高的5只股票。

    评分 = 动量排名分位 × 0.6 + (1 - 波动率排名分位) × 0.4

    逻辑：
    - 每月第一个交易日调仓一次
    - 卖出不在新目标池的股票
    - 等权买入新目标池中未持有的股票
    """

    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n
        self._last_rebalance_month: Optional[int] = None

    def on_bar(
        self,
        trade_date: date,
        features: pd.DataFrame,
        positions: dict[str, int],
        cash: float,
        daily_data: dict[str, dict],
    ) -> list:
        """每个交易日的回调。返回 TradeIntent 列表。"""
        intents: list = []
        from core.backtest.engine import TradeIntent

        # 仅在月初第一个交易日调仓
        if not self._is_first_trading_day_of_month(trade_date):
            return intents

        # 用因子数据计算综合评分
        if features.empty:
            return intents

        scores = self._score_stocks(features)
        if scores.empty:
            return intents

        # 选 top N
        target_codes = set(scores.nlargest(self.top_n).index)

        # 卖出不在目标池的持仓
        for code, shares in list(positions.items()):
            if shares <= 0:
                continue
            if code not in target_codes:
                row = daily_data.get(code, {})
                price = float(row.get("close", 0))
                if price > 0:
                    intents.append(TradeIntent(
                        signal_id=f"rebal_sell_{code}_{trade_date}",
                        code=code, side="sell", price=price, shares=shares,
                    ))

        # 等权买入目标池（等值分配）
        if len(target_codes) > 0:
            per_stock_cash = cash / len(target_codes)
            for code in target_codes:
                if positions.get(code, 0) > 0:
                    continue  # 已持有，不重复买入（等权不做再平衡微调）
                row = daily_data.get(code, {})
                price = float(row.get("close", 0))
                if price <= 0:
                    continue
                ideal_s = int(per_stock_cash / price)
                shares = (ideal_s // 100) * 100  # 向下取整到手的倍数
                if shares >= 100:
                    intents.append(TradeIntent(
                        signal_id=f"rebal_buy_{code}_{trade_date}",
                        code=code, side="buy", price=price, shares=shares,
                    ))

        return intents

    # -- 内部 --
    @staticmethod
    def _is_first_trading_day_of_month(dt: date) -> bool:
        from core.common.calendar import get_calendar
        cal = get_calendar()
        prev = cal.prev_trading_day(dt)
        return prev.month != dt.month

    @staticmethod
    def _score_stocks(features: pd.DataFrame) -> pd.Series:
        """综合动量排名 + 低波动排名的评分。"""
        scores = pd.Series(1.0, index=features.index)

        # 动量排名分位（越高越好）
        if "momentum_20d" in features.columns:
            mom = features["momentum_20d"].dropna()
            if len(mom) > 0:
                mom_rank = mom.rank(pct=True)
                scores.loc[mom_rank.index] = mom_rank * 0.6

        # 波动率倒数排名分位（越低波动越高分）
        if "volatility_20d" in features.columns:
            vol = features["volatility_20d"].dropna()
            if len(vol) > 0:
                # 反转：低波动→高排名
                inv_vol_rank = (1.0 - vol.rank(pct=True))
                common = scores.index.intersection(inv_vol_rank.index)
                scores.loc[common] = scores.loc[common] + inv_vol_rank.loc[common] * 0.4

        return scores.clip(lower=0.0)


# ---------------------------------------------------------------------------
# 2. 数据加载
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_raw_data(start: date, end: date) -> dict[date, dict[str, dict]]:
    """从 akshare 拉取日K线，缓存在内存中（大时间段会消耗较多内存）。"""
    from core.data.sources.akshare import AkShareSource

    logger.info("正在从 akshare 拉取 %s ~ %s 日K线数据...", start, end)
    ds = AkShareSource()
    raw_df = ds.get_daily_kline(start, end)

    if raw_df.empty:
        logger.warning("akshare 返回空数据！")
        return {}

    # 按日期组织：{date: {code: {open, high, low, close, ...}}}
    result: dict[date, dict] = {}
    for _, row in raw_df.iterrows():
        td = row.get("trade_date")
        if isinstance(td, pd.Timestamp):
            td = td.date()
        if td is None:
            continue
        if td not in result:
            result[td] = {}
        result[td][str(row["code"])] = {
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("close", 0)),
            "pre_close": float(row.get("pre_close", 0)),
            "volume": float(row.get("volume", 0)),
            "amount": float(row.get("amount", 0)),
            "is_st": bool(row.get("is_st", False)),
            "is_suspended": bool(row.get("is_suspended", False)),
            "turnover": float(row.get("turnover", 0)),
        }
    logger.info("数据加载完成: %d 个交易日", len(result))
    return result


def make_data_loader():
    """返回 data_loader 回调函数。首次调用拉取全量数据，后续从缓存读取。"""
    # 闭包变量
    cache: Optional[dict] = None

    def loader(trade_date: date) -> pd.DataFrame:
        nonlocal cache
        if cache is None:
            # 拉取数据范围：回测前后各多取一些（用于因子计算的滚动窗口）
            data_start = trade_date - timedelta(days=365)  # 多1年
            data_end = trade_date + timedelta(days=30)
            cache = _load_raw_data(data_start, data_end)

        rows = []
        for code, fields in (cache or {}).get(trade_date, {}).items():
            rows.append({"code": code, **fields})
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    return loader


# ---------------------------------------------------------------------------
# 3. 因子计算（简化版：不依赖 FeatureStore，直接基于行情数据计算）
# ---------------------------------------------------------------------------
def make_feature_loader(raw_data: dict):
    """从已加载的行情数据计算技术因子。"""

    def loader(trade_date: date, codes: list[str]) -> pd.DataFrame:
        from core.common.calendar import get_calendar
        cal = get_calendar()

        # 取最近 N 个交易日的数据计算因子
        lookback_dates = cal.get_prev_n_trading_days(trade_date, 60)
        lookback_dates = [d for d in reversed(lookback_dates) if d <= trade_date]

        # 构建 60 天的价格序列
        price_series: dict[str, list[float]] = {}
        volume_series: dict[str, list[float]] = {}
        for dt in lookback_dates:
            if dt not in raw_data:
                continue
            for code in raw_data[dt]:
                close = raw_data[dt][code].get("close", 0)
                vol = raw_data[dt][code].get("volume", 0)
                if close > 0:
                    price_series.setdefault(code, []).append(close)
                    volume_series.setdefault(code, []).append(vol)

        # 计算因子
        results: list[dict] = []
        for code in (codes or []):
            prices = price_series.get(code, [])
            if len(prices) < 22:
                continue

            close_arr = np.array(prices, dtype=float)
            rets = np.diff(close_arr) / close_arr[:-1]

            momentum_20d = (close_arr[-1] / close_arr[-min(21, len(close_arr))] - 1.0) if len(close_arr) >= 21 else np.nan
            volatility_20d = np.nanstd(rets[-min(20, len(rets)):]) * np.sqrt(252) if len(rets) >= 5 else np.nan

            vols = volume_series.get(code, [])
            volume_ratio = vols[-1] / np.mean(vols[-min(21, len(vols)):]) if len(vols) >= 5 else 1.0

            results.append({
                "code": code,
                "momentum_20d": float(momentum_20d) if not np.isnan(momentum_20d) else 0.0,
                "volatility_20d": float(volatility_20d) if not np.isnan(volatility_20d) else 0.0,
                "volume_ratio": float(volume_ratio),
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index("code")
        return df

    return loader


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    from core.backtest.engine import BacktestEngine

    # 回测区间
    start_date = date(2024, 1, 2)
    end_date = date(2025, 12, 31)

    # 创建引擎
    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=1_000_000,
    )

    # 数据加载器
    data_loader = make_data_loader()

    # 因子加载器（闭包，共享 raw data）
    raw_data = None

    def feature_loader(trade_date: date, codes: list[str]) -> pd.DataFrame:
        nonlocal raw_data
        if raw_data is None:
            # 获取已缓存的数据
            raw_data = _load_raw_data(
                start_date - timedelta(days=365),
                end_date + timedelta(days=30),
            )
        fl = make_feature_loader(raw_data)
        return fl(trade_date, codes)

    # 策略
    strategy = MomentumLowVolStrategy(top_n=5)

    # 执行回测
    logger.info("开始回测 %s ~ %s (资金=¥%.0f)", start_date, end_date, 1_000_000)
    result = engine.run(
        strategy,
        data_loader=data_loader,
        feature_loader=feature_loader,
    )

    # 输出结果
    metrics = result.summary()
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:.<30s} {v:>10.4f}")
        else:
            print(f"  {k:.<30s} {v!s:>10s}")
    print("=" * 60)

    if not result.trade_records.empty:
        print(f"\n总成交 {len(result.trade_records)} 笔")
        filled = result.trade_records[result.trade_records["status"] == "filled"]
        print(f"成交 {len(filled)} 笔")


if __name__ == "__main__":
    main()
