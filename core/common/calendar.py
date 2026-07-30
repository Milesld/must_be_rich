"""A股交易日历。

基于 akshare 获取A股交易日历，提供交易日判断、区间获取、
月末/季末/年末检测等功能。首次拉取后缓存到本地 Parquet 文件。

使用方式：
    cal = get_calendar()
    cal.is_trading_day(date(2026, 6, 8))   # True
    cal.next_trading_day(date(2026, 6, 6))  # date(2026, 6, 8)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".quant_system" / "calendar"
_CACHE_FILE = _CACHE_DIR / "trading_calendar.parquet"
_CACHE_YEAR_RANGE = (2000, 2030)


class TradingCalendar:
    """A股交易日历，支持 2000-2030 年份范围。

    首次初始化时从 akshare 拉取全量日历并缓存到本地 Parquet 文件。
    后续初始化直接使用缓存，网络不可用时也回退到缓存。
    """

    def __init__(self, force_refresh: bool = False, offline_ok: bool = True) -> None:
        """初始化交易日历。

        Args:
            force_refresh: 强制从 akshare 重新拉取，忽略本地缓存。
            offline_ok: 网络不可用时是否允许使用内置估算日历（测试/离线环境用）。
        """
        self._trading_days: set[date] = set()
        self._trading_days_sorted: list[date] = []
        self._load(force_refresh=force_refresh, offline_ok=offline_ok)

    # —— 公共查询方法 ——

    def is_trading_day(self, d: date) -> bool:
        """判断是否为交易日。"""
        return d in self._trading_days

    def next_trading_day(self, d: date) -> date:
        """返回 d 之后的下一个交易日（不含 d 自身）。

        Raises:
            ValueError: 如果在缓存范围内找不到下一个交易日。
        """
        from bisect import bisect_right

        idx = bisect_right(self._trading_days_sorted, d)
        if idx >= len(self._trading_days_sorted):
            raise ValueError(f"在 {d} 之后找不到交易日")
        return self._trading_days_sorted[idx]

    def prev_trading_day(self, d: date) -> date:
        """返回 d 之前的上一个交易日（不含 d 自身）。

        Raises:
            ValueError: 如果在缓存范围内找不到上一个交易日。
        """
        from bisect import bisect_left

        idx = bisect_left(self._trading_days_sorted, d) - 1
        if idx < 0:
            raise ValueError(f"在 {d} 之前找不到交易日")
        return self._trading_days_sorted[idx]

    def get_trading_days(self, start: date, end: date) -> list[date]:
        """返回 [start, end] 区间内所有交易日（含端点）。"""
        return [d for d in self._trading_days_sorted if start <= d <= end]

    def get_next_n_trading_days(self, d: date, n: int) -> list[date]:
        """返回 d 之后的 n 个交易日（不含 d）。"""
        results: list[date] = []
        cursor = d
        for _ in range(n):
            cursor = self.next_trading_day(cursor)
            results.append(cursor)
        return results

    def get_prev_n_trading_days(self, d: date, n: int) -> list[date]:
        """返回 d 之前的 n 个交易日（不含 d）。"""
        results: list[date] = []
        cursor = d
        for _ in range(n):
            cursor = self.prev_trading_day(cursor)
            results.append(cursor)
        return results

    def is_month_end(self, d: date) -> bool:
        """判断 d 是否为当月最后一个交易日。

        d 是日历缓存的最后一天时无法看到下一个交易日，按"是最后一个交易日"处理，
        而不是抛 ValueError（末日调仓判断不该让整个流程崩掉）。
        """
        if not self.is_trading_day(d):
            return False
        next_day = self._next_trading_day_or_none(d)
        if next_day is None:
            return True
        return next_day.month != d.month

    def is_quarter_end(self, d: date) -> bool:
        """判断 d 是否为当季最后一个交易日。"""
        if not self.is_trading_day(d):
            return False
        next_day = self._next_trading_day_or_none(d)
        if next_day is None:
            return True
        # 季度切换：月份跨过一个能被3整除的边界
        current_quarter = (d.month - 1) // 3
        next_quarter = (next_day.month - 1) // 3
        return current_quarter != next_quarter or next_day.year != d.year

    def is_year_end(self, d: date) -> bool:
        """判断 d 是否为当年最后一个交易日。"""
        if not self.is_trading_day(d):
            return False
        next_day = self._next_trading_day_or_none(d)
        if next_day is None:
            return True
        return next_day.year != d.year

    def _next_trading_day_or_none(self, d: date) -> date | None:
        """下一个交易日；超出日历缓存范围返回 None。"""
        try:
            return self.next_trading_day(d)
        except ValueError:
            return None

    def count_trading_days(self, start: date, end: date) -> int:
        """返回 [start, end] 区间内的交易日数量。"""
        return len(self.get_trading_days(start, end))

    @property
    def all_trading_days(self) -> list[date]:
        """返回所有已知交易日的排序列表。"""
        return list(self._trading_days_sorted)

    # —— 内部方法 ——

    def _load(self, force_refresh: bool = False, offline_ok: bool = True) -> None:
        """加载交易日历：先试网络拉取（5秒超时），失败则用缓存或估算。

        ★ 估算日历（周一至五，节假日全算交易日）仅作离线兜底，
        缓存中以 estimated 标记区分：估算缓存每次初始化都优先尝试
        联网替换为真实日历，防止一次离线运行永久污染后续所有回测。
        """
        # 1. 真实日历缓存存在且不强制刷新 → 直接用；
        #    估算缓存则跳过此分支，继续尝试联网获取真实日历
        if not force_refresh and _CACHE_FILE.exists():
            try:
                if not self._cache_is_estimated():
                    self._load_from_cache()
                    return
                logger.info("本地缓存为估算日历（含节假日误差），尝试联网替换为真实日历")
            except Exception:
                logger.warning("缓存损坏，尝试网络拉取")

        # 2. 尝试网络拉取（5 秒超时，失败不卡住）
        import threading

        net_result: list = []
        net_error: list = []

        def _fetch_net():
            try:
                self._fetch_from_akshare()
                net_result.append(True)
            except Exception as e:
                net_error.append(str(e))

        t = threading.Thread(target=_fetch_net, daemon=True)
        t.start()
        t.join(timeout=5.0)

        if net_result:
            # 网络拉取成功 → 保存缓存（真实日历，estimated=False）
            self._save_to_cache(estimated=False)
            return

        # 3. 网络失败 → 回退（估算缓存也可用：过节假日误差 < 无日历）
        if net_error:
            logger.info("交易日历网络拉取超时/失败(%s)，使用本地缓存或估算", net_error[0][:80])
        if _CACHE_FILE.exists():
            try:
                if self._cache_is_estimated():
                    logger.warning(
                        "正在使用估算日历缓存（节假日被当作交易日），"
                        "回测结果在节假日附近会失真。联网后重跑将自动替换为真实日历。"
                    )
                self._load_from_cache()
                return
            except Exception:
                logger.warning("缓存损坏，回退估算日历")
        if offline_ok:
            logger.info("无缓存，使用估算日历（周一至周五）")
            self._build_estimated_calendar()
            self._save_to_cache(estimated=True)
        else:
            raise RuntimeError(
                "交易日历不可用：无本地缓存且 offline_ok=False。"
                "请先连网调用 get_calendar(force_refresh=True) 拉取。"
            )

    def _load_from_cache(self) -> None:
        """从本地 Parquet 缓存加载。"""
        df = pd.read_parquet(_CACHE_FILE)
        self._trading_days = {d.date() for d in pd.to_datetime(df["trade_date"])}
        self._trading_days_sorted = sorted(self._trading_days)

    @staticmethod
    def _cache_is_estimated() -> bool:
        """判断缓存是否为估算日历（由 _save_to_cache 的 estimated 列标记）。

        旧版缓存无 estimated 列：用启发式兜底——真实 A 股日历不含春节
        （检查若干年份的大年初一附近是否被标为交易日）。
        """
        try:
            df = pd.read_parquet(_CACHE_FILE)
        except Exception:
            return False
        if "estimated" in df.columns:
            return bool(df["estimated"].iloc[0])
        # 旧缓存启发式：2024-02-12（春节初三，周一）绝不可能是真实交易日
        days = {d.date() for d in pd.to_datetime(df["trade_date"])}
        from datetime import date as _d
        spring_festival_probes = [_d(2024, 2, 12), _d(2025, 1, 29), _d(2023, 1, 24)]
        return any(p in days for p in spring_festival_probes)

    def _fetch_from_akshare(self) -> None:
        """从 akshare 拉取A股交易日历（仅在 force_refresh=True 时调用）。"""
        import akshare as ak

        df = ak.tool_trade_date_hist_sina()
        df.columns = ["trade_date"]
        self._trading_days = {
            d.date() for d in pd.to_datetime(df["trade_date"])
            if _CACHE_YEAR_RANGE[0] <= d.year <= _CACHE_YEAR_RANGE[1]
        }
        self._trading_days_sorted = sorted(self._trading_days)
        logger.info(
            "从 akshare 拉取交易日历完成：%d 个交易日",
            len(self._trading_days_sorted),
        )

    def _build_estimated_calendar(self) -> None:
        """离线模式下构建估算日历：周一至周五排除元旦/春节/国庆等粗略处理。"""
        import datetime as dt

        self._trading_days = set()
        for year in range(_CACHE_YEAR_RANGE[0], _CACHE_YEAR_RANGE[1] + 1):
            d = dt.date(year, 1, 1)
            while d.year == year:
                # 周末排除
                if d.weekday() < 5:
                    self._trading_days.add(d)
                d += dt.timedelta(days=1)
        self._trading_days_sorted = sorted(self._trading_days)
        logger.warning(
            "使用估算交易日历（%d 天），不含节假日排除。建议在联网后重新初始化。",
            len(self._trading_days_sorted),
        )

    def _save_to_cache(self, estimated: bool = False) -> None:
        """保存到本地 Parquet 缓存。

        Args:
            estimated: 是否为估算日历。估算缓存会在下次初始化时优先
                       尝试联网替换，不会永久生效。
        """
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"trade_date": self._trading_days_sorted})
        df["estimated"] = estimated
        df.to_parquet(_CACHE_FILE, index=False)
        logger.debug("交易日历已缓存到 %s (estimated=%s)", _CACHE_FILE, estimated)


# —— 全局单例 ——

_calendar_instance: Optional[TradingCalendar] = None


def get_calendar() -> TradingCalendar:
    """获取 TradingCalendar 全局单例。"""
    global _calendar_instance
    if _calendar_instance is None:
        _calendar_instance = TradingCalendar()
    return _calendar_instance
