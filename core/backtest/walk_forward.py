"""Walk-Forward 滚动验证。

金融时序不能用随机 K-Fold 交叉验证——会引入前视偏差。
Walk-Forward 按时间顺序滚动训练和测试窗口，保证训练数据
严格在测试数据之前。

Purge 机制：在训练窗口和测试窗口之间留出清除期，防止
训练窗口末尾的数据泄露到测试窗口开头。
"""

from __future__ import annotations

import logging
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd

from core.common.calendar import get_calendar

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """时序 Walk-Forward 滚动验证。

    使用方式：
        wfv = WalkForwardValidator(train_window_years=5, test_window_months=3)
        windows = wfv.get_windows(start, end)
        for train_s, train_e, test_s, test_e in windows:
            model = train(train_s, train_e)
            predictions[test_s:test_e] = predict(model, test_s, test_e)
    """

    def __init__(
        self,
        train_window_years: int = 5,
        test_window_months: int = 3,
        purge_days: int = 5,
        min_train_years: int = 2,
    ) -> None:
        """初始化。

        Args:
            train_window_years: 训练窗口长度（日历年）。
            test_window_months: 测试窗口长度（日历月，每次前滚的步长）。
            purge_days: 训练/测试之间的清除期（交易日）。
                设为5天可清除训练窗口末尾的短期信号残留。
            min_train_years: 最小训练窗口（年）。不足时不产生窗口。
        """
        self.train_window_years = train_window_years
        self.test_window_months = test_window_months
        self.purge_days = purge_days
        self.min_train_years = min_train_years

    def get_windows(
        self, start: date, end: date,
    ) -> list[tuple[date, date, date, date]]:
        """生成 Walk-Forward 窗口序列。

        Returns:
            [(train_start, train_end, test_start, test_end), ...]
            每个窗口的训练/测试日期均为日历日（后续使用时再过滤为交易日）。

        示例：
            2015-01-01 ~ 2025-12-31, train=5年, test=3月
            Window 1: train(2015~2019), test(2020 Q1)
            Window 2: train(2015 Q2~2020 Q1), test(2020 Q2)
            ...
        """
        windows: list[tuple[date, date, date, date]] = []

        train_s = start
        train_e = self._add_years(train_s, self.train_window_years)

        while True:
            test_s = train_e
            # 加 purge 间隔
            cal = get_calendar()
            test_s_purged = test_s
            for _ in range(self.purge_days):
                try:
                    test_s_purged = cal.next_trading_day(test_s_purged)
                except ValueError:
                    break

            test_e = self._add_months(test_s_purged, self.test_window_months)

            if test_e > end:
                break
            if test_s_purged >= end:
                break

            # 检查最小训练窗口
            train_years = (train_e - train_s).days / 365.25
            if train_years >= self.min_train_years:
                windows.append((train_s, train_e, test_s_purged, test_e))

            # 前滚
            train_s = self._add_months(train_s, self.test_window_months)
            train_e = self._add_months(train_e, self.test_window_months)

            if train_e >= end:
                break

        logger.info(
            "生成 %d 个 Walk-Forward 窗口 (%s ~ %s)",
            len(windows), start, end,
        )
        return windows

    def validate(
        self,
        model_train_fn: Callable[[date, date], Any],
        model_predict_fn: Callable[[Any, date, date], pd.DataFrame],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """执行 Walk-Forward 滚动验证。

        Args:
            model_train_fn: 训练函数 f(train_start, train_end) → model。
            model_predict_fn: 预测函数 f(model, test_start, test_end) → DataFrame。
            start: 数据起始日。
            end: 数据截止日。

        Returns:
            所有测试窗口的拼接预测结果 DataFrame。
        """
        windows = self.get_windows(start, end)
        if not windows:
            logger.warning("无有效 Walk-Forward 窗口: %s ~ %s", start, end)
            return pd.DataFrame()

        all_predictions: list[pd.DataFrame] = []

        for i, (train_s, train_e, test_s, test_e) in enumerate(windows):
            logger.info(
                "WF 窗口 %d/%d: train=[%s, %s] test=[%s, %s]",
                i + 1, len(windows), train_s, train_e, test_s, test_e,
            )

            try:
                model = model_train_fn(train_s, train_e)
                preds = model_predict_fn(model, test_s, test_e)
                if preds is not None and len(preds) > 0:
                    preds["wf_window"] = i
                    preds["train_start"] = train_s
                    preds["train_end"] = train_e
                    all_predictions.append(preds)
            except Exception as e:
                logger.error("WF 窗口 %d 失败: %s", i + 1, e)
                continue

        if not all_predictions:
            return pd.DataFrame()

        result = pd.concat(all_predictions, ignore_index=True)
        logger.info(
            "Walk-Forward 完成: %d 窗口, 总预测 %d 条",
            len(windows), len(result),
        )
        return result

    # ── 辅助 ─────────────────────────────────

    @staticmethod
    def _add_months(d: date, months: int) -> date:
        return d + relativedelta(months=months)

    @staticmethod
    def _add_years(d: date, years: int) -> date:
        return d + relativedelta(years=years)
