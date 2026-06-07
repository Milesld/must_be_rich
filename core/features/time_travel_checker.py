"""时间旅行安全检查器（★系统最重要的安全模块★）。

防止回测中的前视偏差（Look-ahead Bias）——这是量化系统中最致命、最隐蔽的错误。

核心逻辑：在回测的每一个时间点，严格校验所有使用的数据的时间戳不晚于该时间点。
特别关注财务数据中的 report_date 和 announce_date 关系：
绝不允许 announce_date > date 的数据被使用。

使用方式：
    checker = TimeTravelChecker()
    checker.check(date(2020, 2, 10), {
        "market_daily": {"max_trade_date": date(2020, 2, 10)},
        "financial_statements": {"max_announce_date": date(2020, 2, 15)},  # ← 违规！
    })
    # → 抛出 TimeTravelError
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEBUG = os.environ.get("ENV", "dev") == "dev"


class TimeTravelError(Exception):
    """前视偏差错误——回测使用了尚未发生的数据。

    这表示某个特征计算管线在回测时间点使用了未来的信息。
    必须修复后才能继续回测。
    """

    def __init__(
        self,
        as_of_date: date,
        violations: list[TimeTravelViolation],
    ) -> None:
        self.as_of_date = as_of_date
        self.violations = violations
        msg_lines = [
            f"TimeTravelError: 回测日期 {as_of_date} 检测到 {len(violations)} 条前视偏差违规："
        ]
        for v in violations[:10]:
            msg_lines.append(f"  - {v}")
        if len(violations) > 10:
            msg_lines.append(f"  ... 还有 {len(violations) - 10} 条")
        super().__init__("\n".join(msg_lines))


@dataclass
class TimeTravelViolation:
    """单条前视偏差违规记录。"""
    data_source: str           # 数据源/表名
    field_name: str            # 违规字段
    observed_max_value: Any    # 该字段在数据中的最大值（未来值）
    as_of_date: date           # 回测截止日期
    detail: str = ""           # 额外说明
    violating_codes: list[str] = field(default_factory=list)  # 涉及股票代码


class TimeTravelChecker:
    """时间旅行安全检查器。

    在回测的每个时间点调用 check()，验证所有数据来源的时间戳
    严格不晚于回测日期。如果发现违规，抛出 TimeTravelError。
    """

    def __init__(self, strict_mode: bool = True) -> None:
        """初始化检查器。

        Args:
            strict_mode: True 时违规直接抛出异常阻断。
                         False 时仅告警记录到 audit log（仅用于调试）。
        """
        self.strict_mode = strict_mode
        self._total_checks = 0
        self._total_violations = 0

    def check(
        self,
        as_of_date: date,
        data_sources: dict[str, dict[str, Any]],
    ) -> bool:
        """检查所有数据源是否在 as_of_date 之前可用。

        Args:
            as_of_date: 回测时间点（这天开盘时的可用信息）。
            data_sources: 数据源信息字典。
                格式: {
                    "market_daily": {"max_trade_date": date(2020, 2, 10)},
                    "financial_statements": {
                        "max_announce_date": date(2020, 2, 5),
                        "codes_with_violations": ["600519", ...]  # 可选
                    },
                    "factor_values": {"calc_dates": [date(2020, 2, 9), ...]},
                }

        Returns:
            True 如果全部通过。

        Raises:
            TimeTravelError: 在 strict_mode=True 且检测到违规时抛出。
        """
        self._total_checks += 1
        violations: list[TimeTravelViolation] = []

        for source_name, meta in data_sources.items():
            vs = self._check_source(source_name, as_of_date, meta)
            violations.extend(vs)

        if violations:
            self._total_violations += len(violations)
            logger.debug(
                "TimeTravel 检查: as_of=%s, violations=%d",
                as_of_date, len(violations),
            )
            if self.strict_mode:
                raise TimeTravelError(as_of_date, violations)
            # 非严格模式：记录告警但继续
            for v in violations:
                logger.warning("TimeTravel告警(非严格): %s", v)

        return len(violations) == 0

    # ── 专用检查方法 ────────────────────────────

    def check_financials(
        self,
        as_of_date: date,
        df_meta: dict[str, Any],
    ) -> list[TimeTravelViolation]:
        """检查财务报表数据是否有前视偏差。

        核心规则：announce_date（实际披露日）必须 <= as_of_date。
        如果系统错误地使用了 report_date（报告期截止日），
        而 announce_date > as_of_date，说明数据尚未披露——这是典型的前视偏差。

        Args:
            as_of_date: 回测日期。
            df_meta: 财务报表元信息。需要包含 "max_announce_date"，
                     可选 "max_report_date"（用于对比检测常见错误）。

        Returns:
            违规列表。
        """
        violations: list[TimeTravelViolation] = []

        max_announce = df_meta.get("max_announce_date")
        if max_announce is not None and max_announce > as_of_date:
            violations.append(TimeTravelViolation(
                data_source="financial_statements",
                field_name="announce_date",
                observed_max_value=max_announce,
                as_of_date=as_of_date,
                detail=(
                    f"存在 announce_date={max_announce} > 回测日期={as_of_date} 的财报数据。"
                    "这意味着特征计算使用了当时尚未披露的财务信息。"
                    "请确认因子管线使用的是 announce_date 而非 report_date 作为可用时间点。"
                ),
                violating_codes=df_meta.get("codes_with_violations", []),
            ))

        # 额外检查：如果提供的 max_report_date 远大于 as_of_date 但
        # announce_date 正常，这本身不是问题，但记录 DEBUG 供排查
        max_report = df_meta.get("max_report_date")
        if max_report is not None and max_report > as_of_date:
            if max_announce is None or max_announce <= as_of_date:
                logger.debug(
                    "TimeTravel: as_of=%s, max_report_date=%s > as_of_date 但 announce_date 合法（正常）",
                    as_of_date, max_report,
                )

        return violations

    # ── 内部方法 ───────────────────────────────

    def _check_source(
        self,
        source_name: str,
        as_of_date: date,
        meta: dict[str, Any],
    ) -> list[TimeTravelViolation]:
        """检查单个数据源。"""
        violations: list[TimeTravelViolation] = []

        # 按字段名和时间类型匹配
        date_field_map = {
            "max_trade_date": "trade_date",
            "max_announce_date": "announce_date",
            "max_report_date": "report_date",
            "max_calc_date": "calc_date",
            "calc_dates": "calc_date",           # 列表形式的因子计算日期
            "max_date": "date",
        }

        for meta_key, field_name in date_field_map.items():
            if meta_key not in meta:
                continue

            value = meta[meta_key]
            # 支持列表或单个日期
            if isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, date) and v > as_of_date:
                        violations.append(TimeTravelViolation(
                            data_source=source_name,
                            field_name=field_name,
                            observed_max_value=v,
                            as_of_date=as_of_date,
                            detail=f"{source_name} 中的 {field_name}={v} 晚于回测日期 {as_of_date}",
                        ))
                        break
            elif isinstance(value, date) and value > as_of_date:
                violations.append(TimeTravelViolation(
                    data_source=source_name,
                    field_name=field_name,
                    observed_max_value=value,
                    as_of_date=as_of_date,
                    detail=f"{source_name} 中的 {field_name}={value} 晚于回测日期 {as_of_date}",
                    violating_codes=meta.get("codes_with_violations", []),
                ))

        return violations

    # ── 统计 ──────────────────────────────────

    @property
    def stats(self) -> dict[str, int]:
        """返回检查统计。"""
        return {
            "total_checks": self._total_checks,
            "total_violations": self._total_violations,
        }

    def reset_stats(self) -> None:
        """重置统计计数器。"""
        self._total_checks = 0
        self._total_violations = 0


def validate_pit_financials(
    as_of_date: date,
    announce_dates: list[date],
    codes: Optional[list[str]] = None,
) -> list[str]:
    """Point-in-Time 快速验证：检查所有 announce_date 是否 <= as_of_date。

    这是一个便捷函数，可以直接传入 DataFrame 的 announce_date 列。

    Args:
        as_of_date: 回测基准日期。
        announce_dates: 财报实际披露日列表。
        codes: 关联的股票代码列表。

    Returns:
        违规的 announce_date 列表（字符串格式），可直接用于日志。
    """
    violations: list[str] = []
    for i, ad in enumerate(announce_dates):
        if ad > as_of_date:
            code_str = codes[i] if codes and i < len(codes) else f"index_{i}"
            violations.append(
                f"code={code_str} announce_date={ad} > as_of_date={as_of_date}"
            )
    return violations
