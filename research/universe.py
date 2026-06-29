#!/usr/bin/env python3
"""动态股票池 — PIT（point-in-time）候选宇宙构建。

解决幸存者偏差：原先的固定 `data_source.codes` 是 2026 年视角手选的赢家，
回测 2023–2025 等于"知道答案再考试"。本模块改为每月初用"当时已知"的信息
从行业板块成分中筛选候选宇宙，再在宇宙内选股。

★ 残留偏差（诚实标注，未消除）：
    akshare 的 stock_board_industry_cons_em 返回的是"当前"行业成分，
    回放到 2023 年仍有轻度幸存者偏差——2023 年的真实成分可能不同，
    且已退市的股票无法获取。含退市股的 PIT 成分库免费数据源基本拿不到。
    本方案比"手选 2026 赢家"好一个数量级，但不是零偏差。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 行业成分本地缓存：成分股慢变（月度才调），缓存到磁盘防限流+加速。
_CONS_CACHE_DIR = Path.home() / ".quant_system" / "industry_cons"
_CONS_CACHE_TTL_DAYS = 7   # 缓存有效期（天）


def _cons_cache_path(industry: str) -> Path:
    # 行业名可能含特殊字符，用安全文件名
    safe = "".join(c if c.isalnum() else "_" for c in industry)
    return _CONS_CACHE_DIR / f"{safe}.json"


def _load_cons_cache(industry: str) -> list[str] | None:
    """读行业成分缓存；过期或不存在返回 None。"""
    p = _cons_cache_path(industry)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
        ts = datetime.fromisoformat(obj["fetched_at"])
        if (datetime.now() - ts).days > _CONS_CACHE_TTL_DAYS:
            return None
        codes = obj.get("codes", [])
        return codes if codes else None
    except Exception:
        return None


def _save_cons_cache(industry: str, codes: list[str]) -> None:
    try:
        _CONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cons_cache_path(industry).write_text(json.dumps(
            {"industry": industry, "fetched_at": datetime.now().isoformat(), "codes": codes},
            ensure_ascii=False,
        ))
    except Exception as e:
        logger.debug("行业 %s 成分缓存写入失败: %s", industry, e)


def _fetch_industry_cons(industry: str) -> list[str] | None:
    """拉单个行业成分（重试3次+退避），返回 6 位代码列表或 None。"""
    import akshare as ak
    for attempt in range(3):
        try:
            df = ak.stock_board_industry_cons_em(symbol=industry)
            if df is not None and len(df) > 0:
                code_col = "代码" if "代码" in df.columns else df.columns[1]
                out: list[str] = []
                for raw in df[code_col].astype(str):
                    c = raw.strip().zfill(6)
                    if c.isdigit() and len(c) == 6:
                        out.append(c)
                return out
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def build_industry_universe(industries: list[str]) -> list[str]:
    """从 akshare 行业板块取成分股代码（去重保序）。

    优先读本地缓存（{_CONS_CACHE_TTL_DAYS} 天有效）；缓存缺失/过期才联网，
    联网成功后回写缓存。这样防东财限流（RemoteDisconnected）+ 加速反复回测。

    Args:
        industries: akshare 行业板块名列表，如 ["半导体", "光学光电子"]。
                    准确名称可用 ak.stock_board_industry_name_em() 核对。

    Returns:
        所有行业成分股代码的并集（6位代码字符串）。

    ★ 返回的是"当前"成分，含轻度幸存者偏差（见模块 docstring）。
    """
    codes: list[str] = []
    seen: set[str] = set()
    failed: list[str] = []

    for industry in industries:
        cons = _load_cons_cache(industry)
        src = "缓存"
        if cons is None:
            cons = _fetch_industry_cons(industry)
            src = "联网"
            if cons:
                _save_cons_cache(industry, cons)
        if not cons:
            logger.warning("行业 %s 成分股拉取失败（缓存无 + 联网重试3次失败）", industry)
            failed.append(industry)
            continue
        added = 0
        for c in cons:
            if c not in seen:
                codes.append(c)
                seen.add(c)
                added += 1
        logger.info("行业 %s: %d 只成分股（%s）", industry, len(cons), src)

    if failed:
        # 关键行业拉空会让候选宇宙残缺、回测失真，必须显式报错而非静默继续
        raise RuntimeError(
            f"以下行业成分股拉取失败，候选宇宙不完整: {failed}。"
            f"东财接口可能临时限流（RemoteDisconnected），稍后重跑即可；"
            f"或核对行业名 (ak.stock_board_industry_name_em())。"
            f"成功拉取的行业已缓存到 {_CONS_CACHE_DIR}。"
        )

    logger.info("候选宇宙合计 %d 只（%d 个行业并集）", len(codes), len(industries))
    return codes


# 上市日缓存：{code: 上市 date 或 None}，避免重复联网拉取
_LISTING_DATE_CACHE: dict[str, "date | None"] = {}


def _fetch_one_listing_date(code: str) -> tuple[str, "date | None"]:
    """拉取单只股票的真实上市日（供进程池调用，须为模块顶层函数）。

    用 akshare stock_individual_info_em 的"上市时间"字段。拿不到返回 None。
    """
    import akshare as ak

    try:
        df = ak.stock_individual_info_em(symbol=code)
        # 返回长表：item/value 两列
        row = df[df["item"] == "上市时间"]
        if row.empty:
            return code, None
        raw = str(row["value"].iloc[0]).strip()
        # 形如 "20100617" 或 "2010-06-17"
        raw = raw.replace("-", "")
        if len(raw) >= 8 and raw[:8].isdigit():
            from datetime import date as _date
            return code, _date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
        return code, None
    except Exception:
        return code, None


def fetch_listing_dates(codes: list[str], max_workers: int | None = None) -> dict[str, "date | None"]:
    """并发拉取一批股票的真实上市日（进程池，带缓存）。

    Returns:
        {code: 上市 date 或 None}。拿不到的为 None（PIT 筛选中按"未知上市日"处理）。
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    if max_workers is None:
        max_workers = int(os.environ.get("FETCH_WORKERS", "8"))

    todo = [c for c in codes if c not in _LISTING_DATE_CACHE]
    if todo:
        logger.info("正在拉取 %d 只股票的真实上市日（进程池）...", len(todo))
        done = 0
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_fetch_one_listing_date, c) for c in todo]
            for fut in as_completed(futures):
                try:
                    code, d = fut.result()
                except Exception:
                    code, d = None, None
                if code is not None:
                    _LISTING_DATE_CACHE[code] = d
                done += 1
                if done % 50 == 0 or done == len(todo):
                    logger.info("  上市日拉取进度: %d/%d", done, len(todo))
    return {c: _LISTING_DATE_CACHE.get(c) for c in codes}


def filter_universe_pit(
    candidates: list[str],
    as_of_date: date,
    raw_data: dict[date, dict],
    min_listing_months: int = 12,
    min_avg_amount: float = 200_000_000.0,
    lookback_days: int = 60,
    exclude_st: bool = True,
    pool_size: int | None = None,
    listing_dates: dict[str, "date | None"] | None = None,
) -> list[str]:
    """PIT 筛选：只用 as_of_date 之前的数据，确定当月可投宇宙。

    Args:
        candidates: 候选代码（来自 build_industry_universe）。
        as_of_date: 筛选基准日（月初调仓日）。只用此日期之前的数据。
        raw_data: {date: {code: {close, amount, is_st, ...}}} 全量行情。
        min_listing_months: 最小上市月数（剔除次新）。
        min_avg_amount: 过去 lookback_days 日均成交额下限（流动性，单位元）。
        lookback_days: 流动性统计回溯天数。
        exclude_st: 是否剔除 ST（用 as_of_date 当日 is_st 状态）。
        pool_size: 宇宙容量上限。None=不限；否则按日均成交额降序取 top。
        listing_dates: {code: 真实上市 date}。给定时用真实上市日判断次新；
                       否则回退到"raw_data 内交易日数"近似（仅当数据起点早于
                       回测起点足够时才准确）。

    Returns:
        通过 PIT 筛选的代码列表（按日均成交额降序）。
    """
    # as_of_date 之前的交易日（升序），用于流动性与上市时长统计
    past_dates = sorted(d for d in raw_data if d < as_of_date)
    if not past_dates:
        return []

    # 每只股票的历史交易日（升序）
    code_dates: dict[str, list[date]] = {}
    for d in past_dates:
        for code in raw_data[d]:
            code_dates.setdefault(code, []).append(d)

    # 上市时长门槛：优先用真实上市日；否则回退到"数据内交易日数"近似（约 21 日/月）
    from dateutil.relativedelta import relativedelta
    listing_cutoff = as_of_date - relativedelta(months=min_listing_months)
    min_listing_days = int(min_listing_months * 21)
    recent = past_dates[-lookback_days:]
    listing_dates = listing_dates or {}

    scored: list[tuple[str, float]] = []
    for code in candidates:
        dates_for_code = code_dates.get(code)
        if not dates_for_code:
            continue  # 该候选在回测数据中无任何记录

        # 次新剔除
        ld = listing_dates.get(code)
        if ld is not None:
            # 真实上市日：上市日须早于 (as_of - min_listing_months)
            if ld > listing_cutoff:
                continue
        else:
            # 回退：用数据内交易日数近似（数据起点不够早时会偏严）
            if len(dates_for_code) < min_listing_days:
                continue

        # ST 剔除：用 as_of_date 之前最近一个交易日的 is_st 状态
        if exclude_st:
            last_d = dates_for_code[-1]
            if raw_data.get(last_d, {}).get(code, {}).get("is_st", False):
                continue

        # 流动性：过去 lookback_days 日均成交额
        amts = [
            float(raw_data[d][code].get("amount", 0))
            for d in recent
            if code in raw_data.get(d, {})
        ]
        if not amts:
            continue
        avg_amt = sum(amts) / len(amts)
        if avg_amt < min_avg_amount:
            continue

        scored.append((code, avg_amt))

    # 按日均成交额降序
    scored.sort(key=lambda x: -x[1])
    result = [c for c, _ in scored]
    if pool_size is not None and pool_size > 0:
        result = result[:pool_size]
    return result
