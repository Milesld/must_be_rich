#!/usr/bin/env python3
"""westock-data 数据源适配层（provider: westock）。

封装对 westock-data skill（node CLI，腾讯自选股独立数据源）的调用，
把输出转成回测引擎要的格式。用于绕开 akshare 东财接口的限流
（RemoteDisconnected）。

westock CLI 调用：node <仓库>/westock-data/scripts/index.js <子命令> ... --raw

已验证字段（kline --raw）：
    {symbol, date, open, last, high, low, volume, amount, exchange}
    last=收盘价, exchange=换手率(%), amount=成交额(元)；缺 pre_close（用前一日 close 回填）。
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_INDEX_JS = _PROJECT_DIR / "westock-data" / "scripts" / "index.js"

# 行业成分/上市日本地缓存（慢变数据，防限流+加速）
_CACHE_DIR = Path.home() / ".quant_system" / "westock_cache"
_CONS_TTL_DAYS = 7
_LISTING_TTL_DAYS = 30


def _node_exe() -> str:
    """从 PATH 定位 node 可执行文件，缺失时抛清晰错误。"""
    exe = shutil.which("node")
    if not exe:
        raise RuntimeError(
            "未找到 node 可执行文件（PATH 中无 node）。westock 数据源需要 Node.js ≥18。"
            "请安装 Node 或确认它在 PATH 中。"
        )
    return exe


def _run_westock(args: list[str], timeout: float = 120.0) -> str:
    """调用 westock CLI，返回 stdout 文本。非零退出抛异常。"""
    if not _INDEX_JS.exists():
        raise RuntimeError(f"westock CLI 不存在: {_INDEX_JS}")
    cmd = [_node_exe(), str(_INDEX_JS), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"westock 调用失败 (exit {proc.returncode}): {' '.join(args)}\n{proc.stderr[:500]}"
        )
    return proc.stdout


def _run_westock_json(args: list[str], timeout: float = 120.0):
    """调用 westock CLI（带 --raw）并解析 JSON。

    westock --raw 有两种结构：
    - kline/sector：直接返回数组 [{...}]
    - profile 等：返回 {success, data: {...}}（下钻 data）
    统一返回内层数据（数组或 dict）。
    """
    out = _run_westock([*args, "--raw"], timeout=timeout)
    out = out.strip()
    if not out:
        return None
    try:
        obj = json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"westock 输出非 JSON: {' '.join(args)}\n{out[:300]}") from e
    # 显式识别错误响应 {success: false, error: {...}}，避免下游把它当数据迭代
    if isinstance(obj, dict) and obj.get("success") is False:
        err = obj.get("error") or {}
        raise RuntimeError(
            f"westock 返回错误 [{err.get('code', '?')}]: "
            f"{err.get('message', '未知错误')} ({' '.join(args)})"
        )
    # 下钻 {success, data} 包装
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    return obj


# ══════════════════════════════════════════════════════════════
# 缓存工具
# ══════════════════════════════════════════════════════════════

def _cache_get(key: str, ttl_days: int):
    p = _CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
        ts = datetime.fromisoformat(obj["fetched_at"])
        if (datetime.now() - ts).days > ttl_days:
            return None
        return obj.get("data")
    except Exception:
        return None


def _cache_put(key: str, data) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (_CACHE_DIR / f"{key}.json").write_text(json.dumps(
            {"fetched_at": datetime.now().isoformat(), "data": data}, ensure_ascii=False,
        ))
    except Exception as e:
        logger.debug("westock 缓存写入失败 %s: %s", key, e)


# ══════════════════════════════════════════════════════════════
# 行业成分
# ══════════════════════════════════════════════════════════════

def westock_industry_cons(board_codes: list[str]) -> list[str]:
    """取申万板块成分股代码并集（去重保序）。

    Args:
        board_codes: westock pt 板块码列表，如 ["pt01801081"]（申万二级-半导体）。

    Returns:
        纯 6 位成分代码并集（与 raw_data/PIT 链路一致；westock 前缀在 kline 时再加）。

    ★ 返回"当前"成分，回放历史仍有轻度幸存者偏差（与 akshare 同）。
    """
    codes: list[str] = []
    seen: set[str] = set()
    for board in board_codes:
        cached = _cache_get(f"cons_{board}", _CONS_TTL_DAYS)
        if cached is not None:
            cons = cached
            src = "缓存"
        else:
            try:
                data = _run_westock_json(["sector", "constituent", board])
                cons = [_strip_prefix(str(r["code"])) for r in (data or [])
                        if isinstance(r, dict) and r.get("code")]
            except RuntimeError as ex:
                # 联网失败时回退过期缓存（成分月度慢变，过期数据远好于中断）
                stale = _cache_get(f"cons_{board}", ttl_days=365)
                if stale:
                    logger.warning("westock 板块 %s 联网失败(%s)，回退过期缓存(%d只)",
                                   board, ex, len(stale))
                    cons = stale
                    src = "过期缓存"
                    for c in cons:
                        if c not in seen:
                            codes.append(c)
                            seen.add(c)
                    continue
                raise
            if not cons:
                logger.warning("westock 板块 %s 返回空成分", board)
                raise RuntimeError(f"westock 板块 {board} 成分为空，候选宇宙不完整")
            _cache_put(f"cons_{board}", cons)
            src = "联网"
        added = 0
        for c in cons:
            if c not in seen:
                codes.append(c)
                seen.add(c)
                added += 1
        logger.info("westock 板块 %s: %d 只成分（%s，新增 %d）", board, len(cons), src, added)
    logger.info("westock 候选宇宙合计 %d 只（%d 个板块并集）", len(codes), len(board_codes))
    return codes


# ══════════════════════════════════════════════════════════════
# 上市日
# ══════════════════════════════════════════════════════════════

def westock_listing_dates(codes: list[str]) -> dict[str, "date | None"]:
    """逐只取真实上市日（profile 命令），带缓存。

    Returns:
        {code: 上市 date 或 None}
    """
    result: dict[str, date | None] = {}
    todo: list[str] = []
    for c in codes:
        cached = _cache_get(f"listing_{c}", _LISTING_TTL_DAYS)
        if cached is not None:
            result[c] = date.fromisoformat(cached) if cached else None
        else:
            todo.append(c)

    if todo:
        logger.info("westock 拉取 %d 只股票上市日（profile）...", len(todo))
        for i, c in enumerate(todo):
            ld = _fetch_one_listing(c)
            result[c] = ld
            _cache_put(f"listing_{c}", ld.isoformat() if ld else "")
            if (i + 1) % 50 == 0:
                logger.info("  上市日进度: %d/%d", i + 1, len(todo))
    return result


def _fetch_one_listing(code: str) -> "date | None":
    """从 profile 解析单只上市日。profile 字段名因 westock 版本而异，做多键兜底。"""
    try:
        data = _run_westock_json(["profile", to_westock_code(code)], timeout=30.0)
    except Exception:
        return None
    if not data:
        return None
    # profile 可能返回 dict 或单元素 list
    obj = data[0] if isinstance(data, list) and data else data
    if not isinstance(obj, dict):
        return None
    # 常见上市日字段名兜底（westock profile 实测为 listedDate）
    for k in ("listedDate", "listingDate", "ipoDate", "listDate", "上市时间", "上市日期", "ListDate"):
        v = obj.get(k)
        if v:
            return _parse_listing_date(str(v))
    return None


def _parse_listing_date(raw: str) -> "date | None":
    raw = raw.strip().replace("-", "").replace("/", "")[:8]
    if len(raw) == 8 and raw.isdigit():
        try:
            return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
        except ValueError:
            return None
    return None


# ══════════════════════════════════════════════════════════════
# 行情 K 线
# ══════════════════════════════════════════════════════════════

# westock 单只单次范围查询有 ~244 条硬上限（约一年），超出只返回最近段。
# 故把长区间切成 ~半年(180 自然日)子窗逐段拉，每段 ~120 交易日 < 244，拼全历史。
_SEG_DAYS = 180


def _date_segments(start: date, end: date, seg_days: int = _SEG_DAYS):
    """把 [start, end] 切成不超过 seg_days 的连续子区间。"""
    from datetime import timedelta
    segs = []
    cur = start
    while cur <= end:
        seg_end = min(cur + timedelta(days=seg_days - 1), end)
        segs.append((cur, seg_end))
        cur = seg_end + timedelta(days=1)
    return segs


def _fetch_one_segment(code: str, s: str, e: str, max_retries: int = 2) -> list[dict]:
    """拉单只单段 K 线，返回标准化 row 列表（含 code）。带行情磁盘缓存。

    密集请求会触发 westock 限流（返回 None）。策略：段间小延时 + 拉空重试
    max_retries 次（退避 3s/6s/9s...）。个股(277只)用默认 2 次求快；行业轮动
    (十几条)可传更大值(如 4)确保拉全。注意空数组=该段真无数据(次新)，不重试。
    """
    cache_key = f"kline_{code}_{s}_{e}"
    cached = _cache_get(cache_key, ttl_days=3650)  # 历史行情不变，长期缓存
    if cached is not None:
        return cached

    import time
    time.sleep(0.15)  # 段间基础延时，降低触发限流概率

    data = None
    for attempt in range(max_retries + 1):  # 首次 + max_retries 次重试
        try:
            data = _run_westock_json(
                ["kline", to_westock_code(code), "--period", "day",
                 "--start", s, "--end", e, "--fq", "qfq"],
                timeout=60.0,
            )
        except Exception as ex:
            logger.warning("westock kline %s %s~%s 调用异常: %s", code, s, e, ex)
            data = None
        if isinstance(data, list):
            break  # 拿到数组（含空数组=该段真无数据，不重试）
        if attempt < max_retries:
            time.sleep(3.0 * (attempt + 1))  # 限流退避 3s/6s/9s...

    # 非数组（None/str）= 限流或错误；空数组 = 该段真无数据（次新）
    if not isinstance(data, list):
        logger.warning("westock kline %s %s~%s 重试后仍非数组（疑限流），跳过该段",
                       code, s, e)
        return []
    rows: list[dict] = []
    for r in data:
        if not isinstance(r, dict):
            continue
        d = r.get("date")
        close = r.get("last", r.get("close"))
        if not d or close in (None, 0):
            continue
        rows.append({
            "code": code,
            "date": str(d)[:10],
            "open": r.get("open", close),
            "high": r.get("high", close),
            "low": r.get("low", close),
            "close": close,
            "volume": r.get("volume", 0),
            "amount": r.get("amount", 0),
            "turnover": r.get("exchange", 0),  # exchange 实为换手率%
        })
    _cache_put(cache_key, rows)
    return rows


def westock_kline(codes: list[str], start: date, end: date, batch: int | None = None,
                  max_retries: int = 2):
    """拉日 K 线（前复权），分段突破 244 条上限，合并为统一 DataFrame。

    单只 × 多段（每段 ≤ _SEG_DAYS 自然日），逐段拉再拼接去重。
    每只每段带磁盘缓存（历史行情不变，TTL 长），重跑秒出。
    batch 参数保留兼容签名，实际按单只分段拉（westock 单只查询更稳）。
    max_retries：单段拉空重试次数。个股用默认 2（快）；行业轮动数据少可传 4（拉全）。

    Returns:
        DataFrame[code, trade_date, open, high, low, close, volume, amount, turnover]
        （code 纯 6 位）。无数据返回 None。
    """
    import pandas as pd

    segs = _date_segments(start, end)
    rows: list[dict] = []
    total = len(codes)
    for idx, code in enumerate(codes):
        for (cs, ce) in segs:
            rows.extend(_fetch_one_segment(code, cs.strftime("%Y-%m-%d"),
                                           ce.strftime("%Y-%m-%d"), max_retries=max_retries))
        if (idx + 1) % 25 == 0 or idx + 1 == total:
            logger.info("  westock 行情进度: %d/%d 只（每只 %d 段）", idx + 1, total, len(segs))

    if not rows:
        return None
    df = pd.DataFrame(rows)
    # date 字符串 → date 对象
    df["trade_date"] = df["date"].map(lambda x: date.fromisoformat(x))
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        df[col] = df[col].astype(float)
    df = df[["code", "trade_date", "open", "high", "low", "close", "volume", "amount", "turnover"]]
    return df.drop_duplicates(subset=["code", "trade_date"])



def _strip_prefix(symbol: str) -> str:
    """westock 代码 sh688593/sz300053/bj920139 → 纯 6 位码（与现有 raw_data 一致）。"""
    s = symbol.strip()
    for p in ("sh", "sz", "bj"):
        if s.startswith(p):
            return s[len(p):]
    return s


def to_westock_code(code: str) -> str:
    """纯 6 位码 → westock 前缀码（sh6.../sz0../sz3../bj...）。"""
    code = code.strip()
    if code.startswith(("sh", "sz", "bj")):
        return code
    if code.startswith("6"):
        return "sh" + code
    if code.startswith(("0", "3")):
        return "sz" + code
    if code.startswith(("8", "4", "92")):
        return "bj" + code
    return "sh" + code


# ══════════════════════════════════════════════════════════════
# 基本面财报（时间序列，供 PIT 对齐）
# ══════════════════════════════════════════════════════════════

def _to_float(v) -> "float | None":
    """财报值是字符串（"50613911.01"），转 float；None/空/'-' → None。"""
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s in ("", "-", "None", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_announce_date(raw: str) -> "date | None":
    """InfoPublDate 形如 '2025-04-22 00:00:00 +0800 CST' → date。"""
    if not raw:
        return None
    head = str(raw).strip()[:10]  # 2025-04-22
    try:
        return date.fromisoformat(head)
    except ValueError:
        return None


def westock_financials(codes: list[str], num: int = 12) -> dict[str, list[dict]]:
    """拉财报并整理成按披露日升序的时间序列，供 PIT 对齐。

    finance --raw 结构：{"sections": [income[], balance[], cashflow[]]}，值为字符串。
    只有 income 段有 InfoPublDate；balance 借用 income 同报告期(EndDate)的披露日。

    Returns:
        {code: [{"announce_date": date, "end_date": str,
                 "roe_ttm": float, "revenue_yoy": float}, ...]}（按 announce_date 升序）
        缺失项不写入该期（PIT 取数时缺失因子按中性处理）。
    """
    result: dict[str, list[dict]] = {}
    total = len(codes)
    for idx, code in enumerate(codes):
        series = _fetch_one_financials(code, num)
        if series:
            result[code] = series
        if (idx + 1) % 25 == 0 or idx + 1 == total:
            logger.info("  westock 财报进度: %d/%d 只", idx + 1, total)
    return result


def _fetch_one_financials(code: str, num: int) -> list[dict]:
    """拉单只财报时间序列，带磁盘缓存（季度更新，TTL 30 天）。"""
    cache_key = f"fin_{code}_{num}"
    cached = _cache_get(cache_key, ttl_days=30)
    if cached is not None:
        # 缓存里 announce_date 存的是 isoformat 字符串，转回 date
        out = []
        for r in cached:
            ad = date.fromisoformat(r["announce_date"]) if r.get("announce_date") else None
            if ad is None:
                continue
            out.append({**r, "announce_date": ad})
        return out

    try:
        data = _run_westock_json(["finance", to_westock_code(code), "--num", str(num)], timeout=60.0)
    except Exception as ex:
        logger.warning("westock finance %s 失败: %s", code, ex)
        return []
    # finance 不走 {success,data} 包装，是 {sections:[...]}；_run_westock_json 下钻 data
    # 时若无 data 键会原样返回，故这里兼容两种
    sections = None
    if isinstance(data, dict):
        sections = data.get("sections")
    if not sections or len(sections) < 2:
        return []

    income_rows = sections[0] or []
    balance_rows = sections[1] or []

    # 按 EndDate 索引（过滤脏行）
    inc_by_end: dict[str, dict] = {}
    for r in income_rows:
        if not isinstance(r, dict):
            continue
        ed = r.get("EndDate")
        if ed:
            inc_by_end[str(ed)] = r
    bal_by_end: dict[str, dict] = {}
    for r in balance_rows:
        if not isinstance(r, dict):
            continue
        ed = r.get("EndDate")
        se = _to_float(r.get("SEWithoutMI"))
        if ed and se is not None:  # 过滤净资产为 None 的脏行
            bal_by_end[str(ed)] = r

    series: list[dict] = []
    for end_date, inc in inc_by_end.items():
        ad = _parse_announce_date(inc.get("InfoPublDate", ""))
        if ad is None:
            continue
        rec: dict = {"announce_date": ad, "end_date": end_date}

        np_ttm = _to_float(inc.get("NPParentCompanyOwnersTTM"))
        rev_ttm = _to_float(inc.get("OperatingRevenueTTM"))
        gross_ttm = _to_float(inc.get("GrossProfitTTM"))
        bal = bal_by_end.get(end_date)
        se = _to_float(bal.get("SEWithoutMI")) if bal else None

        # ── 原始量保留（B 类 PB/PE 计算需 PIT 净利/净资产）──
        if np_ttm is not None:
            rec["_np_ttm"] = np_ttm
        if se and se > 0:
            rec["_equity"] = se

        # roe_ttm = 归母净利TTM / 期末净资产
        if np_ttm is not None and se and se > 0:
            rec["roe_ttm"] = np_ttm / se
        # net_margin_ttm = 归母净利TTM / 营收TTM
        if np_ttm is not None and rev_ttm and rev_ttm > 0:
            rec["net_margin_ttm"] = np_ttm / rev_ttm
        # gross_margin = 毛利TTM / 营收TTM
        if gross_ttm is not None and rev_ttm and rev_ttm > 0:
            rec["gross_margin"] = gross_ttm / rev_ttm

        # 同比（营收/净利）：本期 TTM / 去年同报告期 TTM − 1
        prev_end = _prev_year_end_date(end_date)
        prev_inc = inc_by_end.get(prev_end) if prev_end else None
        if prev_inc:
            prev_rev = _to_float(prev_inc.get("OperatingRevenueTTM"))
            if rev_ttm is not None and prev_rev and prev_rev > 0:
                rec["revenue_yoy"] = rev_ttm / prev_rev - 1.0
            prev_np = _to_float(prev_inc.get("NPParentCompanyOwnersTTM"))
            if np_ttm is not None and prev_np and prev_np > 0:
                rec["net_profit_yoy"] = np_ttm / prev_np - 1.0

        series.append(rec)

    series.sort(key=lambda r: r["announce_date"])
    # 缓存（announce_date 转字符串）
    _cache_put(cache_key, [{**r, "announce_date": r["announce_date"].isoformat()} for r in series])
    return series


def _prev_year_end_date(end_date: str) -> "str | None":
    """报告期 EndDate (YYYY-MM-DD) → 去年同报告期。"""
    try:
        d = date.fromisoformat(str(end_date)[:10])
        return date(d.year - 1, d.month, d.day).isoformat()
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════
# 总股本（用于 PB/PE 估值因子；westock 无历史股本，取当前值）
# ══════════════════════════════════════════════════════════════

def westock_total_shares(codes: list[str], batch: int = 50) -> dict[str, float]:
    """批量取当前总股本（quote 的 total_shares，单位=股）。带缓存。

    ★ 取当前值——westock 不提供历史股本。用于 PB/PE 时有小瑕疵
      （增发/回购致历史股本不同），但远小于直接用今日 PB/PE。
    """
    result: dict[str, float] = {}
    todo: list[str] = []
    for c in codes:
        cached = _cache_get(f"shares_{c}", ttl_days=30)
        if cached is not None:
            result[c] = float(cached)
        else:
            todo.append(c)

    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        wcodes = ",".join(to_westock_code(c) for c in chunk)
        try:
            data = _run_westock_json(["quote", wcodes], timeout=60.0)
        except Exception as ex:
            logger.warning("westock quote 股本批次失败: %s", ex)
            continue
        rows = data if isinstance(data, list) else [data]
        for r in rows:
            if not isinstance(r, dict):
                continue
            sym = r.get("symbol") or r.get("code")
            ts = _to_float(r.get("total_shares"))
            if sym and ts and ts > 0:
                code6 = _strip_prefix(str(sym))
                result[code6] = ts
                _cache_put(f"shares_{code6}", ts)
    logger.info("westock 总股本加载: %d/%d 只", len(result), len(codes))
    return result


