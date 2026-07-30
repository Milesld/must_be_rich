#!/usr/bin/env python3
"""本地行情数据仓（路线图第 6 阶段）：回测只读本地，联网只为增量更新。

存储布局（data/warehouse/，已被 .gitignore 的 data/ 规则覆盖）：
    kline/{code}.parquet          单只全历史日线（trade_date 升序，唯一）
    kline_meta.json               {code: {first,last,updated_at,n_rows}}（加速覆盖判定）
    constituents/{board}/{YYYY-MM-DD}.json   行业成分每日快照（攒 PIT 成分库）

★ qfq 复权基线错位的根治（替代 westock_source._seams_ok 的段缝启发式）：
qfq 价格以「拉取当天」为基准回算，任何除权除息（含段缝检查测不出的小额分红）
都会让**全历史**价格整体平移。因此增量更新时总是多拉最近 OVERLAP_DAYS 个
交易日，与仓内已有行重叠比对：任一重叠日 close 偏差超 RebaseTolerance
→ 判定复权基线已变 → 废弃该股全历史重拉。upsert_kline 返回的 action
告知调用方（"rebase_detected"时调用方应全量重拉后再 upsert）。

成分快照：每天存一份行业成分（慢变数据，重复日覆盖），constituents_asof
取 ≤ 指定日的最近快照——攒一年即得自建 PIT 成分库，把幸存者偏差从
「文档里诚实标注」变成「真正解决」（路线图 2.3）。
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_DIR = Path(__file__).resolve().parent.parent
WAREHOUSE_DIR = _PROJECT_DIR / "data" / "warehouse"

KLINE_COLUMNS = ["trade_date", "open", "high", "low", "close",
                 "volume", "amount", "turnover"]
# 增量更新时与仓内重叠比对的窗口（自然日；覆盖 ≥10 个交易日）
OVERLAP_DAYS = 15
# 重叠日 close 相对偏差超过此值 → 判定 qfq 基线已变（0.5%，远小于任何除权幅度，
# 又大于浮点/数据源舍入噪声）
REBASE_TOLERANCE = 0.005


def _kline_dir() -> Path:
    return WAREHOUSE_DIR / "kline"


def _meta_path() -> Path:
    return WAREHOUSE_DIR / "kline_meta.json"


def _cons_dir(board: str) -> Path:
    return WAREHOUSE_DIR / "constituents" / board


def _kline_path(code: str) -> Path:
    return _kline_dir() / f"{code}.parquet"


# ══════════════════════════════════════════════════════════════
# 元数据（覆盖范围速查，避免为判断覆盖而读几百个 parquet）
# ══════════════════════════════════════════════════════════════

def load_meta() -> dict:
    p = _meta_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        logger.warning("warehouse 元数据损坏，将从 parquet 重建")
        return rebuild_meta()


def _save_meta(meta: dict) -> None:
    p = _meta_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(p, lambda tmp: tmp.write_text(
        json.dumps(meta, ensure_ascii=False, indent=1)))


def _atomic_write(target: Path, writer) -> None:
    """tmp + rename 原子落盘：进程中途挂掉不会留下半截文件。"""
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        writer(tmp)
        tmp.replace(target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def rebuild_meta() -> dict:
    """扫描全部 parquet 重建元数据（元数据丢失/损坏时的自愈路径）。"""
    meta: dict = {}
    if not _kline_dir().exists():
        return meta
    for p in sorted(_kline_dir().glob("*.parquet")):
        try:
            df = pd.read_parquet(p, columns=["trade_date"])
        except Exception:
            logger.warning("warehouse 损坏的 parquet: %s（跳过）", p.name)
            continue
        if df.empty:
            continue
        meta[p.stem] = {
            "first": str(df["trade_date"].min()),
            "last": str(df["trade_date"].max()),
            "n_rows": int(len(df)),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    _save_meta(meta)
    return meta


def coverage(code: str, meta: dict | None = None) -> tuple[date, date] | None:
    """单只覆盖区间 (first, last)；无数据返回 None。"""
    m = (meta if meta is not None else load_meta()).get(code)
    if not m:
        return None
    return date.fromisoformat(m["first"]), date.fromisoformat(m["last"])


def warehouse_covers(code: str, start: date, end: date,
                     meta: dict | None = None) -> bool:
    """判断 [start, end] 能否完全由仓内数据服务（回测免联网的闸门）。

    头部条件用 fetch_start（该股历史上全量拉取时请求的最早起点）而非
    首行日期：次新股 first > start 是因为上市晚，不是缺数据——只要
    fetch_start ≤ start 就说明「更早的数据拉过了，确实不存在」。
    """
    m = (meta if meta is not None else load_meta()).get(code)
    if not m:
        return False
    if date.fromisoformat(m["last"]) < end:
        return False
    head = m.get("fetch_start") or m["first"]
    return date.fromisoformat(head) <= start


# ══════════════════════════════════════════════════════════════
# K 线读写
# ══════════════════════════════════════════════════════════════

def read_kline(code: str, start: date | None = None,
               end: date | None = None) -> pd.DataFrame | None:
    """读单只日线（含 code 列），可选日期过滤。无文件返回 None。"""
    p = _kline_path(code)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        # 截断/损坏的 parquet 不该让整个回测崩掉：当作无数据，交由上游重拉
        logger.warning("warehouse %s parquet 损坏（%s），当作无数据", code, e)
        return None
    if start is not None:
        df = df[df["trade_date"] >= start]
    if end is not None:
        df = df[df["trade_date"] <= end]
    if df.empty:
        return None
    df = df.copy()
    df["code"] = code
    return df[["code", *KLINE_COLUMNS]]


def read_kline_many(codes: list[str], start: date | None = None,
                    end: date | None = None) -> pd.DataFrame | None:
    """批量读取拼接（列结构与 westock_kline 返回一致）。"""
    frames = [df for c in codes if (df := read_kline(c, start, end)) is not None]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """标准化入仓行：列裁剪、类型统一、按日去重排序。"""
    out = df.copy()
    if "trade_date" not in out.columns and "date" in out.columns:
        out["trade_date"] = out["date"]
    out["trade_date"] = out["trade_date"].map(
        lambda x: x if isinstance(x, date) and not isinstance(x, datetime)
        else (x.date() if hasattr(x, "date") else date.fromisoformat(str(x)[:10])))
    for col in KLINE_COLUMNS[1:]:
        out[col] = pd.to_numeric(out.get(col, 0), errors="coerce").fillna(0.0).astype(float)
    out = out[KLINE_COLUMNS]
    out = out[out["close"] > 0]
    return out.drop_duplicates(subset=["trade_date"], keep="last") \
              .sort_values("trade_date").reset_index(drop=True)


def upsert_kline(code: str, new_df: pd.DataFrame,
                 meta: dict | None = None,
                 fetch_start: date | None = None) -> str:
    """新数据入仓（按 trade_date 合并），带 qfq 基线错位检测。

    Args:
        code: 6 位代码。
        new_df: 新拉的行（需含 trade_date + OHLCV 列；建议与仓内至少
                重叠 1 个交易日以便基线比对）。
        meta: 传入则就地更新（批量场景由调用方最后统一 _save_meta）；
              不传则本函数自行加载并保存。
        fetch_start: 本次拉取请求的起始日（全量拉取时传入）。记入元数据
                后，warehouse_covers 可区分「次新股上市晚」与「头部缺数据」。

    Returns:
        action：
        - "created"          新建（仓内原无此股）
        - "appended"         正常合并（重叠日价格一致）
        - "rebase_detected"  重叠日 close 偏差超容差 → 已废弃仓内旧数据，
                             只保留 new_df。调用方应随后全量重拉该股再
                             upsert（本次 new_df 通常只是尾部窗口）。
        - "noop"             新数据为空
    """
    own_meta = meta is None
    if own_meta:
        meta = load_meta()

    new_rows = _normalize(new_df)
    if new_rows.empty:
        return "noop"

    old = None
    p = _kline_path(code)
    if p.exists():
        old = pd.read_parquet(p)

    action = "created" if old is None or old.empty else "appended"
    if old is not None and not old.empty:
        old_by_date = old.set_index("trade_date")["close"]
        overlap = [d for d in new_rows["trade_date"] if d in old_by_date.index]
        for d in overlap:
            new_close = float(new_rows.loc[new_rows["trade_date"] == d, "close"].iloc[0])
            old_close = float(old_by_date[d])
            if old_close > 0 and abs(new_close / old_close - 1.0) > REBASE_TOLERANCE:
                logger.warning(
                    "warehouse %s 重叠日 %s close %.4f→%.4f（偏差 %.2f%%）："
                    "qfq 复权基线已变（除权/分红），废弃仓内旧数据",
                    code, d, old_close, new_close,
                    abs(new_close / old_close - 1.0) * 100)
                old = None
                action = "rebase_detected"
                break

    merged = new_rows if old is None or old.empty else _normalize(
        pd.concat([old, new_rows], ignore_index=True))

    _kline_dir().mkdir(parents=True, exist_ok=True)
    _atomic_write(p, lambda tmp: merged.to_parquet(tmp, index=False))
    prev_fetch_start = meta.get(code, {}).get("fetch_start")
    if action == "rebase_detected":
        prev_fetch_start = None  # 旧数据已废弃，头部覆盖需重新建立
    starts = [s for s in (prev_fetch_start,
                          fetch_start.isoformat() if fetch_start else None) if s]
    meta[code] = {
        "first": str(merged["trade_date"].min()),
        "last": str(merged["trade_date"].max()),
        "n_rows": int(len(merged)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **({"fetch_start": min(starts)} if starts else {}),
    }
    if own_meta:
        _save_meta(meta)
    return action


def save_meta_bulk(meta: dict) -> None:
    """批量 upsert 后统一落盘元数据。"""
    _save_meta(meta)


# ══════════════════════════════════════════════════════════════
# 行业成分每日快照（PIT 成分库）
# ══════════════════════════════════════════════════════════════

def snapshot_constituents(board: str, codes: list[str],
                          snap_date: date | None = None) -> Path:
    """存一份行业成分快照（同日重跑覆盖）。"""
    d = snap_date or date.today()
    dir_ = _cons_dir(board)
    dir_.mkdir(parents=True, exist_ok=True)
    p = dir_ / f"{d.isoformat()}.json"
    _atomic_write(p, lambda tmp: tmp.write_text(json.dumps(
        {"board": board, "date": d.isoformat(), "codes": list(codes)},
        ensure_ascii=False)))
    return p


def constituents_asof(board: str, as_of: date) -> list[str] | None:
    """取 ≤ as_of 的最近成分快照；无任何快照返回 None（调用方回退当前成分）。"""
    dir_ = _cons_dir(board)
    if not dir_.exists():
        return None
    dates = sorted(
        p.stem for p in dir_.glob("*.json")
        if len(p.stem) == 10 and p.stem <= as_of.isoformat())
    if not dates:
        return None
    obj = json.loads((dir_ / f"{dates[-1]}.json").read_text())
    return list(obj.get("codes", []))


def constituent_boards() -> dict[str, int]:
    """已有快照的板块及各自快照数（status 展示用）。"""
    base = WAREHOUSE_DIR / "constituents"
    if not base.exists():
        return {}
    return {d.name: len(list(d.glob("*.json")))
            for d in sorted(base.iterdir()) if d.is_dir()}
