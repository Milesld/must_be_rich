#!/usr/bin/env python3
"""本地数据仓每日更新（路线图第 6 阶段）。建议每日收盘后跑一次（可挂 cron）。

    # 按策略配置增量更新行情 + 存当日行业成分快照
    python scripts/update_data.py update --config configs/strategy_semiconductor_westock.yaml

    # 首次建仓提速：把 westock 分段缓存（~/.quant_system/westock_cache/kline_*.json）
    # 导入仓库，不联网
    python scripts/update_data.py import-cache

    # 仓库状态
    python scripts/update_data.py status

update 流程（每只股票）：
1. 仓内已有 → 从 (仓内最后日 − OVERLAP_DAYS) 拉到今天，重叠日比对 close：
   一致则增量合并；偏差超容差 = 除权后 qfq 基线变了 → 全量重拉该股。
   这从机制上根治了段缝启发式检不出的小额分红错位（见 warehouse.py 模块注释）。
2. 仓内没有 → 按 config 区间（start_date − lookback）全量拉。
3. 行业成分快照：dynamic 模式的每个板块存一份当日成分 JSON——
   攒一年即自建 PIT 成分库（幸存者偏差的长期解法）。
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("update_data")

from research import warehouse as wh


def _fetch_range(code: str, start: date, end: date):
    """联网拉单只区间（westock，qfq）。返回 DataFrame 或 None。"""
    from research.westock_source import westock_kline
    try:
        return westock_kline([code], start, end)
    except Exception as ex:
        logger.warning("拉取 %s 失败: %s", code, ex)
        return None


def cmd_update(config_path: str) -> None:
    from research.run_backtest_demo import load_config, _get_codes, _parse_date, _provider

    config = load_config(config_path)
    if _provider(config) != "westock":
        print(f"  ✗ 目前仅支持 westock provider（配置为 {_provider(config)}）")
        return

    codes = _get_codes(config)
    if not codes:
        print("  ✗ 候选代码为空")
        return

    lookback = config["data_source"].get("lookback_days", 400)
    full_start = _parse_date(config["backtest"]["start_date"]) - timedelta(days=lookback)
    today = date.today()

    meta = wh.load_meta()
    stats = {"appended": 0, "created": 0, "rebase_detected": 0,
             "up_to_date": 0, "failed": 0}

    print(f"  数据仓更新: {len(codes)} 只 → {wh.WAREHOUSE_DIR}")
    for i, code in enumerate(codes):
        cov = wh.coverage(code, meta)
        if cov is None:
            df = _fetch_range(code, full_start, today)
            if df is None or df.empty:
                stats["failed"] += 1
                continue
            wh.upsert_kline(code, df, meta, fetch_start=full_start)
            stats["created"] += 1
        else:
            _first, last = cov
            if last >= today:
                stats["up_to_date"] += 1
                continue
            pull_start = last - timedelta(days=wh.OVERLAP_DAYS)
            df = _fetch_range(code, pull_start, today)
            if df is None or df.empty:
                stats["failed"] += 1
                continue
            action = wh.upsert_kline(code, df, meta, fetch_start=None)
            if action == "rebase_detected":
                # 基线已变：仓内旧数据已废弃，全量重拉恢复完整历史。
                # 必须先清掉该股的分段磁盘缓存——否则全量重拉会命中 TTL=3650 天的
                # 旧基线缓存段，只有含今日的那段是新基线，仓里留下混合复权序列。
                from research.westock_source import _purge_kline_cache
                purged = _purge_kline_cache(code)
                if purged:
                    logger.info("%s 基线变化，清除 %d 个 kline 缓存段后重拉", code, purged)
                full_df = _fetch_range(code, full_start, today)
                if full_df is not None and not full_df.empty:
                    wh.upsert_kline(code, full_df, meta, fetch_start=full_start)
                stats["rebase_detected"] += 1
            else:
                stats[action] = stats.get(action, 0) + 1
        if (i + 1) % 25 == 0 or i + 1 == len(codes):
            print(f"    进度 {i + 1}/{len(codes)}")

    wh.save_meta_bulk(meta)
    print(f"  行情: 新建 {stats['created']} | 增量 {stats['appended']} | "
          f"复权重拉 {stats['rebase_detected']} | 已最新 {stats['up_to_date']} | "
          f"失败 {stats['failed']}")

    # ── 行业成分当日快照（dynamic 模式）──
    uni = config.get("data_source", {}).get("universe", {})
    boards = uni.get("industries", []) if isinstance(uni, dict) else []
    for board in boards:
        try:
            from research.westock_source import westock_industry_cons
            cons = westock_industry_cons([board])
            if cons:
                p = wh.snapshot_constituents(board, cons)
                print(f"  成分快照: {board} {len(cons)} 只 → {p.name}")
        except Exception as ex:
            logger.warning("成分快照 %s 失败: %s", board, ex)


def cmd_import_cache() -> None:
    """把 westock 分段缓存导入仓库（首次建仓提速，不联网）。

    同一只股票的多个段文件合并入仓；upsert 内部的重叠比对天然校验
    段间基线一致性（不一致时保留最新拉取的段，告警提示重拉）。
    """
    import json as _json
    cache_dir = Path.home() / ".quant_system" / "westock_cache"
    if not cache_dir.exists():
        print(f"  ✗ 缓存目录不存在: {cache_dir}")
        return

    import pandas as pd

    # kline_{code}_{start}_{end}.json，按 code 归组、按段起始日排序
    seg_files: dict[str, list[Path]] = {}
    for p in cache_dir.glob("kline_*.json"):
        parts = p.stem.split("_")
        if len(parts) == 4:
            seg_files.setdefault(parts[1], []).append(p)

    if not seg_files:
        print("  缓存中无 kline 段文件")
        return

    print(f"  导入 {len(seg_files)} 只股票的缓存段 → {wh.WAREHOUSE_DIR}")
    meta = wh.load_meta()
    n_ok = n_empty = 0
    for i, (code, files) in enumerate(sorted(seg_files.items())):
        rows: list[dict] = []
        seg_starts: list[str] = []
        for p in sorted(files, key=lambda x: x.stem.split("_")[2]):
            try:
                obj = _json.loads(p.read_text())
            except Exception:
                continue
            seg_starts.append(p.stem.split("_")[2])
            rows.extend(obj.get("data") or [])
        if not rows:
            n_empty += 1
            continue
        df = pd.DataFrame(rows)
        df["trade_date"] = df["date"].map(date.fromisoformat)
        fetch_start = date.fromisoformat(min(seg_starts)) if seg_starts else None
        wh.upsert_kline(code, df, meta, fetch_start=fetch_start)
        n_ok += 1
        if (i + 1) % 50 == 0 or i + 1 == len(seg_files):
            print(f"    进度 {i + 1}/{len(seg_files)}")
    wh.save_meta_bulk(meta)
    print(f"  完成: 入仓 {n_ok} 只 | 空缓存 {n_empty} 只")


def cmd_status() -> None:
    meta = wh.load_meta()
    print(f"\n  数据仓: {wh.WAREHOUSE_DIR}")
    if not meta:
        print("  （空仓——先跑 import-cache 或 update）")
        return
    lasts = sorted(m["last"] for m in meta.values())
    firsts = sorted(m["first"] for m in meta.values())
    n_rows = sum(m.get("n_rows", 0) for m in meta.values())
    print(f"  行情: {len(meta)} 只 | 共 {n_rows:,} 行 | "
          f"区间 {firsts[0]} ~ {lasts[-1]}")
    print(f"  最旧尾部: {lasts[0]}（{sum(1 for m in meta.values() if m['last'] == lasts[0])} 只）"
          f" | 最新尾部: {lasts[-1]}（{sum(1 for m in meta.values() if m['last'] == lasts[-1])} 只）")
    boards = wh.constituent_boards()
    if boards:
        for b, n in boards.items():
            print(f"  成分快照: {b} × {n} 天")
    else:
        print("  成分快照: 无（update 时自动存）")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="本地数据仓每日更新")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("update", help="按配置增量更新行情 + 当日成分快照")
    p_up.add_argument("--config", required=True)

    sub.add_parser("import-cache", help="把 westock 分段缓存导入仓库（不联网）")
    sub.add_parser("status", help="仓库覆盖状态")

    args = parser.parse_args()
    if args.cmd == "update":
        cmd_update(args.config)
    elif args.cmd == "import-cache":
        cmd_import_cache()
    elif args.cmd == "status":
        cmd_status()


if __name__ == "__main__":
    main()
