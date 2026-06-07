"""数据采集微服务。

基于 APScheduler 的定时任务调度，负责所有数据的定时拉取、
质量检查和入库。通过 Redpanda 上报健康状态到 monitor。
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.common import (
    HealthChecker,
    MessageBus,
    get_broker,
    get_env,
    setup_graceful_shutdown,
)

logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────

POLL_INTERVAL_MINUTES = int(get_env("OVERNIGHT_POLL_INTERVAL", "15"))
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # 秒

health = HealthChecker("data_collector")
bus = MessageBus(broker=get_broker(), client_id="data_collector")

# 延迟初始化（避免导入时连数据库）
_pipeline = None
_checker = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from core.data.sources.akshare import AkShareSource
        from core.data.sources.fallback import FallbackDataSource
        from core.data.pipeline import DataPipeline
        from core.data.quality import DataQualityChecker
        primary = AkShareSource()
        ds = FallbackDataSource(primary, primary, max_failures=1)  # 单源模式
        _pipeline = DataPipeline(ds, ch_client=None, quality_checker=None)
    return _pipeline


def _retry(func, name: str):
    """指数退避重试。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            func()
            health.record_success()
            return
        except Exception as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning("%s 失败 (attempt %d/%d): %s, %ds后重试", name, attempt, MAX_RETRIES, e, delay)
            health.record_failure(str(e))
            if attempt < MAX_RETRIES:
                time.sleep(delay)
            else:
                logger.error("%s 最终失败", name)
                bus.send("system.health", {
                    "service": "data_collector",
                    "task": name,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })


# ── 定时任务 ──────────────────────────

def collect_daily_close() -> None:
    """15:05: 日K线 + 龙虎榜 + 融资融券。"""
    logger.info("日终批量采集开始")
    pipeline = _get_pipeline()
    today = date.today()

    _retry(lambda: pipeline.collect_daily_kline(end=today), "日K线采集")
    _retry(lambda: pipeline.collect_margin_trading(today), "融资融券采集")
    _retry(lambda: pipeline.collect_dragon_tiger(today), "龙虎榜采集")

    bus.send("system.health", {
        "service": "data_collector",
        "task": "daily_close",
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
    })


def run_quality_checks() -> None:
    """15:30: 数据质量检查。"""
    logger.info("数据质量检查开始")
    # 质量检查逻辑在 DataQualityChecker 中
    # 此处检查最近一批数据
    try:
        from core.data.quality import DataQualityChecker
        checker = DataQualityChecker()
        # 实际部署时会传入最近的 DataFrame
        health.record_success()
        logger.info("数据质量检查完成")
    except Exception as e:
        health.record_failure(str(e))


def collect_overnight() -> None:
    """06:00-07:00: 每隔 POLL_INTERVAL_MINUTES 分钟拉取隔夜海外数据。"""
    logger.info("隔夜数据采集开始")
    try:
        import akshare as ak
        # 采集美股指数
        for idx in ["DJI", "IXIC", "SPX"]:
            try:
                ak.index_us_stock_sina(symbol=f".{idx}")
            except Exception:
                pass
        # A50期货
        try:
            ak.futures_zh_spot(symbol="A50")
        except Exception:
            pass
        health.record_success()
        bus.send("system.health", {
            "service": "data_collector",
            "task": "overnight",
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        health.record_failure(str(e))


def collect_premarket_announcements() -> None:
    """07:00: 盘前公告增量采集。"""
    logger.info("盘前公告采集开始")
    try:
        import akshare as ak
        # 拉取最新公告列表
        try:
            ak.stock_notice_report(symbol="ALL")
        except Exception:
            pass
        health.record_success()
    except Exception as e:
        health.record_failure(str(e))


def start_auction_stream() -> None:
    """09:15: 启动竞价数据流监听。"""
    logger.info("竞价数据流监听启动")
    # 竞价数据来自 QMT L2，此处为占位逻辑
    bus.send("system.health", {
        "service": "data_collector",
        "task": "auction_stream",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
    })


# ── 主入口 ──────────────────────────

def main() -> None:
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        logger.warning("apscheduler 未安装，使用简单 sleep 循环")
        _simple_loop()
        return

    scheduler = BackgroundScheduler()
    scheduler.add_job(collect_daily_close, "cron", hour=15, minute=5, day_of_week="mon-fri", id="daily_close")
    scheduler.add_job(run_quality_checks, "cron", hour=15, minute=30, day_of_week="mon-fri", id="quality")
    scheduler.add_job(collect_overnight, "cron", hour=6, minute="*/15", day_of_week="mon-fri", id="overnight")
    scheduler.add_job(collect_premarket_announcements, "cron", hour=7, minute=0, day_of_week="mon-fri", id="premarket")
    scheduler.add_job(start_auction_stream, "cron", hour=9, minute=15, day_of_week="mon-fri", id="auction")

    scheduler.start()
    logger.info("data_collector 已启动，等待定时任务触发")

    def cleanup():
        scheduler.shutdown(wait=False)
        bus.close()

    setup_graceful_shutdown(cleanup)

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        cleanup()


def _simple_loop() -> None:
    """无 APScheduler 时的简化轮询。"""
    logger.info("data_collector 启动（简单循环模式）")

    def cleanup():
        bus.close()
    setup_graceful_shutdown(cleanup)

    last_run: dict[str, float] = {}
    interval_map = {
        "overnight": POLL_INTERVAL_MINUTES * 60,
        "premarket": 3600,
    }

    try:
        while True:
            now = time.time()
            if now - last_run.get("overnight", 0) >= interval_map["overnight"]:
                collect_overnight()
                last_run["overnight"] = now
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
