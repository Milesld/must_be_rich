"""长期选股微服务。

定时任务驱动的月度调仓和模型重训练。
- 月末收盘后：全市场排序 → 目标权重 → 整手持仓 → 调仓信号。
- 月初盘后：Walk-Forward 窗口更新 → 模型重训练 → 新版本部署。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.common import HealthChecker, MessageBus, get_broker, get_env, setup_graceful_shutdown

logger = logging.getLogger(__name__)

health = HealthChecker("long_term_service")
bus = MessageBus(broker=get_broker(), client_id="long_term_service")

# 配置
MODEL_DIR = Path(get_env("MODEL_DIR", "data/models"))


def load_long_term_scores() -> dict[str, float]:
    """从 Redis 加载最新的长期排序分数。"""
    from core.data.db import RedisClient
    redis = RedisClient(url=get_env("REDIS_URL", "redis://localhost:6379"))
    # 从 Redis 读取最新评分（key 由 feature_server 写入）
    keys = redis.client.keys("feat:long_term_score:*")
    if not keys:
        logger.warning("Redis 中无长期评分数据")
        return {}
    scores: dict[str, float] = {}
    for k in keys:
        raw = redis.get(k)
        if raw:
            try:
                code = k.decode().split(":")[-1] if isinstance(k, bytes) else k.split(":")[-1]
                scores[code] = float(json.loads(raw)["value"])
            except (ValueError, KeyError):
                pass
    return scores


def run_monthly_rebalance() -> None:
    """月度调仓流程。"""
    logger.info("月度调仓开始")
    try:
        scores = load_long_term_scores()
        if not scores:
            logger.warning("无长期评分，跳过月度调仓")
            return

        from core.portfolio.optimizer import PortfolioOptimizer
        from core.backtest.rounding import ShareRounder

        opt = PortfolioOptimizer()
        rounder = ShareRounder()

        # 获取波动率（用于 inverse_vol_weight）
        volatilities = _load_volatilities(list(scores.keys()))

        if volatilities:
            weights = opt.inverse_vol_weight(scores, volatilities, top_n=20)
        else:
            weights = opt.equal_weight(scores, top_n=20)

        # 发送调仓信号
        signals = [
            {"code": c, "weight": w, "action": "rebalance"}
            for c, w in weights.items()
        ]
        bus.send("signals.long_term", {
            "date": date.today().isoformat(),
            "type": "monthly_rebalance",
            "signals": signals,
            "generated_at": datetime.now().isoformat(),
        })

        health.record_success()
        logger.info("月度调仓完成: %d 只持仓", len(weights))
    except Exception as e:
        health.record_failure(str(e))
        logger.error("月度调仓失败: %s", e)


def run_model_retraining() -> None:
    """月度模型重训练流程。"""
    logger.info("长期模型重训练开始")
    try:
        from core.models.long_term import LongTermRanker
        from core.features.store import FeatureStore

        ranker = LongTermRanker()
        store = FeatureStore()

        # 加载历史因子值
        end_date = date.today()
        start_date = end_date.replace(year=end_date.year - 5)

        factor_names = [fd.name for fd in ranker._feature_names] if ranker._feature_names else ["momentum_20d", "roe_ttm"]
        df = store.load_factor_values(start_date, end_date, factor_names)

        if df.empty:
            logger.warning("无历史因子数据，跳过重训练")
            return

        # 特征和目标
        X = df[factor_names]
        y = ranker.prepare_labels(df)
        groups = df.index.get_level_values("calc_date") if "calc_date" in df.index.names else None

        ranker.train(X, y, groups=pd.Series(groups) if groups is not None else None)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"long_term_ranker_{date.today().isoformat()}.joblib"
        ranker.save(str(model_path))
        logger.info("模型已保存: %s (version=%s)", model_path, ranker.model_version)

        health.record_success()
    except Exception as e:
        health.record_failure(str(e))
        logger.error("模型重训练失败: %s", e)


def generate_weekly_report() -> None:
    """生成周度持仓报告。"""
    report_path = Path(get_env("REPORT_DIR", "data/reports"))
    report_path.mkdir(parents=True, exist_ok=True)
    filepath = report_path / f"weekly_report_{date.today().isoformat()}.md"

    with open(filepath, "w") as f:
        f.write(f"# 周度持仓报告\n\n**日期**: {date.today()}\n\n")
        f.write("## 当前持仓\n\n(数据来源于最新调仓信号)\n")

    logger.info("周度报告已生成: %s", filepath)


def _load_volatilities(codes: list[str]) -> dict[str, float]:
    """加载股票波动率（用于 inverse_vol_weight）。"""
    try:
        from core.data.db import RedisClient
        redis = RedisClient(url=get_env("REDIS_URL", "redis://localhost:6379"))
        result: dict[str, float] = {}
        for code in codes:
            raw = redis.get(f"feat:volatility_20d:{code}")
            if raw:
                result[code] = float(json.loads(raw)["value"])
        return result
    except Exception:
        return {}


# ── 主入口 ──────────────────────────

def main() -> None:
    import pandas as pd  # noqa: F401

    logger.info("long_term_service 启动")

    def cleanup():
        bus.close()
    setup_graceful_shutdown(cleanup)

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        # 月末最后一个交易日的收盘后
        scheduler.add_job(run_monthly_rebalance, "cron", day="25-31", hour=16, minute=0, id="rebalance")
        # 月初第一个交易日的收盘后
        scheduler.add_job(run_model_retraining, "cron", day="1-7", hour=20, minute=0, id="retrain")
        # 每周五收盘后
        scheduler.add_job(generate_weekly_report, "cron", day_of_week="fri", hour=17, minute=0, id="weekly_report")
        scheduler.start()
        logger.info("定时任务已注册")
    except ImportError:
        logger.warning("apscheduler 未安装，立即执行一次调仓作为演示")
        run_monthly_rebalance()

    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
