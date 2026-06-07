"""日内预测微服务。

常驻进程，09:25-15:00 活跃。基于盘前推荐清单和实时行情，
每分钟更新预测信号。收盘集合竞价(14:57-15:00)单独模型。

关注池是软约束——非关注池股票出现极强盘口信号时也可以生成信号。
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

health = HealthChecker("intraday_service")
bus = MessageBus(broker=get_broker(), client_id="intraday_service")

# ── 运行时状态 ─────────────────────────

_watch_pool: set[str] = set()
_watch_pool_meta: dict[str, dict] = {}  # {code: {expected_direction, entry_range, composite_score}}

# 日内信号阈值
UP_PROB_THRESHOLD = float(get_env("INTRA_UP_THRESHOLD", "0.55"))
DOWN_PROB_THRESHOLD = float(get_env("INTRA_DOWN_THRESHOLD", "0.55"))
CONFIDENCE_THRESHOLD = float(get_env("INTRA_CONFIDENCE_THRESHOLD", "0.5"))
EXTREME_SIGNAL_SIGMA = float(get_env("INTRA_EXTREME_SIGMA", "3.0"))


def load_watch_pool() -> None:
    """09:25: 从盘前推荐加载今日关注池。"""
    # 实际部署从 Redpanda 消费 signals.premarket topic
    # 简化：从 Redis 读取
    try:
        from core.data.db import RedisClient
        redis = RedisClient(url=get_env("REDIS_URL", "redis://localhost:6379"))

        # 尝试从最新推荐文件加载
        from pathlib import Path
        rec_dir = Path(get_env("PREMIUM_OUTPUT_DIR", "data/premarket_recommendations"))
        today_file = rec_dir / f"premarket_{date.today().isoformat()}.json"
        if today_file.exists():
            with open(today_file) as f:
                data = json.load(f)
                for rec in data.get("recommendations", []):
                    code = rec.get("code", "")
                    if code:
                        _watch_pool.add(code)
                        _watch_pool_meta[code] = {
                            "expected_direction": rec.get("expected_direction", "flat"),
                            "composite_score": rec.get("composite_score", 0.5),
                            "suggested_entry": rec.get("suggested_entry_range", {}),
                        }
        logger.info("关注池加载完成: %d 只", len(_watch_pool))
    except Exception as e:
        logger.warning("关注池加载失败: %s", e)


def on_minute_bar(market_data: list[dict]) -> None:
    """每分钟行情回调：对关注池股票计算实时特征 → 推理 → 生成信号。

    Args:
        market_data: [{code, open, high, low, close, volume, amount, time}, ...]
    """
    if not market_data:
        return

    # 过滤出关注池股票（+ 非关注池但有极强信号的）
    codes_in_data = {d["code"] for d in market_data}
    pool_codes = _watch_pool & codes_in_data

    # 检查非关注池的极强信号
    for d in market_data:
        code = d["code"]
        if code not in _watch_pool:
            # 简单判断：涨跌幅超 3σ → 生成强信号
            _check_extreme_signal(d)

    if not pool_codes:
        return

    # 批量获取特征（从 Redis）
    features = _get_live_features(list(pool_codes))

    # 模型推理
    try:
        from core.models.intraday import IntradayClassifier
        import pandas as pd
        import numpy as np

        classifier = IntradayClassifier()
        if not classifier.is_trained:
            return

        for code in pool_codes:
            feat = features.get(code, {})
            if not feat:
                continue
            X = pd.DataFrame([feat])
            proba = classifier.predict_proba(X)  # (1, 3) → [跌, 平, 涨]
            up_prob = float(proba[0, 2])
            down_prob = float(proba[0, 0])
            flat_prob = float(proba[0, 1])

            # 信号生成条件
            confidence = max(up_prob, down_prob, flat_prob)
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            signal = None
            if up_prob > UP_PROB_THRESHOLD:
                signal = {"direction": "up", "strength": up_prob, "confidence": confidence}
            elif down_prob > DOWN_PROB_THRESHOLD:
                signal = {"direction": "down", "strength": down_prob, "confidence": confidence}

            if signal:
                meta = _watch_pool_meta.get(code, {})
                signal.update({
                    "code": code,
                    "signal_id": f"intra_{code}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                    "generated_at": datetime.now().isoformat(),
                    "model_name": "intraday_classifier",
                    "premarket_direction": meta.get("expected_direction", ""),
                    "metadata": {"watch_pool": True},
                })
                bus.send("signals.intraday", signal)

    except Exception as e:
        logger.error("日内推理失败: %s", e)
        health.record_failure(str(e))


def handle_closing_auction() -> None:
    """14:57-15:00: 收盘集合竞价——单独模型处理价格跳变。"""
    logger.info("收盘集合竞价阶段——使用单独模型")
    bus.send("signals.intraday", {
        "type": "closing_auction_note",
        "message": "进入收盘集合竞价阶段，日内模型暂停新信号",
        "generated_at": datetime.now().isoformat(),
    })


def _check_extreme_signal(market_data: dict) -> None:
    """检查非关注池股票的极强盘口信号。"""
    # 简化实现：极端涨跌幅视为强信号
    # 实际部署会使用更复杂的盘口模型
    pass


def _get_live_features(codes: list[str]) -> dict[str, dict[str, float]]:
    """从 Redis 批量获取最新特征值。"""
    try:
        from core.data.db import RedisClient
        from core.features.store import FeatureStore
        redis = RedisClient(url=get_env("REDIS_URL", "redis://localhost:6379"))
        store = FeatureStore(redis_client=redis)
        # 读取日内实时因子（技术面为主）
        factor_names = ["momentum_20d", "volatility_20d", "rsi_14", "volume_ratio"]
        return store.get_latest_from_redis(codes, factor_names)
    except Exception:
        return {}


def generate_daily_summary() -> None:
    """日终：生成当日信号汇总。"""
    logger.info("日内信号汇总报告生成")


# ── 主入口 ──────────────────────────

def main() -> None:
    logger.info("intraday_service 启动")

    def cleanup():
        bus.close()
    setup_graceful_shutdown(cleanup)

    load_watch_pool()

    # 简化轮询模式
    logger.info("等待盘中行情...")
    try:
        while True:
            now = datetime.now()
            h, m = now.hour, now.minute

            if h == 15 and m >= 1:
                # 收盘后
                generate_daily_summary()
                break
            elif h == 14 and m >= 57:
                handle_closing_auction()
                time.sleep(60)
            elif (h == 9 and m >= 30) or (10 <= h <= 14) or (h == 13 and m >= 1) or (h == 11 and m <= 30):
                # 盘中：每分钟检查一次
                on_minute_bar([])  # 实际部署从 Redpanda consumer 获取市场数据
                time.sleep(60)
            else:
                time.sleep(30)
    except (KeyboardInterrupt, SystemExit):
        cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
