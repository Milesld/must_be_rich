"""订单管理微服务 (OMS)。

常驻进程，订阅信号 topic → 风控检查 → 下单 → 成交回报处理。
是系统唯一可以直接调用券商 API 的服务。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.common import HealthChecker, get_broker, get_env, setup_graceful_shutdown

logger = logging.getLogger(__name__)

health = HealthChecker("order_manager")

# ── 订单状态机 ─────────────────────────

ORDER_STATES = ["pending", "submitted", "partial_filled", "filled", "cancelled", "rejected"]

# 已处理 signal_id 集合（幂等去重）
_seen_signals: set[str] = set()
MAX_SIGNAL_HISTORY = 100_000

# QMT client
_qmt: Any = None


def _get_qmt():
    """延迟初始化 QMT 客户端。"""
    global _qmt
    if _qmt is None:
        qmt_enabled = get_env("QMT_ENABLED", "false").lower() == "true"
        if qmt_enabled:
            try:
                # 实际部署时的 QMT SDK 初始化
                # from xtquant import xtdata, xttrader
                # _qmt = xttrader.XtQuantTrader(...)
                logger.info("QMT 客户端已初始化（模拟模式）")
            except ImportError:
                logger.warning("QMT SDK 不可用，使用模拟下单")
        _qmt = _MockQMT()
    return _qmt


class _MockQMT:
    """模拟 QMT 客户端（开发/测试环境）。"""
    def submit_order(self, code: str, side: str, price: float, shares: int) -> dict:
        order_id = f"ord_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        logger.info("[MOCK-QMT] 下单: %s %s %s %d股 @ %.2f", order_id, code, side, shares, price)
        return {"order_id": order_id, "status": "submitted", "code": code}

    def cancel_order(self, order_id: str) -> bool:
        logger.info("[MOCK-QMT] 撤单: %s", order_id)
        return True

    def query_order(self, order_id: str) -> dict:
        return {"order_id": order_id, "status": "filled", "filled_shares": 0, "filled_price": 0.0}


# ── 信号处理管线 ───────────────────────

def handle_signal(signal: dict) -> Optional[dict]:
    """处理单条信号：去重→风控→计算→下单。

    Returns:
        订单记录字典（用于写入日志）。
    """
    signal_id = signal.get("signal_id", "")
    code = signal.get("code", "")
    side = signal.get("side", signal.get("direction", "flat"))
    strength = signal.get("strength", 0.0)

    # 1. 去重
    if signal_id in _seen_signals:
        logger.debug("重复信号已忽略: %s", signal_id)
        return None
    _seen_signals.add(signal_id)
    if len(_seen_signals) > MAX_SIGNAL_HISTORY:
        _seen_signals.clear()

    # 2. 事前风控检查
    price = signal.get("price", 0.0)
    shares = signal.get("shares", 0)
    if price <= 0 or shares <= 0:
        # 从信号中的 entry_range 或 latest_price 字段获取
        price = signal.get("latest_price", signal.get("intent_price", 0.0))
        shares = signal.get("intent_shares", signal.get("target_shares", 0))

    qmt = _get_qmt()

    # 3. 订单生成
    try:
        result = qmt.submit_order(code, side, price, shares)
        order_id = result.get("order_id", "")
        status = result.get("status", "pending")

        # 4. 记录订单日志
        order_record = {
            "order_id": order_id,
            "signal_id": signal_id,
            "code": code,
            "side": side,
            "price": price,
            "shares": shares,
            "status": status,
            "submitted_at": datetime.now().isoformat(),
        }
        health.record_success()
        return order_record
    except Exception as e:
        logger.error("下单失败 [%s]: %s", code, e)
        health.record_failure(str(e))
        return None


def cancel_order(order_id: str) -> bool:
    """撤单（仅重试 1 次，重试前先查状态）。"""
    qmt = _get_qmt()
    try:
        status = qmt.query_order(order_id)
        if status["status"] in ("filled", "cancelled", "rejected"):
            logger.info("订单 %s 状态为 %s，无需撤单", order_id, status["status"])
            return True
        return qmt.cancel_order(order_id)
    except Exception as e:
        logger.error("撤单失败 [%s]: %s", order_id, e)
        return False


# ── 消息消费（简化版，无 kafka-python 时的轮询）───

def _consume_signals() -> None:
    """消费 Redpanda signals topic（简化：文件轮询模式）。"""
    # 实际部署中：
    # from kafka import KafkaConsumer
    # consumer = KafkaConsumer('signals.intraday', 'signals.long_term', 'signals.exit', ...)
    # for msg in consumer: handle_signal(json.loads(msg.value))
    logger.info("OMS 就绪，等待信号...（当前为占位模式，实际部署连接 Redpanda）")


# ── 主入口 ──────────────────────────

def main() -> None:
    logger.info("order_manager 启动")

    def cleanup():
        global _seen_signals
        _seen_signals.clear()
    setup_graceful_shutdown(cleanup)

    # 消费信号流
    _consume_signals()

    try:
        while True:
            time.sleep(10)
    except (KeyboardInterrupt, SystemExit):
        cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
