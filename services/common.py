"""微服务公共工具。

提供所有服务共用的样板代码：健康检查、信号发送、优雅退出。
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ── 配置────────────────────────────────

def get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

def get_broker() -> str:
    return get_env("REDPANDA_BROKER", "localhost:9092")

def get_redis_url() -> str:
    return get_env("REDIS_URL", "redis://localhost:6379")

# ── 健康检查 ──────────────────────────

class HealthChecker:
    """服务健康状态追踪。"""

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        self.start_time = time.time()
        self._status = "healthy"
        self._last_error: Optional[str] = None
        self._stats: dict[str, Any] = {"successes": 0, "failures": 0, "last_success": None}

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time

    def record_success(self) -> None:
        self._stats["successes"] += 1
        self._stats["last_success"] = datetime.now().isoformat()

    def record_failure(self, error: str) -> None:
        self._stats["failures"] += 1
        self._last_error = error
        if self._stats["failures"] > 10:
            self._status = "degraded"

    def status(self) -> dict:
        return {
            "service": self.service_name,
            "status": self._status,
            "uptime": self.uptime,
            "last_error": self._last_error,
            "stats": self._stats,
        }


# ── 优雅退出 ──────────────────────────

def setup_graceful_shutdown(cleanup_fn: Callable[[], None] | None = None) -> None:
    """注册 SIGTERM/SIGINT 处理，优雅退出。"""

    def handler(signum: int, frame: Any) -> None:
        logger.info("收到信号 %s，正在优雅退出...", signum)
        if cleanup_fn:
            cleanup_fn()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# ── Redpanda 消息发送 ──────────────────

class MessageBus:
    """Redpanda/Kafka 消息总线客户端。"""

    def __init__(self, broker: str, client_id: str) -> None:
        self._broker = broker
        self._client_id = client_id
        self._producer: Any = None

    def _ensure_producer(self) -> Any:
        if self._producer is None:
            try:
                from kafka import KafkaProducer
                import json
                self._producer = KafkaProducer(
                    bootstrap_servers=self._broker,
                    client_id=self._client_id,
                    value_serializer=lambda v: json.dumps(v, ensure_ascii=False, default=str).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    retries=2,
                    max_block_ms=5000,
                )
                logger.info("Redpanda producer 已连接: %s", self._broker)
            except ImportError:
                logger.warning("kafka-python 未安装，消息总线降级为日志模式")
                self._producer = _LogProducer()
        return self._producer

    def send(self, topic: str, payload: dict, key: str = "") -> None:
        try:
            producer = self._ensure_producer()
            producer.send(topic, value=payload, key=key or None)
            if not isinstance(producer, _LogProducer):
                producer.flush(timeout=3)
        except Exception as e:
            logger.error("发送消息到 %s 失败: %s", topic, e)

    def close(self) -> None:
        if self._producer and not isinstance(self._producer, _LogProducer):
            self._producer.close()
            self._producer = None


class _LogProducer:
    """无 kafka-python 时的降级方案：将消息写入日志。"""
    def send(self, topic: str, value: Any, key: str | None = None) -> None:
        logger.info("[MOCK-KAFKA] topic=%s key=%s payload=%s", topic, key, str(value)[:200])
    def flush(self, timeout: int = 0) -> None:
        pass
    def close(self) -> None:
        pass
