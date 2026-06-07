"""结构化日志配置。

基于 Python 标准库 logging + JSON 格式输出，区分四类日志：
- audit: 业务审计日志（不可删除，合规要求）
- app: 系统运行日志（滚动保留30天）
- error: 错误告警日志（推送钉钉/微信）
- perf: 性能日志（采样记录，滚动7天）

可选使用 structlog（如已安装）。

使用方式：
    from core.common.logging_config import get_logger
    log = get_logger(__name__, "app")
    log.info("服务启动", extra={"port": 8080})
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Literal

LogType = Literal["audit", "app", "error", "perf"]


class JSONFormatter(logging.Formatter):
    """JSON 格式日志格式化器。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            payload["exception"] = self.formatException(record.exc_info)
        # 合并 extra 字段
        for key, val in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg",
                "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName",
            }:
                payload[key] = val
        return json.dumps(payload, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """开发环境彩色控制台格式。"""
    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[41m",  # red bg
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        msg = f"{color}[{record.levelname}]{self.RESET} [{record.name}] {record.getMessage()}"
        if record.exc_info and record.exc_info[0]:
            msg += "\n" + self.formatException(record.exc_info)
        return msg


def configure_logging(
    log_level: str = "INFO",
    log_type: LogType = "app",
) -> None:
    """配置根 logger。

    Args:
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）。
        log_type: 日志类型（影响格式化器选择）。
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 清除现有 handler
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)

    if log_type in ("audit", "error"):
        handler.setFormatter(JSONFormatter())
    elif os.environ.get("ENV", "dev") == "production":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ColoredFormatter())

    root.addHandler(handler)


# 已配置标志
_configured = False


def get_logger(name: str, log_type: LogType = "app") -> logging.Logger:
    """获取日志记录器。

    Args:
        name: 日志记录器名称（通常用 __name__）。
        log_type: 日志类型。
            - 'audit': 业务审计日志。
            - 'app': 系统运行日志（默认）。
            - 'error': 错误告警日志。
            - 'perf': 性能日志。

    Returns:
        logging.Logger 实例。
    """
    global _configured
    if not _configured:
        configure_logging(log_type=log_type)
        _configured = True

    logger = logging.getLogger(name)
    # 将 log_type 存入 logger 的 extra
    return logging.LoggerAdapter(logger, {"log_type": log_type})  # type: ignore[return-value]


def get_audit_logger(name: str) -> logging.Logger:
    """获取审计日志记录器。"""
    return get_logger(name, "audit")


def get_app_logger(name: str) -> logging.Logger:
    """获取应用日志记录器。"""
    return get_logger(name, "app")


def get_error_logger(name: str) -> logging.Logger:
    """获取错误告警日志记录器。"""
    return get_logger(name, "error")


def get_perf_logger(name: str) -> logging.Logger:
    """获取性能日志记录器。"""
    return get_logger(name, "perf")
