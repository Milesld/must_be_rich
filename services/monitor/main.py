"""监控告警微服务。

常驻进程，消费系统健康事件和风控告警，
集成钉钉/邮件/企业微信通知，暴露 Prometheus metrics 端点。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.common import HealthChecker, get_broker, get_env, setup_graceful_shutdown

logger = logging.getLogger(__name__)

health = HealthChecker("monitor")

# ── 告警通道配置 ────────────────────────

DINGTALK_WEBHOOK = get_env("DINGTALK_WEBHOOK", "")
WEWORK_WEBHOOK = get_env("WEWORK_WEBHOOK", "")
EMAIL_SMTP_HOST = get_env("EMAIL_SMTP_HOST", "")

# 服务心跳记录 {service_name: last_heartbeat_timestamp}
_heartbeats: dict[str, float] = {}

# Prometheus metrics 存储（简化版内存收集器）
_metrics: dict[str, Any] = {
    "order_count": 0,
    "signal_count": 0,
    "risk_rejection_count": 0,
    "trades_today": [],
}


# ── 告警发送 ──────────────────────────

def send_alert(alert: dict) -> None:
    """根据 severity 分发告警到对应通道。

    CRITICAL → 钉钉 + 电话(预留)
    ERROR    → 钉钉 + 邮件
    WARN     → 企业微信
    """
    severity = alert.get("severity", "WARN")
    message = alert.get("message", "")
    alert_type = alert.get("alert_type", "unknown")

    formatted = f"[{severity}] {alert_type}: {message}"

    if severity in ("CRITICAL", "ERROR"):
        _send_dingtalk(formatted)

    if severity == "ERROR":
        _send_email(f"[量化系统告警] {alert_type}", formatted)

    if severity == "WARN":
        _send_wework(formatted)

    logger.info("告警已发送: %s", formatted)


def _send_dingtalk(message: str) -> None:
    if not DINGTALK_WEBHOOK:
        logger.debug("钉钉 webhook 未配置")
        return
    try:
        import requests
        requests.post(DINGTALK_WEBHOOK, json={
            "msgtype": "markdown",
            "markdown": {"title": "量化系统告警", "text": message},
        }, timeout=5)
    except Exception:
        pass


def _send_wework(message: str) -> None:
    if not WEWORK_WEBHOOK:
        return
    try:
        import requests
        requests.post(WEWORK_WEBHOOK, json={
            "msgtype": "text",
            "text": {"content": message},
        }, timeout=5)
    except Exception:
        pass


def _send_email(subject: str, body: str) -> None:
    if not EMAIL_SMTP_HOST:
        return
    # 预留邮件发送接口


# ── 事件消费 ──────────────────────────

def handle_health_event(event: dict) -> None:
    """处理服务健康事件。"""
    svc = event.get("service", "unknown")
    _heartbeats[svc] = time.time()
    logger.debug("心跳: %s", svc)


def handle_risk_alert(event: dict) -> None:
    """处理风控告警事件。"""
    _metrics["risk_rejection_count"] += 1
    send_alert(event)


def handle_order_fill(event: dict) -> None:
    """处理成交回报事件。"""
    _metrics["order_count"] += 1
    _metrics["trades_today"].append(event)
    if len(_metrics["trades_today"]) > 10_000:
        _metrics["trades_today"] = _metrics["trades_today"][-5_000:]


def handle_signal_event(event: dict) -> None:
    """处理信号事件。"""
    _metrics["signal_count"] += 1


# ── 日报生成 ──────────────────────────

def generate_daily_report() -> str:
    """每日收盘后生成日报。"""
    report = [
        f"# 量化系统日报",
        f"**日期**: {date.today().isoformat()}",
        f"",
        f"## 今日概览",
        f"- 订单总数: {_metrics['order_count']}",
        f"- 信号总数: {_metrics['signal_count']}",
        f"- 风控拒绝: {_metrics['risk_rejection_count']}",
        f"",
        f"## 服务心跳",
    ]
    for svc, ts in sorted(_heartbeats.items()):
        ago = time.time() - ts
        status_text = "正常" if ago < 300 else f"⚠ {int(ago)}秒前"
        report.append(f"- {svc}: {status_text}")

    report_text = "\n".join(report)

    # 存档
    report_dir = Path(get_env("REPORT_DIR", "data/reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"daily_report_{date.today().isoformat()}.md"
    with open(path, "w") as f:
        f.write(report_text)

    logger.info("日报已生成: %s", path)
    return report_text


# ── Prometheus Metrics ──────────────────

def _serve_metrics_http() -> None:
    """暴露 Prometheus metrics 端点。"""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                # 生成 Prometheus 格式的 metrics
                lines = [
                    "# HELP quant_orders_total Total orders processed",
                    f"quant_orders_total {_metrics['order_count']}",
                    "# HELP quant_signals_total Total signals generated",
                    f"quant_signals_total {_metrics['signal_count']}",
                    "# HELP quant_risk_rejections_total Total risk rejections",
                    f"quant_risk_rejections_total {_metrics['risk_rejection_count']}",
                    "# HELP quant_service_heartbeat_seconds Last heartbeat timestamp",
                ]
                for svc, ts in _heartbeats.items():
                    lines.append(
                        f'quant_service_heartbeat_seconds{{service="{svc}"}} {ts:.0f}'
                    )
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write("\n".join(lines).encode())
            elif self.path == "/health":
                panels = {
                    "services": {svc: (time.time() - ts < 300) for svc, ts in _heartbeats.items()},
                    "metrics_summary": {
                        "orders": _metrics["order_count"],
                        "signals": _metrics["signal_count"],
                        "rejections": _metrics["risk_rejection_count"],
                    },
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(panels).encode())
            else:
                self.send_response(404)
                self.end_headers()

    port = int(get_env("MONITOR_PORT", "9090"))
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    logger.info("monitor HTTP(Prometheus) 监听端口 %d", port)
    server.serve_forever()


# ── 主入口 ──────────────────────────

def main() -> None:
    logger.info("monitor 启动")

    def cleanup():
        try:
            generate_daily_report()
        except Exception:
            pass
    setup_graceful_shutdown(cleanup)

    _serve_metrics_http()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
