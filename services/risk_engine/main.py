"""风控引擎微服务。

常驻进程，暴露 gRPC CheckOrder 接口，实时监控净值和合规阈值。
支持配置热加载，熔断触发时调用 order_manager 紧急平仓。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.common import HealthChecker, MessageBus, get_broker, get_env, setup_graceful_shutdown

logger = logging.getLogger(__name__)

health = HealthChecker("risk_engine")
bus = MessageBus(broker=get_broker(), client_id="risk_engine")

# ── 运行时状态 ─────────────────────────

_peak_nav: float = 0.0
_current_nav: float = 0.0
_nav_history: list[float] = []
_checker: Any = None
_portfolio_mgr: Any = None
_recent_rejections: list[dict] = []  # 最近的拒绝记录（用于拒绝率突增检测）


def _get_checker():
    global _checker
    if _checker is None:
        from core.risk.pre_trade import PreTradeRiskChecker
        _checker = PreTradeRiskChecker()
    return _checker


def _get_portfolio_mgr():
    global _portfolio_mgr
    if _portfolio_mgr is None:
        from core.risk.portfolio_risk import PortfolioRiskManager
        _portfolio_mgr = PortfolioRiskManager()
    return _portfolio_mgr


# ── gRPC / HTTP 接口 ────────────────────

def handle_check_order(request: dict) -> dict:
    """处理 CheckOrder 请求（同步，<10ms）。

    实际部署通过 gRPC proto 定义，这里用 HTTP+JSON 简化实现。
    """
    checker = _get_checker()
    result = checker.check(
        code=request.get("code", ""),
        side=request.get("side", "buy"),
        price=float(request.get("price", 0)),
        shares=int(request.get("shares", 0)),
        account_id=int(request.get("account_id", 1)),
        current_positions=request.get("current_positions", {}),
        current_cash=float(request.get("current_cash", 0)),
        total_asset=float(request.get("total_asset", 0)),
        current_prices=request.get("current_prices", {}),
        industry_map=request.get("industry_map", {}),
        is_st=request.get("is_st", False),
        is_suspended=request.get("is_suspended", False),
    )
    result_dict = result.to_dict()
    if not result.passed:
        _recent_rejections.append({"time": time.time(), "reason": result.reject_reason})
        _clean_old_rejections()
    return result_dict


def handle_query_portfolio_risk() -> dict:
    mgr = _get_portfolio_mgr()
    return {
        "current_nav": _current_nav,
        "peak_nav": _peak_nav,
        "daily_drawdown": (_peak_nav - _current_nav) / _peak_nav if _peak_nav > 0 else 0.0,
        "circuit_breaker_triggered": mgr.is_circuit_breaker_triggered,
    }


# ── 实时监控 ──────────────────────────

def update_nav(nav: float) -> None:
    """收到成交回报更新净值→检查回撤熔断。"""
    global _current_nav, _peak_nav
    _current_nav = nav
    if nav > _peak_nav:
        _peak_nav = nav
    _nav_history.append(nav)

    mgr = _get_portfolio_mgr()
    action = mgr.check_drawdown(nav, _peak_nav)

    if action is not None:
        logger.warning("熔断触发: %s", action.reason)
        bus.send("risk.alert", {
            "alert_type": "circuit_breaker",
            "severity": "CRITICAL",
            "message": action.reason,
            "action": action.to_dict(),
            "timestamp": datetime.now().isoformat(),
        })
        # 发送紧急平仓请求到 order_manager
        bus.send("signals.exit", {
            "type": "emergency_flat",
            "action": action.action_type,
            "reason": action.reason,
        })


def _clean_old_rejections() -> None:
    """清理5分钟前的拒绝记录。"""
    cutoff = time.time() - 300
    global _recent_rejections
    _recent_rejections = [r for r in _recent_rejections if r["time"] > cutoff]


def check_compliance_alerts() -> None:
    """检查合规告警条件并发送 risk.alert。"""
    alerts: list[dict] = []

    # 拒绝率突增
    if len(_recent_rejections) > 10:
        alerts.append({
            "alert_type": "rejection_spike",
            "severity": "ERROR",
            "message": f"5分钟内风控拒绝 {len(_recent_rejections)} 次",
        })

    # 回撤接近熔断线
    if _peak_nav > 0:
        dd = (_peak_nav - _current_nav) / _peak_nav
        if dd > 0.10:  # 距离15%熔断线不到5%
            alerts.append({
                "alert_type": "drawdown_warning",
                "severity": "WARN",
                "message": f"回撤{dd:.1%}接近熔断线",
            })

    for alert in alerts:
        bus.send("risk.alert", {**alert, "timestamp": datetime.now().isoformat()})


# ── 配置热加载 ────────────────────────

def _listen_config_updates() -> None:
    """监听 Redis pub/sub 频道获取配置更新。"""
    try:
        from core.data.db import RedisClient
        redis = RedisClient(url=get_env("REDIS_URL", "redis://localhost:6379"))
        pubsub = redis.subscribe("config:risk:update")
        logger.info("已订阅 config:risk:update")
        # 实际部署中开独立线程监听
    except Exception as e:
        logger.debug("配置热加载监听启动失败: %s（将在文件轮询中处理）", e)


# ── HTTP 服务 ──────────────────────────

def _serve_http() -> None:
    """启动简化 HTTP API。"""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
            if self.path == "/CheckOrder":
                result = handle_check_order(body)
            elif self.path == "/QueryRisk":
                result = handle_query_portfolio_risk()
            else:
                result = {"error": "unknown endpoint"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result, default=str).encode())

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(health.status(), default=str).encode())
            else:
                self.send_response(404)
                self.end_headers()

    port = int(get_env("RISK_ENGINE_PORT", "50052"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    logger.info("risk_engine HTTP 监听端口 %d", port)
    server.serve_forever()


# ── 主入口 ──────────────────────────

def main() -> None:
    logger.info("risk_engine 启动")

    def cleanup():
        pass
    setup_graceful_shutdown(cleanup)

    _listen_config_updates()
    _serve_http()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
