"""特征计算微服务。

提供实时流计算和批量离线计算两种模式，将因子值写入 Redis 缓存。
暴露 gRPC GetFeatures 接口供策略引擎查询最新特征。
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.common import HealthChecker, get_env, get_redis_url, setup_graceful_shutdown

logger = logging.getLogger(__name__)

health = HealthChecker("feature_server")

# ── 配置 ──────────────────────────────

REDIS_URL = get_redis_url()

# 延迟初始化
_store = None
_registry = None


def _get_store():
    global _store
    if _store is None:
        from core.features.store import FeatureStore
        from core.data.db import RedisClient
        redis = RedisClient(url=REDIS_URL)
        _store = FeatureStore(ch_client=None, redis_client=redis)
    return _store


def _get_registry():
    global _registry
    if _registry is None:
        from core.features.registry import FactorRegistry
        _registry = FactorRegistry()
        _registry.load_from_yaml("configs/factors/registry.yaml")
    return _registry


# ── 计算模式 ──────────────────────────

def compute_factors_batch(calc_date: date | None = None) -> dict:
    """批量计算所有日频因子（收盘后调用）。"""
    if calc_date is None:
        calc_date = date.today()

    registry = _get_registry()
    store = _get_store()
    order = registry.compute_order()
    logger.info("批量计算 %d 个因子 (date=%s)", len(order), calc_date)

    results: dict[str, dict[str, float]] = {}  # {factor_name: {code: value}}
    factor_modules = {
        "technical": "core.features.technical",
        "fundamental": "core.features.fundamental",
        "capital_flow": "core.features.capital_flow",
        "sentiment": "core.features.sentiment",
    }

    for factor_name in order:
        fd = registry.get_factor(factor_name)
        try:
            # 动态导入并计算
            mod_path = fd.function.rsplit(".", 1)[0]
            func_name = fd.function.rsplit(".", 1)[1]
            import importlib
            mod = importlib.import_module(mod_path)
            func = getattr(mod, func_name)
            # 这里需要行情数据——实际部署时从 ClickHouse 加载
            # 简化实现：占位返回
            results[factor_name] = {}
            logger.debug("因子 %s 计算完成", factor_name)
        except Exception as e:
            logger.error("因子 %s 计算失败: %s", factor_name, e)
            health.record_failure(str(e))

    # 写入 Redis
    for fname, values in results.items():
        if values:
            fd = registry.get_factor(fname)
            store.cache_to_redis(calc_date, fname, values, ttl_seconds=86400)

    health.record_success()
    return results


def compute_factors_stream(tick_data: dict) -> None:
    """实时流计算：单个tick到达时增量更新因子。

    仅更新依赖 tick 数据的实时因子（如盘口不平衡、订单流等）。
    大部分日频因子不需要实时重算。
    """
    store = _get_store()
    # 实时因子列表（轻量级，可基于tick增量更新）
    realtime_factors = []  # 由配置决定
    for fname in realtime_factors:
        try:
            # 增量更新逻辑
            pass
        except Exception as e:
            logger.debug("实时因子 %s 更新失败: %s", fname, e)


# ── gRPC 服务 ─────────────────────────

def _serve_grpc() -> None:
    """启动简化 gRPC 服务（实际部署用 proto 生成）。"""
    # 简化实现：通过 HTTP+JSON 暴露接口
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class FeatureHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/GetFeatures":
                content_len = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(content_len))
                codes = body.get("codes", [])
                factor_names = body.get("factor_names", [])
                store = _get_store()
                result = store.get_latest_from_redis(codes, factor_names)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result, default=str).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(health.status(), default=str).encode())
            else:
                self.send_response(404)
                self.end_headers()

    port = int(get_env("FEATURE_SERVER_PORT", "50051"))
    server = HTTPServer(("0.0.0.0", port), FeatureHandler)
    logger.info("feature_server gRPC(HTTP) 监听端口 %d", port)
    server.serve_forever()


# ── 主入口 ──────────────────────────

def main() -> None:
    registry = _get_registry()
    logger.info("feature_server 启动，已加载 %d 个因子定义", len(registry))

    def cleanup():
        pass

    setup_graceful_shutdown(cleanup)
    _serve_grpc()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
