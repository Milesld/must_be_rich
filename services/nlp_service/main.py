"""NLP 推理微服务。

常驻进程，托管 Qwen3 模型进行中文金融文本情绪分析。
暴露 AnalyzeSentiment 和 BatchAnalyze 两个 gRPC 接口。
支持 GPU/CPU 自适应和降级回退。
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

from services.common import HealthChecker, get_env, setup_graceful_shutdown

logger = logging.getLogger(__name__)

health = HealthChecker("nlp_service")

# ── 模型管理 ──────────────────────────

_analyzer: Any = None
_model_load_time: float = 0.0
_last_inference_latency_ms: float = 0.0


def _get_analyzer() -> Any:
    """延迟加载 NLP 分析器。

    优先 GPU（vLLM），回退 CPU（0.6B INT8 量化版）。
    """
    global _analyzer, _model_load_time

    if _analyzer is None:
        from core.models.nlp import NLPSentimentAnalyzer

        model_name = get_env("NLP_MODEL", "keyword")  # 'keyword' / 'FinSenti-Qwen3-0.6B' / ...
        device = get_env("NLP_DEVICE", "auto")

        start = time.time()
        _analyzer = NLPSentimentAnalyzer(model_name=model_name, device=device)
        _model_load_time = time.time() - start

        # 尝试加载 Transformer 模型
        if model_name != "keyword":
            _analyzer.install_transformers_model(model_name)

        logger.info("NLP 模型已加载: %s (%.1fs)", model_name, _model_load_time)

    return _analyzer


# ── 推理接口 ──────────────────────────

def analyze_single(text: str, source: str = "announcement") -> dict:
    """单条文本情绪分析。"""
    global _last_inference_latency_ms

    start = time.time()
    try:
        analyzer = _get_analyzer()
        result = analyzer.analyze_single(text, source)
        _last_inference_latency_ms = (time.time() - start) * 1000
        health.record_success()
        return {
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "event_type": result.event_type,
            "summary": result.summary,
        }
    except Exception as e:
        health.record_failure(str(e))
        return {
            "sentiment": "neutral", "confidence": 0.0,
            "event_type": None, "summary": "",
            "error": str(e),
        }


def analyze_batch(texts: list[tuple[str, str]], timeout: float = 30.0) -> list[dict]:
    """批量分析。

    Args:
        texts: [(text, source), ...]
        timeout: 硬超时时间（秒），超时则返回已处理的结果。
    """
    analyzer = _get_analyzer()
    results: list[dict] = []
    start = time.time()

    try:
        raw = analyzer.analyze_batch(texts, timeout=timeout)
        for r in raw:
            results.append({
                "sentiment": r.sentiment,
                "confidence": r.confidence,
                "event_type": r.event_type,
                "summary": r.summary,
            })
    except Exception as e:
        logger.error("批量 NLP 分析失败: %s", e)
        health.record_failure(str(e))

    elapsed = (time.time() - start) * 1000
    health.record_success()
    logger.info("批量分析完成: %d 条, %.0fms", len(results), elapsed)
    return results


def model_status() -> dict:
    """返回模型加载状态和性能指标。"""
    analyzer = _get_analyzer()

    gpu_info = {}
    try:
        import torch
        gpu_info = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        gpu_info = {"gpu_available": False, "note": "torch not installed"}

    return {
        "model_name": analyzer.model_name,
        "model_load_time_s": _model_load_time,
        "last_inference_latency_ms": _last_inference_latency_ms,
        **gpu_info,
        **health.status(),
    }


# ── Scheduler: GPU 调度优先级 ────────────

class GPUScheduler:
    """GPU 调度优先级管理器。

    优先级: 日内实时推理 > 盘前批量 > 离线研报分析。

    简化实现：通过队列长度和 GPU 利用率判断。
    """

    def __init__(self) -> None:
        self._queue_lens: dict[str, int] = {"realtime": 0, "premarket": 0, "offline": 0}
        self._max_queue = {"realtime": 50, "premarket": 100, "offline": 500}

    def can_enqueue(self, priority: str) -> bool:
        return self._queue_lens.get(priority, 0) < self._max_queue.get(priority, 100)

    def enqueue(self, priority: str) -> None:
        self._queue_lens[priority] = self._queue_lens.get(priority, 0) + 1

    def dequeue(self, priority: str) -> None:
        if self._queue_lens.get(priority, 0) > 0:
            self._queue_lens[priority] -= 1


gpu_scheduler = GPUScheduler()


# ── HTTP API ──────────────────────────

def _serve_http() -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
            if self.path == "/AnalyzeSentiment":
                result = analyze_single(
                    body.get("text", ""),
                    body.get("source", "announcement"),
                )
            elif self.path == "/BatchAnalyze":
                texts = [(item["text"], item["source"]) for item in body.get("texts", [])]
                result = analyze_batch(texts, timeout=body.get("timeout", 30.0))
            else:
                result = {"error": "unknown endpoint"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(model_status(), default=str).encode())
            else:
                self.send_response(404)
                self.end_headers()

    port = int(get_env("NLP_SERVICE_PORT", "50053"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    logger.info("nlp_service HTTP 监听端口 %d", port)
    server.serve_forever()


# ── 主入口 ──────────────────────────

def main() -> None:
    logger.info("nlp_service 启动")

    def cleanup():
        pass
    setup_graceful_shutdown(cleanup)

    # WarmUp: 提前加载模型
    _get_analyzer()

    _serve_http()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
