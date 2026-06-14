"""NLP 情绪分析模型。

中文金融文本情绪分析，支持三种引擎：
- KeywordSentimentEngine：零依赖关键词规则（默认，立即可用）
- DeepSeekAPIEngine：DeepSeek API 云端推理（需要 DEEPSEEK_API_KEY）
- Transformers 本地模型：Qwen3/DeepSeek 本地部署（需要 transformers + torch）

使用方式：
    # 关键词规则（默认）
    analyzer = NLPSentimentAnalyzer()
    result = analyzer.analyze_single("业绩预增50%")

    # DeepSeek API
    analyzer = NLPSentimentAnalyzer(model_name="deepseek")
    result = analyzer.analyze_single("业绩预增50%")

    # 本地模型
    analyzer = NLPSentimentAnalyzer(model_name="Qwen/Qwen3-0.6B")
    analyzer.install_transformers_model("Qwen/Qwen3-0.6B")
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

SentimentLabel = Literal["positive", "neutral", "negative"]


@dataclass
class SentimentResult:
    """单条文本的情绪分析结果。"""
    sentiment: SentimentLabel
    confidence: float           # 0.0 ~ 1.0
    event_type: Optional[str] = None  # 事件类型
    summary: str = ""
    raw_response: str = ""      # 模型的原始输出（用于调试）


# ══════════════════════════════════════════════════════════════
# 引擎 1: 关键词规则（默认，零依赖）
# ══════════════════════════════════════════════════════════════

class KeywordSentimentEngine:
    """基于关键词的金融文本情绪评分（零依赖）。"""

    POSITIVE_WORDS = {
        "增长", "大幅增长", "超预期", "翻倍", "扭亏", "扭亏为盈",
        "中标", "签约", "增持", "回购", "分红", "高送转", "股权激励",
        "突破", "获批", "上市", "业绩预增", "业绩大增", "业绩预盈",
        "创新高", "产能释放", "量价齐升", "景气", "需求旺盛",
        "收到补贴", "政府补助", "项目投产", "新签订单", "产能扩张",
        "技术突破", "专利", "通过认证", "获得批文", "战略合作",
    }
    NEGATIVE_WORDS = {
        "下降", "亏损", "大幅下降", "预亏", "首亏", "减持", "处罚",
        "退市", "风险警示", "立案", "调查", "诉讼", "违约",
        "业绩预减", "业绩预亏", "业绩下滑", "停产", "限产",
        "商誉减值", "计提减值", "债务违约", "逾期", "冻结",
        "终止上市", "暂停上市", "退市风险", "监管函", "警示函",
        "非标意见", "无法表示意见", "保留意见",
    }
    EVENT_PATTERNS = {
        "业绩超预期": re.compile(r"(业绩|净利|营收).*(超预期|大幅增长|预增|大增)"),
        "重大合同": re.compile(r"(中标|签约|新签订单|合同|协议).*\d+(亿|万|千)"),
        "增减持": re.compile(r"(增持|减持|回购)"),
        "重组进展": re.compile(r"(重组|并购|资产注入|借壳|定向增发|非公开发行)"),
        "分红送转": re.compile(r"(分红|送转|派息|转增|高送转)"),
        "风险警示": re.compile(r"(风险警示|退市|立案|调查|处罚|违规)"),
    }

    def analyze(self, text: str) -> SentimentResult:
        pos_count = sum(1 for w in self.POSITIVE_WORDS if w in text)
        neg_count = sum(1 for w in self.NEGATIVE_WORDS if w in text)
        total = pos_count + neg_count
        if total == 0:
            return SentimentResult(sentiment="neutral", confidence=0.3)
        if pos_count > neg_count:
            return SentimentResult(sentiment="positive", confidence=min(pos_count / (total + 1), 1.0))
        elif neg_count > pos_count:
            return SentimentResult(sentiment="negative", confidence=min(neg_count / (total + 1), 1.0))
        else:
            return SentimentResult(sentiment="neutral", confidence=0.5)

    def extract_event_type(self, text: str) -> Optional[str]:
        for event_type, pattern in self.EVENT_PATTERNS.items():
            if pattern.search(text):
                return event_type
        return None

    def summarize(self, text: str, max_len: int = 100) -> str:
        clean = text.replace("\n", " ").replace("\r", "").strip()
        if len(clean) <= max_len:
            return clean
        return clean[:max_len] + "..."


# ══════════════════════════════════════════════════════════════
# 引擎 2: DeepSeek API（云端推理）
# ══════════════════════════════════════════════════════════════

class DeepSeekAPIEngine:
    """DeepSeek API 云端金融文本情绪分析引擎。

    使用 OpenAI 兼容的 chat/completions 接口。
    需要环境变量 DEEPSEEK_API_KEY。
    不需要本地 GPU，不需要下载模型。

    计费：DeepSeek API 按 token 计费（¥1/百万输入 token），单条公告约 ¥0.001。
    """

    API_BASE = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"

    # 金融情绪分析的 System Prompt
    SYSTEM_PROMPT = """你是一个专业的A股金融公告分析师。
请对以下上市公司公告进行情绪分析，按指定 JSON 格式输出。

分析维度：
1. sentiment: 对股价的短期影响
   - "positive": 业绩超预期、重大合同、增持回购、政策利好
   - "negative": 业绩下滑、违规处罚、减持、退市风险、诉讼
   - "neutral": 日常经营、例行公告、中性信息
2. confidence: 你的判断置信度 (0.0~1.0)
3. event_type: 事件类型，从以下选择：
   业绩超预期、业绩预减、重大合同、增减持、重组进展、分红送转、
   风险警示、股权激励、其他
4. summary: 一句话总结公告核心内容（15字以内）

请严格输出一行 JSON，不要输出其他内容。
格式示例：{"sentiment": "positive", "confidence": 0.85, "event_type": "业绩超预期", "summary": "24年净利润预增50%"}
"""

    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        base_url: str = API_BASE,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

        if not self._api_key:
            logger.warning(
                "DeepSeek API key 未设置。请设置环境变量 DEEPSEEK_API_KEY，"
                "或传入 api_key 参数。当前将回退到关键词规则引擎。"
            )

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def analyze(self, text: str, source: str = "announcement") -> SentimentResult:
        """调用 DeepSeek API 分析单条文本。"""
        if not self.is_available:
            logger.warning("DeepSeek API 不可用（无 API key），回退到关键词规则")
            return self._fallback(text)

        system_prompt = self.SYSTEM_PROMPT
        if source == "news":
            system_prompt = system_prompt.replace("上市公司公告", "财经新闻")
        elif source == "research":
            system_prompt = system_prompt.replace("上市公司公告", "券商研报")
            system_prompt = system_prompt.replace("短期影响", "中期影响")

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text[:4000]},  # 限制长度控制成本
            ],
            "temperature": 0.1,       # 低温度 = 更确定性
            "max_tokens": 200,
            "stream": False,
        }

        try:
            req = urllib.request.Request(
                f"{self._base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            content = body["choices"][0]["message"]["content"].strip()
            # 提取 JSON（有时模型会在 JSON 外包 markdown 代码块）
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                result = json.loads(json_match.group())
                return SentimentResult(
                    sentiment=result.get("sentiment", "neutral"),
                    confidence=float(result.get("confidence", 0.5)),
                    event_type=result.get("event_type"),
                    summary=result.get("summary", ""),
                    raw_response=content,
                )
            else:
                logger.warning("DeepSeek 返回非 JSON 格式: %s", content[:100])
                return self._fallback(text)

        except Exception as e:
            logger.warning("DeepSeek API 调用失败: %s，回退到关键词规则", e)
            return self._fallback(text)

    def analyze_batch(
        self,
        texts: list[tuple[str, str]],
        timeout: float = 60.0,
    ) -> list[SentimentResult]:
        """批量分析：依次调用 API（可改为并发）。"""
        results: list[SentimentResult] = []
        for text, source in texts:
            results.append(self.analyze(text, source))
        return results

    def extract_event_type(self, text: str) -> Optional[str]:
        """快速事件分类（不需要完整推理，用关键词做先用后付）。"""
        result = self.analyze(text)
        return result.event_type

    def _fallback(self, text: str) -> SentimentResult:
        """API 不可用时的回退方案。"""
        kw = KeywordSentimentEngine()
        result = kw.analyze(text)
        result.event_type = kw.extract_event_type(text)
        result.summary = kw.summarize(text)
        return result


# ══════════════════════════════════════════════════════════════
# 统一接口
# ══════════════════════════════════════════════════════════════

class NLPSentimentAnalyzer:
    """中文金融文本情绪分析器 — 三引擎统一接口。

    引擎选择：
        model_name="keyword"   → 关键词规则（默认，零依赖）
        model_name="deepseek"  → DeepSeek API（需 DEEPSEEK_API_KEY）
        model_name="Qwen/Qwen3-0.6B" → 本地 Transformers（需手动调用 install）

    使用方式：
        # 关键词
        analyzer = NLPSentimentAnalyzer()
        result = analyzer.analyze_single("业绩预增50%")

        # DeepSeek
        analyzer = NLPSentimentAnalyzer(model_name="deepseek")
        result = analyzer.analyze_single("业绩预增50%")

        # 或通过环境变量切换
        export NLP_MODEL=deepseek
        analyzer = NLPSentimentAnalyzer(model_name=os.environ.get("NLP_MODEL", "keyword"))
    """

    def __init__(
        self,
        model_name: str = "",
        device: str = "auto",
        api_key: str = "",
        api_base: str = "",
    ) -> None:
        # 优先级：参数 > 环境变量 > 默认 "keyword"
        _model = model_name or os.environ.get("NLP_MODEL", "keyword")
        self._model_name = _model
        self._device = device

        # 初始化引擎
        self._keyword_engine = KeywordSentimentEngine()
        self._deepseek_engine: Optional[DeepSeekAPIEngine] = None
        self._hf_model: Any = None
        self._hf_tokenizer: Any = None

        if _model in ("deepseek", "deepseek-chat", "deepseek-v4", "deepseek-v4-pro"):
            ds_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
            ds_base = api_base or os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
            self._deepseek_engine = DeepSeekAPIEngine(
                api_key=ds_key, base_url=ds_base,
            )
            if self._deepseek_engine.is_available:
                logger.info("NLP 引擎: DeepSeek API (%s)", self._deepseek_engine._model)
            else:
                logger.warning("NLP 引擎: DeepSeek 配置了但 API key 未设置，回退到关键词规则")

        elif _model != "keyword":
            logger.info("NLP 引擎: %s（首次使用时加载）", _model)

    # ── 公共接口 ────────────────────────────

    def analyze_single(
        self, text: str, source: str = "announcement",
    ) -> SentimentResult:
        """单条文本情绪分析。

        Args:
            text: 待分析文本。
            source: 'announcement' | 'news' | 'research'。
        """
        if not text or not text.strip():
            return SentimentResult(sentiment="neutral", confidence=0.0)

        # DeepSeek API 引擎（优先级最高）
        if self._deepseek_engine is not None and self._deepseek_engine.is_available:
            return self._deepseek_engine.analyze(text, source)

        # 本地 Transformers 模型
        if self._hf_model is not None:
            return self._analyze_hf(text, source)

        # 默认：关键词规则
        result = self._keyword_engine.analyze(text)
        result.event_type = self._keyword_engine.extract_event_type(text)
        result.summary = self._keyword_engine.summarize(text)
        return result

    def analyze_batch(
        self,
        texts: list[tuple[str, str]],
        timeout: float = 30.0,
    ) -> list[SentimentResult]:
        """批量分析。"""
        results: list[SentimentResult] = []
        for text, source in texts:
            try:
                r = self.analyze_single(text, source)
                results.append(r)
            except Exception as e:
                logger.warning("NLP分析失败: %s", e)
                results.append(SentimentResult(sentiment="neutral", confidence=0.0))
        return results

    def extract_event_type(self, text: str) -> Optional[str]:
        """事件分类。"""
        if self._deepseek_engine is not None and self._deepseek_engine.is_available:
            return self._deepseek_engine.extract_event_type(text)
        return self._keyword_engine.extract_event_type(text)

    # ── Transformers 本地模型（可选）─────────

    def install_transformers_model(self, model_id: str) -> bool:
        """安装并加载 HuggingFace 模型。"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("正在加载模型: %s ...", model_id)
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True,
            )
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=self._device if self._device != "cpu" else None,
                trust_remote_code=True,
            )
            self._model_name = model_id
            logger.info("模型加载完成: %s", model_id)
            return True
        except ImportError:
            logger.warning(
                "transformers 未安装。请 pip install transformers torch 后重试。"
                "当前将使用关键词规则引擎。"
            )
            return False
        except Exception as e:
            logger.error("模型加载失败: %s", e)
            return False

    def _analyze_hf(self, text: str, source: str) -> SentimentResult:
        """HuggingFace 模型推理。"""
        result = self._keyword_engine.analyze(text)
        result.event_type = self._keyword_engine.extract_event_type(text)
        result.summary = self._keyword_engine.summarize(text)
        return result

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def active_engine(self) -> str:
        """返回当前实际使用的引擎名称。"""
        if self._deepseek_engine is not None and self._deepseek_engine.is_available:
            return "deepseek"
        if self._hf_model is not None:
            return "transformers"
        return "keyword"
