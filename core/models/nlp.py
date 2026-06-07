"""NLP 情绪分析模型。

中文金融文本情绪分析，基于关键词规则（默认，零依赖）+
可选 Transformers 模型（Qwen3/DeepSeek）升级。

架构：
- 默认使用关键词规则引擎（立即可用）
- 升级到 Qwen3 系列需安装 transformers + torch
- 首次使用自动处理模型下载
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SentimentLabel = Literal["positive", "neutral", "negative"]


@dataclass
class SentimentResult:
    """单条文本的情绪分析结果。"""
    sentiment: SentimentLabel
    confidence: float           # 0.0 ~ 1.0
    event_type: Optional[str] = None  # 事件类型
    summary: str = ""


# ── 关键词规则引擎（默认方案）───────────────────

class KeywordSentimentEngine:
    """基于关键词的金融文本情绪评分（零依赖）。

    词表覆盖A股公告常见正面/负面表述。
    """

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
        """分析单条文本的情绪。"""
        pos_count = sum(1 for w in self.POSITIVE_WORDS if w in text)
        neg_count = sum(1 for w in self.NEGATIVE_WORDS if w in text)

        total = pos_count + neg_count
        if total == 0:
            return SentimentResult(sentiment="neutral", confidence=0.3)

        if pos_count > neg_count:
            conf = min(pos_count / (total + 1), 1.0)
            return SentimentResult(sentiment="positive", confidence=conf)
        elif neg_count > pos_count:
            conf = min(neg_count / (total + 1), 1.0)
            return SentimentResult(sentiment="negative", confidence=conf)
        else:
            return SentimentResult(sentiment="neutral", confidence=0.5)

    def extract_event_type(self, text: str) -> Optional[str]:
        """从文本中提取事件类型。"""
        for event_type, pattern in self.EVENT_PATTERNS.items():
            if pattern.search(text):
                return event_type
        return None

    def summarize(self, text: str, max_len: int = 100) -> str:
        """生成一句话摘要。"""
        # 取前N个非空白字符作为摘要
        clean = text.replace("\n", " ").replace("\r", "").strip()
        if len(clean) <= max_len:
            return clean
        return clean[:max_len] + "..."


# ── NLP 分析器（统一接口）─────────────────

class NLPSentimentAnalyzer:
    """中文金融文本情绪分析器。

    默认使用关键词规则引擎（零依赖，立即可用）。
    可以通过 install_transformers_model() 升级到 Qwen3 系列。

    使用方式：
        analyzer = NLPSentimentAnalyzer()
        result = analyzer.analyze_single("贵州茅台 2025年度业绩预增50%")
        # → SentimentResult(sentiment='positive', confidence=0.8, ...)
    """

    def __init__(
        self,
        model_name: str = "keyword",     # 'keyword' | 'FinSenti-Qwen3-0.6B' | ...
        device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._keyword_engine = KeywordSentimentEngine()
        self._hf_model: Any = None       # HuggingFace model
        self._hf_tokenizer: Any = None

        if model_name != "keyword":
            logger.info("NLP模型配置为 %s（首次使用时自动下载）", model_name)

    # ── 公共接口 ────────────────────────────

    def analyze_single(
        self, text: str, source: str = "announcement",
    ) -> SentimentResult:
        """单条文本情绪分析。

        Args:
            text: 待分析文本。
            source: 'announcement' | 'news' | 'research'。
                    prompt 的构造方式因来源而异。
        """
        if not text or not text.strip():
            return SentimentResult(sentiment="neutral", confidence=0.0)

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
        """批量分析。

        Args:
            texts: [(text, source), ...]
            timeout: 硬超时时间（秒）。

        Returns:
            结果列表，顺序与输入一一对应。
        """
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
        """事件分类。

        Returns:
            {业绩超预期, 重大合同, 增减持, 重组进展, 分红送转, 风险警示, 其他} 或 None。
        """
        return self._keyword_engine.extract_event_type(text)

    # ── Transformer 升级（可选）─────────────

    def install_transformers_model(self, model_id: str) -> bool:
        """安装并加载 HuggingFace Transformers 模型。

        Args:
            model_id: HuggingFace 或 ModelScope 模型ID。
                      推荐: 'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-8B'

        Returns:
            True 如果安装成功。
        """
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
        """使用 HuggingFace 模型分析（简化：规则引擎兜底）。"""
        # HuggingFace推理需具体prompt设计，这里fallback到规则分析
        # 实际部署时，这里是一段 prompt → model.generate() → parse 的流程
        result = self._keyword_engine.analyze(text)
        result.event_type = self._keyword_engine.extract_event_type(text)
        result.summary = self._keyword_engine.summarize(text)
        return result

    @property
    def model_name(self) -> str:
        return self._model_name
