"""盘前专属因子。

隔夜增量信息因子，用于盘前推荐子系统。
部分因子依赖第5批NLP模型——已预留接口，当前提供基于关键词的规则版本。

统一签名：f(data: pd.DataFrame, params: dict) -> pd.Series
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

# ADR 映射表路径
_ADR_MAPPING_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "premarket" / "adr_mapping.yaml"


# ── 隔夜映射 ─────────────────────────────────────

def overnight_adr_mapped(data: pd.DataFrame, params: dict) -> pd.Series:
    """隔夜中概映射信号。

    读取 configs/premarket/adr_mapping.yaml 中的映射表，
    将隔夜中概股涨跌幅映射到对应的A股标的。

    依赖列：adr_code, adr_overnight_change

    params: {}
    """
    if "adr_code" not in data.columns or "adr_overnight_change" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    # 加载映射表
    try:
        with open(_ADR_MAPPING_PATH) as f:
            mapping_config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        return pd.Series(np.nan, index=data.index)

    mappings = mapping_config.get("mappings", [])
    # adr_code → {a_stock, weight}
    adr_map: dict[str, dict] = {}
    for m in mappings:
        adr_map[m["adr"]] = m

    result = pd.Series(np.nan, index=data.index)
    for adr_code, info in adr_map.items():
        if info["a_stock"] is None or info["weight"] <= 0:
            continue
        mask = data["adr_code"].str.upper() == adr_code.upper()
        if mask.any():
            # 映射信号 = 中概涨跌幅 × 映射权重
            change = data.loc[mask, "adr_overnight_change"].astype(float)
            result.loc[mask] = change * info["weight"]

    return result


def a50_futures_overnight(data: pd.DataFrame, params: dict) -> pd.Series:
    """A50期货隔夜涨跌幅（全市场信号，所有行返回相同值）。

    依赖列：a50_change（富时A50期货涨跌幅）。

    params: {}
    """
    if "a50_change" not in data.columns:
        return pd.Series(np.nan, index=data.index)
    return pd.Series(data["a50_change"].iloc[0] if len(data) > 0 else np.nan, index=data.index)


def hsi_futures_overnight(data: pd.DataFrame, params: dict) -> pd.Series:
    """恒指期货隔夜涨跌幅。

    依赖列：hsi_change。

    params: {}
    """
    if "hsi_change" not in data.columns:
        return pd.Series(np.nan, index=data.index)
    return pd.Series(data["hsi_change"].iloc[0] if len(data) > 0 else np.nan, index=data.index)


# ── NLP预留接口（当前规则版本，第5批后无缝替换）───

def announcement_sentiment_score(data: pd.DataFrame, params: dict) -> pd.Series:
    """公告NLP情绪评分（★当前为规则版本★）。

    基于公告标题中的关键词进行简单的规则评分。
    第5批NLP模型完成后，替换为 Qwen3 推理结果。

    评分范围：-1（强烈负面）~ +1（强烈正面），0为中性。

    依赖列：announcement_title（公告标题）。

    params: {}
    """
    if "announcement_title" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    # 正面/负面关键词
    positive_words = {
        "增长", "大幅增长", "超预期", "翻倍", "扭亏", "中标", "签约",
        "增持", "回购", "分红", "股权激励", "突破", "获批", "上市",
        "业绩预增", "业绩大增", "业绩预盈",
    }
    negative_words = {
        "下降", "亏损", "大幅下降", "预亏", "首亏", "减持", "处罚",
        "退市", "风险警示", "立案", "调查", "诉讼", "违约",
        "业绩预减", "业绩预亏", "业绩下滑",
    }

    titles = data["announcement_title"].fillna("").astype(str)
    scores = pd.Series(0.0, index=data.index)

    for word in positive_words:
        scores[data["announcement_title"].str.contains(word, na=False)] += 0.3
    for word in negative_words:
        scores[data["announcement_title"].str.contains(word, na=False)] -= 0.3

    # 裁剪到 [-1, 1]
    return scores.clip(-1.0, 1.0)


def theme_heat_score(data: pd.DataFrame, params: dict) -> pd.Series:
    """盘前题材热度（★当前为规则版本★）。

    基于新闻标题中的关键词聚类映射到概念板块，统计频率后排序。

    第5批NLP+聚类模型完成后替换。

    依赖列：news_titles（当日的新闻标题列表，可以是聚合后的字符串）。

    params: {}
    """
    # 概念板块关键词映射
    theme_keywords: dict[str, list[str]] = params.get("theme_keywords", {
        "AI": ["人工智能", "大模型", "算力", "ChatGPT", "AGI", "机器学习"],
        "芯片": ["芯片", "半导体", "光刻", "GPU", "CPU", "存储", "HBM"],
        "新能源汽车": ["新能源", "锂电", "固态电池", "钠电池", "充电桩"],
        "光伏": ["光伏", "太阳能", "钙钛矿", "组件", "硅料"],
        "低空经济": ["低空", "无人机", "飞行汽车", "eVTOL", "空管"],
        "机器人": ["机器人", "人形", "具身智能", "自动化"],
        "医药": ["创新药", "CRO", "CDMO", "ADC", "GLP-1"],
        "消费": ["消费", "白酒", "免税", "预制菜"],
    })

    if "news_titles" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    titles = data["news_titles"].fillna("").astype(str).str.cat(sep=" ")

    scores: dict[str, float] = {}
    for theme, keywords in theme_keywords.items():
        count = sum(1 for kw in keywords if kw in titles)
        scores[theme] = count

    # 归一化到 0~1
    max_count = max(scores.values()) if scores else 1
    normalized = {k: v / max_count for k, v in scores.items()} if max_count > 0 else {}

    result = pd.Series(0.0, index=data.index)
    # 每个 code 取对应行业的主题热度
    # 简化：返回全局热度的平均值
    avg_heat = np.mean(list(normalized.values())) if normalized else 0.0
    result[:] = avg_heat
    return result


# ── 龙虎榜复盘 ──────────────────────────────────

def dragon_tiger_review_score(data: pd.DataFrame, params: dict) -> pd.Series:
    """龙虎榜复盘综合评分。

    综合机构净买入额分位 + 游资参与度 + 买卖力量对比。

    依赖列：dragon_tiger_net（净买入，万元），is_known_trader（知名游资参与标志）

    params: {}
    """
    if "dragon_tiger_net" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    net = data["dragon_tiger_net"].fillna(0).astype(float)
    # 截面排名分位（0~1）
    net_rank = net.groupby(data["trade_date"]).rank(pct=True).fillna(0.5)

    # 游资参与加分
    trader_bonus = data.get("is_known_trader", pd.Series(0, index=data.index)).fillna(0).astype(float) * 0.3

    # 买卖力量比
    buy = data.get("dragon_tiger_buy", pd.Series(0, index=data.index)).fillna(0).astype(float)
    sell = data.get("dragon_tiger_sell", pd.Series(0, index=data.index)).fillna(0).astype(float)
    buy_sell_ratio = buy / sell.replace(0, float("nan"))
    buy_sell_rank = buy_sell_ratio.groupby(data["trade_date"]).rank(pct=True).fillna(0.5)

    # 综合：净买30% + 游资30% + 买卖比40%
    combined = net_rank * 0.3 + trader_bonus * 0.3 + buy_sell_rank * 0.4
    return combined.clip(0.0, 1.0)


def limit_up_review_signal(data: pd.DataFrame, params: dict) -> pd.Series:
    """昨日涨停股今日高开概率（基于历史统计）。

    依赖列：yesterday_limit_up（昨日是否涨停）。

    params: {lookback_days: 60}
    """
    if "yesterday_limit_up" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    lookback = params.get("lookback_days", 60)
    was_limit_up = data["yesterday_limit_up"].astype(bool)
    today_gap_up = (
        (data["open"] - data["pre_close"]) / data["pre_close"].replace(0, float("nan"))
    ) > 0.01  # 高开 > 1%

    # 滚动窗口内"昨日涨停→今日高开"的条件概率
    events = was_limit_up.groupby(data["code"]).rolling(lookback, min_periods=1).sum().droplevel(0)
    successes = (was_limit_up & today_gap_up).groupby(data["code"]).rolling(
        lookback, min_periods=1
    ).sum().droplevel(0)
    prob = successes / events.replace(0, float("nan"))
    return prob


# ── 集合竞价 ─────────────────────────────────────

def auction_strength_score(data: pd.DataFrame, params: dict) -> pd.Series:
    """竞价强度综合评分 = 偏离度×0.4 + 量比×0.3 + 不平衡度×0.3。

    依赖列：auction_price, pre_close, auction_volume, volume。
    """
    premium = auction_open_premium(data, params)
    vol_ratio = auction_volume_ratio(data, params)

    # 买卖不平衡度
    if "auction_buy_vol" in data.columns and "auction_sell_vol" in data.columns:
        total = (data["auction_buy_vol"] + data["auction_sell_vol"]).replace(0, float("nan"))
        imbalance = (data["auction_buy_vol"] - data["auction_sell_vol"]) / total
        imbalance_filled = imbalance.fillna(0)
    else:
        imbalance_filled = pd.Series(0.0, index=data.index)

    # 归一化各分量
    premium_norm = _safe_rank_norm(premium)
    vol_ratio_norm = _safe_rank_norm(vol_ratio)
    imbalance_norm = _safe_rank_norm(imbalance_filled)

    return (premium_norm * 0.4 + vol_ratio_norm * 0.3 + imbalance_norm * 0.3).clip(0.0, 1.0)


def auction_fake_order_risk(data: pd.DataFrame, params: dict) -> pd.Series:
    """竞价虚假申报风险评分。

    9:15-9:20的申报量远大于9:20-9:25的最终匹配量 → 虚假申报嫌疑。
    评分范围 0（安全）~ 1（高度可疑）。

    依赖列：auction_order_count_pre_920, auction_order_count_post_920。

    params: {threshold: 3.0}  — 申报量/最终匹配量比值阈值。
    """
    if "auction_order_count_pre_920" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    pre = data["auction_order_count_pre_920"].fillna(0).astype(float)
    post_col = data.get("auction_order_count_post_920", pd.Series(0, index=data.index))
    post = post_col.fillna(1).replace(0, 1).astype(float)

    ratio = pre / post
    threshold = params.get("threshold", 3.0)

    risk = ratio / threshold
    return risk.clip(0.0, 1.0)


# ── 辅助 ──────────────────────────────────────

def days_to_next_event(data: pd.DataFrame, params: dict) -> pd.Series:
    """距下一重大事件（业绩披露日/股东大会/限售解禁）的剩余天数。

    返回剩余天数，999表示未来90天内无重大事件。

    依赖列：next_event_date。

    params: {}
    """
    if "next_event_date" not in data.columns:
        return pd.Series(999, index=data.index)

    trade_dates = pd.to_datetime(data["trade_date"])
    event_dates = pd.to_datetime(data["next_event_date"])
    days = (event_dates - trade_dates).dt.days.fillna(999).clip(upper=999)
    return days


def _safe_rank_norm(series: pd.Series) -> pd.Series:
    """安全地将 series 按截面排名归一化到 0~1。"""
    if series.isna().all():
        return pd.Series(0.5, index=series.index)
    return series.groupby(level=0).rank(pct=True).fillna(0.5)
