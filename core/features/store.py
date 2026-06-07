"""特征存储。

负责因子的持久化（ClickHouse）和实时缓存（Redis）：
- 离线回测：从 ClickHouse 批量加载因子历史值
- 实时推理：从 Redis 读取最新因子值（亚毫秒延迟）

使用方式：
    store = FeatureStore(ch_client=ch, redis_client=redis)
    store.save_factor_values(date, "momentum_20d", {"600519": 0.034}, "v2.1")
    df = store.load_factor_values(start, end, ["momentum_20d", "roe_ttm"])
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Redis key 模板
_REDIS_KEY_PREFIX = "feat"
_REDIS_KEY_TEMPLATE = f"{_REDIS_KEY_PREFIX}:{{factor_name}}:{{code}}"


class FeatureStore:
    """特征存储：ClickHouse（历史）+ Redis（实时缓存）。

    批量写入和读取走 ClickHouse，实时特征查询走 Redis。
    """

    def __init__(
        self,
        ch_client: Any = None,      # ClickHouseClient
        redis_client: Any = None,   # RedisClient
    ) -> None:
        """初始化特征存储。

        Args:
            ch_client: ClickHouse 客户端（None 时跳过数据库操作）。
            redis_client: Redis 客户端（None 时跳过缓存操作）。
        """
        self._ch = ch_client
        self._redis = redis_client

    # ── ClickHouse 持久化 ────────────────────────────

    def save_factor_values(
        self,
        calc_date: date,
        factor_name: str,
        values: dict[str, float],
        version: str,
    ) -> int:
        """保存因子值到 ClickHouse。

        Args:
            calc_date: 计算日期。
            factor_name: 因子名。
            values: {code: value} 字典。
            version: 因子版本号。

        Returns:
            写入的行数。
        """
        if self._ch is None:
            logger.debug("ClickHouse 未配置，跳过因子持久化: %s", factor_name)
            return 0

        rows = [
            {"calc_date": calc_date, "factor_name": factor_name,
             "code": code, "value": value, "version": version}
            for code, value in values.items()
            if value is not None and not (isinstance(value, float) and pd.isna(value))
        ]

        if not rows:
            return 0

        df = pd.DataFrame(rows)
        self._ch.insert_df("factor_values", df)
        logger.debug("保存因子 %s: %d 条 (version=%s)", factor_name, len(rows), version)
        return len(rows)

    def save_factor_values_batch(
        self,
        calc_date: date,
        factor_values: dict[str, dict[str, float]],
        versions: dict[str, str],
    ) -> int:
        """批量保存多个因子值。

        Args:
            calc_date: 计算日期。
            factor_values: {factor_name: {code: value}} 嵌套字典。
            versions: {factor_name: version} 版本映射。

        Returns:
            写入的总行数。
        """
        total = 0
        for factor_name, values in factor_values.items():
            version = versions.get(factor_name, "v1.0")
            total += self.save_factor_values(calc_date, factor_name, values, version)
        return total

    def load_factor_values(
        self,
        start: date,
        end: date,
        factor_names: list[str],
        codes: Optional[list[str]] = None,
        version: Optional[str] = None,
    ) -> pd.DataFrame:
        """从 ClickHouse 批量加载因子历史值。

        Args:
            start: 起始日期（含）。
            end: 截止日期（含）。
            factor_names: 因子名列表。
            codes: 股票代码过滤（None=全市场）。
            version: 版本过滤（None=最新版本）。

        Returns:
            宽表格式 DataFrame：行 = (date, code)，列 = factor_names。
        """
        if self._ch is None:
            logger.warning("ClickHouse 未配置，返回空 DataFrame")
            return pd.DataFrame()

        factor_list = "', '".join(factor_names)
        sql = f"""
            SELECT calc_date, code, factor_name, value, version
            FROM factor_values
            WHERE factor_name IN ('{factor_list}')
              AND calc_date >= '{start.isoformat()}'
              AND calc_date <= '{end.isoformat()}'
        """
        if codes:
            code_list = "', '".join(codes)
            sql += f"\n  AND code IN ('{code_list}')"

        df = self._ch.query_df(sql)

        if df.empty:
            return pd.DataFrame(columns=["calc_date", "code"] + factor_names)

        # 如果指定了版本，过滤
        if version is not None:
            df = df[df["version"] == version]

        # Pivot: 长表 → 宽表
        pivoted = df.pivot_table(
            index=["calc_date", "code"],
            columns="factor_name",
            values="value",
            aggfunc="last",  # 同一日期+因子的最新版本
        ).reset_index()

        logger.info(
            "加载因子: %s, %d 行 × %d 因子 (%s ~ %s)",
            factor_names, len(pivoted), len(factor_names), start, end,
        )
        return pivoted

    # ── Redis 实时缓存 ───────────────────────────

    def cache_to_redis(
        self,
        calc_date: date,
        factor_name: str,
        values: dict[str, float],
        ttl_seconds: int = 86400,
    ) -> int:
        """将最新因子值写入 Redis 缓存。

        Key 格式: feat:{factor_name}:{code}
        默认 TTL: 24小时（日频因子）/ 300秒（实时因子）。

        Args:
            calc_date: 计算日期。
            factor_name: 因子名。
            values: {code: value}。
            ttl_seconds: TTL（秒）。

        Returns:
            成功写入的 key 数量。
        """
        if self._redis is None:
            return 0

        count = 0
        ttl = ttl_seconds  # 日频因子默认24小时
        for code, value in values.items():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            key = _REDIS_KEY_TEMPLATE.format(factor_name=factor_name, code=code)
            payload = json.dumps({"value": value, "date": calc_date.isoformat()})
            self._redis.set(key, payload, ttl=ttl)
            count += 1

        logger.debug("Redis 缓存: %s, %d 条", factor_name, count)
        return count

    def get_latest_from_redis(
        self,
        codes: list[str],
        factor_names: list[str],
    ) -> dict[str, dict[str, Optional[float]]]:
        """从 Redis 批量读取最新因子值（用于实时推理）。

        Args:
            codes: 股票代码列表。
            factor_names: 因子名列表。

        Returns:
            {code: {factor_name: value}} 嵌套字典。
            不存在的 key 返回 None。
        """
        if self._redis is None:
            return {}

        keys: list[str] = []
        key_map: list[tuple[str, str]] = []  # [(key, (code, factor_name))]
        for fn in factor_names:
            for c in codes:
                k = _REDIS_KEY_TEMPLATE.format(factor_name=fn, code=c)
                keys.append(k)
                key_map.append((c, fn))

        raw = self._redis.mget(keys)

        result: dict[str, dict[str, Optional[float]]] = {c: {} for c in codes}
        for (code, fn), val in zip(key_map, raw):
            if val is not None:
                try:
                    parsed = json.loads(val)
                    result[code][fn] = float(parsed["value"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    result[code][fn] = None
            else:
                result[code][fn] = None

        return result

    def cache_batch_to_redis(
        self,
        calc_date: date,
        factor_values: dict[str, dict[str, float]],
        ttl_seconds: int = 86400,
    ) -> int:
        """批量缓存多个因子到 Redis（Pipeline 优化）。"""

        if self._redis is None:
            return 0

        count = 0
        pipe = self._redis.pipeline()
        for factor_name, values in factor_values.items():
            for code, value in values.items():
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    continue
                key = _REDIS_KEY_TEMPLATE.format(factor_name=factor_name, code=code)
                payload = json.dumps({"value": value, "date": calc_date.isoformat()})
                pipe.set(key, payload, ex=ttl_seconds)
                count += 1

        pipe.execute()
        logger.debug("Redis Pipeline 批量缓存: %d 条", count)
        return count

    # ── 健康检查 ─────────────────────────────────

    def ping(self) -> dict[str, bool]:
        """检查 ClickHouse 和 Redis 连接状态。"""
        status: dict[str, bool] = {}

        if self._ch is not None:
            try:
                self._ch.client.execute("SELECT 1")
                status["clickhouse"] = True
            except Exception:
                status["clickhouse"] = False

        if self._redis is not None:
            status["redis"] = self._redis.ping()

        return status
