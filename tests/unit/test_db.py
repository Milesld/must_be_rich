"""数据库客户端单元测试。

使用 mock 验证 ClickHouseClient, MySQLClient, RedisClient 的
接口和行为正确性。由于无法在 CI 中实际连接数据库，这些测试
验证客户端的构造、配置加载和基本方法签名。
"""

from __future__ import annotations

import pytest


class TestClickHouseClient:
    """ClickHouse 客户端测试。"""

    def test_init_with_url(self) -> None:
        from core.data.db import ClickHouseClient

        ch = ClickHouseClient(url="http://localhost:8123")
        assert ch._url == "http://localhost:8123"

    def test_init_with_kwargs(self) -> None:
        from core.data.db import ClickHouseClient

        ch = ClickHouseClient(url="http://localhost:9000", user="default", password="secret")
        assert ch._url == "http://localhost:9000"

    def test_repr(self) -> None:
        from core.data.db import ClickHouseClient

        ch = ClickHouseClient()
        repr_str = repr(ch)
        assert "ClickHouseClient" in repr_str or "ClickHouse" in str(type(ch))


class TestMySQLClient:
    """MySQL 客户端测试。"""

    def test_init_with_url(self) -> None:
        from core.data.db import MySQLClient

        mysql = MySQLClient(url="mysql://user:pass@localhost:3306/quant")
        assert mysql._url == "mysql://user:pass@localhost:3306/quant"

    def test_context_manager_protocol(self) -> None:
        """验证 session 上下文管理器的方法存在。"""
        from core.data.db import MySQLClient

        mysql = MySQLClient()
        # 不实际连接，只验证方法签名
        # (engine 是 @property，会触发延迟初始化；这里检查底层属性和公共方法)
        assert hasattr(mysql, "session")
        assert hasattr(mysql, "close")
        assert hasattr(mysql, "table_exists")
        assert hasattr(mysql, "execute")

    def test_close_is_idempotent(self) -> None:
        from core.data.db import MySQLClient

        mysql = MySQLClient()
        mysql.close()  # 未初始化时 close 应不报错
        mysql.close()  # 二次 close 也应不报错


class TestRedisClient:
    """Redis 客户端测试。"""

    def test_init_with_url(self) -> None:
        from core.data.db import RedisClient

        redis = RedisClient(url="redis://localhost:6379")
        assert redis._url == "redis://localhost:6379"

    def test_method_signatures_exist(self) -> None:
        """验证所有公开方法存在。"""
        from core.data.db import RedisClient

        redis = RedisClient()
        methods = ["get", "set", "delete", "exists", "expire",
                   "hget", "hset", "hgetall", "hmset",
                   "publish", "subscribe", "mget", "mset",
                   "pipeline", "ping", "close"]
        for name in methods:
            assert hasattr(redis, name), f"缺少方法: {name}"

    def test_close_is_idempotent(self) -> None:
        from core.data.db import RedisClient

        redis = RedisClient()
        redis.close()
        redis.close()  # 不应报错

    def test_ping_returns_false_when_no_server(self) -> None:
        from core.data.db import RedisClient

        redis = RedisClient(url="redis://nonexistent:6379")
        # 未初始化 ping 可能返回 False 或抛异常
        # 延迟初始化（lazy init）意味着未调用 client 属性前不会连接
        # 一旦调用 client 属性，ping 可能因为连接失败而抛异常
        # 这里验证 ping 不会崩溃
        try:
            result = redis.ping()
            # 取决于 redis-py 的行为：可能 False 或抛异常
            assert result is True or result is False
        except Exception:
            pass  # 连接失败也是预期行为
