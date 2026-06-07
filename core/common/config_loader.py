"""配置加载器。

从 configs/ 目录加载所有 YAML 配置，支持环境变量替换、热加载、
属性式访问和字典式访问。

使用方式：
    loader = ConfigLoader()
    max_ratio = loader.risk.thresholds.single_stock_max_ratio
    max_ratio = loader["risk"]["thresholds"]["single_stock_max_ratio"]

热加载：
    风控阈值类配置监听 Redis pub/sub 频道 "config:risk:update"，
    收到更新通知后自动重载对应文件。如果 Redis 不可用，退化为
    文件轮询（30秒间隔）。
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env_vars(value: Any) -> Any:
    """递归解析值中的 ${ENV_VAR:default} 占位符。"""
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            var_name = m.group(1)
            default = m.group(2) if m.group(2) is not None else ""
            return os.environ.get(var_name, default)
        return _ENV_VAR_PATTERN.sub(_replace, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


class _ConfigNode:
    """支持属性访问和字典访问的配置节点。"""

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(f"配置项 '{name}' 不存在") from None
        if isinstance(value, dict):
            return _ConfigNode(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, dict):
            return _ConfigNode(value)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"ConfigNode({list(self._data.keys())})"

    def get(self, key: str, default: Any = None) -> Any:
        """安全获取配置项，不存在时返回默认值。"""
        value = self._data.get(key, default)
        if isinstance(value, dict):
            return _ConfigNode(value)
        return value

    def to_dict(self) -> dict[str, Any]:
        """递归转换为普通 dict。"""
        result: dict[str, Any] = {}
        for k, v in self._data.items():
            if isinstance(v, _ConfigNode):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


class ConfigLoader:
    """配置加载器，加载 configs/ 下所有 YAML 文件。

    支持：
    - 本地 YAML 文件加载
    - ${ENV_VAR:default} 环境变量替换
    - 热加载（Redis pub/sub 或文件轮询）
    - 属性式和字典式双重访问
    """

    def __init__(
        self,
        configs_dir: str | Path | None = None,
    ) -> None:
        """初始化配置加载器。

        Args:
            configs_dir: 配置文件目录。默认从项目根目录的 configs/ 加载。
        """
        self._configs_dir = Path(configs_dir) if configs_dir else _CONFIGS_DIR
        self._data: dict[str, Any] = {}
        self._redis_client: Any = None
        self._polling_thread: Optional[threading.Thread] = None
        self._stop_polling = threading.Event()
        self._watched_files: set[str] = set()

        self.load_all()

    # —— 公共方法 ——

    def load_all(self) -> None:
        """加载 configs/ 下所有 YAML 文件。"""
        if not self._configs_dir.exists():
            raise FileNotFoundError(
                f"配置目录不存在: {self._configs_dir}。"
                f"请确认项目根目录下存在 configs/ 目录。"
            )

        yaml_files = sorted(self._configs_dir.rglob("*.yaml"))
        if not yaml_files:
            logger.warning("配置目录 %s 下没有找到 .yaml 文件", self._configs_dir)
            return

        for yaml_path in yaml_files:
            self._load_yaml(yaml_path)

    def reload(self, rel_path: str) -> None:
        """重载单个配置文件。

        Args:
            rel_path: 相对于 configs/ 的路径，如 'risk/thresholds.yaml'。
        """
        yaml_path = self._configs_dir / rel_path
        if not yaml_path.exists():
            logger.warning("配置文件不存在，跳过重载: %s", yaml_path)
            return
        self._load_yaml(yaml_path)
        logger.info("配置热加载完成: %s", rel_path)

    # —— 热加载 ——

    def enable_hot_reload(
        self,
        redis_url: str = "redis://localhost:6379",
        watched_files: list[str] | None = None,
    ) -> None:
        """启用热加载支持。

        优先使用 Redis pub/sub，不可用时退化为文件轮询。

        Args:
            redis_url: Redis 连接地址。
            watched_files: 需要监控的文件列表（相对于 configs/ 的路径）。
        """
        self._watched_files = set(watched_files or ["risk/thresholds.yaml"])

        try:
            import redis as _redis
            self._redis_client = _redis.from_url(redis_url)
            # 尝试 Redis 订阅
            pubsub = self._redis_client.pubsub()
            pubsub.subscribe("config:risk:update")
            self._stop_polling.clear()
            self._polling_thread = threading.Thread(
                target=self._redis_listener,
                daemon=True,
                name="config-hot-reload-redis",
            )
            self._polling_thread.start()
            logger.info("配置热加载已启用 (Redis pub/sub)")
        except Exception:
            logger.info("Redis 不可用，配置热加载退化为文件轮询 (30s)")
            self._redis_client = None
            self._stop_polling.clear()
            self._polling_thread = threading.Thread(
                target=self._file_polling_listener,
                daemon=True,
                name="config-hot-reload-file",
            )
            self._polling_thread.start()

    def disable_hot_reload(self) -> None:
        """停止热加载。"""
        self._stop_polling.set()
        if self._polling_thread is not None:
            self._polling_thread.join(timeout=5.0)
        logger.info("配置热加载已停止")

    # —— 属性式访问 ——

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(
                f"顶级配置 '{name}' 不存在。可用: {list(self._data.keys())}"
            ) from None
        if isinstance(value, dict):
            return _ConfigNode(value)
        return value

    # —— 字典式访问 ——

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, dict):
            return _ConfigNode(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """安全获取顶级配置项。"""
        return self._data.get(key, default)

    def keys(self) -> Any:
        """返回顶级配置键列表。"""
        return list(self._data.keys())

    def __repr__(self) -> str:
        return f"ConfigLoader({list(self._data.keys())})"

    # —— 内部方法 ——

    def _load_yaml(self, yaml_path: Path) -> None:
        """加载单个 YAML 文件并入配置树。"""
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        if raw is None:
            return

        # 环境变量替换
        resolved = _resolve_env_vars(raw)

        # 按文件路径嵌套: boards/sse_main.yaml → data['boards']['sse_main']
        rel = yaml_path.relative_to(self._configs_dir)
        parts = list(rel.parts)
        parts[-1] = parts[-1].replace(".yaml", "")

        root = self._data
        for part in parts[:-1]:
            if part not in root:
                root[part] = {}
            root = root[part]
        root[parts[-1]] = resolved

    def _redis_listener(self) -> None:
        """Redis pub/sub 热加载监听线程。"""
        while not self._stop_polling.is_set():
            try:
                pubsub = self._redis_client.pubsub()
                pubsub.subscribe("config:risk:update")
                for message in pubsub.listen():
                    if self._stop_polling.is_set():
                        break
                    if message["type"] == "message":
                        data = message.get("data")
                        if isinstance(data, bytes):
                            data = data.decode("utf-8")
                        file_path = data if isinstance(data, str) else "risk/thresholds.yaml"
                        self.reload(file_path)
            except Exception:
                logger.debug("Redis 连接中断，5秒后重试")
                time.sleep(5)

    def _file_polling_listener(self) -> None:
        """文件轮询热加载监听线程，每30秒检查 mtime。"""
        mtimes: dict[str, float] = {}
        # 初始化 mtime
        for rel_path in self._watched_files:
            fpath = self._configs_dir / rel_path
            mtimes[rel_path] = fpath.stat().st_mtime if fpath.exists() else 0

        while not self._stop_polling.is_set():
            time.sleep(30)
            for rel_path in self._watched_files:
                fpath = self._configs_dir / rel_path
                if not fpath.exists():
                    continue
                current_mtime = fpath.stat().st_mtime
                if current_mtime > mtimes.get(rel_path, 0):
                    mtimes[rel_path] = current_mtime
                    self.reload(rel_path)


# —— 全局单例 ——

_config_instance: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """获取 ConfigLoader 全局单例。"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance
