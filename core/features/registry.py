"""因子注册与版本管理。

FactorRegistry 管理系统所有因子的定义、计算函数、参数和版本。
从 configs/factors/registry.yaml 加载初始因子定义，支持动态注册新因子。

核心功能：
- 因子定义加载与注册
- 依赖关系 DAG 拓扑排序（保证因子按正确顺序计算）
- 版本管理（参数变更时自动递增版本号）

使用方式：
    registry = FactorRegistry()
    registry.load_from_yaml("configs/factors/registry.yaml")
    dag = registry.compute_order()  # ['momentum_20d', 'roe_ttm']
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


@dataclass
class FactorDefinition:
    """因子元信息定义。"""
    name: str                           # 内部名（如 'momentum_20d'）
    display_name: str                   # 显示名（如 "20日动量"）
    category: str                       # 分类：technical/fundamental/capital_flow/sentiment/premarket
    function: str                       # 计算函数路径（如 "core.features.technical.momentum"）
    params: dict = field(default_factory=dict)   # 函数参数字典
    version: str = "v1.0"                       # 当前版本号
    depends_on: list[str] = field(default_factory=list)  # 依赖的因子名（或数据字段名）
    missing_threshold: float = 0.05              # 缺失率告警阈值
    point_in_time: bool = False                  # 是否需要PIT处理
    description: str = ""

    def increment_version(self) -> str:
        """自动递增版本号（v1.0 → v1.1 → v2.0）。"""
        parts = self.version.lstrip("v").split(".")
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        minor += 1
        if minor >= 10:
            major += 1
            minor = 0
        self.version = f"v{major}.{minor}"
        return self.version


class FactorRegistry:
    """因子注册表，管理所有因子的定义和计算依赖关系。"""

    def __init__(self) -> None:
        self._factors: dict[str, FactorDefinition] = {}

    # ── 加载 ──────────────────────────────────

    def load_from_yaml(self, path: str) -> int:
        """从 YAML 文件批量加载因子定义。

        Args:
            path: YAML 文件路径（如 'configs/factors/registry.yaml'）。

        Returns:
            加载的因子数量。
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        count = 0
        for item in data.get("factors", []):
            fd = FactorDefinition(
                name=item["name"],
                display_name=item.get("display_name", item["name"]),
                category=item.get("category", "unknown"),
                function=item.get("function", ""),
                params=item.get("params", {}),
                version=item.get("version", "v1.0"),
                depends_on=item.get("depends_on", []),
                missing_threshold=item.get("missing_threshold", 0.05),
                point_in_time=item.get("point_in_time", False),
                description=item.get("description", ""),
            )
            self.register(fd)
            count += 1

        logger.info("从 %s 加载了 %d 个因子定义", path, count)
        return count

    # ── 注册与查询 ───────────────────────────

    def register(self, fd: FactorDefinition) -> None:
        """注册（或覆盖）一个因子定义。

        Args:
            fd: 因子定义对象。
        """
        self._factors[fd.name] = fd
        logger.debug("注册因子: %s (v%s, category=%s)", fd.name, fd.version, fd.category)

    def get_factor(self, name: str) -> FactorDefinition:
        """获取单个因子定义。

        Raises:
            KeyError: 因子名不存在。
        """
        if name not in self._factors:
            raise KeyError(f"因子 '{name}' 未注册。可用: {list(self._factors.keys())}")
        return self._factors[name]

    def list_factors(self, category: Optional[str] = None) -> list[FactorDefinition]:
        """列出所有（或指定分类的）因子定义。

        Args:
            category: 因子分类过滤。None 返回全部。
        """
        if category is None:
            return list(self._factors.values())
        return [fd for fd in self._factors.values() if fd.category == category]

    def get_dependencies(self, name: str) -> list[str]:
        """返回某因子的所有依赖（递归展开）。

        例如：如果 A 依赖 B，B 依赖 C，则 get_dependencies('A') → ['B', 'C']。
        """
        fd = self.get_factor(name)
        all_deps: list[str] = []
        visited: set[str] = set()

        def _collect(n: str) -> None:
            if n in visited:
                return
            visited.add(n)
            if n in self._factors:
                for dep in self._factors[n].depends_on:
                    if dep not in visited:
                        all_deps.append(dep)
                        _collect(dep)

        for dep in fd.depends_on:
            _collect(dep)

        return all_deps

    # ── DAG 拓扑排序 ─────────────────────────

    def compute_order(self, factor_names: Optional[list[str]] = None) -> list[str]:
        """根据因子间 depends_on 关系计算计算拓扑序（Kahn算法）。

        Args:
            factor_names: 需要的因子列表。None 表示全部已注册因子。

        Returns:
            拓扑排序列表：先计算排前面的因子（被依赖的在前）。
        """
        targets = set(factor_names) if factor_names else set(self._factors.keys())

        # 构建 DAG：只包含 targets 中的节点及其传递依赖
        all_nodes: set[str] = set()
        for name in targets:
            if name in self._factors:
                all_nodes.add(name)
                for dep in self.get_dependencies(name):
                    all_nodes.add(dep)

        # 过滤出在因子注册表中的真正节点（有些 depends_on 是数据字段名）
        factor_nodes = [n for n in all_nodes if n in self._factors]
        # 不在注册表中的识别为外部数据依赖（如 'close_adj', 'net_profit'）

        # 构建入度表
        in_degree: dict[str, int] = {n: 0 for n in factor_nodes}
        adj: dict[str, list[str]] = {n: [] for n in factor_nodes}

        for name in factor_nodes:
            fd = self._factors[name]
            for dep in fd.depends_on:
                if dep in factor_nodes:   # 只追踪因子间依赖，忽略外部数据依赖
                    adj[dep].append(name)
                    in_degree[name] += 1

        # Kahn 算法
        queue: deque[str] = deque(n for n in factor_nodes if in_degree[n] == 0)
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(factor_nodes):
            remaining = set(factor_nodes) - set(order)
            logger.warning(
                "因子依赖可能存在循环: 未排序节点=%s", remaining,
            )
            # 强制追加剩余节点（打破循环）
            order.extend(remaining)

        logger.debug("因子计算拓扑序: %s", " → ".join(order))
        return order

    def validate(self) -> dict[str, Any]:
        """验证所有因子定义的完整性。

        Returns:
            包含 errors 和 warnings 列表的字典。
        """
        errors: list[str] = []
        warnings: list[str] = []

        for name, fd in self._factors.items():
            if not fd.function:
                errors.append(f"因子 '{name}' 缺少计算函数")

            # 检查依赖的因子是否都已注册
            for dep in fd.depends_on:
                if dep not in self._factors:
                    warnings.append(f"因子 '{name}' 依赖 '{dep}'，但'{dep}'未注册（可能是外部数据字段）")

        return {"errors": errors, "warnings": warnings}

    def __len__(self) -> int:
        return len(self._factors)

    def __contains__(self, name: str) -> bool:
        return name in self._factors

    def __repr__(self) -> str:
        cats: dict[str, int] = defaultdict(int)
        for fd in self._factors.values():
            cats[fd.category] += 1
        cat_summary = ", ".join(f"{k}={v}" for k, v in cats.items())
        return f"FactorRegistry({len(self._factors)} factors: {cat_summary})"
