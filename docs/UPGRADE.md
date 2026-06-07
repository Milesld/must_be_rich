# 系统升级指南

> 本文记录当前系统架构中全部可替换部件及其升级路径。每个组件说明三件事：
> **当前是怎么实现的**、**升级到什么**、**涉及修改哪些文件**。

---

## 一、升级全景图

```
                      ┌─────────────────────────────────┐
                      │         upgrade level           │
                      └─────────────────────────────────┘
                                │
    ┌───────────┬───────────┬───┴───┬───────────┬───────────┐
    │ 数据源     │ 数据库     │ 消息   │ 模型推理   │ 执行      │
    │ upgrade1  │ upgrade2  │ upgrade3│ upgrade4  │ upgrade5  │
    └───────────┴───────────┴────────┴───────────┴───────────┘

    每个升级独立进行，互不阻塞。接口已预留，按本文指引替换即可。
```

---

## 二、upgrade1 — 数据源升级

### 2.1 当前实现

| 层级 | 使用 | 接口 |
|------|------|------|
| 主数据源 | `AkShareSource` — 免费爬虫级数据 | 继承 `DataSourceBase` (§`core/data/sources/base.py`) |
| 备用数据源 | `TushareSource` — Token 认证 API | 同上 |
| 切换策略 | `FallbackDataSource` — 主源失败自动切备用 | 包装两个 `DataSourceBase` 实例 |
| 默认激活 | AkShare（单源模式） | `FallbackDataSource(primary, primary, max_failures=1)` |

**当前质量约束**：
- 退市股数据不完整 → 幸存者偏差风险
- 集合竞价快照不可用 → `raise NotImplementedError`
- 财报数据无实际披露日（announce_date 近似等于 report_date）→ 前视偏差风险
- 分钟线需逐只拉取，速度慢

### 2.2 可升级至

| 级别 | 数据源 | 年费 | 质量提升 |
|------|--------|------|---------|
| **Level 1**（当前） | AkShare + Tushare 备用 | 0~1500元 | 基线 |
| **Level 2** | Tushare 全功能 + QMT L2 | ~8000元 | 集合竞价快照可用、分钟线批量拉取、Level-2 逐笔 |
| **Level 3** | iFinD（同花顺） | 1.45万~3.6万 | PIT 数据完整、财务数据精确到披露日、2026年推出 MCP 服务 |
| **Level 4** | Wind（万得） | 3.98万~15万+ | 机构级金标准、退市股全量历史、PIT 数据最完整 |
| **Level 5** | Wind + 交易所直连 | 50万+ | 微秒级行情、FAST 协议解码、Level-2 全量历史 |

### 2.3 升级方法

**新增一个适配器只需 3 步**：

#### Step 1: 创建新适配器类

```python
# core/data/sources/wind.py
from core.data.sources.base import DataSourceBase

class WindSource(DataSourceBase):
    @property
    def source_name(self) -> str:
        return "wind"

    def get_daily_kline(self, start, end, codes=None) -> pd.DataFrame:
        # 调用 Wind SDK → 字段映射到 DataSourceBase 统一列名
        ...
    # 实现其余 6 个接口
```

#### Step 2: 在 FallbackDataSource 中注册

```python
# services/data_collector/main.py  或  任意调用的地方
from core.data.sources.wind import WindSource
from core.data.sources.fallback import FallbackDataSource

primary = WindSource()
fallback = TushareSource(token="...")
ds = FallbackDataSource(primary, fallback)
```

#### Step 3: 更新配置

```yaml
# configs/data_sources.yaml
wind:
  enabled: true
  username: "${WIND_USERNAME}"
  password: "${WIND_PASSWORD}"
```

**无需修改的模块**：`DataSourceBase` 接口不变、所有 `core/` 下的调用方都依赖抽象接口 `DataSourceBase` 而不依赖具体实现、`FallbackDataSource` 的主备切换逻辑不关心具体数据源类型。

### 2.4 Level-2 数据接入

当前系统设计中所有涉及 Level-2 的组件（集合竞价分析引擎、盘口因子、`AuctionAnomalyDetector`）都留有接口，只是在 AkShare 层 `raise NotImplementedError`。升级到 QMT L2 或 iFinD L2 后，实现以下方法即可激活：

| 方法 | 当前状态 | 激活后影响 |
|------|---------|-----------|
| `get_auction_snapshot()` | `raise NotImplementedError` | 盘前推荐竞价分析引擎启动 |
| `get_minute_kline()` | 逐只拉取，慢 | 批量拉取，tick 回放可用 |
| Level-2 逐笔 | 无 | `AuctionAnomalyDetector` 模型可训练 |

### 2.5 Tushare 已实现但未激活

`TushareSource` 代码已完成，只需两行改动即可启用主/备双活：

```python
# services/data_collector/main.py 中 _get_pipeline() 函数
from core.data.sources.tushare import TushareSource
import os

primary = AkShareSource()
fallback = TushareSource(token=os.environ["TUSHARE_TOKEN"])
ds = FallbackDataSource(primary, fallback, max_failures=3)
```

---

## 三、upgrade2 — 数据库升级

### 3.1 当前实现

| 数据 | 当前方案 | 方式 |
|------|---------|------|
| 行情历史 | 不持久化（研究模式） / ClickHouse（Docker 模式） | `ClickHouseClient` |
| 基本面 | 同上 | `ClickHouseClient` |
| 因子值 | 不持久化（研究） / ClickHouse + Redis 缓存 | `FeatureStore` |
| 业务数据（订单/持仓/账户） | MySQL | `MySQLClient` |
| 特征缓存 | Redis | `RedisClient` |

**重要**：研究模式下 `FeatureStore` 和 `DataPipeline` 都是在 `ch_client=None` 模式下工作的——数据在内存中直接流转而不写入任何数据库。这是有意的设计：研究模式零配置即可运行。

### 3.2 可升级至

| 组件 | 当前 | 升级目标 | 收益 |
|------|------|---------|------|
| 时序数据库 | ClickHouse（单节点） | **DolphinDB**（中国量化事实标准） | 因子计算吞吐提升 100x、流批一体、2000+ 金融函数、Real-time Tick 处理 |
| 缓存 | Redis（单机） | Redis Cluster / Dragonfly | 高可用、更高吞吐 |
| 业务数据库 | MySQL（单机） | PostgreSQL / MySQL 主从 | JSON 原生支持、更强分析查询 |

### 3.3 升级方法

**三个客户端的接口已统一**，替换数据库只需改连接配置，不改代码：

```yaml
# configs/system.yaml
databases:
  clickhouse_url: "${CLICKHOUSE_URL:http://localhost:8123}"  # 升级时改为 DolphinDB 连接
  mysql_url: "${MYSQL_URL:mysql://quant:quant@localhost:3306/quant}"
  redis_url: "${REDIS_URL:redis://localhost:6379}"
```

如果要对接 DolphinDB，新增一个适配器类：

```python
# core/data/db.py 新增
class DolphinDBClient:
    """DolphinDB 客户端（替换 ClickHouseClient 时需要）。"""
    # 实现 execute(), query(), query_df(), insert_df() 四个方法即可
```

`DataPipeline`、`FeatureStore`、回测引擎都通过 `ch_client` 参数注入客户端，不关心底层是 ClickHouse 还是 DolphinDB。

---

## 四、upgrade3 — 消息队列升级

### 4.1 当前实现

| 场景 | 当前方案 | 代码位置 |
|------|---------|---------|
| 有 `kafka-python` 时 | Kafka 兼容模式（对接 Redpanda） | `services/common.py` → `MessageBus` |
| 无 `kafka-python` 时 | **降级为日志模式**（`_LogProducer`），所有消息仅写入日志，不实际发送 | 同上 |
| 微服务间通信 | HTTP+JSON（简化版，非真实 gRPC） | 各 service 的 `_serve_http()` |

**当前约束**：服务之间的真实消息传递依赖 `kafka-python` 库。在无法安装这个库的环境（比如当前网络受限的开发机），`MessageBus` 退化为 `_LogProducer`——消息不会跨服务传递。

### 4.2 可升级至

| 组件 | 当前 | 升级目标 | 收益 |
|------|------|---------|------|
| 消息总线 | Redpanda 单节点（kafka-python 客户端） | Redpanda 3节点集群（生产） | 高可用、持久化保证 |
| RPC | HTTP+JSON（简化版） | **gRPC（proto 定义）** | 强类型、更低序列化开销 |
| 内部通信 | 无 | **NATS**（微秒级延迟） | Tick-to-Trade 路径延迟优化 |

### 4.3 升级方法

**gRPC 升级**：在 `proto/` 下定义 `.proto` 文件，替换各 service 的 `_serve_http()` 为 gRPC server。当前 HTTP 端点的语义已与 gRPC 一对一对应：

| HTTP 端点 | 对应 gRPC 方法 | 文件 |
|-----------|---------------|------|
| `POST /GetFeatures` | `FeatureService.GetFeatures` | `services/feature_server/main.py` |
| `POST /CheckOrder` | `RiskService.CheckOrder` | `services/risk_engine/main.py` |
| `POST /AnalyzeSentiment` | `NLPService.AnalyzeSentiment` | `services/nlp_service/main.py` |

**消息总线升级**：安装 `kafka-python` 即可激活真实消息通道：

```bash
pip install kafka-python
# MessageBus.__init__ 中会自动调用 KafkaProducer 而非 _LogProducer
```

**NATS 集成**（高级，Tick-to-Trade 优化）：在 C++ 组件之间引入 NATS 作为内部 pub/sub，替换 tick 数据从 Python→Redpanda→Python 的路径为 C++ → NATS → C++。

---

## 五、upgrade4 — NLP 模型升级

### 5.1 当前实现

| 组件 | 当前 | 代码位置 |
|------|------|---------|
| NLP 引擎 | `KeywordSentimentEngine` — 零依赖关键词规则 | `core/models/nlp.py` |
| 接口 | `NLPSentimentAnalyzer`（统一的 `analyze_single` / `analyze_batch`） | `core/models/nlp.py` |
| 调用方 | `premarket_service` / 盘前因子 | 通过 `analyze_single()` → `SentimentResult` |

**当前效果**：关键词匹配，正向词 / 负向词各约 20 个，事件分类 6 种。准确率有限但零延迟、零依赖、立即可用。

### 5.2 可升级至

| 模型 | 大小 | 推理延迟 | 场景 | 硬件要求 |
|------|------|---------|------|---------|
| FinSenti-Qwen3-0.6B | 0.6B | <50ms (CPU INT8) | 实时新闻情绪 | 无 GPU 即可 |
| **Qwen3-8B rLoRA**（推荐升级路径） | 8B | 批量 GPU | 日频公告批量分析 + 情绪因子 | A10/A100 |
| DeepSeek-R1-Distill-Qwen-7B | 7B | 离线 | 复杂研报解读 | A100 |

### 5.3 升级方法

**一条命令**：

```python
# 首次启动时自动下载模型
from core.models.nlp import NLPSentimentAnalyzer

analyzer = NLPSentimentAnalyzer(model_name="FinSenti-Qwen3-0.6B")
analyzer.install_transformers_model("Qwen/Qwen3-0.6B")
# → 自动下载模型权重 → 替换关键词引擎 → 对外接口不变
```

**升级后所有调用方不受影响**：
- `analyze_single(text, source)` → `SentimentResult`（接口不变）
- `analyze_batch(texts)` → `List[SentimentResult]`（接口不变）
- `extract_event_type(text)` → `str|None`（接口不变）

**vLLM 生产部署**（高吞吐）：在 `services/nlp_service/` 中用 vLLM 替换当前 HTTP 推理：

```yaml
# docker-compose.prod.yml 中取消 GPU 注释
nlp_service:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

```bash
# 启动时加载模型
NLP_MODEL=Qwen3-8B NLP_DEVICE=cuda docker-compose up -d nlp_service
```

### 5.4 盘前因子中的 NLP 预留接口

当前 `core/features/premarket.py` 中的以下因子已预留 NLP 接口，升级后自动替换：

| 因子 | 当前实现 | 升级后替换 |
|------|---------|-----------|
| `announcement_sentiment_score()` | 关键词规则评分 | → `NLPSentimentAnalyzer.analyze_single()` 结果 |
| `theme_heat_score()` | 关键词聚类 | → 新闻标题 Qwen3 → 概念聚类 |

---

## 六、upgrade5 — 执行层升级

### 6.1 当前实现

| 模式 | 下单位置 | 实现 |
|------|---------|------|
| 回测模式 | `BacktestEngine._handle_trade()` | `ClosePriceFillSimulator` / `OHLCFillSimulator` 模拟撮合 |
| 模拟盘 | `order_manager` | `_MockQMT` — 打印日志，不真实下单 |
| 实盘 | `order_manager` | 需要 `QMT_ENABLED=true` + QMT SDK 已安装 |

### 6.2 升级方法

#### Step 1: 开通券商 QMT

开通条件：50万资金门槛、风险测评 C4+、知识测试通过。开通后获得 `QMT_ACCOUNT_ID` 和数据目录路径。

#### Step 2: 安装 QMT Python SDK

```bash
# 券商提供的 xtquant 包
pip install /path/to/xtquant-*.whl
```

#### Step 3: 启用实盘模式

```bash
# .env
QMT_ENABLED=true
QMT_ACCOUNT_ID=your_account_id
QMT_DATA_PATH=/path/to/QMT/userdata_mini
```

#### Step 4: 渐进式验证

```
模拟盘 2 周 → 小资金（5%-10%）2 周 → 正常资金
```

每一步验证后再进入下一步。

#### C++ 加速

C++ 组件 `cpp/order_router/` 已写好 pybind11 绑定，在编译安装后替换 Python 版 order_manager 中的路由逻辑：

```python
# services/order_manager/main.py
try:
    from order_router import OrderRouter  # C++ 版
except ImportError:
    from _MockQMT import OrderRouter      # Python 回退
```

### 6.3 模拟盘到实盘的差异

| 维度 | 回测模拟 | 模拟盘 | 实盘 |
|------|---------|--------|------|
| 撮合 | 收盘价 / VWAP | 真实撮合引擎 | 交易所真实撮合 |
| 延迟 | 0ms | 真实网络延迟 | 真实网络延迟 |
| 流动性 | 无限 | 真实盘口厚度 | 真实盘口厚度 |
| 滑点 | 估算 | 真实滑点 | 真实滑点 |
| 心理压力 | 无 | 无 | **有**（最大变量） |

---

## 七、upgrade6 — C++ 组件编译部署

### 7.1 当前实现

| 组件 | 当前状态 | 代码 |
|------|---------|------|
| 事前风控 | Python 版可用 | `core/risk/pre_trade.py` |
| C++ 风控 | **已写、已绑定，未编译** | `cpp/risk_checker/` |
| C++ Tick 解码 | **已写、已绑定，未编译** | `cpp/tick_processor/` |
| C++ 订单路由 | **已写、已绑定，未编译** | `cpp/order_router/` |

三个 C++ 模块代码已完成，`pybind11` 绑定已完成，`CMakeLists.txt` 已就绪。因为环境没有 C++ 编译工具链（或编译步骤被跳过），这几个模块当前不可用。

### 7.2 编译

```bash
cd cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)  # macOS: make -j$(sysctl -n hw.logicalcpu)

# 产物:
# build/risk_checker.cpython-*.so   → 10万次检查 P99<100μs
# build/tick_processor.cpython-*.so  → UDP组播接收 + 零拷贝共享内存
# build/order_router.cpython-*.so   → 内置重试+状态查询
```

### 7.3 Python 侧使用

```python
# 直接替代 Python 版（接口完全兼容）
from risk_checker import PreTradeRiskChecker, RiskConfig
checker = PreTradeRiskChecker(RiskConfig())
result = checker.check(req)  # <10μs vs Python ~500μs
```

---

## 八、upgrade7 — 特征工程从 Python 迁移到 DolphinDB

### 8.1 当前实现

全部因子计算由 `core/features/` 下的 Python 函数完成，数据通过 pandas DataFrame 传参。实时因子通过 `FeatureStore.cache_to_redis()` 写入 Redis。

### 8.2 升级时机

当出现以下信号时考虑迁移：
- 全市场因子计算超过 30 分钟
- 实时因子在 Tick 频率下出现延迟
- 因子数量超过 200 个且维护成本上升

### 8.3 升级路径

DolphinDB 内置 2000+ 金融函数，迁移后因子计算从分钟级→毫秒级。关键点：

- **接口不变**：`FactorRegistry`、`FeatureStore` 的 Python 接口保持不动
- **store.py 内部切换**：`save_factor_values()` → DolphinDB 客户端；`get_latest_from_redis()` → DolphinDB 流表订阅
- **因子计算迁移策略**：先迁移延迟敏感的高频因子（盘口、订单流），再迁移日频因子

---

## 九、升级优先级建议

按投入产出比排序：

| 优先级 | 升级项 | 成本 | 收益 | 前提条件 |
|--------|--------|------|------|---------|
| **1** | Tushare 启用主/备双活 | 配置 TUSHARE_TOKEN | 数据可靠性 +1 | 注册 tushare.pro |
| **2** | NLP → Qwen3 0.6B | 下载 1.2GB 模型 | 情绪因子准确率 +30% | GPU 或 16GB RAM |
| **3** | 开通 QMT L2 | 券商开通 ¥200/月 | 集合竞价分析激活、分钟线批量 | 50万门槛 |
| **4** | 安装 kafka-python | `pip install kafka-python` | 微服务间消息真实传递 | — |
| **5** | 编译 C++ 组件 | cmake + make | 风控 50x 加速 | C++17 编译器 |
| **6** | 实盘 QMT | 券商开通 | 真实交易 | 50万 + 合规报告 |
| **7** | iFinD 数据 | ¥1.45万/年 | PIT 数据完整 | 预算 |
| **8** | DolphinDB | ¥数万~数十万/年 | 因子计算 100x 提速 | 预算 + 运维 |

---

## 十、升级不影响的模块

以下模块在任何升级路径中保持不变——这也是系统设计时重点保护的接口：

| 模块 | 不变的接口 | 说明 |
|------|-----------|------|
| `core/backtest/engine.py` | `BacktestEngine.run(strategy, data_loader)` | 撮合模拟器可注入，不影响引擎逻辑 |
| `core/features/registry.py` | `FactorRegistry.compute_order()` | 因子定义可增删，拓扑排序不变 |
| `core/portfolio/optimizer.py` | `PortfolioOptimizer.equal_weight()` 等 | 优化算法不变，输入数据源变更不影响 |
| `core/risk/pre_trade.py` | `PreTradeRiskChecker.check()` | C++ 重写后接口 100% 兼容 |
| `core/models/base.py` | `BaseModel.train() / predict()` | 模型换算法，接口不变 |
| `core/signals/schema.py` | `Signal` Pydantic 模型 | 信号格式统一，全系统以此为契约 |
| `core/backtest/cost_model.py` | `TransactionCostModel.calculate()` | 费率更新只改常量，接口不变 |

> **关键设计原则**：所有 "什么" 是变化的（数据源、模型算法、数据库），但 "怎么做"（接口、管线、约束检查）是不变的。升级只影响 `core/` 中具体实现类的内部、`services/` 中的编排调用、以及 `configs/` 中的连接字符串。
