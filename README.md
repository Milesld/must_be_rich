# A股量化交易系统

基于 **长期选股 + 盘前推荐 + 日内预测** 三子系统分层聚焦的量化交易平台。

> ⚠️ **风险声明**：本文不构成任何投资建议。量化交易存在本金损失风险，实盘前必须经过充分回测与模拟验证。没有任何系统能稳定准确地预测股价。

---

## 系统架构

```
长期选股（月度，全市场）
    │ 底池筛选 + 方向约束
    ▼
盘前推荐（日频，~30-80只精选）
    │ 关注清单 + 入场区间
    ▼
日内预测（分钟，~5-15只高频跟踪）
```

三子系统在时间维度上层层聚焦：数据层共享、模型层独立、资金层硬隔离。

### 技术栈

| 组件 | 方案 |
|------|------|
| 语言 | Python 3.12（策略/研究）+ C++17（风控/执行） |
| ML模型 | LightGBM（表格排序分类）+ CatBoost + Qwen3（NLP情绪） |
| 数据源 | AkShare（免费主源）/ Tushare（备用）/ QMT L2（升级路径） |
| 消息中间件 | Redpanda（Kafka兼容，开发环境降级为日志模式） |
| 部署 | Docker Compose（9个微服务）+ Prometheus + Grafana |

---

## 快速开始

### 1. 初始化

```bash
./scripts/setup.sh
```

### 2. 研究模式 — 直接跑回测

```bash
python research/run_backtest_demo.py
```

输出：年化收益、夏普比率、最大回撤、月度热力图。

### 3. 实盘模式 — Docker 一键启动

```bash
docker-compose up -d
./scripts/check_health.sh
```

### 4. 清理

```bash
./scripts/clean_data.sh
```

---

## 项目结构

```
quant-system/
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── configs/                  # YAML 配置（板块规则/模型超参/因子/风控阈值）
├── core/                     # 核心库（纯Python，被所有服务依赖）
│   ├── common/               #   交易日历/复权/配置/日志
│   ├── data/                 #   数据源适配器/管线/质量检查
│   ├── features/             #   因子计算（56个因子/5类）
│   ├── models/               #   LightGBM模型/NLP情绪/评估
│   ├── backtest/             #   事件驱动回测引擎
│   ├── portfolio/            #   组合优化/凯利/退出监控
│   ├── risk/                 #   事前风控/组合风控/极端预案
│   └── signals/              #   信号格式/总线
├── services/                 # 微服务（编排调度层）
│   ├── data_collector/       #   定时数据采集
│   ├── feature_server/       #   因子计算服务
│   ├── long_term_service/    #   长期选股（月度调仓）
│   ├── premarket_service/    #   盘前推荐（08:00触发）
│   ├── intraday_service/     #   日内预测（09:30-15:00）
│   ├── risk_engine/          #   风控引擎（<10ms）
│   ├── order_manager/        #   订单管理（唯一可调用券商API）
│   ├── nlp_service/          #   NLP情绪分析（Qwen3）
│   └── monitor/              #   监控告警（Prometheus metrics）
├── cpp/                      # C++17 高性能组件
│   ├── risk_checker/         #   事前风控（P99<100μs）
│   ├── tick_processor/       #   Tick行情解码（FAST/二进制）
│   └── order_router/         #   订单路由（QMT封装）
├── tests/                    # 单元测试/集成测试/回归测试
│   ├── unit/
│   ├── integration/
│   └── regression/
├── scripts/                  # 运维脚本
│   ├── setup.sh              #   首次初始化
│   ├── daily_run.sh          #   每日自动运行
│   ├── check_health.sh       #   健康检查
│   ├── backup_db.sh          #   数据库备份
│   └── clean_data.sh         #   清理数据
├── research/                 # 研究笔记本和示例
│   └── run_backtest_demo.py  #   完整回测示例
└── docs/                     # 文档
    ├── A股量化交易系统详细方案.md    # 系统级方案（做什么、为什么）
    ├── 系统实现设计指南.md           # 工程实现指南（代码结构/表设计/API契约）
    ├── OPERATIONS.md                 # 操作指南（怎么启动/运行/清理）
    ├── UPGRADE.md                    # 升级指南（7个可替换部件）
    ├── runbook.md                    # 生产运维手册（故障排查/降级方案）
    ├── prompts.md                    # 8批次开发prompt全集
    └── GITHUB_SETUP.md               # GitHub上传指南
```

---

## 文档导航

| 文档 | 阅读对象 | 内容 |
|------|---------|------|
| `docs/OPERATIONS.md` | **所有人** | 怎么启动、怎么跑回测、怎么清理 |
| `docs/A股量化交易系统详细方案.md` | 量化研究员 | 方法论、因子体系、合规要求、风控策略 |
| `docs/系统实现设计指南.md` | 开发工程师 | 代码结构、数据库schema、API契约、配置规范、测试策略 |
| `docs/UPGRADE.md` | 架构师/高级开发 | 6个可替换部件的升级路径和具体步骤 |
| `docs/runbook.md` | 运维/SRE | 故障排查、降级操作、检查清单 |
| `docs/prompts.md` | 开发者 | 8批次开发的完整prompt（用于AI辅助开发） |

---

## 测试

```bash
python -m pytest tests/ -v    # 全部测试（248个）
python -m pytest tests/unit/  # 仅单元测试
```

---

## 许可证

MIT License

---

> **一句话总纲**：风险最小化的发力点 80% 在组合结构与相关性；三套系统模型解耦、数据共享、资金硬隔离，时间维度层层聚焦；一切权重和退出规则都要能扛住 A股的 T+1 与涨跌停才算数；过拟合与情绪化是最大的两个敌人。
