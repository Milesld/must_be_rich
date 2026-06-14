# 系统操作指南

> 本文回答"怎么用这个系统"的所有问题——从首次初始化到日常运行再到清理。

---

## 一、脚本清单

| 脚本 | 功能 | 需要网络 | 需要 Docker |
|------|------|---------|------------|
| `scripts/setup.sh` | 首次初始化：检查依赖、创建目录、加载交易日历、跑测试 | ✗（已缓存） | ✗ |
| `scripts/init_calendar.py` | 独立初始化交易日历（setup.sh 内部调用） | ✗（已缓存） | ✗ |
| `scripts/init_db.py` | 创建 ClickHouse + MySQL 表结构 | ✗ | ✓ |
| `scripts/daily_run.sh` | 研究模式全天自动运行（06:00→16:00） | ✓ | ✗ |
| `scripts/check_health.sh` | 检查微服务 /health 端点 | ✗ | ✓ |
| `scripts/backup_db.sh` | 备份 MySQL + ClickHouse schema + Redis RDB | ✗ | ✓ |
| `scripts/clean_data.sh` | 清理运行时数据，回到初始状态 | ✗ | ✗ |
| `research/run_backtest_demo.py` | 跑单次回测（configs/strategy.yaml 驱动） | ✗（已缓存） | ✗ |
| `research/factor_optimizer.py` | TPE 自动搜索最优因子组合 | ✗（已缓存） | ✗ |

---

## 二、首次初始化

```bash
# 一条命令完成全部初始化
./scripts/setup.sh
```

这个脚本做了：
1. 检查 Python 依赖是否齐全
2. 检查 lightgbm 是否安装
3. 创建 `data/` 下所有运行时目录
4. 从 akshare 拉取交易日历（缓存到 `~/.quant_system/calendar/`）
5. 运行全部单元测试验证系统正常

初始化完成后你就可以直接跑回测了：

```bash
python research/run_backtest_demo.py
```

---

## 三、两种运行模式

### 模式A：研究模式（不需要 Docker，直接从 Python 跑）

适合：因子研究、策略回测、模型训练、盘前推荐验证。

```bash
# 跑一次回测（用 configs/strategy.yaml 中的配置）
python research/run_backtest_demo.py

# 因子自动优化 — TPE 自适应搜索最优因子组合和权重
# 候选池自动排除无数据源的因子（基本面/资金流/竞价等需要额外数据的不会入选）
python research/factor_optimizer.py --task long_term --rounds 200
python research/factor_optimizer.py --task premarket --rounds 100
python research/factor_optimizer.py --task intraday --rounds 100

# 加速：短窗口搜索 + 全区间验证 Top-3
python research/factor_optimizer.py --task long_term --search-years 0.5 --rounds 100

# 限制因子数防止过拟合
python research/factor_optimizer.py --task long_term --min-factors 3 --max-factors 5
```

**数据流向**：akshare 在线拉取 → 内存中 → 直接回测/模型推理。数据不会写入数据库。

**需要做什么**：只需要 `python` + `pip install` 几个核心依赖。不需要 Docker、不需要 MySQL、不需要 Redis。

### 模式B：实盘模式（Docker Compose 一键启动全部微服务）

适合：真实交易日的全自动化运行。

> ⚠ **实盘模式当前为架构就绪状态**：8 个微服务的代码已完成，Docker 编排已就绪，但两个关键依赖需要你自行配置——① 券商 QMT 账户（50万资金门槛）② 消息真实传递（需 `kafka-python` 库）。研究模式可直接使用，不需要以下步骤。

#### 第1步：确认环境

```bash
# 确保 Docker 已安装
docker --version

# 确保 .env 文件存在（首次从模板复制）
cp .env.example .env
```

#### 第2步：启动基础设施（首次需等镜像拉取）

```bash
docker-compose up -d
# 启动后包含：Redpanda(消息总线) + ClickHouse(行情) + MySQL(业务) + Redis(缓存)
```

#### 第3步：初始化数据库表

```bash
python scripts/init_db.py
```

#### 第4步：启动微服务（逐项检查）

```bash
# 启动所有微服务
docker-compose up -d
./scripts/check_health.sh

# 各服务端口和用途：
#   feature_server :50051  — 因子实时查询
#   risk_engine    :50052  — 事前风控检查 + 熔断
#   nlp_service    :50053  — 公告 NLP 情绪分析
#   monitor        :9090   — Prometheus /metrics + 日报
```

#### 第5步：验证服务正常

```bash
docker-compose logs data_collector | tail -20       # 确认数据采集正常
docker-compose logs premarket_service | tail -20     # 确认盘前推荐正常
docker-compose logs intraday_service | tail -20      # 确认日内服务就绪
docker-compose logs risk_engine | tail -10            # 确认风控引擎就绪
```

#### 第6步：渐进式投入实盘

```
阶段1（模拟模式）：
  QMT_ENABLED=false   ← .env 中保持关闭
  所有信号仅在日志中输出，不会真实下单
  运行 2 周，观察信号质量、数据到达率、系统稳定性

阶段2（模拟盘验证）：
  QMT_ENABLED=true    ← .env 中开启
  券商开通 QMT 模拟账户
  系统连接到 QMT 仿真撮合环境
  运行 2 周，验证下单→成交回报→持仓更新的全链路

阶段3（小资金实盘）：
  使用真实券商账户，总资金的 5-10%
  继续运行 2-4 周
  确认真实交易成本、滑点、心理压力在可承受范围内

阶段4（正常资金）：
  逐步放大到目标仓位
```

**不跳步**：每个阶段至少持续 2 周。回测漂亮 ≠ 实盘赚钱，模拟盘赚钱 ≠ 实盘扛得住——沉没成本心理和亏损恐惧只有真实资金才能检验。

#### 实盘系统如何工作（端到端）

每个交易日的时间线：

```
┌─ 06:00 ─────────────────────────────────────────────┐
│ data_collector 拉取隔夜数据                           │
│   · 美股指数、A50期货、中概股涨跌幅                   │
│   · 商品/汇率/公告                                    │
│   → 存入 ClickHouse，心跳上报 monitor                 │
├─ 08:00 ─────────────────────────────────────────────┤
│ premarket_service 盘前推荐流水线                      │
│   · 加载长期评分（从 Redis）                          │
│   · 计算盘前因子（隔夜映射/A50/竞价等）                │
│   · NLP 分析隔夜公告情绪（调用 nlp_service）           │
│   · 跑模型推理（FusionRanker）→ 生成今日推荐清单      │
│   → 推荐清单 JSON 写入 signals.premarket topic         │
│   · 硬截止 08:45，超时丢弃未完成任务                   │
├─ 09:15 ─────────────────────────────────────────────┤
│ premarket_service 竞价分析引擎                        │
│   · 9:15-9:20 申报量监控（仅标记，不决策）             │
│   · 9:20-9:25 真实意图判断                            │
│   · 竞价异常检测 → 修正推荐置信度                      │
│   → 09:25 发送 signals.premarket (final=true)         │
├─ 09:30 ─────────────────────────────────────────────┤
│ intraday_service 日内预测                             │
│   · 加载盘前推荐清单作为关注池                         │
│   · 每分钟对关注池股票计算实时特征                     │
│   · LightGBM 三分类推理 → 涨/跌/平概率                 │
│   · 阈值判断 → 生成买入/卖出信号                       │
│   → 信号发送到 signals.intraday topic                 │
├─ 09:30-15:00（盘中持续）────────────────────────────│
│ order_manager 实时消费信号                            │
│   每收到一条信号：                                     │
│     ① signal_id 幂等去重                              │
│     ② 同步调用 risk_engine.CheckOrder()（<10ms）       │
│        · 单票上限≤20%、行业≤40%                        │
│        · 日申报<20000笔、秒申报<300笔                   │
│        · ST/停牌/黑名单过滤                            │
│        · 可用资金检查                                  │
│     ③ 风控不通过 → 记录 risk_log → 丢弃信号           │
│     ④ 风控通过 → 整手化 → 调用 QMT SDK 下单            │
│     ⑤ 监听成交回报 → 更新持仓 → 记录 order_log        │
│     ⑥ 下单超时 → 最多重试 1 次（重试前先查订单状态）   │
│                                                        │
│ risk_engine 实时监控                                   │
│   · 每笔成交 → 更新净值 → 检查回撤熔断线                │
│   · 回撤触发 → signals.exit (emergency_flat)            │
│   · 拒绝率突增 → risk.alert → 钉钉/微信通知            │
├─ 14:57-15:00 ───────────────────────────────────────┤
│ intraday_service 收盘集合竞价阶段                      │
│   · 暂停新信号生成，单独处理竞价跳变风险                │
├─ 15:00 ─────────────────────────────────────────────┤
│ data_collector 日终数据采集                            │
│   · 日K线入库、龙虎榜、融资融券                         │
│   · 运行数据质量检查                                   │
│ monitor 生成日报                                       │
│   · 今日信号汇总、成交统计、持仓快照                    │
│   · 回撤监控、合规检查                                 │
├─ 15:30 ─────────────────────────────────────────────┤
│ feature_server 日终批量因子重算                        │
│   · 全部日频因子重算 → 写入 ClickHouse + Redis         │
│                                                       │
│ （如果是月末最后一个交易日 → long_term_service 月度调仓）│
└──────────────────────────────────────────────────────┘
```

#### 信号链路

```
策略引擎(intraday_service/premarket_service/long_term_service)
    │ 信号: {signal_id, code, side, strength, confidence, ...}
    ▼
Redpanda topic (signals.intraday / signals.long_term / signals.premarket)
    │
    ▼
order_manager (唯一可调用券商 API 的服务)
    ├─ signal_id 去重
    ├─ gRPC → risk_engine.CheckOrder() → approved?
    ├─ 整手化 (ShareRounder)
    ├─ QMT SDK submit_order()
    ├─ 成交回报监听 → 更新持仓
    └─ 所有操作记录到 order_log + signal_log
```

#### 实盘前置条件清单

| 条件 | 说明 | 检查方式 |
|------|------|---------|
| Docker 已安装 | 启动基础设施 | `docker --version` |
| `kafka-python` 已安装 | 消息总线真实传递（否则日志模式） | `python -c "import kafka"` |
| QMT 账户已开通 | 50 万门槛，风险测评 C4+ | 券商确认 |
| QMT SDK 已安装 | xtquant Python 包 | `python -c "import xtquant"` |
| `.env` 已配置 | QMT_ENABLED=true, TUSHARE_TOKEN, 数据库连接 | `cat .env` |
| 程序化交易已报告 | 2025 年 7 月起新规 | 向券商提交报告 |

#### 实盘模式与研究模式的差异

| 维度 | 研究模式 | 实盘模式 |
|------|---------|---------|
| 数据存储 | 内存中，跑完即释放 | ClickHouse 持久化 |
| 信号处理 | 回测引擎模拟撮合 | QMT 真实下单 |
| 消息传递 | 直接 Python 调用 | Redpanda 异步 topic |
| 风控检查 | 回测中模拟 | C++ 独立进程同步调用 |
| 配置文件 | 直接改 YAML | YAML + 环境变量 (.env) |
| 启动方式 | `python research/run_backtest_demo.py` | `docker-compose up -d` |
| 适用场景 | 策略研发、因子验证 | 真实交易日自动化 |

---

## 三-B、三个子系统在实盘中的使用

系统有三个独立的子系统，按时间维度分工。每个可以单独运行，也可以组合运行。

```
时间轴：
  每月初盘后  →  long_term_service 月度调仓
  每日 08:00  →  premarket_service 盘前推荐
  每日 09:30  →  intraday_service  盘中预测
```

### 子系统1：长期股票推荐（月度调仓）

**运行方式**：`long_term_service` 是 Docker 容器，由 APScheduler 定时触发。

**触发时机**：
- 每月 25-31 日 16:00（月末盘后）：自动跑月度调仓，生成下月持仓清单
- 每月 1-7 日 20:00（月初盘后）：自动重训练 LightGBM 模型
- 每周五 17:00：生成周度持仓报告

**输出的结果**：
- 调仓信号 → `signals.long_term` topic（{code, weight, action: "rebalance"}）
- 模型文件 → `data/models/long_term_ranker_<日期>.joblib`
- 周报 → `data/reports/weekly_report_<日期>.md`

**你如何查看**：
```bash
# 查看最新长期评分（Redis）
docker-compose exec redis redis-cli KEYS "feat:long_term_score:*" | head -10

# 查看调仓信号日志
docker-compose logs long_term_service | grep "月度调仓"

# 查看周报
cat data/reports/weekly_report_*.md | tail -50
```

**配置文件**：`configs/strategy.yaml` 中的 `factors`（19 个长期因子可选）和 `strategy.rebalance_frequency: monthly`。

**研究模式等价操作**：
```bash
# 手动跑一次月度调仓
python -m services.long_term_service.main
```

---

### 子系统2：盘前推荐（每日 08:00）

**运行方式**：`premarket_service` 是 Docker 容器，通过循环睡眠检测到 08:00 自动触发。

**触发时机**：
- 08:00-08:45：主流水线（长期评分 + 隔夜数据 + NLP 公告 + 模型推理 → 推荐清单）
- 08:45：硬截止时间，超时丢弃未完成任务
- 09:15-09:25：竞价修正阶段，实时检测异常并调整置信度
- 09:25：发送最终推荐（`signals.premarket`，`final=true`）

**输出的结果**：
```json
{
  "date": "2026-06-13",
  "generated_at": "08:45:00",
  "market_regime": "震荡偏多",
  "recommendations": [
    {"code": "600519", "rank": 1, "composite_score": 0.81, "expected_direction": "up", ...},
    {"code": "300750", "rank": 2, "composite_score": 0.76, "expected_direction": "up", ...},
    ...
  ]
}
```

**你如何查看**：
```bash
# 查看今日推荐清单
cat data/premarket_recommendations/premarket_$(date +%Y-%m-%d).json | python -m json.tool | head -80

# 实时跟踪盘前服务日志
docker-compose logs -f premarket_service

# 手动触发一次盘前推荐（测试用）
python -m services.premarket_service.main
```

**配置文件**：`configs/strategy.yaml` 中的 `factors`（16 个盘前因子可选）和 `data_source.codes`（117 只股票池）。

**研究模式等价操作**：
```bash
# 直接在本地跑盘前推荐，输出到 data/premarket_recommendations/
python -m services.premarket_service.main
```

---

### 子系统3：盘中推荐（09:30-15:00 实时）

**运行方式**：`intraday_service` 是 Docker 容器，09:25 加载盘前推荐清单作为关注池，09:30-15:00 每分钟对关注池股票做实时因子计算和模型推理。

**触发时机**：
- 09:25：加载今日盘前推荐清单（从 `data/premarket_recommendations/` 读取）
- 09:30~14:57：每分钟对关注池股票：算实时特征 → LightGBM 三分类（涨/跌/平）→ 阈值判断 → 生成买卖信号
- 14:57-15:00：收盘集合竞价单独处理，暂停新信号
- 15:00：生成当日信号汇总

**输出的结果**：
- 买卖信号 → `signals.intraday` topic（{code, direction: up/down, strength, confidence, ...}）
- 信号被 `order_manager` 消费 → 风控检查 → QMT 下单

**你如何查看**：
```bash
# 跟踪日内信号生成
docker-compose logs -f intraday_service

# 查看当日产生的信号数量
docker-compose logs intraday_service --since $(date +%Y-%m-%d) | grep "signal" | wc -l
docker-compose logs intraday_service --since $(date +%Y-%m-%d) | grep "信号" | wc -l

# 启动日终信号汇总（收盘后手动触发）
docker-compose exec intraday_service python -m services.intraday_service.main --generate-summary
```

**配置文件**：`configs/strategy.yaml` 中的 `factors`（19 个日内因子可选）。

**研究模式等价操作**：
```bash
# 在本地跑日内服务（会循环等待盘中时段）
python -m services.intraday_service.main
```

---

### 三个子系统如何串联

```
long_term_service（每月）
    │ 长期评分写入 Redis
    │ 调仓信号 → signals.long_term → order_manager → QMT
    ▼
premarket_service（每日）
    │ 读取 Redis 中最新长期评分
    │ + 隔夜海外数据 + 盘前公告NLP
    │ → 生成今日推荐清单
    │ → signals.premarket
    ▼
intraday_service（盘中实时）
    │ 读取今日推荐清单作为关注池
    │ 每分钟实时特征 + 模型推理
    │ → 买卖信号 → signals.intraday
    ▼
order_manager
    消费 signals.intraday + signals.long_term
    → risk_engine.CheckOrder() 风控检查
    → QMT SDK 下单
```

**你可以只开启其中一两个**：

| 场景 | 需要开启的服务 | 配置 |
|------|-------------|------|
| 只要月度调仓 | `long_term_service` | `regime.enabled: true`（建议配市场状态判断） |
| 只要盘前推荐 | `premarket_service` | `data_source.codes` 设好股票池 |
| 只要日内预测 | `premarket_service` + `intraday_service` | 盘前后动一个，日内才有信号 |
| 全自动 | `docker-compose up -d`（三个都跑） | 不需要额外配置 |

**研究模式下独立跑**（不需要 Docker）：

```bash
# 长期选股 — 直接用 demo 跑回测，验证因子/参数
python research/run_backtest_demo.py

# 盘前推荐 — 手动触发一次，看今天的推荐清单
python -m services.premarket_service.main

# 日内预测 — 手动启动，盘中实时打印信号
python -m services.intraday_service.main
```

---

## 四、日常操作

### 某个交易日手动触发盘前推荐

```bash
docker-compose exec premarket_service python -m services.premarket_service.main
# 或本地直接跑：
python -m services.premarket_service.main
```

### 查看今天推荐了哪些股票

```bash
cat data/premarket_recommendations/premarket_$(date +%Y-%m-%d).json | python -m json.tool | head -40
```

### 查看日内信号日志

```bash
docker-compose logs intraday_service --tail 100
```

### 查看监控面板

浏览器打开 `http://localhost:9090/metrics` — Prometheus 格式的实时指标。

### 检查系统是否正常

```bash
./scripts/check_health.sh
```

输出示例：
```
=== 量化系统健康检查 (localhost) ===

  ✓ feature_server
  ✓ risk_engine
  ✓ nlp_service
  ✓ monitor

✓ 所有服务正常
```

### 每周日备份数据库

```bash
./scripts/backup_db.sh
```
备份存放在 `data/backups/<日期>/` 下，自动保留 30 天。

### "如何启用 DeepSeek API 做 NLP 情绪分析？"

系统默认使用关键词规则引擎（免费、零延迟、不需任何配置）。如果要用上大语言模型：

```bash
# 1. 获取 API key: https://platform.deepseek.com
# 2. 在 .env 中设置
echo 'DEEPSEEK_API_KEY=sk-your-key-here' >> .env
echo 'NLP_MODEL=deepseek' >> .env

# 3. 直接使用——所有调用方无需改代码
python -c "
from core.models.nlp import NLPSentimentAnalyzer
analyzer = NLPSentimentAnalyzer(model_name='deepseek')
result = analyzer.analyze_single('贵州茅台2025年净利润预增50%超预期')
print(f'情绪: {result.sentiment}, 置信度: {result.confidence:.0%}')
print(f'事件类型: {result.event_type}')
print(f'摘要: {result.summary}')
"
```

**三种引擎对比**：

| 维度 | 关键词规则 | DeepSeek API | 本地 Qwen3 |
|------|----------|-------------|-----------|
| 准确率 | 低（~60%） | 高（~85%） | 高（~84%） |
| 延迟 | <1ms | ~500ms | ~50ms(0.6B) |
| 成本 | 免费 | ¥0.001/条 | 免费（但需GPU） |
| 依赖 | 无 | 网络 + API key | transformers+torch+12GB+显存 |
| 适用场景 | 快速原型 | 盘前批量(100条/天) | 实时(<100ms) |

**切换引擎**：只需改环境变量或参数，所有调用方不受影响。

```

### "DeepSeek API 在盘前推荐中怎么用？"

盘前推荐服务 `premarket_service` 自动读取 `NLP_MODEL` 环境变量。设置后隔夜公告分析和盘前 `announcement_sentiment_score` 因子自动使用 DeepSeek：

```bash
# .env 中设置
NLP_MODEL=deepseek
DEEPSEEK_API_KEY=sk-xxx

# 重启盘前服务
docker-compose restart premarket_service
```

研究模式直接设置环境变量后运行：

```bash
DEEPSEEK_API_KEY=sk-xxx python research/run_backtest_demo.py
```

---

## 五、清理

### 场景1："回测跑乱了，想重新开始"

```bash
# 只清理回测缓存和训练出的模型，保留交易日历和市场数据
rm -rf data/checkpoints/ data/models/ data/logs/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

### 场景2："想彻底回到刚 git clone 的状态"

```bash
./scripts/clean_data.sh
```

这个脚本清理**全部**运行时数据（交易日历、检查点、模型、报告、日志、备份、Python缓存、本地 lightgbm），回到项目初始状态。

### 场景3："只想清理日志释放磁盘"

```bash
rm -rf data/logs/*.log
```

日志每天收盘自动滚动，30 天前的由 `daily_run.sh` 自动删除。

---

## 六、运行产生的数据都存在哪

| 位置 | 内容 | 产生者 | 清理方式 |
|------|------|--------|---------|
| `~/.quant_system/calendar/` | 交易日历 `.parquet` | 首次 `get_calendar()` | `rm -rf ~/.quant_system/` |
| `data/checkpoints/` | 数据管线 JSON | `DataPipeline` | `rm -rf data/checkpoints/` |
| `data/models/` | 训练好的 `.joblib` | `long_term_service` | `rm -rf data/models/` |
| `data/reports/` | 周报 `.md` | `long_term_service` / `monitor` | `rm -rf data/reports/` |
| `data/premarket_recommendations/` | 盘前推荐 `.json` | `premarket_service` | `rm -rf data/premarket_recommendations/` |
| `data/logs/` | 服务日志 | 所有微服务 | `rm -rf data/logs/` |
| `data/backups/` | DB备份 | `backup_db.sh` | `rm -rf data/backups/` |
| 项目目录 `lightgbm/` | 手动安装的 lightgbm | `pip install` 失败的替代方案 | `rm -rf lightgbm/ lightgbm-*.dist-info/` |

---

## 七、常见场景速查

### "我刚 clone 了这个项目，怎么开始？"

```bash
./scripts/setup.sh              # 一条命令初始化
python research/run_backtest_demo.py  # 跑个回测看看效果
```

### "我改了因子计算逻辑，想验证效果"

```bash
python -m pytest tests/unit/test_factors.py -v  # 跑因子测试
python research/run_backtest_demo.py              # 跑完整回测对比
```

### "我想在生产环境部署"

```bash
# 第1步：确认环境变量
cp .env.example .env
vim .env   # 填入 TUSHARE_TOKEN, QMT 配置等

# 第2步：启动基础设施
docker-compose up -d

# 第3步：初始化数据库表
python scripts/init_db.py

# 第4步：启动所有微服务（已在 docker-compose 中编排）
./scripts/check_health.sh

# 第5步：观察日志确认各服务正常
docker-compose logs premarket_service --tail 10
docker-compose logs intraday_service --tail 10

# 第6步：渐进验证（详见上文"模式B：实盘模式" → 第6步）
```

### "系统跑了几个月，磁盘满了"

```bash
du -sh data/*                    # 看哪个目录最大
rm -rf data/logs/*.log           # 日志通常是罪魁祸首
rm -rf data/backups/*/           # 或者清理旧备份
```

### "我换了电脑/重装了系统，lightgbm 怎么装？"

在不能直接 `pip install` 的环境（公司网络/代理限制）：

```bash
# 1. 用手机从 https://pypi.org/project/lightgbm/#files 下载 .whl 到 ~/Downloads/
#    macOS ARM64 选 lightgbm-4.x.x-py3-none-macosx_12_0_arm64.whl

# 2. 解压到项目目录
cd $TMPDIR
unzip -o ~/Downloads/lightgbm-4.*macosx*arm64.whl
cp -r $TMPDIR/lightgbm* /path/to/quant-system/
cd /path/to/quant-system

# 3. 修复动态库路径
install_name_tool -change @rpath/libomp.dylib \
  "$(python -c 'import sklearn; print(sklearn.__path__[0])')/.dylibs/libomp.dylib" \
  lightgbm/lib/lib_lightgbm.dylib

# 验证
python -c "import lightgbm; print(lightgbm.__version__)"
```

### "代理环境导致 akshare 拉数据卡住"

如果 Clash Verge/代理关闭后终端里还残留 `HTTP_PROXY` 等环境变量：

```bash
# 检查是否有残留代理变量
env | grep -i proxy

# 一次性清除
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

# 持久化（加到 ~/.zshrc 末尾，关闭 Clash 后自动清理）
pgrep -x "Clash Verge" >/dev/null 2>&1 || {
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
}
```

### "交易日历一直显示估算日历"

删除旧缓存让它重新拉取真实 A 股日历：

```bash
rm -f ~/.quant_system/calendar/trading_calendar.parquet
python -c "from core.common.calendar import get_calendar; cal=get_calendar(); print(f'{len(cal.all_trading_days)} 个交易日')"
```

### "如何管理多个策略配置？"

新建不同的 YAML 文件，互相隔离、互不影响：

```bash
cp configs/strategy.yaml configs/strategy_weekly.yaml       # 周度调仓版
cp configs/strategy.yaml configs/strategy_200stocks.yaml    # 扩股票池版
cp configs/strategy.yaml configs/strategy_value.yaml        # 纯价值因子版
cp configs/strategy.yaml configs/strategy_momentum.yaml     # 纯动量版
```

运行时指定对应配置文件：

```bash
python research/run_backtest_demo.py configs/strategy_weekly.yaml
python research/factor_optimizer.py --config configs/strategy_weekly.yaml --task long_term
```

### "跑 optimizer 太慢怎么办？"

三种加速方式，从最快到最准：

```bash
# 最快：只搜索最近半年，Top-3 自动全区间验证（约 2 分钟）
python research/factor_optimizer.py --task long_term --rounds 100 --search-years 0.5

# 中等：搜索最近 1 年（约 4 分钟）
python research/factor_optimizer.py --task long_term --rounds 200 --search-years 1

# 最准但不加速：全区间搜索（约 10 分钟）
python research/factor_optimizer.py --task long_term --rounds 200
```

工作原理：短窗口 TPE 搜索 → 找到 Top-3 → 用完整回测区间各验证一次，输出带 `✓ 全区间验证` 标记的才是真实指标。

### "优化器结果怎么看是不是过拟合？"

关注优化器输出的过拟合诊断：

- **夏普 > 2.0**：极高概率过拟合，实盘预计衰减 40-60%
- **夏普 > 1.5**：可能过拟合，建议换个时间段验证
- **因子数 ≥ 8 且交易日 < 1000**：因子偏多

最可靠的验证方式：换一个模型从未见过的回测区间（如 2019-2021），同样因子组合重跑，看夏普是否稳定。

### "我换股票池/换调仓频率，需要重新搜索因子吗？"

- 调仓频率变（月→周）：**需要重新搜索**。信号窗口变了，最优因子组合会不同
- 股票池微调（98→150 蓝筹）：**可以先不搜**，用现有组合跑一次看夏普变化
- 股票池大变（蓝筹→全市场 500 只）：**必须重新搜索**。加入了小盘股，流动性权重应该上调
- 两者同时改：**必须重新搜索**

### "不同板块应该用不同的因子吗？"

**是的。** 用同一套因子在全市场选股，TPE 找到的是在所有板块平均表现最好的因子组合，会牺牲板块特有的 Alpha。

举例：

```
白酒板块（强趋势、低波动）：动量因子 Rank IC = 0.08，非常有效
芯片板块（高波动、轮动快）：换手率+振幅更有效，动量反而可能是反向指标
银行板块（走势平缓、低增长）： 低波动+股息率更有效，动量几乎无效

统一因子：动量 71% + 换手率 20% → 在白酒上大赚、芯片上小赚、银行上完全无效
分板块因子：每板块独立搜索 → 白酒用动量、芯片用振幅、银行用低波动
```

**当前系统的实现方式**：为每个板块建一个独立配置文件，分别跑 optimizer 搜索最优因子。

```bash
# 为三个板块各建一个配置
# 1. 白酒/消费
cp configs/strategy.yaml configs/strategy_liquor.yaml
# 编辑 strategy_liquor.yaml：codes 只保留白酒/食品板块的 5 只
# 跑 optimizer：python research/factor_optimizer.py --config configs/strategy_liquor.yaml --task long_term --rounds 100

# 2. AI/半导体
cp configs/strategy.yaml configs/strategy_semiconductor.yaml
# 编辑 strategy_semiconductor.yaml：codes 只保留 AI 芯片到数据中心的 44 只
# 跑 optimizer

# 3. 银行/金融
cp configs/strategy.yaml configs/strategy_finance.yaml
# 同上
```

**预期结果**：三个板块的最优因子组合会完全不同。白酒可能仍然是动量+换手率，芯片可能变成振幅+量比+RSI，银行可能变成低波动+股息率+PE。

**在实盘中怎么组合**：长期选股服务同时运行三个板块的策略，各自选出 N 只，汇总后等权或按信号强度分配资金。

### "什么是 Walk-Forward 验证？"

模拟真实交易中"不知道未来"的验证方式：

```
第1轮: 训练(2023-01~2024-12) → 测试(2025 Q1)
第2轮: 训练(2023-04~2025-03) → 测试(2025 Q2)
第3轮: 训练(2023-07~2025-06) → 测试(2025 Q3)
...

每一轮只用"到今天为止"的数据选因子/训练，预测"接下来3个月"
如果多轮都稳 → 策略真的有效
如果只有某几轮好 → 过拟合
```

系统里 `core/backtest/walk_forward.py` 的 `WalkForwardValidator` 已实现此逻辑。

### "我想卸载，彻底删除"

```bash
./scripts/clean_data.sh --force  # 清理运行时数据
docker-compose down -v           # 停止并删除容器+数据卷
rm -rf ~/.quant_system/          # 交易日历缓存
cd .. && rm -rf quant-system/    # 删除项目本身
```
