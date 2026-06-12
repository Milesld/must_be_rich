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

### 1. 初始化（首次，只需要一次）

```bash
./scripts/setup.sh
```

### 2. 跑一次回测

```bash
python research/run_backtest_demo.py
```

输出示例：
```
============================================================
回测结果
============================================================
  period........................ 2024-01-02 ~ 2025-12-31
  trading_days.................. 470
  initial_capital............... 1000000
  final_nav..................... 1156247.32
  total_return_pct.............. 15.6247
  annual_return_pct............. 7.8124
  annual_volatility_pct......... 18.5231
  sharpe_ratio.................. 0.3142
  max_drawdown_pct.............. -12.3456
  calmar_ratio.................. 0.6331
  win_rate_pct.................. 52.38
  total_cost.................... 12530.45
============================================================
```

### 3. 运行测试

```bash
python -m pytest tests/ -q    # 全部测试（248个）
```

### 4. 清理数据

```bash
./scripts/clean_data.sh
```

---

## 使用指南

### 配置驱动模式

所有参数集中在 `configs/strategy.yaml`，修改后重跑即可生效：

```bash
python research/run_backtest_demo.py                # 默认配置
python research/run_backtest_demo.py configs/my.yaml # 自定义配置
```

### 三种用法（按复杂度递进）

```bash
# ① 手工调参 — 改 YAML → 跑回测
vim configs/strategy.yaml
python research/run_backtest_demo.py

# ② 自动搜索 — 逐个加因子看增量效果
python research/factor_optimizer.py --task long_term --rounds 100

# ③ 安装 optuna 后性能更强（TPE自适应搜索，收敛快3~10倍）
pip install optuna
python research/factor_optimizer.py --task long_term --rounds 200
```

### 不同任务如何选因子

不同任务的收益驱动因素不同，应使用不同的因子池：

| 任务 | 命令 | 因子池 | 说明 |
|------|------|--------|------|
| **长期选股** | `--task long_term` | 30个（基本面+长动量+质量） | 月频、低噪声，估值/成长/ROE预测力最强 |
| **盘前推荐** | `--task premarket` | 16个（隔夜信号+竞价+NLP） | 日频、竞价和隔夜海外是增量信息 |
| **日内预测** | `--task intraday` | 19个（盘口+短动量+量价） | 分钟级、量比/RSI/布林带最敏感 |
| 不限任务 | 不传 `--task` | 全部技术面 | 探索性质 |

**原则**：不要把高频因子用于月频策略（是噪声），也不要把基本面用于分钟策略（没变化）。

### 当前 demo 启用的因子（2个，可通过 optimizer 自动寻找更好的组合）

| 因子 | 代码位置 | 计算方式 |
|------|---------|---------|
| `momentum_20d` | `_build_feature_loader()` | 20日累计收益率 |
| `volatility_20d` | `_build_feature_loader()` | 20日年化波动率 |

### 系统中已实现但 demo 未启用的因子（56个）

#### 技术面因子（14个）— `core/features/technical.py`

| 因子 | 函数名 | 参数 | 说明 |
|------|--------|------|------|
| N日动量 | `momentum()` | `{window: 20}` | 累计收益率 |
| Alpha动量 | `alpha_momentum()` | `{window: 20}` | 剔除市场beta的纯动量 |
| 年化波动率 | `volatility()` | `{window: 20}` | 日收益年化标准差 |
| 振幅 | `amplitude()` | `{window: 20}` | (高-低)/昨收 的均值 |
| ATR | `atr()` | `{window: 14}` | 平均真实波幅 |
| 换手率 | `turnover()` | `{window: 5}` | N日平均换手率 |
| 量比 | `volume_ratio()` | `{window: 20}` | 当日量/20日均量 |
| Amihud非流动 | `amihud_illiq()` | `{window: 20}` | \|收益率\|/成交额 |
| RSI | `rsi()` | `{window: 14}` | 相对强弱指标 |
| MACD(DIF) | `macd_dif()` | `{fast:12, slow:26, signal:9}` | 快慢线差值 |
| MACD(DEA) | `macd_dea()` | 同上 | 信号线 |
| MACD(柱) | `macd_hist()` | 同上 | 柱状图 |
| 布林带位置 | `bollinger_position()` | `{window:20, num_std:2}` | 0=下轨, 1=上轨 |
| 均线排列 | `ma_alignment()` | `{short:5, mid:20, long:60}` | +1多头, -1空头 |

#### 基本面因子（17个）— `core/features/fundamental.py`

> ★全部支持 Point-in-Time 模式（防前视偏差）

| 因子 | 函数名 | 说明 |
|------|--------|------|
| 市盈率 | `pe_ttm()` | PE(TTM) |
| 市净率 | `pb()` | PB |
| 市销率 | `ps_ttm()` | PS(TTM) |
| 市现率 | `pcf_ttm()` | PCF |
| PEG | `peg()` | PE/盈利增速 |
| 股息率 | `dividend_yield()` | 每股股息/股价 |
| 盈利收益率 | `ep_ttm()` | 1/PE |
| ROE | `roe_ttm()` | 净资产收益率 |
| ROA | `roa_ttm()` | 总资产收益率 |
| ROIC | `roic_ttm()` | 投入资本回报率 |
| 毛利率趋势 | `gross_margin_trend()` | 近4季度斜率 |
| 净利率 | `net_margin_ttm()` | 净利率 |
| 营收增速 | `revenue_yoy()` | 同比 |
| 净利增速 | `net_profit_yoy()` | 同比 |
| 负债率 | `debt_ratio()` | 资产负债率 |
| 流动比率 | `current_ratio()` | — |
| 速动比率 | `quick_ratio()` | — |
| 现金流比 | `cf_ratio_ttm()` | 经营CF/净利润 |
| 自由现金流收益 | `free_cf_yield()` | FCF/市值 |

#### 资金面因子（7个）— `core/features/capital_flow.py`

| 因子 | 函数名 | 说明 |
|------|--------|------|
| 主力净流入 | `main_force_net_inflow()` | N日累计大单净流入 |
| 主力参与度 | `main_force_inflow_ratio()` | 大单净流入/成交额 |
| 融资余额变化 | `margin_balance_change()` | N日变化率 |
| 龙虎榜净买 | `dragon_tiger_net_buy()` | 机构净买排名分位 |
| 机构席位 | `dragon_tiger_institution_count()` | 机构出现次数 |
| 北向季度变化 | `northbound_quarter_change()` | ★仅季度频率 |
| 限售解禁天数 | `lockup_expiry_days()` | 距下次解禁天数 |

#### 情绪/事件因子（8个）— `core/features/sentiment.py`

| 因子 | 函数名 | 说明 |
|------|--------|------|
| 涨停家数 | `limit_up_count()` | 全市场统计 |
| 连板高度 | `limit_up_chain_height()` | 最高连板数 |
| 跌停家数 | `limit_down_count()` | 全市场统计 |
| 炸板率 | `board_break_ratio()` | 曾涨停后开板比例 |
| 赚钱效应 | `limit_up_ratio()` | 涨停数/上涨数 |
| 竞价溢价 | `auction_open_premium()` | 虚拟开盘vs昨收 |
| 竞价比 | `auction_volume_ratio()` | 竞价量/20日均量 |
| 业绩预告 | `performance_forecast_surprise()` | 大增→+2, 首亏→-3 |

#### 盘前专属因子（10个）— `core/features/premarket.py`

| 因子 | 函数名 | 说明 |
|------|--------|------|
| 隔夜中概映射 | `overnight_adr_mapped()` | 美股中概→A股方向 |
| A50期货 | `a50_futures_overnight()` | 隔夜涨跌幅 |
| 恒指期货 | `hsi_futures_overnight()` | 隔夜涨跌幅 |
| 公告情绪 | `announcement_sentiment_score()` | 关键词规则(可升级NLP) |
| 龙虎榜复盘 | `dragon_tiger_review_score()` | 综合评分 |
| 涨停复盘 | `limit_up_review_signal()` | 昨涨停→今高开概率 |
| 题材热度 | `theme_heat_score()` | 概念提及频率 |
| 竞价强度 | `auction_strength_score()` | 偏离+量比+不平衡 |
| 虚假申报 | `auction_fake_order_risk()` | 9:15-9:20大单后撤 |
| 事件天数 | `days_to_next_event()` | 距下次重大事件 |

---

### 全部可调参数

| 参数 | 文件 | 行位置（约） | 默认值 | 调整方式 |
|------|------|------------|--------|---------|
| **持仓数量** | `research/run_backtest_demo.py` | `MomentumLowVolStrategy(top_n=5)` | 5 | 改数字，如 `top_n=10` |
| **动量权重** | 同上 | `_score_stocks()` 中 `mom_rank * 0.6` | 0.6 | 改权重，和=1.0 |
| **低波动权重** | 同上 | 同上 `inv_vol_rank * 0.4` | 0.4 | 同上 |
| **回测起始日** | 同上 | `main()` 中 `start_date` | 2024-01-02 | 改日期 |
| **回测截止日** | 同上 | `main()` 中 `end_date` | 2025-12-31 | 改日期 |
| **初始资金** | 同上 | `BacktestEngine(initial_capital=1_000_000)` | 100万 | 改数字 |
| **股票池** | 同上 | `DEMO_CODES` 列表 | 80只 | 增删代码 |
| **股票数量** | 同上 | `codes[:80]` | 80 | 改数字，如 `codes[:200]` |
| **调仓频率** | 同上 | `_is_first_trading_day_of_month()` | 每月初 | 改为每周/每季 |
| **佣金率** | `configs/system.yaml` | `default_commission_rate` | 0.00015(万1.5) | 改YAML |
| **印花税率** | 同上 | `stamp_duty_rate` | 0.0005 | 改YAML |
| **单票上限** | `configs/risk/thresholds.yaml` | `single_stock_max_ratio` | 0.20(20%) | 改YAML |
| **行业上限** | 同上 | `industry_max_ratio` | 0.40(40%) | 改YAML |
| **回撤熔断线** | 同上 | `drawdown_daily_circuit` | 0.15 | 改YAML |
| **波动率目标** | 同上 | `vol_target` | 0.15 | 改YAML |
| **止损ATR倍数** | `core/portfolio/exit_monitor.py` | `stop_loss_atr_mult` | 2.0 | 改常量 |
| **止损比例** | 同上 | `stop_loss_pct` | 0.08 | 改常量 |
| **止盈目标** | 同上 | `take_profit_target` | 0.20 | 改常量 |
| **移动止盈回撤** | 同上 | `trailing_stop_pct` | 0.10 | 改常量 |
| **时间止损天数** | 同上 | `max_hold_days` | 60 | 改常量 |
| **因子计算窗口** | `core/features/technical.py` | 各函数 `params` | 各不同 | 改字典值 |

---

### 如何启用更多因子

以在 demo 中加入 `rsi_14`（14日 RSI）为例，只需两步：

**第1步**：在 `_build_feature_loader()` 里加一行计算逻辑：

```python
# 在 research/run_backtest_demo.py 的 _build_feature_loader() 中
# 找到 results.append({...}) 那几行，在 momentum_20d 下面加：

"rsi_14": float(_calc_rsi(arr, 14)) if len(arr) >= 15 else 50.0,
```

**第2步**：在 `_score_stocks()` 里把这个新因子纳入综合评分：

```python
# 在 _score_stocks() 中加：
if "rsi_14" in features.columns:
    rsi = features["rsi_14"].dropna()
    # RSI 30以下=超卖(买入信号)，70以上=超买(卖出信号)
    # 归一化到 0~1：RSI越低分越高
    rsi_score = (70 - rsi.clip(30, 70)) / 40
    scores.loc[rsi_score.index] = scores.loc[rsi_score.index] + rsi_score * 0.2
```

然后用 `python research/run_backtest_demo.py` 重跑看效果。每个因子都这样两步接入：**1. 计算** → **2. 融入评分**。

---

### 组合优化器（5种加权策略）

在 `core/portfolio/optimizer.py` 中已实现，demo 用的是最简单的「等权」。如果需要更复杂的仓位分配：

| 策略 | 方法 | 何时用 | 参数 |
|------|------|--------|------|
| 等权 | `equal_weight()` | 基线，最稳健 | `top_n` |
| 信号强度加权 | `signal_weighted()` | 模型置信度驱动 | `max_single` |
| 波动率倒数加权 | `inverse_vol_weight()` | 降低波动最有效 | `top_n` |
| 风险平价 | `risk_parity()` | 风险贡献均衡 | `top_n` |
| 最小CVaR | `min_cvar()` | 尾部风险最小化 | `top_n, alpha` |

### 数据源切换

| 场景 | 修改方式 |
|------|---------|
| 当前 | 新浪接口（`stock_zh_a_daily`），通过 `finance.sina.com.cn` |
| 切换到东方财富 | 把 `_fetch()` 里的 `stock_zh_a_daily` 改成 `stock_zh_a_hist` |
| 添加 Tushare 备用 | 设置环境变量 `TUSHARE_TOKEN`，在 `FallbackDataSource` 中注册 |
| 添加更多股票 | 修改 `DEMO_CODES` 列表，可扩展到 200、500、1000 只 |

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
│   └── run_backtest_demo.py  #   完整回测示例（含所有可调参数）
└── docs/                     # 文档
    ├── A股量化交易系统详细方案.md    # 系统级方案（做什么、为什么）
    ├── 系统实现设计指南.md           # 工程实现指南（代码结构/表设计/API契约）
    ├── OPERATIONS.md                 # 操作指南
    ├── UPGRADE.md                    # 升级指南（7个可替换部件）
    ├── runbook.md                    # 生产运维手册
    ├── prompts.md                    # 8批次开发prompt全集
    └── GITHUB_SETUP.md               # GitHub上传指南
```

---

## 文档导航

| 文档 | 阅读对象 | 内容 |
|------|---------|------|
| **README.md（本文）** | **所有人** | 快速开始、因子清单、可调参数、使用示例 |
| `docs/A股量化交易系统详细方案.md` | 量化研究员 | 方法论、因子体系、合规要求、风控策略 |
| `docs/系统实现设计指南.md` | 开发工程师 | 代码结构、数据库schema、API契约、配置规范、测试策略 |
| `docs/OPERATIONS.md` | 运维 | 怎么启动、怎么清理、脚本参考 |
| `docs/UPGRADE.md` | 架构师 | 6个可替换部件的升级路径 |
| `docs/runbook.md` | SRE | 故障排查、降级操作、检查清单 |

---

## 许可证

MIT License

---

> **一句话总纲**：风险最小化的发力点 80% 在组合结构与相关性；三套系统模型解耦、数据共享、资金硬隔离，时间维度层层聚焦；一切权重和退出规则都要能扛住 A股的 T+1 与涨跌停才算数；过拟合与情绪化是最大的两个敌人。
