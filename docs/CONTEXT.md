# 新对话上下文 — 请先阅读本文件

> **写给新 Claude Code 会话的第一条消息**：下面是一段可以直接粘贴到新对话中的提示词，它会引导新会话自动加载所有必要上下文。

---

## 使用方式

在新对话中粘贴以下完整提示词：

```
请先阅读以下文件来了解这个A股量化交易系统的全部上下文：

【核心必读】
1. README.md — 项目概述、因子清单、快速开始
2. docs/OPERATIONS.md — 完整操作指南（如何启动、回测、优化、实盘）
3. docs/CONTEXT.md（本文件）— 所有已做的工作、关键决策、已知问题

【按需阅读】
4. docs/A股量化交易系统详细方案.md — 系统级方案（方法论）
5. docs/系统实现设计指南.md — 工程实现指南（代码结构、表设计）
6. docs/UPGRADE.md — 升级路径（7个可替换部件）
7. docs/PRESENTATION.md — 汇报材料（PPT+易拉宝设计）

【关键技术选型】
8. docs/quant_tech_stack_report.md — 技术栈调研报告

【当前配置】
9. configs/strategy.yaml — 策略配置（因子、权重、股票池、回测区间）

【核心代码】
10. research/run_backtest_demo.py — 回测主入口
11. research/factor_optimizer.py — 因子优化器
12. core/backtest/engine.py — 回测引擎
13. core/models/nlp.py — NLP三引擎（关键词+DeepSeek API+本地Qwen3）
14. core/portfolio/regime.py — 市场状态判断

【测试验证】
15. 运行 python -m pytest tests/unit/ -q 确认系统正常

---

请先读完这些文件，然后告诉我：
1. 你理解这个系统做了什么事
2. 当前系统的状态是什么
3. 你觉得接下来可以做什么

这样我就能确认上下文已经衔接上了。
```

---

## 如果你是新的 Claude Code 会话，你应该知道以下信息

### 1. 项目状态

**已完成**：8 批次全量开发完成，248 个单元/集成/回归测试通过。

核心代码：
- `core/` — 全部模块完成（common/data/features/models/backtest/portfolio/risk/signals）
- `services/` — 9 个微服务代码完成，Docker 编排就绪
- `research/` — 回测主入口 + Optuna TPE 因子优化器
- `configs/` — strategy.yaml（117只股票/5因子/regime/DeepSeek NLP）
- `docs/` — 12 篇文档

### 2. 当前策略配置（configs/strategy.yaml）

```yaml
backtest:
  start_date: "2025-01-02"
  end_date: "2026-05-12"
  initial_capital: 1000000

strategy:
  top_n: 5
  rebalance_frequency: monthly
  optimizer: equal_weight

regime:
  enabled: true  # 市场状态判断已启用
  min_position_ratio: 0.30
  max_position_ratio: 0.90

factors:
  momentum_60d:  enabled, weight 0.71
  turnover_5d:   enabled, weight 0.20
  turnover_20d:  enabled, weight 0.05
  amihud_illiq:  enabled, weight 0.03
  momentum_20d:  enabled, weight 0.01

股票池: 117 只（蓝筹44 + AI半导体44 + 机器人13 + 电力8 + 苹果供应链8）
```

### 3. 跨时间段验证结果

| 区间 | 夏普 | 年化 | 最大回撤 | 备注 |
|------|------|------|---------|------|
| 2019-2021 | 0.89 | 12.7% | -9.2% | 外推验证，COVID期间 |
| 2024-2025 | 1.50 | 17.1% | -8.2% | 训练窗口 |
| 2026 H1 | 0.15 | 4.1% | -9.1% | 样本太小(105天)，不具统计意义 |

### 4. 关键决策和结论

1. **动量因子在 98 只蓝筹精选池上效果最好**（TPE 给了 71% 权重），换到 117 只混合池后振幅取代了 60 日动量成为主力
2. **基本面因子需要真实财报数据**：当前 demo 通过东方财富 ths 接口拉取了 ROE/营收增速/净利增速/负债率等 14 个指标，但 optimizer 在月频池上最终没有选择它们（2024-2025 年动量和低波动主导）
3. **市场状态判断在保护下行风险**：regime 启用在 2024-2025 窗口会将仓位控制在 65-85%，收益下降但回撤更可控
4. **分板块选因子有理论价值但尚未实现**：当前 117 只股票用同一组因子，TPE 找到的是"最通用"而非"最精确"的组合。可以按白酒/芯片/银行分别建 YAML、分别跑 optimizer

### 5. 已修复的关键 Bug

1. **引擎 `feature_loader` 只传持仓股票代码** → 第二个月起策略只看自己的持仓，永远不调仓 → 已修复为传全市场代码
2. **`TradeRecord` 没有 `realized_pnl`** → 胜率统计只能用"卖出笔数/总笔数" → 已修复为真实盈亏统计
3. **`calendar.py` 代理下卡死** → `socket.setdefaulttimeout` 不生效 → 最终改为不主动联网（优先缓存→失败用估算日历）
4. **`regime.py` Timestamp vs date 比较错误** → `market_data.index <= pd.Timestamp(as_of_date)` 在 index 是 date 类型时报 TypeError → 已修复
5. **`yaml.dump` 把 `"000001"` 转成整数 `1`** → 部分深市股票拉取失败 → 已改用固定格式写入带引号的代码

### 6. 环境特定注意事项

1. **网络受代理限制**：不能直连 PyPI/push2his.eastmoney.com，但能直连 finance.sina.com.cn。**akshare 数据拉取使用新浪接口**
2. **lightgbm 是离线安装的**：wheel 文件在 `~/Downloads/` 下，手动解压到项目目录，`install_name_tool` 修复了 `libomp.dylib` 路径。在 `.gitignore` 中已排除，重新部署时需要重新安装
3. **交易日历已缓存**：在 `~/.quant_system/calendar/trading_calendar.parquet`，二次启动不再拉取。清理用 `rm -f ~/.quant_system/calendar/trading_calendar.parquet`

### 7. NLP 引擎状态

| 引擎 | 状态 | 使用方式 |
|------|------|---------|
| 关键词规则（KeywordSentimentEngine） | ✅ 可用 | 默认 |
| DeepSeek API（DeepSeekAPIEngine） | ✅ 代码已完成 | `export DEEPSEEK_API_KEY=sk-xxx && export NLP_MODEL=deepseek` |
| 本地 Qwen3（Transformers） | ✅ 代码骨架已完成 | `analyzer.install_transformers_model("Qwen/Qwen3-0.6B")` |

### 8. 计划中但未完成的工作

1. Walk-Forward 验证接入 optimizer（`core/backtest/walk_forward.py` 已实现但未挂到 `factor_optimizer.py`）
2. 分板块并行选股（创建多个 `strategy_*.yaml` 并分别跑 optimizer）
3. 基本面因子实效验证（当前 optimizer 排除它们是因为 demo 数据精度不足，接入真实 Level-1+ 财报数据后应重新搜索）
4. C++ 组件编译（`cpp/` 代码和 CMakeLists.txt 就绪，但未在本机编译）
5. 实盘 QMT 对接（`order_manager` 代码就绪，需满足 50 万门槛 + 合规报告）
6. `kafka-python` 安装后激活真实消息总线（当前服务间通信降级为日志模式）

### 9. 三步快速验证系统正常

```bash
python -c "from core.common.calendar import get_calendar; print(len(get_calendar().all_trading_days))"
# 应输出: 6543 个交易日

python -m pytest tests/unit/ -q
# 应全部 PASS 或部分 SKIP (lightgbm 相关)

python research/run_backtest_demo.py
# 应输出回测结果
```

---

## 文件索引

| 文件 | 描述 |
|------|------|
| `docs/CONTEXT.md` | 本文件 |
| `docs/OPERATIONS.md` | 操作指南（初始化/回测/优化/实盘/NLP/清理/故障排查） |
| `docs/PRESENTATION.md` | 汇报材料（14页PPT + 易拉宝） |
| `docs/UPGRADE.md` | 升级指南（7个可替换部件） |
| `docs/A股量化交易系统详细方案.md` | 系统级方案 |
| `docs/系统实现设计指南.md` | 工程实现指南 |
| `docs/runbook.md` | 生产运维手册 |
| `docs/prompts.md` | 8批次开发prompt全集 |
| `docs/GITHUB_SETUP.md` | GitHub上传步骤 |
| `docs/quant_tech_stack_report.md` | 技术栈调研 |
| `configs/strategy.yaml` | 当前策略配置 |
| `research/run_backtest_demo.py` | 回测主入口 |
| `research/factor_optimizer.py` | 因子优化器 |
| `.env.example` | 环境变量模板 |
