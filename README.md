# A股量化交易系统

个人量化研究与交易系统：**本地数据仓 → 研究回测 → 因子检验 → 每日信号 → 模拟盘对账 → 手动/半自动下单**。

> ⚠️ **风险声明**：本文不构成任何投资建议。量化交易存在本金损失风险，实盘前必须经过充分回测与模拟验证。没有任何系统能稳定准确地预测股价。

---

## 一、系统架构

主路径是 `research/` 研究链（`services/` 下的微服务是早期原型，**已封存不维护**，见 `services/README.md`）：

```
┌─ 数据层 ────────────────────────────────────────────────┐
│ scripts/update_data.py   每日增量：行情入本地 Parquet 仓  │
│                          + 行业成分快照（攒 PIT 成分库）   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─ 研究层 ────────────────────────────────────────────────┐
│ research/factor_ic.py         因子 IC 检验（先证明有信息量）│
│ research/factor_optimizer.py  因子组合搜索（只搜通过检验的）│
│ research/run_backtest_demo.py 回测验证                    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─ 执行层 ────────────────────────────────────────────────┐
│ research/pick_stocks.py    每日选股信号                   │
│ research/paper_trading.py  模拟盘双账本对账（实盘前闸门）  │
│ → 手动下单                                               │
└─────────────────────────────────────────────────────────┘
```

**第一设计原则：回测的策略 = 实盘跑的策略。** 回测、每日选股、模拟盘三处共用同一套因子计算、打分、组合规则代码（同一个 `feature_loader` / `_score_stocks` / `select_target_portfolio`），费用、滑点、涨跌停口径也一致。

### 技术栈

| 组件 | 方案 |
|------|------|
| 语言 | Python 3.12+ |
| 数据源 | westock-data（腾讯源，主）/ AkShare（备）→ 本地 Parquet 数据仓 |
| 回测引擎 | 自研（`core/backtest/`：真实佣金印花税/滑点/板块涨跌停/封板拒单） |
| 因子检验 | RankIC / ICIR / IC 衰减 / 五分层回测 |
| 组合优化 | Optuna TPE + Walk-Forward + IC 门禁 |
| 组合约束 | rank buffer 换手控制 / 单行业上限 / 单票权重上限 / 行业市值中性化 |

改进历史与方法论依据见 `docs/REVIEW_AND_ROADMAP_2026-07.md`（8 阶段全部完成，2026-07）。

---

## 二、快速开始

### 1. 环境准备

```bash
# Python 依赖
pip install -e .            # 或 make install

# westock 数据源需要 Node.js ≥ 18（node 在 PATH 中即可）
node --version

# 可选：lightgbm 模型测试需要 libomp（没有则相关测试自动 skip）
brew install libomp
```

### 2. 建本地数据仓（首次）

```bash
# 若之前跑过回测，直接把 westock 分段缓存导入仓库（不联网，秒级）
python scripts/update_data.py import-cache

# 查看仓库覆盖状态
python scripts/update_data.py status

# 之后每日收盘后增量更新（也负责攒行业成分快照，建议挂 cron）
python scripts/update_data.py update --config configs/strategy_bank_westock.yaml
```

数据仓建好后，回测对仓内覆盖的股票**完全不联网**——数据源限流也不影响研究。

### 3. 跑一次回测

```bash
python research/run_backtest_demo.py configs/strategy_bank_westock.yaml          # 验证区间
python research/run_backtest_demo.py configs/strategy_bank_westock.yaml --full   # 完整训练区间
```

输出包含：收益/夏普/回撤等指标、相对沪深300 的超额与信息比率、每次调仓的买卖清单与费用明细、现金不足注资提示。

### 4. 运行测试

```bash
python -m pytest tests/ -q     # 全量（313 passed / 23 skipped 为正常基线）
```

> skip 的 23 个主要是 lightgbm 用例（本机缺 libomp）。**passed 数下降才是回归**。

---

## 三、日常工作流

### 每日例行（收盘后，约 5 分钟）

```bash
# 1. 数据仓增量更新 + 当日成分快照（每个在用的池都跑一遍）
python scripts/update_data.py update --config configs/strategy_bank_westock.yaml
python scripts/update_data.py update --config configs/strategy_semiconductor_westock.yaml

# 2. 模拟盘：生成今日调仓信号（落虚拟订单，幂等可重跑）
python research/paper_trading.py signal \
    --config configs/strategy_semiconductor_westock.yaml \
    --pool semiconductor --budget 100000

# 3. 模拟盘：结算之前的挂单（按次日开盘价撮合）+ 推进净值
python research/paper_trading.py settle --pool semiconductor

# 4.（需要实盘参考时）看今天该持有什么
python research/pick_stocks.py
#    （跑哪些池、各池预算在 pick_stocks.py 的 main() 里配置）
```

### 每月复盘

```bash
# 模拟盘对账：模拟净值 vs 回测预期净值，看执行偏差
python research/paper_trading.py report --pool semiconductor
```

报告核心是**双账本对比**：真实账本按次日开盘价有条件成交（涨停拒买/跌停拒卖/停牌顺延），影子账本按信号日收盘价无条件成交（即回测引擎的乐观假设）。两条净值曲线之差 = 回测隐藏的执行偏差。**攒 3~6 个月数据、偏差可接受，才考虑实盘。**

---

## 四、研究工作流（找因子 → 组权重 → 验证）

### 第 1 步：因子 IC 检验（先证明因子有信息量）

```bash
python research/factor_ic.py --config configs/strategy_bank_westock.yaml
```

对候选池的每个因子输出：月度截面 RankIC 均值 / ICIR / t 值 / 正率、IC 衰减（1/3/6 个月）、五分层回测（Q5−Q1 spread 与单调性）、pass/fail 判定（默认 ICIR ≥ 0.3）。报告落 `data/reports/factor_ic/ic_{pool}.json`。

反向因子（波动率/PE 等）已按方向调整——好因子的 ICIR 恒为正，直接看数值即可。

> 2026-07 首跑结论：银行池只有低波动/反转类通过（volatility/amplitude/atr/bollinger/rsi）；**动量类因子在银行、半导体两池 ICIR 均为负**——直接在全因子池上跑优化器搜出的动量组合基本是过拟合。这就是为什么必须先跑这一步。

### 第 2 步：优化器搜权重（只在通过检验的因子里搜）

```bash
python research/factor_optimizer.py --task long_term \
    --config configs/strategy_bank_westock.yaml \
    --ic-report data/reports/factor_ic/ic_bank_westock.json \
    --rounds 200

# 常用参数
#   --search-years 1      短窗口搜索加速 2~4 倍，Top-3 自动全区间验证
#   --min-icir 0.5        提高 IC 门槛
#   --min-factors 3 --max-factors 6
```

不传 `--ic-report` 则退化为全候选池搜索（不推荐）。输出含过拟合诊断（夏普 >2 告警）与 Walk-Forward 子窗口稳定性（配置 `optimizer.use_walk_forward` 后）。

### 第 3 步：把最优组合写回 YAML，回测验证

优化器结尾会打印可直接粘贴的 `factors:` 配置段。改完后：

```bash
python research/run_backtest_demo.py configs/strategy_bank_westock.yaml --full
```

关注**相对基准**口径：超额年化、信息比率（IR > 0.5 才算有持续超额能力），绝对夏普受市场涨跌影响大。

---

## 五、策略配置说明（configs/strategy_*.yaml）

一个配置 = 一个策略池。当前在用：

| 配置 | 股票池 | 说明 |
|------|--------|------|
| `strategy_bank_westock.yaml` | 银行（申万一级，dynamic） | 价值池，基本面因子实验田 |
| `strategy_semiconductor_westock.yaml` | 半导体（申万二级，dynamic） | 成长池 |
| `strategy_multi_sector_westock.yaml` | 多板块并集（top_n=20） | 扩宇宙模板：组合约束+中性化全开 |
| `strategy_robotics.yaml` 等 | 固定代码列表（akshare） | 旧池，限流时不可用 |

### 关键配置段

```yaml
backtest:
  start_date: '2023-01-01'       # 训练区间
  end_date: '2025-12-31'
  validate_start: '2026-01-01'   # 可选：验证区间（默认回测用它，--full 用训练区间）
  validate_end: '2026-06-25'
  initial_capital: 200000

data_source:
  provider: westock              # westock | sina | eastmoney
  universe:
    mode: dynamic                # dynamic=行业成分每月 PIT 重建 | fixed=固定 codes
    industries: ["pt01801780"]   # westock 申万板块码（search <名> --type sector 核对）
    pool_size: 30                # 每月候选上限（按日均成交额取 top）
    min_listing_months: 12       # 剔除次新
    min_avg_amount: 200000000    # 流动性门槛（60 日均成交额）

strategy:
  top_n: 3                       # 持仓只数
  rebalance_frequency: monthly   # monthly | weekly | daily
  # ── 组合约束（均可选，不配 = 关闭）──
  rank_buffer: 6                 # 已持有排名 ≤6 保留（换手控制，须 ≥ top_n）
  max_per_sector: 6              # 单行业最多只数（多板块宇宙用）
  max_weight_per_stock: 0.10     # 单票权重上限（超额留现金）
  neutralize:                    # 因子中性化（打分前回归取残差）
    industry: false              #   行业中性化（需 dynamic 多板块宇宙）
    mktcap: true                 #   市值中性化（close × westock 股本）

regime:
  enabled: true                  # 市场状态判断（沪深300 趋势/波动/宽度/海外）
  min_position_ratio: 0.2        # 熊市最低仓位
  max_position_ratio: 0.9

factors:                         # 因子开关与权重（权重自动归一化）
  volatility_20d:
    enabled: true
    weight: 0.4
    params: {window: 20}
  # ... 全部因子见任一配置文件，enabled: false 的不参与打分

optimizer:                       # 优化器专用（可选）
  time_decay_halflife: 1.0       # 时间衰减夏普半衰期（年）
  use_walk_forward: false        # WF 目标（mean − λ·std）
```

新建策略池：复制最接近的 YAML → 改 `industries`/`factors` → 先跑 `factor_ic.py` 再跑优化器。

---

## 六、数据仓机制（为什么可靠）

- **存储**：`data/warehouse/kline/{code}.parquet` 单股单文件 + `kline_meta.json` 覆盖速查；行业成分快照在 `data/warehouse/constituents/{board}/{date}.json`。
- **复权正确性**：westock 只提供前复权（qfq）价，除权除息后全历史会整体平移。`update` 每次多拉最近 15 天与仓内重叠比对，close 偏差 >0.5% 即判定基线已变 → 自动废弃该股全历史重拉。**坚持每日更新，仓内复权基线恒为最新。**
- **回测读取**：`_load_real_data` 对仓内覆盖的代码直接读 parquet，未覆盖才联网；联网失败自动降级到本地部分。
- **PIT 成分库**：每日快照攒起来后（`warehouse.constituents_asof`），可回放「当时的行业成分」，真正消除幸存者偏差（当前 `universe.py` 仍用当前成分，待快照积累后接入）。

---

## 七、项目结构

```
must_be_rich/
├── configs/                       # 策略/系统 YAML（唯一的调参入口）
│   ├── strategy_*.yaml            #   策略池配置（见第五节）
│   ├── boards/                    #   板块交易规则（涨跌幅/最小申报，回测自动加载）
│   └── system.yaml                #   佣金/印花税等
├── research/                      # ★ 主路径：研究与执行链
│   ├── run_backtest_demo.py       #   回测入口 + 策略/因子计算/数据加载
│   ├── factor_ic.py               #   因子 RankIC/ICIR 检验
│   ├── factor_optimizer.py        #   Optuna 因子组合搜索（--ic-report 门禁）
│   ├── pick_stocks.py             #   每日选股
│   ├── paper_trading.py           #   模拟盘（signal/settle/report）
│   ├── portfolio_rules.py         #   组合约束（buffer/行业上限/权重上限）
│   ├── warehouse.py               #   本地 Parquet 数据仓
│   ├── westock_source.py          #   westock CLI 适配（行情/财报/股本/成分）
│   ├── universe.py                #   动态宇宙 PIT 筛选
│   └── sector_rotation.py         #   行业轮动（独立轻量回测）
├── core/                          # 核心库
│   ├── backtest/                  #   回测引擎/约束/费用模型
│   ├── features/                  #   因子库/中性化/验证工具
│   ├── portfolio/                 #   regime 检测/组合优化器
│   └── common/                    #   交易日历（防缓存投毒）等
├── scripts/
│   ├── update_data.py             #   ★ 每日数据仓更新（update/import-cache/status）
│   └── setup.sh 等                #   初始化/清理
├── tests/                         # 313 个测试（unit/integration/regression）
├── data/                          # 运行时数据（.gitignore，不入库）
│   ├── warehouse/                 #   行情仓 + 成分快照
│   ├── paper_trading/             #   模拟盘账本 JSON
│   └── reports/factor_ic/         #   IC 检验报告
├── docs/
│   ├── REVIEW_AND_ROADMAP_2026-07.md  # ★ 全库审查与 8 阶段改进记录
│   └── OPERATIONS.md 等           #   操作手册/升级指南
├── westock-data/                  # westock CLI（node，.gitignore）
└── services/                      # ⚠ 已封存的微服务原型（不维护）
```

---

## 八、常见问题

**Q：westock 报 `SKILL_006: fetch failed`？**
腾讯源限流或网络问题。仓内已覆盖的股票不受影响（回测走本地）；增量更新稍后重试即可。

**Q：回测结果和之前不一致？**
检查三点：① 数据仓是否更新过（除权重拉会改历史价格，这是修正不是 bug）；② 配置里 `validate_start/end` 存在时默认跑验证区间，加 `--full` 跑完整区间；③ 动态宇宙每月成分可能随缓存刷新变化。

**Q：测试有 23 个 skip 正常吗？**
正常。主要是 lightgbm 用例（本机缺 libomp，`brew install libomp` 后自动恢复）。

**Q：怎么加一个新因子？**
① 在 `run_backtest_demo._compute_factor_value` 加计算分支；② 在各 `configs/strategy_*.yaml` 的 `factors:` 段注册（`enabled/weight/params`）；③ 若是「越低越好」的因子，加入 `ConfigDrivenStrategy.INVERSE_EXACT/PREFIXES`；④ 跑 `factor_ic.py` 检验有没有信息量，通过了再进优化器候选池（`factor_optimizer.FACTOR_POOL`）。

**Q：什么时候可以实盘？**
路线图的答案：模拟盘攒 3~6 个月对账数据，「模拟盘净值 vs 回测预期」的执行偏差稳定且可接受之后。在此之前实盘等于用真钱验证回测假设。

---

## 许可证

MIT License

---

> **一句话总纲**：数据正确性 > 一切；回测的策略必须等于实盘跑的策略；先证明因子有信息量再搜权重；实盘之前先让模拟盘证明「回测 ≈ 现实」；过拟合与情绪化是最大的两个敌人。
