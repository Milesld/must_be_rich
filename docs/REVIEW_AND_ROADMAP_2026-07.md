# 代码库审查与改进路线图（2026-07-14）

> 本文档来自 2026-07-14 的一次全库审查（覆盖回测引擎、约束、成本模型、数据源、
> 宇宙构建、优化器、行业轮动，并运行了测试套件：194 通过 / 1 失败 / 1 个文件无法加载）。
> 后续改进工作按本文档的「三、行动顺序」逐步推进，每完成一项在对应条目打勾并注明日期。
>
> **总体结论**：回测框架的工程质量不错（PIT 对齐、Walk-Forward 验证、真实费用模型都有），
> 但存在一个会直接扭曲所有回测结果的高优先级 bug（板块涨跌停配置未加载），
> 以及若干中等问题。改进方向是把研究路径（research/）做扎实，而不是修微服务空壳。

---

## 一、已发现的 Bug 与问题清单

### 🔴 高优先级

#### 1.1 创业板/科创板涨跌停被按 ±10% 检查（影响所有回测结果）

**现象**：`research/run_backtest_demo.py`（约 1601 行）创建引擎时没有传 `config`：

```python
engine = BacktestEngine(
    start_date=start_date, end_date=end_date,
    initial_capital=initial_capital,
)
```

**链路**：`BacktestEngine.__init__` 里 `board_cfg = getattr(config, "boards", None)`
拿到 `None` → `TradingConstraints(None)` 的 `_boards={}` →
`update_daily_info` 中 `rules.get("price_limit", 0.10)` 全部落到默认 ±10%。

**后果**：股票池里大量 300/688 开头的股票（半导体、机器人池几乎全是），
它们合法的 +10%~+20% 日内波动会被误判为「价格超过涨停价」而**拒单**。
`configs/boards/` 下五个板块配置（sse_main/sse_star/szse_gem/szse_main/bse）
在回测链路里从未被加载。这会系统性扭曲每一次回测和优化器搜索的结果——
动量策略在大涨日买不进，回测收益比真实规则下偏低且失真。

**修复方案**：把板块配置传进引擎；或在 `TradingConstraints` 里内置按代码前缀的
默认涨跌幅表（688→0.20、300/301→0.20、bj(8/4/920)→0.30、其余→0.10），
使「不传配置」时的默认行为也正确。修好后**历史回测结论需全部重跑**。

### 🟠 中优先级

#### 1.2 前复权数据 + 永久分段缓存 = 除权后数据错位

`research/westock_source.py` 的 `_fetch_one_segment` 把 qfq（前复权）K 线按
180 自然日分段缓存，TTL 3650 天（注释称「历史行情不变」）。
但**前复权价格是以「今天」为基准回算的**——任何一次分红/送股除权后，
复权基准变了：旧缓存段还是老基准，新拉的段是新基准，拼接处出现假跳空，
直接污染动量、波动率、振幅等所有价格因子。

**修复方案**（二选一）：
- 正确做法：缓存不复权价 + 复权因子，读取时实时复权；
- 低成本做法：拼接时校验相邻段边界处价格连续性（如缝隙 >5% 且当日非涨跌停），
  不连续则废弃该股全部缓存段重拉。

#### 1.3 交易日历缓存投毒

`core/common/calendar.py` 的 `_load`：离线时用「周一到周五」估算日历
（春节、国庆等节假日全被当成交易日），**并把估算结果写入同一个缓存文件**；
而缓存一旦存在，后续初始化永远优先使用缓存。
只要有一次离线运行，之后所有回测都在错误日历上跑。

**修复方案**：估算日历不落盘；或落盘时打 `estimated=true` 标记，
下次初始化发现该标记就强制联网刷新。
**临时恢复手段**：删除 `~/.quant_system/calendar/` 目录。

#### 1.4 滑点口径自相矛盾

`core/backtest/cost_model.py` 的 `_calc_slippage` 注释说滑点「不实际扣除
（非真实成本），用于评估策略在真实市场中的可行性」，但它被计入
`CostBreakdown.total`，而 `engine.py` 的 `_apply_fill` 按 `cb.total` 扣现金——
**实际上扣了**。回测报告里又把它显示为「滑(估)」，读报告的人会以为收益未含滑点。

**修复方案**（二选一，保持口径一致即可）：
- 把 slippage 从 `total` 中拿出来，报告单列「若含滑点则收益为 X」；
- 保留扣除行为，修正注释与报告文案为「滑点已计入成本扣除」。

#### 1.5 封板检查形同虚设

`core/backtest/constraints.py` 的封板判断要求 `volume <= 0` 才认定封板。
日线数据里成交量几乎不可能为 0（一字涨停日也有集合竞价成交）。
结果：回测里可以在一字涨停日照常买入——恰好**高估动量类策略**（当前主力策略）。

**修复方案**：改为「收盘价 ≈ 涨停价（容差 0.1%）即视为封板，买单拒绝；
收盘价 ≈ 跌停价即卖单拒绝」，宁可保守。可加开关配置。

#### 1.6 pick_stocks.py 与回测打分不一致（回测和实盘不是同一个策略）

`research/pick_stocks.py` 硬编码 `financials = {}`、不加载 `total_shares`
（注释还停留在「ROE 为伪造值」的旧时代，westock 真财报接入后没同步）。
一旦某个配置启用了基本面因子（如银行池的 roe_ttm/pb），
回测用真 PIT 财报打分、实盘选股却把这些因子全按 0 算。

**修复方案**：`pick_stocks` 复用 `run_backtest_demo.main` 的数据加载分支——
provider 为 westock 时同样调用 `westock_financials` + `westock_total_shares`。

#### 1.7 测试问题（2 处）

- `tests/unit/test_factors.py::TestFundamentalFactors::test_with_financials_no_announce_date`
  失败：`core/features/fundamental.py::_merge_financials` 在 financials 缺
  `announce_date` 列时，用 datetime64 的 `trade_date` 对 object 类型的
  `report_date` 做 `merge_asof`，抛 `MergeError: Incompatible merge dtype`。
  修复：merge 前把 `report_date` 转成 datetime64。
- `tests/unit/test_models.py` 整个文件无法收集：lightgbm 加载失败，
  缺 `libomp.dylib`。环境修复：`brew install libomp`。

### 🟡 小问题

- **收盘价撮合 + 收盘价决策**：`on_bar` 用当日收盘价打分并下单、又按当日收盘价
  成交，隐含「能在收盘瞬间同时看到信号并成交」，偏乐观。引擎里已有
  `OHLCFillSimulator`（VWAP 撮合）但默认没用上。可改为次日开盘价撮合或 VWAP。
- **仓库卫生**：根目录 ~3.5MB 的 `optimizer_*.log`、`temp_output.log`、
  空的 `run.sh` 应加进 `.gitignore` 并清掉。
- **死代码**：`_load_financials`（伪造 ROE = net_profit_yoy/revenue_yoy 那个函数）
  已无调用方，应直接删除避免误用。
- **行业轮动池覆盖窄**：`research/sector_rotation.py` 逻辑干净，
  但行业指数池只有 12 个且是手选的，覆盖面不足以称「轮动全市场」。

---

## 二、对照业界系统的差距分析

**定位判断**（改进方向的前提）：当前实际在跑的是**「研究脚本」路径**
（`run_backtest_demo.py` + `pick_stocks.py` + 手动下单）。
`services/` 里的 9 个微服务是空壳：盘前服务喂随机数
（`np.random.normal` 伪造 A50 数据）、日内服务行情硬编码 `[]`、模型从未从磁盘加载。
因此改进应该**把研究路径做扎实，而不是去修微服务**。

### 2.1 没有因子有效性检验层（最大的方法论缺口）

业界流程：单因子 IC/RankIC 分析 → 分层回测（quintile spread）→
因子相关性/正交化 → 再谈组合。
当前是跳过这一步，直接用 Optuna 在 20+ 因子 × 权重空间里搜「哪个组合回测夏普高」，
本质是**在挑最幸运的过拟合**，即使有 walk-forward 的 mean−λ·std 也只是缓解。

**建议**：新建 `research/factor_ic.py`（alphalens 风格，数据结构已齐，约百余行）：
- 对每个候选因子计算月度 RankIC 序列 → IC 均值、ICIR、IC 衰减（1/3/6 个月）；
- 分五层（quintile）回测，看 top-bottom spread 是否单调；
- 把 ICIR < 0.3（或自定阈值）的因子从优化器候选池剔除；
- 优化器只在「被证明有信息量」的因子里搜权重。

### 2.2 股票池太窄，横截面没有意义

单行业约 30 只里选 top 3：横截面排名统计意义弱，
且「行业内选股」和「选对行业」两个决策被耦合。
业界做法：中证800/全市场打分 + 行业中性化
（每个行业内标准化因子值，再全市场排名）。
`core/features/neutralizer.py` 已经写好但长期路径没用上。

**建议**：宇宙扩到至少中证800；因子做行业+市值中性化；
`top_n` 从 3 扩到 20~30 分散持仓。top_n=3 的组合单票风险大到
任何因子 alpha 都会被噪声淹没。

### 2.3 数据层应本地化建仓，而不是每次回测联网拉

业界标配（哪怕个人量化）：本地数据库每日增量更新一次
（**不复权价 + 复权因子** + ST/停牌状态 + 成分股快照），回测只读本地。
当前每次回测都要过 westock CLI/akshare + 分段缓存，慢、脆，
且有 1.2 的 qfq 缓存问题。

**建议**：新增 `scripts/update_data.py`，每日收盘后把行情增量写入
DuckDB/Parquet（`data/` 目录和 `scripts/init_db.py` 的架子已在）。
**顺手每天存一份当日行业成分股快照**——攒一年就有了自己的 PIT 成分库，
幸存者偏差从「文档里诚实标注」变成「真正解决」。

### 2.4 组合构建太原始

当前：等权 top-N + regime 仓位缩放。没有个股权重上限、行业敞口约束、换手率约束。
`core/portfolio/optimizer.py`（318 行）和 `kelly.py` 写了但长期路径没用。

**建议**（比任何因子改进都更能降低实盘回撤）：
- 单票权重 ≤ 10%；
- 单行业敞口 ≤ 30%；
- 月换手率上限（如单边 50%）。

### 2.5 缺少「模拟盘」闭环（"真正可用"的关键一步）

回测到实盘之间没有验证环节。业界做法是 paper trading：
每天 `pick_stocks` 输出目标持仓 → 记录成虚拟组合 →
用次日真实价格跟踪虚拟净值 → 攒 3~6 个月对比「模拟盘净值 vs 回测预期」。

成本极低（每日 cron + JSON 账本 + 对账脚本），
却能暴露回测里所有隐藏的乐观假设（撮合价、涨停买入、滑点）。
**在实盘投钱之前，先让系统证明它的回测和现实是一致的。**

### 2.6 架构取舍建议

`services/` 微服务、Redpanda、C++ 风控是「机构架构」，
对单人 + 20 万资金的场景是负资产。业界个人量化的成熟形态：
**本地数据仓 + 研究回测 + 每日信号生成 + 模拟盘对账 + 手动/半自动下单**。
把这条链做到每一环都可信，比九个微服务有用得多。
建议删掉或 README 里明确封存 `services/`，研究路径就是主路径。

---

## 三、行动顺序（按此推进，完成一项勾一项）

| # | 状态 | 事项 | 内容细节 | 为什么排这个位置 |
|---|------|------|----------|------------------|
| 1 | ✅ 2026-07-15 (c8ece6d) | 修涨跌停板块配置 bug + 封板检查 | 1.1：板块配置注入引擎（或内置前缀默认表）；1.5：封板判断改为收盘价≈涨跌停价 | 所有历史回测结果失真，修好才有可信的基线；**修完需重跑各池回测刷新结论**（对比数据见附录 B） |
| 2 | ✅ 2026-07-15 (14d290a) | 修数据正确性问题 | 1.2：qfq 缓存段缝连续性校验（超板块涨跌停×1.5+2% 判基线错位→废弃该股缓存重拉，见 `westock_source._seams_ok`）；1.3：估算日历落盘打 estimated 标记，下次初始化优先联网替换真实日历（见 `calendar._cache_is_estimated`，旧缓存用春节探针启发式识别）。新增 17 个回归测试（`test_calendar_cache.py` / `test_westock_seams.py`）。当前本机日历缓存已验证为真实日历（春节探针未命中）。残留：小额分红级别的基线错位检不出，根治待第 6 阶段本地数据仓 | 数据正确性 > 一切；这两个问题会静默污染未来所有结果 |
| 3 | ✅ 2026-07-15 (34e9630) | 修一致性与测试 | 1.6：pick_stocks 与回测同分支加载 westock 真财报+总股本（`_provider(config)=="westock"` 时调用 `westock_financials`/`westock_total_shares`）；1.4：滑点口径统一为「已计入成本扣除」（cost_model 注释 + 回测报告文案「滑¥x(已扣)」，行为不变，保守口径）；1.7a：`_merge_financials` 统一转 datetime64 临时 key 再 merge_asof（修 dtype MergeError，且保留原行序保证 Series 对齐）；1.7b：test_models.py 捕获 OSError（缺 libomp 时优雅 skip 而非无法收集）。**libomp 本体未装成**：本机 conda 环境与 `~/.local` 均无写权限，需手动 `brew install libomp` 或 `conda install llvm-openmp` 后 20 个 lightgbm 用例自动恢复。全量测试 240 passed / 23 skipped / 0 failed | 保证「回测的策略 = 实盘跑的策略」，测试全绿 |
| 4 | ✅ 2026-07-15 | 搭 paper trading 闭环 | `research/paper_trading.py`：`signal`（收盘后生成调仓信号，与回测/pick_stocks 同一套打分）→ `settle`（次日**开盘价**撮合，开盘涨停拒买/跌停拒卖/停牌顺延超 10 交易日撤单，费用含滑点与回测同口径）→ `report`（对账报告）。★双账本设计：真实账本按次日开盘成交 vs 影子账本按信号日收盘无条件成交（=回测的乐观假设），两条净值曲线之差=回测隐藏的执行偏差（隔夜跳空/封板/停牌）。账本 JSON 落 `data/paper_trading/`。14 个单元测试（`test_paper_trading.py`）。简化项：无 regime 仓位缩放、订单无部分成交 | 用最低成本验证「回测≈现实」，是实盘前的必要闸门 |
| 5 | ⬜ | 因子 IC 检验层 | `research/factor_ic.py`：RankIC/ICIR/IC 衰减 + 分层回测；ICIR 阈值过滤优化器候选池 | 从「搜过拟合」转向「组合已验证的信号」 |
| 6 | ⬜ | 本地数据仓 + 每日成分快照 | `scripts/update_data.py` 增量入 DuckDB/Parquet（不复权+复权因子）；每日行业成分快照攒 PIT 成分库 | 长期解决幸存者偏差和限流，回测提速 |
| 7 | ⬜ | 扩宇宙 + 中性化 + 组合约束 | 宇宙扩中证800；行业+市值中性化（复用 neutralizer.py）；单票≤10%/单行业≤30%/换手上限 | 让策略统计上站得住 |
| 8 | ⬜ | 清理 | 封存/删除 services/ 空壳；删 `_load_financials` 死代码；log 文件出库进 .gitignore；（可选）撮合改次日开盘/VWAP | 减少维护幻觉，收尾 |

### 推进约定

- 每完成一项：把状态改为 ✅ 并注明完成日期与提交 hash；
- 第 1 项完成后必须重跑 semiconductor/robotics/bank 各池回测，
  在本文档附录记录修复前后的指标差异（夏普/年化/拒单笔数）；
- 第 4 项（paper trading）跑起来之后，每月在本文档追加一次对账摘要；
- 顺序允许微调（如 2 和 3 可并行），但 1→2 必须在任何新的回测结论之前完成。

---

## 附录 A：审查时的测试基线（2026-07-14）

```
python -m pytest tests/ --ignore=tests/unit/test_models.py
→ 194 passed, 1 failed, 3 skipped

失败: tests/unit/test_factors.py::TestFundamentalFactors::test_with_financials_no_announce_date
  (pandas MergeError: datetime64 vs object dtype in merge_asof)
无法收集: tests/unit/test_models.py (lightgbm: libomp.dylib 缺失)
```

代码规模参考：`research/` 6 个脚本约 4,200 行；`core/` 约 13,800 行；
`services/` 9 个微服务（均为空壳，不在改进范围内）。

---

## 附录 B：第 1 阶段修复记录与前后对比（2026-07-15）

### 改动内容

1. `core/backtest/constraints.py`
   - 新增 `_DEFAULT_PRICE_LIMITS` 内置板块涨跌幅表（主板 0.10 / 创业板科创板 0.20 / 北交所 0.30，与 configs/boards 一致），无配置时也按正确幅度检查；
   - 封板判断从「volume≤0 且贴板」改为「收盘价≈涨/跌停价（容差 0.2%）即封板」——旧条件在日线数据上永不触发（涨停日也有集合竞价成交量），检查形同虚设。
2. `core/backtest/engine.py`
   - 新增 `_load_boards_config()`：无 config 时自动从 `configs/boards/*.yaml` 加载板块规则；
   - `BacktestEngine.__init__` 新增 `board_config` 参数，优先级：显式传入 > config.boards > 自动加载。
   - 受益方无需改动：`run_backtest_demo.py` 与 `factor_optimizer.py` 的引擎创建点现在自动获得正确板块规则（含科创板 min_shares=200、价格笼子等）。
3. `research/westock_source.py`（顺手修复，跑基线时发现）
   - `_run_westock_json` 显式识别 `{success:false, error:{...}}` 错误响应并抛出带错误码的异常（旧代码会把错误 dict 的 `"error"` 字符串当数据迭代 → `AttributeError: 'str' object has no attribute 'get'`）；
   - `westock_industry_cons` 联网失败时回退过期缓存（成分月度慢变，过期数据远好于中断）。
4. `tests/unit/test_constraints.py` 新增 9 个回归测试：
   - 板块默认涨跌幅（创业板 +15% 放行 / +21% 拒绝、科创板 +19% 放行、北交所 +25% 放行、主板 +12% 拒绝）；
   - 封板（有成交量的涨停贴板拒买、跌停贴板拒卖）；
   - 引擎自动加载 boards 配置。

### 回测前后对比（完整训练区间 2023-01-01 ~ 2025-12-31）

修复前数据用「旧行为模拟脚本」（`data/reports/fix_boards_baseline/run_before_emulation.py`：
monkeypatch 引擎不加载 boards、全板块 ±10%、封板检查放行）在同一份缓存数据上跑出，
排除数据差异干扰。

| 池 | 指标 | 修复前 | 修复后 | 差异 |
|---|---|---|---|---|
| 半导体 (westock, 178只dynamic) | final_nav | 191,437 | 217,978 | **+26,541 (+13.9%)** |
| | annual_return | −1.51% | +3.03% | **+4.54pp** |
| | sharpe_ratio | −0.211 | +0.045 | +0.256 |
| | excess_annual (vs 沪深300) | −7.72% | −3.18% | +4.54pp |
| | total_trades | 107 | 108 | +1 |
| 银行 (westock, 84只dynamic) | 全部指标 | — | — | **无差异**（银行全是主板股，±10% 本来就对）|

**解读**：
- 半导体池 143/178 只是 20% 涨跌幅板块（688/300/301），修复影响显著：
  年化从 −1.5% 转正为 +3.0%。旧代码把创业板/科创板 10~20% 的合法上涨日
  误判为「超涨停」拒买、又把对应的下跌日误判为「超跌停」拒卖，
  既错过强势买点又卡住止损卖出，双向扭曲。
- 银行池指标完全一致，符合预期（纯主板池），也验证修复没有引入行为漂移。
- robotics/apple/ai_app 三个池仍是 akshare fixed 模式（限流中，未重跑）；
  它们的池内 300 开头股票占比高，结论方向与半导体一致，待数据源可用时补跑。

### 测试

```
python -m pytest tests/ --ignore=tests/unit/test_models.py
→ 202 passed, 1 failed（失败项为既有的 fundamental merge_asof bug，属第 3 阶段范围）
```

### 遗留说明

- westock 网络当日（2026-07-15）大面积 `SKILL_006: fetch failed`（腾讯源限流或接口变动），
  基线回测全部依赖 6/30 之前的本地缓存完成；缓存覆盖完整（半导体 178/178、银行 84/84）。
- 修复前模拟脚本保留在 `data/reports/fix_boards_baseline/`，前后原始输出同目录。
