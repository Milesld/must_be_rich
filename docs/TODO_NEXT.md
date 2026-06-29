# 后续待办（2026-06-28 改造之后）

> 本文件记录"动态股票池 + Regime 解耦 + 优化器重构 + 现金处理"五项改造**已完成**后，
> 仍需人工执行/验证/决策的事项。改造背景见 docs/TODO.md 与 git 历史。
> 改造涉及文件：`research/universe.py`(新)、`research/run_backtest_demo.py`、
> `research/factor_optimizer.py`、`research/pick_stocks.py`、`core/backtest/engine.py`、
> 四个 `configs/strategy_*.yaml`、`tests/unit/test_universe.py`(新)。

---

## A. 必须先做的验证（联网，本地 miniforge base 缺 akshare/optuna/pyarrow 跑不了）

> 当前 shell 的 python 没装依赖，且 pip 被代理拦截（403）。须在平时跑回测的环境执行。

### A1. 环境与接口连通性自检
```bash
python -c "import akshare, optuna, pyarrow; print('deps ok')"
# 核对四个池配置里的行业名是否与 akshare 实际板块名一致：
python -c "import akshare as ak; print(ak.stock_board_industry_name_em()['板块名称'].tolist())"
```
**关键**：四个 YAML 里的 `data_source.universe.industries` 是我**按常识猜的板块名**，
未经接口核对。若某行业名不存在，`build_industry_universe` 会 warning 并跳过该行业，
导致候选宇宙变小或为空。务必用上面命令核对并改正。当前填写的：
- 半导体: `半导体`, `光学光电子`
- 机器人: `通用设备`, `专用设备`, `汽车零部件`
- 苹果链: `消费电子`, `电子元件`, `家用电器`
- AI应用: `软件开发`, `计算机设备`, `互联网服务`

### A2. 宽基指数接口连通性
```bash
python -c "import akshare as ak; print(ak.stock_zh_index_daily(symbol='sh000300').tail(3)); print(ak.stock_zh_index_daily(symbol='sh000985').tail(3))"
```
若 `sh000985`（中证全指）拉不到，`_load_benchmark_index` 只会缺 csi_all 键，
regime 仍用 csi300 做趋势/波动，可接受。若 `sh000300` 也失败，会回退到池内伪指数
（日志 debug "宽基指数不可用..."），届时需换指数接口。

### A3. 回测冒烟（逐池）
```bash
python research/run_backtest_demo.py configs/strategy_semiconductor.yaml
# robotics / apple_chain / ai_app 同样各跑一遍
```
确认：
- [ ] 日志只出现一次 "正在拉取 N 只股票"（数据只加载一次）。
- [ ] 出现多条 "当月宇宙 YYYY-MM-DD: N 只"，且 N 逐月**有变化**（动态生效）。
- [ ] 若现金不足，出现 "现金不足：…已假定追加注资"，结果区显示 `total_injected`。
- [ ] 候选宇宙不为空（若为 0，多半是 A1 行业名错或 PIT 门槛太严）。

### A4. 优化器性能 + Walk-Forward
```bash
python research/factor_optimizer.py --task long_term --config configs/strategy_semiconductor.yaml --rounds 30
```
- [ ] 确认数据加载日志只出现一次（_load_shared_data 生效）。
- [ ] 临时把 yaml `optimizer.use_walk_forward: true` 跑一次，确认报告输出
      "Walk-Forward 子窗口夏普" 各段明细。注意 WF 模式下每个组合回测 ×窗口数，
      30 轮可能较慢，先小轮数验证。

### A5. 既有测试不回归（带 pytest 的环境）
```bash
pytest tests/unit/test_engine.py tests/unit/test_constraints.py tests/unit/test_universe.py -q
```
test_engine.py 有 5 处 `BacktestResult(...)` 构造，`total_injected` 已设默认值 0.0，
应兼容；仍需实跑确认。

---

## B. 验证通过后的调参/决策（按优先级）

### B1. ★★★ 对比验证：动态池 vs 旧固定池
同一组因子，分别用 `universe.mode: dynamic` 和 `fixed` 在 2026H1 验证窗口跑，
对比夏普/回撤。**预期 dynamic 指标下降**——这是幸存者偏差被移除的正确结果，
不是 bug。若 dynamic 反而更高，要警惕宇宙构建是否仍漏了未来信息。

### B2. ★★★ PIT 门槛调优
当前默认：`min_listing_months: 12`, `min_avg_amount: 2亿`, `pool_size: 30`。
- 若每月宇宙过小（<10 只），放宽 min_avg_amount 或扩 industries。
- 若过大（>50），收紧或调小 pool_size。
- 关注：宇宙规模在 2023 vs 2025 是否合理（早期上市股少，宇宙可能偏小）。

### B3. ★★ regime 维度权重调优
当前初值 `trend 0.30 / volatility 0.25 / breadth 0.15 / overseas 0.30 / valuation 0.00`。
overseas 从原 0.35 降到 0.30。在真实宽基接入后，可回测验证：overseas 权重多少
最优；breadth 维度目前仍部分依赖池内 limit_up 推算，可考虑全部改用中证全指。

### B4. ★★ wf_lambda 与窗口参数调优
WF 目标 = mean − λ·std，λ 默认 0.5。λ 越大越偏好稳定。配合
wf_train_years/wf_test_months 调整。等实盘或更长历史积累后更有意义。

### B5. ★ AI 应用池重组（接 docs/TODO.md #3）
该池历史最弱（夏普 1.33）。dynamic 化后行业暂填 软件开发/计算机设备/互联网服务。
可考虑改为"AI 基础设施"（算力/数据中心）或并入半导体池。重组后重跑 optimizer。

---

## C. 本轮明确**未做**、需要时另起的事项

### C1. 基本面因子（pe/pb/roe 等）—— 当前全程不加载
原因：`_load_financials` 里 ROE 是**伪造值**（`净利增速/营收增速`，非真 ROE），
THS 数据精度不足，故被优化器 `ALL_NO_DATA` 过滤，且已从 demo/optimizer/pick_stocks
的调用链移除（financials={}）。
**要做需**：① 换数据源（如 `ak.stock_financial_analysis_indicator` 东财财务指标，
有真 ROE/ROA/毛利率）；② **按财报披露日做 PIT 对齐**（年报次年4月才披露，回测
当下只能用已披露的最近报告期，否则前视偏差）；③ 重新纳入候选池。
工程量大，独立立项。

### C2. 动态池的残留幸存者偏差（无法用免费源彻底消除）
akshare `stock_board_industry_cons_em` 返回**当前**行业成分，回放到 2023 年：
真实成分可能不同、已退市股拿不到。已在 `research/universe.py` docstring 标注。
彻底消除需含退市股的 PIT 成分库（付费数据源，如 Wind/聚宽）。短期接受现状。

### C3. docs/TODO.md 里仍未做的项
- #1 止损/止盈加入月度框架（约 10-15 行，`on_bar` 非调仓日加规则）——优先级高，
  改造稳定后可做。
- #4 信号加权（top-2/3 场景意义微乎其微，优先级最低）。
- #5 多策略并行调度（运维项）。
- #8 接入实盘（前提：验证窗口夏普>1 持续3个月——尚未满足）。

### C4. services/ 微服务路径仍非功能性
见记忆 [[dual-factor-implementations]]：premarket/intraday 服务模型从不 load、
intraday bar feed 硬编码 []、premarket 喂随机噪声。本轮改造**只动了长期选股回测路径**
（research/），未碰服务。若要上线服务需单独修复。

---

## D. 实现细节备忘（改下次回看）

- 动态宇宙数据流：`_get_codes`(dynamic→行业并集,带 `_INDUSTRY_UNIVERSE_CACHE` 缓存)
  → `_load_real_data` 一次性拉全 → `_build_monthly_universe` 按月 PIT 切片
  → `on_bar`/`_add_cross_sectional_factors`/`pick_stocks` 用 `_universe_for_date` 取当月。
- 横截面因子（alpha_momentum/sector_relative_strength）的"市场基准"已改为**当月宇宙**，
  不再是全部拉取代码。
- regime 优先用 csi300（`_build_market_proxy`），宽基缺失才回退池内伪指数（有 debug 日志）。
- optimizer：`run_single_backtest(shared_data=...)` 不再自己联网；`_walk_forward_score`
  复用 `core/backtest/walk_forward.py` 的 `get_windows`；只 semiconductor 之外的池也已配 WF 参数（默认关）。
- 现金：`engine.total_injected` 累计，`summary()` 出 `total_injected`/`effective_capital`；
  收益率口径仍按 initial_capital，effective 仅参考。
