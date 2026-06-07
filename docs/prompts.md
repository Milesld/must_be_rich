# 分批开发 Prompt 集

> 本文包含 8 个批次的开发 prompt，按依赖顺序排列。每个 prompt 可直接作为与 Claude Code 对话的输入。
> 
> 使用方式：按批次顺序依次将 prompt 发送给 Claude Code。同一批次内的任务可拆分并行。
> 
> **前提**：项目目录下已有以下两篇文档：
> - `A股量化交易系统详细方案.md` — 系统级方案（业务逻辑、因子体系、合规、风控策略）
> - `系统实现设计指南.md` — 工程实现指南（代码结构、表设计、API契约、管线时序、配置规范）

---

## 第1批：项目骨架 + 公共工具 + 配置体系

> **依赖**：无
> **复杂度**：低
> **可并行**：否

```
【角色与上下文】
你是一个A股量化交易系统的开发工程师。请先阅读项目目录下的两份文档：
- A股量化交易系统详细方案.md（了解业务背景）
- 系统实现设计指南.md（了解代码结构、配置规范）
然后再开始写代码。

【本次任务】
初始化项目骨架，包括：

1. pyproject.toml — Python 3.12 项目配置
   依赖清单：lightgbm, catboost, pandas(>=2.2), numpy, scikit-learn,
   pydantic(>=2), grpcio, redis(hiredis), clickhouse-driver, sqlalchemy,
   akracer, tushare, structlog, pyyaml, pytest, pytest-asyncio
   （akracer 是 python 包名，正确的包名可能是 akshare，请以 pip 可安装的实际包名为准）

2. core/common/ 公共工具模块：
   a) calendar.py — 交易日历
      - 基于 akshare 获取A股交易日历
      - TradingCalendar 类：
        · is_trading_day(date) -> bool
        · next_trading_day(date) -> date
        · prev_trading_day(date) -> date
        · get_trading_days(start, end) -> List[date]
        · get_next_n_trading_days(date, n) -> List[date]
        · get_prev_n_trading_days(date, n) -> List[date]
        · is_month_end(date) -> bool（月末最后一个交易日）
        · is_quarter_end(date) -> bool（季末最后一个交易日）
        · is_year_end(date) -> bool
      - 首次从 akshare 拉取全量日历，缓存到本地 Parquet 文件
      - 支持 2000-2030 年份范围
      - 全局单例：get_calendar() 函数

   b) adjustment.py — 复权处理
      - 提供前复权计算：to_forward_adjusted(df, price_cols) -> df
      - 复权因子来源于日线数据的 adj_factor 字段
      - 支持从原始价格反推前复权价格

   c) logging_config.py — 结构化日志
      - 基于 structlog，JSON 格式输出
      - 区分 audit/app/error/perf 四类日志
      - 提供 get_logger(name, log_type) 工厂函数

   d) config_loader.py — 配置加载器
      - ConfigLoader 类，支持加载 configs/ 下所有 YAML 文件
      - YAML 中的 ${ENV_VAR:default} 占位符支持环境变量覆盖
      - 热加载支持：风控阈值类配置监听 Redis pub/sub 频道 "config:risk:update"
        （如果 Redis 不可用则退化为文件轮询，30秒间隔）
      - 配置访问通过属性或字典方式：config.risk.thresholds.single_stock_max_ratio
      - 所有配置项有类型注解和默认值

3. configs/ 目录下所有 YAML 配置模板文件：
   按照实现指南第五章的规范创建：
   - boards/sse_main.yaml, boards/sse_star.yaml, boards/szse_main.yaml, 
     boards/szse_gem.yaml, boards/bse.yaml（板块交易规则）
   - models/long_term.yaml, models/intraday.yaml, models/premarket.yaml
   - factors/registry.yaml（因子注册，初始包含 momentum_20d, roe_ttm 两个示例因子）
   - risk/thresholds.yaml, risk/blacklist.yaml
   - data_sources.yaml（AkShare + Tushare 连接配置）
   - brokers.yaml（QMT 连接配置模板）
   - system.yaml（时区=Asia/Shanghai, 语言, 日志级别, 环境标识）

4. pyproject.toml 额外配置：
   - pytest 配置（testpaths = ["tests"], pythonpath = ["."]）
   - ruff 代码检查配置（line-length=100, target-version=py312）
   - mypy 类型检查配置（strict=true）

5. Makefile：
   make install    — pip install -e ".[dev]"
   make test       — pytest -xvs
   make test-cov   — pytest --cov=core --cov-report=term
   make lint       — ruff check . && mypy core/
   make clean      — 清理 __pycache__ 和 .pyc

6. .env.example：
   TUSHARE_TOKEN=
   REDIS_URL=redis://localhost:6379
   CLICKHOUSE_URL=http://localhost:8123
   MYSQL_URL=mysql://user:pass@localhost:3306/quant
   LOG_LEVEL=INFO
   ENV=dev

【约束条件】
- 所有 Python 代码使用 typing annotations（Python 3.12 语法）
- 类和函数都需要 docstring（Google 风格，简短即可）
- 日志使用 structlog，不要用 print
- 配置加载器要能处理文件不存在的情况（给出清晰的错误信息）
- 不要做超出本次任务范围的实现（如数据库连接、具体因子计算等后续批次才做）
```

---

## 第2批：数据层 — 数据库初始化 + 数据源适配器 + 质量检查

> **依赖**：第1批（公共工具 + 配置）
> **复杂度**：中
> **可并行**：数据源适配器之间可以并行开发（akshare.py 和 tushare.py 互不依赖）

```
【角色与上下文】
你是一个A股量化交易系统的数据层开发工程师。请先阅读：
- A股量化交易系统详细方案.md：第三章数据层设计
- 系统实现设计指南.md：第二节数据库表设计、第四节每日数据管线时序
- 已完成模块：core/common/ — calendar.py, config_loader.py, logging_config.py

【本次任务】
实现 core/data/ 下的数据层，包括：

1. core/data/db.py — 数据库连接管理
   - ClickHouseClient 类：连接池管理，提供 execute() 和 query() 方法
   - MySQLClient 类：基于 SQLAlchemy，提供 session 上下文管理器
   - RedisClient 类：连接池，提供 get/set/hgetall/publish 方法
   - 所有客户端支持从配置加载连接参数

2. core/data/sources/base.py — 数据源抽象基类
   - DataSourceBase 抽象类，定义接口：
     · get_daily_kline(start, end, codes=None) -> pd.DataFrame
     · get_minute_kline(date, codes=None) -> pd.DataFrame
     · get_financials(codes, report_dates=None) -> pd.DataFrame
     · get_margin_trading(date) -> pd.DataFrame
     · get_dragon_tiger(date) -> pd.DataFrame
     · get_auction_snapshot(date) -> pd.DataFrame
     · source_name -> str（数据源名称）
   - 每个方法返回的 DataFrame 有统一的列名规范

3. core/data/sources/akshare.py — AkShare 数据源适配器（免费主源）
   实现 DataSourceBase 的全部接口，数据从 akshare 库拉取。
   关键：实现 get_daily_kline() 时需包含 adj_factor（前复权因子）和 turnover（换手率）。
   对于 akshare 不支持的接口（如 auction_snapshot），raise NotImplementedError 并说明。

4. core/data/sources/tushare.py — Tushare 数据源适配器（备用/补充）
   实现 DataSourceBase 全部接口，数据从 tushare.pro_bar() 等接口拉取。
   需要 token 认证，从配置中读取 TUSHARE_TOKEN。

5. core/data/sources/fallback.py — 主备切换数据源
   - FallbackDataSource 类，接收 primary 和 fallback 两个 DataSourceBase 实例
   - 每次调用先尝试 primary，失败或超时（5秒）则切换 fallback
   - 记录切换事件到日志
   - 连续失败 N 次后默认使用 fallback（N 可配置，默认3）

6. core/data/pipeline.py — 数据管线
   - DataPipeline 类，负责每日定时数据采集的编排：
     · collect_daily_kline(date) — 采集日K线 → 数据质量检查 → 写入 ClickHouse
     · collect_financials() — 增量采集最新财报 → 质量检查 → 写入 ClickHouse
     · collect_margin_trading(date) — 采集融资融券 → 写入 ClickHouse
     · collect_dragon_tiger(date) — 采集龙虎榜 → 写入 ClickHouse
   - 支持断点续传：记录每次采集的 checkpoint，中断后从断点继续
   - 支持增量模式：只采集上次采集时间之后的新数据

7. core/data/quality.py — 数据质量检查
   - 行情到达率检查：当日应有N只股票，实收M只，缺失率超过阈值告警
     （应有数量基于上一交易日的正常交易股票数）
   - 价格合理性检查：
     · 涨跌幅不超过对应板块的涨跌停限制
     · open/high/low/close 逻辑关系：low <= open/close <= high 且 low <= high
     · 成交量 > 0（非停牌股票）
   - 财务数据勾稽校验：资产 = 负债 + 权益（允许 1% 误差）
   - 每个检查返回 QualityReport 对象：{passed: bool, issues: [{field, expected, actual, severity}]}
   - severity 分 WARN/ERROR，ERROR 级别的 issue 阻塞数据入库

【约束条件】
- 所有 DataFrame 列的命名规范遵循实现指南中的表 schema
- 数据源适配器内部处理数据清洗（去重、字段映射、类型转换）
- pipeline.py 不硬编码数据源，通过依赖注入 DataSourceBase 实例
- 每个模块附带单元测试（使用 mock 数据源，不联网）
- 数据源适配器的重试逻辑在 FallbackDataSource 中统一处理，不在单个适配器中实现
```

---

## 第3批：特征工程 — 因子注册、计算、验证

> **依赖**：第2批（数据层）、第1批（配置 + 日历）
> **复杂度**：中
> **可并行**：五个因子模块可以并行开发（technical.py, fundamental.py, capital_flow.py, sentiment.py, premarket.py）

```
【角色与上下文】
你是一个A股量化交易系统的特征工程开发工程师。请先阅读：
- A股量化交易系统详细方案.md：第四章特征工程（五类因子体系、规范性要求）
- 系统实现设计指南.md：configs/factors/registry.yaml 规范
- 已完成模块：
  · core/common/ — 日历、配置加载、日志
  · core/data/ — data_pipeline 提供 get_daily_kline() 等数据查询接口，
    ClickHouse 中有 market_daily, financial_statements 等表

【本次任务】
实现 core/features/ 下的特征工程模块。这是一组互不依赖的子任务，可以分批实现：

--- 子任务3A：因子注册与版本管理 ---
core/features/registry.py
- FactorRegistry 类：
  · 从 configs/factors/registry.yaml 加载因子定义
  · register(name, function, params, version) — 注册新因子
  · get_factor(name) -> FactorDefinition
  · list_factors(category=None) -> List[FactorDefinition]
  · get_dependencies(name) -> List[str] — 返回该因子的依赖因子（用于计算顺序）
- FactorDefinition 数据类：name, display_name, category, function, params, version, depends_on, missing_threshold
- 计算 DAG：根据因子间的 depends_on 关系自动推导计算拓扑序
- 版本管理：每次修改因子参数时生成新版本号（自动增量），旧版本数据不覆盖

--- 子任务3B：技术面因子 ---
core/features/technical.py
实现以下因子（按实现指南 4.1.1 节清单）：
- momentum_20d / momentum_60d：N日累计收益率
- volatility_20d：20日年化波动率
- amplitude_20d：20日平均振幅
- atr_14：14日平均真实波幅
- turnover_5d / turnover_20d：N日平均换手率
- volume_ratio：当日成交量 / 20日均量
- amihud_illiq：Amihud非流动性指标 |return|/amount
- rsi_14：14日RSI
- macd_dif / macd_dea / macd_hist：MACD指标
- bollinger_position：(close - lower_band) / (upper_band - lower_band)
- ma_alignment：短期(5)/中期(20)/长期(60)均线的多头/空头排列评分
- alpha_momentum_20d：剔除市场beta和行业收益后的纯alpha动量

每个因子函数签名：f(data: pd.DataFrame, params: dict) -> pd.Series
data 参数至少包含 code, trade_date, open, high, low, close, volume, amount, adj_factor

--- 子任务3C：基本面因子 ---
core/features/fundamental.py
实现以下因子：
- pe_ttm, pb, ps_ttm, pcf_ttm：估值因子
- peg：PE / 盈利增速
- dividend_yield：股息率
- ep_ttm：1/PE（盈利收益率）
- roe_ttm, roa_ttm, roic_ttm：盈利质量
- gross_margin_trend：毛利率近4季度趋势
- net_margin_ttm：净利率
- revenue_yoy, net_profit_yoy：成长性
- debt_ratio, current_ratio, quick_ratio：财务健康
- cf_ratio_ttm：经营现金流/净利润
- free_cf_yield：自由现金流/市值

★关键★ 所有基本面因子必须支持 Point-in-Time 模式：
- 通过 core/common/calendar.py 获取交易日历
- 在每个回测日期，只使用 announce_date <= 该日期的财报数据
- 参数 point_in_time=True 时启用，默认 True

--- 子任务3D：资金面因子 ---
core/features/capital_flow.py
实现：
- main_force_net_inflow_5d / _20d：主力资金累计净流入
- main_force_inflow_ratio：大单净流入/总成交额
- margin_balance_change_5d：融资余额近5日变化
- dragon_tiger_net_buy：最近一次龙虎榜机构净买入额分位数
- dragon_tiger_institution_count：最近龙虎榜机构席位数量
- northbound_quarter_change：北向资金季度持仓变化（仅季度频率！标记为低频因子）
- lockup_expiry_days：距下次限售解禁的天数，若90天内无解禁则返回默认值

--- 子任务3E：情绪/事件因子 ---
core/features/sentiment.py
实现：
- limit_up_count：全市场涨停家数
- limit_up_chain_height：最高连板高度
- limit_down_count：全市场跌停家数
- board_break_ratio：炸板率 = 曾涨停后开板数/曾涨停数
- limit_up_ratio：涨停家数/上涨家数（赚钱效应代理指标）
- auction_open_premium：个股集合竞价虚拟开盘价相对昨收的涨跌幅
- auction_volume_ratio：竞价匹配量/近20日均量
- performance_forecast_surprise：基于业绩预告类型（大增/略增/续盈/略减/大减）的量化映射

--- 子任务3F：盘前专属因子 ---
core/features/premarket.py
实现：
- overnight_adr_mapped：隔夜中概映射信号
  （需维护映射表：configs/premarket/adr_mapping.yaml）
- a50_futures_overnight：A50期货隔夜涨跌幅
- hsi_futures_overnight：恒指期货隔夜涨跌幅
- announcement_sentiment_score：公告NLP情绪评分（依赖第5批NLP模型，此处先预留接口）
- dragon_tiger_review_score：龙虎榜复盘综合评分（机构净买+游资参与度+买卖力量比）
- limit_up_review_signal：昨日涨停股今日高开概率（基于历史统计）
- theme_heat_score：盘前题材热度（新闻标题聚类→概念提及频率，依赖第5批NLP模型，预留接口）
- auction_strength_score：竞价强度综合评分（偏离度+量比+不平衡度）
- auction_fake_order_risk：竞价虚假申报风险评分（9:15-9:20申报量/最终匹配量的偏离）
- days_to_next_event：距下一重大事件的剩余天数

--- 子任务3G：特征工程规范性工具 ---
1. core/features/neutralizer.py
   - 行业中性化：对 factor_value ~ industry_dummies 做回归，取残差
   - 市值中性化：对 factor_value ~ log(market_cap) 做回归，取残差
   - 行业+市值双重中性化
   - 行业分类支持申万一级（默认）/中信一级/GICS，通过参数切换

2. core/features/validation.py
   - 缺失率检查：FactorMissingRateChecker
     · 单因子缺失率超过注册时的 missing_threshold → 告警
     · 全局因子缺失热力图
   - 异常值检测：MAD 方法（±5MAD 标记为异常）
   - 分布漂移检测：当前窗口因子值分布 vs 历史窗口的 KS 检验

3. core/features/time_travel_checker.py
   ★这是整个系统最重要的安全模块★
   - TimeTravelChecker 类：
     · check(date, data_source_info) -> bool
     · 给定一个回测日期，校验 data_source_info 中的所有数据的时间戳 <= date
     · 核心检查：财务数据中 report_date 和 announce_date 的关系
       绝不允许 announce_date > date 的数据被使用
     · 检查失败的调用抛出 TimeTravelError（自定义异常），附带详细的违规数据信息
     · 在 DEBUG 模式下记录每次检查的通过情况

--- 子任务3H：特征存储 ---
core/features/store.py
- FeatureStore 类：
  · save_factor_values(date, factor_name, values: {code: value}, version) 
    → 写入 ClickHouse factor_values 表
  · load_factor_values(start, end, factor_names, codes=None) -> pd.DataFrame
    → 从 ClickHouse 读取因子历史值（宽表格式：行=date*code，列=factor_names）
  · cache_to_redis(date, factor_name, values: {code: value})
    → 写入 Redis：key 模式 feat:{factor_name}:{code}，TTL 24小时
  · get_latest_from_redis(codes, factor_names) -> {code: {factor: value}}
    → 从 Redis 批量读取最新因子值（用于实时推理）

【约束条件】
- 所有因子函数使用相同的函数签名协议（见子任务3B模板）
- 因子计算内部不直接读写数据库——数据通过参数传入，计算结果通过 store 写入
- 每个因子模块附带单元测试：测试因子在正常数据、缺失数据、极端值数据下的输出
- TimeTravelChecker 的测试是最重要的——需要构造"含前视偏差的数据"来验证检查器能捕获
- 盘前因子中依赖 NLP 的预留接口，先用规则方法实现一版（如基于关键词的情绪评分），
  第5批 NLP 模块完成后可以无缝替换
- 因子版本管理的逻辑：参数变更 → 版本号递增 → 新值写入新行 → 回测时通过 manifest 锁定版本
```

---

## 第4批：回测引擎

> **依赖**：第3批（特征工程）、第2批（数据层）、第1批（公共工具）
> **复杂度**：高（整个系统中最复杂的模块）
> **可并行**：内部子模块 cost_model.py / constraints.py / rounding.py 可以和 engine.py 并行开发

```
【角色与上下文】
你是一个量化交易回测引擎开发工程师。请先阅读：
- A股量化交易系统详细方案.md：第七章回测系统
- 系统实现设计指南.md：第六节订单管理系统、第七节测试策略（回测可复现性）、第十节坑#1#3#4

【已有基础】
- core/common/calendar.py — 交易日历
- core/common/config_loader.py — 配置加载（含板块规则、风控阈值）
- core/data/pipeline.py — 数据查询接口（get_daily_kline 等）
- core/features/registry.py — 因子注册与计算
- core/features/store.py — 因子加载（load_factor_values）和缓存
- configs/boards/*.yaml — 各板块交易规则（涨跌幅、最小单位、价格笼子等）

【本次任务】
实现 core/backtest/ 下的完整回测引擎。

--- 子任务4A：交易成本模型 ---
core/backtest/cost_model.py

class TransactionCostModel:
    """
    精确建模A股所有交易成本，严格遵循方案2.4和2.5节定义。
    """
    def calculate(self, code: str, side: Literal['buy','sell'], 
                  price: float, shares: int, is_etf: bool = None) -> CostBreakdown:
        """
        参数：
        - code: 股票代码 '600519'
        - side: 买/卖
        - price: 成交价
        - shares: 成交股数
        - is_etf: 是否ETF（None则根据code前缀自动判断：51xxxx为ETF）

        成本计算规则：
        1. 印花税：仅卖出方 0.05%，ETF免征
        2. 佣金：万1.5，买卖双向；最低5元/笔（不可减免）
           例：3000元交易→万1.5→4.5元→实收5元
           例：50000元交易→万1.5→7.5元→实收7.5元
        3. 过户费：0.01‰（十万分之一），买卖双向
        4. 滑点估算（不实际扣除，用于评估策略可行性）：
           · 中证800成分股：0.1%
           · 其他主板：0.2%
           · 创业板/科创板：0.3%
           · 小盘股（<50亿市值）：0.5%

        返回 CostBreakdown 数据类：{commission, stamp_duty, transfer_fee, slippage_est, total}
        """

    def is_csi800(self, code: str) -> bool:
        """判断是否中证800成分股（用于滑点分层），基于配置中的成分股列表"""

    def estimate_slippage(self, code: str, price: float, shares: int, side: str) -> float:
        """估算滑点金额 = 成交金额 × 对应滑点率"""

class CostBreakdown:
    commission: float
    stamp_duty: float
    transfer_fee: float
    slippage_est: float
    total: float  # = sum of above

--- 子任务4B：交易约束检查 ---
core/backtest/constraints.py

class TradingConstraints:
    """
    检查每笔交易意图是否满足A股交易规则约束。
    """

    def __init__(self):
        self._today_bought: Set[str] = set()   # 当日买入的股票（T+1约束）
        self._limit_status: Dict[str, str] = {}  # 涨跌停状态 {code: 'limit_up'/'limit_down'/'normal'}

    def check_price_limit(self, code: str, price: float, date: date) -> Tuple[bool, str]:
        """
        检查价格是否在当日涨跌停范围内。
        返回 (合法, 原因)
        从配置中读取该股票对应板块的涨跌幅限制（区分ST股的时间段）。
        """

    def check_t_plus_1(self, code: str, side: str, date: date) -> Tuple[bool, str]:
        """
        T+1检查：当日买入的股票不能卖出。
        """

    def check_suspension(self, code: str, date: date) -> Tuple[bool, str]:
        """
        停牌检查：status='停牌'的股票不可交易。
        """

    def check_price_cage(self, code: str, price: float, board: str) -> Tuple[bool, str]:
        """
        价格笼子（连续竞价阶段，科创板/创业板适用）：
        - 买入申报价 ≤ 买入基准价 × 102%
        - 卖出申报价 ≥ 卖出基准价 × 98%
        超出范围→废单。
        """

    def check_limit_sealed(self, code: str, side: str) -> Tuple[bool, str]:
        """
        封板检查（坑#3）：
        - 涨停封板→买单不可成交（区分"触板"和"封板"）
        - 跌停封板→卖单不可成交
        封板判断：封单量/流通盘 > 阈值 且 未开板。
        """

    def check_all(self, code: str, side: str, price: float, 
                  date: date, board: str) -> ConstraintCheckResult:
        """汇总所有检查，任何一项不通过即拒绝。"""

    def mark_bought(self, code: str):
        """记录当日买入（用于T+1检查）"""

    def reset_day(self):
        """新交易日开始时调用，清空当日买入列表"""

class ConstraintCheckResult:
    passed: bool
    reject_reason: Optional[str]
    checks_detail: Dict[str, bool]  # {check_name: passed}

--- 子任务4C：整手化算法 ---
core/backtest/rounding.py

class ShareRounder:
    """
    将理想股数按板块规则向下取整到可执行单位。
    坑#4：整手化是迭代收敛过程，不是一次性取整。
    """

    def round_down(self, code: str, ideal_shares: float) -> int:
        """
        单只股票的整手化：
        - 主板：必须是100的整数倍
        - 创业板：100的整数倍
        - 科创板：≥200股，超200后1股递增（如201, 250）；<200→0（买不起）
        - 北交所：100的整数倍
        """

    def round_portfolio(self, 
                         ideal_weights: Dict[str, float],
                         total_capital: float,
                         current_prices: Dict[str, float],
                         account_id: int) -> Tuple[Dict[str, int], Dict[str, float], int]:
        """
        组合级整手化（核心算法）：

        输入：理想权重 {code: weight}, 总资金, 当前价格 {code: price}
        处理流程：
        1. 理想金额 = 权重 × 总资金
        2. 理想股数 = 理想金额 / 价格
        3. round_down 取整
        4. 检查每只是否达标：
           - 最小配置金额 ≥ max(5000元, 总资金×0.5%)
           - 单笔佣金占比 < 0.1%
        5. 不达标 → 砍掉 → 资金重新分配给剩余股票 → 回到步骤1
        6. 迭代直到所有保留的股票都满足约束
        返回：(最终整手持仓, 实际权重, 保留股票数量)
        """

--- 子任务4D：事件驱动回测引擎 ---
core/backtest/engine.py

class BacktestEngine:
    """
    A股事件驱动回测引擎。按交易日历逐日推进，每天按固定事件顺序处理。

    使用示例：
    engine = BacktestEngine(start_date, end_date, initial_capital, config)
    result = engine.run(strategy)
    """

    def __init__(self, start_date: date, end_date: date,
                 initial_capital: float, config: dict):
        self.cost_model = TransactionCostModel()
        self.constraints = TradingConstraints()
        self.rounder = ShareRounder()
        # ...

    def run(self, strategy: 'Strategy') -> BacktestResult:
        """
        每日事件处理顺序：
        ┌─ 开盘前 ─────────────────────────────────────┐
        │ 1. 加载当日行情数据                            │
        │ 2. 更新持仓市值（mark-to-market）               │
        │ 3. 更新涨跌停/停牌状态                         │
        │ 4. 加载当日因子值（通过 FeatureStore）           │
        ├─ 盘中 ───────────────────────────────────────┤
        │ 5. 调用 strategy.on_bar(date, features,        │
        │       positions, cash, constraints)            │
        │    → 返回 List[TradeIntent] {code, side,       │
        │       price, shares, signal_id}                │
        │ 6. 逐笔处理 TradeIntent：                      │
        │    a) constraints.check_all()                  │
        │    b) rounder.round_down()                    │
        │    c) cost_model.calculate()                  │
        │    d) 更新持仓 + 现金 + 当日买入列表             │
        ├─ 收盘前 ─────────────────────────────────────┤
        │ 7. T+1约束：检查"当日买入"集合中的卖出意图→拒绝  │
        │ 8. 收盘集合竞价(14:57-15:00)单独处理：          │
        │    部分未成交单在收盘价撮合                     │
        ├─ 收盘后 ─────────────────────────────────────┤
        │ 9. 记录当日快照（净值、持仓、现金、交易记录）      │
        │ 10. 检查组合级风控（回撤熔断）                  │
        │ 11. constraints.reset_day()                   │
        └──────────────────────────────────────────────┘
        """

    def _handle_trade(self, intent: TradeIntent, date: date) -> TradeRecord:
        """
        处理单笔交易：约束检查 → 整手化 → 撮合 → 成本计算 → 记录。
        返回 TradeRecord 包含成交详情（价格、股数、成本拆分）。
        """

class TradeIntent:
    signal_id: str
    code: str
    side: Literal['buy','sell']
    price: float        # 意图价格（限价单价格）
    shares: int          # 意图股数（整手化前的理想股数）

class TradeRecord:
    trade_id: str
    signal_id: str
    code: str
    side: str
    intent_price: float
    intent_shares: int
    filled_price: float
    filled_shares: int
    cost_breakdown: CostBreakdown
    status: Literal['filled','partial','rejected','unfilled']
    reject_reason: Optional[str]

class BacktestResult:
    """回测结果对象，支持链式分析"""
    daily_nav: pd.DataFrame      # 每日净值
    daily_positions: pd.DataFrame
    trade_records: pd.DataFrame  # 所有成交记录
    signals_history: pd.DataFrame # 所有信号记录

    def summary(self) -> dict:  # 年化收益/波动/夏普/IR/MDD/胜率/盈亏比
    def plot_nav(self) -> Figure:  # Plotly净值曲线
    def monthly_returns_heatmap(self) -> Figure:
    def to_report(self, path: str):  # 生成 Markdown 回测报告

--- 子任务4E：Walk-Forward 滚动验证 ---
core/backtest/walk_forward.py

class WalkForwardValidator:
    """
    时序滚动验证，防过拟合。
    """

    def __init__(self, train_window_years: int = 5, 
                 test_window_months: int = 3,
                 purge_days: int = 5):
        """
        - train_window_years: 训练窗口长度（年）
        - test_window_months: 测试窗口长度（月），每次前滚的步长
        - purge_days: 训练/测试之间的清除期（天）
          防止训练窗口末尾的数据泄露到测试窗口开头
        """

    def get_windows(self, start: date, end: date) -> List[Tuple[date, date, date, date]]:
        """返回 [(train_start, train_end, test_start, test_end), ...]"""

    def validate(self, model_train_fn, model_predict_fn, 
                 start: date, end: date) -> pd.DataFrame:
        """
        model_train_fn(train_start, train_end) -> model
        model_predict_fn(model, test_start, test_end) -> pd.DataFrame (预测结果)
        返回所有测试窗口的拼接预测结果
        """

--- 子任务4F：回测报告生成 ---
core/backtest/report.py

class BacktestReport:
    """
    从 BacktestResult 生成标准回测报告。
    """

    def __init__(self, result: BacktestResult, benchmark_code: str = '000300'):
        """benchmark_code: 比较基准，默认沪深300"""

    def metrics(self) -> dict:
        """
        计算标准指标集：
        - 年化收益率
        - 年化波动率
        - 夏普比率 (Sharpe Ratio)
        - 信息比率 (IR，相对基准)
        - 最大回撤 (MDD) 及回撤区间
        - Calmar比率
        - 胜率（盈利交易占比）
        - 盈亏比（平均盈利/平均亏损）
        - 平均持仓天数
        - 换手率（年化）
        - 总交易成本及占比
        """

    def generate(self, output_path: str):
        """生成 Markdown 报告 + Plotly 图表到 output_path"""

    def ic_analysis(self, factor_values: pd.DataFrame, 
                    forward_returns: pd.DataFrame) -> pd.DataFrame:
        """因子IC序列、ICIR、IC衰减曲线"""

【约束条件】
- 所有金额运算使用 Decimal 或整数（分为单位），不使用浮点。提供 to_decimal / from_decimal 转换函数
- 回测引擎支持注入自定义撮合模拟器（默认用"历史日线收盘价撮合"，
  可通过参数注入"tick回放撮合器"用于高频策略回测）
- 每笔 TradeRecord 记录完整的 signal_id → trade_id 链路，可追溯到原始信号
- constraints.py 中涨跌停限制从配置动态读取，不硬编码具体数值
- engine.py 的 run() 方法中途支持暂停/继续（通过保存状态到 JSON 文件）
- 每一个子模块都有独立的单元测试
- 边界测试必须覆盖：
  · 1000元/5000元/10000元/100万元订单的成本计算
  · 小资金账户(5万) + 高价股(茅台)的整手化 → 应该被砍掉
  · 涨停封板日的买单 → 应被拒绝
  · 停牌股票的交易意图 → 应被拒绝
  · T+1违规（当日买入当日卖） → 应被拒绝
  · 空回测段（某测试窗口无可交易信号） → 不崩
```

---

## 第5批：模型层 — LightGBM + NLP + 评估

> **依赖**：第3批（特征工程）、第1批（配置）
> **复杂度**：中
> **可并行**：长期/日内/盘前/NLP 四个模型模块可并行开发

```
【角色与上下文】
你是一个量化交易系统的模型层开发工程师。请先阅读：
- A股量化交易系统详细方案.md：第五章模型层设计、第六章盘前推荐子系统
- 系统实现设计指南.md

【已有基础】
- core/features/ — 所有因子已实现，通过 FeatureStore 加载
- core/features/store.py — load_factor_values(start, end, factor_names, codes)
- core/features/registry.py — 因子注册和拓扑序
- configs/models/*.yaml — 模型超参配置
- configs/factors/registry.yaml — 因子注册

【本次任务】
实现 core/models/ 下的模型层。

--- 子任务5A：模型基类 ---
core/models/base.py

class BaseModel(ABC):
    """所有模型的抽象基类"""
    model_name: str
    model_version: str

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, 
              sample_weights: Optional[pd.Series] = None):
        """训练模型"""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """推理（返回原始分数/概率）"""

    def save(self, path: str):
        """保存模型到文件（joblib/ pickle）"""

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """从文件加载模型"""

    def get_params(self) -> dict:
        """获取模型参数"""

--- 子任务5B：长期选股模型 ---
core/models/long_term.py

class LongTermRanker(BaseModel):
    """
    基于 LightGBM LambdaRank 的多因子横截面排序模型。
    用途：月度全市场扫描，输出排序分数（越高越好）。
    """

    def __init__(self, config_path: str = "configs/models/long_term.yaml"):
        """
        从配置文件加载超参：num_leaves, learning_rate, max_depth,
        feature_fraction, bagging_fraction, min_data_in_leaf 等
        """

    def prepare_labels(self, data: pd.DataFrame, 
                       forward_period: int = 20) -> pd.Series:
        """
        标签构造：未来N个交易日的累计收益率（前复权）。
        也可以用分组排序标签（LambdaRank 所需格式）。
        """

    def train(self, X, y, sample_weights=None, 
              groups: Optional[pd.Series] = None):
        """
        groups: 日期分组（同一天的数据必须在同一个query组内，LambdaRank要求）
        """

    def predict_ranks(self, X: pd.DataFrame) -> pd.Series:
        """返回排序分数（越高越好，rank按降序）"""

    def feature_importance(self) -> pd.DataFrame:
        """特征重要性（gain/split双维度）"""

class LongTermClassifier(BaseModel):
    """
    LightGBM 分类模型：预测未来N日是否跑赢基准（中证800）。
    输出跑赢基准的概率（0~1）。
    """

class LinearBaseline(BaseModel):
    """
    多因子线性基线模型：加权或OLS回归。
    用途：作为非线性模型的最小可解释基线——先跑通线性再上LightGBM。
    """

--- 子任务5C：日内预测模型 ---
core/models/intraday.py

class IntradayClassifier(BaseModel):
    """
    日内涨/跌/平三分类 LightGBM 模型。
    输入：实时特征快照 + "距收盘剩余时间"特征
    输出：三分类概率 {up: p1, flat: p2, down: p3}
    """

    def prepare_labels(self, data, target_time: str = 'close'):
        """
        标签构造：当前时刻→目标时刻（收盘）的价格变化方向。
        target_time='close' 时目标为收盘价；也可以是 'next_30min'。
        """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回 (n_samples, 3) 的概率矩阵"""

class IntradayQuantileRegressor:
    """
    分位数回归模型（LightGBM quantile objective）：
    输出收盘价的条件分位数（如 P10, P50, P90），
    用于获得预测区间而非单一预测值。
    """

class DistributionPredictor:
    """
    NGBoost / XGBoostLSS 包装器：全分布建模，输出不确定性估计。
    实验性质，作为分位数回归的升级替代方案。
    """

--- 子任务5D：盘前推荐模型 ---
core/models/premarket/

1. overnight_mapping.py — OvernightMappingModel
   - LightGBM 回归/分类模型
   - 输入：隔夜海外市场特征 + 个股特征 → 输出：次日开盘涨跌方向及幅度
   - 标签：次日开盘价 vs 前日收盘价的涨跌幅

2. gap_classifier.py — GapClassifier
   - LightGBM 三分类 + 概率校准（CalibratedClassifierCV）
   - 输入：个股特征 + 公告NLP特征 → 输出：高开/平开/低开概率
   - 标签分桶：>+1%高开, -1%~+1%平开, <-1%低开
   - 概率必须做 Platt Scaling 校准（校准后的概率才能用于后续融合）

3. fusion_ranker.py — FusionRanker
   - LightGBM LambdaRank
   - 输入：长期排序分数(复用) + 全部隔夜/盘前信号 → 输出：综合推荐排序
   - 标签：当日日内最大涨幅（作为"值得关注"的代理标签）
   - 关键设计：长期评分是软特征输入而非硬约束——强隔夜信号可突破低长期评分

4. auction_anomaly.py — AuctionAnomalyDetector
   - Isolation Forest + 规则引擎
   - 输入：竞价快照特征 → 输出：异常评分（0=正常, 1=高度可疑）
   - 规则层：9:15-9:20申报量远大于最终匹配量的认作"虚假申报"

--- 子任务5E：NLP 模型 ---
core/models/nlp.py

class NLPSentimentAnalyzer:
    """
    Qwen3 系列模型封装，用于中文金融文本情绪分析。
    支持三种尺寸，按场景选择：
    - FinSenti-Qwen3-0.6B (INT8, CPU可运行, <50ms) — 实时
    - Qwen3-8B rLoRA (GPU, 批量) — 日频批处理
    - DeepSeek-R1-Distill-Qwen-7B — 复杂研报离线
    """

    def __init__(self, model_name: str = "FinSenti-Qwen3-0.6B", 
                 device: str = "auto"):
        """
        自动检测GPU可用性，优先GPU，回退CPU。
        模型首次加载时从 HuggingFace/ModelScope 下载。
        """

    def analyze_single(self, text: str, source: str = "announcement") -> SentimentResult:
        """
        单条文本情绪分析。source 影响 prompt 的构造方式：
        - 'announcement': 上市公司公告（正式、结构化）
        - 'news': 财经新闻（非正式、可能有标题党）
        - 'research': 券商研报（长、有深度）

        返回：{sentiment: 'positive'/'neutral'/'negative', 
                confidence: 0.0~1.0, 
                event_type: str|None,  # 仅 announcement/news 模式
                summary: str}          # 一句话摘要
        """

    def analyze_batch(self, texts: List[Tuple[str, str]]) -> List[SentimentResult]:
        """
        批量分析。texts 中每项为 (text, source)。
        使用 vLLM 批量推理，GPU 批处理优化。
        硬超时时间：30 秒（防止拖过盘前推荐窗口）。
        """

    def extract_event_type(self, text: str) -> Optional[str]:
        """
        事件分类：{业绩超预期, 重大合同, 增减持, 重组进展, 
                   分红送转, 风险警示, 其他}
        """

class SentimentResult:
    sentiment: Literal['positive','neutral','negative']
    confidence: float
    event_type: Optional[str]
    summary: str

--- 子任务5F：模型评估 ---
core/models/evaluation.py

class ModelEvaluator:
    """模型评估工具箱"""

    @staticmethod
    def rank_ic(predictions: pd.Series, returns: pd.Series) -> float:
        """Rank IC (Spearman correlation)"""

    @staticmethod
    def icir(ic_series: pd.Series) -> float:
        """ICIR = IC均值 / IC标准差"""

    @staticmethod
    def ic_decay(predictions: pd.Series, 
                 forward_returns: pd.DataFrame) -> pd.Series:
        """IC在不同前向期的衰减曲线"""

    @staticmethod
    def calibration_curve(y_true, y_prob, n_bins=10) -> Tuple[np.ndarray, np.ndarray]:
        """概率校准曲线（Reliability Diagram）"""

    @staticmethod
    def layered_returns(predictions: pd.Series, returns: pd.Series, 
                        n_quantiles: int = 10) -> pd.DataFrame:
        """
        分层收益分析：按预测分位数分组，计算每组的平均收益。
        如果收益单调递增 → 模型排序能力好。
        """

    @staticmethod
    def direction_accuracy(y_true_direction, y_pred_direction) -> float:
        """方向准确率（日内模型的关键评估指标）"""

    @staticmethod
    def rolling_monitoring(predictions, returns, window=20) -> pd.DataFrame:
        """Rolling IC 监控：用于模型衰减检测"""

【约束条件】
- 模型 save/load 使用 joblib（LightGBM原生支持），版本号纳入文件名
- predict() 返回 numpy array，不返回 DataFrame（减少序列化开销）
- NLP 模型首次加载时自动处理下载，不需要用户手动准备模型文件
- 每个模型类附带一个简短的 smoke test（生成随机数据→训练→预测→不崩溃即可）
- 盘前融合排序模型的输入中，"长期评分"通过参数传入（来自 LongTermRanker.predict_ranks），
  而非在 FusionRanker 内部重新计算——保证解耦
- 所有模型训练前固定随机种子（np.random.seed + lightgbm random_state），
  同时将种子值存入模型元信息
```

---

## 第6批：组合管理 + 风控引擎

> **依赖**：第3批（特征工程）、第1批（配置 + 日历 + 板块规则）
> **复杂度**：中
> **可并行**：组合管理和风控引擎可以并行开发

```
【角色与上下文】
你是一个量化交易系统的组合管理与风控模块开发工程师。请先阅读：
- A股量化交易系统详细方案.md：第九章仓位管理、第十章退出机制、第十三章风险最小化
- 系统实现设计指南.md：第二节 position 表、第六节 OMS 链路

【已有基础】
- core/common/calendar.py — 交易日历
- core/common/config_loader.py — 风控阈值配置（risk/thresholds.yaml）
- core/features/ — 因子值（通过 store 加载）
- configs/risk/thresholds.yaml — 风控阈值
- configs/boards/*.yaml — 板块规则（含涨跌停）

【本次任务】

--- 子任务6A：市场状态识别 ---
core/portfolio/regime.py

class MarketRegimeDetector:
    """
    市场状态识别，用于动态调整总仓位水平。

    不依赖主观感觉，综合四个客观指标：
    """

    def __init__(self, config: dict):
        """从配置加载各项参数（均线窗口、分位阈值等）"""

    def detect(self, date: date) -> RegimeResult:
        """
        综合四个维度：

        1. 趋势信号 (0~1分)：
           - 指数(上证/沪深300)与200日均线的关系
           - 均线多空排列（5/20/60/120日）
           → 多头排列→1分，空头排列→0分，交叉→0.5分

        2. 波动率信号 (0~1分)：
           - 已实现波动率在历史(3年)中的分位数
           - 高波动(>80分位)→低分，低波动(<20分位)→高分
           → 实现方式：rolling 252日波动率 → rank 分位

        3. 估值信号 (0~1分)：
           - 全市场PE/PB在历史(10年)中的分位数
           - 低估值(<30分位)→高分，高估值(>70分位)→低分
           → 分位数来自 PE-TTM / PB 的滚动窗口

        4. 市场宽度信号 (0~1分)：
           - 上涨家数占比
           - 创20日新高家数占比
           - 涨停家数温度（涨停数/全市场比例的分位）

        综合 = 加权平均（权重可配置，默认等权）
        返回 RegimeResult：{regime_label, score, dimension_scores, suggested_position_ratio}
        """
        pass

class RegimeResult:
    regime_label: str  # "牛市高波动" / "牛市低波动" / "熊市高波动" / "震荡市" / "极端恐慌"
    composite_score: float  # 0.0(极度悲观) ~ 1.0(极度乐观)
    dimension_scores: Dict[str, float]  # {trend: x, volatility: y, valuation: z, breadth: w}
    suggested_position_ratio: float  # 建议的总仓位比例

--- 子任务6B：组合优化器 ---
core/portfolio/optimizer.py

class PortfolioOptimizer:
    """
    将持仓权重或排序分数转化为实际可执行的仓位方案。
    提供多个优化策略，由简到繁。
    """

    def __init__(self, config: dict):
        """加载单票上限、行业上限等约束"""

    def equal_weight(self, scores: Dict[str, float], 
                     top_n: int = 15) -> Dict[str, float]:
        """
        等权（基线策略）。
        取 score 最高的 top_n 只股票，每只等权。
        """

    def inverse_vol_weight(self, scores: Dict[str, float],
                           volatilities: Dict[str, float],
                           top_n: int = 15) -> Dict[str, float]:
        """
        波动率倒数加权（推荐第二阶段使用）。
        权重 = 1/vol / sum(1/vol)，波动越低的股票权重越高。
        显著改善回撤。
        """

    def risk_parity(self, scores: Dict[str, float],
                    returns: pd.DataFrame,
                    top_n: int = 15) -> Dict[str, float]:
        """
        风险平价：使每只股票对组合的风险贡献相等。
        输入 returns 为历史收益率矩阵（过去60~120日）。
        使用迭代算法求解风险平价权重。
        """

    def min_cvar(self, scores: Dict[str, float],
                 returns: pd.DataFrame,
                 top_n: int = 15,
                 alpha: float = 0.05) -> Dict[str, float]:
        """
        最小CVaR优化：在预期收益>=0的约束下最小化CVaR。
        使用 Ledoit-Wolf 收缩协方差估计（而非样本协方差）来降低估计误差。
        """

    def signal_weighted(self, scores: Dict[str, float],
                        max_single: float = 0.20) -> Dict[str, float]:
        """
        按信号强度/置信度加权。
        max_single: 单票上限（超过的截断并重分配）。
        """

    def apply_constraints(self, weights: Dict[str, float],
                          industry_map: Dict[str, str],
                          max_industry: float = 0.40) -> Dict[str, float]:
        """
        后处理：施加行业集中度约束（单行业≤40%），超过的截断并重分配。
        """

--- 子任务6C：凯利公式 ---
core/portfolio/kelly.py

class KellyCalculator:
    """
    凯利公式（股票连续收益版）：f* = μ / σ²
    实际使用半凯利（f = 0.5 × f*）或 1/4 凯利。
    """

    def calculate_f_star(self, returns: pd.Series, 
                         risk_free_rate: float = 0.02) -> float:
        """
        输入历史收益率序列，计算最优仓位比例 f*。
        μ = 年化超额收益, σ² = 年化方差
        """

    def half_kelly(self, returns, rfr=0.02) -> float:
        """半凯利：f = 0.5 × f*"""

    def quarter_kelly(self, returns, rfr=0.02) -> float:
        """1/4凯利：f = 0.25 × f*（最保守，推荐新手使用）"""

--- 子任务6D：退出监控 ---
core/portfolio/exit_monitor.py

class ExitMonitor:
    """
    监控持仓的退出条件，分为长期仓和短期仓两套逻辑。
    """

    def __init__(self, config: dict):
        """加载止损止盈参数"""

    # --- 短期仓退出（机械执行） ---
    def check_stop_loss(self, code: str, entry_price: float,
                        current_price: float, atr: float) -> ExitSignal | None:
        """
        ATR 倍数止损 / 固定比例止损。
        只能往盈利方向移动，永不下移或取消。
        """

    def check_take_profit(self, code: str, entry_price: float,
                          current_price: float, highest_price: float) -> ExitSignal | None:
        """
        固定目标止盈 / trailing stop（从最高点回撤X%就卖）。
        """

    def check_time_stop(self, code: str, entry_date: date, 
                        current_date: date, max_hold_days: int) -> ExitSignal | None:
        """时间止损：持有超过N天仍未盈利或未达预期→退出"""

    # --- 长期仓退出（逻辑检查，非机械止损） ---
    def check_logic_breakdown(self, code: str, 
                               fundamental_factors: Dict[str, float]) -> ExitAlert | None:
        """
        逻辑检查清单：
        - ROE 连续两季下降 >30%
        - 经营现金流/净利润 < 0.5 且持续恶化
        - 负债率超过行业均值2σ
        - 出现财务造假/违规公告
        → 不自动卖出，生成 ExitAlert 供人工确认
        """

    def check_valuation_extreme(self, code: str, pe_ttm: float,
                                 pe_history: pd.Series) -> ExitAlert | None:
        """估值严重高估（超出历史2σ）→ 预警"""

class ExitSignal:
    code: str
    exit_type: str   # 'stop_loss' / 'take_profit' / 'time_stop'
    reason: str
    urgency: str     # 'immediate'(短期仓) / 'review'(长期仓)

class ExitAlert:
    code: str
    alert_type: str  # 'logic_breakdown' / 'valuation_extreme'
    detail: str
    requires_human_confirm: bool = True

--- 子任务6E：风控引擎 ---
core/risk/pre_trade.py  — 事前风控
core/risk/portfolio_risk.py  — 组合级风控
core/risk/extreme_scenario.py  — 极端行情预案

class PreTradeRiskChecker:
    """
    事前风控检查（下单前同步调用，<10ms）。
    这部分后续第8批会用C++重写，但Python版先跑通逻辑。
    """

    def check(self, code: str, side: str, price: float, shares: int,
              account_id: int, current_positions: Dict, 
              current_cash: float) -> RiskCheckResult:
        """
        检查清单：
        1. 单票上限 ≤20%（硬上限25%）
        2. 行业集中度 ≤40%
        3. 日累计申报笔数 < 20000
        4. 秒申报笔数 < 300（使用滑动窗口计数器）
        5. 撤单率 ≤70%（1秒窗口）
        6. 涨跌停/停牌状态
        7. 可用资金 ≥ 所需资金 + 预估成本
        8. 黑名单过滤（ST/退市风险/合规禁止池）
        9. 单笔金额上限 ≤ 50万（可配置）
        """

class PortfolioRiskManager:
    """
    组合级实时风控。
    """

    def check_drawdown(self, current_nav: float, peak_nav: float) -> RiskAction | None:
        """
        回撤熔断：
        - 当日回撤 > 15% → 强制减短期仓至0
        - 5日滚动回撤 > 25% → 强制减全仓至30%
        """

    def check_var(self, positions: Dict, returns_matrix: pd.DataFrame,
                  confidence: float = 0.95) -> float:
        """计算组合 VaR（历史模拟法）"""

    def vol_targeting(self, current_vol: float, target_vol: float,
                      current_position_ratio: float) -> float:
        """
        波动率目标：adjustment = target_vol / current_vol
        当前波动率 > 目标 → 降仓；当前波动率 < 目标 → 可适当加仓
        返回建议的目标仓位比例
        """

class ExtremeScenarioHandler:
    """
    极端行情下的系统行为（坑#8：「卖不出去」是必然场景）。
    """

    def on_circuit_breaker(self, positions: Dict, 
                           market_status: str) -> List[TradeIntent]:
        """
        回撤熔断触发时：
        1. 优先减可成交、流动性好的持仓（按Amihud非流动性排序）
        2. 减仓顺序：交易仓(全部)→核心仓(至30%)
        """

    def on_liquidity_crisis(self, positions: Dict) -> List[TradeIntent]:
        """
        千股跌停/停牌潮：
        1. 优先减可成交的（哪些还能卖→按流动性排序）
        2. 无法卖出的持仓→计算股指期货对冲手数(IF/IC/IM)
        3. 预设的资金缓冲确保不会被强平
        结果记录到 extreme_scenario.log
        """

class RiskCheckResult:
    passed: bool
    reject_reason: Optional[str]
    checks_detail: Dict[str, bool]
    warn_only: List[str]  # 不拒绝但需告警的项

class RiskAction:
    action_type: str  # 'reduce_trading' / 'reduce_all' / 'hedge_with_futures' / 'halt'
    target_positions: Dict[str, int]
    reason: str

【约束条件】
- 风控检查中涉及金额的计算使用 Decimal
- 所有风控拒绝必须记录到 risk_log 表（包含检查时的完整上下文）
- 回撤熔断可以配置为"自动执行"或"告警后人工确认"两种模式
- 市场状态的阈值（如"极度恐慌"的定义）全部配置化，不从代码中写死
- 退出信号格式与第4批定义的 TradeIntent 保持一致，可直接传入 OMS
```

---

## 第7批：微服务

> **依赖**：第4批（回测引擎）、第5批（模型层）、第6批（组合管理 + 风控）
> **复杂度**：高（涉及多个独立可部署服务，服务间通信复杂）
> **可并行**：各微服务可以拆分给多个 agent 并行开发

```
【角色与上下文】
你是一个量化交易系统的微服务开发工程师。请先阅读：
- 系统实现设计指南.md：第一节代码结构(services/)、第三节API契约、第四节管线时序

【已有基础】
- core/ 下所有模块已实现（data, features, models, backtest, portfolio, risk, signals, common）
- 所有服务的核心逻辑在 core/ 中，services/ 只负责编排、调度、对外暴露接口
- Redpanda topic 列表和 gRPC proto 定义见实现指南第三节

【本次任务】
实现 services/ 下的所有微服务。由于互相不依赖，可以拆分给多个 agent 并行开发。

============================================================
微服务共7个，请按以下分组并行开发：
- 组A: data_collector (1个agent)
- 组B: feature_server (1个agent)
- 组C: long_term_service + order_manager (1个agent, 2个服务)
- 组D: premarket_service (1个agent)
- 组E: intraday_service + risk_engine (1个agent, 2个服务)
- 组F: nlp_service + monitor (1个agent, 2个服务)
============================================================

=== 组A：data_collector ===
services/data_collector/main.py
- 基于 APScheduler 的定时任务调度器
- 注册 cron 任务：
  · 15:05: collect_daily_close  — 日K线 + 龙虎榜 + 融资融券
  · 15:30: run_quality_checks   — 数据质量检查
  · 06:00-07:00: collect_overnight  — 每隔15分钟拉取一次，直到全部到齐
  · 07:00: collect_premarket_announcements  — 盘前公告增量
  · 09:15: start_auction_stream  — 启动竞价数据流监听
- 每个任务失败后指数退避重试（最多3次）
- 任务执行状态上报到 monitor 服务（通过 Redpanda topic: system.health）
- Dockerfile：python:3.12-slim 基础镜像，依赖通过 requirements.txt 安装

=== 组B：feature_server ===
services/feature_server/main.py
- gRPC 服务端，暴露 GetFeatures RPC 方法
- 实现两个计算模式：
  1. 实时流计算（订阅 market.tick → DolphinDB 流引擎 → 增量更新因子）
  2. 批量计算（定时任务：收盘后批量重算日频因子）
- 因子计算结果写入 Redis（key: feat:{factor_name}:{code}）
- 因子版本管理：计算时从 FactorRegistry 获取当前因子版本
- 启动时预加载所有因子定义到内存
- 健康检查端点：返回最近一次计算时间、成功/失败统计

=== 组C：long_term_service + order_manager ===

1. services/long_term_service/main.py
   - 长期选股服务（定时任务驱动，非实时）
   - 月度调仓流程（每月最后一个交易日收盘后触发）：
     · 加载全市场最新因子值
     · LongTermRanker.predict_ranks() → 全市场排序
     · PortfolioOptimizer.inverse_vol_weight() 或 risk_parity() → 目标权重
     · ShareRounder.round_portfolio() → 整手持仓
     · 与当前持仓 diff → 调仓信号列表
     · 信号发送到 Redpanda topic: signals.long_term
   - 月度重训练流程（每月第一个交易日盘后触发）：
     · WalkForwardValidator 计算新窗口
     · LongTermRanker.train() 重训练
     · 保存新模型版本 → 部署
   - 每周生成持仓报告（Markdown 格式，发送到用户）

2. services/order_manager/main.py
   - OMS 服务（常驻进程）
   - 订阅 Redpanda topics：signals.long_term, signals.intraday, signals.exit
   - 信号处理流水线（对应实现指南第六节）：
     · 告融合去重（signal_id 幂等）
     · 事前风控检查（gRPC 同步调用 risk_engine）
     · 资金/仓位计算（调用 core/portfolio/rounding.py）
     · 订单生成 → 调用 QMT SDK 下单
     · 成交回报监听 → 更新持仓
     · TODO部分（仅成交）→ 撤单后重试1次
   - 所有订单记录写入 order_log 表
   - 订单状态机：pending → submitted → partial_filled → filled / cancelled / rejected
   - 异常处理：网络超时最多重试1次，重试前先查订单状态（防止重复下单）

=== 组D：premarket_service ===
services/premarket_service/main.py
- 盘前推荐服务（每日 08:00 定时触发，cron: 0 8 * * 1-5）
- 主流水线（对应实现指南第四节时序）：
  1. 加载长期评分（从 Redis 读取最新 long_term_ranker 输出）
  2. 计算盘前因子（调用 feature_server 或直接调用 core/features/premarket.py）
  3. 如有隔夜公告 → nlp_service.BatchAnalyze() → 公告情绪 + 事件分类
  4. OvernightMappingModel.predict() → 开盘方向预估
  5. GapClassifier.predict_proba() → 高开/平开/低开概率
  6. FusionRanker.predict_ranks() → 综合推荐排序
  7. 生成推荐清单 JSON → Redpanda topic: signals.premarket
  8. 通知用户（企业微信webhook / APP推送）
- 硬截止时间：08:45，超时丢弃未完成任务，释放 GPU 资源
- 09:15-09:25：订阅 market.auction topic → AuctionAnomalyDetector
  → 实时修正推荐（标记异常标的、调节置信度）
  → 09:25 发送 signals.premarket (final=true)
- 降级方案：如果 NLP 服务不可用，公告情绪退化为关键词规则方法
- 每日推荐结果存档（JSON 文件，以日期命名）

=== 组E：intraday_service + risk_engine ===

1. services/intraday_service/main.py
   - 日内预测服务（常驻进程，09:25-15:00 活跃）
   - 09:25：加载今日盘前推荐清单作为关注池
   - 09:30-15:00：订阅 market.minute topic
     · 每分钟对关注池内的股票计算实时特征
     · IntradayClassifier.predict_proba() → 涨/跌/平概率
     · 信号满足条件（如 up_prob > 0.6 且 confidence > 0.5）→ 生成信号
     · 信号发送到 Redpanda topic: signals.intraday
   - 收盘集合竞价(14:57-15:00)：单独模型，预测集合竞价跳变方向
   - 日终：当日信号汇总报告
   - 关注池是软约束——非关注池的股票如果出现极强的盘口信号（如 >3σ），也可以生成信号

2. services/risk_engine/main.py
   - 风控引擎服务（常驻进程）
   - 暴露 gRPC 接口：CheckOrder(同步, <10ms), 组合风险查询
   - 实时监控：
     · 订阅所有成交回报 → 实时更新净值 → 检查回撤熔断线
     · 订阅 signals.* → 统计申报/撤单率 → 检查合规阈值
   - 告警触发：
     · 风控拒绝率突增（5分钟内>10次拒绝）→ Redpanda topic: risk.alert
     · 回撤接近熔断线（距离<2%）→ risk.alert
     · 申报笔数接近合规上限（>80%）→ risk.alert
   - 熔断触发时的自动执行逻辑（调用 order_manager 的紧急平仓接口）
   - 支持配置热加载（监听 Redis pub/sub 频道 config:risk:update）

=== 组F：nlp_service + monitor ===

1. services/nlp_service/main.py
   - NLP 推理服务（常驻进程，vLLM 托管 Qwen3 模型）
   - 暴露 gRPC 接口：
     · AnalyzeSentiment(text, source) → SentimentResult
     · BatchAnalyze([(id, text, source)]) → [{id, SentimentResult}]
   - vLLM 推理配置：INT8 量化，max_model_len=4096，gpu_memory_utilization=0.85
   - 启动时加载模型到 GPU（WarmUp）
   - GPU 调度优先级：日内实时推理 > 盘前批量 > 离线研报分析
   - 健康检查：返回模型加载状态、GPU 利用率、最近一次推理延迟
   - 降级方案：如果 GPU OOM，small 模型（0.6B）可回退到 CPU 推理

2. services/monitor/main.py
   - 监控告警服务（常驻进程）
   - 订阅 Redpanda topics：system.health, risk.alert, orders.fill
   - 告警集成：
     · 严重级别(CRITICAL)：钉钉群机器人 + 电话（预留接口）
     · 错误级别(ERROR)：钉钉群机器人 + 邮件
     · 警告级别(WARN)：企业微信
   - 仪表板数据源：通过 Prometheus metrics endpoint 暴露核心指标
     http://monitor:9090/metrics
   - 每日收盘后，汇总当日关键指标（成交额、信号数、最大回撤等）生成日报
   - 健康检查面板：显示所有服务的最近心跳时间
   - Grafana 仪表板 JSON：3个面板
     · 实时交易面板：账户总览、持仓分布、实时行情
     · AI模型性能面板：预测vs真实、特征重要性、Rolling IC
     · 策略回测面板：净值曲线、月度热力图、滚动夏普

【约束条件】
- 所有服务通过环境变量获取配置（Redpanda broker 地址、gRPC 端口、Redis URL 等）
- 每个服务独立 Dockerfile（python:3.12-slim），通过 docker-compose.yml 编排
- 服务间通信使用 gRPC（同步）+ Redpanda（异步），不直接调用其他服务的 HTTP API
- 所有服务的 main.py 包含 graceful shutdown（SIGTERM 处理）
- 服务健康检查统一格式：{service: str, status: 'healthy'/'degraded'/'unhealthy', uptime: float, last_error: str|None}
- 每个服务附带 README.md：服务职责、启动方式、依赖的其他服务、环境变量清单
- 任务之间的边界：order_manager 是唯一一个可以调用券商API的服务（其他服务不能直接下单）
```

---

## 第8批：C++ 组件 + 测试完善 + 部署配置

> **依赖**：第7批（微服务，特别是 risk_engine 和 order_manager）
> **复杂度**：中（C++ 编程 + 性能优化）
> **可并行**：C++ 组件之间、与 Python 测试之间可以并行

```
【角色与上下文】
你是一个量化交易系统的高性能组件与部署工程师。请先阅读：
- 系统实现设计指南.md：第一节代码结构(cpp/)、第九节部署运维
- 方案文档第六~七章回测和风控的性能要求

【已有基础】
- services/risk_engine/ — 风控服务（Python版，已实现逻辑）
- services/order_manager/ — OMS 服务（已实现订单路由逻辑）
- services/data_collector/ — 数据采集（已有 tick 数据流接收逻辑）
- 所有 Python 层代码已实现并通过测试

【本次任务】

--- 子任务8A：C++ 事前风控检查器 ---
cpp/risk_checker/

将 PreTradeRiskChecker（Python）中延迟敏感的检查项用 C++ 重写：
- 单票上限、行业集中度、申报笔数检查
- 涨跌停范围计算
- 黑名单查找（std::unordered_set）

要求：
- 编译为共享库 librich_checker.so (Linux) / librich_checker.dylib (macOS)
- 通过 pybind11 暴露到 Python
- Python 侧接口不变：check(code, side, price, shares, ...) -> RiskCheckResult
- 提供 C++ benchmark：连续调用 100000 次，P99 延迟 < 100 微秒
- CMakeLists.txt 配置：C++17，-O3 优化，依赖 spdlog（日志）, pybind11
- 需要附带 smoke test（Python 侧调用验证结果与 Python 版一致）

--- 子任务8B：tick 行情解码器 ---
cpp/tick_processor/

- 从 UDP multicast 接收交易所 L2 行情组播包
- 解码 FAST 协议（上交所）或二进制格式（深交所）
- 转换为统一的内存数据结构（TickData struct）
- 通过共享内存（shared memory）暴露给 Python 进程，零拷贝
- 提供 Python 绑定：ctx = TickReceiver(multicast_addr); tick = ctx.recv()

--- 子任务8C：订单路由适配器 ---
cpp/order_router/

- QMT SDK 的 C++ 封装层
- 暴露 Python 接口：submit_order(code, side, price, qty) -> order_id
- 内置重试逻辑：第一次超时后重试，但重试前先查订单状态
- 网络异常处理：超时 100ms，超时后不重试，标记为 UNKNOWN 状态，人工确认

--- 子任务8D：完善测试 ---
tests/

完善以下测试（大部分模块在第2-6批已有单元测试，这一步补充缺失的）：

1. tests/integration/
   - test_data_pipeline.py：从 akshare 拉取真实数据 → 质量检查 → 写入 ClickHouse
     （使用 Docker Compose 中的测试 ClickHouse 实例）
   - test_signal_to_order.py：信号生成 → OMS → 风控 → 下单 → 成交回报，全链路
     （使用 mock 的 QMT SDK）
   - test_premarket_flow.py：隔夜数据采集 → NLP → 因子计算 → 模型推理 → 输出
     （使用 mock 的 NLP 服务和前一日真实数据）
   - test_long_term_rebalance.py：月度调仓全流程

2. tests/regression/
   - test_lookahead.py：构造含前视偏差的数据 → 验证 TimeTravelChecker 能捕获
   - test_survivorship.py：构造含退市股的完整历史数据 → 回测验证退市股交易记录存在
   - test_reproducibility.py：固定 seed + 锁定数据 → 跑两次回测 → 对比净值和每笔成交 checksum
   - test_cost_boundaries.py：
     · 1000元买单→佣金=5元（最低）; 50000元→佣金=7.5元; 100万元→佣金=150元
     · 印花税仅卖方; ETF免印花税

--- 子任务8E：部署配置 ---
根目录文件：

1. docker-compose.yml
   - 所有微服务的编排
   - 包含 Redpanda（单节点，开发环境）、ClickHouse（单节点）、MySQL、Redis
   - 服务间网络隔离（backend 网络）
   - 环境变量注入
   - 健康检查（healthcheck 指令）
   - volumes 挂载：./data:/data（行情数据持久化）

2. docker-compose.prod.yml
   - 生产环境覆盖配置
   - Redpanda 3节点集群
   - ClickHouse 2分片×2副本
   - Redis Cluster 3主3从
   - MySQL 主从
   - C++ 组件直接宿主机部署（非容器）
   - 资源限制（CPU/memory limits）

3. .env.example（完善版）
   所有服务需要的环境变量，分类注释

4. docs/runbook.md
   - 系统启动步骤
   - 各服务依赖关系图
   - 常见故障排查：
     · 数据源不可达 → 检查 FallbackDataSource 日志 → 切换备用源
     · 行情延迟 → 检查 Redpanda consumer lag → 扩分区
     · 订单发送失败 → 检查 QMT 连接状态 → 重启 order_manager
     · 模型预测异常 → 检查 Rolling IC → 触发重训练
   - 降级操作手册：
     · 盘前推荐服务故障 → 使用前一日推荐清单
     · NLP 服务故障 → 退化为关键词规则
     · 风控引擎故障 → 停止所有自动交易，手工模式

5. scripts/
   - init_db.py：执行 ClickHouse 和 MySQL 的建表 SQL
   - daily_run.sh：每日日程入口（crontab 注册）
   - weekly_report.sh：周报生成脚本
   - check_health.sh：遍历所有服务的 /health 端点
   - backup_db.sh：数据库备份脚本

【约束条件】
- C++ 代码遵循 Google C++ Style
- 所有 C++ 编译产物附带版本号（从 Git tag 自动获取）
- 集成测试在 CI 中通过 Docker Compose 启动完整环境运行
- runbook.md 是给运维人员看的，用中文写，步骤要具体到命令
```
