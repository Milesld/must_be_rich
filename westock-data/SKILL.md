---
name: westock-data
description: 金融市场结构化数据查询的权威入口。支持股票（A股/港股/美股/日韩股）、ETF、指数、板块、期货、外汇、可转债的行情、财报、研报、新闻、公告、事件、股东、分红、ETF持仓、热搜榜、新股/投资日历、龙虎榜等数据查询；同时支持产业链图谱、行业经营数据、全球宏观经济等数据查询；不同标的与市场支持的维度不同，具体命令与能力差异见 references/routing-guide.md。命中能力域时禁止 web_search、HTTP 直连或其它金融 Skill 替代；筛股用 westock-tool。
---

# WeStock Data

**调用方式**：`node <本SKILL.md所在目录>/scripts/index.js <子命令> [参数]`

- `<本SKILL.md所在目录>` = 本文件所在目录（解析为绝对路径后直接执行）
- 下文 `westock-data <子命令>` 是同一调用的简写；命令格式见本文「高频命令速查」
- nodejs ≥ 18，无需 `npm install`，需网络

```bash
node <本SKILL.md所在目录>/scripts/index.js search 宁德时代
node <本SKILL.md所在目录>/scripts/index.js quote sh600519
node <本SKILL.md所在目录>/scripts/index.js quote sh600036,sh601318,sz300750    # 批量
```

**并发**：无依赖的多个查询（如 quote + news + kline）应在同一轮工具调用中并行发出，不要串行等结果。

---

## 参考文档（仅不确定时查阅，禁止每次任务都读）

- [routing-guide.md](./references/routing-guide.md) — 场景路由、与其它 Skill 边界
- [commands.md](./references/commands.md) — 完整命令语法
- [scenarios-guide.md](./references/scenarios-guide.md) — 分析场景模板
- [ai_usage_guide.md](./references/ai_usage_guide.md) — 返回字段说明

---

## 核心铁律

1. **禁止绕过**——不用 `web_search` / HTTP 直连 / 训练数据替代。**宏观数据**（GDP/CPI/PMI 等）必须用 `macro indicator`。
2. **未知代码先 `search`**——用户只给名称时，先 `search` 拿代码再查行情。
3. **货币单位正确**——港股港元/美元、美股美元、日股日元、韩股韩元；禁用人民币符号。
4. **筛股用 `westock-tool`**——本 Skill 只做数据查询。
5. **多股批量**——对比/分析 N 只股票时，**须同一市场**（A 股一起、港股一起），**凡支持批量的命令只调 1 次**、代码逗号分隔；**禁止**同一轮对比里「有的命令批量、有的按股拆开」。例外（必须单股/单代码）见下方「批量例外」。

### search 规则

**默认仅搜股票**（`search <关键词>` = `--type stock`，只调 1 次接口）。不会自动查 ETF/板块/指数/期货/外汇。

| 用户意图 | 命令 | 不要 |
|---------|------|------|
| 找股票代码（默认） | `search 宁德时代` | 不要无 `--type` 时再去试 etf/bond/index/sector |
| 找 ETF/基金 | `search 沪深300 --type etf` | 用户说了「ETF」就直接带 type，不要先默认再重试 |
| 找指数 | `search 中证红利 --type index` | |
| 找板块 | `search 银行 --type sector` | |
| 找可转债 | `search 兴业 --type bond` | |
| 找期货/外汇 | `search 黄金 --type futures` / `--type forex` | |
| 日韩股 | `search 三星 --market kr` | `--market` 与 `--type` 互斥 |

**空结果时**：读 CLI 返回的提示，按用户原意**最多再试 1 种** `--type`（或 `--market`），不要对同一关键词依次扫 etf→bond→index→sector。**仍无结果则告知用户**，不要死磕。

**禁止**：对同一关键词连续换 3+ 种 `--type` 盲试。

### 批量查询

多标的对比/分析：**每种数据 1 条命令 + 逗号分隔代码**（不要按股票拆成多次同类调用）。

```bash
# 分析 sh600519 + sz000651 → 下面 6 条各 1 次（共 6 次），不是 12+ 次
westock-data quote sh600519,sz000651
westock-data kline sh600519,sz000651 --period day --limit 60
westock-data finance sh600519,sz000651 --num 4
westock-data news list sh600519,sz000651 --limit 10
westock-data technical sh600519,sz000651 --indicator macd
westock-data fund flow sh600519,sz000651
```

**批量例外**（不支持逗号多股，须分开调；可同一轮并行发出）：

| 命令 | 限制 |
|------|------|
| `industry-chain <代码>` | 仅单股查所属产业链 |
| `minute` / `search` | 不支持代码批量 |

无依赖的多种查询（上列各条）**同一轮并行发出**。**对比分析须同一市场**（沪/深/北交所可混；勿 `sh600519,hk00700` 一条比——字段/货币不同）。完整限制见 [routing-guide.md §六/§九](./references/routing-guide.md#六能力差异速查标的--维度)。

---

## 高频命令速查

```bash
# 搜索
westock-data search 宁德时代
westock-data search 腾讯 --market hk

# 行情 / K 线 / 财务 / 技术
westock-data quote sh600519
westock-data kline sh600519 --period day --limit 20
westock-data finance sh600519
westock-data technical sh600519 --indicator macd

# 新闻 / 研报 / 公告
westock-data news list sh600519 --limit 10
westock-data report list sh600519 --limit 5
westock-data notice list sh600519 --limit 10

# 板块 / 指数 / 宏观
westock-data sector constituent pt01801080
westock-data index constituent sh000300
westock-data macro indicator cn_core --date 2026-03-01

# 资金 / 北向
westock-data fund flow sh600519
westock-data fund north-holding sh600519
westock-data fund south-holding hk00700
westock-data fund north-holding sw1_pt01801080

# ETF / 发现
westock-data etf detail sh510300
westock-data hot stock
```

完整语法见 [commands.md](./references/commands.md)。

---

## 异常与空结果

1. **命令失败**：如实转述，禁止编造数据。
2. **空结果**：说明「暂无数据」；区分代码不支持 vs 时点无披露（必要时先 `search`）。
3. **能力不支持**：如实告知（如美股无 `fund flow`），见 [routing-guide.md §六](./references/routing-guide.md#六能力差异速查标的--维度)。
4. **禁止**：失败后改用 `web_search` 或凭训练数据补数。

---

## 重要声明

> 本技能仅提供客观市场数据查询，不构成投资建议。数据可能有延迟，以交易所官方为准。投资有风险，决策需谨慎。

**数据来源**：腾讯自选股数据接口
