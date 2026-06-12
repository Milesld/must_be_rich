# 系统操作指南

> 本文回答"怎么用这个系统"的所有问题——从首次初始化到日常运行再到清理。

---

## 一、脚本清单

| 脚本 | 功能 | 需要网络 | 需要 Docker |
|------|------|---------|------------|
| `scripts/setup.sh` | 首次初始化：检查依赖、创建目录、加载交易日历、跑测试 | ✓ | ✗ |
| `scripts/init_db.py` | 创建 ClickHouse + MySQL 表结构 | ✗ | ✓(需DB实例) |
| `scripts/daily_run.sh` | 全天自动运行（06:00→16:00） | ✓ | ✓ |
| `scripts/check_health.sh` | 检查所有服务 /health 端点 | ✗ | ✓ |
| `scripts/backup_db.sh` | 备份 MySQL + ClickHouse schema + Redis RDB | ✗ | ✓ |
| `scripts/clean_data.sh` | 清理运行时数据，回到初始状态 | ✗ | ✗ |
| `research/run_backtest_demo.py` | 研究模式：直接用 Python 跑回测 | ✓(首次) | ✗ |

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

# 因子自动优化 — 搜索最优因子组合和权重
# Optuna TPE 自适应搜索（推荐，收敛快）
python research/factor_optimizer.py --task long_term --rounds 200
# 随机搜索（无 optuna 时自动回退）
python research/factor_optimizer.py --task long_term --rounds 100

# 不同任务用不同的因子池
python research/factor_optimizer.py --task long_term    # 长期选股（30个因子）
python research/factor_optimizer.py --task premarket    # 盘前推荐（16个因子）
python research/factor_optimizer.py --task intraday     # 日内预测（19个因子）
```

**数据流向**：akshare 在线拉取 → 内存中 → 直接回测/模型推理。数据不会写入数据库。

**需要做什么**：只需要 `python` + `pip install` 几个核心依赖。不需要 Docker、不需要 MySQL、不需要 Redis。

### 模式B：实盘模式（Docker Compose 一键启动全部微服务）

适合：真实交易日的全自动化运行。

```bash
# 第1步：（首次）初始化数据库表
python scripts/init_db.py
# 需要 ClickHouse 和 MySQL 实例运行中（docker-compose up 已启动它们）

# 第2步：启动所有服务
docker-compose up -d

# 第3步：验证启动
./scripts/check_health.sh

# 第4步：注册每日自动运行（选做）
crontab -e
# 添加这行：
# 0 6 * * 1-5 cd /path/to/project && docker-compose up -d && sleep 30 && ./scripts/daily_run.sh
```

**服务运行后**：
- 每天 06:00 自动采集隔夜数据
- 每天 08:00 自动生成盘前推荐 → 推送到 `signals.premarket`
- 每天 09:30-15:00 日内预测实时运行 → 推送 `signals.intraday`
- 每天 15:00 后自动日终结算 → 生成日报
- 月末自动触发月度调仓
- order_manager 收到信号后执行真实下单（需要 QMT 已连接且 `QMT_ENABLED=true`）

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
python scripts/init_db.py       # 建表
docker-compose up -d             # 启动全部服务
./scripts/check_health.sh        # 验证
# 等一个完整交易日后检查 data/logs/ 确认无错误
```

### "系统跑了几个月，磁盘满了"

```bash
du -sh data/*                    # 看哪个目录最大
rm -rf data/logs/*.log           # 日志通常是罪魁祸首
rm -rf data/backups/*/           # 或者清理旧备份
```

### "我想卸载，彻底删除"

```bash
./scripts/clean_data.sh --force  # 清理运行时数据
docker-compose down -v           # 停止并删除容器+数据卷
rm -rf ~/.quant_system/          # 交易日历缓存
cd .. && rm -rf quant-system/    # 删除项目本身
```
