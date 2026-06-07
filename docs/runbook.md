# 量化交易系统运维手册 (Runbook)

## 一、系统启动

### 1.1 首次启动

```bash
# 1. 克隆项目并安装依赖
cd quant-system
pip install -e ".[dev]"

# 2. 初始化数据库
python scripts/init_db.py

# 3. 启动所有服务
docker-compose up -d

# 4. 验证启动
./scripts/check_health.sh
```

### 1.2 服务依赖关系

```
                    ┌─────────────┐
                    │   monitor   │ ← 消费所有健康/告警事件
                    └──────┬──────┘
                           │
  ┌────────────┬───────────┼───────────┬────────────┐
  │            │           │           │            │
  ▼            ▼           ▼           ▼            ▼
data_     feature_    premarket_   intraday_    long_term_
collector  server      service      service      service
  │            │           │           │            │
  └────────────┴───────────┴───────────┴────────────┘
                           │
                    ┌──────┴──────┐
                    │  Redpanda   │ ← 消息总线
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         risk_engine   order_manager  nlp_service
```

### 1.3 停止系统

```bash
# 正常停止
docker-compose down

# 停止并清理数据卷（危险！）
docker-compose down -v
```

---

## 二、常见故障排查

### 2.1 数据源不可达

**症状**: data_collector 日志出现连续采集失败

**排查步骤**:
1. 检查 FallbackDataSource 日志: `docker-compose logs data_collector | grep "失败"`
2. 检查网络连通性: 在容器内执行 `curl -I https://finance.sina.com.cn`
3. 检查 AkShare 版本: `python -c "import akshare; print(akshare.__version__)"`

**修复**:
```bash
# 方案A: 切换到备用源
# 修改 configs/data_sources.yaml: primary -> "tushare", fallback -> "akshare"

# 方案B: 更新 AkShare
pip install --upgrade akshare

# 方案C: 重启数据采集服务
docker-compose restart data_collector
```

### 2.2 行情延迟

**症状**: intraday_service 日志显示 "行情延迟 > 1s"

**排查步骤**:
1. 检查 Redpanda consumer lag:
   ```bash
   docker exec -it quant-system-redpanda-1 rpk topic describe market.minute
   ```
2. 检查网络延迟: `ping <broker_host>`

**修复**:
```bash
# 增加 topic 分区数
docker exec quant-system-redpanda-1 rpk topic alter-config market.minute --set partitions=6

# 重启消费者
docker-compose restart intraday_service
```

### 2.3 订单发送失败

**症状**: order_manager 日志出现 "下单失败" / "QMT连接超时"

**排查步骤**:
1. 检查 QMT 连接状态:
   ```bash
   curl http://localhost:${QMT_HTTP_PORT}/status  # 取决于 QMT 版本
   ```
2. 检查 QMT 账户登录状态、可用资金
3. 查看 order_log 表最近错误:
   ```sql
   SELECT * FROM order_log WHERE status='rejected' ORDER BY submitted_at DESC LIMIT 10;
   ```

**修复**:
```bash
# 重启 order_manager
docker-compose restart order_manager

# 如果 QMT 崩溃，重启 QMT 后再重启 order_manager
# QMT 重启步骤参照券商文档
```

### 2.4 模型预测异常

**症状**: 监控面板 Rolling IC 连续N周 < 0.02；模型预测值突然跳变

**排查步骤**:
1. 检查模型版本和加载时间:
   ```bash
   curl http://localhost:50051/health | jq '.model_version'
   ```
2. 检查预测值分布: Grafana → AI模型性能面板 → 预测分布散点图
3. 对比线上/离线特征一致性: feature_server 日志

**修复**:
```bash
# 触发手动重训练
docker-compose exec long_term_service python -m services.long_term_service.main --task retrain

# 如果因子版本不匹配，更新因子注册表后重启 feature_server
docker-compose restart feature_server
```

### 2.5 Redis 连接断开

**症状**: 所有服务报 Redis connection refused

**修复**:
```bash
docker-compose restart redis
# 等待 Redis 就绪后重启依赖服务
docker-compose restart feature_server long_term_service premarket_service intraday_service
```

---

## 三、降级操作手册

### 3.1 盘前推荐服务故障

**症状**: premarket_service 在 08:45 前未生成推荐报告

**操作**:
1. 使用前一日推荐清单: 系统自动回退到 `data/premarket_recommendations/` 下最新 JSON
2. 如果前一日推荐也不可用:
   ```bash
   # 手动生成简化推荐（仅基于长期评分 + 等权）
   python -c "
   from core.portfolio.optimizer import PortfolioOptimizer
   opt = PortfolioOptimizer()
   # 从 Redis 读取最新长期评分...
   "
   ```

### 3.2 NLP 服务故障

**症状**: nlp_service 不可用（GPU OOM / 模型加载失败）

**操作**:
1. 系统自动降级: 公告情绪退化为关键词规则
2. 设置环境变量强制使用关键词模式:
   ```bash
   export NLP_MODEL=keyword
   docker-compose up -d nlp_service
   ```
3. 无 GPU 时使用 CPU 推理 (0.6B 模型):
   ```bash
   export NLP_MODEL=FinSenti-Qwen3-0.6B
   export NLP_DEVICE=cpu
   ```

### 3.3 风控引擎故障

**症状**: risk_engine /health 不可达，或 CheckOrder 超时

**操作**:
1. **停止所有自动交易**:
   ```bash
   docker-compose stop order_manager intraday_service
   ```
2. 切换到手工模式: 系统信号仅作参考展示，不自动下单
3. 检查风控引擎日志:
   ```bash
   docker-compose logs risk_engine --tail 100
   ```
4. 修复后重启:
   ```bash
   docker-compose restart risk_engine
   docker-compose start order_manager intraday_service
   ```

---

## 四、日常运维

### 4.1 每日检查清单

| 时间 | 检查项 | 命令 |
|------|--------|------|
| 06:30 | 隔夜数据到齐 | `docker-compose logs data_collector --since 6h \| grep "完成"` |
| 08:50 | 盘前推荐已生成 | `ls -la data/premarket_recommendations/` |
| 09:30 | 日内服务已启动 | `./scripts/check_health.sh` |
| 盘中 | 行情无延迟 | Grafana → 实时交易面板 |
| 15:30 | 日终数据入库 | `docker-compose logs data_collector --since 1h \| grep "入库"` |

### 4.2 每周维护

```bash
# 数据库备份
./scripts/backup_db.sh

# 模型性能周报
docker-compose exec monitor python -m services.monitor.main --generate-weekly-report

# 清理旧日志
find data/logs -name "*.log" -mtime +30 -delete
```

### 4.3 磁盘空间管理

```bash
# 检查各数据卷占用
docker system df -v

# 清理 ClickHouse 旧分区（保留最近3年）
# ClickHouse 自动 TTL 策略在建表时配置
```

---

## 五、账户与权限

| 组件 | 用户 | 用途 |
|------|------|------|
| MySQL | quant/quant | 业务数据（订单、持仓、账户） |
| ClickHouse | quant/quant | 行情和基本面历史数据 |
| Redis | 无认证 | 因子缓存、特征值 |
| Redpanda | 无认证(内网) | 消息总线 |
| Monitor (9090) | 公开 | Prometheus metrics |
| QMT | 券商账户 | 交易下单（需50万+资金门槛） |

---

## 六、紧急联系方式

| 角色 | 渠道 | 场景 |
|------|------|------|
| 系统告警 | 钉钉群机器人 | 自动推送 CRITICAL/ERROR |
| 开发者 | 企业微信 | 非紧急通知 |
| 券商技术支持 | 电话 | QMT 故障、交易异常 |
