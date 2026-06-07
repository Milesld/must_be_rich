#!/bin/bash
# 数据库备份 — crontab: 0 2 * * 0 /app/scripts/backup_db.sh

set -e
BACKUP_DIR="data/backups/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

echo "=== 数据库备份 $(date '+%Y-%m-%d %H:%M:%S') ==="

# MySQL 备份
echo "备份 MySQL..."
docker exec quant-system-mysql-1 mysqldump -u quant -pquant quant \
    account position signal_log order_log risk_log \
    > "$BACKUP_DIR/mysql_dump.sql" 2>/dev/null
echo "  → $BACKUP_DIR/mysql_dump.sql ($(wc -c < "$BACKUP_DIR/mysql_dump.sql") bytes)"

# ClickHouse 备份（仅元数据）
echo "备份 ClickHouse..."
docker exec quant-system-clickhouse-1 clickhouse-client \
    --query "SHOW CREATE TABLE market_daily" > "$BACKUP_DIR/ch_schema.sql" 2>/dev/null
echo "  → $BACKUP_DIR/ch_schema.sql"

# Redis RDB 备份
echo "备份 Redis..."
docker exec quant-system-redis-1 redis-cli BGSAVE > /dev/null 2>&1
sleep 2
cp data/redis/dump.rdb "$BACKUP_DIR/redis_dump.rdb" 2>/dev/null || echo "  (Redis 无持久化文件)"
echo "  → $BACKUP_DIR/redis_dump.rdb"

# 清理超过30天的备份
find data/backups -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \; 2>/dev/null || true

echo "备份完成: $BACKUP_DIR"
