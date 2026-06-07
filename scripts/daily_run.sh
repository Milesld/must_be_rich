#!/bin/bash
# 每日自动运行入口 — crontab 注册:
#   0 6 * * 1-5 /app/scripts/daily_run.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOGDIR="$PROJECT_DIR/data/logs"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── 06:00 隔夜数据采集 ──────────
log "=== 隔夜数据采集 ==="
python -m services.data_collector.main 2>&1 | tee -a "$LOGDIR/overnight.log"

# ── 08:00 盘前推荐 ─────────────
log "=== 盘前推荐 ==="
python -m services.premarket_service.main 2>&1 | tee -a "$LOGDIR/premarket.log"

# ── 09:30-15:00 日内预测 ───────
log "=== 日内预测服务就绪 ==="
python -m services.intraday_service.main 2>&1 | tee -a "$LOGDIR/intraday.log" &

INTRA_PID=$!

# ── 15:05 日终数据采集 ──────────
# 等待收盘后执行
WAIT_UNTIL="15:05"
log "等待 $WAIT_UNTIL 执行日终采集..."
while [ "$(date +%H:%M)" '<' "$WAIT_UNTIL" ]; do sleep 60; done

log "=== 日终数据采集 ==="
python -m services.data_collector.main --task daily_close 2>&1 | tee -a "$LOGDIR/daily_close.log"

# ── 15:30 数据质量检查 ──────────
sleep 1500  # 15:25→15:30
log "=== 数据质量检查 ==="
python -m services.data_collector.main --task quality 2>&1 | tee -a "$LOGDIR/quality.log"

# ── 16:00 持仓快照 + 日报 ────────
log "=== 日内信号汇总 ==="
python -m services.monitor.main --generate-report 2>&1 | tee -a "$LOGDIR/daily_report.log"

# ── 月末最后一个交易日: 月度调仓 ──
if python -c "
from core.common.calendar import get_calendar
from datetime import date
cal = get_calendar()
print('yes' if cal.is_month_end(date.today()) else 'no')
" | grep -q yes; then
    log "=== 月度调仓 ==="
    python -m services.long_term_service.main --task rebalance 2>&1 | tee -a "$LOGDIR/rebalance.log"
fi

# 清理
kill $INTRA_PID 2>/dev/null || true
log "=== 本日运行完成 ==="
