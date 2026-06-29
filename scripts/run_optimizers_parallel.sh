#!/usr/bin/env bash
# 并行跑 4 个池的因子搜索。
# 每池一个后台进程，各自独立 log；FETCH_WORKERS 限制每池数据拉取并发，
# 避免 4池×并发 把数据源打爆（总并发 = 4 × FETCH_WORKERS）。
# 用法见文件末尾注释。

set -u
cd "$(dirname "$0")/.."   # 切到项目根目录（脚本放在 scripts/ 下）

POOLS=(semiconductor robotics apple_chain ai_app)
ROUNDS=100
export FETCH_WORKERS="${FETCH_WORKERS:-4}"   # 每池数据拉取进程数（可被外部覆盖）

echo "并行启动 ${#POOLS[@]} 个池，每池 FETCH_WORKERS=${FETCH_WORKERS} $(date)"

pids=()
for cfg in "${POOLS[@]}"; do
  python research/factor_optimizer.py \
    --task long_term \
    --config "configs/strategy_${cfg}.yaml" \
    --rounds "${ROUNDS}" \
    > "optimizer_${cfg}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "  ${cfg}: PID ${pid} → optimizer_${cfg}.log"
done

echo "全部已后台启动，等待完成..."
fail=0
for i in "${!POOLS[@]}"; do
  if wait "${pids[$i]}"; then
    echo "  ✓ ${POOLS[$i]} done $(date)"
  else
    echo "  ✗ ${POOLS[$i]} FAILED (见 optimizer_${POOLS[$i]}.log) $(date)"
    fail=1
  fi
done

echo "ALL DONE $(date)"
exit $fail

# ── 如何运行 ──
#   并行后台运行（4 池同时，总进度写入 optimizer_all.log）：
#     nohup bash scripts/run_optimizers_parallel.sh > optimizer_all.log 2>&1 &
#   每个池的详细输出在 optimizer_<池名>.log。
#   想更激进/保守，覆盖每池并发：
#     FETCH_WORKERS=8 nohup bash scripts/run_optimizers_parallel.sh > optimizer_all.log 2>&1 &
#   监控总进度： tail -f optimizer_all.log
#   监控某池：   tail -f optimizer_semiconductor.log
#   停止全部：   kill $(pgrep -f factor_optimizer.py)
