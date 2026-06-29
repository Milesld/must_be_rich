#!/usr/bin/env bash
# 后台跑 4 个池的因子搜索（串行，避免并发拉取限流/原生崩溃）。
# 用法见文件末尾注释。

set -u
cd "$(dirname "$0")/.."   # 切到项目根目录（脚本放在 scripts/ 下）

POOLS=(semiconductor robotics apple_chain ai_app)
ROUNDS=100

for cfg in "${POOLS[@]}"; do
  echo "==================== ${cfg} $(date) ===================="
  python research/factor_optimizer.py \
    --task long_term \
    --config "configs/strategy_${cfg}.yaml" \
    --rounds "${ROUNDS}" \
    > "optimizer_${cfg}.log" 2>&1
  echo "==================== ${cfg} done $(date) ===================="
done

echo "ALL DONE $(date)"

# ── 如何运行 ──
#   后台运行并把总进度写入 optimizer_all.log：
#     nohup bash scripts/run_optimizers.sh > optimizer_all.log 2>&1 &
#   每个池的详细输出在 optimizer_<池名>.log。
#   监控： tail -f optimizer_all.log
#   停止： kill <启动时打印的 PID>
