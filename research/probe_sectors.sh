#!/bin/bash
# 探测 钢铁/计算机/房地产/家居家电 在 westock 里可用的指数代码。
# 背景：原配置里 4 个 cs 前缀码(cs930606/cs930651/cs931775/cs931241)
# westock 不认，kline 一律 fetch failed。用本脚本找 sz399/sh000 前缀替代码。
# 用法：bash research/probe_sectors.sh
set -u
cd "$(dirname "$0")/.." || exit 1
IDX=westock-data/scripts/index.js

echo "================ 0) 网络健康探针（已知能拉）================"
probe() {
  local out
  out=$(node "$IDX" kline "$1" --period day --start 2024-01-01 --end 2024-02-28 --fq qfq --raw 2>/dev/null)
  local n; n=$(echo "$out" | grep -o '"date"' | wc -l | tr -d ' ')
  local f; f=$(echo "$out" | grep -o 'fetch failed' | head -1)
  printf "  %-12s rows=%s %s\n" "$1" "$n" "${f:+[NET-FAIL]}"
}
probe sz399986   # 中证银行，正常应 rows>0
echo "  ↑ 若上面是 [NET-FAIL]，说明当前网络到腾讯行情被墙/限流，先解决网络再跑下面"
echo ""

echo "================ 1) search 查指数码 ================"
for name in 钢铁 计算机 房地产 家用电器; do
  echo "---- search '$name' --type index ----"
  node "$IDX" search "$name" --type index --raw 2>/dev/null \
    | grep -oE '"(code|name)": *"[^"]*"' | paste - - | head -10
done
echo ""

echo "================ 2) 候选码 kline 验证（2024-01~02，rows>0 即可用）================"
echo "[钢铁候选]";   for c in sz399440 sh000986 sz399959; do probe $c; done
echo "[计算机候选]"; for c in sz399363 sh000935 sz399608 sh000971; do probe $c; done
echo "[房地产候选]"; for c in sz399948 sh000952 sz399393; do probe $c; done
echo "[家电候选]";   for c in sz399997 sh000992 sz399996; do probe $c; done
echo ""
echo "把 rows>0 的码告诉我，我改进 configs/sector_rotation.yaml 后重跑。"
