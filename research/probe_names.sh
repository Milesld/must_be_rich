#!/bin/bash
# 核对补齐用的 4 个替代指数码到底是什么指数(名称)，防止选错板块。
# 因 search 对计算机/房地产/家电只返回 cs 码(拉不到)，替代码是靠 kline 能拉反推的，
# 必须确认其真实行业名称与钢铁/计算机/房地产/家电一致。
set -u
cd "$(dirname "$0")/.." || exit 1
IDX=westock-data/scripts/index.js

echo "用 quote 查各码名称（quote 通常带指数名）："
for c in sz399440 sz399363 sz399393 sz399996 \
         sh000935 sz399608 sh000971 sz399948 sh000952 sz399997 sh000992; do
  name=$(node "$IDX" quote "$c" --raw 2>/dev/null \
    | grep -oE '"(name|cn_name|zh_name)": *"[^"]*"' | head -1)
  printf "  %-12s %s\n" "$c" "$name"
done
echo ""
echo "若 quote 无名称字段，改用 search 反查（按码在结果里找名）已知不行，"
echo "则贴出任一码的完整 quote 原文：node $IDX quote sz399363 --raw"
