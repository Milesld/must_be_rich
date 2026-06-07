#!/bin/bash
# 清理所有运行时生成的数据，回到初始状态
# 使用方式: ./scripts/clean_data.sh [--force]
#   --force: 不询问直接清理

set -e

if [ "$1" != "--force" ]; then
    echo "将清理以下数据目录:"
    echo ""
    echo "  ~/.quant_system/calendar/         交易日历缓存"
    echo "  data/checkpoints/                 数据管线断点"
    echo "  data/models/                      训练好的模型"
    echo "  data/reports/                     持仓报告"
    echo "  data/premarket_recommendations/   盘前推荐"
    echo "  data/logs/                        系统日志"
    echo "  data/backups/                     数据库备份"
    echo "  .pytest_cache/                    测试缓存"
    echo "  lightgbm/                         本地安装的 lightgbm"
    echo "  lightgbm-*.dist-info/             lightgbm 元信息"
    echo ""
    read -p "确认清理？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消"
        exit 0
    fi
fi

echo "正在清理..."

# 交易日历缓存
rm -rf ~/.quant_system/calendar/
echo "  ✓ ~/.quant_system/calendar/"

# 项目运行时数据
for d in data/checkpoints data/models data/reports data/premarket_recommendations data/logs data/backups; do
    if [ -d "$d" ]; then
        rm -rf "$d"
        echo "  ✓ $d"
    fi
done

# 测试缓存
rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Python缓存"

# LightGBM 本地安装
rm -rf lightgbm/ lightgbm-*.dist-info/ 2>/dev/null || true

echo ""
echo "清理完成。项目回到初始状态。"
echo ""
echo "如需重新开始："
echo "  1. 确保 lightgbm wheel 在 ~/Downloads/ 中"
echo "  2. 运行: python research/run_backtest_demo.py"
