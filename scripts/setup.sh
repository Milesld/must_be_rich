#!/bin/bash
# 首次初始化 — 一条命令完成项目启动前的所有准备
# 使用方式: ./scripts/setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║   量化交易系统 — 首次初始化              ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 第1步：检查 Python 环境 ──────────
echo "→ 第1步：检查 Python 环境..."
python -c "import pandas, numpy, sklearn, yaml" 2>/dev/null && echo "  ✓ 核心依赖已就绪" || {
    echo "  ✗ 缺少依赖，请先安装: pip install pandas numpy scikit-learn pyyaml akshare tushare"
}

# ── 第2步：检查 lightgbm ─────────────
echo "→ 第2步：检查 lightgbm..."
python -c "import lightgbm" 2>/dev/null && echo "  ✓ lightgbm 已安装 ($(python -c 'import lightgbm; print(lightgbm.__version__)'))" || {
    echo "  ✗ lightgbm 未安装"
    echo "  请手动安装："
    echo "    1. 下载 .whl: https://pypi.org/project/lightgbm/#files"
    echo "    2. cd \$TMPDIR && unzip ~/Downloads/lightgbm-*.whl"
    echo "    3. cp -r \$TMPDIR/lightgbm* $PROJECT_DIR/"
    echo "    4. install_name_tool -change @rpath/libomp.dylib \\"
    echo "       \"\$(python -c 'import sklearn; print(sklearn.__path__[0])')/.dylibs/libomp.dylib\" \\"
    echo "       $PROJECT_DIR/lightgbm/lib/lib_lightgbm.dylib"
}

# ── 第3步：创建运行时目录 ────────────
echo "→ 第3步：创建运行时数据目录..."
mkdir -p data/{checkpoints,models,reports,premarket_recommendations,logs,backups}
echo "  ✓ data/ 目录已创建"

# ── 第4步：初始化交易日历 ────────────
echo "→ 第4步：初始化交易日历（需要网络）..."
python -c "
from core.common.calendar import get_calendar
cal = get_calendar()
print(f'  ✓ 交易日历已加载: {len(cal.all_trading_days)} 个交易日')
" 2>/dev/null || echo "  ⚠ 交易日历初始化失败（网络不可用？），使用估算日历"

# ── 第5步：运行单元测试 ───────────────
echo "→ 第5步：运行单元测试验证..."
python -m pytest tests/ -q --tb=line 2>/dev/null && echo "  ✓ 测试通过" || echo "  ⚠ 测试未全部通过（可能缺少依赖或网络）"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   初始化完成！                          ║"
echo "╠══════════════════════════════════════════╣"
echo "║                                          ║"
echo "║  研究模式（直接跑回测）：                ║"
echo "║    python research/run_backtest_demo.py  ║"
echo "║                                          ║"
echo "║  实盘模式（Docker 一键启动）：           ║"
echo "║    docker-compose up -d                  ║"
echo "║                                          ║"
echo "║  清理数据回到初始状态：                  ║"
echo "║    ./scripts/clean_data.sh               ║"
echo "║                                          ║"
echo "╚══════════════════════════════════════════╝"
