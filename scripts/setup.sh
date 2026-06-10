#!/bin/bash
# 首次初始化 — 每步独立运行，失败不中断后续步骤
# 使用方式: ./scripts/setup.sh

PASS=0
FAIL=0
WARN=0

green()  { echo "  ✓ $*"; PASS=$((PASS+1)); }
red()    { echo "  ✗ $*"; FAIL=$((FAIL+1)); }
yellow() { echo "  ⚠ $*"; WARN=$((WARN+1)); }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$(which python3 2>/dev/null || which python 2>/dev/null)}"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   量化交易系统 — 首次初始化              ║"
echo "╚══════════════════════════════════════════╝"
echo "  Python: $PYTHON_BIN"
echo "  项目:   $PROJECT_DIR"
echo ""

# ── 第1步：Python 环境 ──────────────────
echo "→ 第1步：检查 Python 环境..."

$PYTHON_BIN --version 2>&1 && green "$PYTHON_BIN 可用" || { red "$PYTHON_BIN 不可用"; }

$PYTHON_BIN -c "import pandas, numpy, sklearn, yaml" 2>/dev/null \
  && green "核心依赖已就绪 (pandas, numpy, sklearn, yaml)" \
  || yellow "部分核心依赖缺失"

$PYTHON_BIN -c "import distutils" 2>/dev/null \
  && green "distutils 可用" \
  || yellow "distutils 不可用 (Python 3.12+ 需要 setuptools)"

# ── 第2步：lightgbm ──────────────────────
echo "→ 第2步：检查 lightgbm..."

$PYTHON_BIN -c "import lightgbm" 2>/dev/null \
  && green "lightgbm $($PYTHON_BIN -c 'import lightgbm;print(lightgbm.__version__)')" \
  || yellow "lightgbm 未安装（部分模型测试会跳过）"

# ── 第3步：创建目录 ──────────────────────
echo "→ 第3步：创建运行时目录..."
for d in checkpoints models reports premarket_recommendations logs backups; do
    mkdir -p "data/$d"
done
green "data/ 目录已创建"

# ── 第4步：交易日历 ──────────────────────
echo "→ 第4步：初始化交易日历..."
$PYTHON_BIN scripts/init_calendar.py 2>&1 && green "交易日历就绪" || yellow "交易日历有问题（不影响使用）"

# ── 第5步：运行测试 ──────────────────────
echo "→ 第5步：运行核心测试..."
# 跳过包含 lightgbm 的测试（加载 native 库很慢，在代理环境可能卡住）
$PYTHON_BIN -m pytest \
    tests/unit/test_db.py \
    tests/unit/test_constraints.py \
    tests/unit/test_cost_model.py \
    tests/unit/test_rounding.py \
    tests/unit/test_portfolio_risk.py \
    tests/unit/test_quality.py \
    tests/unit/test_time_travel.py \
    tests/regression/ \
    tests/integration/ \
    -q --tb=line 2>&1 | tail -5

# ── 总结 ────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   初始化完成                              ║"
echo "╠══════════════════════════════════════════╣"
echo "║  通过: $PASS  警告: $WARN  失败: $FAIL          ║"
echo "╠══════════════════════════════════════════╣"
echo "║                                          ║"
echo "║  研究模式（直接跑回测）：                ║"
echo "║    python research/run_backtest_demo.py  ║"
echo "║                                          ║"
echo "║  运行全部测试：                          ║"
echo "║    python -m pytest tests/ -q            ║"
echo "║                                          ║"
echo "║  清理数据：                              ║"
echo "║    ./scripts/clean_data.sh               ║"
echo "║                                          ║"
echo "╚══════════════════════════════════════════╝"
