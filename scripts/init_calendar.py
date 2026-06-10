"""交易日历初始化 — 供 setup.sh 调用。"""
import sys
from pathlib import Path

# 确保项目根目录在 Python path 中
_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

try:
    from core.common.calendar import get_calendar
    cal = get_calendar()
    num = len(cal.all_trading_days)
    print(f"  OK 交易日历已加载: {num} 个交易日")
except Exception as e:
    print(f"  WARN 网络拉取失败({e})，使用估算日历")
    try:
        from core.common.calendar import TradingCalendar
        cal = TradingCalendar(offline_ok=True)
        num = len(cal.all_trading_days)
        print(f"  OK 交易日历已就绪（估算模式）: {num} 个交易日")
    except Exception as e2:
        print(f"  ERR 交易日历不可用: {e2}")
        sys.exit(1)
