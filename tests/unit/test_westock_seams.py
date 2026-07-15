"""westock qfq 缓存段缝连续性校验回归测试（路线图第 2 阶段）。

场景：kline 按段缓存（每段 ≤244 自然日）。qfq 前复权以拉取当天为
基准回算，个股除权除息后新拉段与旧缓存段基线不一致，段缝处出现假跳空。
修复：拼接时校验段缝相邻收盘价涨跌幅，超板块涨跌停×1.5+2% 判为基线
错位，废弃该股全部缓存段重拉。
"""

from __future__ import annotations

from datetime import date

import pytest

from research import westock_source as ws


def _seg(closes: list[float], start_day: int = 1) -> list[dict]:
    """构造一段标准化 kline rows（只填测试关心的字段）。"""
    return [
        {"code": "600000", "date": f"2026-01-{start_day + i:02d}",
         "open": c, "high": c, "low": c, "close": c, "volume": 1000, "amount": 1e6,
         "turnover": 1.0}
        for i, c in enumerate(closes)
    ]


class TestBoardPriceLimit:
    def test_prefix_mapping(self):
        assert ws._board_price_limit("600000") == 0.10
        assert ws._board_price_limit("300750") == 0.20
        assert ws._board_price_limit("688981") == 0.20
        assert ws._board_price_limit("920139") == 0.30


class TestSeamsOk:
    def test_continuous_segments_pass(self):
        segs = [_seg([10.0, 10.2, 10.1]), _seg([10.3, 10.5], start_day=4)]
        assert ws._seams_ok("600000", segs) is True

    def test_baseline_mismatch_detected(self):
        """典型 qfq 基线错位：10 送 10 除权后旧段价格约为新段 2 倍。"""
        old_seg = _seg([20.0, 20.4, 20.2])          # 旧缓存（旧基准）
        new_seg = _seg([10.1, 10.3], start_day=4)   # 新拉段（新基准，价格减半）
        assert ws._seams_ok("600000", [old_seg, new_seg]) is False

    def test_limit_up_within_threshold_passes(self):
        """主板正常涨停 10% 在阈值内（阈值 = 10%×1.5+2% = 17%）。"""
        segs = [_seg([10.0]), _seg([11.0], start_day=2)]
        assert ws._seams_ok("600000", segs) is True

    def test_gem_20pct_jump_passes_but_50pct_fails(self):
        """创业板阈值 = 20%×1.5+2% = 32%：20% 涨停放行，50% 错位检出。"""
        assert ws._seams_ok("300750", [_seg([10.0]), _seg([12.0], start_day=2)]) is True
        assert ws._seams_ok("300750", [_seg([10.0]), _seg([15.0], start_day=2)]) is False

    def test_empty_segments_skipped(self):
        """次新股早期段为空数组，不应影响校验。"""
        segs = [[], _seg([10.0, 10.1]), [], _seg([10.2], start_day=4)]
        assert ws._seams_ok("600000", segs) is True

    def test_single_segment_always_ok(self):
        assert ws._seams_ok("600000", [_seg([10.0, 30.0, 5.0])]) is True


class TestPurgeAndRefetch:
    def test_purge_kline_cache_removes_only_target_code(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ws, "_CACHE_DIR", tmp_path)
        (tmp_path / "kline_600000_2025-01-01_2025-08-31.json").write_text("[]")
        (tmp_path / "kline_600000_2025-09-01_2026-04-30.json").write_text("[]")
        (tmp_path / "kline_000001_2025-01-01_2025-08-31.json").write_text("[]")

        n = ws._purge_kline_cache("600000")
        assert n == 2
        assert not list(tmp_path.glob("kline_600000_*.json"))
        assert list(tmp_path.glob("kline_000001_*.json"))

    def test_fetch_code_segments_refetches_on_seam_break(self, monkeypatch):
        """段缝错位 → 废弃缓存 → 重拉后使用连续数据。"""
        segs = [(date(2026, 1, 1), date(2026, 1, 3)), (date(2026, 1, 4), date(2026, 1, 6))]

        old_data = {  # 首轮：第一段来自旧缓存（基线错位，价格翻倍）
            "2026-01-01": _seg([20.0, 20.2, 20.4]),
            "2026-01-04": _seg([10.1, 10.2, 10.3], start_day=4),
        }
        fresh_data = {  # 重拉：两段基线一致
            "2026-01-01": _seg([10.0, 10.1, 10.2]),
            "2026-01-04": _seg([10.1, 10.2, 10.3], start_day=4),
        }
        state = {"purged": False}

        def fake_fetch(code, s, e, max_retries=2):
            return (fresh_data if state["purged"] else old_data)[s]

        def fake_purge(code):
            state["purged"] = True
            return 2

        monkeypatch.setattr(ws, "_fetch_one_segment", fake_fetch)
        monkeypatch.setattr(ws, "_purge_kline_cache", fake_purge)

        rows = ws._fetch_code_segments("600000", segs, max_retries=2)
        assert state["purged"] is True
        closes = [r["close"] for r in rows]
        assert closes == [10.0, 10.1, 10.2, 10.1, 10.2, 10.3]

    def test_fetch_code_segments_no_refetch_when_continuous(self, monkeypatch):
        """段缝连续时不触发废弃重拉。"""
        segs = [(date(2026, 1, 1), date(2026, 1, 3)), (date(2026, 1, 4), date(2026, 1, 6))]
        calls = []

        def fake_fetch(code, s, e, max_retries=2):
            calls.append(s)
            return _seg([10.0, 10.1]) if s == "2026-01-01" else _seg([10.2], start_day=4)

        def fake_purge(code):
            raise AssertionError("should not purge")

        monkeypatch.setattr(ws, "_fetch_one_segment", fake_fetch)
        monkeypatch.setattr(ws, "_purge_kline_cache", fake_purge)

        rows = ws._fetch_code_segments("600000", segs, max_retries=2)
        assert len(calls) == 2  # 每段只拉一次
        assert len(rows) == 3
