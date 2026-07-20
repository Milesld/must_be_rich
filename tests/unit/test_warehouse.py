"""warehouse.py 测试（路线图第 6 阶段）。全部在 tmp_path 上运行，不碰真实仓库。"""

from datetime import date, timedelta

import pandas as pd
import pytest

from research import warehouse as wh


@pytest.fixture(autouse=True)
def _isolate_warehouse(tmp_path, monkeypatch):
    monkeypatch.setattr(wh, "WAREHOUSE_DIR", tmp_path / "warehouse")


def _df(dates_closes: list[tuple[str, float]], **extra) -> pd.DataFrame:
    rows = []
    for d, c in dates_closes:
        rows.append({
            "trade_date": date.fromisoformat(d),
            "open": c * 0.99, "high": c * 1.01, "low": c * 0.98, "close": c,
            "volume": 1000.0, "amount": c * 1000, "turnover": 1.5, **extra,
        })
    return pd.DataFrame(rows)


class TestUpsertAndRead:
    def test_create_then_read(self):
        action = wh.upsert_kline("600000", _df([("2024-01-02", 10.0), ("2024-01-03", 10.5)]),
                                 fetch_start=date(2024, 1, 1))
        assert action == "created"
        df = wh.read_kline("600000")
        assert len(df) == 2
        assert list(df.columns) == ["code", *wh.KLINE_COLUMNS]
        assert df["close"].tolist() == [10.0, 10.5]

    def test_append_merges_and_dedupes(self):
        wh.upsert_kline("600000", _df([("2024-01-02", 10.0), ("2024-01-03", 10.5)]))
        action = wh.upsert_kline("600000", _df([("2024-01-03", 10.5), ("2024-01-04", 11.0)]))
        assert action == "appended"
        df = wh.read_kline("600000")
        assert len(df) == 3
        assert df["trade_date"].is_monotonic_increasing

    def test_date_filter(self):
        wh.upsert_kline("600000", _df([("2024-01-02", 10.0), ("2024-01-03", 10.5),
                                       ("2024-01-04", 11.0)]))
        df = wh.read_kline("600000", start=date(2024, 1, 3), end=date(2024, 1, 3))
        assert len(df) == 1
        assert wh.read_kline("600000", start=date(2025, 1, 1)) is None

    def test_read_missing_returns_none(self):
        assert wh.read_kline("999999") is None

    def test_read_many_concat(self):
        wh.upsert_kline("600000", _df([("2024-01-02", 10.0)]))
        wh.upsert_kline("000001", _df([("2024-01-02", 20.0)]))
        df = wh.read_kline_many(["600000", "000001", "999999"])
        assert set(df["code"]) == {"600000", "000001"}

    def test_zero_close_rows_dropped(self):
        action = wh.upsert_kline("600000", _df([("2024-01-02", 0.0), ("2024-01-03", 10.0)]))
        assert action == "created"
        assert len(wh.read_kline("600000")) == 1


class TestRebaseDetection:
    def test_consistent_overlap_appends(self):
        wh.upsert_kline("600000", _df([("2024-01-02", 10.0), ("2024-01-03", 10.5)]))
        action = wh.upsert_kline("600000", _df([("2024-01-03", 10.5), ("2024-01-04", 11.0)]))
        assert action == "appended"

    def test_shifted_baseline_detected_and_old_discarded(self):
        """除权后 qfq 全历史平移：重叠日 close 偏差超容差 → 废弃旧数据。"""
        wh.upsert_kline("600000", _df([("2024-01-02", 10.0), ("2024-01-03", 10.5)]))
        # 10 派 2 除权后 qfq 基准变化，历史价整体下移 ~2%
        action = wh.upsert_kline("600000", _df([("2024-01-03", 10.29), ("2024-01-04", 10.78)]))
        assert action == "rebase_detected"
        df = wh.read_kline("600000")
        # 旧数据废弃，只剩新段
        assert df["trade_date"].tolist() == [date(2024, 1, 3), date(2024, 1, 4)]
        assert df["close"].tolist() == [10.29, 10.78]

    def test_small_rounding_noise_tolerated(self):
        wh.upsert_kline("600000", _df([("2024-01-03", 10.50)]))
        action = wh.upsert_kline("600000", _df([("2024-01-03", 10.51), ("2024-01-04", 11.0)]))
        assert action == "appended"  # 0.1% < 0.5% 容差

    def test_rebase_clears_fetch_start(self):
        wh.upsert_kline("600000", _df([("2024-01-02", 10.0)]), fetch_start=date(2023, 1, 1))
        wh.upsert_kline("600000", _df([("2024-01-02", 12.0)]))  # 20% 偏移
        meta = wh.load_meta()
        assert "fetch_start" not in meta["600000"]  # 头部覆盖声明随旧数据一起作废


class TestMetaAndCoverage:
    def test_coverage(self):
        wh.upsert_kline("600000", _df([("2024-01-02", 10.0), ("2024-03-04", 11.0)]))
        assert wh.coverage("600000") == (date(2024, 1, 2), date(2024, 3, 4))
        assert wh.coverage("999999") is None

    def test_warehouse_covers_with_fetch_start(self):
        """次新股：first > start 但 fetch_start ≤ start → 视为覆盖。"""
        wh.upsert_kline("301999", _df([("2024-06-03", 10.0), ("2024-12-31", 11.0)]),
                        fetch_start=date(2023, 1, 1))
        assert wh.warehouse_covers("301999", date(2023, 6, 1), date(2024, 12, 31))
        # 尾部不足 → 不覆盖
        assert not wh.warehouse_covers("301999", date(2023, 6, 1), date(2025, 6, 1))

    def test_warehouse_covers_without_fetch_start(self):
        """无 fetch_start（旧数据）：按首行日期保守判断。"""
        wh.upsert_kline("600000", _df([("2024-01-02", 10.0), ("2024-12-31", 11.0)]))
        assert not wh.warehouse_covers("600000", date(2023, 6, 1), date(2024, 12, 31))
        assert wh.warehouse_covers("600000", date(2024, 1, 2), date(2024, 12, 31))

    def test_rebuild_meta_from_parquet(self):
        wh.upsert_kline("600000", _df([("2024-01-02", 10.0)]))
        wh._meta_path().unlink()
        meta = wh.rebuild_meta()
        assert meta["600000"]["first"] == "2024-01-02"

    def test_bulk_meta_flow(self):
        meta = wh.load_meta()
        wh.upsert_kline("600000", _df([("2024-01-02", 10.0)]), meta)
        wh.upsert_kline("000001", _df([("2024-01-02", 20.0)]), meta)
        wh.save_meta_bulk(meta)
        assert set(wh.load_meta()) == {"600000", "000001"}


class TestConstituentSnapshots:
    def test_snapshot_and_asof(self):
        wh.snapshot_constituents("pt018", ["600000", "000001"], date(2024, 1, 10))
        wh.snapshot_constituents("pt018", ["600000", "300001"], date(2024, 2, 10))
        assert wh.constituents_asof("pt018", date(2024, 1, 15)) == ["600000", "000001"]
        assert wh.constituents_asof("pt018", date(2024, 3, 1)) == ["600000", "300001"]
        # 快照日之前 → None（调用方回退当前成分）
        assert wh.constituents_asof("pt018", date(2023, 12, 31)) is None
        assert wh.constituents_asof("no_such_board", date(2024, 1, 15)) is None

    def test_same_day_overwrites(self):
        wh.snapshot_constituents("pt018", ["600000"], date(2024, 1, 10))
        wh.snapshot_constituents("pt018", ["000001"], date(2024, 1, 10))
        assert wh.constituents_asof("pt018", date(2024, 1, 10)) == ["000001"]

    def test_boards_listing(self):
        wh.snapshot_constituents("pt018", ["600000"], date(2024, 1, 10))
        wh.snapshot_constituents("pt018", ["600000"], date(2024, 1, 11))
        assert wh.constituent_boards() == {"pt018": 2}
