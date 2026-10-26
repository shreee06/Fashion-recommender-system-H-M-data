"""Guards for the final-test script: the dry run must never touch the test week."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_reranker  # noqa: E402
import run_test_final  # noqa: E402
from fashion_recsys.config import make_split, to_day  # noqa: E402

SPLIT = make_split(to_day(date(2018, 9, 20)), to_day(date(2020, 9, 22)))
TEST_START = to_day(SPLIT.test_start)  # 2020-09-16


def fake_D(days):
    return {"day": np.asarray(days)}


def test_assert_no_test_data_rejects_test_rows_and_accepts_truncated():
    with pytest.raises(AssertionError):
        run_test_final.assert_no_test_data(fake_D([TEST_START - 3, TEST_START]), SPLIT)
    run_test_final.assert_no_test_data(fake_D([TEST_START - 3, TEST_START - 1]), SPLIT)


def test_load_D_truncation_drops_every_test_row(monkeypatch):
    """load_D('test_start') must return only rows before 2020-09-16 while the split still
    comes from the full data range."""
    n = 40
    day = np.sort(np.r_[np.arange(to_day(date(2018, 9, 20)), to_day(date(2020, 9, 23)))[::25], [TEST_START - 1, TEST_START, TEST_START + 3, to_day(date(2020, 9, 22))]])
    n = len(day)
    arrays = (day.astype(np.int32), np.zeros(n, np.int32), np.zeros(n, np.int32))
    monkeypatch.setattr(run_reranker, "load_transaction_codes", lambda p: (*arrays, ["c0"], ["a0"]))
    monkeypatch.setattr(run_reranker, "load_price_channel", lambda p: (np.zeros(n, np.float32), np.ones(n, np.int8)))

    class Tbl:
        def __init__(self, cols):
            self.cols = cols

        def to_pandas(self):
            import pandas as pd
            return pd.DataFrame({c: [0] for c in self.cols})

        def __getitem__(self, c):
            class Col:
                def to_numpy(self, zero_copy_only=False):
                    return np.array([30.0])
            return Col()

    monkeypatch.setattr(run_reranker.pq, "read_table", lambda path, columns=None: Tbl(columns))
    monkeypatch.setattr(run_reranker, "json", __import__("json"))
    monkeypatch.setattr(run_reranker.json, "loads", lambda s: {})
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: "{}")
    D, split, val_start, _ = run_reranker.load_D(truncate_before="test_start")
    assert split.test_start == SPLIT.test_start and split.test_end == SPLIT.test_end  # from the full range
    assert D["day"].max() < TEST_START and len(D["day"]) == len(D["cust"]) == len(D["art"]) == len(D["price"]) == len(D["chan"])
    assert (day >= TEST_START).sum() == 3 and len(D["day"]) == n - 3
    run_test_final.assert_no_test_data(D, split)


def test_dry_run_never_writes_the_run_record_and_never_asks_for_the_flag(monkeypatch, tmp_path):
    calls = []

    class FakeGuard:
        def __init__(self, *a): pass
        def start_watchdog(self): pass
        def stage(self, *a): pass
        def peak_mb(self): return 1

    monkeypatch.setattr(run_test_final, "RECORD", tmp_path / "test_run_record.json")
    monkeypatch.setattr(run_test_final, "MemGuard", FakeGuard)
    monkeypatch.setattr(run_test_final, "RES", tmp_path)
    expected = run_test_final.EXPECTED_VAL_MAP

    def fake_eval(week, guard):
        calls.append(week)
        return {"results": {k: {"map@12": {"mean": v}} for k, v in expected.items()}}

    monkeypatch.setattr(run_test_final, "evaluate_week", fake_eval)
    run_test_final.main(["--dry-run-on-validation"])
    assert calls == ["validation"]
    assert not (tmp_path / "test_run_record.json").exists()
    assert (tmp_path / "validation_reference.json").exists()


def test_dry_run_stops_when_numbers_differ(monkeypatch, tmp_path):
    class FakeGuard:
        def __init__(self, *a): pass
        def start_watchdog(self): pass
        def stage(self, *a): pass
        def peak_mb(self): return 1

    monkeypatch.setattr(run_test_final, "MemGuard", FakeGuard)
    monkeypatch.setattr(run_test_final, "RES", tmp_path)
    off = {k: {"map@12": {"mean": v + 0.001}} for k, v in run_test_final.EXPECTED_VAL_MAP.items()}
    monkeypatch.setattr(run_test_final, "evaluate_week", lambda week, guard: {"results": off})
    with pytest.raises(SystemExit):
        run_test_final.main(["--dry-run-on-validation"])
    assert not (tmp_path / "validation_reference.json").exists()


def test_test_week_refuses_without_flag_and_when_record_exists(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="i-am-done-tuning"):
        run_test_final.main([])
    rec = tmp_path / "test_run_record.json"
    rec.write_text("{}")
    monkeypatch.setattr(run_test_final, "RECORD", rec)
    with pytest.raises(SystemExit, match="exists"):
        run_test_final.main(["--i-am-done-tuning"])
    assert rec.read_text() == "{}"  # untouched
