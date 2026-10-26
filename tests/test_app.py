"""Streamlit smoke tests: run WITHOUT any H&M data present."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "app.py")


@pytest.fixture(autouse=True)
def no_demo_file(monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_RECS_PATH", str(tmp_path / "does_not_exist.json"))


def test_results_page_renders_from_saved_files():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "Fashion recommendation engine: results" in [t.value for t in at.title]
    assert len(at.dataframe) >= 3  # test table, validation-vs-test, paired differences
    text = " ".join(c.value for c in at.caption)
    assert "evaluated exactly once".lower() in text.lower() and "68,984" in text


def test_results_page_metric_switch():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="vt_metric").set_value("NDCG@12").run()  # the validation-vs-test metric selector
    assert at.radio(key="vt_metric").value == "NDCG@12"
    assert not at.exception


def test_demo_page_explains_missing_data():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Demo").run()
    assert not at.exception
    assert any("not included" in i.value for i in at.info)


def test_demo_page_renders_when_a_demo_file_exists(monkeypatch, tmp_path):
    """Synthetic demo file (not H&M data): the page shows the customer picker and both tables."""
    import json
    f = tmp_path / "demo.json"
    art = {"article_id": "0000000001", "name": "n", "type": "t", "colour": "c", "department": "d"}
    f.write_text(json.dumps({"cutoff": "2020-09-09", "customers": [{
        "customer": "abcd1234...", "cold": False, "n_purchases": 3, "recent": [art],
        "recommendations": [{"rank": 1, **art, "bought_next_week": True}], "n_bought_next_week": 2}]}))
    monkeypatch.setenv("DEMO_RECS_PATH", str(f))
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Demo").run()
    assert not at.exception and len(at.selectbox) == 1 and len(at.dataframe) == 2
