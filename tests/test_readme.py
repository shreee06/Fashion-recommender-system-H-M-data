"""README.md must be exactly what scripts/render_readme_tables.py renders from the saved result files."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_readme_tables as rr  # noqa: E402


def test_readme_is_in_sync_with_result_files():
    current = (ROOT / "README.md").read_text()
    count = re.search(r"\*\*(\d+) tests\*\*", current).group(1)
    assert rr.render(count) == current, "README.md is stale: run python scripts/render_readme_tables.py"


def test_readme_has_no_unfilled_placeholders_and_no_kaggle_rank_claim():
    text = (ROOT / "README.md").read_text()
    assert "{{" not in text and "}}" not in text
    assert "leaderboard" in text and "No Kaggle leaderboard score" in text
