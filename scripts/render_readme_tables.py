"""Render README.md from docs/README.template.md and the saved result files.

Every `{{table:NAME}}` and `{{NAME}}` placeholder is filled from evals/results/ via
fashion_recsys.report, so the README numbers cannot drift from the files.

    python scripts/render_readme_tables.py            # write README.md
    python scripts/render_readme_tables.py --check    # exit 1 if README.md is out of date
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashion_recsys import report as R  # noqa: E402

TEMPLATE, README = ROOT / "docs" / "README.template.md", ROOT / "README.md"


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def ci(m, lo, hi):
    return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"


def tables():
    t = {}
    t["test"] = md(["System", "MAP@12 [95% CI]", "Recall@12 [95% CI]", "NDCG@12 [95% CI]", "Coverage@12", "Already-bought share", "MAP cold", "MAP warm"],
                   [[r["system"], *(ci(r[k], r[k + "_lo"], r[k + "_hi"]) for k, _ in R.METRICS), f"{r['coverage@12']:.4f}",
                     f"{r['already_bought_share']:.3f}", f"{r['map@12_cold']:.4f}", f"{r['map@12_warm']:.4f}"] for r in R.test_rows()])
    t["paired_test"] = md(["A - B", "Metric", "Difference [95% CI]", "CI excludes 0"],
                          [[r["a_minus_b"], r["metric"], f"{r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]", "yes" if r["excludes_zero"] else "no"]
                           for r in R.paired_rows()])
    t["val_vs_test"] = md(["System", "Metric", "Validation", "Test", "Change", "Test CI contains validation value"],
                          [[r["system"], r["metric"], f"{r['validation']:.4f}", f"{r['test']:.4f}",
                            f"{r['change']:+.4f} ({r['change_pct']:+.1%})" + (" **drop**" if r["change"] < 0 else ""),
                            "yes" if r["test_ci_contains_validation"] else "no"] for r in R.val_vs_test_rows()])
    t["ladder"] = md(["Stage (validation week)", "MAP@12 [95% CI]", "Recall@12 [95% CI]", "NDCG@12 [95% CI]"],
                     [[r["stage"], *(ci(r[k], *r[k + "_ci"]) for k, _ in R.METRICS)] for r in R.ladder_rows()])
    t["rr_ablation"] = md(["Variant (validation week)", "MAP@12 [95% CI]", "Recall@12 [95% CI]", "NDCG@12 [95% CI]"],
                          [[r["variant"], *(ci(r[k], *r[k + "_ci"]) for k, _ in R.METRICS)] for r in R.reranker_ablation_rows()])
    wanted = [r for r in R.reranker_ablation_paired() if "without" in r["a_minus_b"]]
    t["rr_ablation_paired"] = md(["Full model minus ablated model", "Metric", "Difference [95% CI]", "CI excludes 0"],
                                 [[r["a_minus_b"], r["metric"], f"{r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]", "yes" if r["excludes_zero"] else "no"] for r in wanted])
    t["tt_ablation"] = md(["Two-tower variant (validation week)", "MAP@12 [95% CI]", "Recall@12 [95% CI]", "NDCG@12 [95% CI]"],
                          [[r["variant"], *(ci(r[k], *r[k + "_ci"]) for k, _ in R.METRICS)] for r in R.two_tower_ablation_rows()])
    t["importance"] = md(["Feature (full reranker)", "Share of total gain"], [[f, f"{v:.4f}"] for f, v in R.importance_rows(10)])
    t["tuning"] = md(["Logged trials (evals/results/tuning_log.csv, smoke runs excluded)", "Count"], [[k, v] for k, v in sorted(R.tuning_summary().items())])
    return t


def pytest_count():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    return m.group(1) if m else "unknown"


def render(test_count):
    text = TEMPLATE.read_text()
    tabs, facts = tables(), R.facts()
    facts["tests"] = test_count
    text = re.sub(r"\{\{table:(\w+)\}\}", lambda m: tabs[m.group(1)], text)
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: str(facts[m.group(1)]), text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    current = README.read_text() if README.exists() else ""
    m = re.search(r"\*\*(\d+) tests\*\*", current)
    count = m.group(1) if (args.check and m) else pytest_count()
    out = render(count)
    if args.check:
        sys.exit(0 if out == current else "README.md is out of date: run scripts/render_readme_tables.py")
    README.write_text(out)
    print("wrote README.md")


if __name__ == "__main__":
    main()
