"""Local Streamlit app: saved results and a precomputed-recommendation demo.

The "Results" page reads only the committed files in evals/results/. The "Demo" page needs the
untracked data/processed/demo_recs.json (made by scripts/make_demo_recs.py from the H&M data,
which is not included in the repository) and explains what is missing otherwise.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashion_recsys import report as R  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the validated default palette
DEMO_PATH = Path(os.environ.get("DEMO_RECS_PATH", ROOT / "data" / "processed" / "demo_recs.json"))

st.set_page_config(page_title="Fashion recommender: results", layout="wide")
page = st.sidebar.radio("Page", ["Results", "Demo"])


def style(fig, title, ytitle):
    fig.update_layout(title=title, yaxis_title=ytitle, margin=dict(l=10, r=10, t=50, b=10), bargap=0.35,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(rangemode="tozero", gridcolor="rgba(128,128,128,0.25)")
    return fig


def results_page():
    test = R.load("test_final.json")
    rec = R.load("test_run_record.json")
    st.title("Fashion recommendation engine: results")
    st.caption(f"Test week 2020-09-16 to 2020-09-22, {test['n_evaluated']:,} customers who bought that week "
               f"({test['cold_start_share']:.2%} cold-start). Evaluated exactly once with frozen models "
               f"(run record status: {rec['status']}, commit {rec['commit'][:7]}). 95% bootstrap CIs "
               f"({test['bootstrap']['resamples']} resamples). Numbers are read from evals/results/.")
    rows = R.test_rows()

    st.subheader("Test week")
    metric = st.radio("Metric", [k for k, _ in R.METRICS], format_func=dict(R.METRICS).get, horizontal=True)
    fig = go.Figure(go.Bar(
        x=[r["system"] for r in rows], y=[r[metric] for r in rows], marker_color=BLUE, name="Test week",
        error_y=dict(type="data", symmetric=False, array=[r[metric + "_hi"] - r[metric] for r in rows],
                     arrayminus=[r[metric] - r[metric + "_lo"] for r in rows]),
        text=[f"{r[metric]:.4f}" for r in rows], textposition="outside",
        hovertemplate="%{x}<br>%{y:.4f}<extra></extra>"))
    st.plotly_chart(style(fig, f"{dict(R.METRICS)[metric]} on the test week (error bars: 95% CI)", dict(R.METRICS)[metric]), use_container_width=True)
    table = pd.DataFrame([{
        "System": r["system"], **{n: f"{r[k]:.4f} [{r[k + '_lo']:.4f}, {r[k + '_hi']:.4f}]" for k, n in R.METRICS},
        "Coverage@12": round(r["coverage@12"], 4), "Already-bought share": round(r["already_bought_share"], 3),
        "MAP cold": round(r["map@12_cold"], 4), "MAP warm": round(r["map@12_warm"], 4)} for r in rows])
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.subheader("Validation vs test")
    vt = R.val_vs_test_rows()
    sel = st.radio("Metric ", [n for _, n in R.METRICS], horizontal=True, key="vt_metric")
    sub = [r for r in vt if r["metric"] == sel]
    fig = go.Figure([go.Bar(name="Validation week", x=[r["system"] for r in sub], y=[r["validation"] for r in sub], marker_color=BLUE),
                     go.Bar(name="Test week", x=[r["system"] for r in sub], y=[r["test"] for r in sub], marker_color=ORANGE)])
    fig.update_layout(barmode="group")
    st.plotly_chart(style(fig, f"{sel}: validation vs test", sel), use_container_width=True)
    st.dataframe(pd.DataFrame([{"System": r["system"], "Metric": r["metric"], "Validation": round(r["validation"], 4),
                                "Test": round(r["test"], 4), "Change": f"{r['change']:+.4f} ({r['change_pct']:+.1%})",
                                "Test CI contains validation value": r["test_ci_contains_validation"]} for r in vt]),
                 hide_index=True, use_container_width=True)

    st.subheader("Paired differences on the test week")
    st.dataframe(pd.DataFrame([{"A - B": r["a_minus_b"], "Metric": r["metric"],
                                "Difference [95% CI]": f"{r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]",
                                "CI excludes 0": r["excludes_zero"]} for r in R.paired_rows()]), hide_index=True, use_container_width=True)
    st.caption("The H&M data is not included in this repository (Kaggle competition rules); this page shows aggregate results only.")


def demo_page():
    st.title("Demo: recommendations for sample customers")
    if not DEMO_PATH.exists():
        st.info("The demo needs precomputed recommendations made from the H&M data, which is not included in this repository "
                "(Kaggle competition rules). Download the data, then run `python scripts/make_demo_recs.py` "
                f"to create `{DEMO_PATH.name}` locally; it stays untracked.")
        return
    demo = json.loads(DEMO_PATH.read_text())
    st.caption(f"Recommendations as of {demo['cutoff']} from the frozen reranker, with what each customer actually bought in the following week. "
               "Customer ids are shortened; no images are shown.")
    names = [f"{c['customer']}  ({'cold-start' if c['cold'] else str(c['n_purchases']) + ' past purchases'})" for c in demo["customers"]]
    pick = st.selectbox("Customer", range(len(names)), format_func=names.__getitem__)
    c = demo["customers"][pick]
    left, right = st.columns(2)
    with left:
        st.subheader("Recent purchases")
        st.dataframe(pd.DataFrame(c["recent"]), hide_index=True, use_container_width=True) if c["recent"] else st.write("No purchase history.")
    with right:
        st.subheader("Top 12 recommendations")
        st.dataframe(pd.DataFrame(c["recommendations"]), hide_index=True, use_container_width=True)
    st.write(f"Hits in the next week: {sum(r['bought_next_week'] for r in c['recommendations'])} of {len(c['recommendations'])}; "
             f"the customer bought {c['n_bought_next_week']} distinct articles that week.")


(results_page if page == "Results" else demo_page)()
