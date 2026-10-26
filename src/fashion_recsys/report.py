"""Read the saved result files in evals/results/ and turn them into table rows.

Used by scripts/render_readme_tables.py and app/app.py so that every number shown to a reader
comes straight from the JSON/CSV files and cannot drift from them.
"""
import csv
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "evals" / "results"
METRICS = [("map@12", "MAP@12"), ("recall@12", "Recall@12"), ("ndcg@12", "NDCG@12")]
LABELS = {
    "popular_last_1w": "Popular, last 1 week",
    "repurchase+last_4w": "Repurchase + popular 4w",
    "combination_step5": "Step 5 combination",
    "two_tower_alone": "Two-tower alone",
    "reranker": "LightGBM reranker",
}
SYSTEMS = list(LABELS)


def load(name):
    return json.loads((RES / name).read_text())


def label(name):
    return LABELS.get(name, name)


def test_rows():
    d = load("test_final.json")
    rows = []
    for s in SYSTEMS:
        r = d["results"][s]
        row = {"system": label(s), "key": s, "coverage@12": r["coverage@12"], "already_bought_share": r["already_bought_share"],
               "map@12_cold": r["map@12_cold"], "map@12_warm": r["map@12_warm"]}
        for k, _ in METRICS:
            row[k], row[k + "_lo"], row[k + "_hi"] = r[k]["mean"], r[k]["ci_lo"], r[k]["ci_hi"]
        rows.append(row)
    return rows


def val_vs_test_rows():
    val, test = load("validation_reference.json")["results"], load("test_final.json")["results"]
    rows = []
    for s in SYSTEMS:
        for k, name in METRICS:
            v, t = val[s][k]["mean"], test[s][k]["mean"]
            rows.append({"system": label(s), "metric": name, "validation": v, "test": t, "change": t - v, "change_pct": (t - v) / v,
                         "test_ci_contains_validation": test[s][k]["ci_lo"] <= v <= test[s][k]["ci_hi"]})
    return rows


def paired_rows(file="test_final.json", rename=None):
    rows = []
    for ab, d in load(file)["paired"].items():
        a, b = ab.split(" - ")
        for k, name in METRICS:
            x = d[k]
            rows.append({"a_minus_b": f"{(rename or label)(a)} - {(rename or label)(b)}", "metric": name, "diff": x["diff"],
                         "ci_lo": x["ci_lo"], "ci_hi": x["ci_hi"], "excludes_zero": x["excludes_zero"]})
    return rows


def ladder_rows():
    """Validation-week progression, all on the same 72,019 buyers (values from the step result files)."""
    ic, cd, rr = load("item_cf_validation.json")["results"], load("candidates_validation.json")["results"], load("reranker_validation.json")["results"]
    src = [("Popular, last 4 weeks", ic["popular_last_4w"]), ("Repurchase + popular 4w", ic["repurchase+last_4w"]),
           ("Item-CF alone (200k-customer sample)", cd["item_cf_step4 (200k sample, 26w, 20k items, binary, K=3)"]),
           ("Item-CF alone (all customers, recency)", cd["item_cf_new (best config)"]),
           ("Step 4 combination", cd["combination_step4 (repurchase top8 + step-4 CF + pop4w)"]),
           ("Step 5 combination", rr["combination_step5"]), ("Two-tower alone", rr["two_tower_alone"]),
           ("LightGBM reranker", rr["reranker (full)"])]
    return [{"stage": n, **{k: r[k]["mean"] for k, _ in METRICS}, **{k + "_ci": (r[k]["ci_lo"], r[k]["ci_hi"]) for k, _ in METRICS}} for n, r in src]


def reranker_ablation_rows():
    rr = load("reranker_validation.json")
    rows = []
    for name in ("reranker (full)", "reranker without two-tower features", "reranker without content features"):
        r = rr["results"][name]
        rows.append({"variant": name, **{k: r[k]["mean"] for k, _ in METRICS}, **{k + "_ci": (r[k]["ci_lo"], r[k]["ci_hi"]) for k, _ in METRICS}})
    return rows


def reranker_ablation_paired():
    return paired_rows("reranker_validation.json", rename=lambda x: x)


def two_tower_ablation_rows():
    tt = load("two_tower_validation.json")["results"]
    rows = []
    for name in ("two_tower (chosen)", "two_tower ID-only item tower", "two_tower logQ flipped"):
        r = tt[name]
        rows.append({"variant": name, **{k: r[k]["mean"] for k, _ in METRICS}, **{k + "_ci": (r[k]["ci_lo"], r[k]["ci_hi"]) for k, _ in METRICS}})
    return rows


def importance_rows(top=10):
    imp = load("reranker_validation.json")["feature_importance_gain_share"]
    return list(imp.items())[:top]


def tuning_summary():
    """Number of logged trials per method (smoke-run rows excluded)."""
    counts = {}
    with open(RES / "tuning_log.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["method"].startswith("smoke_") or r["run"].endswith("-SMOKE"):
                continue
            counts[r["method"]] = counts.get(r["method"], 0) + 1
    return counts


def facts():
    t, rr, tt, rec = load("test_final.json"), load("reranker_validation.json"), load("two_tower_validation.json"), load("test_run_record.json")
    val = load("validation_reference.json")
    trials = tt["trials"]
    at_cap = sum(1 for x in trials if x["best_epoch"] == x["epochs_run"])
    main = tt["final_runs"]["two_tower (chosen)"]
    rm = load("runtime_memory.json")
    gb = lambda b: f"{b / 1e9:.2f} GB"
    mb = lambda b: f"{b / 1e6:.0f} MB"
    conv, scale = rm["conversion"], rm["data_scale"]["rows"]
    return {
        "n_transactions": f"{scale['transactions_train']:,}", "n_customers_all": f"{scale['customers']:,}", "n_articles_all": f"{scale['articles']:,}",
        "csv_transactions": gb(conv["csv_bytes"]["transactions_train"]), "parquet_transactions": mb(conv["parquet_bytes"]["transactions_train"]),
        "reduction_transactions": f"{conv['transactions_reduction_pct']:.1f}%",
        "csv_total": gb(conv["total_csv_bytes"]), "parquet_total": mb(conv["total_parquet_bytes"]), "reduction_total": f"{conv['total_reduction_pct']:.1f}%",
        "block_mb": str(conv["streamed_block_size_mb"]), "machine_ram_mb": f"{rm['machine']['ram_mb']:,}",
        "machine_cpus": str(rm["machine"]["logical_cpus"]), "machine_threads": str(rm["machine"]["cpu_threads_used_for_training_and_inference"]),
        "chunk_customers": f"{rm['memory_guard']['feature_build_chunk_customers']:,}", "guard_mb": f"{rm['memory_guard']['abort_below_mem_available_mb']:,}",
        "freeze_peak": f"{rm['freeze_run']['peak_rss_mb']:,}", "freeze_min_avail": f"{rm['freeze_run']['min_mem_available_mb']:,}",
        "dry_peak": f"{rm['dry_run_validation']['peak_rss_mb']:,}", "dry_runtime": f"{rm['dry_run_validation']['runtime_s']:.0f}",
        "test_peak_logged": f"{rm['final_test_run']['peak_rss_mb']:,}", "test_min_avail": f"{rm['final_test_run']['min_mem_available_mb']:,}",
        "tt_script_peak": f"{rm['two_tower_training_script']['peak_rss_mb']:,}", "tt_script_hours": f"{rm['two_tower_training_script']['runtime_s'] / 3600:.1f}",
        "tt_script_trials": str(rm["two_tower_training_script"]["trials"]),
        "rr_script_peak": f"{rm['reranker_validation_script']['peak_rss_mb']:,}", "rr_script_min": f"{rm['reranker_validation_script']['runtime_s'] / 60:.0f}",
        "n_test": f"{t['n_evaluated']:,}", "n_test_cold": f"{t['n_cold']:,}", "cold_test": f"{t['cold_start_share']:.2%}",
        "n_val": f"{val['n_evaluated']:,}", "cold_val": f"{val['cold_start_share']:.2%}",
        "catalogue_size": f"{tt['population']['catalogue_size']:,}",
        "in_catalogue": f"{tt['population']['relevant_pairs_in_catalogue']:.2%}",
        "out_catalogue": f"{1 - tt['population']['relevant_pairs_in_catalogue']:.2%}",
        "rr_train_customers": f"{rr['training']['customers_sampled']:,}", "rr_fit_customers": f"{rr['training']['fit_customers']:,}",
        "rr_rows": f"{rr['training']['fit_rows']:,}", "rr_features": str(rr['training']['n_features']),
        "rr_iter": str(rr["best_iteration"]),
        "pool_size": f"{rr['pool_ceiling']['mean_pool_size']:.0f}", "pool_recall": f"{rr['pool_ceiling']['pool_recall_mean']:.4f}",
        "rr_recall12": f"{rr['pool_ceiling']['reranked_recall@12']:.4f}",
        "tt_params": f"{main['n_params']:,}", "tt_sec_epoch": f"{main['sec_per_epoch']:.0f}", "tt_trials": str(len(trials)),
        "tt_trials_at_cap": str(at_cap), "tt_epoch_cap": str(max(x["epochs_run"] for x in trials)),
        "tt_hours": f"{tt['runtime_s'] / 3600:.1f}",
        "run_commit": rec["commit"][:7], "run_time": rec["timestamp"], "run_status": rec["status"],
        "test_peak_mb": f"{t['peak_rss_mb']:,}", "test_runtime": f"{t['runtime_s']:.0f}",
        "bootstrap": f"{t['bootstrap']['resamples']} resamples, seed {t['bootstrap']['seed']}",
        "test_range": "2020-09-16 to 2020-09-22", "val_range": "2020-09-09 to 2020-09-15",
    }
