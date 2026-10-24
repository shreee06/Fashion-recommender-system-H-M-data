"""Evaluate the baselines on the VALIDATION week only and write evals/results/."""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashion_recsys import baselines  # noqa: E402
from fashion_recsys.config import (  # noqa: E402
    HISTORY_WEEKS, SAMPLE_SEED, SAMPLE_SIZE, history_start, make_split, sample_customers, to_day,
)
from fashion_recsys.data import load_transaction_codes  # noqa: E402
from fashion_recsys.evaluation import build_eval_set, evaluate_cold_warm  # noqa: E402

RES = ROOT / "evals" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--history-weeks", type=int, default=HISTORY_WEEKS)
    args = ap.parse_args()
    t0 = time.time()

    day, cust, art, customer_ids, article_ids = load_transaction_codes(ROOT / "data" / "processed")
    n_articles = len(article_ids)
    split = make_split(day.min(), day.max())
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "split.json").write_text(json.dumps(split.to_json(), indent=2) + "\n")

    eligible = np.unique(cust)
    sampled = sample_customers(eligible, args.sample_size, args.seed)
    val_start, val_end = to_day(split.val_start), to_day(split.val_end)
    customers, truths, hist = build_eval_set(day, cust, art, sampled, val_start, val_end)

    def run(recs):
        return evaluate_cold_warm(recs, truths, hist, n_articles)

    rec_lists = {}
    results = {}
    glob = baselines.global_popular(day, art, val_start, n_articles)
    pop1 = baselines.recent_popular(day, art, val_start, 1, n_articles)
    pop4 = baselines.recent_popular(day, art, val_start, 4, n_articles)
    for name, top in (("popular_global", glob), ("popular_last_1w", pop1), ("popular_last_4w", pop4)):
        results[name] = run([top.tolist()] * len(customers))
    for name, fb in (("repurchase+last_1w", pop1), ("repurchase+last_4w", pop4)):
        results[name] = run(baselines.repurchase(day, cust, art, val_start, customers, fb))

    rep = ("repurchase+last_1w", "repurchase+last_4w")
    chosen = max(rep, key=lambda n: results[n]["overall"]["map@12"])
    best = max(results, key=lambda n: results[n]["overall"]["map@12"])

    out = {
        "split": split.to_json(),
        "eval_week": "validation",
        "train_cutoff": split.val_start.isoformat(),
        "sample": {
            "eligible_customers_with_purchase": int(len(eligible)),
            "sample_size_requested": args.sample_size,
            "sample_size_used": int(len(sampled)),
            "seed": args.seed,
            "bought_in_validation_week": int(len(customers)),
            "cold_start_share_of_evaluated": results["popular_global"]["cold_share"],
        },
        "history_window": {"weeks": args.history_weeks, "start": history_start(split, args.history_weeks).isoformat(),
                           "note": "window for model training; popularity/repurchase baselines here use all data before the cutoff or their own stated window"},
        "catalog_size": n_articles,
        "repurchase_fallback_chosen": chosen,
        "best_baseline_by_map": best,
        "baselines": results,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        "runtime_s": round(time.time() - t0, 1),
    }
    (RES / "baselines_validation.json").write_text(json.dumps(out, indent=2) + "\n")

    f = lambda x: "n/a" if x is None else f"{x:.4f}"
    L = ["# Baselines on the validation week", "",
         f"Validation week {split.val_start} to {split.val_end}; training data before {split.val_start}. "
         f"Sample: {len(sampled):,} customers (seed {args.seed}), of which {len(customers):,} bought in the validation week "
         f"and are evaluated. Cold-start (no purchase before the cutoff): {results['popular_global']['cold_share']:.2%} of evaluated customers. "
         f"Coverage = distinct recommended articles / {n_articles:,} catalog articles.", "",
         "| baseline | MAP@12 | Recall@12 | NDCG@12 | coverage@12 | MAP cold | MAP warm | NDCG cold | NDCG warm |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        o, g = r["overall"], r["by_group"]
        L.append(f"| {n} | {f(o['map@12'])} | {f(o['recall@12'])} | {f(o['ndcg@12'])} | {f(o['coverage@12'])} | "
                 f"{f(g.get('cold', {}).get('map@12'))} | {f(g.get('warm', {}).get('map@12'))} | "
                 f"{f(g.get('cold', {}).get('ndcg@12'))} | {f(g.get('warm', {}).get('ndcg@12'))} |")
    L += ["", f"Repurchase fallback chosen on validation MAP@12: **{chosen}**. Best baseline overall: **{best}**.", ""]
    (RES / "baselines_validation.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"peak RSS {out['peak_rss_mb']} MB, {out['runtime_s']}s")


if __name__ == "__main__":
    main()
