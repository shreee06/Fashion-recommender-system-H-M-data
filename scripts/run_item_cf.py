"""Item-item CF + combinations on the VALIDATION week, with bootstrap CIs.

Populations:
  * evaluated: ALL customers who bought in the validation week (all methods)
  * training (item-CF similarity): the seeded 200k customer sample
Training/popularity data are strictly before the validation start. Test week is never touched.
"""
import argparse
import csv
import json
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashion_recsys import baselines as bl  # noqa: E402
from fashion_recsys.config import SAMPLE_SEED, SAMPLE_SIZE, make_split, sample_customers, to_day  # noqa: E402
from fashion_recsys.data import load_transaction_codes  # noqa: E402
from fashion_recsys.evaluation import (  # noqa: E402
    bootstrap_ci, build_eval_set, candidate_recall, coverage_at_k, paired_bootstrap, per_customer_metrics,
)
from fashion_recsys.item_cf import ItemCF  # noqa: E402

RES = ROOT / "evals" / "results"
LOG = RES / "tuning_log.csv"
METRICS = ("map@12", "recall@12", "ndcg@12")
BOOT_SEED, N_BOOT, L = 123, 1000, 100


def log_row(run, method, param, value, n_eval, m):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "run", "method", "param", "value", "population", "n_customers", "map@12", "recall@12", "ndcg@12"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), run, method, param, value,
                    "all validation-week buyers", n_eval] + [f"{m[k].mean():.6f}" for k in METRICS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-weeks", type=int, default=26)
    ap.add_argument("--max-items", type=int, default=20_000)
    ap.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--k-grid", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    ap.add_argument("--rep-grid", type=int, nargs="+", default=[2, 4, 6, 8, 12])
    args = ap.parse_args()
    t0 = time.time()
    run = datetime.now().strftime("%Y%m%d-%H%M%S")

    day, cust, art, customer_ids, article_ids = load_transaction_codes(ROOT / "data" / "processed")
    n_art = len(article_ids)
    split = make_split(day.min(), day.max())
    cutoff = to_day(split.val_start)
    sampled = sample_customers(np.unique(cust), args.sample_size, args.seed)

    customers, truths, hist = build_eval_set(day, cust, art, np.arange(len(customer_ids)), cutoff, to_day(split.val_end))
    n_eval = len(customers)
    cold = hist == 0
    in_sample = np.isin(customers, sampled)
    print(f"evaluated {n_eval:,} customers; cold {cold.mean():.2%}; in 200k sample {in_sample.sum():,}", flush=True)

    pop_g = bl.global_popular(day, art, cutoff, n_art, L).tolist()
    pop1 = bl.recent_popular(day, art, cutoff, 1, n_art, L).tolist()
    pop4 = bl.recent_popular(day, art, cutoff, 4, n_art, L).tolist()
    recents = bl.recent_items(day, cust, art, cutoff, customers, L)

    t = time.time()
    cf = ItemCF(args.window_weeks, args.max_items).fit(day, cust, art, cutoff, n_art, train_customers=sampled)
    print(f"fit CF: {len(cf.items):,} items, {cf.sim.nnz:,} sim entries, {cf.n_train_customers:,} training customers, {time.time()-t:.0f}s", flush=True)

    def score(lists):
        return per_customer_metrics(lists, truths)

    # ---- tune K (seed items) for item-CF alone, on validation
    cf_lists = {}
    best_k, best_map = None, -1
    for K in args.k_grid:
        cf_lists[K] = cf.recommend(day, cust, art, customers, k_seeds=K, n=L)
        m = score(bl.fill(cf_lists[K], pop4, L))
        log_row(run, "item_cf", "k_seeds", K, n_eval, m)
        print(f"  K={K}: MAP@12 {m['map@12'].mean():.4f}", flush=True)
        if m["map@12"].mean() > best_map:
            best_k, best_map = K, m["map@12"].mean()
    # ---- tune number of repurchase items kept first in the combination
    cf_best = cf_lists[best_k]
    best_r, best_map_r = None, -1
    for R in args.rep_grid:
        combo = bl.fill([bl.fill([r[:R]], cf_l, L)[0] for r, cf_l in zip(recents, cf_best)], pop4, L)
        m = score(combo)
        log_row(run, "repurchase+cf+pop", f"n_repurchase(k_seeds={best_k})", R, n_eval, m)
        print(f"  R={R}: MAP@12 {m['map@12'].mean():.4f}", flush=True)
        if m["map@12"].mean() > best_map_r:
            best_r, best_map_r = R, m["map@12"].mean()

    # ---- final methods, all as 100-long ranked lists (metrics use the first 12)
    methods = {
        "popular_global": [pop_g] * n_eval,
        "popular_last_1w": [pop1] * n_eval,
        "popular_last_4w": [pop4] * n_eval,
        "repurchase+last_1w": bl.fill(recents, pop1, L),
        "repurchase+last_4w": bl.fill(recents, pop4, L),
        f"item_cf(K={best_k})": bl.fill(cf_best, pop4, L),
        f"repurchase(top{best_r})+item_cf+pop4w": bl.fill(
            [bl.fill([r[:best_r]], c, L)[0] for r, c in zip(recents, cf_best)], pop4, L),
    }
    # the full-repurchase variant of the combination (literal spec: all recents first)
    methods["repurchase(all)+item_cf+pop4w"] = bl.fill([bl.fill([r], c, L)[0] for r, c in zip(recents, cf_best)], pop4, L)
    names = list(methods)

    per = {n: score(l) for n, l in methods.items()}
    table = {}
    for n in names:
        row = {"n_customers": n_eval, "coverage@12": coverage_at_k(methods[n], n_art)}
        for k in METRICS:
            mean, lo, hi = bootstrap_ci(per[n][k], N_BOOT, BOOT_SEED)
            row[k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
            row[k + "_cold"] = float(per[n][k][cold].mean())
            row[k + "_warm"] = float(per[n][k][~cold].mean())
        row["map@12_in_200k_sample"] = float(per[n]["map@12"][in_sample].mean())
        table[n] = row

    cand = {}
    for n in (f"item_cf(K={best_k})", "repurchase+last_4w", "popular_last_4w", f"repurchase(top{best_r})+item_cf+pop4w", "repurchase(all)+item_cf+pop4w"):
        cand[n] = {f"recall@{c}": candidate_recall(methods[n], truths, c) for c in (12, 50, 100)}

    # union candidate pool for a later reranker (unranked): own recents + CF top-100 + popularity top-100
    unions = [set(r) | set(c) | set(pop4) for r, c in zip(recents, cf_best)]
    cand["union(repurchase<=100 + item_cf top100 + pop4w top100)"] = {
        "recall@pool": float(np.mean([len(u & t) / len(t) for u, t in zip(unions, truths)])),
        "mean_pool_size": float(np.mean([len(u) for u in unions])),
    }

    combo_n, rep_n, cf_n = f"repurchase(top{best_r})+item_cf+pop4w", "repurchase+last_4w", f"item_cf(K={best_k})"
    pairs = [(combo_n, rep_n), ("repurchase(all)+item_cf+pop4w", rep_n), (cf_n, rep_n), (cf_n, "popular_last_4w"), (rep_n, "repurchase+last_1w")]
    paired = {}
    for a, b in pairs:
        paired[f"{a} - {b}"] = {k: paired_bootstrap(per[a][k], per[b][k], N_BOOT, BOOT_SEED) for k in METRICS}

    out = {
        "split": split.to_json(), "eval_week": "validation", "train_cutoff": split.val_start.isoformat(),
        "populations": {
            "evaluated": "all customers who bought in the validation week",
            "n_evaluated": n_eval, "cold_start_share": float(cold.mean()),
            "n_evaluated_in_200k_sample": int(in_sample.sum()),
            "cf_training_customers": f"seeded sample of {len(sampled):,} (seed {args.seed}); {cf.n_train_customers:,} had window purchases",
        },
        "item_cf": {"window_weeks": args.window_weeks, "max_items": args.max_items, "modelled_items": int(len(cf.items)),
                    "max_neighbors": cf.max_neighbors, "k_seeds_chosen": best_k, "n_repurchase_chosen": best_r,
                    "exclude_seen": True, "similarity_entries": int(cf.sim.nnz)},
        "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "ci": "95% percentile"},
        "results": table, "candidate_recall": cand, "paired": paired,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, "runtime_s": round(time.time() - t0, 1),
    }
    (RES / "item_cf_validation.json").write_text(json.dumps(out, indent=2) + "\n")

    f = lambda d: f"{d['mean']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]"
    M = ["# Item-CF and combinations on the validation week", "",
         f"Validation week {split.val_start} to {split.val_end}; everything trained on data before {split.val_start}. "
         f"**Evaluated population: all {n_eval:,} customers who bought in the validation week** "
         f"({cold.mean():.2%} cold-start; {int(in_sample.sum()):,} of them are in the 200k sample). "
         f"Item-CF similarity is trained on the seeded 200k customer sample only, using the last {args.window_weeks} weeks, "
         f"the {len(cf.items):,} most popular articles of that window, {cf.max_neighbors} neighbours per article, "
         f"seeds = last K={best_k} distinct purchases (K chosen on validation, see tuning_log.csv). "
         f"95% CIs: customer bootstrap, {N_BOOT} resamples, seed {BOOT_SEED}.", "",
         "| method | MAP@12 [95% CI] | Recall@12 [95% CI] | NDCG@12 [95% CI] | coverage@12 | MAP cold | MAP warm |", "|---|---|---|---|---|---|---|"]
    for n, r in table.items():
        M.append(f"| {n} | {f(r['map@12'])} | {f(r['recall@12'])} | {f(r['ndcg@12'])} | {r['coverage@12']:.4f} | {r['map@12_cold']:.4f} | {r['map@12_warm']:.4f} |")
    M += ["", "## Paired differences (same customers, bootstrap)", "", "| A - B | metric | diff | 95% CI | excludes 0 |", "|---|---|---|---|---|"]
    for ab, d in paired.items():
        for k in METRICS:
            x = d[k]
            M.append(f"| {ab} | {k} | {x['diff']:+.4f} | [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}] | {'yes' if x['excludes_zero'] else 'no'} |")
    M += ["", "## Candidate recall (mean share of each customer's relevant items found in the top N)", "",
          "| method | top 12 | top 50 | top 100 |", "|---|---|---|---|"]
    for n, r in cand.items():
        if "recall@pool" in r:
            continue
        M.append(f"| {n} | {r['recall@12']:.4f} | {r['recall@50']:.4f} | {r['recall@100']:.4f} |")
    for n, r in cand.items():
        if "recall@pool" in r:
            M += ["", f"Unranked union pool, {n}: recall {r['recall@pool']:.4f} with mean pool size {r['mean_pool_size']:.1f} (upper bound for a reranker over this pool)."]
    M.append("")
    (RES / "item_cf_validation.md").write_text("\n".join(M))
    print("\n".join(M))
    print(f"peak RSS {out['peak_rss_mb']} MB, {out['runtime_s']}s")


if __name__ == "__main__":
    main()
