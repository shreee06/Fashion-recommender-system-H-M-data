"""Step 5: better candidate generators, evaluated on the VALIDATION week only.

Evaluated population = all customers who bought in the validation week (same as step 4).
Every source uses only transactions before the validation start plus static metadata.
"""
import argparse
import itertools
import json
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fashion_recsys import baselines as bl  # noqa: E402
from fashion_recsys.config import SAMPLE_SEED, SAMPLE_SIZE, make_split, sample_customers, to_day  # noqa: E402
from fashion_recsys.data import load_transaction_codes  # noqa: E402
from fashion_recsys.evaluation import (  # noqa: E402
    bootstrap_ci, build_eval_set, candidate_recall, coverage_at_k, paired_bootstrap, per_customer_metrics,
)
from fashion_recsys.item_cf import ItemCF  # noqa: E402
from fashion_recsys.segments import (  # noqa: E402
    AGE_LABELS, age_bucket, customer_segment_popular, dominant_group, group_popular,
)
from fashion_recsys.variants import VariantCandidates  # noqa: E402
from run_item_cf import LOG, METRICS, BOOT_SEED, N_BOOT, log_row  # noqa: E402

RES = ROOT / "evals" / "results"
L = 100
T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def merge_unique(lists, n):
    """Round-robin interleave of ranked lists, deduplicated, truncated to n."""
    out, seen = [], set()
    for i in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if i < len(x) and x[i] not in seen:
                seen.add(x[i])
                out.append(x[i])
    return out[:n]


def combine(rep, srcs, order, caps, R, fallback):
    """repurchase top-R, then each source in `order` (capped), then fallback; exactly 12."""
    res = []
    for i, r in enumerate(rep):
        out, seen = list(r[:R]), set(r[:R])
        for name, cap in zip(order, caps):
            added = 0
            for x in srcs[name][i]:
                if len(out) >= 12 or added >= cap:
                    break
                if x not in seen:
                    seen.add(x)
                    out.append(x)
                    added += 1
        for x in fallback:
            if len(out) >= 12:
                break
            if x not in seen:
                seen.add(x)
                out.append(x)
        res.append(out[:12])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="+", default=[4, 8, 26])
    ap.add_argument("--n-items", type=int, nargs="+", default=[20_000, 40_000])
    ap.add_argument("--taus", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--k-grid", type=int, nargs="+", default=[3, 5, 10])
    args = ap.parse_args()
    run = datetime.now().strftime("%Y%m%d-%H%M%S")

    day, cust, art, customer_ids, article_ids = load_transaction_codes(ROOT / "data" / "processed")
    n_art, n_cust = len(article_ids), len(customer_ids)
    split = make_split(day.min(), day.max())
    cutoff = to_day(split.val_start)
    customers, truths, hist = build_eval_set(day, cust, art, np.arange(n_cust), cutoff, to_day(split.val_end))
    n_eval = len(customers)
    cold = hist == 0
    say(f"evaluated {n_eval:,} customers, cold {cold.mean():.2%}")

    arts = pq.read_table(ROOT / "data/processed/articles.parquet", columns=["product_code", "index_group_no"]).to_pandas()
    product_code = arts["product_code"].astype("category").cat.codes.to_numpy()
    article_group = arts["index_group_no"].astype("category").cat.codes.to_numpy().astype(np.int64)
    n_groups = int(article_group.max()) + 1
    age = pq.read_table(ROOT / "data/processed/customers.parquet", columns=["age"])["age"].to_numpy(zero_copy_only=False).astype(float)
    bucket = age_bucket(age)

    score = lambda lists: per_customer_metrics(lists, truths)
    pop4 = bl.recent_popular(day, art, cutoff, 4, n_art, L).tolist()
    recents = bl.recent_items(day, cust, art, cutoff, customers, L)
    rep_pop4 = bl.fill(recents, pop4, L)
    pers = {}  # per-customer metrics of the methods used in the final table

    def fill_pop(lists):
        return bl.fill(lists, pop4, L)

    # ------------------------------------------------------------------ 1. item-CF
    sampled = sample_customers(np.unique(cust), SAMPLE_SIZE, SAMPLE_SEED)
    old = ItemCF(26, 20_000).fit(day, cust, art, cutoff, n_art, train_customers=sampled)
    old_lists = old.recommend(day, cust, art, customers, k_seeds=3, n=L)
    pers["item_cf_step4 (200k sample, 26w, 20k items, binary, K=3)"] = score(fill_pop(old_lists))
    say("step-4 item-CF reproduced: MAP", pers[next(iter(pers))]["map@12"].mean())
    del old

    grid = {}  # config -> per-customer metrics

    def run_cf(w, N, tau, K, keep=False):
        t = time.time()
        cf = ItemCF(w, N).fit(day, cust, art, cutoff, n_art, recency_tau_weeks=tau)
        lists = cf.recommend(day, cust, art, customers, k_seeds=K, n=L)
        m = score(fill_pop(lists))
        key = (w, N, tau, K)
        grid[key] = m
        log_row(run, "item_cf_all_customers", "window_weeks/max_items/tau_weeks/k_seeds",
                f"{w}/{N}/{'none' if tau is None else tau}/{K}", n_eval, m)
        say(f"CF w={w} N={N} tau={tau} K={K}: MAP {m['map@12'].mean():.4f} "
            f"(items {len(cf.items):,}, sim {cf.sim.nnz:,}, {time.time()-t:.0f}s)")
        return cf, lists, m

    for w, N in itertools.product(args.windows, args.n_items):
        run_cf(w, N, None, 3)
    best = max(grid, key=lambda k: grid[k]["map@12"].mean())
    w, N = best[0], best[1]
    for tau in args.taus:
        run_cf(w, N, tau, 3)
    best = max(grid, key=lambda k: grid[k]["map@12"].mean())
    for K in args.k_grid:
        if K != 3:
            run_cf(best[0], best[1], best[2], K)
    best = max(grid, key=lambda k: grid[k]["map@12"].mean())
    say("best CF config (window, N, tau, K):", best)
    cf, cf_lists, _ = run_cf(*best)
    cf_info = {"window_weeks": best[0], "max_items": best[1], "tau_weeks": best[2], "k_seeds": best[3],
               "modelled_items": int(len(cf.items)), "similarity_entries": int(cf.sim.nnz),
               "training_customers": int(cf.n_train_customers)}
    del cf
    pers["item_cf_all_customers_binary (26w, 20k, K=3)"] = grid[(26, 20_000, None, 3)]
    pers["item_cf_new (best config)"] = grid[best]

    # ------------------------------------------------------------------ 2. variants
    best_v, best_vm = None, -1
    for pw in (1, 4, 8):
        v = VariantCandidates(pw).fit(day, art, cutoff, product_code, n_art)
        m = score(fill_pop(v.recommend(recents, 5, L)))
        log_row(run, "variants", "pop_weeks/k_seeds", f"{pw}/5", n_eval, m)
        say(f"variants pop_weeks={pw} k=5: MAP {m['map@12'].mean():.4f}")
        if m["map@12"].mean() > best_vm:
            best_v, best_vm = (pw, 5), m["map@12"].mean()
    for ks in (3, 10):
        v = VariantCandidates(best_v[0]).fit(day, art, cutoff, product_code, n_art)
        m = score(fill_pop(v.recommend(recents, ks, L)))
        log_row(run, "variants", "pop_weeks/k_seeds", f"{best_v[0]}/{ks}", n_eval, m)
        say(f"variants pop_weeks={best_v[0]} k={ks}: MAP {m['map@12'].mean():.4f}")
        if m["map@12"].mean() > best_vm:
            best_v, best_vm = (best_v[0], ks), m["map@12"].mean()
    v = VariantCandidates(best_v[0]).fit(day, art, cutoff, product_code, n_art)
    var_lists = v.recommend(recents, best_v[1], L)
    pers["variants_alone (+pop4w fill)"] = score(fill_pop(var_lists))

    # ------------------------------------------------------------------ 3. segment popularity
    sizes_all = np.bincount(bucket, minlength=6)
    sizes_eval = np.bincount(bucket[customers], minlength=6)
    best_s, best_sm = None, -1
    for sw in (1, 4, 8):
        tops = customer_segment_popular(day, cust, art, bucket, 6, cutoff, sw, n_art, L)
        lists = [tops[b].tolist() for b in bucket[customers]]
        m = score(lists)
        log_row(run, "segment_pop_age", "weeks", sw, n_eval, m)
        say(f"age-bucket popularity weeks={sw}: MAP {m['map@12'].mean():.4f}")
        if m["map@12"].mean() > best_sm:
            best_s, best_sm = sw, m["map@12"].mean()
    tops = customer_segment_popular(day, cust, art, bucket, 6, cutoff, best_s, n_art, L)
    age_lists = [tops[b].tolist() for b in bucket[customers]]
    pers[f"popular_by_age_bucket ({best_s}w)"] = score(age_lists)

    dom = dominant_group(day, cust, art, article_group, n_groups, cutoff, n_cust)
    best_g, best_gm = None, -1
    for gw in (1, 4, 8):
        gtops = {g: t.tolist() for g, t in group_popular(day, art, article_group, n_groups, cutoff, gw, n_art, L).items()}
        lists = [gtops[g] if g >= 0 else pop4 for g in dom[customers]]
        m = score(fill_pop(lists))
        log_row(run, "segment_pop_index_group", "weeks", gw, n_eval, m)
        say(f"index-group popularity weeks={gw}: MAP {m['map@12'].mean():.4f}")
        if m["map@12"].mean() > best_gm:
            best_g, best_gm = gw, m["map@12"].mean()
    gtops = {g: t.tolist() for g, t in group_popular(day, art, article_group, n_groups, cutoff, best_g, n_art, L).items()}
    grp_lists = [gtops[g] if g >= 0 else pop4 for g in dom[customers]]
    pers[f"popular_by_index_group ({best_g}w, cold -> global pop4w)"] = score(grp_lists)
    pers["popular_last_4w (global)"] = score([pop4] * n_eval)
    pers["repurchase+last_4w"] = score(rep_pop4)

    # ------------------------------------------------------------------ 4. combination
    seg_name = "segage" if best_sm >= best_gm else "seggrp"
    srcs = {"cf": cf_lists, "var": var_lists, "segage": age_lists, "seggrp": grp_lists}
    names = ["cf", "var", seg_name]
    say(f"combination: segment source = {seg_name}")
    comb_best, comb_bm = None, -1

    def try_comb(order, caps, R):
        nonlocal comb_best, comb_bm
        m = score(combine(recents, srcs, order, caps, R, pop4))
        log_row(run, "combination", "order/caps/R", f"{'>'.join(order)}/{'-'.join(map(str, caps))}/{R}", n_eval, m)
        if m["map@12"].mean() > comb_bm:
            comb_best, comb_bm = (order, caps, R), m["map@12"].mean()

    for order in itertools.permutations(names):
        for R in (4, 6, 8):
            try_comb(order, (3, 3, 100), R)
    say("combination stage A best:", comb_best, f"MAP {comb_bm:.4f}")
    order, _, R = comb_best
    for c1, c2 in itertools.product((1, 2, 3, 4, 6), repeat=2):
        try_comb(order, (c1, c2, 100), R)
    say("combination stage B best:", comb_best, f"MAP {comb_bm:.4f}")
    order, caps, R = comb_best
    final_lists = combine(recents, srcs, order, caps, R, pop4)
    pers["combination_new"] = score(final_lists)
    s4 = bl.fill([bl.fill([r[:8]], c, L)[0] for r, c in zip(recents, old_lists)], pop4, L)
    pers["combination_step4 (repurchase top8 + step-4 CF + pop4w)"] = score(s4)

    # ------------------------------------------------------------------ results table
    cov_lists = {"popular_last_4w (global)": [pop4] * n_eval, "repurchase+last_4w": rep_pop4,
                 "combination_new": final_lists, "combination_step4 (repurchase top8 + step-4 CF + pop4w)": s4}
    table = {}
    for n, m in pers.items():
        row = {"n_customers": n_eval}
        for k in METRICS:
            mean, lo, hi = bootstrap_ci(m[k], N_BOOT, BOOT_SEED)
            row[k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
            row[k + "_cold"], row[k + "_warm"] = float(m[k][cold].mean()), float(m[k][~cold].mean())
        table[n] = row
    pn = list(pers)
    f_ = lambda x: next(n for n in pn if n.startswith(x))
    pairs = [
        (f_("item_cf_all_customers_binary"), f_("item_cf_step4")),
        (f_("item_cf_new"), f_("item_cf_all_customers_binary")),
        (f_("item_cf_new"), f_("item_cf_step4")),
        (f_("variants_alone"), "popular_last_4w (global)"),
        (f_("popular_by_age_bucket"), "popular_last_4w (global)"),
        (f_("popular_by_index_group"), "popular_last_4w (global)"),
        ("combination_new", "repurchase+last_4w"),
        ("combination_new", f_("combination_step4")),
        (f_("combination_step4"), "repurchase+last_4w"),
    ]
    paired = {f"{a} - {b}": {k: paired_bootstrap(pers[a][k], pers[b][k], N_BOOT, BOOT_SEED) for k in METRICS} for a, b in pairs}

    # ------------------------------------------------------------------ 5. candidate recall
    own = {"repurchase (recents)": recents, "item_cf_step4": old_lists, "item_cf_new": cf_lists,
           "variants": var_lists, "popular_by_age_bucket": age_lists, "popular_by_index_group": grp_lists,
           "popular_last_4w": [pop4] * n_eval}
    cand = {}
    for n, ls in own.items():
        cand[n] = {f"recall@{c}": candidate_recall(ls, truths, c) for c in (12, 50, 100)}
        cand[n]["mean_list_length"] = float(np.mean([len(x) for x in ls]))
    parts = ["repurchase (recents)", "item_cf_new", "variants", "popular_by_age_bucket", "popular_by_index_group", "popular_last_4w"]
    merged = [merge_unique([own[p][i] for p in parts], 10_000) for i in range(n_eval)]
    cand["UNION (round-robin merge of: " + ", ".join(parts) + ")"] = {
        **{f"recall@{c}": candidate_recall(merged, truths, c) for c in (12, 50, 100)},
        "recall@full_pool": float(np.mean([len(set(u) & t) / len(t) for u, t in zip(merged, truths)])),
        "mean_list_length": float(np.mean([len(u) for u in merged])),
    }
    union_pool = cand[next(k for k in cand if k.startswith("UNION"))]

    out = {
        "split": split.to_json(), "eval_week": "validation", "train_cutoff": split.val_start.isoformat(),
        "population": {"evaluated": "all customers who bought in the validation week", "n_evaluated": n_eval,
                       "cold_start_share": float(cold.mean())},
        "age_buckets": {"labels": AGE_LABELS, "customers_all": sizes_all.tolist(), "customers_evaluated": sizes_eval.tolist()},
        "item_cf_new": cf_info, "variants": {"pop_weeks": best_v[0], "k_seeds": best_v[1]},
        "segment_popularity": {"age_weeks": best_s, "index_group_weeks": best_g, "segment_used_in_combination": seg_name,
                               "sales_channel_segments": "not evaluated (channel not in the cached arrays)"},
        "combination": {"order": list(order), "caps": list(caps), "n_repurchase": R,
                        "form": "repurchase top-R, then each source in order up to its cap (last uncapped), then global pop4w, deduplicated, exactly 12"},
        "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "ci": "95% percentile"},
        "results": table, "paired": paired, "candidate_recall": cand,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        "runtime_s": round(time.time() - T0, 1),
    }
    (RES / "candidates_validation.json").write_text(json.dumps(out, indent=2) + "\n")

    f = lambda d: f"{d['mean']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]"
    M = ["# Candidate generators on the validation week", "",
         f"Validation week {split.val_start} to {split.val_end}; all sources use only data before {split.val_start} plus static metadata. "
         f"**Evaluated population: all {n_eval:,} validation-week buyers** ({cold.mean():.2%} cold-start). 95% CIs: customer bootstrap, {N_BOOT} resamples, seed {BOOT_SEED}. "
         f"Standalone sources are filled with global pop4w to 12 items.", "",
         f"Item-CF (new): trained on ALL customers, window {best[0]} weeks, top {best[1]:,} articles ({len(pers) and cf_info['modelled_items']:,} modelled), 100 neighbours, "
         f"recency tau = {best[2]} weeks, K = {best[3]} seed items.", "",
         "Age buckets (customers in customers.csv / among evaluated): " + ", ".join(f"{l}: {a:,}/{e:,}" for l, a, e in zip(AGE_LABELS, sizes_all, sizes_eval)) + ".", "",
         f"Combination: repurchase top-{R}, then {' > '.join(order)} with caps {caps} (last uncapped), then pop4w.", "",
         "| method | MAP@12 [95% CI] | Recall@12 [95% CI] | NDCG@12 [95% CI] | MAP cold | MAP warm |", "|---|---|---|---|---|---|"]
    for n, r in table.items():
        M.append(f"| {n} | {f(r['map@12'])} | {f(r['recall@12'])} | {f(r['ndcg@12'])} | {r['map@12_cold']:.4f} | {r['map@12_warm']:.4f} |")
    M += ["", "## Paired differences (A - B, same customers)", "", "| A - B | metric | diff | 95% CI | excludes 0 |", "|---|---|---|---|---|"]
    for ab, d in paired.items():
        for k in METRICS:
            x = d[k]
            M.append(f"| {ab} | {k} | {x['diff']:+.4f} | [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}] | {'yes' if x['excludes_zero'] else 'no'} |")
    M += ["", "## Candidate recall (each source's own list, unfilled)", "",
          "| source | recall@12 | recall@50 | recall@100 | mean list length |", "|---|---|---|---|---|"]
    for n, r in cand.items():
        M.append(f"| {n} | {r['recall@12']:.4f} | {r['recall@50']:.4f} | {r['recall@100']:.4f} | {r['mean_list_length']:.1f} |")
    M += ["", f"Union pool: recall {union_pool['recall@full_pool']:.4f} at mean pool size {union_pool['mean_list_length']:.1f} "
              "(this is the ceiling for a reranker over the full pool).", ""]
    (RES / "candidates_validation.md").write_text("\n".join(M))
    say("done; peak RSS", out["peak_rss_mb"], "MB")


if __name__ == "__main__":
    main()
