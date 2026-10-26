"""Step 7: candidate pool + LightGBM lambdarank reranker.

Train: cutoff 2020-09-02 (labels = purchases 2020-09-02..08), customers sampled from the buyers of that week.
Validate: cutoff 2020-09-09, all validation-week buyers (same 72,019 as before).
The test week is never touched.
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
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fashion_recsys import baselines as bl  # noqa: E402
from fashion_recsys.config import make_split, to_day  # noqa: E402
from fashion_recsys.data import load_price_channel, load_transaction_codes  # noqa: E402
from fashion_recsys.evaluation import bootstrap_ci, build_eval_set, candidate_recall, paired_bootstrap, per_customer_metrics  # noqa: E402
from fashion_recsys.item_cf import ItemCF  # noqa: E402
from fashion_recsys.reranker import (  # noqa: E402
    CONTENT_FEATURES, FEATURES, TT_FEATURES, Meta, build_pool, cutoff_stats, fit_ranker, labels_for,
    pair_features, top_k_per_customer,
)
from fashion_recsys.segments import age_bucket, customer_segment_popular  # noqa: E402
from fashion_recsys.variants import VariantCandidates  # noqa: E402
from run_candidates import combine  # noqa: E402
from run_item_cf import LOG, METRICS, BOOT_SEED, N_BOOT  # noqa: E402

RES, PROC = ROOT / "evals" / "results", ROOT / "data" / "processed"
T0 = time.time()
CHUNK = 5000


def say(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def log_trial(run, method, param, value, population, n, m):
    with open(LOG, "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), run, method, param, value, population, n]
                               + [f"{m[k]:.6f}" for k in METRICS])


class Ctx:
    """Everything a cutoff needs to generate candidates and features."""

    def __init__(self, D, cutoff, tt_file):
        d = D
        self.D, self.cutoff = d, cutoff
        cj = d["cfg"]
        ic, vc, sc = cj["item_cf_new"], cj["variants"], cj["segment_popularity"]
        self.stats = cutoff_stats(d["day"], d["cust"], d["art"], d["price"], d["chan"], cutoff, d["n_art"], d["n_cust"])
        self.cf = ItemCF(ic["window_weeks"], ic["max_items"]).fit(d["day"], d["cust"], d["art"], cutoff, d["n_art"], recency_tau_weeks=ic["tau_weeks"])
        self.k_seeds, self.var_k = ic["k_seeds"], vc["k_seeds"]
        self.var = VariantCandidates(vc["pop_weeks"]).fit(d["day"], d["art"], cutoff, d["product_code"], d["n_art"])
        self.tops = customer_segment_popular(d["day"], d["cust"], d["art"], d["bucket"], 6, cutoff, sc["age_weeks"], d["n_art"], 100)
        self.pop4 = bl.recent_popular(d["day"], d["art"], cutoff, 4, d["n_art"], 100).tolist()
        z = np.load(tt_file)
        assert int(z["cutoff_day"]) == cutoff
        self.tt_customers, self.tt_codes, self.tt_scores = z["customers"], z["codes"], z["scores"]

    def sources(self, chunk):
        d = self.D
        recents = bl.recent_items(d["day"], d["cust"], d["art"], self.cutoff, chunk, 100)
        cf = self.cf.recommend(d["day"], d["cust"], d["art"], chunk, k_seeds=self.k_seeds, n=100)
        var = self.var.recommend(recents, self.var_k, 100)
        age = [self.tops[b].tolist() for b in d["bucket"][chunk]]
        pos = np.searchsorted(self.tt_customers, chunk)
        assert np.array_equal(self.tt_customers[pos], chunk), "customer missing from two-tower file"
        return {"recent": recents, "cf": cf, "var": var, "agepop": age, "pop": [self.pop4] * len(chunk),
                "tt": self.tt_codes[pos], "tt_score": self.tt_scores[pos]}

    def pool_features(self, chunk, src):
        d = self.D
        pcust, part, ranks, tts = build_pool(src, len(chunk), d["n_art"])
        X, a_cnt = pair_features(self.stats, d["meta"], d["day"], d["cust"], d["art"], chunk, pcust, part, ranks, tts, d["n_art"])
        return pcust, part, X, a_cnt


def build_matrix(ctx, customers, truths):
    Xs, ys, gs = [], [], []
    for s in range(0, len(customers), CHUNK):
        chunk = customers[s : s + CHUNK]
        pcust, part, X, _ = ctx.pool_features(chunk, ctx.sources(chunk))
        Xs.append(X)
        ys.append(labels_for(pcust, part, truths[s : s + CHUNK], ctx.D["n_art"]))
        gs.append(np.bincount(pcust, minlength=len(chunk)))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-customers", type=int, default=30_000)
    ap.add_argument("--val-limit", type=int, default=0, help="dev only: evaluate on the first N validation customers")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run = "rerank-" + datetime.now().strftime("%Y%m%d-%H%M%S")

    day, cust, art, customer_ids, article_ids = load_transaction_codes(PROC)
    price, chan = load_price_channel(PROC)
    n_art, n_cust = len(article_ids), len(customer_ids)
    split = make_split(day.min(), day.max())
    val_start = to_day(split.val_start)
    train_cut = val_start - 7
    arts = pq.read_table(PROC / "articles.parquet", columns=["product_code", "product_type_no", "colour_group_code", "department_no"]).to_pandas()
    codes = lambda c: arts[c].astype("category").cat.codes.to_numpy().astype(np.int64)
    ages = pq.read_table(PROC / "customers.parquet", columns=["age"])["age"].to_numpy(zero_copy_only=False).astype(np.float32)
    meta = Meta(codes("product_code"), codes("product_type_no"), codes("colour_group_code"), codes("department_no"), ages)
    D = dict(day=day, cust=cust, art=art, price=price, chan=chan, n_art=n_art, n_cust=n_cust, meta=meta,
             product_code=meta.product_code, bucket=age_bucket(ages), cfg=json.loads((RES / "candidates_validation.json").read_text()))

    # ---------------- training set: customers sampled from the buyers of the training label week
    ctx_tr = Ctx(D, train_cut, PROC / "two_tower_candidates_heldout.npz")
    say("training-cutoff sources ready")
    rng = np.random.default_rng(0)
    pool_customers = ctx_tr.tt_customers.astype(np.int64)
    n_tr = min(args.train_customers, len(pool_customers))
    sampled = np.sort(rng.choice(pool_customers, n_tr, replace=False))
    perm = rng.permutation(n_tr)
    n_fit = int(0.8 * n_tr)
    fit_c, inner_c = np.sort(sampled[perm[:n_fit]]), np.sort(sampled[perm[n_fit:]])
    sets = {}
    for name, cs in (("fit", fit_c), ("inner", inner_c)):
        cu, tr, _ = build_eval_set(day, cust, art, cs, train_cut, train_cut + 6)
        assert np.array_equal(cu, cs)
        sets[name] = (cu, tr)
    Xf, yf, gf = build_matrix(ctx_tr, *sets["fit"])
    Xi, yi, gi = build_matrix(ctx_tr, *sets["inner"])
    say(f"training matrix: {len(fit_c):,} fit customers, {Xf.shape[0]:,} rows ({yf.sum():,.0f} positives); "
        f"inner-validation {len(inner_c):,} customers, {Xi.shape[0]:,} rows ({yi.sum():,.0f} positives); {Xf.shape[1]} features; "
        f"mean pool {Xf.shape[0] / len(fit_c):.0f}")
    pop_label = f"inner held-out customers of the training week (cutoff 2020-09-02), {len(inner_c)} customers"

    def inner_map(booster, cols):
        score = booster.predict(Xi[:, cols], num_iteration=booster.best_iteration)
        pc = np.repeat(np.arange(len(inner_c)), gi)
        lists = top_k_per_customer(pc, np.arange(len(pc)), score, len(inner_c), 12)
        start = np.cumsum(gi) - gi
        hits = []
        for i, l in enumerate(lists):
            rel = np.flatnonzero(yi[start[i] : start[i] + gi[i]] > 0)
            l_local = [x - start[i] for x in l]
            rel_set = set(rel.tolist())
            ap_, h = 0.0, 0
            for r_, x in enumerate(l_local, 1):
                if x in rel_set:
                    h += 1
                    ap_ += h / r_
            hits.append(ap_ / min(12, len(sets["inner"][1][i])))
        return float(np.mean(hits))

    all_cols = list(range(len(FEATURES)))
    grid = [dict(num_leaves=31, learning_rate=0.1, min_data_in_leaf=100),
            dict(num_leaves=63, learning_rate=0.1, min_data_in_leaf=100),
            dict(num_leaves=31, learning_rate=0.05, min_data_in_leaf=100),
            dict(num_leaves=15, learning_rate=0.1, min_data_in_leaf=100),
            dict(num_leaves=31, learning_rate=0.1, min_data_in_leaf=400)]
    if args.smoke:
        grid = grid[:1]
    best_p, best_m, best_model = None, -1, None
    for p in grid:
        t = time.time()
        b = fit_ranker(Xf, yf, gf, Xi, yi, gi, FEATURES, p, args.rounds)
        m = inner_map(b, all_cols)
        say(f"trial {p}: inner MAP@12 {m:.4f}, best iter {b.best_iteration}, {time.time()-t:.0f}s")
        log_trial(run, "lgbm_lambdarank", "num_leaves/lr/min_data/best_iter", f"{p['num_leaves']}/{p['learning_rate']}/{p['min_data_in_leaf']}/{b.best_iteration}",
                  pop_label, len(inner_c), {"map@12": m, "recall@12": float("nan"), "ndcg@12": float(b.best_score["valid_0"]["ndcg@12"])})
        if m > best_m:
            best_p, best_m, best_model = p, m, b
    say("chosen params:", best_p, f"inner MAP@12 {best_m:.4f}")

    models = {"reranker (full)": (best_model, all_cols)}
    for label, drop in (("reranker without two-tower features", TT_FEATURES), ("reranker without content features", CONTENT_FEATURES)):
        cols = [i for i, f in enumerate(FEATURES) if f not in drop]
        names = [FEATURES[i] for i in cols]
        b = fit_ranker(Xf[:, cols], yf, gf, Xi[:, cols], yi, gi, names, best_p, args.rounds)
        m = inner_map(b, cols)
        say(f"ablation {label}: inner MAP@12 {m:.4f}, best iter {b.best_iteration}")
        log_trial(run, "lgbm_ablation", label, f"best_iter={b.best_iteration}", pop_label, len(inner_c),
                  {"map@12": m, "recall@12": float("nan"), "ndcg@12": float(b.best_score["valid_0"]["ndcg@12"])})
        models[label] = (b, cols)
    del Xf, yf, Xi, yi
    train_rows = int(gf.sum())

    imp = best_model.feature_importance(importance_type="gain")
    importance = {f: float(v) for f, v in sorted(zip(FEATURES, imp / max(imp.sum(), 1)), key=lambda x: -x[1])}

    # ---------------- validation week: all buyers, chunked
    customers, truths, hist = build_eval_set(day, cust, art, np.arange(n_cust), val_start, to_day(split.val_end))
    if args.val_limit:
        customers, truths, hist = customers[: args.val_limit], truths[: args.val_limit], hist[: args.val_limit]
    n_eval, cold = len(customers), hist == 0
    ctx = Ctx(D, val_start, PROC / "two_tower_candidates_validation.npz")
    say(f"validation: {n_eval:,} customers; sources ready")
    cj = D["cfg"]["combination"]
    order, caps, R = cj["order"], cj["caps"], cj["n_repurchase"]

    methods = ["repurchase+last_4w", "combination_step5", "two_tower_alone"] + list(models)
    per = {m: {k: [] for k in METRICS} for m in methods}
    all_lists = {m: [] for m in methods}
    pool_recall, pool_size, bought = [], [], {m: [0, 0] for m in methods}
    for s in range(0, n_eval, CHUNK):
        chunk, tr = customers[s : s + CHUNK], truths[s : s + CHUNK]
        src = ctx.sources(chunk)
        pcust, part, X, a_cnt = ctx.pool_features(chunk, src)
        lab = labels_for(pcust, part, tr, D["n_art"])
        pool_recall += (np.bincount(pcust, weights=lab, minlength=len(chunk)) / np.array([len(t) for t in tr])).tolist()
        pool_size += np.bincount(pcust, minlength=len(chunk)).tolist()
        lists = {"repurchase+last_4w": bl.fill(src["recent"], ctx.pop4, 12),
                 "combination_step5": combine(src["recent"], {"cf": src["cf"], "var": src["var"], "segage": src["agepop"]}, order, caps, R, ctx.pop4),
                 "two_tower_alone": src["tt"][:, :12].tolist()}
        for name, (b, cols) in models.items():
            sc = b.predict(X[:, cols], num_iteration=b.best_iteration)
            lists[name] = top_k_per_customer(pcust, part, sc, len(chunk), 12)
        # share of recommended items the customer already bought before the cutoff
        n_pre = ctx.stats.n_pre
        row_of = np.full(n_cust, -1, dtype=np.int64)
        row_of[chunk] = np.arange(len(chunk))
        rr = row_of[cust[:n_pre]]
        mk = rr >= 0
        hk = np.unique(rr[mk] * n_art + art[:n_pre][mk].astype(np.int64))
        for m, ls in lists.items():
            assert all(len(l) == 12 and len(set(l)) == 12 for l in ls), m
            q = np.repeat(np.arange(len(ls)), 12) * n_art + np.fromiter((x for l in ls for x in l), dtype=np.int64)
            pos = np.clip(np.searchsorted(hk, q), 0, len(hk) - 1)
            bought[m][0] += int((hk[pos] == q).sum())
            bought[m][1] += len(q)
            p = per_customer_metrics(ls, tr)
            for k in METRICS:
                per[m][k].append(p[k])
            all_lists[m] += ls
        say(f"  chunk {s // CHUNK + 1}/{-(-n_eval // CHUNK)} done")
    per = {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in per.items()}
    for m, v in (("repurchase+last_4w", 0.0242), ("combination_step5", 0.0261), ("two_tower_alone", 0.0106)):
        say(f"check {m}: MAP {per[m]['map@12'].mean():.4f} (earlier report {v})")
        if not args.val_limit:
            assert abs(per[m]["map@12"].mean() - v) < 6e-5, m

    table = {}
    for m in methods:
        r = {"coverage@12": len({x for l in all_lists[m] for x in l}) / n_art, "already_bought_share": bought[m][0] / bought[m][1]}
        for k in METRICS:
            mean, lo, hi = bootstrap_ci(per[m][k], N_BOOT, BOOT_SEED)
            r[k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
            r[k + "_cold"], r[k + "_warm"] = float(per[m][k][cold].mean()), float(per[m][k][~cold].mean())
        r["recall_all12_cold"], r["recall_all12_warm"] = r["recall@12_cold"], r["recall@12_warm"]
        table[m] = r
    full = "reranker (full)"
    pairs = [(full, "combination_step5"), (full, "repurchase+last_4w"), (full, "two_tower_alone"),
             (full, "reranker without two-tower features"), (full, "reranker without content features")]
    paired = {f"{a} - {b}": {k: paired_bootstrap(per[a][k], per[b][k], N_BOOT, BOOT_SEED) for k in METRICS} for a, b in pairs}
    pr = np.array(pool_recall)
    ceiling = {"pool_recall_mean": float(pr.mean()), "mean_pool_size": float(np.mean(pool_size)),
               "pool_recall_cold": float(pr[cold].mean()), "pool_recall_warm": float(pr[~cold].mean()),
               "reranked_recall@12": table[full]["recall@12"]["mean"],
               "share_of_pool_recall_captured_in_top12": table[full]["recall@12"]["mean"] / float(pr.mean())}

    out = {"split": split.to_json(), "population": {"n_evaluated": n_eval, "cold_start_share": float(cold.mean())},
           "training": {"cutoff": str(np.datetime64("1970-01-01") + np.timedelta64(train_cut, "D")), "customers_sampled": int(n_tr),
                        "fit_customers": int(len(fit_c)), "inner_validation_customers": int(len(inner_c)), "fit_rows": train_rows,
                        "sampling": "customers drawn at random (seed 0) from the buyers of the training label week; all pool candidates kept (no negative subsampling)",
                        "n_features": len(FEATURES), "features": FEATURES},
           "chosen_params": best_p, "best_iteration": int(best_model.best_iteration), "inner_map@12": best_m,
           "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED}, "results": table, "paired": paired,
           "feature_importance_gain_share": importance, "pool_ceiling": ceiling,
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, "runtime_s": round(time.time() - T0, 1)}
    if args.smoke or args.val_limit:
        say(json.dumps({k: out[k] for k in ("chosen_params", "pool_ceiling", "peak_rss_mb")}, indent=1))
        say({m: round(table[m]["map@12"]["mean"], 4) for m in methods})
        return
    (RES / "reranker_validation.json").write_text(json.dumps(out, indent=2) + "\n")

    f4 = lambda d: f"{d['mean']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]"
    M = ["# Candidate pool + LightGBM reranker on the validation week", "",
         f"Evaluated population: all {n_eval:,} validation-week buyers ({cold.mean():.2%} cold-start). 95% CIs: customer bootstrap, {N_BOOT} resamples, seed {BOOT_SEED}. "
         f"Trained on cutoff {out['training']['cutoff']}: {len(fit_c):,} fit customers ({train_rows:,} candidate rows) + {len(inner_c):,} inner-validation customers, sampled at random (seed 0) "
         f"from the {len(pool_customers):,} buyers of that label week; every pool candidate kept (no negative subsampling). "
         f"LightGBM lambdarank, params {best_p}, best iteration {out['best_iteration']}, {len(FEATURES)} features, tuned on the inner customers of the training week only.", "",
         "| method | MAP@12 [95% CI] | Recall@12 [95% CI] | NDCG@12 [95% CI] | coverage@12 | already-bought share | MAP cold | MAP warm | Recall cold | Recall warm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in table.items():
        M.append(f"| {n} | {f4(r['map@12'])} | {f4(r['recall@12'])} | {f4(r['ndcg@12'])} | {r['coverage@12']:.4f} | {r['already_bought_share']:.3f} | "
                 f"{r['map@12_cold']:.4f} | {r['map@12_warm']:.4f} | {r['recall@12_cold']:.4f} | {r['recall@12_warm']:.4f} |")
    M += ["", "## Paired differences (A - B)", "", "| A - B | metric | diff | 95% CI | excludes 0 |", "|---|---|---|---|---|"]
    for ab, d in paired.items():
        for k in METRICS:
            x = d[k]
            M.append(f"| {ab} | {k} | {x['diff']:+.4f} | [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}] | {'yes' if x['excludes_zero'] else 'no'} |")
    M += ["", "## Pool ceiling vs achieved", "",
          f"Mean pool size {ceiling['mean_pool_size']:.1f} candidates per customer. Pool recall (share of relevant items anywhere in the pool): "
          f"{ceiling['pool_recall_mean']:.4f} (cold {ceiling['pool_recall_cold']:.4f}, warm {ceiling['pool_recall_warm']:.4f}). "
          f"Reranked top-12 recall: {ceiling['reranked_recall@12']:.4f}, i.e. {ceiling['share_of_pool_recall_captured_in_top12']:.1%} of the ceiling.", "",
          "## Feature importance (share of total gain, full model)", "", "| feature | share |", "|---|---|"]
    M += [f"| {f} | {v:.4f} |" for f, v in importance.items()]
    M.append("")
    (RES / "reranker_validation.md").write_text("\n".join(M))
    say("done; peak RSS", out["peak_rss_mb"], "MB")


if __name__ == "__main__":
    main()
