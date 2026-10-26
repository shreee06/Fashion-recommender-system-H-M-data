"""Part A of the final evaluation: freeze the reranker and pre-declare the cold-start rule.

Uses ONLY data before 2020-09-02 for training and the 6,000 inner tuning customers of the
training week (labels 2020-09-02..08) for the cold-start decision. Neither the validation
nor the test week is read.

1. Rebuilds the exact training matrices of scripts/run_reranker.py (same seed, same
   customers) and refits the LightGBM ranker with the already chosen hyperparameters. The
   original run did not save the model, so this is a deterministic reconstruction; it is
   verified against the logged inner-customer MAP@12 and best iteration before it is saved.
2. Decides the cold-start fallback rule on the inner customers only.
"""
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fashion_recsys.evaluation import build_eval_set, per_customer_metrics  # noqa: E402
from fashion_recsys.reranker import FEATURES, fit_ranker, labels_for, top_k_per_customer  # noqa: E402
from run_item_cf import LOG, METRICS  # noqa: E402
from run_reranker import CHUNK, PROC, RES, Ctx, build_matrix, load_D, say  # noqa: E402

PARAMS = dict(num_leaves=31, learning_rate=0.1, min_data_in_leaf=400)  # chosen in Step 7 on inner customers
EXPECTED_ITER, EXPECTED_INNER_MAP = 45, 0.037150  # from evals/results/tuning_log.csv (Step 7)


def main():
    D, split, val_start, train_cut = load_D()
    day, cust, art = D["day"], D["cust"], D["art"]
    ctx = Ctx(D, train_cut, PROC / "two_tower_candidates_heldout.npz")
    # identical sampling to scripts/run_reranker.py
    rng = np.random.default_rng(0)
    pool_customers = ctx.tt_customers.astype(np.int64)
    n_tr = min(30_000, len(pool_customers))
    sampled = np.sort(rng.choice(pool_customers, n_tr, replace=False))
    perm = rng.permutation(n_tr)
    n_fit = int(0.8 * n_tr)
    fit_c, inner_c = np.sort(sampled[perm[:n_fit]]), np.sort(sampled[perm[n_fit:]])
    cu, tr_f, _ = build_eval_set(day, cust, art, fit_c, train_cut, train_cut + 6)
    assert np.array_equal(cu, fit_c)
    cu, tr_i, hist_i = build_eval_set(day, cust, art, inner_c, train_cut, train_cut + 6)
    assert np.array_equal(cu, inner_c)
    Xf, yf, gf = build_matrix(ctx, fit_c, tr_f)
    Xi, yi, gi = build_matrix(ctx, inner_c, tr_i)
    say(f"fit rows {Xf.shape[0]:,}; inner rows {Xi.shape[0]:,}")
    assert Xf.shape[0] == 7_728_754, "training matrix differs from the Step 7 run"
    booster = fit_ranker(Xf, yf, gf, Xi, yi, gi, FEATURES, PARAMS, 400)
    say("best iteration", booster.best_iteration)
    assert booster.best_iteration == EXPECTED_ITER
    del Xf, yf, Xi, yi

    # inner-customer evaluation: reranker vs the cold-start fallback (age-bucket popularity, Step 5 cold list)
    lists_rr, lists_fb = [], []
    for s in range(0, len(inner_c), CHUNK):
        chunk = inner_c[s : s + CHUNK]
        src = ctx.sources(chunk)
        pcust, part, X, _ = ctx.pool_features(chunk, src)
        sc = booster.predict(X, num_iteration=booster.best_iteration)
        lists_rr += top_k_per_customer(pcust, part, sc, len(chunk), 12)
        lists_fb += [l[:12] for l in src["agepop"]]
    per_rr = per_customer_metrics(lists_rr, tr_i)
    per_fb = per_customer_metrics(lists_fb, tr_i)
    inner_map = float(per_rr["map@12"].mean())
    say(f"reproduction check: inner MAP@12 {inner_map:.6f} (Step 7 log {EXPECTED_INNER_MAP})")
    assert abs(inner_map - EXPECTED_INNER_MAP) < 2e-6, "reconstruction does not match the Step 7 model"

    cold = hist_i == 0
    res = {"n_inner_customers": int(len(inner_c)), "n_cold": int(cold.sum()), "cold_share": float(cold.mean())}
    for name, per in (("reranker", per_rr), ("fallback_age_bucket_popularity", per_fb)):
        res[name] = {"cold": {k: float(per[k][cold].mean()) for k in METRICS},
                     "warm": {k: float(per[k][~cold].mean()) for k in METRICS},
                     "all": {k: float(per[k].mean()) for k in METRICS}}
    use_fallback = res["fallback_age_bucket_popularity"]["cold"]["map@12"] > res["reranker"]["cold"]["map@12"]
    res["rule"] = ("customers with no purchase before the cutoff get the age-bucket popularity top-12 (Step 5 cold list) "
                   "instead of the reranker" if use_fallback else "fallback does not improve cold MAP@12 on the inner customers; no fallback")
    res["decision_criterion"] = "fallback adopted iff its cold-customer MAP@12 on the inner tuning customers exceeds the reranker's (point estimate)"
    res["fallback_adopted"] = bool(use_fallback)
    res["reconstruction"] = {"best_iteration": int(booster.best_iteration), "inner_map@12": inner_map, "step7_logged_inner_map@12": EXPECTED_INNER_MAP}
    (RES / "cold_fallback_decision.json").write_text(json.dumps(res, indent=2) + "\n")
    say(json.dumps(res, indent=1))

    run = "freeze-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    pop = f"inner tuning customers of the training week (cutoff 2020-09-02), {len(inner_c)} customers, cold only n={int(cold.sum())}"
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        for name in ("reranker", "fallback_age_bucket_popularity"):
            w.writerow([datetime.now().isoformat(timespec="seconds"), run, "cold_start_rule", "system", name, pop, int(cold.sum())]
                       + [f"{res[name]['cold'][k]:.6f}" for k in METRICS])
        w.writerow([datetime.now().isoformat(timespec="seconds"), run, "cold_start_rule", "decision", "fallback_adopted=" + str(use_fallback),
                    pop, int(cold.sum())] + ["nan"] * 3)

    (ROOT / "models").mkdir(exist_ok=True)
    booster.save_model(str(ROOT / "models" / "lgbm_reranker.txt"), num_iteration=booster.best_iteration)
    (ROOT / "models" / "lgbm_reranker_meta.json").write_text(json.dumps(
        {"params": PARAMS, "best_iteration": int(booster.best_iteration), "features": FEATURES, "train_cutoff_day": int(train_cut)}, indent=1))
    say("saved models/lgbm_reranker.txt")


if __name__ == "__main__":
    main()
