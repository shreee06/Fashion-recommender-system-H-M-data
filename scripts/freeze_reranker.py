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
import gc
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fashion_recsys.evaluation import build_eval_set, per_customer_metrics  # noqa: E402
from fashion_recsys.reranker import FEATURES, labels_for, top_k_per_customer  # noqa: E402
from run_item_cf import LOG, METRICS  # noqa: E402
from run_reranker import CHUNK, PROC, RES, Ctx, build_matrix, load_D, say  # noqa: E402

MIN_AVAILABLE_KB = 1_000_000  # abort below ~1 GB MemAvailable instead of letting the machine swap
MEM_LOG = PROC / "freeze_memory.log"  # untracked (data/ is git-ignored)


def _kb(path, key):
    for line in open(path):
        if line.startswith(key):
            return int(line.split()[1])
    return -1


def mem_line(stage):
    return (f"{datetime.now():%H:%M:%S} {stage:<34} RSS {_kb('/proc/self/status', 'VmRSS:') / 1024:7.0f} MB  "
            f"peak {_kb('/proc/self/status', 'VmHWM:') / 1024:7.0f} MB  MemAvailable {_kb('/proc/meminfo', 'MemAvailable:') / 1024:7.0f} MB")


def stage(name):
    gc.collect()
    line = mem_line(name)
    with open(MEM_LOG, "a") as f:
        f.write(line + "\n")
    say(line)
    if _kb("/proc/meminfo", "MemAvailable:") < MIN_AVAILABLE_KB:
        abort(f"MemAvailable below 1 GB at stage '{name}'")


def abort(msg):
    with open(MEM_LOG, "a") as f:
        f.write(f"ABORT: {msg}\n")
    print(f"ABORT (no files written): {msg}", flush=True)
    os._exit(3)


def watchdog():
    """Poll MemAvailable every 2 s (also while LightGBM trains) and exit before the machine swaps."""
    while True:
        if _kb("/proc/meminfo", "MemAvailable:") < MIN_AVAILABLE_KB:
            abort("MemAvailable dropped below 1 GB: " + mem_line("watchdog"))
        time.sleep(2)


def build_matrix_lean(ctx, customers, truths, label):
    """Same rows as run_reranker.build_matrix, written into one preallocated float32 array
    (no list-of-chunks + concatenate copy)."""
    X = y = None
    groups, n = [], 0
    for s0 in range(0, len(customers), CHUNK):
        chunk = customers[s0 : s0 + CHUNK]
        pcust, part, Xc, _ = ctx.pool_features(chunk, ctx.sources(chunk))
        yc = labels_for(pcust, part, truths[s0 : s0 + CHUNK], ctx.D["n_art"])
        if X is None:  # capacity guess from the first chunk, grown if ever exceeded
            cap = int(len(Xc) / len(chunk) * len(customers) * 1.15) + 1
            X, y = np.empty((cap, Xc.shape[1]), dtype=np.float32), np.empty(cap, dtype=np.float32)
        if n + len(Xc) > len(X):
            cap = int((n + len(Xc)) * 1.3)
            X2, y2 = np.empty((cap, X.shape[1]), dtype=np.float32), np.empty(cap, dtype=np.float32)
            X2[:n], y2[:n] = X[:n], y[:n]
            X, y = X2, y2
        X[n : n + len(Xc)], y[n : n + len(Xc)] = Xc, yc
        n += len(Xc)
        groups.append(np.bincount(pcust, minlength=len(chunk)))
        del Xc, yc, pcust, part
        stage(f"{label}: chunk {s0 // CHUNK + 1}")
    return X[:n], y[:n], np.concatenate(groups)


def make_datasets(Xtr, ytr, gtr, Xva, yva, gva, params):
    """LightGBM Datasets built straight from the float32 arrays, with the SAME params that
    reranker.fit_ranker passes to lgb.train (data_random_seed etc. decide the bin sampling)."""
    import lightgbm as lgb

    from fashion_recsys.reranker import CATEGORICAL, LGB_BASE

    full = {**LGB_BASE, **params}
    dtr = lgb.Dataset(Xtr, ytr, group=gtr, feature_name=FEATURES, categorical_feature=[f for f in CATEGORICAL if f in FEATURES],
                      params=full, free_raw_data=True)
    dtr.construct()
    dva = lgb.Dataset(Xva, yva, group=gva, reference=dtr, params=full, free_raw_data=True)  # constructed inside lgb.train
    return dtr, dva, full


def train_lean(dtr, dva, full, rounds=400, early=30):
    import lightgbm as lgb

    return lgb.train(full, dtr, num_boost_round=rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(early, verbose=False)])


PARAMS = dict(num_leaves=31, learning_rate=0.1, min_data_in_leaf=400)  # chosen in Step 7 on inner customers
EXPECTED_ITER, EXPECTED_INNER_MAP = 45, 0.037150  # from evals/results/tuning_log.csv (Step 7)


def main():
    MEM_LOG.write_text("")
    threading.Thread(target=watchdog, daemon=True).start()
    stage("start")
    D, split, val_start, train_cut = load_D()
    day, cust, art = D["day"], D["cust"], D["art"]
    stage("data loaded")
    ctx = Ctx(D, train_cut, PROC / "two_tower_candidates_heldout.npz")
    D["price"] = D["chan"] = None  # only needed by cutoff_stats, which Ctx has just run
    stage("training-cutoff sources ready")
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
    Xi, yi, gi = build_matrix_lean(ctx, inner_c, tr_i, "inner")
    Xf, yf, gf = build_matrix_lean(ctx, fit_c, tr_f, "fit")
    say(f"fit rows {Xf.shape[0]:,}; inner rows {Xi.shape[0]:,}")
    if Xf.shape[0] != 7_728_754:
        abort(f"training matrix has {Xf.shape[0]:,} rows, Step 7 had 7,728,754: not the same data")
    dtr, dva, full = make_datasets(Xf, yf, gf, Xi, yi, gi, PARAMS)
    del Xf, yf  # the Dataset holds the binned copy; the 1 GB float32 matrix is no longer needed
    stage("lightgbm training dataset constructed, raw matrix freed")
    booster = train_lean(dtr, dva, full, 400)
    del Xi, yi, dtr, dva
    stage("model trained, datasets freed")
    say("best iteration", booster.best_iteration)
    if booster.best_iteration != EXPECTED_ITER:
        abort(f"best iteration {booster.best_iteration} != Step 7's {EXPECTED_ITER}")

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
    stage("inner customers scored")
    if abs(inner_map - EXPECTED_INNER_MAP) > 3e-4:
        abort(f"inner MAP@12 {inner_map:.6f} differs from Step 7's {EXPECTED_INNER_MAP} by {inner_map - EXPECTED_INNER_MAP:+.6f} (> 0.0003)")

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
    stage("model saved")
    say("saved models/lgbm_reranker.txt")


if __name__ == "__main__":
    main()
