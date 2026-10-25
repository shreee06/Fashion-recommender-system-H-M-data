"""Step 6: two-tower retrieval. Trains on weekly cutoffs strictly before the validation
start, early-stops / tunes on a held-out TRAINING week, then evaluates the VALIDATION week.
The test week is never touched."""
import argparse
import copy
import json
import resource
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fashion_recsys import baselines as bl  # noqa: E402
from fashion_recsys.config import make_split, to_day  # noqa: E402
from fashion_recsys.data import load_transaction_codes  # noqa: E402
from fashion_recsys.evaluation import (  # noqa: E402
    bootstrap_ci, build_eval_set, candidate_recall, paired_bootstrap, per_customer_metrics,
)
from fashion_recsys.item_cf import ItemCF  # noqa: E402
from fashion_recsys.segments import age_bucket, customer_segment_popular, dominant_group, group_popular  # noqa: E402
from fashion_recsys.two_tower import TwoTower, build_examples, concat_examples, retrieve, train_epoch  # noqa: E402
from fashion_recsys.variants import VariantCandidates  # noqa: E402
from run_candidates import combine, merge_unique  # noqa: E402
from run_item_cf import LOG, METRICS, BOOT_SEED, N_BOOT  # noqa: E402

RES = ROOT / "evals" / "results"
PROC = ROOT / "data" / "processed"
SEED, T0 = 0, time.time()
torch.set_num_threads(4)


def say(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


@dataclass(frozen=True)
class Cfg:
    d: int = 64
    K: int = 20
    tau_in: float = 8.0
    temp: float = 0.07
    lr: float = 2e-3
    bs: int = 2048
    logq: bool = True
    shared: bool = True
    content: bool = True
    train_weeks: int = 12
    lookback: int = 26

    def short(self):
        return ";".join(f"{k}={int(v) if isinstance(v, bool) else v}" for k, v in self.__dict__.items())


def subset(ex, idx):
    """Examples restricted to rows `idx` (sorted), targets re-packed."""
    from fashion_recsys.two_tower import Examples
    lens = np.diff(ex.tgt_start)[idx]
    flat = np.concatenate([ex.tgt_items[ex.tgt_start[i] : ex.tgt_start[i + 1]] for i in idx]) if len(idx) else np.zeros(0, np.int64)
    return Examples(ex.customers[idx], ex.cutoff[idx], ex.items[idx], ex.weights[idx], ex.feats[idx],
                    np.r_[0, np.cumsum(lens)].astype(np.int64), flat, ex.max_input_day, ex.min_target_day, ex.max_target_day)


def truths_of(ex):
    return [set(ex.tgt_items[ex.tgt_start[i] : ex.tgt_start[i + 1]].tolist()) for i in range(len(ex.customers))]


def log_trial(run, method, param, value, population, n, m):
    with open(LOG, "a", newline="") as f:
        import csv
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), run, method, param, value, population, n]
                               + [f"{m[k]:.6f}" for k in METRICS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, no tuning, no outputs (timing check)")
    ap.add_argument("--skip-tuning", action="store_true")
    args = ap.parse_args()
    run = "2tower-" + datetime.now().strftime("%Y%m%d-%H%M%S")

    day, cust, art, customer_ids, article_ids = load_transaction_codes(PROC)
    n_art, n_cust = len(article_ids), len(customer_ids)
    split = make_split(day.min(), day.max())
    val_start = to_day(split.val_start)
    arts = pq.read_table(PROC / "articles.parquet", columns=["product_type_no", "colour_group_code", "department_no", "index_group_no", "product_code"]).to_pandas()
    content = {k: arts[c].astype("category").cat.codes.to_numpy().astype(np.int64)
               for k, c in (("product_type", "product_type_no"), ("colour", "colour_group_code"),
                            ("department", "department_no"), ("index_group", "index_group_no"))}
    ages = pq.read_table(PROC / "customers.parquet", columns=["age"])["age"].to_numpy(zero_copy_only=False).astype(float)
    bucket = age_bucket(ages)

    def catalog_at(cutoff, weeks=26):
        m = (day >= cutoff - 7 * weeks) & (day < cutoff)
        return np.flatnonzero(np.bincount(art[m], minlength=n_art) > 0)

    # ---------------- training / held-out examples (all strictly before the validation start)
    held_cut = val_start - 7  # held-out TRAINING week: val_start-7 .. val_start-1
    ex_cache = {}

    def train_examples(cfg):
        key = (cfg.K, cfg.tau_in, cfg.train_weeks, cfg.lookback)
        if key not in ex_cache:
            ex_cache.clear()
            parts = [build_examples(day, cust, art, val_start - 7 * k, n_art, cfg.K, cfg.tau_in, cfg.lookback, limit_day=val_start)
                     for k in range(2, cfg.train_weeks + 2)]
            ex_cache[key] = concat_examples(parts)
        return ex_cache[key]

    def held_examples(cfg):
        ex = build_examples(day, cust, art, held_cut, n_art, cfg.K, cfg.tau_in, cfg.lookback, limit_day=val_start)
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(ex.customers), size=min(8000, len(ex.customers)), replace=False))
        return subset(ex, idx)

    held_catalog = catalog_at(held_cut)

    def held_metrics(model, hex_, truths):
        codes, _ = retrieve(model, hex_, bucket, held_catalog, top=12)
        m = per_customer_metrics(codes.tolist(), truths)
        return {k: float(v.mean()) for k, v in m.items()}

    def run_trial(cfg, label, max_epochs):
        t_all = time.time()
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        ex = train_examples(cfg)
        hex_ = held_examples(cfg)
        truths = truths_of(hex_)
        model = TwoTower(n_art, content, d=cfg.d, share_embeddings=cfg.shared, use_content=cfg.content)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        log_q = None
        if cfg.logq:
            lens = np.diff(ex.tgt_start)
            w = np.repeat(1.0 / lens, lens)
            q = np.bincount(ex.tgt_items, weights=w, minlength=n_art)
            log_q = np.log(q / q.sum() + 1e-12)
        best, best_state, bad, hist, ep_times = -1.0, None, 0, [], []
        for ep in range(max_epochs):
            t = time.time()
            loss = train_epoch(model, opt, ex, bucket, rng, cfg.bs, cfg.temp, log_q)
            ep_times.append(time.time() - t)
            hm = held_metrics(model, hex_, truths)
            hist.append({"epoch": ep + 1, "loss": loss, **hm})
            say(f"  [{label}] epoch {ep + 1}: loss {loss:.4f}  held-out MAP@12 {hm['map@12']:.4f}  ({ep_times[-1]:.0f}s)")
            if hm["map@12"] > best:
                best, best_state, bad = hm["map@12"], copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
        model.load_state_dict(best_state)
        hm = held_metrics(model, hex_, truths)
        info = {"label": label, "cfg": cfg.short(), "n_params": model.n_params(), "epochs_run": len(hist),
                "best_epoch": int(np.argmax([h["map@12"] for h in hist]) + 1), "sec_per_epoch": float(np.mean(ep_times)),
                "train_examples": int(len(ex.customers)), "total_sec": time.time() - t_all, "history": hist, "held_out": hm}
        return model, info

    population_held = f"held-out training week (cutoff {(split.val_start).isoformat()} - 7d), 8000 sampled buyers"

    base = Cfg()
    if args.smoke:
        model, info = run_trial(base, "smoke", 1)
        print(json.dumps({k: v for k, v in info.items() if k != "history"}, indent=2))
        say("peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
        return

    # ---------------- tuning on the held-out training week (greedy, one factor at a time)
    trials = []

    def trial(cfg, label):
        model, info = run_trial(cfg, label, args.max_epochs)
        log_trial(run, "two_tower", "config", cfg.short(), population_held, 8000, info["held_out"])
        trials.append(info)
        say(f"TRIAL {label}: held-out MAP@12 {info['held_out']['map@12']:.4f} ({info['best_epoch']} epochs best)")
        return model, info

    best_cfg, best_map, best_model = base, -1, None

    def consider(cfg, label):
        nonlocal best_cfg, best_map, best_model
        model, info = trial(cfg, label)
        if info["held_out"]["map@12"] > best_map:
            best_cfg, best_map, best_model = cfg, info["held_out"]["map@12"], model

    if args.skip_tuning:
        consider(base, "default")
    else:
        consider(base, "default (temp 0.07)")
        for t in (0.03, 0.15):
            consider(replace(base, temp=t), f"temp {t}")
        consider(replace(best_cfg, logq=not best_cfg.logq), "logq flipped")
        for K in (10, 40):
            consider(replace(best_cfg, K=K), f"K {K}")
        consider(replace(best_cfg, shared=not best_cfg.shared), "shared flipped")
        for lr in (1e-3, 4e-3):
            consider(replace(best_cfg, lr=lr), f"lr {lr}")
    say("chosen config:", best_cfg.short(), f"held-out MAP {best_map:.4f}")

    # ---------------- validation-week population (same as steps 4/5)
    cutoff = val_start
    customers, truths, hist = build_eval_set(day, cust, art, np.arange(n_cust), cutoff, to_day(split.val_end))
    n_eval, cold = len(customers), None
    cold = hist == 0
    val_catalog = catalog_at(cutoff)
    in_cat = np.zeros(n_art, dtype=bool)
    in_cat[val_catalog] = True
    rel_total = sum(len(t) for t in truths)
    rel_in_cat = sum(sum(in_cat[list(t)]) for t in truths)
    say(f"validation: {n_eval:,} customers, catalogue {len(val_catalog):,} articles; "
        f"{rel_in_cat / rel_total:.2%} of relevant (customer, article) pairs are inside the catalogue")

    # exclusion matrix: everything each validation customer bought before the cutoff
    row_of = np.full(n_cust, -1, dtype=np.int64)
    row_of[customers] = np.arange(n_eval)
    hm_ = (day < cutoff) & (row_of[cust] >= 0)
    excl = sp.csr_matrix((np.ones(hm_.sum(), dtype=np.float32), (row_of[cust[hm_]], art[hm_])), shape=(n_eval, n_art))

    def val_example(cfg):
        return build_examples(day, cust, art, cutoff, n_art, cfg.K, cfg.tau_in, cfg.lookback,
                              customers=customers, with_targets=False, limit_day=None)

    def eval_val(model, cfg):
        ex = val_example(cfg)
        assert np.array_equal(ex.customers, customers)
        codes, scores = retrieve(model, ex, bucket, val_catalog, top=100)
        codes_x, scores_x = retrieve(model, ex, bucket, val_catalog, top=100, exclude=excl)
        return codes, scores, codes_x, scores_x

    def summarise(codes):
        lists = codes.tolist()
        per = per_customer_metrics(lists, truths)
        return lists, per, {"recall@50": candidate_recall(lists, truths, 50), "recall@100": candidate_recall(lists, truths, 100)}

    # ---------------- final runs: chosen, logQ flipped, ID-only (all evaluated on validation)
    finals = {}
    for label, cfg in (("two_tower (chosen)", best_cfg),
                       ("two_tower logQ flipped", replace(best_cfg, logq=not best_cfg.logq)),
                       ("two_tower ID-only item tower", replace(best_cfg, content=False))):
        if label == "two_tower (chosen)" and best_model is not None:
            model, info = best_model, next(t for t in trials if t["cfg"] == cfg.short())
        else:
            model, info = trial(cfg, label)
        codes, scores, codes_x, scores_x = eval_val(model, cfg)
        lists, per, rec = summarise(codes)
        lists_x, per_x, rec_x = summarise(codes_x)
        finals[label] = dict(model=model, info=info, cfg=cfg, codes=codes, scores=scores, lists=lists, per=per, rec=rec,
                             lists_x=lists_x, per_x=per_x, rec_x=rec_x)
        log_trial(run, "two_tower_final", "config", cfg.short(), "all validation-week buyers", n_eval,
                  {k: float(per[k].mean()) for k in METRICS})
        say(f"VAL {label}: MAP@12 {per['map@12'].mean():.4f}  recall@100 {rec['recall@100']:.4f}  "
            f"| excluding seen: MAP@12 {per_x['map@12'].mean():.4f}")
    main_f = finals["two_tower (chosen)"]

    # ---------------- save the model (untracked) and candidate scores for the reranker (untracked)
    (ROOT / "models").mkdir(exist_ok=True)
    torch.save({"state_dict": main_f["model"].state_dict(), "cfg": main_f["cfg"].short()}, ROOT / "models" / "two_tower_best.pt")
    np.savez_compressed(PROC / "two_tower_candidates_validation.npz", customers=customers.astype(np.int32), cutoff_day=cutoff,
                        codes=main_f["codes"].astype(np.int32), scores=main_f["scores"].astype(np.float32))
    hall = build_examples(day, cust, art, held_cut, n_art, best_cfg.K, best_cfg.tau_in, best_cfg.lookback, limit_day=val_start)
    hc, hs = retrieve(main_f["model"], hall, bucket, held_catalog, top=100)
    np.savez_compressed(PROC / "two_tower_candidates_heldout.npz", customers=hall.customers.astype(np.int32), cutoff_day=held_cut,
                        codes=hc.astype(np.int32), scores=hs.astype(np.float32))
    say("saved candidate files")

    # ---------------- reproduce Step 5 sources for paired comparisons and the union pool
    cj = json.loads((RES / "candidates_validation.json").read_text())
    ic, vc, sc, cb = cj["item_cf_new"], cj["variants"], cj["segment_popularity"], cj["combination"]
    L = 100
    pop4 = bl.recent_popular(day, art, cutoff, 4, n_art, L).tolist()
    recents = bl.recent_items(day, cust, art, cutoff, customers, L)
    cf = ItemCF(ic["window_weeks"], ic["max_items"]).fit(day, cust, art, cutoff, n_art, recency_tau_weeks=ic["tau_weeks"])
    cf_lists = cf.recommend(day, cust, art, customers, k_seeds=ic["k_seeds"], n=L)
    del cf
    product_code = arts["product_code"].astype("category").cat.codes.to_numpy()
    var_lists = VariantCandidates(vc["pop_weeks"]).fit(day, art, cutoff, product_code, n_art).recommend(recents, vc["k_seeds"], L)
    tops = customer_segment_popular(day, cust, art, bucket, 6, cutoff, sc["age_weeks"], n_art, L)
    age_lists = [tops[b].tolist() for b in bucket[customers]]
    ag = arts["index_group_no"].astype("category").cat.codes.to_numpy().astype(np.int64)
    dom = dominant_group(day, cust, art, ag, int(ag.max()) + 1, cutoff, n_cust)
    gt = {g: t.tolist() for g, t in group_popular(day, art, ag, int(ag.max()) + 1, cutoff, sc["index_group_weeks"], n_art, L).items()}
    grp_lists = [gt[g] if g >= 0 else pop4 for g in dom[customers]]
    srcs = {"cf": cf_lists, "var": var_lists, "segage": age_lists}
    comb = combine(recents, srcs, cb["order"], cb["caps"], cb["n_repurchase"], pop4)
    cf_filled = bl.fill(cf_lists, pop4, L)
    per_cf, per_comb = per_customer_metrics(cf_filled, truths), per_customer_metrics(comb, truths)
    say(f"reproduced Step 5: item-CF MAP {per_cf['map@12'].mean():.4f}, combination MAP {per_comb['map@12'].mean():.4f}")
    assert abs(per_cf["map@12"].mean() - 0.0098) < 6e-5 and abs(per_comb["map@12"].mean() - 0.0261) < 6e-5

    parts = [recents, cf_lists, var_lists, age_lists, grp_lists, [pop4] * n_eval]
    tt = main_f["lists"]
    union5 = [merge_unique([p[i] for p in parts], 10_000) for i in range(n_eval)]
    union6 = [merge_unique([p[i] for p in parts] + [tt[i]], 10_000) for i in range(n_eval)]

    def pool(lists):
        return {"recall@12": candidate_recall(lists, truths, 12), "recall@50": candidate_recall(lists, truths, 50),
                "recall@100": candidate_recall(lists, truths, 100),
                "recall@full_pool": float(np.mean([len(set(u) & t) / len(t) for u, t in zip(lists, truths)])),
                "mean_pool_size": float(np.mean([len(u) for u in lists]))}
    p5, p6 = pool(union5), pool(union6)
    only_tt = float(np.mean([len((set(tt[i]) - set(union5[i])) & truths[i]) / len(truths[i]) for i in range(n_eval)]))
    say(f"union pool: {p5['recall@full_pool']:.4f} @ {p5['mean_pool_size']:.1f} -> {p6['recall@full_pool']:.4f} @ {p6['mean_pool_size']:.1f}")

    # ---------------- results
    def row(per):
        r = {}
        for k in METRICS:
            mean, lo, hi = bootstrap_ci(per[k], N_BOOT, BOOT_SEED)
            r[k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
            r[k + "_cold"], r[k + "_warm"] = float(per[k][cold].mean()), float(per[k][~cold].mean())
        return r

    table = {"item_cf_new (Step 5, filled)": row(per_cf), "combination_step5": row(per_comb)}
    rec_tab = {}
    for label, f in finals.items():
        table[label] = row(f["per"])
        table[label + " [exclude already-bought]"] = row(f["per_x"])
        rec_tab[label] = {**f["rec"], "recall@100_cold": candidate_recall([f["lists"][i] for i in np.flatnonzero(cold)], [truths[i] for i in np.flatnonzero(cold)], 100),
                          "recall@100_warm": candidate_recall([f["lists"][i] for i in np.flatnonzero(~cold)], [truths[i] for i in np.flatnonzero(~cold)], 100)}
        rec_tab[label + " [exclude already-bought]"] = f["rec_x"]
    main_per, main_per_x = main_f["per"], main_f["per_x"]
    pairs = {
        "two_tower - item_cf_new": (main_per, per_cf),
        "two_tower - combination_step5": (main_per, per_comb),
        "two_tower[exclude seen] - item_cf_new": (main_per_x, per_cf),
        "two_tower[exclude seen] - combination_step5": (main_per_x, per_comb),
        "two_tower(ID+content) - two_tower(ID-only)": (main_per, finals["two_tower ID-only item tower"]["per"]),
        "two_tower(logQ as chosen) - two_tower(logQ flipped)": (main_per, finals["two_tower logQ flipped"]["per"]),
    }
    paired = {n: {k: paired_bootstrap(a[k], b[k], N_BOOT, BOOT_SEED) for k in METRICS} for n, (a, b) in pairs.items()}

    out = {
        "split": split.to_json(), "population": {"n_evaluated": n_eval, "cold_start_share": float(cold.mean()),
                                                 "catalogue_size": int(len(val_catalog)),
                                                 "relevant_pairs_in_catalogue": rel_in_cat / rel_total},
        "training": {"cutoffs": [(split.val_start - __import__("datetime").timedelta(days=7 * k)).isoformat() for k in range(2, best_cfg.train_weeks + 2)],
                     "held_out_cutoff": (split.val_start - __import__("datetime").timedelta(days=7)).isoformat(),
                     "note": "targets of every training cutoff end before the validation start; early stopping and tuning use the held-out week"},
        "chosen_cfg": best_cfg.short(),
        "trials": [{k: v for k, v in t.items() if k != "history"} | {"history": t["history"]} for t in trials],
        "final_runs": {l: {k: v for k, v in f["info"].items()} for l, f in finals.items()},
        "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED}, "results": table, "paired": paired,
        "candidate_recall_two_tower": rec_tab,
        "union_pool": {"before (Step 5 sources)": p5, "after (+ two-tower top-100)": p6,
                       "relevant_share_found_only_by_two_tower": only_tt},
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, "runtime_s": round(time.time() - T0, 1),
    }
    (RES / "two_tower_validation.json").write_text(json.dumps(out, indent=2) + "\n")

    f4 = lambda d: f"{d['mean']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]"
    mi = main_f["info"]
    M = ["# Two-tower retrieval on the validation week", "",
         f"Evaluated population: all {n_eval:,} customers who bought in the validation week ({cold.mean():.2%} cold-start). "
         f"Retrieval is over the modelled catalogue of {len(val_catalog):,} articles (sold in the 26 weeks before the cutoff); "
         f"{rel_in_cat / rel_total:.2%} of relevant (customer, article) pairs are inside it. 95% CIs: customer bootstrap, {N_BOOT} resamples, seed {BOOT_SEED}.", "",
         f"Chosen config (tuned on a held-out TRAINING week, not on the validation week): `{best_cfg.short()}`. "
         f"{mi['n_params']:,} parameters; {mi['sec_per_epoch']:.0f} s/epoch on CPU; best epoch {mi['best_epoch']} of {mi['epochs_run']}.", "",
         "| method | MAP@12 [95% CI] | Recall@12 [95% CI] | NDCG@12 [95% CI] | MAP cold | MAP warm |", "|---|---|---|---|---|---|"]
    for n, r in table.items():
        M.append(f"| {n} | {f4(r['map@12'])} | {f4(r['recall@12'])} | {f4(r['ndcg@12'])} | {r['map@12_cold']:.4f} | {r['map@12_warm']:.4f} |")
    M += ["", "## Paired differences (A - B)", "", "| A - B | metric | diff | 95% CI | excludes 0 |", "|---|---|---|---|---|"]
    for ab, d in paired.items():
        for k in METRICS:
            x = d[k]
            M.append(f"| {ab} | {k} | {x['diff']:+.4f} | [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}] | {'yes' if x['excludes_zero'] else 'no'} |")
    M += ["", "## Two-tower candidate recall", "", "| variant | recall@50 | recall@100 | recall@100 cold | recall@100 warm |", "|---|---|---|---|---|"]
    for n, r in rec_tab.items():
        M.append(f"| {n} | {r['recall@50']:.4f} | {r['recall@100']:.4f} | {r.get('recall@100_cold', float('nan')):.4f} | {r.get('recall@100_warm', float('nan')):.4f} |")
    M += ["", "## Union pool (round-robin merge, deduplicated)", "", "| pool | recall@12 | recall@50 | recall@100 | recall (full pool) | mean pool size |", "|---|---|---|---|---|---|"]
    for n, p in (("Step 5 sources", p5), ("+ two-tower top-100", p6)):
        M.append(f"| {n} | {p['recall@12']:.4f} | {p['recall@50']:.4f} | {p['recall@100']:.4f} | {p['recall@full_pool']:.4f} | {p['mean_pool_size']:.1f} |")
    M += ["", f"Share of relevant items found only by the two-tower (not in the Step 5 pool): {only_tt:.4f}.", ""]
    (RES / "two_tower_validation.md").write_text("\n".join(M))
    say("done; peak RSS", out["peak_rss_mb"], "MB")


if __name__ == "__main__":
    main()
