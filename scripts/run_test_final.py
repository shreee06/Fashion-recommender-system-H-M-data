"""Final evaluation of the FROZEN systems on the TEST week (2020-09-16 .. 2020-09-22).

* Test mode needs --i-am-done-tuning and runs at most once: it writes
  evals/results/test_run_record.json before touching the test labels and refuses if that file exists.
* Nothing is trained or tuned here. Frozen assets: models/two_tower_best.pt, models/lgbm_reranker.txt,
  the Step 5 candidate configuration (evals/results/candidates_validation.json) and the cold-start
  rule (evals/results/cold_fallback_decision.json).
* Candidates and features use only transactions before the cutoff; the test week is read only to
  build the relevance labels.
* `--dry-run-on-validation` runs the identical pipeline with cutoff 2020-09-09 and labels from the validation
  week. It drops every transaction on or after 2020-09-16 right after loading, never writes
  test_run_record.json, and writes evals/results/validation_reference.json. It is also checked against the
  validation numbers of the earlier steps (stops with an error if any differ by more than 0.0003).
"""
import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fashion_recsys import baselines as bl  # noqa: E402
from fashion_recsys.config import to_day  # noqa: E402
from fashion_recsys.memguard import MemGuard  # noqa: E402
from fashion_recsys.evaluation import bootstrap_ci, build_eval_set, paired_bootstrap, per_customer_metrics  # noqa: E402
from fashion_recsys.reranker import labels_for, top_k_per_customer  # noqa: E402
from fashion_recsys.two_tower import TwoTower, build_examples, retrieve  # noqa: E402
from run_candidates import combine  # noqa: E402
from run_item_cf import METRICS, BOOT_SEED, N_BOOT  # noqa: E402
from run_reranker import CHUNK, PROC, RES, Ctx, load_D, say  # noqa: E402

RECORD = RES / "test_run_record.json"
PROTOCOL = RES / "test_protocol.md"
SYSTEMS = ["popular_last_1w", "repurchase+last_4w", "combination_step5", "two_tower_alone", "reranker"]
# validation MAP@12 of the earlier steps (dry-run reproduction check), tolerance 0.0003
EXPECTED_VAL_MAP = {"repurchase+last_4w": 0.024223, "combination_step5": 0.026091, "two_tower_alone": 0.010597, "reranker": 0.033653}
TOL = 3e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def config_hash():
    cfg = json.loads((RES / "candidates_validation.json").read_text())
    parts = [sha(PROTOCOL) if PROTOCOL.exists() else "no-protocol",
             sha(ROOT / "models" / "two_tower_best.pt"), sha(ROOT / "models" / "lgbm_reranker.txt"),
             sha(ROOT / "models" / "lgbm_reranker_meta.json"), sha(RES / "cold_fallback_decision.json"),
             json.dumps({k: cfg[k] for k in ("item_cf_new", "variants", "segment_popularity", "combination")}, sort_keys=True)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def load_frozen_two_tower(D):
    ck = torch.load(ROOT / "models" / "two_tower_best.pt", map_location="cpu")
    c = dict(kv.split("=") for kv in ck["cfg"].split(";"))
    arts = pq.read_table(PROC / "articles.parquet", columns=["product_type_no", "colour_group_code", "department_no", "index_group_no"]).to_pandas()
    content = {k: arts[col].astype("category").cat.codes.to_numpy().astype(np.int64)
               for k, col in (("product_type", "product_type_no"), ("colour", "colour_group_code"),
                              ("department", "department_no"), ("index_group", "index_group_no"))}
    model = TwoTower(D["n_art"], content, d=int(c["d"]), share_embeddings=bool(int(c["shared"])), use_content=bool(int(c["content"])))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, dict(K=int(c["K"]), tau=float(c["tau_in"]), lookback=int(c["lookback"]))


def assert_no_test_data(D, split):
    """Dry-run guard: no transaction on or after the first test day may be in memory."""
    assert int(D["day"].max()) < to_day(split.test_start), "dry run must not hold any test-week transaction"


def evaluate_week(week, guard):
    import lightgbm as lgb

    torch.set_num_threads(4)
    guard.stage("start")
    # dry run: every transaction on or after the first test day is dropped at load time, then proven absent
    D, split, val_start, _ = load_D(truncate_before="test_start" if week == "validation" else None)
    if week == "validation":
        assert_no_test_data(D, split)
    guard.stage("data loaded")
    day, cust, art, n_art, n_cust = D["day"], D["cust"], D["art"], D["n_art"], D["n_cust"]
    cutoff, end = (to_day(split.test_start), to_day(split.test_end)) if week == "test" else (val_start, to_day(split.val_end))
    assert end == cutoff + 6
    customers, truths, hist = build_eval_set(day, cust, art, np.arange(n_cust), cutoff, end)
    n_eval, cold = len(customers), hist == 0
    say(f"{week}: cutoff {split.test_start if week == 'test' else split.val_start}, {n_eval:,} customers, cold {cold.mean():.2%}")

    # frozen two-tower: encode customers from data before the cutoff and retrieve the top 100
    tt_model, tcfg = load_frozen_two_tower(D)
    m = (day >= cutoff - 182) & (day < cutoff)
    catalog = np.flatnonzero(np.bincount(art[m], minlength=n_art) > 0)
    ex = build_examples(day, cust, art, cutoff, n_art, tcfg["K"], tcfg["tau"], tcfg["lookback"], customers=customers, with_targets=False)
    assert np.array_equal(ex.customers, customers) and ex.max_input_day < cutoff
    tt_codes, tt_scores = retrieve(tt_model, ex, D["bucket"], catalog, top=100)
    if week == "validation":  # identical to the file produced in Step 6?
        z = np.load(PROC / "two_tower_candidates_validation.npz")
        same = np.array_equal(z["codes"], tt_codes)
        say(f"two-tower candidates identical to the Step 6 file: {same}")
        assert same
    ctx = Ctx(D, cutoff, tt=(customers, tt_codes, tt_scores))
    pop1 = bl.recent_popular(day, art, cutoff, 1, n_art, 100).tolist()
    booster = lgb.Booster(model_file=str(ROOT / "models" / "lgbm_reranker.txt"))
    meta = json.loads((ROOT / "models" / "lgbm_reranker_meta.json").read_text())
    assert booster.feature_name() == meta["features"]
    assert json.loads((RES / "cold_fallback_decision.json").read_text())["fallback_adopted"] is False, "frozen rule: no cold-start fallback"
    cj = D["cfg"]["combination"]
    guard.stage("sources, frozen models and two-tower candidates ready")

    per = {s: {k: [] for k in METRICS} for s in SYSTEMS}
    lists_all = {s: [] for s in SYSTEMS}
    bought = {s: [0, 0] for s in SYSTEMS}
    for s0 in range(0, n_eval, CHUNK):
        chunk, tr, cold_c = customers[s0 : s0 + CHUNK], truths[s0 : s0 + CHUNK], cold[s0 : s0 + CHUNK]
        src = ctx.sources(chunk)
        pcust, part, X, _ = ctx.pool_features(chunk, src)
        primary = top_k_per_customer(pcust, part, booster.predict(X, num_iteration=meta["best_iteration"]), len(chunk), 12)
        comb = combine(src["recent"], {"cf": src["cf"], "var": src["var"], "segage": src["agepop"]}, cj["order"], cj["caps"], cj["n_repurchase"], ctx.pop4)
        lists = {"popular_last_1w": [pop1[:12]] * len(chunk), "repurchase+last_4w": bl.fill(src["recent"], ctx.pop4, 12),
                 "combination_step5": comb, "two_tower_alone": src["tt"][:, :12].tolist(),
                 "reranker": primary}
        n_pre = ctx.stats.n_pre
        row_of = np.full(n_cust, -1, dtype=np.int64)
        row_of[chunk] = np.arange(len(chunk))
        rr = row_of[cust[:n_pre]]
        mk = rr >= 0
        hk = np.unique(rr[mk] * n_art + art[:n_pre][mk].astype(np.int64))
        for sname, ls in lists.items():
            assert all(len(l) == 12 and len(set(l)) == 12 for l in ls), sname
            q = np.repeat(np.arange(len(ls)), 12) * n_art + np.fromiter((x for l in ls for x in l), dtype=np.int64)
            pos = np.clip(np.searchsorted(hk, q), 0, len(hk) - 1)
            bought[sname][0] += int((hk[pos] == q).sum())
            bought[sname][1] += len(q)
            p = per_customer_metrics(ls, tr)
            for k in METRICS:
                per[sname][k].append(p[k])
            lists_all[sname] += ls
        guard.stage(f"chunk {s0 // CHUNK + 1}/{-(-n_eval // CHUNK)}")
    per = {s: {k: np.concatenate(v) for k, v in d.items()} for s, d in per.items()}

    table = {}
    for sname in SYSTEMS:
        r = {"coverage@12": len({x for l in lists_all[sname] for x in l}) / n_art, "already_bought_share": bought[sname][0] / bought[sname][1]}
        for k in METRICS:
            mean, lo, hi = bootstrap_ci(per[sname][k], N_BOOT, BOOT_SEED)
            r[k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
            r[k + "_cold"], r[k + "_warm"] = float(per[sname][k][cold].mean()), float(per[sname][k][~cold].mean())
        table[sname] = r
    pairs = [("reranker", "combination_step5"), ("reranker", "repurchase+last_4w")]
    paired = {f"{a} - {b}": {k: paired_bootstrap(per[a][k], per[b][k], N_BOOT, BOOT_SEED) for k in METRICS} for a, b in pairs}
    return {"week": week, "cutoff_day": int(cutoff), "n_evaluated": n_eval, "n_cold": int(cold.sum()), "cold_start_share": float(cold.mean()),
            "cold_fallback_applied": False, "catalogue_size": int(len(catalog)),
            "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED}, "results": table, "paired": paired}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i-am-done-tuning", action="store_true", help="required for the test week: confirms tuning is finished")
    ap.add_argument("--dry-run-on-validation", action="store_true",
                    help="run the identical pipeline on the validation week (test-week rows are dropped at load time)")
    args = ap.parse_args(argv)
    t0 = time.time()

    if args.dry_run_on_validation:
        guard = MemGuard(PROC / "dryrun_memory.log")
        guard.start_watchdog()
        out = evaluate_week("validation", guard)
        check = {}
        for name, want in EXPECTED_VAL_MAP.items():
            got = out["results"][name]["map@12"]["mean"]
            check[name] = {"dry_run": got, "expected": want, "diff": got - want}
            if abs(got - want) > TOL:
                print(f"STOP: {name} validation MAP@12 {got:.6f} differs from {want:.6f} by {got - want:+.6f} (> {TOL})", flush=True)
                sys.exit(1)
        out["reproduction_check"] = check
        out["note"] = "dry run of the final-evaluation code path on the VALIDATION week with the frozen models; no test-week row is loaded"
        out["peak_rss_mb"], out["runtime_s"] = guard.peak_mb(), round(time.time() - t0, 1)
        (RES / "validation_reference.json").write_text(json.dumps(out, indent=2) + "\n")
        guard.stage("dry run complete")
        say("dry run reproduces the earlier validation numbers:", {k: f"{v['dry_run']:.6f} (diff {v['diff']:+.6f})" for k, v in check.items()})
        return

    if not args.i_am_done_tuning:
        sys.exit("Refusing to evaluate on the test week: pass --i-am-done-tuning once tuning is finished.")
    if RECORD.exists():
        sys.exit(f"Refusing to run: {RECORD.name} exists, so the test week has already been evaluated (or a run was started).")
    if git("status", "--porcelain"):
        sys.exit("Refusing to run: the working tree has uncommitted changes; commit the frozen code and protocol first.")
    record = {"status": "started", "commit": git("rev-parse", "HEAD"), "config_hash": config_hash(),
              "timestamp": datetime.now().isoformat(timespec="seconds"), "protocol_sha256": sha(PROTOCOL),
              "two_tower_sha256": sha(ROOT / "models" / "two_tower_best.pt"), "lgbm_sha256": sha(ROOT / "models" / "lgbm_reranker.txt")}
    RECORD.write_text(json.dumps(record, indent=2) + "\n")  # written BEFORE the test labels are read
    guard = MemGuard(PROC / "test_run_memory.log")
    guard.start_watchdog()
    out = evaluate_week("test", guard)
    out["run_record"] = record
    out["peak_rss_mb"] = guard.peak_mb()
    out["runtime_s"] = round(time.time() - t0, 1)
    (RES / "test_final.json").write_text(json.dumps(out, indent=2) + "\n")
    record.update(status="completed", result_sha256=sha(RES / "test_final.json"))
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    write_markdown(out)
    say("test evaluation complete")


def write_markdown(out):
    ref = json.loads((RES / "validation_reference.json").read_text())
    f4 = lambda d: f"{d['mean']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]"
    n, cold = out["n_evaluated"], out["cold_start_share"]
    M = ["# Final test-week evaluation (2020-09-16 to 2020-09-22)", "",
         f"Population: all {n:,} customers who bought in the test week; {out['n_cold']:,} ({cold:.2%}) had no purchase before the cutoff 2020-09-16 (cold-start). "
         f"95% CIs: customer bootstrap, {N_BOOT} resamples, seed {BOOT_SEED}. Frozen systems only (see test_protocol.md); run record in test_run_record.json.", "",
         "| system | MAP@12 [95% CI] | Recall@12 [95% CI] | NDCG@12 [95% CI] | coverage@12 | already-bought share | MAP cold | MAP warm |", "|---|---|---|---|---|---|---|---|"]
    for s, r in out["results"].items():
        M.append(f"| {s} | {f4(r['map@12'])} | {f4(r['recall@12'])} | {f4(r['ndcg@12'])} | {r['coverage@12']:.4f} | {r['already_bought_share']:.3f} | {r['map@12_cold']:.4f} | {r['map@12_warm']:.4f} |")
    M += ["", "## Paired differences (A - B)", "", "| A - B | metric | diff | 95% CI | excludes 0 |", "|---|---|---|---|---|"]
    for ab, d in out["paired"].items():
        for k in METRICS:
            x = d[k]
            M.append(f"| {ab} | {k} | {x['diff']:+.4f} | [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}] | {'yes' if x['excludes_zero'] else 'no'} |")
    M += ["", f"## Validation vs test (validation: {ref['n_evaluated']:,} customers, {ref['cold_start_share']:.2%} cold; same frozen code and models)", "",
          "| system | metric | validation | test | change | test CI overlaps validation value? |", "|---|---|---|---|---|---|"]
    for s in SYSTEMS:
        for k in METRICS:
            v, t = ref["results"][s][k]["mean"], out["results"][s][k]
            ch = t["mean"] - v
            inside = t["ci_lo"] <= v <= t["ci_hi"]
            M.append(f"| {s} | {k} | {v:.4f} | {t['mean']:.4f} | {ch:+.4f} ({ch / v:+.1%}){' DROP' if ch < 0 else ''} | {'yes' if inside else 'no'} |")
    M.append("")
    (RES / "test_final.md").write_text("\n".join(M))


if __name__ == "__main__":
    main()
