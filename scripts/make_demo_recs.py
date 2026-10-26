"""Precompute recommendations for a small sample of customers for the Streamlit demo.

Runs the FROZEN reranker pipeline at the validation cutoff (2020-09-09) for a seeded sample of
validation-week buyers and writes data/processed/demo_recs.json (untracked: it contains H&M data).
Transactions on or after 2020-09-16 are dropped at load time, so the test week is never read.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import lightgbm as lgb  # noqa: E402
from fashion_recsys import baselines as bl  # noqa: E402
from fashion_recsys.config import to_day  # noqa: E402
from fashion_recsys.data import load_transaction_codes  # noqa: E402
from fashion_recsys.evaluation import build_eval_set  # noqa: E402
from fashion_recsys.reranker import top_k_per_customer  # noqa: E402
from fashion_recsys.two_tower import build_examples, retrieve  # noqa: E402
from run_reranker import PROC, Ctx, load_D, say  # noqa: E402
from run_test_final import assert_no_test_data, load_frozen_two_tower  # noqa: E402

N_WARM, N_COLD, SEED = 30, 10, 0


def main():
    D, split, val_start, _ = load_D(truncate_before="test_start")
    assert_no_test_data(D, split)
    day, cust, art, n_art = D["day"], D["cust"], D["art"], D["n_art"]
    cutoff, end = val_start, to_day(split.val_end)
    customers, truths, hist = build_eval_set(day, cust, art, np.arange(D["n_cust"]), cutoff, end)
    rng = np.random.default_rng(SEED)
    cold, warm = np.flatnonzero(hist == 0), np.flatnonzero(hist > 0)
    pick = np.sort(np.r_[rng.choice(warm, N_WARM, replace=False), rng.choice(cold, N_COLD, replace=False)])
    sample, truth, hist = customers[pick], [truths[i] for i in pick], hist[pick]

    tt_model, tcfg = load_frozen_two_tower(D)
    m = (day >= cutoff - 182) & (day < cutoff)
    catalog = np.flatnonzero(np.bincount(art[m], minlength=n_art) > 0)
    ex = build_examples(day, cust, art, cutoff, n_art, tcfg["K"], tcfg["tau"], tcfg["lookback"], customers=sample, with_targets=False)
    codes, scores = retrieve(tt_model, ex, D["bucket"], catalog, top=100)
    ctx = Ctx(D, cutoff, tt=(sample, codes, scores))
    booster = lgb.Booster(model_file=str(ROOT / "models" / "lgbm_reranker.txt"))
    meta = json.loads((ROOT / "models" / "lgbm_reranker_meta.json").read_text())
    src = ctx.sources(sample)
    pcust, part, X, _ = ctx.pool_features(sample, src)
    recs = top_k_per_customer(pcust, part, booster.predict(X, num_iteration=meta["best_iteration"]), len(sample), 12)
    recent = bl.recent_items(day, cust, art, cutoff, sample, 8)

    ids = pq.read_table(PROC / "articles.parquet", columns=["article_id", "prod_name", "product_type_name", "colour_group_name", "department_name"]).to_pandas()
    cust_ids = pq.read_table(PROC / "customers.parquet", columns=["customer_id"])["customer_id"].to_pylist()

    def art_info(a):
        r = ids.iloc[int(a)]
        return {"article_id": r["article_id"], "name": r["prod_name"], "type": r["product_type_name"],
                "colour": r["colour_group_name"], "department": r["department_name"]}

    out = {"cutoff": str(split.val_start), "note": "frozen reranker; labels = validation week; test week not read", "customers": []}
    for i, c in enumerate(sample):
        out["customers"].append({
            "customer": cust_ids[int(c)][:8] + "...", "cold": bool(hist[i] == 0), "n_purchases": int(hist[i]),
            "recent": [art_info(a) for a in recent[i]],
            "recommendations": [{"rank": r + 1, **art_info(a), "bought_next_week": bool(int(a) in truth[i])} for r, a in enumerate(recs[i])],
            "n_bought_next_week": len(truth[i])})
    path = PROC / "demo_recs.json"
    path.write_text(json.dumps(out, indent=1))
    hits = sum(r["bought_next_week"] for c in out["customers"] for r in c["recommendations"])
    say(f"wrote {path} ({len(sample)} customers, {hits} hits in 12-item lists)")


if __name__ == "__main__":
    main()
