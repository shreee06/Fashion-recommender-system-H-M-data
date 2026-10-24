"""Ranking metrics at K=12 (MAP, Recall, NDCG, catalog coverage) with group breakdowns.

`recs` is a sequence of ranked item lists (one per customer, best first) and
`truths` a parallel sequence of sets of relevant items (distinct articles bought
in the target week). Duplicate items in a recommendation list are skipped.
"""
import math

import numpy as np

K = 12


def _dedupe(items, k):
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
            if len(out) == k:
                break
    return out


def average_precision(rec, relevant, k=K):
    """Kaggle-style AP@k: sum of precision@i at every relevant rank i, divided by
    min(k, |relevant|)."""
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, it in enumerate(_dedupe(rec, k), start=1):
        if it in relevant:
            hits += 1
            score += hits / i
    return score / min(k, len(relevant))


def recall_at_k(rec, relevant, k=K):
    if not relevant:
        return 0.0
    return sum(it in relevant for it in _dedupe(rec, k)) / len(relevant)


def ndcg_at_k(rec, relevant, k=K):
    """Binary-gain NDCG@k; the ideal DCG places min(k, |relevant|) hits first."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, it in enumerate(_dedupe(rec, k), start=1) if it in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / idcg


def coverage_at_k(recs, catalog_size, k=K):
    """Share of the catalog (`catalog_size` articles) recommended to at least one customer."""
    distinct = set()
    for r in recs:
        distinct.update(_dedupe(r, k))
    return len(distinct) / catalog_size


def evaluate(recs, truths, catalog_size, k=K):
    """Mean MAP/Recall/NDCG over customers plus catalog coverage."""
    n = len(recs)
    if n == 0:
        return {"n_customers": 0, "map@12": None, "recall@12": None, "ndcg@12": None, "coverage@12": None}
    return {
        "n_customers": n,
        "map@12": float(np.mean([average_precision(r, t, k) for r, t in zip(recs, truths)])),
        "recall@12": float(np.mean([recall_at_k(r, t, k) for r, t in zip(recs, truths)])),
        "ndcg@12": float(np.mean([ndcg_at_k(r, t, k) for r, t in zip(recs, truths)])),
        "coverage@12": coverage_at_k(recs, catalog_size, k),
    }


def evaluate_by_group(recs, truths, groups, catalog_size, k=K):
    """Metrics per group label (e.g. 'cold'/'warm'); `groups` is parallel to `recs`."""
    groups = list(groups)
    out = {}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        out[g] = evaluate([recs[i] for i in idx], [truths[i] for i in idx], catalog_size, k)
    return out


def evaluate_cold_warm(recs, truths, history_counts, catalog_size, k=K):
    """Overall metrics, cold-start share, and cold vs warm breakdown.

    `history_counts[i]` is the number of purchases customer i has before the cutoff;
    cold = 0.
    """
    groups = ["cold" if c == 0 else "warm" for c in history_counts]
    n = len(groups)
    return {
        "overall": evaluate(recs, truths, catalog_size, k),
        "cold_share": (groups.count("cold") / n) if n else None,
        "by_group": evaluate_by_group(recs, truths, groups, catalog_size, k),
    }


def build_eval_set(day, cust, art, sampled, target_start, target_end):
    """Customers (from `sampled`) who bought in [target_start, target_end] (day numbers,
    inclusive), their relevant sets (distinct articles bought in that window), and
    their purchase counts before `target_start` (history, for cold/warm).

    The cutoff for history and training is `target_start`.
    """
    in_sample = np.zeros(max(cust.max(), np.max(sampled)) + 1, dtype=bool)
    in_sample[sampled] = True
    target = (day >= target_start) & (day <= target_end) & in_sample[cust]
    tc, ta = cust[target], art[target]
    order = np.argsort(tc, kind="stable")
    tc, ta = tc[order], ta[order]
    customers, starts = np.unique(tc, return_index=True)
    truths = [set(map(int, x)) for x in np.split(ta, starts[1:])]
    hist = np.bincount(cust[(day < target_start) & in_sample[cust]], minlength=len(in_sample))
    return customers, truths, hist[customers]
