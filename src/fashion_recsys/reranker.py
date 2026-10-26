"""Candidate pool + feature builder + LightGBM ranker utilities.

Every feature of a (customer, article) pair at a cutoff uses only transactions with
day < cutoff (transactions are sorted by day, so that is a prefix) and static metadata.
"""
from dataclasses import dataclass
from itertools import chain

import numpy as np

SOURCES = ["recent", "cf", "var", "agepop", "pop", "tt"]
FEATURES = (
    [f"rank_{s}" for s in SOURCES] + ["tt_score", "n_other_sources"]
    + ["pair_art_count", "pair_art_days_since", "pair_prod_count", "pair_prod_days_since",
       "pair_ptype_count", "pair_colour_count", "pair_dept_count"]
    + ["art_pop_1w", "art_pop_4w", "art_pop_12w", "art_age_days", "art_days_since_sale", "art_price", "price_ratio"]
    + ["cust_age", "cust_n_purchases", "cust_weeks_since_last", "cust_share_ch1", "cust_share_ch2", "cust_avg_price"]
    + ["art_product_type", "art_colour_group", "art_department"]
)
CATEGORICAL = ["art_product_type", "art_colour_group", "art_department"]
TT_FEATURES = ["rank_tt", "tt_score"]
CONTENT_FEATURES = CATEGORICAL


@dataclass
class Meta:
    """Static article/customer metadata (codes are 0-based positions in the parquet files)."""
    product_code: np.ndarray
    product_type: np.ndarray
    colour: np.ndarray
    department: np.ndarray
    age: np.ndarray  # float, NaN when unknown


@dataclass
class CutoffStats:
    """Global statistics computed from transactions strictly before `cutoff`."""
    cutoff: int
    n_pre: int
    pop1: np.ndarray
    pop4: np.ndarray
    pop12: np.ndarray
    art_first: np.ndarray
    art_last: np.ndarray
    art_price: np.ndarray
    cust_n: np.ndarray
    cust_last: np.ndarray
    cust_avg_price: np.ndarray
    cust_ch2: np.ndarray


def cutoff_stats(day, cust, art, price, chan, cutoff, n_art, n_cust):
    n_pre = int(np.searchsorted(day, cutoff, side="left"))  # day is sorted ascending
    d, c, a = day[:n_pre], cust[:n_pre], art[:n_pre]

    def pop(weeks):
        lo = int(np.searchsorted(d, cutoff - 7 * weeks, side="left"))
        return np.bincount(a[lo:], minlength=n_art).astype(np.float32)

    art_first = np.full(n_art, np.nan, dtype=np.float32)
    u, idx = np.unique(a, return_index=True)
    art_first[u] = d[idx]
    art_last = np.full(n_art, np.nan, dtype=np.float32)
    u, idx = np.unique(a[::-1], return_index=True)
    art_last[u] = d[n_pre - 1 - idx]
    cnt_a = np.bincount(a, minlength=n_art)
    art_price = np.bincount(a, weights=price[:n_pre], minlength=n_art) / np.maximum(cnt_a, 1)
    art_price[cnt_a == 0] = np.nan
    cust_n = np.bincount(c, minlength=n_cust)
    cust_last = np.full(n_cust, np.nan, dtype=np.float32)
    u, idx = np.unique(c[::-1], return_index=True)
    cust_last[u] = d[n_pre - 1 - idx]
    cust_avg = np.bincount(c, weights=price[:n_pre], minlength=n_cust) / np.maximum(cust_n, 1)
    cust_avg[cust_n == 0] = np.nan
    ch2 = np.bincount(c, weights=(chan[:n_pre] == 2).astype(np.float64), minlength=n_cust) / np.maximum(cust_n, 1)
    ch2[cust_n == 0] = np.nan
    return CutoffStats(cutoff, n_pre, pop(1), pop(4), pop(12), art_first, art_last,
                       art_price.astype(np.float32), cust_n.astype(np.float32), cust_last,
                       cust_avg.astype(np.float32), ch2.astype(np.float32))


# --------------------------------------------------------------------------- pool
def _flatten(lists):
    lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
    flat = np.fromiter(chain.from_iterable(lists), dtype=np.int64, count=int(lens.sum()))
    local = np.repeat(np.arange(len(lists)), lens)
    starts = np.cumsum(lens) - lens
    rank = np.arange(len(flat)) - np.repeat(starts, lens) + 1
    return local, flat, rank


def build_pool(sources, n_local, n_art):
    """Union of the candidate lists of every source.

    `sources`: dict name -> list (one per customer) of ranked article codes, or for the
    two-tower an (n, k) int array (rank = column + 1) given as dict entry 'tt' together with
    'tt_score' (n, k) float array. Returns (pcust, part, ranks) where ranks[:, SOURCES.index(s)]
    is the 1-based rank in source s (NaN if absent), sorted by (customer, article).
    """
    parts, keys = {}, []
    for s in SOURCES:
        if s == "tt":
            codes = np.asarray(sources["tt"])
            local = np.repeat(np.arange(codes.shape[0]), codes.shape[1])
            flat, rank = codes.ravel().astype(np.int64), np.tile(np.arange(1, codes.shape[1] + 1), codes.shape[0])
        else:
            local, flat, rank = _flatten(sources[s])
        parts[s] = (local * n_art + flat, rank)
        keys.append(parts[s][0])
    ukeys = np.unique(np.concatenate(keys))
    ranks = np.full((len(ukeys), len(SOURCES)), np.nan, dtype=np.float32)
    for j, s in enumerate(SOURCES):
        k, r = parts[s]
        ranks[np.searchsorted(ukeys, k), j] = r
    tt_score = np.full(len(ukeys), np.nan, dtype=np.float32)
    if "tt_score" in sources:
        sc = np.asarray(sources["tt_score"], dtype=np.float32)
        tt_score[np.searchsorted(ukeys, parts["tt"][0])] = sc.ravel()
    return ukeys // n_art, ukeys % n_art, ranks, tt_score


# --------------------------------------------------------------------------- features
def _pair_lookup(hist_keys, hist_days, query_keys, cutoff):
    """Count of earlier purchases and days since the latest one, for each query key."""
    uk, inv, cnt = np.unique(hist_keys, return_inverse=True, return_counts=True)
    last = np.full(len(uk), -1, dtype=np.int64)
    np.maximum.at(last, inv, hist_days)
    pos = np.clip(np.searchsorted(uk, query_keys), 0, max(len(uk) - 1, 0))
    found = (uk[pos] == query_keys) if len(uk) else np.zeros(len(query_keys), dtype=bool)
    count = np.where(found, cnt[pos] if len(uk) else 0, 0).astype(np.float32)
    since = np.where(found, cutoff - last[pos] if len(uk) else 0, np.nan).astype(np.float32)
    return count, since


def pair_features(stats, meta, day, cust, art, chunk_customers, pcust, part, ranks, tt_score, n_art):
    """Feature matrix (rows aligned with the pool) for customers `chunk_customers`
    (pcust indexes into it). Uses only rows before the cutoff."""
    cutoff, n_pre = stats.cutoff, stats.n_pre
    n_cust_total = len(stats.cust_n)
    row_of = np.full(n_cust_total, -1, dtype=np.int64)
    row_of[chunk_customers] = np.arange(len(chunk_customers))
    r_all = row_of[cust[:n_pre]]
    m = r_all >= 0
    hl, ha, hd = r_all[m], art[:n_pre][m].astype(np.int64), day[:n_pre][m].astype(np.int64)

    n_prod = int(meta.product_code.max()) + 1
    n_pt, n_col, n_dep = (int(x.max()) + 1 for x in (meta.product_type, meta.colour, meta.department))
    a_cnt, a_since = _pair_lookup(hl * n_art + ha, hd, pcust * n_art + part, cutoff)
    p_cnt, p_since = _pair_lookup(hl * n_prod + meta.product_code[ha], hd, pcust * n_prod + meta.product_code[part], cutoff)
    pt_cnt, _ = _pair_lookup(hl * n_pt + meta.product_type[ha], hd, pcust * n_pt + meta.product_type[part], cutoff)
    co_cnt, _ = _pair_lookup(hl * n_col + meta.colour[ha], hd, pcust * n_col + meta.colour[part], cutoff)
    de_cnt, _ = _pair_lookup(hl * n_dep + meta.department[ha], hd, pcust * n_dep + meta.department[part], cutoff)

    gc = chunk_customers[pcust]
    art_price = stats.art_price[part]
    cust_avg = stats.cust_avg_price[gc]
    cols = {f"rank_{s}": ranks[:, j] for j, s in enumerate(SOURCES)}
    cols["tt_score"] = tt_score
    cols["n_other_sources"] = np.isfinite(ranks[:, [j for j, s in enumerate(SOURCES) if s != "tt"]]).sum(1).astype(np.float32)
    cols.update(pair_art_count=a_cnt, pair_art_days_since=a_since, pair_prod_count=p_cnt, pair_prod_days_since=p_since,
                pair_ptype_count=pt_cnt, pair_colour_count=co_cnt, pair_dept_count=de_cnt)
    cols.update(art_pop_1w=stats.pop1[part], art_pop_4w=stats.pop4[part], art_pop_12w=stats.pop12[part],
                art_age_days=cutoff - stats.art_first[part], art_days_since_sale=cutoff - stats.art_last[part],
                art_price=art_price, price_ratio=art_price / cust_avg)
    ch2 = stats.cust_ch2[gc]
    cols.update(cust_age=meta.age[gc].astype(np.float32), cust_n_purchases=stats.cust_n[gc],
                cust_weeks_since_last=(cutoff - stats.cust_last[gc]) / 7.0, cust_share_ch1=1.0 - ch2, cust_share_ch2=ch2,
                cust_avg_price=cust_avg)
    cols.update(art_product_type=meta.product_type[part].astype(np.float32), art_colour_group=meta.colour[part].astype(np.float32),
                art_department=meta.department[part].astype(np.float32))
    X = np.empty((len(part), len(FEATURES)), dtype=np.float32)
    for j, f in enumerate(FEATURES):
        X[:, j] = cols[f]
    return X, a_cnt


def labels_for(pcust, part, truths, n_art):
    """1 where the pooled article is in the customer's relevant set."""
    tk = np.sort(np.fromiter(chain.from_iterable(
        (i * n_art + a for a in t) for i, t in enumerate(truths)), dtype=np.int64))
    key = pcust * n_art + part
    pos = np.clip(np.searchsorted(tk, key), 0, max(len(tk) - 1, 0))
    return ((tk[pos] == key) if len(tk) else np.zeros(len(key), dtype=bool)).astype(np.float32)


def top_k_per_customer(pcust, part, score, n_local, k=12):
    """Top-k distinct articles per customer by score (pool is sorted by customer)."""
    starts = np.searchsorted(pcust, np.arange(n_local), side="left")
    ends = np.searchsorted(pcust, np.arange(n_local), side="right")
    out = []
    for s, e in zip(starts, ends):
        sc = score[s:e]
        top = np.argsort(-sc, kind="stable")[:k]
        out.append([int(x) for x in part[s:e][top]])
    return out


LGB_BASE = dict(objective="lambdarank", metric="ndcg", eval_at=[12], seed=0, deterministic=True, force_row_wise=True,
                num_threads=4, verbose=-1, bagging_seed=0, feature_fraction_seed=0, data_random_seed=0)


def fit_ranker(Xtr, ytr, gtr, Xva, yva, gva, feature_names, params, rounds=400, early=30):
    import lightgbm as lgb

    cat = [f for f in CATEGORICAL if f in feature_names]
    dtr = lgb.Dataset(Xtr, ytr, group=gtr, feature_name=feature_names, categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, group=gva, reference=dtr)
    return lgb.train({**LGB_BASE, **params}, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(early, verbose=False)])
