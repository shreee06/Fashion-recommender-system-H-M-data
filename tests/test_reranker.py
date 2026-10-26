import numpy as np
import pytest

from fashion_recsys.reranker import (
    FEATURES, SOURCES, Meta, build_pool, cutoff_stats, fit_ranker, labels_for, pair_features, top_k_per_customer,
)

N_ART, N_CUST = 10, 4


def meta():
    a = np.arange(N_ART)
    return Meta(product_code=a // 3, product_type=a % 4, colour=a % 3, department=a % 2,
                age=np.array([20, 30, np.nan, 50], dtype=np.float32))


def toy_tx():
    # sorted by day; cutoff 20
    day = np.array([5, 10, 12, 15, 18, 21, 22])
    cust = np.array([0, 0, 1, 0, 1, 0, 1])
    art = np.array([3, 3, 4, 5, 3, 9, 9])
    price = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.9], dtype=np.float32)
    chan = np.array([1, 2, 2, 2, 1, 1, 1], dtype=np.int8)
    return day, cust, art, price, chan


def pool_for(chunk, sources):
    return build_pool(sources, len(chunk), N_ART)


def empty_sources(n):
    s = {k: [[] for _ in range(n)] for k in SOURCES if k != "tt"}
    s["tt"] = np.tile(np.arange(2), (n, 1))
    s["tt_score"] = np.full((n, 2), 0.5, dtype=np.float32)
    return s


def feats(day, cust, art, price, chan, chunk, sources, cutoff=20):
    stats = cutoff_stats(day, cust, art, price, chan, cutoff, N_ART, N_CUST)
    pcust, part, ranks, tts = pool_for(chunk, sources)
    X, _ = pair_features(stats, meta(), day, cust, art, chunk, pcust, part, ranks, tts, N_ART)
    return pcust, part, X


def test_build_pool_union_and_ranks_hand_checked():
    src = empty_sources(2)
    src["recent"] = [[3, 1], [2]]
    src["cf"] = [[1], [5]]
    src["pop"] = [[9], [9]]
    src["tt"] = np.array([[1, 7], [2, 3]])
    src["tt_score"] = np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32)
    pcust, part, ranks, tts = build_pool(src, 2, N_ART)
    c0 = pcust == 0
    assert part[c0].tolist() == [1, 3, 7, 9]  # union, sorted by article
    col = {s: j for j, s in enumerate(SOURCES)}
    r = {a: ranks[c0][i] for i, a in enumerate(part[c0])}
    assert r[3][col["recent"]] == 1 and r[1][col["recent"]] == 2 and np.isnan(r[7][col["recent"]])
    assert r[1][col["cf"]] == 1 and r[1][col["tt"]] == 1 and r[7][col["tt"]] == 2 and r[9][col["pop"]] == 1
    assert tts[c0].tolist() == pytest.approx([0.9, np.nan, 0.8, np.nan], nan_ok=True)
    assert part[pcust == 1].tolist() == [2, 3, 5, 9]


def test_pair_features_hand_checked():
    day, cust, art, price, chan = toy_tx()
    src = empty_sources(1)
    src["recent"] = [[3, 4, 9, 5]]
    pcust, part, X = feats(day, cust, art, price, chan, np.array([0]), src)
    f = lambda name, a: X[list(part).index(a), FEATURES.index(name)]
    # customer 0 bought article 3 on days 5 and 10 (cutoff 20): count 2, 10 days since the last one
    assert f("pair_art_count", 3) == 2 and f("pair_art_days_since", 3) == 10
    # article 4 never bought by customer 0 -> count 0, no recency
    assert f("pair_art_count", 4) == 0 and np.isnan(f("pair_art_days_since", 4))
    # product_code = art // 3: articles 3,4,5 share product 1; customer bought 3 (days 5, 10) and 5 (day 15) -> 3 purchases, last 5 days ago
    assert f("pair_prod_count", 4) == 3 and f("pair_prod_days_since", 4) == 5
    # popularity 1w: day >= 13 and < 20 -> rows (15,art5) and (18,art3)
    assert f("art_pop_1w", 3) == 1 and f("art_pop_1w", 5) == 1 and f("art_pop_1w", 4) == 0
    # customer: 3 purchases before 20 (days 5,10,15); channels 1,2,2 -> share ch2 = 2/3
    assert f("cust_n_purchases", 3) == 3 and f("cust_share_ch2", 3) == pytest.approx(2 / 3)
    assert f("cust_weeks_since_last", 3) == pytest.approx(5 / 7)
    assert f("cust_avg_price", 3) == pytest.approx((0.1 + 0.2 + 0.4) / 3)
    assert f("cust_age", 3) == 20


def test_features_ignore_cutoff_and_later():
    day, cust, art, price, chan = toy_tx()
    chunk = np.array([0, 1, 2])
    src = empty_sources(3)
    src["recent"] = [[3, 5], [3, 4], []]
    a = feats(day, cust, art, price, chan, chunk, src)
    # add future rows (day >= cutoff) for the same and other customers/articles, strongly different values
    day2 = np.r_[day, [20, 20, 25, 30, 30]]
    cust2 = np.r_[cust, [0, 1, 2, 0, 1]]
    art2 = np.r_[art, [3, 3, 4, 5, 3]]
    price2 = np.r_[price, np.full(5, 9.0, dtype=np.float32)]
    chan2 = np.r_[chan, np.full(5, 2, dtype=np.int8)]
    b = feats(day2, cust2, art2, price2, chan2, chunk, src)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2], equal_nan=True)


def test_labels_for():
    pcust = np.array([0, 0, 1])
    part = np.array([2, 5, 5])
    y = labels_for(pcust, part, [{5}, {2, 9}], N_ART)
    assert y.tolist() == [0.0, 1.0, 0.0]


def synthetic_ranking(seed=0, n_cust=300, n_cand=20):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_cust * n_cand, len(FEATURES))).astype(np.float32)
    y = np.zeros(n_cust * n_cand, dtype=np.float32)
    rel = rng.integers(0, n_cand, n_cust) + np.arange(n_cust) * n_cand  # one relevant candidate per customer
    y[rel] = 1
    X[rel, FEATURES.index("pair_art_count")] += 4.0  # obviously informative
    X[:, [FEATURES.index(c) for c in ("art_product_type", "art_colour_group", "art_department")]] = rng.integers(0, 5, (len(X), 3))
    return X, y, np.full(n_cust, n_cand), rel


def test_ranker_puts_obvious_candidate_first():
    X, y, g, rel = synthetic_ranking()
    tr, va = slice(0, 200 * 20), slice(200 * 20, None)
    b = fit_ranker(X[tr], y[tr], g[:200], X[va], y[va], g[200:], FEATURES, dict(num_leaves=7, learning_rate=0.2, min_data_in_leaf=5), rounds=50)
    sc = b.predict(X[va]).reshape(100, 20)
    assert (sc.argmax(1) == (rel[200:] - np.arange(200, 300) * 20)).mean() > 0.9


def test_determinism():
    X, y, g, _ = synthetic_ranking()
    p = dict(num_leaves=7, learning_rate=0.2, min_data_in_leaf=5)
    preds = [fit_ranker(X[:4000], y[:4000], g[:200], X[4000:], y[4000:], g[200:], FEATURES, p, rounds=20).predict(X[4000:]) for _ in range(2)]
    assert np.array_equal(preds[0], preds[1])


def test_top_k_exactly_12_distinct():
    rng = np.random.default_rng(0)
    lens = [30, 12, 100]
    pcust = np.repeat(np.arange(3), lens)
    part = np.concatenate([rng.permutation(500)[:n] for n in lens])
    out = top_k_per_customer(pcust, part, rng.random(len(part)), 3, 12)
    assert all(len(o) == 12 and len(set(o)) == 12 for o in out)
    # best-scored article first
    sc = np.array([0.1, 0.9, 0.5] + [0.0] * 9)
    top = top_k_per_customer(np.zeros(12, dtype=int), np.arange(12), sc, 1, 12)[0]
    assert top[:3] == [1, 2, 0]
