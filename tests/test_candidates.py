import numpy as np
import pytest

from fashion_recsys.item_cf import ItemCF
from fashion_recsys.segments import (
    age_bucket, customer_segment_popular, dominant_group, group_popular,
)
from fashion_recsys.variants import VariantCandidates


# ---------- recency weighting (hand-checked) ----------
def recency_toy():
    # cutoff 105. customer 0 bought items 0,1 on day 98 (age 1 week);
    # customer 1 bought items 0,2 on day 77 (age 4 weeks). tau = 2 weeks.
    day = np.array([98, 98, 77, 77])
    cust = np.array([0, 0, 1, 1])
    art = np.array([0, 1, 0, 2])
    return day, cust, art


def test_recency_weights_hand_checked():
    day, cust, art = recency_toy()
    cf = ItemCF(window_weeks=26).fit(day, cust, art, 105, 3, recency_tau_weeks=2)
    # weights: cust0 -> e^-0.5 each (items 0,1); cust1 -> e^-2 each (items 0,2)
    # C01 = e^-0.5 * e^-0.5 = e^-1 ; C02 = e^-2 * e^-2 = e^-4
    # norms (sum of squares): n0 = e^-1 + e^-4, n1 = e^-1, n2 = e^-4
    e1, e4 = np.exp(-1.0), np.exp(-4.0)
    p, S = cf.item_pos, cf.sim.toarray()
    assert S[p[0], p[1]] == pytest.approx(e1 / np.sqrt((e1 + e4) * e1), rel=1e-5)
    assert S[p[0], p[2]] == pytest.approx(e4 / np.sqrt((e1 + e4) * e4), rel=1e-5)
    # unweighted would be 1/sqrt(2) for both; recency makes the fresher pair stronger
    assert S[p[0], p[1]] > S[p[0], p[2]]


def test_no_recency_equals_binary():
    day, cust, art = recency_toy()
    cf = ItemCF().fit(day, cust, art, 105, 3)
    p = cf.item_pos
    assert cf.sim.toarray()[p[0], p[1]] == pytest.approx(1 / np.sqrt(2))  # C01=1, n0=2, n1=1 -> 1/sqrt(2)


def test_chunked_equals_unchunked():
    rng = np.random.default_rng(0)
    cust = rng.integers(0, 50, 400)
    art = rng.integers(0, 12, 400)
    day = rng.integers(0, 100, 400)
    a = ItemCF().fit(day, cust, art, 100, 12, chunk_rows=7)
    b = ItemCF().fit(day, cust, art, 100, 12, chunk_rows=10_000)
    assert np.allclose(a.sim.toarray(), b.sim.toarray(), atol=1e-6)


def test_recency_cf_ignores_cutoff_and_later():
    day, cust, art = recency_toy()
    base = ItemCF().fit(day, cust, art, 105, 5, recency_tau_weeks=2)
    d2 = np.concatenate([day, [105, 106, 110]]); c2 = np.concatenate([cust, [0, 1, 1]]); a2 = np.concatenate([art, [3, 3, 4]])
    leaky = ItemCF().fit(d2, c2, a2, 105, 5, recency_tau_weeks=2)
    assert np.array_equal(base.items, leaky.items) and (base.sim != leaky.sim).nnz == 0


# ---------- variants ----------
def variant_toy():
    # articles 0,1,2 share product 0; article 3 is product 1; article 4 is product 0 but never sold recently
    product = np.array([0, 0, 0, 1, 0])
    # recent (4 weeks before cutoff 100 = days 72..99) sales: art1 x3, art0 x2, art2 x1, art3 x1
    day = np.array([90, 91, 92, 80, 81, 95, 96])
    art = np.array([1, 1, 1, 0, 0, 2, 3])
    return product, day, art


def test_variants_hand_checked():
    product, day, art = variant_toy()
    v = VariantCandidates(pop_weeks=4).fit(day, art, 100, product, 5)
    # customer bought article 0 -> siblings 1 (3 sales), 2 (1 sale); article 4 has no recent sale
    assert v.recommend([[0]], k_seeds=1) == [[1, 2]]
    # customer already has article 1 among recents -> not offered again
    assert v.recommend([[0, 1]], k_seeds=2) == [[2]]
    # product 1 has no siblings
    assert v.recommend([[3]], k_seeds=1) == [[]]
    # seeds limited to the most recent k_seeds
    assert v.recommend([[3, 0]], k_seeds=1) == [[]]


def test_variants_ignore_cutoff_and_later():
    product, day, art = variant_toy()
    base = VariantCandidates().fit(day, art, 100, product, 5)
    d2 = np.concatenate([day, [100, 101, 101, 101]]); a2 = np.concatenate([art, [2, 2, 2, 4]])
    leaky = VariantCandidates().fit(d2, a2, 100, product, 5)
    assert base.by_product == leaky.by_product


# ---------- segments ----------
def test_age_bucket():
    assert age_bucket([16, 24, 25, 34, 35, 44, 45, 54, 55, 90, np.nan]).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]


def seg_toy():
    # customers 0,1 in segment 0; customer 2 in segment 1; window = days 72..99 for cutoff 100
    day = np.array([80, 81, 82, 83, 84, 100, 101])
    cust = np.array([0, 1, 1, 2, 2, 2, 2])
    art = np.array([5, 5, 6, 7, 7, 7, 7])
    return day, cust, art


def test_segment_popular_hand_checked():
    day, cust, art = seg_toy()
    seg = np.array([0, 0, 1])
    r = customer_segment_popular(day, cust, art, seg, 2, 100, 4, 10, k=2)
    assert r[0].tolist() == [5, 6]  # seg0: art5 x2 (cust0, cust1), art6 x1
    assert r[1][0] == 7 and len(r[1]) == 2  # seg1: art7 x2 before the cutoff (the two day>=100 sales ignored)


def test_dominant_group_and_group_popular():
    day, cust, art = seg_toy()
    group = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])  # article -> group
    dom = dominant_group(day, cust, art, group, 2, 100, 4)
    # before cutoff: cust0 bought art5 (g1) ; cust1 art5,art6 (g1,g1); cust2 art7 (g1) x2 ; cust3 nothing
    assert dom.tolist() == [1, 1, 1, -1]
    gp = group_popular(day, art, group, 2, 100, 4, 10, k=3)
    assert set(gp[1].tolist()) == {5, 6, 7} and gp[1][0] == 5  # art5 x2, art7 x2 (tie -> lower code), art6 x1
    assert gp[1].tolist() == [5, 7, 6]


def test_segment_sources_ignore_cutoff_and_later():
    day, cust, art = seg_toy()
    seg = np.array([0, 0, 1])
    group = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
    d2 = np.concatenate([day, [100, 100, 150]]); c2 = np.concatenate([cust, [0, 1, 2]]); a2 = np.concatenate([art, [9, 9, 9]])
    a = customer_segment_popular(day, cust, art, seg, 2, 100, 4, 10, k=3)
    b = customer_segment_popular(d2, c2, a2, seg, 2, 100, 4, 10, k=3)
    assert all(np.array_equal(a[s], b[s]) for s in a)
    assert np.array_equal(dominant_group(day, cust, art, group, 2, 100, 4), dominant_group(d2, c2, a2, group, 2, 100, 4))
    ga, gb = group_popular(day, art, group, 2, 100, 4, 10, 3), group_popular(d2, a2, group, 2, 100, 4, 10, 3)
    assert all(np.array_equal(ga[g], gb[g]) for g in ga)
