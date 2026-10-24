import numpy as np
import pytest

from fashion_recsys.item_cf import ItemCF


def toy():
    # customers 0,1 bought items {0,1}; customer 2 bought {0,2}; days 10..12; cutoff 20
    day = np.array([10, 10, 11, 11, 12, 12])
    cust = np.array([0, 0, 1, 1, 2, 2])
    art = np.array([0, 1, 0, 1, 0, 2])
    return day, cust, art


def test_cosine_values_hand_checked():
    day, cust, art = toy()
    cf = ItemCF(window_weeks=26).fit(day, cust, art, 20, 4)
    # buyers per item: n0=3, n1=2, n2=1 ; co-buyers: C01=2, C02=1, C12=0
    # cos01 = 2/sqrt(3*2) = 0.81650 ; cos02 = 1/sqrt(3*1) = 0.57735
    S = cf.sim.toarray()
    p = cf.item_pos
    assert S[p[0], p[1]] == pytest.approx(2 / np.sqrt(6))
    assert S[p[0], p[2]] == pytest.approx(1 / np.sqrt(3))
    assert S[p[1], p[2]] == 0.0
    assert S[p[0], p[0]] == 0.0
    assert p[3] == -1  # never bought -> not modelled


def test_recommendations_hand_checked():
    day, cust, art = toy()
    cf = ItemCF().fit(day, cust, art, 20, 4)
    # extra customers (not in training baskets) with their own history
    day2 = np.concatenate([day, [15, 15]])
    cust2 = np.concatenate([cust, [3, 4]])
    art2 = np.concatenate([art, [0, 1]])
    cf = ItemCF().fit(day2, cust2, art2, 20, 4, train_customers=np.array([0, 1, 2]))
    recs = cf.recommend(day2, cust2, art2, np.array([3, 4]), k_seeds=1)
    # customer 3 bought item 0: neighbours 1 (0.8165) then 2 (0.5774)
    assert recs[0] == [1, 2]
    # customer 4 bought item 1: only neighbour is item 0 (0.8165)
    assert recs[1] == [0]


def test_k_seeds_uses_most_recent():
    day = np.array([10, 10, 11, 11, 12, 12, 14, 15])
    cust = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    art = np.array([0, 1, 0, 1, 0, 2, 2, 1])
    cf = ItemCF().fit(day, cust, art, 20, 3)
    # customer 3: most recent purchase is item 1 (day 15), then item 2 (day 14)
    only_recent = cf.recommend(day, cust, art, np.array([3]), k_seeds=1)[0]
    both = cf.recommend(day, cust, art, np.array([3]), k_seeds=2)[0]
    assert only_recent == [0]  # item 1's only neighbour is 0 (items 1,2 already seen are excluded)
    assert both == [0]


def test_no_leakage_from_cutoff_onwards():
    day, cust, art = toy()
    base = ItemCF().fit(day, cust, art, 20, 5)
    # add purchases ON the cutoff day and after, linking new item 3 and 4 to item 0
    day2 = np.concatenate([day, [20, 20, 25, 25, 30]])
    cust2 = np.concatenate([cust, [0, 0, 1, 1, 2]])
    art2 = np.concatenate([art, [3, 4, 3, 0, 4]])
    leaky = ItemCF().fit(day2, cust2, art2, 20, 5)
    assert np.array_equal(base.items, leaky.items)
    assert (base.sim != leaky.sim).nnz == 0
    assert leaky.item_pos[3] == -1 and leaky.item_pos[4] == -1


def test_window_excludes_old_purchases():
    day, cust, art = toy()
    cf = ItemCF(window_weeks=1).fit(day, cust, art, 20, 4)  # window = days 13..19 -> empty
    assert len(cf.items) == 0
