import numpy as np

from fashion_recsys.baselines import global_popular, recent_popular, repurchase


def toy():
    # 20 articles; days 0..27
    day = np.array([0, 1, 2, 20, 21, 22, 22, 25, 26, 27])
    cust = np.array([0, 0, 1, 0, 1, 1, 2, 0, 2, 2])
    art = np.array([1, 2, 1, 3, 3, 4, 3, 5, 6, 6])
    return day, cust, art


def test_global_popular_counts_and_exact_12():
    day, cust, art = toy()
    # before day 26: counts: art1=2, art3=3 -> art3 first, then art1; rest fill by code
    rec = global_popular(day, art, 26, 20)
    assert len(rec) == 12 and len(set(rec.tolist())) == 12
    assert rec[0] == 3 and rec[1] == 1


def test_recent_popular_window():
    day, cust, art = toy()
    # 1 week before cutoff 28 = days 21..27: art3 x2 (21,22), art4, art6 x2, art5
    rec = recent_popular(day, art, 28, 1, 20)
    assert rec[:2].tolist() == [3, 6]  # tie between 3 and 6 (2 each) -> lower code first
    assert len(set(rec.tolist())) == 12


def test_repurchase_recent_first_then_fill():
    day, cust, art = toy()
    fb = np.array([3, 6, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13])
    recs = repurchase(day, cust, art, 26, np.array([0, 5]), fb)  # customer 5 has no history
    # customer 0 before day 26: (0,1)(1,2)(20,3)(25,5) -> most recent first: 5,3,2,1
    assert recs[0][:4] == [5, 3, 2, 1]
    assert recs[1] == fb[:12].tolist()
    for r in recs:
        assert len(r) == 12 and len(set(r)) == 12
