import math

import numpy as np
import pytest

from fashion_recsys.evaluation import (
    average_precision,
    build_eval_set,
    coverage_at_k,
    evaluate,
    evaluate_cold_warm,
    ndcg_at_k,
    recall_at_k,
)


def test_average_precision():
    # rec [a,b,c,d], relevant {a,c,x}: hits at rank1 (P=1/1) and rank3 (P=2/3)
    # AP = (1 + 2/3) / min(12, 3) = 0.5555...
    assert average_precision(["a", "b", "c", "d"], {"a", "c", "x"}) == pytest.approx(5 / 9)
    # one relevant item found at rank 2: P=1/2, divided by min(12,1)=1 -> 0.5
    assert average_precision(["z", "a"], {"a"}) == pytest.approx(0.5)
    assert average_precision(["z"], {"a"}) == 0.0


def test_average_precision_caps_denominator_at_12():
    # 20 relevant items, the first 12 recs are all relevant: precision 1 at each rank,
    # sum = 12, denominator min(12, 20) = 12 -> 1.0
    rel = set(range(20))
    assert average_precision(list(range(12)), rel) == pytest.approx(1.0)


def test_duplicates_are_skipped():
    # [a,a,b] dedupes to [a,b]; relevant {a,b}: P@1=1, P@2=1 -> (1+1)/2 = 1.0
    assert average_precision(["a", "a", "b"], {"a", "b"}) == pytest.approx(1.0)
    # duplicates must not crowd out the 12-item cut: 13 items, the 12th distinct is relevant
    rec = ["a"] * 5 + list("bcdefghijkl") + ["m"]  # distinct: a + 11 letters = 12, then m
    assert recall_at_k(rec, {"l"}) == 1.0 and recall_at_k(rec, {"m"}) == 0.0


def test_recall():
    # relevant {a,b,c,d}; recs hit a and c -> 2/4
    assert recall_at_k(["a", "x", "c"], {"a", "b", "c", "d"}) == pytest.approx(0.5)


def test_ndcg():
    # hit at rank 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # |relevant|=2 -> IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}) == pytest.approx(expected)
    # perfect ranking -> 1
    assert ndcg_at_k(["a", "b"], {"a", "b"}) == pytest.approx(1.0)


def test_coverage():
    # distinct recommended across customers: {a,b,c} = 3 of a 10-item catalog
    assert coverage_at_k([["a", "b"], ["b", "c"]], 10) == pytest.approx(0.3)


def test_evaluate_means():
    recs = [["a"], ["z"]]
    truths = [{"a"}, {"a"}]  # AP 1 and 0 -> mean 0.5
    m = evaluate(recs, truths, 10)
    assert m["map@12"] == pytest.approx(0.5) and m["recall@12"] == pytest.approx(0.5)
    assert m["coverage@12"] == pytest.approx(0.2)


def test_cold_warm_split():
    recs = [["a"], ["z"], ["a"]]
    truths = [{"a"}, {"a"}, {"a"}]
    r = evaluate_cold_warm(recs, truths, [0, 3, 0], 10)
    assert r["cold_share"] == pytest.approx(2 / 3)
    assert r["by_group"]["cold"]["map@12"] == pytest.approx(1.0)  # both cold customers hit
    assert r["by_group"]["warm"]["map@12"] == 0.0


def test_build_eval_set():
    day = np.array([1, 2, 10, 10, 11, 10, 10])
    cust = np.array([0, 1, 0, 0, 1, 2, 3])
    art = np.array([5, 5, 6, 6, 7, 8, 9])
    # target window [10,12]; sample excludes customer 3. Customer 2 has no history.
    customers, truths, hist = build_eval_set(day, cust, art, np.array([0, 1, 2]), 10, 12)
    assert customers.tolist() == [0, 1, 2]
    assert truths == [{6}, {7}, {8}]  # duplicates collapse to distinct articles
    assert hist.tolist() == [1, 1, 0]
