import datetime

import numpy as np

from fashion_recsys.config import history_start, make_split, sample_customers, to_day


def test_split_ordered_non_overlapping_and_derived_from_max():
    s = make_split(to_day(datetime.date(2018, 9, 20)), to_day(datetime.date(2020, 9, 22)))
    assert s.train_start <= s.train_end < s.val_start <= s.val_end < s.test_start <= s.test_end
    assert s.train_end + datetime.timedelta(days=1) == s.val_start
    assert s.val_end + datetime.timedelta(days=1) == s.test_start
    assert (s.val_end - s.val_start).days == 6 and (s.test_end - s.test_start).days == 6
    assert s.test_end == datetime.date(2020, 9, 22)


def test_split_follows_data_max():
    s = make_split(0, 100)
    assert s.test_end == datetime.date(1970, 1, 1) + datetime.timedelta(days=100)


def test_history_window():
    s = make_split(to_day(datetime.date(2018, 9, 20)), to_day(datetime.date(2020, 9, 22)))
    assert history_start(s, 26) == s.val_start - datetime.timedelta(weeks=26)
    assert history_start(s, 10_000) == s.train_start


def test_sample_reproducible_and_seed_sensitive():
    pool = np.arange(1000)
    a = sample_customers(pool, 100, seed=7)
    assert np.array_equal(a, sample_customers(pool, 100, seed=7))
    assert np.array_equal(a, sample_customers(pool[::-1], 100, seed=7))  # pool order irrelevant
    assert not np.array_equal(a, sample_customers(pool, 100, seed=8))
    assert len(set(a.tolist())) == 100


def test_sample_capped_at_pool_size():
    assert len(sample_customers(np.arange(5), 100)) == 5
