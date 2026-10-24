"""Temporal split and customer-sample definitions.

Split dates are derived from the last date in the data, never hardcoded:
    TEST       = the last 7 days
    VALIDATION = the 7 days before the test week
    TRAIN      = everything before the validation week
"""
import datetime
from dataclasses import asdict, dataclass

import numpy as np

SAMPLE_SIZE = 200_000
SAMPLE_SEED = 42
HISTORY_WEEKS = 26  # default window of train used for model training
TOP_K = 12
EPOCH = datetime.date(1970, 1, 1)


def to_date(day):
    return EPOCH + datetime.timedelta(days=int(day))


def to_day(date):
    return (date - EPOCH).days


@dataclass(frozen=True)
class Split:
    """Inclusive date ranges. Train ends the day before validation starts."""

    train_start: datetime.date
    train_end: datetime.date
    val_start: datetime.date
    val_end: datetime.date
    test_start: datetime.date
    test_end: datetime.date

    def to_json(self):
        return {k: v.isoformat() for k, v in asdict(self).items()}


def make_split(min_day, max_day):
    """Build the split from the first and last day numbers (days since epoch) in the data."""
    max_day, min_day = int(max_day), int(min_day)
    return Split(
        train_start=to_date(min_day),
        train_end=to_date(max_day - 14),
        val_start=to_date(max_day - 13),
        val_end=to_date(max_day - 7),
        test_start=to_date(max_day - 6),
        test_end=to_date(max_day),
    )


def history_start(split, weeks=HISTORY_WEEKS, cutoff=None):
    """First day of the training window: the last `weeks` weeks before `cutoff`
    (default: start of validation)."""
    cutoff = cutoff or split.val_start
    return max(split.train_start, cutoff - datetime.timedelta(weeks=weeks))


def sample_customers(eligible, n=SAMPLE_SIZE, seed=SAMPLE_SEED):
    """Seeded sample (without replacement) of `n` customers out of `eligible`
    (ids or codes of customers with at least one purchase). Order-independent:
    the pool is sorted first, so the same seed always gives the same sample."""
    pool = np.sort(np.asarray(eligible))
    n = min(n, len(pool))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=n, replace=False))
