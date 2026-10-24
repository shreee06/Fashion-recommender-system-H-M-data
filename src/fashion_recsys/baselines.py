"""Non-personalised and repurchase baselines. All work on integer article codes
(see data.load_transaction_codes) and return exactly TOP_K distinct articles."""
import numpy as np

from .config import TOP_K


def most_popular(day, art, start_day, end_day, n_articles, k=TOP_K):
    """Top-k articles by purchase count for days in [start_day, end_day).

    Ties break towards the lower article code (deterministic). If fewer than k
    articles were bought in the window, the remainder is filled with unbought
    articles in code order so that exactly k distinct articles are returned.
    """
    mask = (day >= start_day) & (day < end_day)
    counts = np.bincount(art[mask], minlength=n_articles)
    order = np.lexsort((np.arange(n_articles), -counts))
    return order[:k].astype(np.int64)


def global_popular(day, art, cutoff_day, n_articles, k=TOP_K):
    return most_popular(day, art, day.min(), cutoff_day, n_articles, k)


def recent_popular(day, art, cutoff_day, weeks, n_articles, k=TOP_K):
    return most_popular(day, art, cutoff_day - 7 * weeks, cutoff_day, n_articles, k)


def recent_items(day, cust, art, cutoff_day, customers, k):
    """Each customer's most recently bought distinct articles before the cutoff
    (most recent first, at most k). Returns a list of lists parallel to `customers`."""
    wanted = np.zeros(max(cust.max(), customers.max()) + 1, dtype=bool)
    wanted[customers] = True
    m = (day < cutoff_day) & wanted[cust]
    d, c, a = day[m], cust[m], art[m]
    # sort by customer, then most recent day first (later rows first within a day)
    pos = np.arange(len(d))
    order = np.lexsort((-pos, -d, c))
    c, a = c[order], a[order]
    starts = np.searchsorted(c, customers, side="left")
    ends = np.searchsorted(c, customers, side="right")
    out = []
    for s, e in zip(starts, ends):
        seen, rec = set(), []
        for x in a[s:e]:
            x = int(x)
            if x not in seen:
                seen.add(x)
                rec.append(x)
                if len(rec) == k:
                    break
        out.append(rec)
    return out


def fill(lists, fallback, k=TOP_K):
    """Extend each ranked list with `fallback` items (skipping duplicates) up to k items."""
    fallback = [int(x) for x in fallback]
    out = []
    for rec in lists:
        rec = list(rec)[:k]
        seen = set(rec)
        for x in fallback:
            if len(rec) == k:
                break
            if x not in seen:
                seen.add(x)
                rec.append(x)
        out.append(rec)
    return out


def repurchase(day, cust, art, cutoff_day, customers, fallback, k=TOP_K):
    """For each customer: their most recently bought distinct articles before the
    cutoff (most recent first), filled up to k with `fallback` (a ranked array).
    Returns a list of k-item lists, parallel to `customers`."""
    return fill(recent_items(day, cust, art, cutoff_day, customers, k), fallback, k)
