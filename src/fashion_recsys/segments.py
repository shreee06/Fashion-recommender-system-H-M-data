"""Segment popularity: most-popular articles of the last weeks before the cutoff within
a customer segment (age bucket, or the customer's dominant index group)."""
import numpy as np

AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+", "unknown"]


def age_bucket(age):
    """Bucket codes 0..5 for an array of ages (NaN/negative = unknown, code 5)."""
    age = np.asarray(age, dtype=np.float64)
    out = np.full(len(age), 5, dtype=np.int64)
    ok = np.isfinite(age) & (age >= 0)
    out[ok] = np.digitize(age[ok], [25, 35, 45, 55])
    return out


def _top(counts, k):
    return np.lexsort((np.arange(len(counts)), -counts))[:k]


def customer_segment_popular(day, cust, art, segment_of_customer, n_segments, cutoff_day, weeks, n_articles, k=100):
    """Top-k articles per segment, counting window purchases of customers in each segment.
    `segment_of_customer[c]` is the segment code of customer c. Returns {segment: array}."""
    m = (day >= cutoff_day - 7 * weeks) & (day < cutoff_day)
    seg = segment_of_customer[cust[m]]
    counts = np.bincount(seg * n_articles + art[m], minlength=n_segments * n_articles).reshape(n_segments, n_articles)
    return {s: _top(counts[s], k) for s in range(n_segments)}


def dominant_group(day, cust, art, article_group, n_groups, cutoff_day, n_customers):
    """Per customer, the article group (e.g. index_group) with most purchases before the
    cutoff; -1 for customers without history."""
    m = day < cutoff_day
    counts = np.bincount(cust[m].astype(np.int64) * n_groups + article_group[art[m]], minlength=n_customers * n_groups)
    counts = counts.reshape(n_customers, n_groups)
    dom = counts.argmax(axis=1)
    dom[counts.sum(axis=1) == 0] = -1
    return dom


def group_popular(day, art, article_group, n_groups, cutoff_day, weeks, n_articles, k=100):
    """Top-k articles per article group by purchases in the window. Returns {group: array}."""
    m = (day >= cutoff_day - 7 * weeks) & (day < cutoff_day)
    counts = np.bincount(art[m], minlength=n_articles)
    out = {}
    for g in range(n_groups):
        c = np.where(article_group == g, counts, -1)  # other groups can never rank above
        top = _top(c, k)
        out[g] = top[c[top] >= 0]
    return out
