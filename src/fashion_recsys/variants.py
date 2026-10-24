"""Variant candidates: other articles sharing the product_code (e.g. other colours) of
articles a customer bought recently, ranked by recent popularity.

Uses only purchases before the cutoff (for popularity and the customer's recent items)
and static article metadata (product_code).
"""
import numpy as np


class VariantCandidates:
    def __init__(self, pop_weeks=4):
        self.pop_weeks = pop_weeks

    def fit(self, day, art, cutoff_day, product_code, n_articles):
        """`product_code[a]` is the product code of article code a. Siblings must have at
        least one sale in the `pop_weeks` weeks before the cutoff to be offered."""
        m = (day >= cutoff_day - 7 * self.pop_weeks) & (day < cutoff_day)
        self.counts = np.bincount(art[m], minlength=n_articles)
        by_product = {}
        # rank by recent popularity, ties -> lower article code
        for a in np.lexsort((np.arange(n_articles), -self.counts)):
            if self.counts[a] > 0:
                by_product.setdefault(product_code[a], []).append(int(a))
        self.by_product = by_product
        self.product_code = product_code
        return self

    def recommend(self, recents, k_seeds=5, n=100):
        """`recents`: per customer, most-recent-first distinct article codes (before the
        cutoff). Seeds = first k_seeds; the customer's own recents are not offered."""
        out = []
        for rec in recents:
            have = set(rec)
            res, seen = [], set()
            for a in rec[:k_seeds]:
                for b in self.by_product.get(self.product_code[a], ()):
                    if b not in have and b not in seen:
                        seen.add(b)
                        res.append(b)
                        if len(res) == n:
                            break
                if len(res) == n:
                    break
            out.append(res)
        return out
