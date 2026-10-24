"""Item-item collaborative filtering from co-purchase baskets.

Similarity is cosine on a binary customer x article matrix built ONLY from
purchases in [cutoff - window_weeks, cutoff). To keep the matrix tractable only
the `max_items` most popular articles of the window are modelled (default 20,000)
and each article keeps its `max_neighbors` most similar neighbours (default 100).
Customers are scored from their most recent K distinct purchased (modelled) articles.
"""
import numpy as np
import scipy.sparse as sp


class ItemCF:
    def __init__(self, window_weeks=26, max_items=20_000, max_neighbors=100):
        self.window_weeks = window_weeks
        self.max_items = max_items
        self.max_neighbors = max_neighbors

    def fit(self, day, cust, art, cutoff_day, n_articles, train_customers=None,
            recency_tau_weeks=None, chunk_rows=100_000):
        """Build the similarity from window purchases strictly before `cutoff_day`.

        `train_customers` (codes) restricts the baskets used; None = everyone.
        `recency_tau_weeks`: if set, a customer's (customer, article) entry is weighted by
        exp(-age_in_weeks / tau) where age is measured from the cutoff to the customer's
        most recent purchase of that article; None gives binary entries.
        The Gram matrix is accumulated over blocks of `chunk_rows` customers to bound memory."""
        self.recency_tau_weeks = recency_tau_weeks
        self.cutoff_day = int(cutoff_day)
        self.window_start = self.cutoff_day - 7 * self.window_weeks
        self.n_articles = n_articles
        m = (day >= self.window_start) & (day < self.cutoff_day)
        if train_customers is not None:
            keep = np.zeros(max(cust.max(), np.max(train_customers)) + 1, dtype=bool)
            keep[train_customers] = True
            m &= keep[cust]
        c, a = cust[m], art[m]

        counts = np.bincount(a, minlength=n_articles)
        order = np.lexsort((np.arange(n_articles), -counts))
        items = order[: self.max_items]
        items = items[counts[items] > 0]
        self.items = items  # article codes, most popular first
        self.item_pos = np.full(n_articles, -1, dtype=np.int64)
        self.item_pos[items] = np.arange(len(items))

        pos = self.item_pos[a]
        ok = pos >= 0
        c, pos = c[ok], pos[ok]
        d = day[m][ok]
        _, crow = np.unique(c, return_inverse=True)
        n_rows = int(crow.max()) + 1 if len(crow) else 0
        # one entry per (customer, article): keep the most recent purchase day
        key = crow.astype(np.int64) * len(items) + pos
        o = np.lexsort((d, key))
        ks = key[o]
        last = np.r_[ks[1:] != ks[:-1], True] if len(ks) else np.zeros(0, dtype=bool)
        o = o[last]
        crow, pos, d = crow[o], pos[o], d[o]
        if recency_tau_weeks is None:
            w = np.ones(len(d), dtype=np.float32)
        else:
            w = np.exp(-((self.cutoff_day - d) / 7.0) / recency_tau_weeks).astype(np.float32)
        X = sp.csr_matrix((w, (crow, pos)), shape=(n_rows, len(items)))
        C = None
        for s0 in range(0, n_rows, chunk_rows):
            Xc = X[s0 : s0 + chunk_rows]
            part = (Xc.T @ Xc).tocsr()
            C = part if C is None else C + part
        if C is None:
            C = sp.csr_matrix((len(items), len(items)), dtype=np.float32)
        n_i = C.diagonal()  # sum of squared entries per article
        C.setdiag(0)
        C.eliminate_zeros()
        inv = 1.0 / np.sqrt(np.maximum(n_i, 1e-12))
        C = sp.diags(inv) @ C @ sp.diags(inv)
        self.sim = self._prune(C.tocsr().astype(np.float32))
        self.n_train_customers = X.shape[0]
        return self

    def _prune(self, S):
        M = self.max_neighbors
        if S.shape[0] == 0:
            return S
        indptr, indices, data = [0], [], []
        for i in range(S.shape[0]):
            lo, hi = S.indptr[i], S.indptr[i + 1]
            idx, val = S.indices[lo:hi], S.data[lo:hi]
            if len(val) > M:
                top = np.argpartition(-val, M - 1)[:M]
                idx, val = idx[top], val[top]
            indices.append(idx)
            data.append(val)
            indptr.append(indptr[-1] + len(val))
        return sp.csr_matrix((np.concatenate(data), np.concatenate(indices), indptr), shape=S.shape)

    def _baskets(self, day, cust, art, customers):
        """Window purchases of `customers` as (row in customers, item position, day, order)."""
        row_of = np.full(max(cust.max(), customers.max()) + 1, -1, dtype=np.int64)
        row_of[customers] = np.arange(len(customers))
        m = (day >= self.window_start) & (day < self.cutoff_day) & (row_of[cust] >= 0)
        m_idx = np.flatnonzero(m)
        pos = self.item_pos[art[m_idx]]
        keep = pos >= 0
        m_idx, pos = m_idx[keep], pos[keep]
        return row_of[cust[m_idx]], pos, day[m_idx], m_idx

    def recommend(self, day, cust, art, customers, k_seeds=5, n=100, exclude_seen=True, batch=5000):
        """Top-n article codes per customer (may be fewer than n; callers fill with popularity)."""
        customers = np.asarray(customers)
        rows, pos, d, orig = self._baskets(day, cust, art, customers)
        # most recent first within customer; one entry per (customer, item)
        order = np.lexsort((-orig, -d, rows))
        rows, pos = rows[order], pos[order]
        key = rows * len(self.items) + pos
        _, first = np.unique(key, return_index=True)
        first.sort()
        rows, pos = rows[first], pos[first]
        starts = np.searchsorted(rows, np.arange(len(customers)), side="left")
        rank = np.arange(len(rows)) - starts[rows]
        seed = rank < k_seeds
        N = len(self.items)
        shape = (len(customers), N)
        Q = sp.csr_matrix((np.ones(seed.sum(), dtype=np.float32), (rows[seed], pos[seed])), shape=shape)
        Seen = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, pos)), shape=shape)
        out = []
        for s in range(0, len(customers), batch):
            sc = (Q[s : s + batch] @ self.sim).tocsr()
            if exclude_seen:
                sc = sc - sc.multiply(Seen[s : s + batch])
                sc.eliminate_zeros()
            for i in range(sc.shape[0]):
                lo, hi = sc.indptr[i], sc.indptr[i + 1]
                idx, val = sc.indices[lo:hi], sc.data[lo:hi]
                if len(val) > n:
                    top = np.argpartition(-val, n - 1)[:n]
                    idx, val = idx[top], val[top]
                o = np.lexsort((idx, -val))
                out.append([int(x) for x in self.items[idx[o]]])
        return out
