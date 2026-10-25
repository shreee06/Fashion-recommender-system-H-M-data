"""Two-tower retrieval model (PyTorch, CPU).

Training example = (customer, weekly cutoff t): the input is the customer's recent
purchases strictly before t (last K distinct articles, recency-weighted) and the
targets are the articles bought in [t, t+7). The customer tower and the item tower
both output L2-normalised vectors; training uses an in-batch sampled softmax.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- examples
@dataclass
class Examples:
    customers: np.ndarray   # (n,) customer codes
    cutoff: np.ndarray      # (n,) cutoff day of each example
    items: np.ndarray       # (n, K) int64 article codes, pad = n_articles
    weights: np.ndarray     # (n, K) float32 recency weights, 0 on padding
    feats: np.ndarray       # (n, 2) float32: log1p(#purchases in lookback), weeks since last / lookback
    tgt_start: np.ndarray   # (n + 1,) offsets into tgt_items (all zero length for eval examples)
    tgt_items: np.ndarray   # flat article codes bought in [cutoff, cutoff + 7)
    max_input_day: int      # largest purchase day used as input (diagnostics / leakage test)
    min_target_day: int
    max_target_day: int


def build_examples(day, cust, art, cutoff, n_articles, K=20, tau_weeks=8.0, lookback_weeks=26,
                   customers=None, with_targets=True, limit_day=None):
    """Examples for one cutoff day.

    With `with_targets`, the examples are the customers (restricted to `customers` if given)
    who bought in [cutoff, cutoff + 7); otherwise exactly `customers`. Inputs only use
    purchases in [cutoff - lookback_weeks, cutoff). If `limit_day` is given, every input and
    target day must be < limit_day, enforced by an assertion (the validation start).
    """
    cutoff = int(cutoff)
    if limit_day is not None:
        assert cutoff + 7 <= limit_day, "target week would reach the validation start"
    n_c = int(cust.max()) + 1
    if with_targets:
        tm = (day >= cutoff) & (day < cutoff + 7)
        if customers is not None:
            allowed = np.zeros(n_c, dtype=bool)
            allowed[customers] = True
            tm &= allowed[cust]
        tc, ta = cust[tm], art[tm]
        order = np.argsort(tc, kind="stable")
        tc, ta = tc[order], ta[order]
        members = np.unique(tc)
        # remove duplicate articles within a customer's target set
        key = tc.astype(np.int64) * n_articles + ta
        _, uidx = np.unique(key, return_index=True)
        keep = np.zeros(len(tc), dtype=bool)
        keep[uidx] = True
        row_of_t = np.searchsorted(members, tc)
        tgt_items = ta[keep].astype(np.int64)
        counts = np.bincount(row_of_t[keep], minlength=len(members))
        tgt_start = np.r_[0, np.cumsum(counts)].astype(np.int64)
        min_t, max_t = (int(day[tm].min()), int(day[tm].max())) if tm.any() else (cutoff, cutoff)
    else:
        members = np.unique(np.asarray(customers))
        tgt_start = np.zeros(len(members) + 1, dtype=np.int64)
        tgt_items = np.zeros(0, dtype=np.int64)
        min_t = max_t = cutoff
    n = len(members)
    if limit_day is not None and with_targets:
        assert max_t < limit_day

    row_of = np.full(n_c, -1, dtype=np.int64)
    row_of[members] = np.arange(n)
    lo = cutoff - 7 * lookback_weeks
    wm = (day >= lo) & (day < cutoff) & (row_of[cust] >= 0)
    ridx = np.flatnonzero(wm)
    r, a, d = row_of[cust[ridx]], art[ridx].astype(np.int64), day[ridx]
    order = np.lexsort((-ridx, -d, r))  # by row, most recent day first, later rows first within a day
    r, a, d = r[order], a[order], d[order]
    n_buy = np.bincount(r, minlength=n)
    last_day = np.full(n, lo - 7, dtype=np.int64)
    if len(r):
        firsts = np.r_[True, r[1:] != r[:-1]]
        last_day[r[firsts]] = d[firsts]
    max_input = int(d.max()) if len(d) else cutoff - 1
    if limit_day is not None:
        assert max_input < limit_day
    # one entry per (row, article): the most recent
    key = r * n_articles + a
    _, uidx = np.unique(key, return_index=True)
    uidx.sort()
    r, a, d = r[uidx], a[uidx], d[uidx]
    starts = np.searchsorted(r, np.arange(n), side="left")
    rank = np.arange(len(r)) - starts[r]
    k = rank < K
    items = np.full((n, K), n_articles, dtype=np.int64)
    weights = np.zeros((n, K), dtype=np.float32)
    items[r[k], rank[k]] = a[k]
    weights[r[k], rank[k]] = np.exp(-((cutoff - d[k]) / 7.0) / tau_weeks)
    feats = np.stack([np.log1p(n_buy), np.minimum((cutoff - last_day) / 7.0, lookback_weeks) / lookback_weeks], 1).astype(np.float32)
    return Examples(members, np.full(n, cutoff), items, weights, feats, tgt_start, tgt_items, max_input, min_t, max_t)


def concat_examples(parts):
    off = np.cumsum([0] + [len(p.tgt_items) for p in parts[:-1]])
    return Examples(
        np.concatenate([p.customers for p in parts]), np.concatenate([p.cutoff for p in parts]),
        np.concatenate([p.items for p in parts]), np.concatenate([p.weights for p in parts]),
        np.concatenate([p.feats for p in parts]),
        np.concatenate([p.tgt_start[:-1] + o for p, o in zip(parts, off)] + [[sum(len(p.tgt_items) for p in parts)]]),
        np.concatenate([p.tgt_items for p in parts]),
        max(p.max_input_day for p in parts), min(p.min_target_day for p in parts), max(p.max_target_day for p in parts))


# --------------------------------------------------------------------------- model
class TwoTower(nn.Module):
    """content: dict name -> int array (n_articles,) of category codes (already 0-based).
    use_content=False gives the ID-only item tower (ablation)."""

    def __init__(self, n_articles, content, n_age_buckets=6, d=64, id_dim=64, cat_dim=16,
                 share_embeddings=True, use_content=True, hidden=128):
        super().__init__()
        self.n_articles, self.d, self.use_content = n_articles, d, use_content
        self.pad = n_articles
        self.id_emb = nn.Embedding(n_articles + 1, id_dim, padding_idx=n_articles)
        self.hist_emb = self.id_emb if share_embeddings else nn.Embedding(n_articles + 1, id_dim, padding_idx=n_articles)
        self.content_names = sorted(content)
        self.register_buffer("content_codes", torch.as_tensor(np.stack([content[k] for k in self.content_names], 1), dtype=torch.long))
        self.cat_embs = nn.ModuleList([nn.Embedding(int(content[k].max()) + 1, cat_dim if k != "index_group" else 8) for k in self.content_names])
        item_in = id_dim + (sum(e.embedding_dim for e in self.cat_embs) if use_content else 0)
        self.item_mlp = nn.Sequential(nn.Linear(item_in, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.age_emb = nn.Embedding(n_age_buckets, 8)
        self.cust_mlp = nn.Sequential(nn.Linear(id_dim + 8 + 2, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def encode_items(self, ids):
        x = [self.id_emb(ids)]
        if self.use_content:
            codes = self.content_codes[ids]
            x += [e(codes[:, j]) for j, e in enumerate(self.cat_embs)]
        return F.normalize(self.item_mlp(torch.cat(x, 1)), dim=-1)

    def encode_customers(self, items, weights, age, feats):
        w = weights / weights.sum(1, keepdim=True).clamp_min(1e-9)  # weighted mean of history embeddings
        bag = F.embedding_bag(items, self.hist_emb.weight, per_sample_weights=w, mode="sum", padding_idx=self.pad)
        return F.normalize(self.cust_mlp(torch.cat([bag, self.age_emb(age), feats], 1)), dim=-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def in_batch_loss(u, v, pos_items, temperature, log_q=None):
    """In-batch sampled softmax. Row i's positive is v[i]; other rows' items are negatives,
    except those equal to the positive article (masked). `log_q` (per batch item) subtracts
    log sampling probability from the logits (logQ correction)."""
    logits = u @ v.T / temperature
    if log_q is not None:
        logits = logits - log_q[None, :]
    same = pos_items[:, None] == pos_items[None, :]
    logits = logits.masked_fill(same & ~torch.eye(len(u), dtype=torch.bool), float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(u)))


# --------------------------------------------------------------------------- training / retrieval
def _batch(model, ex, idx, age_of_customer):
    items = torch.as_tensor(ex.items[idx])
    weights = torch.as_tensor(ex.weights[idx])
    age = torch.as_tensor(age_of_customer[ex.customers[idx]], dtype=torch.long)
    return model.encode_customers(items, weights, age, torch.as_tensor(ex.feats[idx]))


def train_epoch(model, opt, ex, age_of_customer, rng, batch_size, temperature, log_q_table=None):
    model.train()
    n = len(ex.customers)
    lens = np.diff(ex.tgt_start)
    perm = rng.permutation(n)
    # one random target per example per epoch
    pick = ex.tgt_items[ex.tgt_start[:-1] + (rng.random(n) * lens).astype(np.int64)]
    total, steps = 0.0, 0
    for s in range(0, n - batch_size + 1, batch_size):
        idx = perm[s : s + batch_size]
        pos = torch.as_tensor(pick[idx])
        u = _batch(model, ex, idx, age_of_customer)
        v = model.encode_items(pos)
        lq = None if log_q_table is None else torch.as_tensor(log_q_table[pick[idx]], dtype=torch.float32)
        loss = in_batch_loss(u, v, pos, temperature, lq)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.detach())
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def retrieve(model, ex, age_of_customer, catalog, top=100, exclude=None, chunk=1000):
    """Exact top-`top` articles (from `catalog` article codes) per example by dot product.
    `exclude`: optional scipy csr (n_examples x n_articles) of articles to skip.
    Returns (codes (n, top) int64, scores (n, top) float32), best first."""
    model.eval()
    cat = torch.as_tensor(catalog, dtype=torch.long)
    V = torch.cat([model.encode_items(cat[s : s + 20000]) for s in range(0, len(cat), 20000)])
    col_of = None
    if exclude is not None:
        col_of = np.full(model.n_articles, -1, dtype=np.int64)
        col_of[catalog] = np.arange(len(catalog))
    n = len(ex.customers)
    codes = np.zeros((n, top), dtype=np.int64)
    scores = np.zeros((n, top), dtype=np.float32)
    for s in range(0, n, chunk):
        idx = np.arange(s, min(n, s + chunk))
        sc = _batch(model, ex, idx, age_of_customer) @ V.T
        if exclude is not None:
            sub = exclude[s : s + len(idx)].tocoo()
            cols = col_of[sub.col]
            ok = cols >= 0
            sc[torch.as_tensor(sub.row[ok]), torch.as_tensor(cols[ok])] = float("-inf")
        val, pos = sc.topk(top, dim=1)
        codes[idx] = catalog[pos.numpy()]
        scores[idx] = val.numpy()
    return codes, scores
