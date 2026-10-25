import numpy as np
import pytest
import torch

from fashion_recsys.two_tower import (
    TwoTower, build_examples, concat_examples, in_batch_loss, retrieve, train_epoch,
)

N_ART = 40


def content():
    a = np.arange(N_ART)
    return {"product_type": a % 5, "colour": a % 7, "department": a % 3, "index_group": a % 2}


def toy_data(n_cust=400, n_groups=20, seed=0):
    """Customer c belongs to group g: bought article g on day 5, buys article 20+g on day 15."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, n_cust)
    c = np.arange(n_cust)
    day = np.r_[np.full(n_cust, 5), np.full(n_cust, 15)]
    cust = np.r_[c, c]
    art = np.r_[g, 20 + g]
    return day, cust, art, g


def make_model(seed=0, **kw):
    torch.manual_seed(seed)
    return TwoTower(N_ART, content(), d=16, id_dim=16, cat_dim=4, hidden=32, **kw)


def train(model, ex, epochs, seed=0, lr=0.01, bs=100, temp=0.1):
    rng = np.random.default_rng(seed)
    age = np.zeros(1000, dtype=np.int64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return [train_epoch(model, opt, ex, age, rng, bs, temp) for _ in range(epochs)]


# ---------- shapes and normalisation ----------
def test_shapes_and_normalisation():
    day, cust, art, _ = toy_data()
    ex = build_examples(day, cust, art, 14, N_ART, K=5, tau_weeks=4)
    m = make_model()
    idx = np.arange(32)
    u = m.encode_customers(torch.as_tensor(ex.items[idx]), torch.as_tensor(ex.weights[idx]),
                           torch.zeros(32, dtype=torch.long), torch.as_tensor(ex.feats[idx]))
    v = m.encode_items(torch.arange(N_ART))
    assert u.shape == (32, 16) and v.shape == (N_ART, 16)
    assert torch.allclose(u.norm(dim=1), torch.ones(32), atol=1e-5)
    assert torch.allclose(v.norm(dim=1), torch.ones(N_ART), atol=1e-5)
    assert m.n_params() > 0 and ex.items.shape == (400, 5)


def test_id_only_tower_has_fewer_params():
    assert make_model(use_content=False).n_params() < make_model(use_content=True).n_params()


# ---------- examples: inputs, weights, targets ----------
def test_example_inputs_weights_and_targets_hand_checked():
    # customer 0: article 3 on day 7 (age 1 week at cutoff 14), article 4 on day 0 (2 weeks), target article 9 on day 15
    day = np.array([7, 0, 15, 15])
    cust = np.array([0, 0, 0, 1])
    art = np.array([3, 4, 9, 9])
    ex = build_examples(day, cust, art, 14, N_ART, K=3, tau_weeks=2.0)
    assert ex.customers.tolist() == [0, 1]
    assert ex.items[0].tolist() == [3, 4, N_ART]  # most recent first, padded with n_articles
    assert ex.weights[0].tolist() == pytest.approx([np.exp(-0.5), np.exp(-1.0), 0.0], rel=1e-6)
    assert ex.items[1].tolist() == [N_ART] * 3  # no history -> empty bag
    assert ex.tgt_items[ex.tgt_start[0] : ex.tgt_start[1]].tolist() == [9]
    assert ex.feats[1].tolist() == [0.0, 1.0]  # no purchases, "weeks since last" at the cap
    assert ex.feats[0][0] == pytest.approx(np.log1p(2))


def test_k_keeps_most_recent_distinct():
    day = np.array([1, 2, 3, 4, 5, 15])
    cust = np.zeros(6, dtype=int)
    art = np.array([1, 2, 1, 3, 4, 9])  # article 1 bought twice -> counted once, at its latest day
    ex = build_examples(day, cust, art, 14, N_ART, K=3, tau_weeks=2.0)
    assert ex.items[0].tolist() == [4, 3, 1]


# ---------- leakage ----------
def test_no_input_or_target_on_or_after_validation_start():
    val_start = 21
    rng = np.random.default_rng(1)
    day = rng.integers(0, 40, 600)  # includes validation-week and later rows
    cust = rng.integers(0, 30, 600)
    art = rng.integers(0, N_ART, 600)
    parts = [build_examples(day, cust, art, val_start - 7 * k, N_ART, K=5, limit_day=val_start) for k in (1, 2)]
    ex = concat_examples(parts)
    assert ex.max_input_day < val_start and ex.max_target_day < val_start
    # the very last target week ends the day before validation starts
    assert parts[0].max_target_day <= val_start - 1
    # items are only ones bought before each example's cutoff
    for p in parts:
        c = int(p.cutoff[0])
        for i, cc in enumerate(p.customers):
            allowed = set(art[(cust == cc) & (day < c)].tolist()) | {N_ART}
            assert set(p.items[i].tolist()) <= allowed
            targets = set(p.tgt_items[p.tgt_start[i] : p.tgt_start[i + 1]].tolist())
            assert targets <= set(art[(cust == cc) & (day >= c) & (day < c + 7)].tolist())


def test_cutoff_reaching_validation_is_rejected():
    day, cust, art, _ = toy_data()
    with pytest.raises(AssertionError):
        build_examples(day, cust, art, 18, N_ART, limit_day=21)  # target week 18..24 overlaps validation


def test_future_rows_do_not_change_examples():
    day, cust, art, _ = toy_data()
    a = build_examples(day, cust, art, 14, N_ART, K=4)
    day2, cust2, art2 = np.r_[day, [21, 22, 30]], np.r_[cust, [0, 1, 2]], np.r_[art, [11, 12, 13]]
    b = build_examples(day2, cust2, art2, 14, N_ART, K=4)
    assert np.array_equal(a.items, b.items) and np.array_equal(a.tgt_items, b.tgt_items)


# ---------- loss ----------
def test_in_batch_loss_masks_duplicate_positives():
    u = torch.eye(3)
    v = torch.eye(3)
    pos = torch.tensor([0, 0, 1])  # rows 0 and 1 share the positive article -> not negatives of each other
    l_masked = in_batch_loss(u[[0, 0, 1]], v[[0, 0, 1]], pos, 1.0)
    l_unmasked = in_batch_loss(u[[0, 0, 1]], v[[0, 0, 1]], torch.tensor([0, 5, 1]), 1.0)
    assert float(l_masked) < float(l_unmasked)


def test_log_q_correction_changes_loss():
    u, v = F_norm(torch.randn(8, 4)), F_norm(torch.randn(8, 4))
    pos = torch.arange(8)
    assert float(in_batch_loss(u, v, pos, 0.1)) != float(in_batch_loss(u, v, pos, 0.1, torch.randn(8)))


def F_norm(x):
    return torch.nn.functional.normalize(x, dim=-1)


# ---------- determinism ----------
def test_determinism_with_fixed_seeds():
    day, cust, art, _ = toy_data()
    ex = build_examples(day, cust, art, 14, N_ART, K=3)
    runs = []
    for _ in range(2):
        m = make_model(seed=3)
        losses = train(m, ex, epochs=2, seed=5)
        runs.append((losses, [p.detach().clone() for p in m.parameters()]))
    assert runs[0][0] == runs[1][0]
    assert all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[1][1]))


# ---------- learns an obvious pattern ----------
def test_learns_group_pattern():
    day, cust, art, g = toy_data()
    ex = build_examples(day, cust, art, 14, N_ART, K=3, tau_weeks=4)
    m = make_model(seed=1)
    losses = train(m, ex, epochs=40, seed=1)
    assert losses[-1] < 0.5 * losses[0]
    catalog = np.arange(20, 40)  # candidate targets only
    codes, _ = retrieve(m, ex, np.zeros(1000, dtype=np.int64), catalog, top=3)
    assert (codes[:, 0] == 20 + g[ex.customers]).mean() > 0.95
