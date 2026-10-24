import numpy as np
import pytest

from fashion_recsys.evaluation import bootstrap_ci, paired_bootstrap


def vals():
    return np.random.default_rng(1).random(500) ** 3


def test_ci_contains_point_estimate():
    m, lo, hi = bootstrap_ci(vals(), n_boot=1000, seed=3)
    assert lo <= m <= hi and m == pytest.approx(vals().mean())


def test_reproducible_with_same_seed():
    assert bootstrap_ci(vals(), seed=5) == bootstrap_ci(vals(), seed=5)
    assert bootstrap_ci(vals(), seed=5) != bootstrap_ci(vals(), seed=6)


def test_paired_self_difference_is_exactly_zero():
    r = paired_bootstrap(vals(), vals())
    assert r["diff"] == 0.0 and r["ci_lo"] == 0.0 and r["ci_hi"] == 0.0
    assert r["excludes_zero"] is False


def test_paired_detects_clear_difference():
    a = vals()
    r = paired_bootstrap(a + 0.1, a)
    assert r["diff"] == pytest.approx(0.1) and r["excludes_zero"] is True
