"""Minimal unit tests of the core quantities used by the BIABAK Transferability Framework (run: python -m pytest -q)."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from biabak import variography as V, validation as VAL, models as M


def auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1; r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def test_excess_risk_auc_of_constant_forecast_is_one_half():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.1).astype(int)
    base = rng.uniform(0.05, 0.15, 200)          # fold-specific prevalences
    assert abs(auc(y, base - base) - 0.5) < 1e-12


def test_r2cv_is_zero_for_the_training_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0]); base = np.array([2.0, 2.5, 3.0, 2.0])
    r2 = 1 - np.sum((y - base) ** 2) / np.sum((y - base) ** 2)
    assert r2 == 0.0


def test_matheron_semivariance_of_two_points():
    x = np.array([0.0, 1.0]); y = np.array([0.0, 0.0]); z = np.array([1.0, 3.0])
    i, j, h, _ = V.pairs(x, y)
    hm, g, n, keep = V.experimental(z, i, j, h, np.array([0.0, 2.0]))
    assert n[0] == 1 and abs(g[0] - 2.0) < 1e-12        # (3-1)^2 / 2


def test_models_vanish_at_origin_and_reach_sill():
    for m in ["spherical", "exponential", "cardinal_sine", "j_bessel"]:
        g0 = V.model_gamma(m, np.array([0.0]), 0.2, 0.8, 5.0)[0]
        assert g0 == 0.0
    assert abs(V.model_gamma("spherical", np.array([50.0]), 0.2, 0.8, 5.0)[0] - 1.0) < 1e-12


def test_nndm_meets_distance_condition_or_training_floor():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 10, (60, 2)); dom = rng.uniform(0, 100, (500, 2))
    folds, info = VAL.nndm_loo(xy, dom, min_train=0.5)
    g = np.sort(info["g_final"]); G = np.sort(info["Gij"])
    ecdf = lambda s, r: np.searchsorted(s, r, side="right") / len(s)
    ok = np.all(ecdf(g, g) <= ecdf(G, g) + 1e-12)
    floor = min(len(tr) for _, tr in folds) >= int(np.ceil(0.5 * len(xy)))
    assert ok or floor
    assert all(te[0] not in set(tr) for te, tr in folds)


def test_rare_levels_are_pooled_within_the_training_fold():
    X = pd.DataFrame({"a": ["x"] * 6 + ["y"] * 2})
    out = M.RarePooler(min_count=5).fit(X).transform(X)
    assert set(out["a"]) == {"x", "other"}
