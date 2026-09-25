"""Validation designs and generalisation distances.

Each design returns a list of folds; a fold is (test_idx, train_idx). For
leave-one-out designs the training set of each fold can exclude neighbours of
the test point (NNDM, Mila et al., 2022).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def random_splits(n, strat, n_rep, test_size, seed):
    idx = np.arange(n)
    if strat is not None:
        ss = StratifiedShuffleSplit(n_splits=n_rep, test_size=test_size, random_state=seed)
        it = ss.split(idx, strat)
    else:
        ss = ShuffleSplit(n_splits=n_rep, test_size=test_size, random_state=seed)
        it = ss.split(idx)
    return [(te, tr) for tr, te in it]


def kmeans_blocks(xy, k, seed):
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(xy).labels_
    return [(np.where(lab == b)[0], np.where(lab != b)[0]) for b in range(k)], lab


def grid_blocks(xy, size_km, origin):
    ix = np.floor((xy[:, 0] - origin[0]) / size_km).astype(int)
    iy = np.floor((xy[:, 1] - origin[1]) / size_km).astype(int)
    key = ix * 10000 + iy
    labs = np.unique(key, return_inverse=True)[1]
    return [(np.where(labs == b)[0], np.where(labs != b)[0]) for b in range(labs.max() + 1)], labs


def nndm_loo(xy_train, xy_pred, phi=None, min_train=0.5):
    """Nearest-neighbour distance matching leave-one-out (Mila et al., 2022).

    Starting from ordinary LOO, the nearest remaining neighbour of a test point
    is excluded from its training set, one at a time, until the empirical
    distribution of test-to-training nearest distances no longer exceeds the
    distribution of prediction-to-training distances at any distance <= phi.
    """
    n = len(xy_train)
    Gij = cKDTree(xy_train).query(xy_pred, k=1)[0]
    Gij_sorted = np.sort(Gij)
    if phi is None:
        phi = float(Gij.max())
    D = cdist(xy_train, xy_train)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)            # neighbours of each point, nearest first
    k = np.zeros(n, int)                     # number of excluded neighbours per point
    g = D[np.arange(n), order[:, 0]]         # current test-to-training distance
    max_excl = int(np.floor((1 - min_train) * n))

    def ecdf(sorted_vals, r):
        return np.searchsorted(sorted_vals, r, side="right") / len(sorted_vals)

    it = 0
    while True:
        it += 1
        gs = np.sort(g)
        cand = gs[gs <= phi]
        if len(cand) == 0:
            break
        diff = ecdf(gs, cand) - ecdf(Gij_sorted, cand)
        bad = np.where(diff > 1e-12)[0]
        if len(bad) == 0:
            break
        p = None
        for b in bad:
            # smallest distance at which the CV distribution is still too close;
            # take a point at that distance that can still exclude a neighbour
            r = cand[b]
            pts = [q for q in np.where(np.isclose(g, r))[0] if k[q] + 1 < max_excl]
            if pts:
                p = pts[0]
                break
        if p is None:
            break
        k[p] += 1
        g[p] = D[p, order[p, k[p]]]
        if it > 200000:
            break
    folds = []
    for i in range(n):
        excl = set(order[i, :k[i]].tolist()) | {i}
        tr = np.array([j for j in range(n) if j not in excl])
        folds.append((np.array([i]), tr))
    return folds, dict(Gij=Gij, g_final=g, n_excluded=k, phi=phi, iterations=it)


def nearest_train_distance(xy, folds):
    """Distance from each test point to its nearest training point, per fold."""
    out = np.full(len(xy), np.nan)
    rows = []
    for f, (te, tr) in enumerate(folds):
        d = cKDTree(xy[tr]).query(xy[te], k=1)[0]
        rows.append((f, te, d))
    return rows
