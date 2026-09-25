"""Experimental variography, admissible model families and diagnostics.

Model families (all positive definite in two dimensions):
  nugget          gamma(h) = C0                                 (h > 0)
  spherical       C0 + C [1.5 h/a - 0.5 (h/a)^3], h <= a; C0 + C beyond
  exponential     C0 + C [1 - exp(-h/a)]            practical range 3a
  cardinal sine   C0 + C [1 - sin(h/a)/(h/a)]      hole effect, valid up to R^3
  J-Bessel        C0 + C [1 - J0(h/a)]             hole effect, valid in R^2
Fitting uses weighted least squares with Cressie (1985) weights N(h)/gamma(h)^2
as the primary choice, and N(h) weights as a sensitivity option.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import j0

# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------

def pairs(x, y):
    i, j = np.triu_indices(len(x), 1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    h = np.hypot(dx, dy)
    az = np.degrees(np.arctan2(dx, dy)) % 180.0   # azimuth from north, axial
    return i, j, h, az


def direction_mask(az, centre, tol):
    d = np.abs(((az - centre) + 90.0) % 180.0 - 90.0)
    return d <= tol


# ---------------------------------------------------------------------------
# Experimental estimators
# ---------------------------------------------------------------------------

def experimental(z, i, j, h, edges, estimator="matheron", pair_w=None, min_pairs=1, mask=None):
    dz = z[j] - z[i]
    sel = np.ones_like(h, bool) if mask is None else mask
    k = np.digitize(h, edges) - 1
    nb = len(edges) - 1
    ok = sel & (k >= 0) & (k < nb) & (h > 0)
    kk = k[ok]
    n = np.bincount(kk, minlength=nb).astype(float)
    hm = np.bincount(kk, weights=h[ok], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        hm = hm / n
        if estimator == "matheron":
            w = np.ones(ok.sum()) if pair_w is None else pair_w[ok]
            num = np.bincount(kk, weights=w * dz[ok] ** 2, minlength=nb)
            den = np.bincount(kk, weights=w, minlength=nb)
            g = num / (2 * den)
        elif estimator == "cressie_hawkins":
            s = np.bincount(kk, weights=np.sqrt(np.abs(dz[ok])), minlength=nb) / n
            g = (s ** 4) / (0.457 + 0.494 / n) / 2.0
        else:
            raise ValueError(estimator)
    keep = n >= min_pairs
    return hm, g, n, keep


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def model_gamma(name, h, c0, c=0.0, a=1.0):
    h = np.asarray(h, float)
    if name == "nugget":
        g = np.full_like(h, c0)
    elif name == "spherical":
        r = np.minimum(h / a, 1.0)
        g = c0 + c * (1.5 * r - 0.5 * r ** 3)
    elif name == "exponential":
        g = c0 + c * (1 - np.exp(-h / a))
    elif name == "cardinal_sine":
        u = h / a
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(u > 1e-12, np.sin(u) / np.where(u > 1e-12, u, 1.0), 1.0)
        g = c0 + c * (1 - s)
    elif name == "j_bessel":
        g = c0 + c * (1 - j0(h / a))
    else:
        raise ValueError(name)
    return np.where(h == 0, 0.0, g)


MODEL_FAMILIES = ["nugget", "spherical", "exponential", "cardinal_sine", "j_bessel"]


def fit_model(name, hm, g, n, weighting="cressie", range_bounds=(0.5, 150.0), var=None):
    hm, g, n = map(np.asarray, (hm, g, n))
    if name == "nugget":
        w = n if weighting == "n" else n / np.maximum(g, 1e-12) ** 2
        c0 = float(np.sum(w * g) / np.sum(w))
        if weighting == "cressie":          # iterate: weights depend on the model
            for _ in range(20):
                w = n / c0 ** 2
                c0 = float(np.sum(w * g) / np.sum(w))
        res = model_gamma(name, hm, c0) - g
        wsse = float(np.sum((n if weighting == "n" else n / c0 ** 2) * res ** 2))
        return dict(model=name, c0=c0, c=0.0, a=np.nan, wsse=wsse, n_par=1)
    v = var if var is not None else float(np.nanmean(g))
    best = None
    lo, hi = range_bounds
    starts_a = [1.0, 3.0, 8.0, 15.0, 30.0, 60.0]
    starts_c = [0.1, 0.5, 0.9]
    for a0 in starts_a:
        for f in starts_c:
            x0 = [v * (1 - f), v * f, a0]

            def resid(p):
                gm = model_gamma(name, hm, *p)
                if weighting == "n":
                    return np.sqrt(n) * (g - gm)
                return np.sqrt(n) * (g - gm) / np.maximum(gm, 1e-9)

            try:
                r = least_squares(resid, x0, bounds=([0, 0, lo], [10 * v, 10 * v, hi]),
                                  method="trf", max_nfev=4000)
            except Exception:
                continue
            wsse = float(np.sum(r.fun ** 2))
            if best is None or wsse < best[0] - 1e-12:
                best = (wsse, r.x)
    wsse, (c0, c, a) = best
    return dict(model=name, c0=float(c0), c=float(c), a=float(a), wsse=wsse, n_par=3)


def summarise_fit(f):
    sill = f["c0"] + f["c"]
    out = dict(f)
    out["sill"] = sill
    out["nugget_to_sill"] = f["c0"] / sill if sill > 0 else np.nan
    if f["model"] == "spherical":
        out["effective_range_km"] = f["a"]
    elif f["model"] == "exponential":
        out["effective_range_km"] = 3 * f["a"]
    elif f["model"] == "cardinal_sine":
        out["effective_range_km"] = np.nan
        out["first_minimum_km"] = 4.4934 * f["a"]    # first root of tan(u) = u
    elif f["model"] == "j_bessel":
        out["effective_range_km"] = np.nan
        out["first_minimum_km"] = 3.8317 * f["a"]    # first zero of J1
    else:
        out["effective_range_km"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Permutation envelope (random relabelling of values over fixed locations)
# ---------------------------------------------------------------------------

def permutation_envelope(z, i, j, h, edges, n_perm, rng, mask=None):
    sims = np.empty((n_perm, len(edges) - 1))
    for b in range(n_perm):
        zp = rng.permutation(z)
        _, g, _, _ = experimental(zp, i, j, h, edges, mask=mask)
        sims[b] = g
    return sims


def global_deviation_test(obs, sims, keep):
    """Two-sided global test: max standardised deviation over retained bins."""
    s = sims[:, keep]
    o = obs[keep]
    mu, sd = s.mean(0), s.std(0, ddof=1)
    t_obs = np.max(np.abs(o - mu) / sd)
    t_sim = np.max(np.abs(s - mu) / sd, axis=1)
    p = (1 + np.sum(t_sim >= t_obs)) / (1 + len(t_sim))
    return float(t_obs), float(p)


def short_lag_test(obs_g, obs_n, sims, keep_bins):
    """One-sided test of spatial continuity: is the pair-weighted semivariance
    over the retained short-lag bins lower than under random relabelling?"""
    w = obs_n[keep_bins]
    o = np.sum(w * obs_g[keep_bins]) / np.sum(w)
    s = (sims[:, keep_bins] * w).sum(1) / np.sum(w)
    p = (1 + np.sum(s <= o)) / (1 + len(s))
    return float(o), float(s.mean()), float(p)


# ---------------------------------------------------------------------------
# Cell declustering
# ---------------------------------------------------------------------------

def cell_declustering_weights(x, y, cell, n_origins, rng):
    w = np.zeros(len(x))
    for _ in range(n_origins):
        ox, oy = rng.uniform(0, cell, 2)
        ix = np.floor((x - x.min() + ox) / cell).astype(int)
        iy = np.floor((y - y.min() + oy) / cell).astype(int)
        key = ix * 100000 + iy
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        wi = 1.0 / cnt[inv]
        w += wi / wi.sum() * len(x)
    return w / n_origins


# ---------------------------------------------------------------------------
# Ordinary kriging (global neighbourhood) and leave-one-out cross-validation
# ---------------------------------------------------------------------------

def ok_matrix(xy, fit):
    from scipy.spatial.distance import cdist
    D = cdist(xy, xy)
    G = model_gamma(fit["model"], D, fit["c0"], fit["c"], fit["a"])
    n = len(xy)
    A = np.ones((n + 1, n + 1))
    A[:n, :n] = G
    A[n, n] = 0.0
    return A, D


def loo_ordinary_kriging(xy, z, fit):
    from scipy.spatial.distance import cdist
    n = len(z)
    pred = np.empty(n)
    var = np.empty(n)
    base = np.empty(n)
    idx = np.arange(n)
    for k in range(n):
        m = idx != k
        xs, zs = xy[m], z[m]
        G = model_gamma(fit["model"], cdist(xs, xs), fit["c0"], fit["c"], fit["a"])
        A = np.ones((n, n)); A[:n - 1, :n - 1] = G; A[n - 1, n - 1] = 0.0
        g0 = model_gamma(fit["model"], cdist(xs, xy[k:k + 1]).ravel(), fit["c0"], fit["c"], fit["a"])
        b = np.append(g0, 1.0)
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        lam, mu = sol[:-1], sol[-1]
        pred[k] = lam @ zs
        var[k] = lam @ g0 + mu
        base[k] = zs.mean()
    e = z - pred
    r2 = 1 - np.sum(e ** 2) / np.sum((z - base) ** 2)
    msdr = float(np.mean(e ** 2 / np.maximum(var, 1e-12)))
    return dict(r2_loo=float(r2), rmse=float(np.sqrt(np.mean(e ** 2))),
                mae=float(np.mean(np.abs(e))), msdr=msdr,
                mean_error=float(e.mean())), pred, var
