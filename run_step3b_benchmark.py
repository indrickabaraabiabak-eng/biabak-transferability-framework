"""Step 3b: synthetic detection benchmark.

Targets are simulated at the real borehole locations with the real covariate
values, so the network geometry, the clustering and the covariate correlations
are those of the archive. For a latent variable
    Y = sqrt(a) f(X) + sqrt(b) g(s) + sqrt(1 - a - b) e,
f is a random linear combination of the ten continuous covariates, g a Gaussian
random field with exponential covariance (range 15 km), e white noise; each term
is standardised. a is the covariate share and b the spatial share of variance.
Regression uses Y at the 201 productive boreholes; classification thresholds Y
at the 224 boreholes to obtain 10.3% (observed) or 30% unsuccessful boreholes.

Learners are the correctly specified ones (ridge or logistic regression on the
same ten covariates; kriging for the spatial term), so the power obtained is an
upper bound for any learner under the same protocol.
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from pyproj import Transformer

from biabak import config as C, models as M, validation as VAL, variography as V

warnings.filterwarnings("ignore")
OUT = C.OUT_DIR / "step3"; CK = OUT / ("benchmark_checkpoints" if len(sys.argv) < 2 else "benchmark_test"); CK.mkdir(parents=True, exist_ok=True)
A_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.35]
B_LEVELS = [0.0, 0.25, 0.50]
RANGE_KM = 15.0
N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 30
PREVALENCES = [23 / 224, 0.30]
N_PERM = 199

T1 = pickle.load(open(C.OUT_DIR / "step1" / "step1_tables.pkl", "rb"))
bh = T1["S3_borehole_covariates"].reset_index(drop=True)
X = bh[M.CONTINUOUS].astype(float)
X = X.fillna(X.median()).values
Xs = (X - X.mean(0)) / X.std(0)
XY = bh[["x_km", "y_km"]].values
prod = bh.productive.values == 1
dom = np.load(C.OUT_DIR / "step2" / "prediction_domain_xy_km.npy")
F_clf, _ = VAL.nndm_loo(XY, dom, min_train=0.5)
F_reg, _ = VAL.nndm_loo(XY[prod], dom, min_train=0.5)
rs = np.random.default_rng(C.SEED)
te_clf = VAL.random_splits(224, (1 - bh.productive.values), 1, 0.3, C.SEED)
te_reg = VAL.random_splits(int(prod.sum()), None, 1, 0.3, C.SEED)
tr_ = Transformer.from_crs(C.CRS_GEOGRAPHIC, C.CRS_ANALYSIS, always_xy=True)
cx, cy = tr_.transform(C.CLUSTER_CENTRE_LATLON[1], C.CLUSTER_CENTRE_LATLON[0])
outside_reg = np.hypot(XY[prod, 0] - cx / 1000, XY[prod, 1] - cy / 1000) > C.CLUSTER_RADIUS_KM

Dfull = cdist(XY, XY)
L = np.linalg.cholesky(np.exp(-Dfull / RANGE_KM) + 1e-6 * np.eye(len(XY)))
EDGES = np.arange(0, C.HMAX_KM + C.LAG_KM, C.LAG_KM)


def pair_arrays(xy):
    i, j, h, _ = V.pairs(xy[:, 0], xy[:, 1])
    return i, j, h


PA_reg = pair_arrays(XY[prod]); PA_clf = pair_arrays(XY)
PA_out = pair_arrays(XY[prod][outside_reg])


def fast_vgm(z, tr, PA):
    i, j, h = PA
    m = np.zeros(len(z), bool); m[tr] = True
    sel = m[i] & m[j]
    hm, g, n, keep = V.experimental(z, i, j, h, EDGES, mask=sel, min_pairs=C.MIN_PAIRS)
    var = float(np.var(z[tr], ddof=1))
    if keep.sum() < 3 or var <= 0:
        return dict(model="nugget", c0=max(var, 1e-9), c=0.0, a=1.0)
    hm, g, n = hm[keep], g[keep], n[keep]
    best = None
    for a0 in (3.0, 15.0, 40.0):
        x0 = [0.5 * var, 0.5 * var, a0]
        f = lambda p: np.sqrt(n) * (g - V.model_gamma("spherical", hm, *p)) / np.maximum(V.model_gamma("spherical", hm, *p), 1e-9)
        r = least_squares(f, x0, bounds=([0, 0, 0.5], [10 * var, 10 * var, 150]), max_nfev=300)
        c = float(np.sum(r.fun ** 2))
        if best is None or c < best[0]:
            best = (c, r.x)
    c0, c, a = best[1]
    return dict(model="spherical", c0=c0, c=c, a=a)


def ok_pred(z, tr, te, D, fit):
    n = len(tr)
    G = V.model_gamma(fit["model"], D[np.ix_(tr, tr)], fit["c0"], fit["c"], fit["a"])
    A = np.ones((n + 1, n + 1)); A[:n, :n] = G; A[n, n] = 0
    B = np.vstack([V.model_gamma(fit["model"], D[np.ix_(tr, te)], fit["c0"], fit["c"], fit["a"]), np.ones((1, len(te)))])
    lam = np.linalg.lstsq(A, B, rcond=None)[0][:n]
    return lam.T @ z[tr]


def paired_p(loss_base, loss_model):
    d = loss_base - loss_model
    if np.std(d) == 0:
        return 1.0
    return float(stats.ttest_1samp(d, 0.0, alternative="greater").pvalue)


def excess_auc_p(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan
    u = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="greater")
    return u.statistic / (y.sum() * (len(y) - y.sum())), float(u.pvalue)


def evaluate_reg(y, folds, D, PA, Xp):
    n = len(y); p_r = np.empty(n); p_k = np.empty(n); b = np.empty(n)
    for te, tr in folds:
        sc = StandardScaler().fit(Xp[tr])
        m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(sc.transform(Xp[tr]), y[tr])
        p_r[te] = m.predict(sc.transform(Xp[te]))
        p_k[te] = ok_pred(y, tr, te, D, fast_vgm(y, tr, PA))
        b[te] = y[tr].mean()
    idx = np.concatenate([te for te, _ in folds])
    y_, b_ = y[idx], b[idx]
    out = {}
    for name, p in [("ridge", p_r[idx]), ("kriging", p_k[idx])]:
        lb, lm = (y_ - b_) ** 2, (y_ - p) ** 2
        out[name] = dict(r2_cv=1 - lm.sum() / lb.sum(), p=paired_p(lb, lm))
    return out


SINGLE_CLASS = [0]


def evaluate_clf(y, folds, D, PA, Xp, with_ik):
    n = len(y); p_l = np.full(n, np.nan); p_k = np.full(n, np.nan); b = np.empty(n)
    for te, tr in folds:
        b[te] = y[tr].mean()
        if y[tr].min() == y[tr].max():
            # training fold with a single class: the only defensible forecast is the
            # training prevalence; the case is counted in the output
            p_l[te] = y[tr].mean(); p_k[te] = y[tr].mean(); SINGLE_CLASS[0] += 1
            continue
        sc = StandardScaler().fit(Xp[tr])
        m = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xp[tr]), y[tr])
        p_l[te] = m.predict_proba(sc.transform(Xp[te]))[:, 1]
        if with_ik:
            yf = y.astype(float)
            p_k[te] = np.clip(ok_pred(yf, tr, te, D, fast_vgm(yf, tr, PA)), 0, 1)
    idx = np.concatenate([te for te, _ in folds])
    y_, b_ = y[idx], b[idx]
    out = {}
    for name, p in [("logistic", p_l[idx]), ("indicator_kriging", p_k[idx])]:
        if np.isnan(p).all():
            continue
        lb, lm = (b_ - y_) ** 2, (p - y_) ** 2
        auc_, pa = excess_auc_p(y_, p - b_)
        out[name] = dict(bss=1 - lm.mean() / lb.mean(), p_bss=paired_p(lb, lm), auc_excess=auc_, p_auc=pa)
    return out


def structure_p(z, PA, rng):
    i, j, h = PA
    hm, g, n, keep = V.experimental(z, i, j, h, EDGES, min_pairs=C.MIN_PAIRS)
    sims = V.permutation_envelope(z, i, j, h, EDGES, N_PERM, rng)
    short = keep & (EDGES[1:] <= 20.0)
    if short.sum() == 0:
        return np.nan
    return V.short_lag_test(g, n, sims, short)[2]


D_reg = Dfull[np.ix_(prod, prod)]
A_RUN = A_LEVELS if len(sys.argv) < 3 else [float(sys.argv[2])]
B_RUN = B_LEVELS if len(sys.argv) < 4 else [float(sys.argv[3])]
for a in A_RUN:
    for b in B_RUN:
        if a + b > 0.86:
            continue
        ck = CK / f"a{a:.2f}_b{b:.2f}.pkl"
        if ck.exists():
            continue
        t0 = time.time()
        rng = np.random.default_rng(int(1000 * a + 100 * b) + C.SEED)
        rows = []
        for rep in range(N_REP):
            w = rng.normal(size=X.shape[1])
            f = Xs @ w; f = (f - f.mean()) / f.std()
            g = L @ rng.normal(size=len(XY)); g = (g - g.mean()) / g.std()
            e = rng.normal(size=len(XY)); e = (e - e.mean()) / e.std()
            Y = np.sqrt(a) * f + np.sqrt(b) * g + np.sqrt(max(1 - a - b, 0)) * e
            yr = Y[prod]
            base = dict(a=a, b=b, rep=rep)
            for design, folds in [("nndm_loo", F_reg), ("random_single", te_reg)]:
                for model, r in evaluate_reg(yr, folds, D_reg, PA_reg, X[prod]).items():
                    rows.append(dict(base, task="regression", design=design, model=model, prevalence=np.nan, **r))
            for prev in PREVALENCES:
                yc = (Y > np.quantile(Y, 1 - prev)).astype(int)
                for design, folds in [("nndm_loo", F_clf), ("random_single", te_clf)]:
                    SINGLE_CLASS[0] = 0
                    res = evaluate_clf(yc, folds, Dfull, PA_clf, X, with_ik=(prev < 0.2))
                    for model, r in res.items():
                        rows.append(dict(base, task="classification", design=design, model=model, prevalence=prev,
                                         single_class_training_folds=SINGLE_CLASS[0], **r))
            rows.append(dict(base, task="structure", design="permutation_short_lag_0_20km", model="all productive (201)",
                             p=structure_p(yr, PA_reg, rng)))
            rows.append(dict(base, task="structure", design="permutation_short_lag_0_20km", model=f"outside cluster ({int(outside_reg.sum())})",
                             p=structure_p(yr[outside_reg], PA_out, rng)))
        pickle.dump(pd.DataFrame(rows), open(ck, "wb"))
        print(f"a={a:.2f} b={b:.2f}: {N_REP} replicates in {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
