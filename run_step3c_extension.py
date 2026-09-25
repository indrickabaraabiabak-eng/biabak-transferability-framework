"""Benchmark extension requested by the audit:
(1) more replicates for the null scenario and for covariate-only scenarios (ridge and logistic learners),
(2) sensitivity of spatial detection to the range of the simulated field (5 and 40 km).
Binomial (Wilson) intervals are computed in the analysis step."""
import sys, time, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
src = open("run_step3b_benchmark.py").read().split("D_reg = Dfull[np.ix_(prod, prod)]")[0]
sys.argv = ["x"]; ns = {}; exec(compile(src, "bench", "exec"), ns)
g = ns
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
OUT = g["OUT"] / "extension_checkpoints"; OUT.mkdir(parents=True, exist_ok=True)
X, Xs, prod, XY, Dfull = g["X"], g["Xs"], g["prod"], g["XY"], g["Dfull"]
D_reg = Dfull[np.ix_(prod, prod)]

def reg_fast(y, folds, Xp):
    n = len(y); p = np.empty(n); b = np.empty(n)
    for te, tr in folds:
        sc = StandardScaler().fit(Xp[tr])
        p[te] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(sc.transform(Xp[tr]), y[tr]).predict(sc.transform(Xp[te]))
        b[te] = y[tr].mean()
    idx = np.concatenate([te for te, _ in folds]); y_, b_, p_ = y[idx], b[idx], p[idx]
    lb, lm = (y_ - b_) ** 2, (y_ - p_) ** 2
    return dict(r2_cv=1 - lm.sum() / lb.sum(), p=g["paired_p"](lb, lm))

def clf_fast(y, folds, Xp):
    n = len(y); p = np.empty(n); b = np.empty(n)
    for te, tr in folds:
        b[te] = y[tr].mean()
        if y[tr].min() == y[tr].max():
            p[te] = y[tr].mean(); continue
        sc = StandardScaler().fit(Xp[tr])
        p[te] = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xp[tr]), y[tr]).predict_proba(sc.transform(Xp[te]))[:, 1]
    idx = np.concatenate([te for te, _ in folds]); y_, b_, p_ = y[idx], b[idx], p[idx]
    lb, lm = (b_ - y_) ** 2, (p_ - y_) ** 2
    auc_, pa = g["excess_auc_p"](y_, p_ - b_)
    return dict(bss=1 - lm.mean() / lb.mean(), p_bss=g["paired_p"](lb, lm), auc_excess=auc_, p_auc=pa)

# ---- part 1: covariate-only scenarios, more replicates ----
PLAN = [(0.0, 300), (0.05, 100), (0.10, 100), (0.20, 100), (0.35, 100)]
for a, nrep in PLAN:
    ck = OUT / f"cov_a{a:.2f}.pkl"
    if ck.exists(): continue
    t0 = time.time(); rng = np.random.default_rng(int(1000 * a) + 7777 + g["C"].SEED); rows = []
    for rep in range(nrep):
        w = rng.normal(size=X.shape[1]); f = Xs @ w; f = (f - f.mean()) / f.std()
        e = rng.normal(size=len(XY)); e = (e - e.mean()) / e.std()
        Y = np.sqrt(a) * f + np.sqrt(1 - a) * e
        for design, folds in [("nndm_loo", g["F_reg"]), ("random_single", g["te_reg"])]:
            rows.append(dict(a=a, b=0.0, range_km=np.nan, rep=rep, task="regression", design=design, model="ridge", prevalence=np.nan, **reg_fast(Y[prod], folds, X[prod])))
        for prev in g["PREVALENCES"]:
            yc = (Y > np.quantile(Y, 1 - prev)).astype(int)
            for design, folds in [("nndm_loo", g["F_clf"]), ("random_single", g["te_clf"])]:
                rows.append(dict(a=a, b=0.0, range_km=np.nan, rep=rep, task="classification", design=design, model="logistic", prevalence=prev, **clf_fast(yc, folds, X)))
    pickle.dump(pd.DataFrame(rows), open(ck, "wb")); print(f"covariate a={a}: {nrep} reps in {time.time()-t0:.0f}s", flush=True)

# ---- part 2: spatial range sensitivity (a = 0) ----
for rng_km in [5.0, 40.0]:
    L = np.linalg.cholesky(np.exp(-Dfull / rng_km) + 1e-6 * np.eye(len(XY)))
    for b in [0.25, 0.50]:
        ck = OUT / f"range{rng_km:.0f}_b{b:.2f}.pkl"
        if ck.exists(): continue
        t0 = time.time(); rng = np.random.default_rng(int(100 * b) + int(rng_km) + 99 + g["C"].SEED); rows = []
        for rep in range(40):
            gg = L @ rng.normal(size=len(XY)); gg = (gg - gg.mean()) / gg.std()
            e = rng.normal(size=len(XY)); e = (e - e.mean()) / e.std()
            Y = np.sqrt(b) * gg + np.sqrt(1 - b) * e; yr = Y[prod]
            n = len(yr); pk = np.empty(n); bb = np.empty(n)
            for te, tr in g["F_reg"]:
                pk[te] = g["ok_pred"](yr, tr, te, D_reg, g["fast_vgm"](yr, tr, g["PA_reg"])); bb[te] = yr[tr].mean()
            lb, lm = (yr - bb) ** 2, (yr - pk) ** 2
            rows.append(dict(a=0.0, b=b, range_km=rng_km, rep=rep, task="regression", design="nndm_loo", model="kriging", r2_cv=1 - lm.sum() / lb.sum(), p=g["paired_p"](lb, lm)))
            rows.append(dict(a=0.0, b=b, range_km=rng_km, rep=rep, task="structure", design="permutation_short_lag_0_20km", model="all productive (201)", p=g["structure_p"](yr, g["PA_reg"], rng)))
            rows.append(dict(a=0.0, b=b, range_km=rng_km, rep=rep, task="structure", design="permutation_short_lag_0_20km", model=f"outside cluster ({int(g['outside_reg'].sum())})", p=g["structure_p"](yr[g["outside_reg"]], g["PA_out"], rng)))
        pickle.dump(pd.DataFrame(rows), open(ck, "wb")); print(f"range {rng_km} km b={b}: 40 reps in {time.time()-t0:.0f}s", flush=True)
print("extension done", flush=True)
