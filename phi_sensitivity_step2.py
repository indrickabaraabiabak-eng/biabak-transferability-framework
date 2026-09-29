"""Step 2: out-of-fold predictions under every validation design.

Outputs one long table of predictions per task, with the distance from each
test borehole to its nearest training borehole, plus fold metadata.
Checkpoints are written per design so the run can resume.
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from biabak import config as C
from biabak import models as M
from biabak import validation as VAL

warnings.filterwarnings("ignore")
OUT = C.OUT_DIR / "step2" / "phi_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
CK = OUT / "checkpoints"; CK.mkdir(exist_ok=True)

T1 = pickle.load(open(C.OUT_DIR / "step1" / "step1_tables.pkl", "rb"))
full = T1["S3_borehole_covariates"].reset_index(drop=True)
full["dist_fault_km"] = full["dist_fault_km"].astype(float)

# Reuse the exact prediction domain from the primary Step 2 analysis.
XY_DOMAIN = np.load(C.OUT_DIR / "step2" / "prediction_domain_xy_km.npy")

TASKS = {
    "regression": full[full.productive == 1].reset_index(drop=True),
    "classification": full.reset_index(drop=True),
}
origin = full[["x_km", "y_km"]].min().values - 1e-6

PHI_VALUES = {"P90": 80.511, "P95": 93.359, "P100": 128.405}


def designs(task, d):
    xy = d[["x_km", "y_km"]].values
    out = {}
    for label, phi in PHI_VALUES.items():
        folds, info = VAL.nndm_loo(xy, XY_DOMAIN, phi=phi, min_train=0.5)
        dname = f"nndm_phi_{label.lower()}"
        out[dname] = folds
        pickle.dump(info, open(OUT / f"nndm_info_{task}_{label.lower()}.pkl", "wb"))
    return out

def run_fold(task, d, te, tr, seed):
    Xtr, Xte = d.iloc[tr], d.iloc[te]
    xy_tr, xy_te = Xtr[["x_km", "y_km"]].values, Xte[["x_km", "y_km"]].values
    res = {}
    if task == "regression":
        ytr = Xtr.log10_Sc.values
        res["training_mean"] = np.full(len(te), ytr.mean())
        fit = M.fit_variogram_fold(xy_tr, ytr)
        res["ordinary_kriging"] = M.ordinary_kriging(xy_tr, ytr, xy_te, fit)
        res["inverse_distance"] = M.idw(xy_tr, ytr, xy_te)
        pre = M.preprocessor().fit(Xtr)
        Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
        en = M.SpatialENet(inner_k=5, seed=seed).fit(Ztr, ytr, xy=xy_tr)
        res["elastic_net"] = en.predict(Zte)
        for name, est in M.regressors(seed).items():
            res[name] = clone(est).fit(Ztr, ytr).predict(Zte)
        base = ytr.mean()
    else:
        ytr = Xtr.unsuccessful.values if "unsuccessful" in Xtr else 1 - Xtr.productive.values
        res["training_prevalence"] = np.full(len(te), ytr.mean())
        fit = M.fit_variogram_fold(xy_tr, ytr.astype(float))
        res["indicator_kriging"] = np.clip(M.ordinary_kriging(xy_tr, ytr.astype(float), xy_te, fit), 0, 1)
        res["inverse_distance"] = np.clip(M.idw(xy_tr, ytr.astype(float), xy_te), 0, 1)
        pre = M.preprocessor().fit(Xtr)
        Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
        for name, est in M.classifiers(seed).items():
            res[name] = clone(est).fit(Ztr, ytr).predict_proba(Zte)[:, 1]
        base = ytr.mean()
    dnn = cKDTree(xy_tr).query(xy_te, k=1)[0]
    return res, base, dnn, fit


for task, d in TASKS.items():
    d = d.copy()
    d["unsuccessful"] = 1 - d.productive
    t0 = time.time()
    D = designs(task, d)
    print(task, {k: len(v) for k, v in D.items()}, f"designs built in {time.time() - t0:.0f}s", flush=True)
    for dname, folds in D.items():
        ck = CK / f"{task}__{dname}.pkl"
        if ck.exists():
            continue
        t0 = time.time()
        rows, meta = [], []
        for f, (te, tr) in enumerate(folds):
            res, base, dnn, fit = run_fold(task, d, te, tr, C.SEED + f)
            y = d.log10_Sc.values[te] if task == "regression" else d.unsuccessful.values[te]
            for m, p in res.items():
                for k_, idx in enumerate(te):
                    rows.append((task, dname, f, m, int(d.N.values[idx]), float(y[k_]), float(p[k_]),
                                 float(base), float(dnn[k_])))
            meta.append(dict(task=task, design=dname, fold=f, n_train=len(tr), n_test=len(te),
                             test_events=int(d.unsuccessful.values[te].sum()) if task == "classification" else np.nan,
                             train_events=int(d.unsuccessful.values[tr].sum()) if task == "classification" else np.nan,
                             median_test_to_train_km=float(np.median(dnn)),
                             variogram_model=fit["model"], vg_c0=fit["c0"], vg_c=fit["c"], vg_a=fit["a"],
                             vg_nugget_to_sill=fit["c0"] / (fit["c0"] + fit["c"]) if (fit["c0"] + fit["c"]) > 0 else np.nan))
        pr = pd.DataFrame(rows, columns=["task", "design", "fold", "model", "N", "y", "pred", "baseline", "dist_nearest_train_km"])
        pickle.dump((pr, pd.DataFrame(meta)), open(ck, "wb"))
        print(f"  {task} {dname}: {len(folds)} folds in {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
