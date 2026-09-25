"""Sensitivity of NNDM skill to the two added layers (soil unit, distance to fault)."""
import pickle, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from sklearn.base import clone
from biabak import config as C, models as M, validation as VAL
OUT = C.OUT_DIR / "step2"
T1 = pickle.load(open(C.OUT_DIR / "step1" / "step1_tables.pkl", "rb"))
full = T1["S3_borehole_covariates"].reset_index(drop=True); full["dist_fault_km"] = full["dist_fault_km"].astype(float)
full["unsuccessful"] = 1 - full.productive
dom = np.load(OUT / "prediction_domain_xy_km.npy")
CONT0, CAT0 = list(M.CONTINUOUS), list(M.CATEGORICAL)
VARIANTS = {"without soil": (CONT0, [c for c in CAT0 if c != "soil_dominant"]),
            "without fault distance": ([c for c in CONT0 if c != "dist_fault_km"], CAT0),
            "without both": ([c for c in CONT0 if c != "dist_fault_km"], [c for c in CAT0 if c != "soil_dominant"])}
def auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1; r = rankdata(s); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rows = []
for task, d in [("regression", full[full.productive == 1].reset_index(drop=True)), ("classification", full)]:
    folds, _ = VAL.nndm_loo(d[["x_km", "y_km"]].values, dom, min_train=0.5)
    for vname, (cont, cat) in VARIANTS.items():
        M.CONTINUOUS[:] = cont; M.CATEGORICAL[:] = cat
        t0 = time.time(); preds = {}; base = np.empty(len(d))
        for f, (te, tr) in enumerate(folds):
            Xtr, Xte = d.iloc[tr], d.iloc[te]; pre = M.preprocessor().fit(Xtr); Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
            if task == "regression":
                ytr = Xtr.log10_Sc.values; base[te] = ytr.mean()
                preds.setdefault("elastic_net", np.empty(len(d)))[te] = M.SpatialENet(inner_k=5, seed=C.SEED + f).fit(Ztr, ytr, xy=Xtr[["x_km", "y_km"]].values).predict(Zte)
                preds.setdefault("random_forest", np.empty(len(d)))[te] = clone(M.regressors(C.SEED + f)["random_forest"]).fit(Ztr, ytr).predict(Zte)
            else:
                ytr = Xtr.unsuccessful.values; base[te] = ytr.mean()
                for m in ["logistic", "random_forest", "mlp"]:
                    preds.setdefault(m, np.empty(len(d)))[te] = clone(M.classifiers(C.SEED + f)[m]).fit(Ztr, ytr).predict_proba(Zte)[:, 1]
        y = d.log10_Sc.values if task == "regression" else d.unsuccessful.values
        for m, p in preds.items():
            r = dict(task=task, variant=vname, model=m, n_continuous=len(cont), n_categorical=len(cat))
            if task == "regression":
                r["r2_cv"] = 1 - np.sum((y - p) ** 2) / np.sum((y - base) ** 2)
            else:
                r["bss"] = 1 - np.mean((p - y) ** 2) / np.mean((base - y) ** 2); r["auc_excess_risk"] = auc(y, p - base)
            rows.append(r)
        print(task, vname, f"{time.time()-t0:.0f}s", flush=True)
M.CONTINUOUS[:] = CONT0; M.CATEGORICAL[:] = CAT0
pd.DataFrame(rows).to_csv(OUT / "covariate_sensitivity.csv", index=False); print("sensitivity done", flush=True)
