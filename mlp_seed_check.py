"""Robustness of the MLP ranking signal under NNDM to the network's random seed."""
import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from sklearn.base import clone
from biabak import config as C, models as M, validation as VAL
OUT = C.OUT_DIR / "step2"
T1 = pickle.load(open(C.OUT_DIR / "step1" / "step1_tables.pkl", "rb"))
d = T1["S3_borehole_covariates"].reset_index(drop=True); d["unsuccessful"] = 1 - d.productive
dom = np.load(OUT / "prediction_domain_xy_km.npy")
folds, _ = VAL.nndm_loo(d[["x_km", "y_km"]].values, dom, min_train=0.5)
def auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1; r = rankdata(s); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rows = []
for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    p = np.empty(len(d)); b = np.empty(len(d))
    for te, tr in folds:
        Xtr, Xte = d.iloc[tr], d.iloc[te]
        pre = M.preprocessor().fit(Xtr)
        m = clone(M.classifiers(seed)["mlp"]).set_params(random_state=seed).fit(pre.transform(Xtr), Xtr.unsuccessful.values)
        p[te] = m.predict_proba(pre.transform(Xte))[:, 1]; b[te] = Xtr.unsuccessful.mean()
    y = d.unsuccessful.values
    rows.append(dict(seed=seed, auc_excess_risk=auc(y, p - b), auc=auc(y, p),
                     bss=1 - np.mean((p - y) ** 2) / np.mean((b - y) ** 2), sensitivity_at_0_5=(p[y == 1] >= 0.5).mean()))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT / "mlp_seed_check.csv", index=False)
print(R.describe().round(3).to_string())
