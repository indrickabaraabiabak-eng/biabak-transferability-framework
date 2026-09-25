"""One-sided bootstrap tests on the primary estimand (NNDM leave-one-out),
with Holm adjustment across the models of each task and metric."""
import glob, pickle, numpy as np, pandas as pd
from scipy.stats import rankdata
from biabak import config as C
OUT = C.OUT_DIR / "step2"
rng = np.random.default_rng(C.SEED + 23)
T = pickle.load(open(OUT / "step2_tables.pkl", "rb"))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(run, 1)
    return adj

def auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

rows = []
for task in ["regression", "classification"]:
    p, _ = pickle.load(open(OUT / "checkpoints" / f"{task}__nndm_loo.pkl", "rb"))
    for m, g in p.groupby("model"):
        if m in ("training_mean", "training_prevalence"): continue
        y, q, b = g.y.values, g.pred.values, g.baseline.values
        n = len(y); B = 4000
        stats = {"r2_cv": [], "bss": [], "auc_excess_risk": []}
        for _ in range(B):
            i = rng.integers(0, n, n)
            yy, qq, bb = y[i], q[i], b[i]
            if task == "regression":
                stats["r2_cv"].append(1 - np.sum((yy - qq) ** 2) / np.sum((yy - bb) ** 2))
            else:
                stats["bss"].append(1 - np.mean((qq - yy) ** 2) / np.mean((bb - yy) ** 2))
                stats["auc_excess_risk"].append(auc(yy, qq - bb))
        keys = ["r2_cv"] if task == "regression" else ["bss", "auc_excess_risk"]
        for k in keys:
            v = np.array(stats[k]); v = v[np.isfinite(v)]
            null = 0.5 if k == "auc_excess_risk" else 0.0
            if task == "regression": est = 1 - np.sum((y - q) ** 2) / np.sum((y - b) ** 2)
            elif k == "bss": est = 1 - np.mean((q - y) ** 2) / np.mean((b - y) ** 2)
            else: est = auc(y, q - b)
            rows.append(dict(task=task, design="nndm_loo", model=m, metric=k, estimate=est,
                             ci_low=np.percentile(v, 2.5), ci_high=np.percentile(v, 97.5),
                             null_value=null, one_sided_p=(1 + np.sum(v <= null)) / (1 + len(v)), bootstrap_draws=len(v)))
R = pd.DataFrame(rows)
R["p_holm_within_task_metric"] = R.groupby(["task", "metric"]).one_sided_p.transform(lambda s: holm(s.values))
T["primary_nndm_tests"] = R
pickle.dump(T, open(OUT / "step2_tables.pkl", "wb"))
pd.set_option("display.width", 200)
print(R.round(3).to_string())
