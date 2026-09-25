"""Step 2 analysis: skill against fold-specific baselines, uncertainty, and the
relation between skill and generalisation distance."""
from __future__ import annotations

import glob
import pickle

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from biabak import config as C

OUT = C.OUT_DIR / "step2"
rng = np.random.default_rng(C.SEED + 11)
N_BOOT = 2000

preds, metas = [], []
for f in sorted(glob.glob(str(OUT / "checkpoints" / "*.pkl"))):
    p, m = pickle.load(open(f, "rb"))
    preds.append(p); metas.append(m)
P = pd.concat(preds, ignore_index=True)
MT = pd.concat(metas, ignore_index=True)
T = {}
BLOCK = {"kmeans_8", "grid_10km", "grid_20km", "grid_35km", "grid_50km"}
DESIGN_ORDER = ["random_single", "random_repeated_100", "kmeans_8", "grid_10km", "grid_20km",
                "grid_35km", "grid_50km", "nndm_loo"]


# ---------------------------------------------------------------------------
def reg_metrics(g):
    y, p, b = g.y.values, g.pred.values, g.baseline.values
    sse, ssb = np.sum((y - p) ** 2), np.sum((y - b) ** 2)
    return dict(n=len(y), r2_cv=1 - sse / ssb, rmse=np.sqrt(np.mean((y - p) ** 2)),
                mae=np.mean(np.abs(y - p)), rmse_baseline=np.sqrt(np.mean((y - b) ** 2)),
                spearman_obs_pred=pd.Series(y).corr(pd.Series(p), method="spearman") if np.std(p) > 0 else np.nan)


def clf_metrics(g):
    y, p, b = g.y.values, g.pred.values, g.baseline.values
    out = dict(n=len(y), events=int(y.sum()))
    if 0 < y.sum() < len(y):
        out["auc"] = roc_auc_score(y, p) if np.std(p) > 0 else 0.5
        out["average_precision"] = average_precision_score(y, p)
        out["auc_excess_risk"] = roc_auc_score(y, p - b) if np.std(p - b) > 0 else 0.5
    else:
        out["auc"] = out["average_precision"] = out["auc_excess_risk"] = np.nan
    bs, bref = np.mean((p - y) ** 2), np.mean((b - y) ** 2)
    out.update(brier=bs, brier_reference=bref, bss=1 - bs / bref)
    hard = p >= 0.5
    out["sensitivity_at_0.5"] = hard[y == 1].mean() if y.sum() else np.nan
    out["accuracy_at_0.5"] = np.mean(hard == (y == 1))
    tp = np.sum(~hard & (y == 0)); fp = np.sum(~hard & (y == 1)); fn = np.sum(hard & (y == 0))
    out["f1_productive_at_0.5"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    flag = p > b                          # flagged as riskier than the training prevalence
    out["sensitivity_at_prevalence"] = flag[y == 1].mean() if y.sum() else np.nan
    out["specificity_at_prevalence"] = (~flag[y == 0]).mean()
    return out


def _fast(task, y, p, b, key):
    if task == "regression":
        ssb = np.sum((y - b) ** 2)
        return 1 - np.sum((y - p) ** 2) / ssb if ssb > 0 else np.nan
    if key == "bss":
        br = np.mean((b - y) ** 2)
        return 1 - np.mean((p - y) ** 2) / br if br > 0 else np.nan
    if key == "auc_excess_risk":
        p = p - b
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap(g, fn, key, unit, task, nb=N_BOOT):
    """Percentile interval; resampling unit is the fold for block designs and the
    borehole otherwise."""
    y, p, b = g.y.values, g.pred.values, g.baseline.values
    idx = [np.asarray(v) for v in g.groupby(unit).indices.values()]
    vals = np.empty(nb)
    for k in range(nb):
        pick = rng.integers(0, len(idx), len(idx))
        ii = np.concatenate([idx[q] for q in pick])
        vals[k] = _fast(task, y[ii], p[ii], b[ii], key)
    vals = vals[np.isfinite(vals)]
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5), np.mean(vals > 0)) if len(vals) else (np.nan,) * 3


rows = []
for (task, design, model), g in P.groupby(["task", "design", "model"]):
    if design == "random_repeated_100":
        continue
    fn = reg_metrics if task == "regression" else clf_metrics
    r = dict(task=task, design=design, model=model, **fn(g))
    unit = "fold" if design in BLOCK else "N"
    for key in (["r2_cv"] if task == "regression" else ["bss", "auc", "auc_excess_risk"]):
        lo, hi, pos = bootstrap(g.reset_index(drop=True), fn, key, unit, task)
        r[f"{key}_ci_low"], r[f"{key}_ci_high"] = lo, hi
        if key in ("r2_cv", "bss"):
            r[f"{key}_share_bootstrap_positive"] = pos
    r["bootstrap_unit"] = "held-out block" if unit == "fold" else "borehole"
    rows.append(r)

# repeated random splits: one metric per split, then the distribution
rep = []
for (task, model, fold), g in P[P.design == "random_repeated_100"].groupby(["task", "model", "fold"]):
    fn = reg_metrics if task == "regression" else clf_metrics
    rep.append(dict(task=task, model=model, split=fold, **fn(g)))
REP = pd.DataFrame(rep)
T["per_split_random_repeated"] = REP
for (task, model), g in REP.groupby(["task", "model"]):
    keys = ["r2_cv", "rmse"] if task == "regression" else ["auc", "auc_excess_risk", "bss", "sensitivity_at_0.5", "f1_productive_at_0.5", "accuracy_at_0.5"]
    r = dict(task=task, design="random_repeated_100", model=model, n=int(g.n.sum()))
    for k in keys:
        r[f"{k}_median"] = g[k].median()
        r[f"{k}_p2_5"] = g[k].quantile(0.025)
        r[f"{k}_p97_5"] = g[k].quantile(0.975)
        r[f"{k}_min"], r[f"{k}_max"] = g[k].min(), g[k].max()
    rows.append(r)
S = pd.DataFrame(rows)
S["design"] = pd.Categorical(S.design, DESIGN_ORDER, ordered=True)
S = S.sort_values(["task", "design", "model"]).reset_index(drop=True)
T["summary_by_design_model"] = S

# per-fold metrics for block designs
pf = []
for (task, design, model, fold), g in P[P.design.isin(BLOCK)].groupby(["task", "design", "model", "fold"]):
    fn = reg_metrics if task == "regression" else clf_metrics
    with np.errstate(all="ignore"):
        pf.append(dict(task=task, design=design, model=model, fold=fold, **fn(g)))
T["per_fold_block_designs"] = pd.DataFrame(pf)
T["fold_metadata"] = MT

# ---------------------------------------------------------------------------
# Generalisation distance by design, against the prediction domain
# ---------------------------------------------------------------------------
dom = np.load(OUT / "prediction_domain_xy_km.npy")
dist_rows = []
for task in ["regression", "classification"]:
    info = pickle.load(open(OUT / f"nndm_info_{task}.pkl", "rb"))
    q = np.percentile(info["Gij"], [10, 25, 50, 75, 90])
    dist_rows.append(dict(task=task, design="prediction domain (1 km cells of the Centre Region)",
                          n=len(info["Gij"]), p10=q[0], p25=q[1], median=q[2], p75=q[3], p90=q[4],
                          share_within_5km=np.mean(info["Gij"] <= 5)))
    base = P[(P.task == task) & (P.model == P[P.task == task].model.iloc[0])]
    for design in DESIGN_ORDER:
        d = base[base.design == design].dist_nearest_train_km.values
        q = np.percentile(d, [10, 25, 50, 75, 90])
        dist_rows.append(dict(task=task, design=design, n=len(d), p10=q[0], p25=q[1], median=q[2],
                              p75=q[3], p90=q[4], share_within_5km=np.mean(d <= 5)))
T["generalisation_distance"] = pd.DataFrame(dist_rows)

# ---------------------------------------------------------------------------
# Skill as a function of distance to the nearest training borehole
# ---------------------------------------------------------------------------
bins = [0, 1, 2, 5, 10, 20, 40, 200]
lab = ["0-1", "1-2", "2-5", "5-10", "10-20", "20-40", ">40"]
pool = P[P.design.isin(["random_repeated_100", "grid_10km", "grid_20km", "grid_35km", "grid_50km", "nndm_loo"])].copy()
pool["dist_bin"] = pd.cut(pool.dist_nearest_train_km, bins, labels=lab, right=False)
ds = []
for (task, model, b), g in pool.groupby(["task", "model", "dist_bin"], observed=True):
    fn = reg_metrics if task == "regression" else clf_metrics
    key = "r2_cv" if task == "regression" else "bss"
    with np.errstate(all="ignore"):
        m = fn(g)
    ids = g.N.unique()
    gg = g.reset_index(drop=True)
    idx = [np.asarray(v) for v in gg.groupby("N").indices.values()]
    y_, p_, b_ = gg.y.values, gg.pred.values, gg.baseline.values
    boots = []
    for _ in range(500):
        pick = rng.integers(0, len(idx), len(idx))
        ii = np.concatenate([idx[q] for q in pick])
        boots.append(_fast(task, y_[ii], p_[ii], b_[ii], key))
    boots = np.array(boots, float); boots = boots[np.isfinite(boots)]
    ds.append(dict(task=task, model=model, distance_bin_km=b, predictions=len(g), boreholes=len(ids),
                   events=int(g.y.sum()) if task == "classification" else np.nan,
                   skill=m[key], skill_metric=key, ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                   auc=m.get("auc", np.nan)))
T["skill_by_distance"] = pd.DataFrame(ds)

pickle.dump(T, open(OUT / "step2_tables.pkl", "wb"))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
s = S[S.task == "regression"][["design", "model", "n", "r2_cv", "r2_cv_ci_low", "r2_cv_ci_high", "rmse", "r2_cv_median", "r2_cv_p2_5", "r2_cv_p97_5"]]
print(s.round(3).to_string())
c = S[S.task == "classification"][["design", "model", "auc", "auc_ci_low", "auc_ci_high", "average_precision", "bss", "bss_ci_low", "bss_ci_high", "sensitivity_at_0.5", "sensitivity_at_prevalence", "specificity_at_prevalence", "auc_median", "auc_min", "auc_max", "accuracy_at_0.5", "f1_productive_at_0.5"]]
print(c.round(3).to_string())
print(T["generalisation_distance"].round(2).to_string())
