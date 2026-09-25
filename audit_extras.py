"""Analyses added after the internal audit:
M6: excess-risk AUC against pooled and fold-wise AUC on the real fold structures;
V1: distance between each design's distance distribution and the prediction domain;
D7: sensitivity of the domain distance distribution to grid spacing;
D6: composition of the 26-dimensional feature space."""
import pickle, numpy as np, pandas as pd, rasterio, warnings
warnings.filterwarnings("ignore")
from scipy.stats import rankdata, norm, wasserstein_distance, ks_2samp
from scipy.spatial import cKDTree
from pyproj import Transformer
from biabak import config as C, models as M
O2 = C.OUT_DIR / "step2"; OUT = C.OUT_DIR / "audit"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(C.SEED + 404)

def auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# ---- M6 ----
rows = []
for design in ["nndm_loo", "grid_20km", "kmeans_8", "random_repeated_100"]:
    P = pickle.load(open(O2 / "checkpoints" / f"classification__{design}.pkl", "rb"))[0]
    P = P[P.model == "training_prevalence"].reset_index(drop=True)
    y, b, fold = P.y.values.astype(int), P.baseline.values, P.fold.values
    if design == "random_repeated_100":        # one split only, to keep a single pooled set
        m = fold == 0; y, b, fold = y[m], b[m], fold[m]
    for auc_true in [0.5, 0.6, 0.7, 0.8]:
        mu = np.sqrt(2) * norm.ppf(auc_true)
        res = {"pooled": [], "excess": [], "foldwise_mean": [], "folds_defined": []}
        for _ in range(500):
            z = rng.normal(size=len(y)) + mu * y
            p = np.clip(b + 0.02 * z, 1e-4, 1 - 1e-4)
            res["pooled"].append(auc(y, p)); res["excess"].append(auc(y, p - b))
            fw = [auc(y[fold == f], p[fold == f]) for f in np.unique(fold)]
            fw = [v for v in fw if np.isfinite(v)]
            res["foldwise_mean"].append(np.mean(fw) if fw else np.nan); res["folds_defined"].append(len(fw))
        r = dict(design=design, n=len(y), events=int(y.sum()), folds=len(np.unique(fold)), true_auc=auc_true,
                 corr_baseline_outcome=float(np.corrcoef(b, y)[0, 1]))
        for k in ["pooled", "excess", "foldwise_mean"]:
            v = np.array(res[k], float)
            r[f"{k}_mean"] = np.nanmean(v); r[f"{k}_bias"] = np.nanmean(v) - auc_true
            r[f"{k}_sd"] = np.nanstd(v); r[f"{k}_defined_share"] = np.mean(np.isfinite(v))
        r["folds_with_both_classes"] = float(np.mean(res["folds_defined"]))
        rows.append(r)
AUCSIM = pd.DataFrame(rows)

# ---- V1 ----
info = pickle.load(open(O2 / "nndm_info_classification.pkl", "rb")); G = info["Gij"]
dist_rows = []
for design in ["random_single", "random_repeated_100", "kmeans_8", "grid_10km", "grid_20km", "grid_35km", "grid_50km", "nndm_loo"]:
    P = pickle.load(open(O2 / "checkpoints" / f"classification__{design}.pkl", "rb"))[0]
    d = P[P.model == "logistic"].dist_nearest_train_km.values
    dist_rows.append(dict(design=design, wasserstein_km=wasserstein_distance(d, G), ks_statistic=ks_2samp(d, G).statistic))
DIST = pd.DataFrame(dist_rows)

# ---- D7 ----
T1 = pickle.load(open(C.OUT_DIR / "step1" / "step1_tables.pkl", "rb")); bh = T1["S3_borehole_covariates"]
tree = cKDTree(bh[["x_km", "y_km"]].values)
grid_rows = []
with rasterio.open(C.RASTER_DIR / "DEM_100m.tif") as r:
    a = r.read(1); ok = np.isfinite(a) & (a > -1e30)
    t = Transformer.from_crs(r.crs, C.CRS_ANALYSIS, always_xy=True)
    for step, lab in [(5, "0.5 km"), (10, "1 km"), (20, "2 km"), (50, "5 km")]:
        rows_, cols_ = np.where(ok[::step, ::step]); rows_ *= step; cols_ *= step
        xs, ys = rasterio.transform.xy(r.transform, rows_, cols_); px, py = t.transform(np.asarray(xs), np.asarray(ys))
        d = tree.query(np.c_[np.asarray(px) / 1000, np.asarray(py) / 1000], k=1)[0]
        q = np.percentile(d, [10, 50, 90])
        grid_rows.append(dict(grid=lab, cells=len(d), p10_km=q[0], median_km=q[1], p90_km=q[2], share_within_5km=np.mean(d <= 5)))
GRID = pd.DataFrame(grid_rows)

# ---- D6 ----
bh2 = bh.copy(); bh2["dist_fault_km"] = bh2["dist_fault_km"].astype(float)
pre = M.preprocessor().fit(bh2)
names = list(M.CONTINUOUS)
oh = pre.named_transformers_["cat"].named_steps["onehot"]
for var, cats in zip(M.CATEGORICAL, oh.categories_):
    names += [f"{var} = {c}" for c in cats]
FEAT = pd.DataFrame({"feature": names, "type": ["standardized continuous"] * len(M.CONTINUOUS) + ["one-hot level (0/1)"] * (len(names) - len(M.CONTINUOUS)),
                     "weight_in_dissimilarity_index": 1.0})
pickle.dump(dict(auc_sim=AUCSIM, design_distance=DIST, grid=GRID, features=FEAT), open(OUT / "audit_tables.pkl", "wb"))
pd.set_option("display.width", 220)
print(AUCSIM[["design", "folds", "events", "true_auc", "corr_baseline_outcome", "pooled_mean", "excess_mean", "foldwise_mean_mean", "folds_with_both_classes"]].round(3).to_string())
print(DIST.round(3).to_string()); print(GRID.round(2).to_string()); print(len(FEAT), FEAT.feature.tolist())
