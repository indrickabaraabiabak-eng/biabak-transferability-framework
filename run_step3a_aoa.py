"""Step 3a: dissimilarity index and area of applicability (Meyer and Pebesma, 2021).

The feature space is the one the learners see (the fitted preprocessing
pipeline). Predictors are weighted equally because no model reached a robust,
transferable skill, so importance-based weights would be unreliable. The
threshold is derived from the training dissimilarity computed with the NNDM
folds of step 2, as the largest training value inside the upper boxplot whisker.
"""
import pickle
import numpy as np
import pandas as pd
import rasterio
import shapely
import shapefile
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import shape
from shapely.ops import transform, unary_union

from biabak import config as C, models as M, validation as VAL

OUT = C.OUT_DIR / "step3"; OUT.mkdir(parents=True, exist_ok=True)
T1 = pickle.load(open(C.OUT_DIR / "step1" / "step1_tables.pkl", "rb"))
bh = T1["S3_borehole_covariates"].reset_index(drop=True)
bh["dist_fault_km"] = bh["dist_fault_km"].astype(float)

# --- covariates on the 1 km prediction grid --------------------------------
with rasterio.open(C.RASTER_DIR / "DEM_100m.tif") as r:
    a = r.read(1); ok = np.isfinite(a) & (a > -1e30)
    rows, cols = np.where(ok[::10, ::10]); rows *= 10; cols *= 10
    xs, ys = rasterio.transform.xy(r.transform, rows, cols)
    xs, ys = np.asarray(xs), np.asarray(ys)
    to_geo = Transformer.from_crs(r.crs, C.CRS_GEOGRAPHIC, always_xy=True)
    lon, lat = to_geo.transform(xs, ys)
to_utm = Transformer.from_crs(C.CRS_GEOGRAPHIC, C.CRS_ANALYSIS, always_xy=True)
gx, gy = to_utm.transform(lon, lat)
G = pd.DataFrame({"Longitude": lon, "Latitude": lat, "x_km": np.asarray(gx) / 1000, "y_km": np.asarray(gy) / 1000})

for col, (stem, kind) in C.RASTER_COVARIATES.items():
    with rasterio.open(C.RASTER_DIR / f"{stem}.tif") as r:
        arr = r.read(1)
        t = Transformer.from_crs(C.CRS_GEOGRAPHIC, r.crs, always_xy=True)
        x, y = t.transform(G.Longitude.values, G.Latitude.values)
        rr, cc = rasterio.transform.rowcol(r.transform, x, y)
        rr, cc = np.asarray(rr), np.asarray(cc)
        inside = (rr >= 0) & (rr < r.height) & (cc >= 0) & (cc < r.width)
        v = np.full(len(G), np.nan)
        v[inside] = arr[rr[inside], cc[inside]]
        bad = ~np.isfinite(v) | (v < -1e30)
        if r.nodata is not None:
            bad |= v == r.nodata
        v[bad] = np.nan
        G[col] = v
    print(col, "missing cells:", int(G[col].isna().sum()), flush=True)

tr = Transformer.from_crs(C.CRS_GEOGRAPHIC, C.CRS_ANALYSIS, always_xy=True).transform
rs = shapefile.Reader(str(C.SOIL_SHP), encoding="utf-8")
soil = np.full(len(G), None, dtype=object)
for s, rec in zip(rs.shapes(), rs.records()):
    poly = transform(tr, shape(s.__geo_interface__))
    m = shapely.contains_xy(poly, G.x_km.values * 1000, G.y_km.values * 1000)
    soil[m & pd.isna(soil)] = rec["DOMSOI"]
G["soil_dominant"] = soil
rf = shapefile.Reader(str(C.FAULT_SHP))
faults = unary_union([transform(tr, shape(s.__geo_interface__)) for s in rf.shapes()])
pts = shapely.points(G.x_km.values * 1000, G.y_km.values * 1000)
G["dist_fault_km"] = shapely.distance(pts, faults) / 1000
cov_cols = M.CONTINUOUS + M.CATEGORICAL
complete = G[M.CONTINUOUS].notna().all(axis=1) & G[M.CATEGORICAL].notna().all(axis=1)
print("grid cells:", len(G), "complete:", int(complete.sum()), flush=True)

# --- dissimilarity index ----------------------------------------------------
pre = M.preprocessor().fit(bh)
Xb = pre.transform(bh)
Xg = pre.transform(G.loc[complete])
Db = np.sqrt(((Xb[:, None, :] - Xb[None, :, :]) ** 2).sum(-1))
dbar = Db[np.triu_indices(len(Xb), 1)].mean()
dom = np.load(C.OUT_DIR / "step2" / "prediction_domain_xy_km.npy")
folds, _ = VAL.nndm_loo(bh[["x_km", "y_km"]].values, dom, min_train=0.5)
train_di = np.array([Db[te[0], trn].min() / dbar for te, trn in folds])
q1, q3 = np.percentile(train_di, [25, 75])
whisk = q3 + 1.5 * (q3 - q1)
thr = train_di[train_di <= whisk].max()
di = cKDTree(Xb).query(Xg, k=1)[0] / dbar
G.loc[complete, "DI"] = di
G["in_AoA"] = np.where(complete, G["DI"] <= thr, np.nan)
G["dist_nearest_borehole_km"] = cKDTree(bh[["x_km", "y_km"]].values).query(G[["x_km", "y_km"]].values, k=1)[0]

bins = [0, 5, 10, 20, 40, 200]
G["dist_bin"] = pd.cut(G.dist_nearest_borehole_km, bins, right=False, labels=["0-5", "5-10", "10-20", "20-40", ">40"])
by_dist = G[complete].groupby("dist_bin", observed=True).agg(cells=("DI", "size"), share_in_AoA=("in_AoA", "mean"),
                                                             median_DI=("DI", "median")).reset_index()
summary = pd.DataFrame([dict(
    feature_space_dimension=Xb.shape[1], mean_training_distance=dbar, n_training=len(Xb),
    training_DI_median=np.median(train_di), training_DI_q75=q3, whisker=whisk, threshold=thr,
    grid_cells_total=len(G), grid_cells_complete=int(complete.sum()),
    share_domain_in_AoA=float(np.nanmean(G.loc[complete, "in_AoA"].astype(float))),
    domain_DI_median=float(np.median(di)), domain_DI_p90=float(np.percentile(di, 90)))])
lith = pre.named_transformers_["cat"].named_steps["pool"].keep_
keys = list(lith.keys())
unseen = {c: float((~G.loc[complete, c].astype(str).isin(lith[keys[i]])).mean()) for i, c in enumerate(M.CATEGORICAL)}
summary["share_cells_lithology_rare_or_unseen"] = unseen["lithology_code"]
summary["share_cells_landcover_rare_or_unseen"] = unseen["landcover_code"]
summary["share_cells_soil_rare_or_unseen"] = unseen["soil_dominant"]
train_tab = pd.DataFrame({"N": bh.N, "locality": bh.locality, "training_DI_nndm": train_di,
                          "n_training_in_fold": [len(t) for _, t in folds]})
T = {"aoa_summary": summary, "aoa_by_distance": by_dist, "aoa_training_DI": train_tab}
pickle.dump(T, open(OUT / "step3a_tables.pkl", "wb"))
G[["x_km", "y_km", "DI", "in_AoA", "dist_nearest_borehole_km"]].to_pickle(OUT / "aoa_grid.pkl")
pd.set_option("display.width", 200)
print(summary.T.to_string()); print(by_dist.round(3).to_string())
