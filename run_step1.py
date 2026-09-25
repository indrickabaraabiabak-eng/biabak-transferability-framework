"""Step 1 of the transferability audit.

Data control, covariate extraction, and spatial-structure diagnostics performed
before any prediction. Every number reported in the manuscript for these parts
is produced by this script; every table is written in full to the
supplementary workbook.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from pyproj import Transformer

from biabak import config as C
from biabak import data as D
from biabak import variography as V

OUT = C.OUT_DIR / "step1"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(C.SEED)
T = {}      # supplementary tables, written in order
KEY = {}    # key numbers quoted in the text

# ===========================================================================
# 1. Data and covariates
# ===========================================================================
full, extraction_log = D.build_dataset()

# cross-check against the covariate workbook delivered on 5 August 2026
ref = pd.read_excel(C.COVARIATE_XLSX, sheet_name="Forages_224")
pairs_cols = {"elevation_m": "elevation_mnt_m", "slope_deg": "pente_deg",
              "profile_curvature": "courbure_profil", "plan_curvature": "courbure_plan",
              "lineament_density": "densite_lineaments", "rainfall_mm_yr": "pluviometrie_mm_an",
              "aet_mm_yr": "etr_mm_an", "ndvi": "ndvi", "ndbi": "ndbi",
              "lithology_code": "geologie_code", "landcover_code": "occupation_sol_code"}
rows = []
for new, old in pairs_cols.items():
    a, b = full[new].values, ref[old].values
    both = np.isfinite(a) & np.isfinite(b)
    rows.append(dict(covariate=new, raster=C.RASTER_COVARIATES[new][0] + ".tif",
                     n_compared=int(both.sum()),
                     n_identical=int(np.sum(np.isclose(a[both], b[both], rtol=1e-5, atol=1e-6))),
                     n_missing_new=int(np.sum(~np.isfinite(a))),
                     n_missing_reference=int(np.sum(~np.isfinite(b))),
                     n_fallback=int(np.sum(extraction_log[new].str.startswith("fallback"))),
                     max_abs_difference=float(np.nanmax(np.abs(a[both] - b[both])))))
T["S2_extraction_check"] = pd.DataFrame(rows)

cols_out = (["ID", "N", "locality", "Latitude", "Longitude", "x_km", "y_km", "common_support",
             "productive", "Q_m3h", "Sc_m3hm", "T_m2s", "K_ms", "total_depth_m", "weathered_m",
             "basement_m", "static_level_m", "first_strike_m", "log10_Sc", "log10_T"]
            + list(C.RASTER_COVARIATES) + ["lithology_name", "soil_dominant", "soil_fao_unit",
                                            "soil_assignment", "dist_fault_km"])
T["S3_borehole_covariates"] = full[cols_out]
fl = extraction_log.copy()
fl.insert(0, "N", full["N"]); fl.insert(1, "locality", full["locality"])
T["S4_extraction_flags"] = fl[(fl.iloc[:, 2:] != "pixel").any(axis=1)]

# ===========================================================================
# 2. Archive control
# ===========================================================================
prod = full[full.productive == 1]
cs = full[full.common_support == 1]
cs_prod = cs[cs.productive == 1]
T["S1_populations"] = pd.DataFrame([
    dict(population="Consolidated inventory", n=len(full), productive=int(full.productive.sum()),
         unsuccessful=int(full.unsuccessful.sum()), use="classification (main)"),
    dict(population="Productive boreholes", n=len(prod), productive=len(prod), unsuccessful=0,
         use="regression (main)"),
    dict(population="Common raster support", n=len(cs), productive=int(cs.productive.sum()),
         unsuccessful=int(cs.unsuccessful.sum()), use="variography of status; support sensitivity"),
    dict(population="Common support, productive", n=len(cs_prod), productive=len(cs_prod),
         unsuccessful=0, use="variography of log10 Sc and log10 T"),
])

def describe(s, name, pop):
    s = s.dropna()
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return dict(variable=name, population=pop, n=len(s), mean=s.mean(), sd=s.std(ddof=1),
                min=s.min(), p05=q[0.05], p25=q[0.25], median=q[0.5], p75=q[0.75], p95=q[0.95],
                max=s.max(), skewness=stats.skew(s))

desc = []
for v, nm in [("Q_m3h", "discharge Q (m3/h)"), ("Sc_m3hm", "specific capacity Sc (m3/h/m)"),
              ("T_m2s", "transmissivity T (m2/s)"), ("K_ms", "equivalent K (m/s)"),
              ("log10_Sc", "log10 Sc"), ("log10_T", "log10 T")]:
    desc.append(describe(prod[v], nm, "productive (201)"))
for v, nm in [("Q_m3h", "discharge Q (m3/h)"), ("total_depth_m", "total depth (m)"),
              ("weathered_m", "weathered thickness (m)"), ("static_level_m", "static level (m)"),
              ("first_strike_m", "first water strike (m)")]:
    desc.append(describe(full[v], nm, "all (224)"))
for v, nm in [("total_depth_m", "total depth (m)")]:
    desc.append(describe(full[full.productive == 0][v], nm, "unsuccessful (23)"))
    desc.append(describe(prod[v], nm, "productive (201)"))
T["S5_descriptive_statistics"] = pd.DataFrame(desc)
KEY["median_Q_productive"] = float(prod.Q_m3h.median())
KEY["median_Sc_productive"] = float(prod.Sc_m3hm.median())
KEY["median_T_productive"] = float(prod.T_m2s.median())
KEY["n_Q_ge_0p5"] = int((full.Q_m3h >= 0.5).sum())
KEY["n_Q_ge_1"] = int((full.Q_m3h >= 1).sum())
mw = stats.mannwhitneyu(full[full.productive == 0].total_depth_m, prod.total_depth_m)
KEY["depth_dry_vs_productive_MW_p"] = float(mw.pvalue)

def mult(v, step):
    return np.isclose(np.round(v / step) * step, v, atol=1e-6)

digit = []
for pop, sub in [("all (224)", full), ("productive (201)", prod)]:
    digit.append(dict(quantity="total depth multiple of 5 m", population=pop, n=len(sub),
                      count=int(mult(sub.total_depth_m, 5).sum()),
                      percent=100 * mult(sub.total_depth_m, 5).mean()))
    digit.append(dict(quantity="total depth multiple of 1 m", population=pop, n=len(sub),
                      count=int(mult(sub.total_depth_m, 1).sum()),
                      percent=100 * mult(sub.total_depth_m, 1).mean()))
digit.append(dict(quantity="discharge multiple of 0.5 m3/h (Q>0)", population="productive (201)",
                  n=len(prod), count=int(mult(prod.Q_m3h, 0.5).sum()),
                  percent=100 * mult(prod.Q_m3h, 0.5).mean()))
digit.append(dict(quantity="discharge multiple of 1 m3/h (Q>0)", population="productive (201)",
                  n=len(prod), count=int(mult(prod.Q_m3h, 1).sum()),
                  percent=100 * mult(prod.Q_m3h, 1).mean()))
# expected share under a uniform last digit given the recording precision
T["S6_digit_preference"] = pd.DataFrame(digit)

xy = full[["x_km", "y_km"]].values
Dm = cdist(xy, xy); np.fill_diagonal(Dm, np.inf)
nn_idx = Dm.argmin(1)
nnd = pd.DataFrame({"N": full.N, "locality": full.locality, "productive": full.productive,
                    "nearest_N": full.N.values[nn_idx], "nearest_locality": full.locality.values[nn_idx],
                    "nearest_distance_km": Dm.min(1)})
T["S7_nearest_neighbour"] = nnd
KEY["nn_distance_min_m"] = float(Dm.min() * 1000)
KEY["nn_distance_median_km"] = float(np.median(Dm.min(1)))

iu = np.triu_indices(len(full), 1)
hh = Dm[iu]
close = hh < 2.0
cp = pd.DataFrame({
    "N_a": full.N.values[iu[0][close]], "locality_a": full.locality.values[iu[0][close]],
    "N_b": full.N.values[iu[1][close]], "locality_b": full.locality.values[iu[1][close]],
    "distance_m": hh[close] * 1000,
    "Q_a": full.Q_m3h.values[iu[0][close]], "Q_b": full.Q_m3h.values[iu[1][close]],
    "depth_a": full.total_depth_m.values[iu[0][close]], "depth_b": full.total_depth_m.values[iu[1][close]],
    "static_a": full.static_level_m.values[iu[0][close]], "static_b": full.static_level_m.values[iu[1][close]],
})
cp["status_discordant"] = ((cp.Q_a > 0) != (cp.Q_b > 0)).astype(int)
cp["abs_diff_log10_Sc"] = np.abs(full.log10_Sc.values[iu[0][close]] - full.log10_Sc.values[iu[1][close]])
cp["possible_duplicate"] = (cp.distance_m < C.DUPLICATE_CHECK_M).astype(int)
cp = cp.sort_values("distance_m").reset_index(drop=True)
T["S8_close_pairs_under_2km"] = cp
KEY["pairs_lt_100m"] = int((hh < 0.1).sum())
KEY["pairs_lt_500m"] = int((hh < 0.5).sum())
KEY["pairs_lt_1km"] = int((hh < 1.0).sum())
KEY["possible_duplicates"] = int(cp.possible_duplicate.sum())

# ===========================================================================
# 3. Variography
# ===========================================================================
edges = np.arange(0, C.HMAX_KM + C.LAG_KM, C.LAG_KM)
targets = {
    "log10_Sc": (cs_prod, cs_prod.log10_Sc.values, "log10 Sc, common support productive (n=196)"),
    "log10_T": (cs_prod, cs_prod.log10_T.values, "log10 T, common support productive (n=196)"),
    "status": (cs, cs.productive.values.astype(float), "productive status, common support (n=219)"),
}
targets_sens = {
    "log10_Sc_all201": (prod, prod.log10_Sc.values, "log10 Sc, all productive (n=201)"),
    "status_all224": (full, full.productive.values.astype(float), "productive status, all (n=224)"),
}

omni_rows, fit_rows, test_rows, loo_rows = [], [], [], []
store = {}
for key, (sub, z, label) in {**targets, **targets_sens}.items():
    x, y = sub.x_km.values, sub.y_km.values
    i, j, h, az = V.pairs(x, y)
    hm, g_m, n, keep = V.experimental(z, i, j, h, edges, "matheron", min_pairs=C.MIN_PAIRS)
    _, g_ch, _, _ = V.experimental(z, i, j, h, edges, "cressie_hawkins", min_pairs=C.MIN_PAIRS)
    sims = V.permutation_envelope(z, i, j, h, edges, C.N_PERMUTATIONS, rng)
    lo, med, hi = np.percentile(sims, [2.5, 50, 97.5], axis=0)
    var = float(np.var(z, ddof=1))
    for b in range(len(edges) - 1):
        omni_rows.append(dict(target=key, description=label, bin_from_km=edges[b], bin_to_km=edges[b + 1],
                              mean_lag_km=hm[b], n_pairs=int(n[b]), retained=bool(keep[b]),
                              gamma_matheron=g_m[b], gamma_cressie_hawkins=g_ch[b],
                              perm_p2_5=lo[b], perm_median=med[b], perm_p97_5=hi[b],
                              outside_envelope=bool(g_m[b] < lo[b] or g_m[b] > hi[b]),
                              sample_variance=var))
    t_glob, p_glob = V.global_deviation_test(g_m, sims, keep)
    short = keep & (edges[1:] <= 10.0)
    o_s, e_s, p_s = V.short_lag_test(g_m, n, sims, short)
    n_out = int(np.sum(((g_m < lo) | (g_m > hi)) & keep))
    test_rows.append(dict(target=key, description=label, n=len(z), sample_variance=var,
                          bins_retained=int(keep.sum()), bins_outside_95_envelope=n_out,
                          global_max_std_deviation=t_glob, global_test_p=p_glob,
                          short_lag_gamma_0_10km=o_s, short_lag_gamma_permutation_mean=e_s,
                          short_lag_one_sided_p_continuity=p_s))
    fits = {}
    for est, gg in [("matheron", g_m), ("cressie_hawkins", g_ch)]:
        for wt in ["cressie", "n"]:
            for fam in V.MODEL_FAMILIES:
                f = V.fit_model(fam, hm[keep], gg[keep], n[keep], wt, C.RANGE_BOUNDS_KM, var)
                s = V.summarise_fit(f)
                k_par = s["n_par"]; m = int(keep.sum())
                s.update(target=key, estimator=est, weighting=wt,
                         aicc_like=m * np.log(s["wsse"] / m) + 2 * k_par + 2 * k_par * (k_par + 1) / max(m - k_par - 1, 1))
                fit_rows.append(s)
                if est == "matheron" and wt == "cressie":
                    fits[fam] = f
    store[key] = dict(sub=sub, z=z, hm=hm, g=g_m, g_ch=g_ch, n=n, keep=keep, lo=lo, hi=hi,
                      med=med, var=var, fits=fits, i=i, j=j, h=h, az=az)
    if key in targets:
        xyz = np.c_[x, y]
        for fam, f in fits.items():
            r, _, _ = V.loo_ordinary_kriging(xyz, z, f)
            r.update(target=key, model=fam, nugget_to_sill=V.summarise_fit(f)["nugget_to_sill"])
            loo_rows.append(r)

T["S9_experimental_variograms_omni"] = pd.DataFrame(omni_rows)
T["S10_structure_tests"] = pd.DataFrame(test_rows)
fit_df = pd.DataFrame(fit_rows)
fit_df = fit_df[["target", "estimator", "weighting", "model", "c0", "c", "a", "sill", "nugget_to_sill",
                 "effective_range_km", "first_minimum_km", "wsse", "n_par", "aicc_like"]]
T["S11_model_fits"] = fit_df
T["S12_loo_kriging"] = pd.DataFrame(loo_rows)[["target", "model", "nugget_to_sill", "r2_loo", "rmse",
                                               "mae", "msdr", "mean_error"]]

# ---------------------------------------------------------------------------
# Directional variograms with envelopes
# ---------------------------------------------------------------------------
dir_rows, dir_fit_rows = [], []
for key in targets:
    s = store[key]
    for dname, ang in C.DIRECTIONS_DEG.items():
        m = V.direction_mask(s["az"], ang, C.ANGULAR_TOL_DEG)
        hm, g, n, keep = V.experimental(s["z"], s["i"], s["j"], s["h"], edges, mask=m, min_pairs=C.MIN_PAIRS)
        sims = V.permutation_envelope(s["z"], s["i"], s["j"], s["h"], edges, C.N_PERMUTATIONS_DIRECTIONAL, rng, mask=m)
        lo, hi = np.percentile(sims, [2.5, 97.5], axis=0)
        t_glob, p_glob = V.global_deviation_test(g, sims, keep)
        for b in range(len(edges) - 1):
            dir_rows.append(dict(target=key, direction=dname, azimuth_deg=ang, tolerance_deg=C.ANGULAR_TOL_DEG,
                                 bin_from_km=edges[b], bin_to_km=edges[b + 1], mean_lag_km=hm[b],
                                 n_pairs=int(n[b]), retained=bool(keep[b]), gamma=g[b],
                                 perm_p2_5=lo[b], perm_p97_5=hi[b],
                                 outside_envelope=bool(keep[b] and (g[b] < lo[b] or g[b] > hi[b]))))
        for fam in ["nugget", "spherical", "exponential", "cardinal_sine"]:
            f = V.summarise_fit(V.fit_model(fam, hm[keep], g[keep], n[keep], "cressie", C.RANGE_BOUNDS_KM, s["var"]))
            f.update(target=key, direction=dname, total_pairs=int(n[keep].sum()),
                     global_test_p=p_glob, bins_outside=int(np.sum(keep & ((g < lo) | (g > hi)))))
            dir_fit_rows.append(f)
        s.setdefault("dir", {})[dname] = (hm, g, n, keep, lo, hi, p_glob)
T["S13_directional_variograms"] = pd.DataFrame(dir_rows)
T["S14_directional_fits"] = pd.DataFrame(dir_fit_rows)[["target", "direction", "model", "total_pairs",
                                                          "c0", "c", "a", "nugget_to_sill", "effective_range_km",
                                                          "first_minimum_km", "wsse", "bins_outside", "global_test_p"]]

# ---------------------------------------------------------------------------
# Short-range structure (sub-kilometre pairs)
# ---------------------------------------------------------------------------
sb = np.array(C.SHORT_BINS_KM)
short_rows = []
for key in targets:
    s = store[key]
    hm, g, n, _ = V.experimental(s["z"], s["i"], s["j"], s["h"], sb)
    sims = V.permutation_envelope(s["z"], s["i"], s["j"], s["h"], sb, C.N_PERMUTATIONS, rng)
    for b in range(len(sb) - 1):
        col = sims[:, b]
        p_low = (1 + np.sum(col <= g[b])) / (1 + len(col)) if n[b] > 0 else np.nan
        short_rows.append(dict(target=key, bin_from_m=sb[b] * 1000, bin_to_m=sb[b + 1] * 1000,
                               n_pairs=int(n[b]), gamma=g[b],
                               gamma_over_variance=g[b] / s["var"] if n[b] > 0 else np.nan,
                               perm_mean=col.mean(), perm_p2_5=np.percentile(col, 2.5),
                               perm_p97_5=np.percentile(col, 97.5), one_sided_p_lower=p_low))
    # cumulative: all pairs under 500 m and under 1 km
    for cut in [0.5, 1.0]:
        m = s["h"] < cut
        dz2 = (s["z"][s["j"]] - s["z"][s["i"]]) ** 2
        g_obs = dz2[m].mean() / 2
        perm = np.array([((zp[s["j"]] - zp[s["i"]]) ** 2)[m].mean() / 2
                         for zp in (rng.permutation(s["z"]) for _ in range(C.N_PERMUTATIONS))])
        short_rows.append(dict(target=key, bin_from_m=0.0, bin_to_m=cut * 1000, n_pairs=int(m.sum()),
                               gamma=g_obs, gamma_over_variance=g_obs / s["var"], perm_mean=perm.mean(),
                               perm_p2_5=np.percentile(perm, 2.5), perm_p97_5=np.percentile(perm, 97.5),
                               one_sided_p_lower=(1 + np.sum(perm <= g_obs)) / (1 + len(perm))))
T["S15_short_range_structure"] = pd.DataFrame(short_rows)
p_dry = cs.unsuccessful.mean()
KEY["status_expected_discordance_independence"] = float(2 * p_dry * (1 - p_dry))

# ---------------------------------------------------------------------------
# Dense-cluster decomposition and proportional effect
# ---------------------------------------------------------------------------
tr = Transformer.from_crs(C.CRS_GEOGRAPHIC, C.CRS_ANALYSIS, always_xy=True)
cx, cy = tr.transform(C.CLUSTER_CENTRE_LATLON[1], C.CLUSTER_CENTRE_LATLON[0])
clu_rows, pe_rows, var_rows = [], [], []
for key in targets:
    s = store[key]
    sub = s["sub"]
    inside = np.hypot(sub.x_km.values - cx / 1000, sub.y_km.values - cy / 1000) <= C.CLUSTER_RADIUS_KM
    s["inside"] = inside
    ii, jj = inside[s["i"]], inside[s["j"]]
    for cls, m in [("both inside cluster", ii & jj), ("one inside, one outside", ii ^ jj),
                   ("both outside cluster", ~ii & ~jj)]:
        hm, g, n, keep = V.experimental(s["z"], s["i"], s["j"], s["h"], edges, mask=m, min_pairs=C.MIN_PAIRS)
        for b in range(len(edges) - 1):
            clu_rows.append(dict(target=key, pair_class=cls, bin_from_km=edges[b], bin_to_km=edges[b + 1],
                                 mean_lag_km=hm[b], n_pairs=int(n[b]), retained=bool(keep[b]), gamma=g[b]))
        s.setdefault("clu", {})[cls] = (hm, g, n, keep)
    zin, zout = s["z"][inside], s["z"][~inside]
    bf = stats.levene(zin, zout, center="median")
    var_rows.append(dict(target=key, n_inside=int(inside.sum()), n_outside=int((~inside).sum()),
                         radius_km=C.CLUSTER_RADIUS_KM, mean_inside=zin.mean(), mean_outside=zout.mean(),
                         variance_inside=zin.var(ddof=1), variance_outside=zout.var(ddof=1),
                         variance_ratio=zin.var(ddof=1) / zout.var(ddof=1),
                         brown_forsythe_stat=bf.statistic, brown_forsythe_p=bf.pvalue,
                         mann_whitney_p=stats.mannwhitneyu(zin, zout).pvalue))
    xyz = sub[["x_km", "y_km"]].values
    Dl = cdist(xyz, xyz)
    lm, lv, cnt = [], [], []
    for k in range(len(xyz)):
        nb = Dl[k] <= C.LOCAL_WINDOW_KM
        if nb.sum() >= C.LOCAL_MIN_POINTS:
            lm.append(s["z"][nb].mean()); lv.append(s["z"][nb].var(ddof=1)); cnt.append(int(nb.sum()))
    rho = stats.spearmanr(lm, lv)
    s["pe"] = (np.array(lm), np.array(lv))
    pe_rows.append(dict(target=key, window_km=C.LOCAL_WINDOW_KM, min_points=C.LOCAL_MIN_POINTS,
                        n_windows=len(lm), median_points_per_window=float(np.median(cnt)),
                        spearman_local_mean_vs_variance=rho.statistic, p_value=rho.pvalue))
T["S16_cluster_decomposition"] = pd.DataFrame(clu_rows)
T["S17_cluster_variance_comparison"] = pd.DataFrame(var_rows)
T["S18_proportional_effect"] = pd.DataFrame(pe_rows)

# ---------------------------------------------------------------------------
# Specification sensitivity: lag spacing, estimator, weighting, family
# ---------------------------------------------------------------------------
sens_rows = []
for key in targets:
    s = store[key]
    for lag in C.LAG_SENSITIVITY_KM:
        e = np.arange(0, C.HMAX_KM + lag, lag)
        for est in ["matheron", "cressie_hawkins"]:
            hm, g, n, keep = V.experimental(s["z"], s["i"], s["j"], s["h"], e, est, min_pairs=C.MIN_PAIRS)
            for wt in ["cressie", "n"]:
                for fam in ["spherical", "exponential", "cardinal_sine", "j_bessel"]:
                    f = V.summarise_fit(V.fit_model(fam, hm[keep], g[keep], n[keep], wt, C.RANGE_BOUNDS_KM, s["var"]))
                    sens_rows.append(dict(target=key, lag_km=lag, estimator=est, weighting=wt, model=fam,
                                          bins=int(keep.sum()), c0=f["c0"], c=f["c"], a=f["a"],
                                          nugget_to_sill=f["nugget_to_sill"],
                                          effective_range_km=f["effective_range_km"],
                                          first_minimum_km=f.get("first_minimum_km", np.nan), wsse=f["wsse"]))
T["S19_sensitivity_lag_estimator_weighting"] = pd.DataFrame(sens_rows)

dec_rows = []
for key in targets:
    s = store[key]
    sub = s["sub"]
    for cell in C.DECLUSTER_CELLS_KM:
        w = V.cell_declustering_weights(sub.x_km.values, sub.y_km.values, cell, C.DECLUSTER_N_ORIGINS, rng)
        pw = w[s["i"]] * w[s["j"]]
        hm, g, n, keep = V.experimental(s["z"], s["i"], s["j"], s["h"], edges, pair_w=pw, min_pairs=C.MIN_PAIRS)
        zbar = np.sum(w * s["z"]) / np.sum(w)
        vdec = np.sum(w * (s["z"] - zbar) ** 2) / np.sum(w)
        for fam in ["nugget", "spherical", "exponential", "cardinal_sine"]:
            f = V.summarise_fit(V.fit_model(fam, hm[keep], g[keep], n[keep], "cressie", C.RANGE_BOUNDS_KM, vdec))
            dec_rows.append(dict(target=key, cell_km=cell, model=fam, declustered_variance=vdec,
                                 max_weight=w.max(), min_weight=w.min(), c0=f["c0"], c=f["c"], a=f["a"],
                                 nugget_to_sill=f["nugget_to_sill"], effective_range_km=f["effective_range_km"],
                                 wsse=f["wsse"]))
        s.setdefault("dec", {})[cell] = (hm, g, n, keep)
T["S20_sensitivity_declustering"] = pd.DataFrame(dec_rows)

# ---------------------------------------------------------------------------
# Possible duplicate records: refit after keeping one record per pair < 20 m
# ---------------------------------------------------------------------------
dup_rows = []
for key in targets:
    s = store[key]
    sub = s["sub"].reset_index(drop=True)
    P = cdist(sub[["x_km", "y_km"]].values, sub[["x_km", "y_km"]].values)
    drop = set()
    for a_, b_ in zip(*np.where(np.triu(P < C.DUPLICATE_CHECK_M / 1000, 1))):
        if a_ not in drop:
            drop.add(b_)
    keepi = np.array([k not in drop for k in range(len(sub))])
    z2 = s["z"][keepi]
    x2, y2 = sub.x_km.values[keepi], sub.y_km.values[keepi]
    i2, j2, h2, _ = V.pairs(x2, y2)
    hm, g, n, keep = V.experimental(z2, i2, j2, h2, edges, min_pairs=C.MIN_PAIRS)
    for fam in ["spherical", "exponential", "cardinal_sine"]:
        f = V.summarise_fit(V.fit_model(fam, hm[keep], g[keep], n[keep], "cressie", C.RANGE_BOUNDS_KM, float(np.var(z2, ddof=1))))
        dup_rows.append(dict(target=key, records_removed=len(drop), n=len(z2), model=fam,
                             nugget_to_sill=f["nugget_to_sill"], effective_range_km=f["effective_range_km"],
                             a=f["a"], wsse=f["wsse"]))
T["S21_sensitivity_possible_duplicates"] = pd.DataFrame(dup_rows)

# ===========================================================================
# 4. Save
# ===========================================================================
import pickle
with open(OUT / "step1_store.pkl", "wb") as fh:
    pickle.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("i", "j", "h", "az")} for k, v in store.items()}, fh)
with open(OUT / "step1_tables.pkl", "wb") as fh:
    pickle.dump(T, fh)
KEY = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in KEY.items()}
(OUT / "step1_key_numbers.json").write_text(json.dumps(KEY, indent=2))
print(json.dumps(KEY, indent=2))
