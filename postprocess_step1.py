"""Post-processing of step 1 tables: multiple-testing correction and fit flags."""
import pickle, numpy as np, pandas as pd
from biabak import config as C
OUT = C.OUT_DIR / "step1"
T = pickle.load(open(OUT / "step1_tables.pkl", "rb"))

def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, k in enumerate(o):
        run = max(run, (m - r) * p[k]); adj[k] = min(run, 1.0)
    return adj

d = T["S14_directional_fits"]
g = d.groupby(["target", "direction"], as_index=False)[["global_test_p"]].first()
g["global_test_p_holm_12_tests"] = holm(g.global_test_p.values)
T["S14_directional_fits"] = d.merge(g[["target", "direction", "global_test_p_holm_12_tests"]], on=["target", "direction"])
t = T["S10_structure_tests"]
main = t.target.isin(["log10_Sc", "log10_T", "status"])
t.loc[main, "global_test_p_holm_3_targets"] = holm(t.loc[main, "global_test_p"].values)
T["S10_structure_tests"] = t
lo, hi = C.RANGE_BOUNDS_KM
for k in ["S11_model_fits", "S19_sensitivity_lag_estimator_weighting", "S20_sensitivity_declustering",
          "S14_directional_fits", "S21_sensitivity_possible_duplicates"]:
    f = T[k]
    f["range_parameter_at_bound"] = np.where(f.a.isna(), False, (f.a <= lo + 1e-3) | (f.a >= hi - 1e-3))
    T[k] = f
pe = T["S18_proportional_effect"]
pe["note"] = np.where(pe.target == "status",
    "Not interpretable: for a binary variable the local variance p(1-p) is a deterministic function of the local mean.",
    "Overlapping windows share boreholes; the p value assumes independence and is therefore optimistic.")
T["S18_proportional_effect"] = pe
pickle.dump(T, open(OUT / "step1_tables.pkl", "wb"))
print(g.round(3).to_string())
print(t[["target","global_test_p","global_test_p_holm_3_targets","short_lag_one_sided_p_continuity"]].round(3).to_string())
