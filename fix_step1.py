"""Remove the Cressie-Hawkins estimator for the binary status target: the
robust estimator assumes a continuous, approximately Gaussian variable and is
not defined meaningfully for an indicator."""
import pickle, numpy as np
from biabak import config as C
OUT = C.OUT_DIR / "step1"
T = pickle.load(open(OUT / "step1_tables.pkl", "rb")); S = pickle.load(open(OUT / "step1_store.pkl", "rb"))
st = lambda s: s.str.startswith("status")
T["S9_experimental_variograms_omni"].loc[st(T["S9_experimental_variograms_omni"].target), "gamma_cressie_hawkins"] = np.nan
for k in ["S11_model_fits", "S19_sensitivity_lag_estimator_weighting"]:
    f = T[k]; T[k] = f[~(st(f.target) & (f.estimator == "cressie_hawkins"))].reset_index(drop=True)
for k in S:
    if k.startswith("status"): S[k]["g_ch"] = np.full_like(S[k]["g_ch"], np.nan)
pickle.dump(T, open(OUT / "step1_tables.pkl", "wb")); pickle.dump(S, open(OUT / "step1_store.pkl", "wb"))
print(len(T["S19_sensitivity_lag_estimator_weighting"]))
