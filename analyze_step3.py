"""Step 3 analysis: detection power of the protocol, empirical false-positive
rates, where the observed results sit among the simulated ones, and figures."""
import glob
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biabak import config as C

OUT = C.OUT_DIR / "step3"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})

# ---------------------------------------------------------------------------
# Same evaluators applied to the real targets (exact comparability)
# ---------------------------------------------------------------------------
src = open("run_step3b_benchmark.py").read().split("D_reg = Dfull[np.ix_(prod, prod)]")[0]
sys.argv = ["x"]
ns = {}
exec(compile(src, "bench", "exec"), ns)
bh, prod, XY, X = ns["bh"], ns["prod"], ns["XY"], ns["X"]
D_reg = ns["Dfull"][np.ix_(prod, prod)]
y_real_reg = bh.log10_Sc.values[prod]
y_real_clf = (1 - bh.productive.values).astype(int)
real = []
for design, folds in [("nndm_loo", ns["F_reg"]), ("random_single", ns["te_reg"])]:
    for m, r in ns["evaluate_reg"](y_real_reg, folds, D_reg, ns["PA_reg"], X[prod]).items():
        real.append(dict(task="regression", design=design, model=m, prevalence=np.nan, **r))
for design, folds in [("nndm_loo", ns["F_clf"]), ("random_single", ns["te_clf"])]:
    for m, r in ns["evaluate_clf"](y_real_clf, folds, ns["Dfull"], ns["PA_clf"], X, True).items():
        real.append(dict(task="classification", design=design, model=m, prevalence=23 / 224, **r))
rng = np.random.default_rng(C.SEED + 99)
ns["N_PERM"] = 999
real.append(dict(task="structure", design="permutation_short_lag_0_20km", model="all productive (201)",
                 p=ns["structure_p"](y_real_reg, ns["PA_reg"], rng)))
real.append(dict(task="structure", design="permutation_short_lag_0_20km",
                 model=f"outside cluster ({int(ns['outside_reg'].sum())})",
                 p=ns["structure_p"](y_real_reg[ns["outside_reg"]], ns["PA_out"], rng)))
REAL = pd.DataFrame(real)

# ---------------------------------------------------------------------------
# Benchmark results
# ---------------------------------------------------------------------------
B = pd.concat([pickle.load(open(f, "rb")) for f in sorted(glob.glob(str(OUT / "benchmark_checkpoints" / "*.pkl")))],
              ignore_index=True)
B["prevalence"] = B["prevalence"].round(3)
B = B[~((B.model == "indicator_kriging") & (B.prevalence > 0.2))].reset_index(drop=True)
if "single_class_training_folds" in B:
    B["single_class_training_folds"] = B["single_class_training_folds"].fillna(0)
REAL["prevalence"] = REAL["prevalence"].round(3)
reg = B[B.task == "regression"].copy()
reg["detected"] = (reg.r2_cv > 0) & (reg.p < 0.05)
clf = B[B.task == "classification"].copy()
clf["detected_ranking"] = (clf.auc_excess > 0.5) & (clf.p_auc < 0.05)
clf["detected_probability"] = (clf.bss > 0) & (clf.p_bss < 0.05)
stc = B[B.task == "structure"].copy()
stc["detected"] = stc.p < 0.05

def q(s, k): return s.quantile(k)
PW = []
for (des, mod, a, b), g in reg.groupby(["design", "model", "a", "b"]):
    PW.append(dict(task="regression", design=des, model=mod, prevalence=np.nan, a_covariate_share=a, b_spatial_share=b,
                   replicates=len(g), detection_rate=g.detected.mean(), median_estimate=g.r2_cv.median(),
                   estimate_p05=q(g.r2_cv, 0.05), estimate_p95=q(g.r2_cv, 0.95), metric="R2cv"))
for (des, mod, prev, a, b), g in clf.groupby(["design", "model", "prevalence", "a", "b"]):
    PW.append(dict(task="classification (ranking)", design=des, model=mod, prevalence=prev, a_covariate_share=a,
                   b_spatial_share=b, replicates=len(g), detection_rate=g.detected_ranking.mean(),
                   median_estimate=g.auc_excess.median(), estimate_p05=q(g.auc_excess, 0.05),
                   estimate_p95=q(g.auc_excess, 0.95), metric="AUC of excess risk"))
    PW.append(dict(task="classification (probability)", design=des, model=mod, prevalence=prev, a_covariate_share=a,
                   b_spatial_share=b, replicates=len(g), detection_rate=g.detected_probability.mean(),
                   median_estimate=g.bss.median(), estimate_p05=q(g.bss, 0.05), estimate_p95=q(g.bss, 0.95),
                   metric="Brier skill score"))
for (mod, a, b), g in stc.groupby(["model", "a", "b"]):
    PW.append(dict(task="spatial structure", design="permutation, 0-20 km", model=mod, prevalence=np.nan,
                   a_covariate_share=a, b_spatial_share=b, replicates=len(g), detection_rate=g.detected.mean(),
                   metric="one-sided short-lag test"))
PW = pd.DataFrame(PW)

# compatibility of the observed results with each scenario (central 90% of simulated values)
comp = []
for _, r in REAL.iterrows():
    if r.task == "regression":
        sim = reg[(reg.design == r.design) & (reg.model == r.model)]; key, obs = "r2_cv", r.r2_cv
    elif r.task == "classification":
        sim = clf[(clf.design == r.design) & (clf.model == r.model) & (clf.prevalence == r.prevalence)]; key, obs = "auc_excess", r.auc_excess
    else:
        continue
    for (a, b), g in sim.groupby(["a", "b"]):
        lo, hi = g[key].quantile([0.05, 0.95])
        comp.append(dict(task=r.task, design=r.design, model=r.model, metric=key, observed=obs, a_covariate_share=a,
                         b_spatial_share=b, simulated_p05=lo, simulated_p95=hi,
                         percentile_of_observed=100 * np.mean(g[key].values <= obs), compatible_90=bool(lo <= obs <= hi)))
COMP = pd.DataFrame(comp)
T = {"benchmark_design": pd.DataFrame([
        ("covariate share a", ", ".join(map(str, ns["A_LEVELS"]))), ("spatial share b", ", ".join(map(str, ns["B_LEVELS"]))),
        ("spatial covariance", f"exponential, range parameter {ns['RANGE_KM']} km"), ("replicates per scenario", ns["N_REP"]),
        ("prevalences", "0.103 (observed), 0.30"), ("locations", "real 224 boreholes (201 productive for regression)"),
        ("covariates", "the ten continuous covariates at the real boreholes"),
        ("learners", "ridge (alphas by GCV) or logistic (C 0.1) on the ten covariates; ordinary or indicator kriging with per-fold spherical fit"),
        ("designs", "NNDM leave-one-out (primary) and one random 70/30 split"),
        ("detection rule", "estimate beyond the null and one-sided p < 0.05 (paired t test on losses; Mann-Whitney on excess risk; permutation for structure)")],
        columns=["element", "setting"]),
     "benchmark_power": PW, "benchmark_all_replicates": B, "real_data_same_evaluators": REAL,
     "compatibility_observed": COMP}
pickle.dump(T, open(OUT / "step3b_tables.pkl", "wb"))

pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
print(REAL.round(3).to_string())
v = PW[(PW.design.isin(["nndm_loo", "random_single"]))].pivot_table(index=["task", "design", "model", "prevalence", "b_spatial_share"],
      columns="a_covariate_share", values="detection_rate", dropna=False)
print(v.round(2).dropna(how="all").to_string())
print(PW[PW.task == "spatial structure"].pivot_table(index="model", columns=["a_covariate_share", "b_spatial_share"], values="detection_rate").round(2).to_string())
cc = COMP[COMP.design == "nndm_loo"].pivot_table(index=["task", "model", "b_spatial_share"], columns="a_covariate_share", values="compatible_90")
print(cc.to_string())
