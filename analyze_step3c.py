"""Extended benchmark: detection rates with Wilson 95% intervals, one-sided compatibility rule,
range sensitivity, and figures (main Fig. 11 and 12; supplementary AUC-validation figure)."""
import glob, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure as _mf
_orig = _mf.Figure.savefig
def _save(self, fname, *a, **k):
    _orig(self, fname, *a, **k)
    if str(fname).endswith(".png"): _orig(self, str(fname)[:-4] + ".pdf", *a, **k)
_mf.Figure.savefig = _save
from scipy.stats import norm
from biabak import config as C
O3 = C.OUT_DIR / "step3"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})

def wilson(k, n, z=1.959964):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + z ** 2 / n; c = (p + z ** 2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h

T3 = pickle.load(open(O3 / "step3b_tables.pkl", "rb"))
B0 = T3["benchmark_all_replicates"]; REAL = T3["real_data_same_evaluators"]
EXT = pd.concat([pickle.load(open(f, "rb")) for f in sorted(glob.glob(str(O3 / "extension_checkpoints" / "*.pkl")))], ignore_index=True)
EXT["prevalence"] = EXT["prevalence"].round(3); B0 = B0.copy(); B0["prevalence"] = B0["prevalence"].astype(float).round(3)
B0["range_km"] = 15.0
# covariate-only cells: original 30 replicates + extension
cov0 = B0[(B0.b == 0) & (((B0.task == "regression") & (B0.model == "ridge")) | ((B0.task == "classification") & (B0.model == "logistic")))]
cov = pd.concat([cov0.assign(source="initial"), EXT[EXT.range_km.isna()].assign(source="extension")], ignore_index=True)

def detected(df, metric):
    if metric == "r2_cv": return (df.r2_cv > 0) & (df.p < 0.05)
    if metric == "auc_excess": return (df.auc_excess > 0.5) & (df.p_auc < 0.05)
    if metric == "bss": return (df.bss > 0) & (df.p_bss < 0.05)
    return df.p < 0.05

TESTS = [("Regression, ridge", "regression", "ridge", np.nan, "r2_cv"),
         ("Ranking, logistic, 10.3% failures", "classification", "logistic", 0.103, "auc_excess"),
         ("Probability, logistic, 10.3% failures", "classification", "logistic", 0.103, "bss"),
         ("Ranking, logistic, 30% failures", "classification", "logistic", 0.3, "auc_excess"),
         ("Probability, logistic, 30% failures", "classification", "logistic", 0.3, "bss")]
obs = {"r2_cv": REAL[(REAL.task == "regression") & (REAL.model == "ridge") & (REAL.design == "nndm_loo")].r2_cv.iloc[0],
       "auc_excess": REAL[(REAL.task == "classification") & (REAL.model == "logistic") & (REAL.design == "nndm_loo")].auc_excess.iloc[0],
       "bss": REAL[(REAL.task == "classification") & (REAL.model == "logistic") & (REAL.design == "nndm_loo")].bss.iloc[0]}
rows = []
for lab, task, model, prev, metric in TESTS:
    for design in ["nndm_loo", "random_single"]:
        for a, g in cov[(cov.task == task) & (cov.model == model) & (cov.design == design)].groupby("a"):
            if not np.isnan(prev): g = g[np.isclose(g.prevalence, prev)]
            k = int(detected(g, metric).sum()); n = len(g); lo, hi = wilson(k, n)
            vals = g[metric].dropna().values
            r = dict(test=lab, design=design, a_covariate_share=a, replicates=n, detected=k, rate=k / n, wilson_low=lo, wilson_high=hi,
                     median_estimate=np.median(vals), p05_estimate=np.percentile(vals, 5), p95_estimate=np.percentile(vals, 95))
            if design == "nndm_loo" and not (prev == 0.3):
                o = obs[metric]; r["observed"] = o; r["share_simulated_at_or_below_observed"] = np.mean(vals <= o)
                r["incompatible_one_sided_5pct"] = bool(np.mean(vals <= o) < 0.05)
            rows.append(r)
# size-adjusted ranking: threshold at the 95th percentile of the simulated null distribution
for prev in [0.103, 0.3]:
    g = cov[(cov.task == "classification") & (cov.design == "nndm_loo") & np.isclose(cov.prevalence, prev)]
    thr = np.percentile(g[g.a == 0].auc_excess, 95)
    for a, h in g.groupby("a"):
        k = int((h.auc_excess > thr).sum()); n = len(h); lo, hi = wilson(k, n)
        r = dict(test=f"Ranking, logistic, {'10.3' if prev < 0.2 else '30'}% failures, size-adjusted", design="nndm_loo", a_covariate_share=a,
                 replicates=n, detected=k, rate=k / n, wilson_low=lo, wilson_high=hi, null_threshold=thr)
        rows.append(r)
POWER = pd.DataFrame(rows)
# spatial detection by range (a = 0)
sp0 = B0[(B0.a == 0) & (B0.b > 0) & (((B0.task == "regression") & (B0.model == "kriging") & (B0.design == "nndm_loo")) | (B0.task == "structure"))]
sp = pd.concat([sp0, EXT[EXT.range_km.notna()]], ignore_index=True)
srows = []
for (task, model, b, rk), g in sp.groupby(["task", "model", "b", "range_km"]):
    k = int(detected(g, "r2_cv" if task == "regression" else "p").sum()); n = len(g); lo, hi = wilson(k, n)
    srows.append(dict(test="ordinary kriging, R2cv" if task == "regression" else f"short-lag test, {model}", b_spatial_share=b, range_km=rk,
                      replicates=n, detected=k, rate=k / n, wilson_low=lo, wilson_high=hi))
SPATIAL = pd.DataFrame(srows)
A = pickle.load(open(C.OUT_DIR / "audit" / "audit_tables.pkl", "rb"))
pickle.dump(dict(power=POWER, spatial=SPATIAL, extension_replicates=EXT), open(O3 / "step3c_tables.pkl", "wb"))
pd.set_option("display.width", 230)
print(POWER[POWER.design == "nndm_loo"].round(3).to_string()); print(SPATIAL.round(3).to_string())

# ---------------- Figure 11 (main): detection rates with Wilson intervals ----------------
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8))
cfg = [(axs[0, 0], [("Regression, ridge", "#1b9e77", "ridge, " + r"$R^2_{cv}$")], "(a)"),
       (axs[0, 1], [("Ranking, logistic, 10.3% failures, size-adjusted", "#d95f02", "10.3% failures"), ("Ranking, logistic, 30% failures, size-adjusted", "#7570b3", "30% failures")], "(b)"),
       (axs[1, 0], [("Probability, logistic, 10.3% failures", "#d95f02", "10.3% failures"), ("Probability, logistic, 30% failures", "#7570b3", "30% failures")], "(c)")]
for ax, series, let in cfg:
    for lab, col, leg in series:
        g = POWER[(POWER.test == lab) & (POWER.design == "nndm_loo")].sort_values("a_covariate_share")
        ax.fill_between(g.a_covariate_share, g.wilson_low, g.wilson_high, color=col, alpha=0.18, lw=0)
        ax.plot(g.a_covariate_share, g.rate, "o-", color=col, lw=1.4, ms=4, label=leg)
    ax.axhline(0.05, color="grey", lw=0.7, ls="--"); ax.axhline(0.8, color="k", lw=0.7, ls=":")
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel("covariate share a"); ax.set_ylabel("detection rate")
    ax.legend(frameon=False, loc="upper left"); ax.text(-0.2, 1.04, let, transform=ax.transAxes, fontweight="bold", fontsize=10)
ax = axs[1, 1]
styles = {"ordinary kriging, R2cv": ("-", "o", "ordinary kriging"), "short-lag test, all productive (201)": ("--", "s", "short-lag test, all"),
          "short-lag test, outside cluster (95)": (":", "^", "short-lag test, outside cluster")}
cols = {0.25: "#1b9e77", 0.5: "#7570b3"}
for (test, b), g in SPATIAL.groupby(["test", "b_spatial_share"]):
    ls, mk, lab = styles.get(test, ("-", "o", test))
    g = g.sort_values("range_km")
    ax.errorbar(g.range_km + (0.6 if b == 0.5 else -0.6), g.rate, yerr=[g.rate - g.wilson_low, g.wilson_high - g.rate], ls=ls, marker=mk, ms=4,
                color=cols[b], capsize=2, lw=1.1, label=f"{lab}, b = {b}")
ax.set_xticks([5, 15, 40]); ax.set_xlabel("range parameter of the spatial field (km)"); ax.set_ylabel("detection rate")
ax.set_ylim(-0.02, 1.02); ax.axhline(0.8, color="k", lw=0.7, ls=":")
ax.text(-0.2, 1.04, "(d)", transform=ax.transAxes, fontweight="bold", fontsize=10)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(O3 / "Fig_B2new_detection_power.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 12 (main): observed against simulated (one-sided rule) ----------------
fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.0))
for ax, (lab, metric, ylab, let, prev) in zip(axs, [("Regression, ridge", "r2_cv", r"$R^2_{cv}$, ridge", "(a)", np.nan),
                                                     ("Ranking, logistic, 10.3% failures", "auc_excess", "excess-risk AUC, logistic", "(b)", 0.103),
                                                     ("Probability, logistic, 10.3% failures", "bss", "Brier skill score, logistic", "(c)", 0.103)]):
    task = "regression" if metric == "r2_cv" else "classification"
    g = cov[(cov.task == task) & (cov.design == "nndm_loo")]
    if task == "classification": g = g[np.isclose(g.prevalence, 0.103)]
    aa = sorted(g.a.unique()); data = [g[g.a == a_][metric].dropna().values for a_ in aa]
    bp = ax.boxplot(data, positions=range(len(aa)), widths=0.5, whis=(5, 95), showfliers=False, patch_artist=True)
    for k_, box in enumerate(bp["boxes"]):
        inc = POWER[(POWER.test == lab) & (POWER.design == "nndm_loo")].sort_values("a_covariate_share").incompatible_one_sided_5pct.values[k_]
        box.set_facecolor("#f4a6a6" if inc else "#cfe3f3")
    ax.axhline(obs[metric], color="#C44E52", lw=1.3)
    ax.set_xticks(range(len(aa))); ax.set_xticklabels([f"{a_:.2f}" for a_ in aa]); ax.set_xlabel("covariate share a"); ax.set_ylabel(ylab)
    ax.text(-0.3, 1.04, let, transform=ax.transAxes, fontweight="bold", fontsize=10)
from matplotlib.patches import Patch
fig.legend([plt.Line2D([], [], color="#C44E52"), Patch(color="#cfe3f3"), Patch(color="#f4a6a6")],
           ["observed statistic", "compatible (observed above the 5th percentile)", "incompatible (below the 5th percentile)"],
           loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(O3 / "Fig_B3new_observed_vs_simulated.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Supplementary figure: validation of the excess-risk AUC ----------------
S = A["auc_sim"]
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)
for ax, design, lab in zip(axs, ["nndm_loo", "grid_20km", "kmeans_8"], ["NNDM leave-one-out", "20 km blocks", "k-means, 8 blocks"]):
    g = S[S.design == design]
    ax.plot(g.true_auc, g.pooled_mean, "o-", color="#C44E52", label="pooled AUC")
    ax.plot(g.true_auc, g.excess_mean, "s-", color="#1f77b4", label="excess-risk AUC")
    if g.foldwise_mean_mean.notna().any():
        ax.plot(g.true_auc, g.foldwise_mean_mean, "^--", color="#7f7f7f", label="mean fold-wise AUC")
    ax.plot([0.5, 0.8], [0.5, 0.8], color="k", lw=0.7, ls=":")
    ax.set_xlabel("true AUC"); ax.text(-0.2, 1.04, "(" + "abc"[["nndm_loo", "grid_20km", "kmeans_8"].index(design)] + ")", transform=ax.transAxes, fontweight="bold", fontsize=10)
axs[0].set_ylabel("estimated AUC")
for a_ in axs: a_.tick_params(labelleft=True)
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(O3 / "Fig_S_auc_validation.png", bbox_inches="tight"); plt.close(fig)
print("figures written")
