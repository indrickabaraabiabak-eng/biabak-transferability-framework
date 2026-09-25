"""Figures for step 2: validation geometry, skill by design, skill by distance."""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure as _mf
_orig_save = _mf.Figure.savefig
def _save_png_and_pdf(self, fname, *a, **k):
    _orig_save(self, fname, *a, **k)
    f = str(fname)
    if f.endswith(".png"):
        _orig_save(self, f[:-4] + ".pdf", *a, **k)
_mf.Figure.savefig = _save_png_and_pdf


from biabak import config as C

OUT = C.OUT_DIR / "step2"
T = pickle.load(open(OUT / "step2_tables.pkl", "rb"))
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
DLAB = {"random_single": "random 70/30", "random_repeated_100": "100 random 70/30", "kmeans_8": "k-means, 8 blocks",
        "grid_10km": "10 km blocks", "grid_20km": "20 km blocks", "grid_35km": "35 km blocks",
        "grid_50km": "50 km blocks", "nndm_loo": "NNDM leave-one-out"}
DORD = list(DLAB)
S = T["summary_by_design_model"]
REP = T["per_split_random_repeated"]


def letter(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=10)


# Figure 1: distances from held-out boreholes to training boreholes, against the prediction domain
P2 = pickle.load(open(OUT / "nndm_info_classification.pkl", "rb"))
ck = pickle.load(open(OUT / "checkpoints" / "classification__random_single.pkl", "rb"))[0]
fig, ax = plt.subplots(figsize=(4.8, 3.4))
g = np.sort(P2["Gij"]); ax.plot(g, np.arange(1, len(g) + 1) / len(g), color="k", lw=2.2, label="prediction domain (1 km cells)")
cols = {"random_single": "#DD8452", "kmeans_8": "#8172B3", "grid_20km": "#55A868", "nndm_loo": "#4C72B0"}
import glob
for dname, col in cols.items():
    p = pickle.load(open(OUT / "checkpoints" / f"classification__{dname}.pkl", "rb"))[0]
    d = np.sort(p[p.model == "logistic"].dist_nearest_train_km.values)
    ax.plot(d, np.arange(1, len(d) + 1) / len(d), color=col, lw=1.4, label=DLAB[dname])
ax.set_xscale("symlog", linthresh=1); ax.set_xlim(0, 130)
ax.set_xticks([0, 1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["0", "1", "2", "5", "10", "20", "50", "100"])
ax.set_xlabel("distance to nearest training borehole (km)"); ax.set_ylabel("cumulative proportion")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT / "Fig_V1_generalisation_distance.png", bbox_inches="tight"); plt.close(fig)

# Figure 2: regression skill by design
RM = [("inverse_distance", "inverse distance"), ("ordinary_kriging", "ordinary kriging"), ("elastic_net", "Elastic Net"),
      ("random_forest", "random forest"), ("extra_trees", "extremely randomized trees"), ("gradient_boosting", "gradient boosting"),
      ("svr_rbf", "support vector regression"), ("mlp", "perceptron")]
CM = [("inverse_distance", "inverse distance"), ("indicator_kriging", "indicator kriging"), ("logistic", "logistic regression"),
      ("random_forest", "random forest"), ("extra_trees", "extremely randomized trees"), ("gradient_boosting", "gradient boosting"),
      ("mlp", "perceptron"), ("knn", "k-nearest neighbors")]
PAL = plt.cm.tab10(np.linspace(0, 1, 10))
_T = plt.cm.tab10.colors
MODEL_COL = {"inverse_distance": _T[0], "ordinary_kriging": _T[1], "indicator_kriging": _T[1], "elastic_net": _T[2],
             "logistic": _T[2], "random_forest": _T[3], "mlp": _T[4], "extra_trees": _T[5], "gradient_boosting": _T[6],
             "svr_rbf": _T[7], "knn": _T[7]}


def design_panel(ax, task, models, key, rep_key, ylab, ylim=None, ref=0.0):
    w = 0.8 / len(models)
    for mi, (m, lab_) in enumerate(models):
        for di, d in enumerate(DORD):
            x = di + (mi - (len(models) - 1) / 2) * w
            if d == "random_repeated_100":
                v = REP[(REP.task == task) & (REP.model == m)][rep_key].dropna()
                ax.plot([x, x], [v.quantile(0.025), v.quantile(0.975)], color=MODEL_COL[m], lw=1.0)
                ax.scatter(x, v.median(), color=MODEL_COL[m], s=10, marker="D", zorder=3)
            else:
                r = S[(S.task == task) & (S.design == d) & (S.model == m)]
                if r.empty: continue
                r = r.iloc[0]
                lo, hi = r.get(f"{key}_ci_low", np.nan), r.get(f"{key}_ci_high", np.nan)
                if np.isfinite(lo):
                    ax.plot([x, x], [lo, hi], color=MODEL_COL[m], lw=1.0)
                val = r[key]
                if ylim and val < ylim[0]:
                    yb = ylim[0] + 0.02 * (ylim[1] - ylim[0])
                    ax.scatter(x, yb, color=MODEL_COL[m], s=22, marker="v", zorder=4, label=lab_ if di == 0 else None)
                    ax.text(x + 0.03, yb + 0.035 * (ylim[1] - ylim[0]), f"{val:.2f}", fontsize=6, color=MODEL_COL[m], ha="left", va="bottom")
                else:
                    ax.scatter(x, val, color=MODEL_COL[m], s=12, zorder=3, label=lab_ if di == 0 else None)
    ax.axhline(ref, color="k", lw=0.8, ls="--")
    for di in range(len(DORD) - 1):
        ax.axvline(di + 0.5, color="#e0e0e0", lw=0.6)
    ax.set_xticks(range(len(DORD))); ax.set_xticklabels([DLAB[d] for d in DORD], rotation=30, ha="right")
    ax.set_ylabel(ylab)
    if ylim: ax.set_ylim(*ylim)


RM_MAIN = [m for m in RM if m[0] in ("ordinary_kriging", "inverse_distance", "elastic_net", "random_forest", "mlp")]
CM_MAIN = [m for m in CM if m[0] in ("indicator_kriging", "logistic", "random_forest", "extra_trees", "mlp")]
for tag, rm, cm in [("", RM_MAIN, CM_MAIN), ("_full", RM, CM)]:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    design_panel(ax, "regression", rm, "r2_cv", "r2_cv", r"$R^2_{cv}$ ($\log_{10} S_c$)", ylim=(-0.9, 0.35))
    h_ = [plt.Line2D([], [], marker="o", ls="", color=MODEL_COL[m_]) for m_, _ in rm]
    fig.legend(h_, [lab for _, lab in rm], loc="lower center", ncol=min(5, len(rm)) if tag == "" else 4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(OUT / f"Fig_V2_regression_by_design{tag}.png", bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)
    design_panel(axs[0], "classification", cm, "auc_excess_risk", "auc_excess_risk", "excess-risk AUC", ylim=(0.15, 1.0), ref=0.5)
    letter(axs[0], "(a)", -0.12)
    design_panel(axs[1], "classification", cm, "bss", "bss", "Brier skill score", ylim=(-0.5, 0.2))
    letter(axs[1], "(b)", -0.12, 1.04)
    h_ = [plt.Line2D([], [], marker="o", ls="", color=MODEL_COL[m_]) for m_, _ in cm]
    fig.legend(h_, [lab for _, lab in cm], loc="lower center", ncol=min(5, len(cm)) if tag == "" else 4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / f"Fig_V3_classification_by_design{tag}.png", bbox_inches="tight"); plt.close(fig)

# Figure 4: skill against distance to the nearest training borehole
DS = T["skill_by_distance"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
order = ["0-1", "1-2", "2-5", "5-10", "10-20", "20-40", ">40"]
for ax, task, models, ylab, let in [(axs[0], "regression", [("ordinary_kriging", "ordinary kriging"), ("elastic_net", "Elastic Net"), ("random_forest", "random forest"), ("gradient_boosting", "gradient boosting")], r"$R^2_{cv}$ ($\log_{10} S_c$)", "(a)"),
                                    (axs[1], "classification", [("indicator_kriging", "indicator kriging"), ("logistic", "logistic regression"), ("random_forest", "random forest"), ("gradient_boosting", "gradient boosting")], "Brier skill score", "(b)")]:
    for mi, (m, lab_) in enumerate(models):
        g = DS[(DS.task == task) & (DS.model == m)].set_index("distance_bin_km").reindex(order)
        x = np.arange(len(order)) + (mi - 1.5) * 0.12
        ax.errorbar(x, g.skill, yerr=[g.skill - g.ci_low, g.ci_high - g.skill], fmt="o-", ms=3, lw=0.9, capsize=2, label=lab_)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_xlabel("distance to nearest training borehole (km)"); ax.set_ylabel(ylab)
    ax.set_ylim(*((-1.05, 0.3) if task == "regression" else (-0.6, 0.6))); letter(ax, let, -0.2)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
fig.tight_layout(); fig.savefig(OUT / "Fig_V4_skill_by_distance.png", bbox_inches="tight"); plt.close(fig)
print("figures written")
