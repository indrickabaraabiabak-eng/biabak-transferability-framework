"""Figures for step 3."""
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

from matplotlib.colors import ListedColormap

from biabak import config as C

OUT = C.OUT_DIR / "step3"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
A = pickle.load(open(OUT / "step3a_tables.pkl", "rb"))
G = pd.read_pickle(OUT / "aoa_grid.pkl")
thr = float(A["aoa_summary"].threshold.iloc[0])


def letter(ax, s, x=-0.14, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=10)


# Figure A: dissimilarity index and area of applicability
fig, axs = plt.subplots(1, 3, figsize=(7.8, 3.3), gridspec_kw={"width_ratios": [1, 1, 0.8]})
g = G.dropna(subset=["DI"])
sc = axs[0].scatter(g.x_km, g.y_km, c=g.DI, s=0.4, cmap="viridis", vmin=0, vmax=np.percentile(g.DI, 99), rasterized=True)
cb = fig.colorbar(sc, ax=axs[0], shrink=0.8); cb.set_label("dissimilarity index")
axs[0].text(0.02, 0.02, "individual borehole locations withheld", transform=axs[0].transAxes, fontsize=6.5, bbox=dict(fc="white", ec="none", alpha=0.75))
axs[0].set_aspect("equal"); axs[0].set_xlabel("UTM 32N easting (km)"); axs[0].set_ylabel("UTM 32N northing (km)")
letter(axs[0], "(a)", -0.3)
axs[1].scatter(g.x_km, g.y_km, c=np.where(g.in_AoA == 1, 0, 1), s=0.4, cmap=ListedColormap(["#c7e9c0", "#d7301f"]), rasterized=True)
axs[1].set_aspect("equal"); axs[1].set_xlabel("UTM 32N easting (km)"); axs[1].set_ylabel("UTM 32N northing (km)")
share = A["aoa_summary"].share_domain_in_AoA.iloc[0]
letter(axs[1], "(b)", -0.3)
bd = A["aoa_by_distance"]
axs[2].bar(bd.dist_bin.astype(str), 100 * bd.share_in_AoA, color="#74a9cf")
axs[2].set_ylim(0, 100); axs[2].tick_params(axis="x", rotation=45); axs[2].set_xlabel("distance to nearest\nborehole (km)"); axs[2].set_ylabel("share of cells inside AoA (%)")
letter(axs[2], "(c)", -0.35)
fig.tight_layout(); fig.savefig(OUT / "Fig_B1_area_of_applicability.png", bbox_inches="tight"); plt.close(fig)

# Figure B: detection power under NNDM and under one random split
step3b = OUT / "step3b_tables.pkl"
if not step3b.exists():
    print("step3b_tables.pkl absent: benchmark figures skipped; Fig_B1 only written")
    raise SystemExit(0)
T = pickle.load(open(step3b, "rb"))
PW = T["benchmark_power"]
fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.2), sharey=True)
panels = [("regression", "ridge", np.nan, r"regression, ridge"), ("regression", "kriging", np.nan, "regression, ordinary kriging"),
          ("classification (ranking)", "logistic", 0.103, "ranking, logistic, 10.3% failures"),
          ("classification (ranking)", "logistic", 0.3, "ranking, logistic, 30% failures"),
          ("classification (probability)", "logistic", 0.103, "probability (BSS), logistic, 10.3%"),
          ("classification (ranking)", "indicator_kriging", 0.103, "ranking, indicator kriging, 10.3%")]
cols = {0.0: "#1b9e77", 0.25: "#d95f02", 0.5: "#7570b3"}
for ax, (task, model, prev, title), let in zip(axs.ravel(), panels, "abcdef"):
    for des, ls in [("nndm_loo", "-"), ("random_single", ":")]:
        for b, col in cols.items():
            g = PW[(PW.task == task) & (PW.model == model) & (PW.design == des) & (PW.b_spatial_share == b)]
            if not np.isnan(prev):
                g = g[np.isclose(g.prevalence, prev)]
            g = g.sort_values("a_covariate_share")
            if g.empty: continue
            ax.plot(g.a_covariate_share, g.detection_rate, ls, color=col, marker="o", ms=3, lw=1.1,
                    label=f"b = {b}, {'NNDM' if des == 'nndm_loo' else 'random 70/30'}")
    ax.axhline(0.8, color="k", lw=0.6, ls="--"); ax.axhline(0.05, color="grey", lw=0.6, ls="--")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("covariate share a"); letter(ax, f"({let})", -0.18, 1.04)
for ax in axs.ravel(): ax.tick_params(labelleft=True)
for ax in axs[:, 0]: ax.set_ylabel("detection rate")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "Fig_B2_detection_power.png", bbox_inches="tight"); plt.close(fig)

# Figure C: where the observed statistics sit among simulated ones (NNDM, b = 0)
REAL = T["real_data_same_evaluators"]; Ball = T["benchmark_all_replicates"]
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8))
for ax, (task, model, key, lab, let) in zip(axs, [("regression", "ridge", "r2_cv", r"$R^2_{cv}$, ridge", "(a)"),
                                                   ("regression", "kriging", "r2_cv", r"$R^2_{cv}$, kriging", "(b)"),
                                                   ("classification", "logistic", "auc_excess", "excess-risk AUC, logistic regression", "(c)")]):
    for b, off, col in [(0.0, -0.15, cols[0.0]), (0.25, 0.0, cols[0.25]), (0.5, 0.15, cols[0.5])]:
        s = Ball[(Ball.task == task) & (Ball.model == model) & (Ball.design == "nndm_loo") & np.isclose(Ball.b, b)]
        if task == "classification":
            s = s[np.isclose(s.prevalence.astype(float), 23 / 224, atol=1e-3)]
        aa = sorted(s.a.unique())
        data = [s[s.a == a_][key].dropna().values for a_ in aa]
        bp = ax.boxplot(data, positions=np.arange(len(aa)) + off, widths=0.12, patch_artist=True, showfliers=False)
        for p_ in bp["boxes"]: p_.set_facecolor(col); p_.set_alpha(0.7)
    obs = REAL[(REAL.task == task) & (REAL.model == model) & (REAL.design == "nndm_loo")][key].iloc[0]
    ax.axhline(obs, color="#C44E52", lw=1.3, label=f"observed ({obs:.3f})")
    ax.set_xticks(range(len(aa))); ax.set_xticklabels([f"{a_:.2f}" for a_ in aa])
    ax.set_xlabel("covariate share a"); ax.set_ylabel(lab); letter(ax, let, -0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
from matplotlib.patches import Patch
fig.legend([Patch(color=cols[b]) for b in (0.0, 0.25, 0.5)], ["spatial share b = 0", "b = 0.25", "b = 0.5"],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT / "Fig_B3_observed_vs_simulated.png", bbox_inches="tight"); plt.close(fig)
print("figures written")
