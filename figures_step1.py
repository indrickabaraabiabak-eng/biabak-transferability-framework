"""Figures for step 1. English labels, 300 dpi, panel letters in parentheses."""
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

from matplotlib.patches import Circle
from pyproj import Transformer

from biabak import config as C
from biabak import variography as V

OUT = C.OUT_DIR / "step1"
S = pickle.load(open(OUT / "step1_store.pkl", "rb"))
T = pickle.load(open(OUT / "step1_tables.pkl", "rb"))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "mathtext.fontset": "dejavuserif", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
COL = {"nugget": "#7f7f7f", "spherical": "#C44E52", "exponential": "#DD8452",
       "cardinal_sine": "#4C72B0", "j_bessel": "#55A868"}
LAB = {"nugget": "pure nugget", "spherical": "spherical", "exponential": "exponential",
       "cardinal_sine": "cardinal sine (hole effect)", "j_bessel": "J-Bessel (hole effect)"}
YLAB = {"log10_Sc": r"semivariance of $\log_{10} S_c$", "log10_T": r"semivariance of $\log_{10} T$",
        "status": "semivariance of productive status"}
TIT = {"log10_Sc": r"$\log_{10} S_c$ (n = 196)", "log10_T": r"$\log_{10} T$ (n = 196)",
       "status": "productive status (n = 219)"}
hh = np.linspace(0.01, 70, 600)


def panel(ax, s, key, letter, models=("nugget", "spherical", "cardinal_sine", "j_bessel"), legend=False):
    k = s["keep"]
    ax.fill_between(s["hm"][k], s["lo"][k], s["hi"][k], color="#d9d9d9", lw=0, label="95% permutation envelope")
    ax.axhline(s["var"], color="k", ls="--", lw=0.8, label="sample variance")
    sz = 8 + 40 * s["n"][k] / s["n"][k].max()
    ax.scatter(s["hm"][k], s["g"][k], s=sz, c="k", zorder=3, label="experimental (Matheron)")
    if np.isfinite(s["g_ch"][k]).any():
        ax.scatter(s["hm"][k], s["g_ch"][k], s=12, facecolors="none", edgecolors="#555555", lw=0.7,
                   zorder=3, label="experimental (Cressie-Hawkins)")
    for m in models:
        f = s["fits"][m]
        ax.plot(hh, V.model_gamma(m, hh, f["c0"], f["c"], f["a"]), color=COL[m], lw=1.3, label=LAB[m])
    ax.set_xlim(0, 71); ax.set_xlabel("lag distance (km)"); ax.set_ylabel(YLAB[key])
    ax.text(-0.24, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)


# Figure A: omnidirectional variograms with envelope and competing model families
fig, axs = plt.subplots(1, 3, figsize=(7.8, 3.0))
for ax, key, let in zip(axs, ["log10_Sc", "log10_T", "status"], ["(a)", "(b)", "(c)"]):
    panel(ax, S[key], key, let)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.subplots_adjust(wspace=0.45)
fig.savefig(OUT / "Fig_S1_variograms_envelopes.png", bbox_inches="tight"); plt.close(fig)

# Figure B: directional variograms of log10 Sc with envelopes
dfd = T["S14_directional_fits"]
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
for ax, (dname, let) in zip(axs.ravel(), zip(["E-W", "NE-SW", "N-S", "SE-NW"], ["(a)", "(b)", "(c)", "(d)"])):
    hm, g, n, keep, lo, hi, p = S["log10_Sc"]["dir"][dname]
    ax.fill_between(hm[keep], lo[keep], hi[keep], color="#d9d9d9", lw=0)
    ax.axhline(S["log10_Sc"]["var"], color="k", ls="--", lw=0.8)
    ax.scatter(hm[keep], g[keep], s=8 + 40 * n[keep] / n[keep].max(), c="k", zorder=3)
    out = keep & ((g < lo) | (g > hi))
    ax.scatter(hm[out], g[out], s=40, facecolors="none", edgecolors="#C44E52", lw=1.2, zorder=4)
    for m in ["spherical", "cardinal_sine"]:
        r = dfd[(dfd.target == "log10_Sc") & (dfd.direction == dname) & (dfd.model == m)].iloc[0]
        ax.plot(hh, V.model_gamma(m, hh, r.c0, r.c, r.a), color=COL[m], lw=1.2, label=LAB[m])
    ph = dfd[(dfd.target == "log10_Sc") & (dfd.direction == dname)].global_test_p_holm_12_tests.iloc[0]
    ax.text(-0.17, 1.04, let, transform=ax.transAxes, fontweight="bold", fontsize=10)
    ax.set_ylabel(r"semivariance of $\log_{10} S_c$")
for ax in axs[1]:
    ax.set_xlabel("lag distance (km)")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h + [plt.Line2D([], [], color="#d9d9d9", lw=6), plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#C44E52")],
           l + ["95% permutation envelope", "bin outside envelope"], loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
[a_.tick_params(labelbottom=True) for a_ in axs.ravel()]
fig.savefig(OUT / "Fig_S2_directional_envelopes.png", bbox_inches="tight"); plt.close(fig)

# Figure C: identifiability of the fitted structure and LOO kriging skill
sens = T["S19_sensitivity_lag_estimator_weighting"]
loo = T["S12_loo_kriging"]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axs[0]
fams = ["spherical", "exponential", "cardinal_sine", "j_bessel"]
tg = ["log10_Sc", "log10_T", "status"]
rng = np.random.default_rng(1)
for ti, t in enumerate(tg):
    for fi, fm in enumerate(fams):
        v = sens[(sens.target == t) & (sens.model == fm)]
        xpos = ti * 5 + fi
        jit = rng.uniform(-0.25, 0.25, len(v))
        bound = v.range_parameter_at_bound.values
        ax.scatter(xpos + jit[~bound], v.nugget_to_sill.values[~bound], s=10, color=COL[fm], alpha=0.8)
        ax.scatter(xpos + jit[bound], v.nugget_to_sill.values[bound], s=12, marker="x", color=COL[fm], alpha=0.8)
ax.set_xticks([1.5, 6.5, 11.5]); ax.set_xticklabels([TIT[t].split(" (")[0] for t in tg])
ax.set_ylabel("fitted nugget-to-sill ratio"); ax.set_ylim(-0.03, 1.05)
ax.text(-0.12, 1.04, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=10)
hand = [plt.Line2D([], [], marker="o", ls="", color=COL[f], label=LAB[f]) for f in fams]
hand.append(plt.Line2D([], [], marker="x", ls="", color="k", label="range parameter at bound"))
fig.legend(handles=hand, loc="lower center", frameon=False, ncol=5, bbox_to_anchor=(0.5, -0.02))
ax = axs[1]
w = 0.16
for fi, fm in enumerate(["nugget"] + fams):
    v = [loo[(loo.target == t) & (loo.model == fm)].r2_loo.iloc[0] for t in tg]
    ax.bar(np.arange(3) + (fi - 2) * w, v, w, color=COL[fm], label=LAB[fm])
    if fm == "nugget":
        ax.scatter(np.arange(3) + (fi - 2) * w, v, marker="_", s=60, color=COL[fm])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels([TIT[t].split(" (")[0] for t in tg])
ax.set_ylabel(r"leave-one-out $R^2$")
ax.text(-0.25, 1.04, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "Fig_S3_identifiability_loo.png", bbox_inches="tight"); plt.close(fig)

# Figure D: clustering, short-range pairs, proportional effect
s = S["log10_Sc"]
fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.6))
ax = axs[0, 0]
sub = s["sub"]
ins = s["inside"]
# Public-release panel: report the cluster decomposition without displaying
# individual borehole coordinates.
labels = ["outside", "inside"]
counts = [int((~ins).sum()), int(ins.sum())]
ax.bar(labels, counts, color=["#4C72B0", "#C44E52"], width=0.65)
for j, n_ in enumerate(counts):
    ax.text(j, n_ + max(counts) * 0.03, str(n_), ha="center", va="bottom", fontsize=8)
ax.set_ylabel("number of productive boreholes")
ax.set_xlabel(f"relative to the {C.CLUSTER_RADIUS_KM:.0f} km Yaoundé cluster")
ax.set_ylim(0, max(counts) * 1.18)
ax.text(-0.2, 1.04, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=10)
ax = axs[0, 1]
cc = {"both inside cluster": "#C44E52", "one inside, one outside": "#8172B3", "both outside cluster": "#4C72B0"}
for cls, colr in cc.items():
    hm, g, n, keep = s["clu"][cls]
    ax.plot(hm[keep], g[keep], "o-", ms=3, lw=0.8, color=colr, label=f"{cls} ({int(n[keep].sum())} pairs)")
ax.axhline(s["var"], color="k", ls="--", lw=0.8)
ax.set_xlabel("lag distance (km)"); ax.set_ylabel(r"semivariance of $\log_{10} S_c$")
ax.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
ax.text(-0.2, 1.04, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=10)
ax = axs[1, 0]
sr = T["S15_short_range_structure"]
sr = sr[(sr.target == "log10_Sc") & ~((sr.bin_from_m == 0) & (sr.bin_to_m >= 500))]
xl = [f"{int(a)}-{int(b)}" for a, b in zip(sr.bin_from_m, sr.bin_to_m)]
xs = np.arange(len(sr))
ax.errorbar(xs, sr.perm_mean, yerr=[sr.perm_mean - sr.perm_p2_5, sr.perm_p97_5 - sr.perm_mean],
            fmt="none", ecolor="#999999", capsize=3, label="permutation 95% interval")
ax.scatter(xs, sr.gamma, c="k", zorder=3, label="observed")

ax.axhline(s["var"], color="k", ls="--", lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels([f"{a}\n({n} pairs)" for a, n in zip(xl, sr.n_pairs)], rotation=0, fontsize=6.5)
ax.set_xlabel("pair separation (m)"); ax.set_ylabel(r"semivariance of $\log_{10} S_c$")
ax.legend(frameon=False, loc="upper left")
ax.text(-0.2, 1.04, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=10)
ax = axs[1, 1]
lm, lv = s["pe"]
ax.scatter(lm, lv, s=8, c="#4C72B0")
pe = T["S18_proportional_effect"]; r = pe[pe.target == "log10_Sc"].iloc[0]
ax.set_xlabel(r"local mean of $\log_{10} S_c$ (10 km window)")
ax.set_ylabel(r"local variance of $\log_{10} S_c$")
ax.text(-0.2, 1.04, "(d)", transform=ax.transAxes, fontweight="bold", fontsize=10)
fig.tight_layout()
fig.subplots_adjust(hspace=0.75)
fig.savefig(OUT / "Fig_S4_cluster_shortrange_proportional.png", bbox_inches="tight"); plt.close(fig)

# Figure E: inside versus outside the dense cluster, with subset-specific envelopes
tb = T["S17b_structure_tests_by_subset"]
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4), sharex=True)
for r_, key in enumerate(["log10_Sc", "log10_T"]):
    for c_, name in enumerate(["inside cluster", "outside cluster"]):
        ax = axs[r_, c_]
        hm, g, n, keep, lo, hi, var = S[key]["subset"][name]
        ax.fill_between(hm[keep], lo[keep], hi[keep], color="#d9d9d9", lw=0)
        ax.axhline(var, color="k", ls="--", lw=0.8)
        ax.scatter(hm[keep], g[keep], s=8 + 40 * n[keep] / n[keep].max(), c="k", zorder=3)
        q = tb[(tb.target == key) & (tb.subset == name) & (tb.short_lag_cut_km == 20.0)].iloc[0]
        ax.set_ylabel(YLAB[key])
        ax.text(-0.19, 1.04, "(" + "abcd"[r_ * 2 + c_] + ")", transform=ax.transAxes, fontweight="bold", fontsize=10)
        if r_ == 1: ax.set_xlabel("lag distance (km)")
fig.tight_layout()
[a_.tick_params(labelbottom=True) for a_ in axs.ravel()]
fig.savefig(OUT / "Fig_S5_inside_outside_cluster.png", bbox_inches="tight"); plt.close(fig)
print("figures written")
