"""Graphical abstract (1328 x 531 px minimum; produced at about 3140 x 1250 px)."""
import pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from biabak import config as C
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
O2 = C.OUT_DIR / "step2"; O3 = C.OUT_DIR / "step3"
info = pickle.load(open(O2 / "nndm_info_classification.pkl", "rb"))
rs = pickle.load(open(O2 / "checkpoints" / "classification__random_single.pkl", "rb"))[0]
nn = pickle.load(open(O2 / "checkpoints" / "classification__nndm_loo.pkl", "rb"))[0]
T2 = pickle.load(open(O2 / "step2_tables.pkl", "rb")); PT = T2["primary_nndm_tests"]
T3 = pickle.load(open(O3 / "step3b_tables.pkl", "rb")); PW = T3["benchmark_power"]
fig, axs = plt.subplots(1, 3, figsize=(10.5, 4.85), gridspec_kw={"width_ratios": [1.05, 1, 1]})
# 1 where the models are tested
ax = axs[0]
for d, col, lab in [(np.sort(info["Gij"]), "k", "region to predict"),
                    (np.sort(rs[rs.model == "logistic"].dist_nearest_train_km.values), "#DD8452", "random 70/30 test"),
                    (np.sort(nn[nn.model == "logistic"].dist_nearest_train_km.values), "#4C72B0", "NNDM test")]:
    ax.plot(d, np.arange(1, len(d) + 1) / len(d), color=col, lw=2.4 if col == "k" else 1.8, label=lab)
ax.set_xscale("symlog", linthresh=1); ax.set_xlim(0, 130); ax.set_xticks([0, 1, 5, 20, 100]); ax.set_xticklabels(["0", "1", "5", "20", "100"])
ax.set_xlabel("distance to training data (km)"); ax.set_ylabel("cumulative share")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=10)
ax.set_title("1. Test where the map is used", loc="left", fontsize=12.5, fontweight="bold")
# 2 skill at those distances
ax = axs[1]
rows = [("regression", "r2_cv", "ordinary_kriging", "kriging (R$^2_{cv}$)"), ("regression", "r2_cv", "elastic_net", "Elastic Net (R$^2_{cv}$)"),
        ("regression", "r2_cv", "random_forest", "random forest (R$^2_{cv}$)"), ("classification", "bss", "logistic", "logistic (BSS)"),
        ("classification", "bss", "random_forest", "random forest (BSS)")]
for k, (task, met, mod, lab) in enumerate(rows):
    r = PT[(PT.task == task) & (PT.metric == met) & (PT.model == mod)].iloc[0]
    ax.plot([r.ci_low, r.ci_high], [k, k], color="#4C72B0", lw=2.2); ax.plot(r.estimate, k, "o", color="#4C72B0", ms=7)
ax.axvline(0, color="k", lw=1, ls="--"); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[3] for r in rows], fontsize=10.5)
ax.invert_yaxis(); ax.set_xlabel("skill against training-fold baseline")
ax.set_title("2. No transferable skill", loc="left", fontsize=12.5, fontweight="bold")
# 3 what the null result excludes
ax = axs[2]
g = PW[(PW.task == "regression") & (PW.model == "ridge") & (PW.design == "nndm_loo") & (PW.b_spatial_share == 0)].sort_values("a_covariate_share")
ax.axvspan(-0.01, 0.2, color="#fde0c5", lw=0, label="compatible with the archive")
ax.plot(g.a_covariate_share, g.detection_rate, "o-", color="#C44E52", lw=2, ms=6, label="detection rate")
ax.axhline(0.8, color="grey", lw=0.8, ls=":")
ax.text(0.345, 0.62, "0.35:\nexcluded", ha="right", va="center", fontsize=10.5, color="#C44E52")
ax.set_xlim(-0.01, 0.36); ax.set_ylim(0, 1.05)
ax.set_xlabel("covariate share of variance"); ax.set_ylabel("detection rate")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=10)
ax.set_title("3. State what a null excludes", loc="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(w_pad=3.0)
OUT = C.OUT_DIR / "manuscript_figures"
OUT.mkdir(parents=True, exist_ok=True)
png_path = OUT / "Graphical_abstract.png"
pdf_path = OUT / "Graphical_abstract.pdf"
fig.savefig(png_path, bbox_inches="tight", pad_inches=0.15)
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.15)
from PIL import Image; print(Image.open(png_path).size)
