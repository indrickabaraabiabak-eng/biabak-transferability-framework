"""Generate the updated Supplementary Figure S5 from final Step 3C results.

The figure compares final extended-benchmark detection rates under:
  - NNDM leave-one-out
  - one random 70/30 split

It uses ONLY values stored in outputs/step3/step3c_tables.pkl.
No benchmark values are hard-coded.

Output:
  outputs/step3/Fig_S5_final_NNDM_vs_random.png
  outputs/step3/Fig_S5_final_NNDM_vs_random.pdf
"""

from __future__ import annotations

import pickle
import numpy as np
import matplotlib.pyplot as plt

from biabak import config as C

OUT = C.OUT_DIR / "step3"

with open(OUT / "step3c_tables.pkl", "rb") as f:
    T = pickle.load(f)

P = T["power"].copy()

PANELS = [
    ("Regression, ridge", "Regression, ridge", "(a)"),
    ("Ranking, logistic, 10.3% failures", "Ranking, logistic, 10.3% failures", "(b)"),
    ("Probability, logistic, 10.3% failures", "Probability, logistic, 10.3% failures", "(c)"),
    ("Ranking, logistic, 30% failures", "Ranking, logistic, 30% failures", "(d)"),
    ("Probability, logistic, 30% failures", "Probability, logistic, 30% failures", "(e)"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})

fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.0), sharey=True)
axs = axs.ravel()

styles = {
    "nndm_loo": dict(ls="-", marker="o", label="NNDM leave-one-out"),
    "random_single": dict(ls=":", marker="s", label="random 70/30 split"),
}

for ax, (test_name, title, letter) in zip(axs[:5], PANELS):
    for design in ["nndm_loo", "random_single"]:
        g = P[(P["test"] == test_name) & (P["design"] == design)].sort_values("a_covariate_share")

        if g.empty:
            raise RuntimeError(f"Missing Step3C data for {test_name!r}, {design!r}")

        x = g["a_covariate_share"].to_numpy(float)
        y = g["rate"].to_numpy(float)
        lo = g["wilson_low"].to_numpy(float)
        hi = g["wilson_high"].to_numpy(float)

        st = styles[design]
        ax.errorbar(
            x, y,
            yerr=np.vstack([y - lo, hi - y]),
            linestyle=st["ls"],
            marker=st["marker"],
            markersize=3.5,
            linewidth=1.2,
            capsize=2,
            label=st["label"],
        )

    ax.axhline(0.05, lw=0.7, ls="--")
    ax.axhline(0.80, lw=0.7, ls=":")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.015, 0.365)
    ax.set_xticks([0.00, 0.05, 0.10, 0.20, 0.35])
    ax.set_xlabel("covariate share a")
    ax.set_title(title, fontsize=8)
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=9)

axs[0].set_ylabel("detection rate")
axs[3].set_ylabel("detection rate")

axs[5].axis("off")
h, l = axs[0].get_legend_handles_labels()
axs[5].legend(h, l, loc="upper left", frameon=False)
axs[5].text(
    0.0, 0.55,
    "Horizontal guides:\n-- 0.05 nominal level\n:  0.80 reference detection rate",
    transform=axs[5].transAxes,
    va="top",
    fontsize=7.3,
)

fig.tight_layout()
png = OUT / "Fig_S5_final_NNDM_vs_random.png"
pdf = OUT / "Fig_S5_final_NNDM_vs_random.pdf"
fig.savefig(png, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(png)
print(pdf)

used = P[
    P["test"].isin([x[0] for x in PANELS])
    & P["design"].isin(["nndm_loo", "random_single"])
][
    ["test", "design", "a_covariate_share", "replicates",
     "rate", "wilson_low", "wilson_high"]
].sort_values(["test", "design", "a_covariate_share"])

print("\nRows used:")
print(used.round(3).to_string(index=False))
