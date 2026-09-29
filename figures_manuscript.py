"""Figures 1 and 2 of the manuscript (protocol diagram and study area)."""
import pickle, numpy as np, pandas as pd, shapefile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure as _mf
_orig_save = _mf.Figure.savefig
def _save_png_and_pdf(self, fname, *a, **k):
    _orig_save(self, fname, *a, **k)
    f = str(fname)
    if f.endswith(".png"):
        _orig_save(self, f[:-4] + ".pdf", *a, **k)
_mf.Figure.savefig = _save_png_and_pdf

from matplotlib.patches import FancyBboxPatch, Circle, Polygon as MPoly
from matplotlib.collections import PatchCollection, LineCollection
from shapely.geometry import shape
from pyproj import Transformer
from biabak import config as C
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 9, "savefig.dpi": 300})
OUT = C.OUT_DIR / "manuscript_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- Figure 1: the audit protocol ----------------
fig, ax = plt.subplots(figsize=(7.4, 4.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
cols_x = [1, 21, 41, 61, 81]; W = 18
heads = ["Inputs", "Diagnostics", "Validation", "Inference", "Reporting"]
boxes = {0: [("point observations", False), ("pre-decision" + chr(10) + "covariates", False), ("prediction domain", False)],
         1: [("archive audit", False), ("structure tested" + chr(10) + "against chance", False)],
         2: [("leakage-controlled\nfitting", False), ("distance-matched\ndesigns (NNDM)", False)],
         3: [("skill vs fold\nbaselines", False), ("excess-risk AUC", True), ("area of\napplicability", False), ("detection\nbenchmark", True)],
         4: [("generalization\ndistances", False), ("skill with\nintervals", False), ("detection limits", True)]}
for c, x in enumerate(cols_x):
    ax.add_patch(FancyBboxPatch((x, 54), W, 6, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#34495e", ec="none"))
    ax.text(x + W / 2, 57, heads[c], ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    items = boxes[c]; n = len(items); hbox = 9.5; gap = (52 - n * hbox) / (n + 1)
    for k, (txt, new_) in enumerate(items):
        y = 52 - (k + 1) * gap - (k + 1) * hbox
        ax.add_patch(FancyBboxPatch((x, y), W, hbox, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    fc="#fdebd0" if new_ else "#eaf0f6", ec="#d35400" if new_ else "#5d6d7e", lw=1.4 if new_ else 0.8))
        ax.text(x + W / 2, y + hbox / 2, txt, ha="center", va="center", fontsize=7.6, linespacing=1.15)
    if c < 4:
        ax.annotate("", xy=(cols_x[c + 1] - 0.3, 30), xytext=(x + W + 0.3, 30), arrowprops=dict(arrowstyle="-|>", color="#34495e", lw=1.2))
ax.add_patch(FancyBboxPatch((1, 0.5), 3, 2.2, boxstyle="round,pad=0.2", fc="#fdebd0", ec="#d35400", lw=1.4))
ax.text(5.5, 1.6, "component introduced in this study", va="center", fontsize=7.5)
fig.savefig(f"{OUT}/Figure_1.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 2: study area ----------------
T1 = pickle.load(open(C.OUT_DIR / "step1" / "step1_tables.pkl", "rb"))
bh = T1["S3_borehole_covariates"]
geo = shapefile.Reader(str(C.GEOLOGY_DIR / "Formations géologiques"), encoding="utf-8", encodingErrors="replace")
fields = [f[0] for f in geo.fields[1:]]
namefield = [f for f in fields if "nom" in f.lower() or "name" in f.lower() or "form" in f.lower()]
namefield = namefield[0] if namefield else fields[0]
recs = geo.records()
import unicodedata
def norm(t):
    return unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower().strip()
GROUPS = [("Quartzites and schists", "#f2e6a0", ["quartzite", "schistes, quartz"]),
          ("Micaschists", "#c9b07a", ["micashiste", "micaschiste"]),
          ("Mafic rocks (amphibolites, diorites, gabbros, mafic granulites)", "#6aa56a", ["amphibolite", "diorite", "gabbro", "granulites mafiques", "ultrab"]),
          ("Migmatitic gneisses", "#e39c9c", ["migmati"]),
          ("Paragneisses and other gneisses", "#c79fd6", ["paragneiss", "granulitiques a grenat", "leucocrate"]),
          ("Orthogneisses and TTG gneisses", "#9ec3e6", ["orthogneiss", "ttg"]),
          ("Granitoids and syenites", "#f4b183", ["granit", "monzon", "syenit", "granodiorit", "leucogranit"])]
def group(name):
    n = norm(name)
    if "granodiorit" in n:
        return "Granitoids and syenites"
    for lab, col, keys in GROUPS:
        if any(k in n for k in keys):
            return lab
    return "Other"
colmap = {lab: col for lab, col, _ in GROUPS}; colmap["Other"] = "#dddddd"
fig, ax = plt.subplots(figsize=(6.6, 6.2))
patches, cols = [], []
for k, r in enumerate(recs):
    try:
        s = geo.shape(k)
    except Exception:
        continue
    if not s.points:
        continue
    pts = np.array(s.points); parts = list(s.parts) + [len(pts)]
    for a, b in zip(parts[:-1], parts[1:]):
        patches.append(MPoly(pts[a:b], closed=True)); cols.append(colmap[group(r[namefield])])
pc = PatchCollection(patches, facecolor=cols, edgecolor="white", linewidth=0.15, alpha=0.75); ax.add_collection(pc)
used = sorted({group(r[namefield]) for r in recs})
flt = shapefile.Reader(str(C.FAULT_SHP))
segs = [np.array(s.points) for s in flt.shapes()]
ax.add_collection(LineCollection(segs, colors="#8b0000", linewidths=0.9, label="mapped faults"))
cs = bh.common_support == 1
p = cs & (bh.productive == 1); u = bh.productive == 0; ex = ~cs
# Public-release figure: individual borehole locations are omitted because the
# provider conditions do not permit redistribution of record-level coordinates.
ax.plot([], [], linestyle="none", marker="", label=f"borehole locations withheld (n={len(bh)})")
lat0, lon0 = C.CLUSTER_CENTRE_LATLON
r_deg = C.CLUSTER_RADIUS_KM / 111.0
ax.add_patch(plt.matplotlib.patches.Ellipse((lon0, lat0), 2 * r_deg / np.cos(np.deg2rad(lat0)), 2 * r_deg, fill=False, ls="--", lw=1.0, ec="k"))
ax.plot(lon0, lat0, marker="*", ms=10, c="gold", mec="k", zorder=6); ax.text(lon0 + 0.18, lat0 - 0.14, "Yaoundé", fontsize=8)
ax.plot([], [], ls="--", c="k", label=f"dense cluster ({C.CLUSTER_RADIUS_KM:.0f} km radius)")
ax.set_xlim(10.3, 13.2); ax.set_ylim(3.0, 6.35); ax.set_aspect(1 / np.cos(np.deg2rad(4.5)))
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
x0, y0 = 11.95, 3.15; L = 100 / (111.32 * np.cos(np.deg2rad(3.2)))
ax.plot([x0, x0 + L], [y0, y0], c="k", lw=2); ax.text(x0 + L / 2, y0 + 0.05, "100 km", ha="center", fontsize=8)
ax.annotate("N", xy=(13.05, 6.2), xytext=(13.05, 5.9), ha="center", arrowprops=dict(arrowstyle="-|>", color="k"), fontsize=9)
leg1 = ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=7.2)
ax.add_artist(leg1)
from matplotlib.patches import Patch
hl = [Patch(facecolor=colmap[lab], edgecolor="none", alpha=0.75, label=lab) for lab, _, _ in GROUPS if lab in used]
ax.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=7.2, title="Lithological groups", title_fontsize=7.5)
fig.savefig(f"{OUT}/Figure_2.png", bbox_inches="tight"); plt.close(fig)
print("groups used:", used, "other:", sum(group(r[namefield]) == "Other" for r in recs))
