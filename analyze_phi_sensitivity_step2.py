import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from biabak import config as C

OUT = C.OUT_DIR / "step2" / "phi_sensitivity"
files = sorted(glob.glob(str(OUT / "checkpoints" / "*.pkl")))
P = pd.concat([pickle.load(open(f, "rb"))[0] for f in files], ignore_index=True)

rows = []
for (task, design, model), g in P.groupby(["task", "design", "model"]):
    if model in ("training_mean", "training_prevalence"):
        continue
    y = g.y.values
    pred = g.pred.values
    base = g.baseline.values
    r = dict(task=task, design=design, model=model)
    if task == "regression":
        r["r2_cv"] = 1 - np.sum((y - pred) ** 2) / np.sum((y - base) ** 2)
    else:
        r["bss"] = 1 - np.mean((pred - y) ** 2) / np.mean((base - y) ** 2)
        r["auc_excess_risk"] = roc_auc_score(y, pred - base) if len(np.unique(y)) == 2 else np.nan
    rows.append(r)

S = pd.DataFrame(rows)
S.to_csv(OUT / "phi_sensitivity_summary.csv", index=False)
with pd.ExcelWriter(OUT / "phi_sensitivity_summary.xlsx", engine="openpyxl") as xw:
    S.to_excel(xw, sheet_name="summary", index=False)

print(S.round(3).to_string(index=False))
print("\nSaved:", OUT / "phi_sensitivity_summary.csv")
print("Saved:", OUT / "phi_sensitivity_summary.xlsx")
