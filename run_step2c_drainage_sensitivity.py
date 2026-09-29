"""NNDM sensitivity analysis for adding drainage density as a 14th covariate.

This script does NOT modify the primary Step 2 pipeline or its outputs.
It compares:
  - baseline_13: the original 13 covariates
  - plus_drainage_14: the same 13 covariates + drainage_density

Both variants use:
  - exactly the same boreholes
  - exactly the same NNDM folds
  - exactly the same seeds
  - the same fold-wise preprocessing, including median imputation

Drainage NoData values remain NaN and are therefore imputed *inside each
training fold* by the existing biabak.models preprocessor, exactly like other
continuous covariates.

Only covariate-based models are rerun because kriging and IDW do not use
environmental covariates and therefore cannot change when drainage is added.
"""

from __future__ import annotations

import argparse
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from biabak import config as C
from biabak import models as M
from biabak import validation as VAL

warnings.filterwarnings("ignore")


def sample_drainage(df: pd.DataFrame, raster_path: Path) -> np.ndarray:
    """Nearest-cell sampling in the raster's own CRS; invalid cells -> NaN."""
    vals = np.full(len(df), np.nan, dtype=float)

    with rasterio.open(raster_path) as r:
        tr = Transformer.from_crs(C.CRS_GEOGRAPHIC, r.crs, always_xy=True)
        xs, ys = tr.transform(
            df["Longitude"].to_numpy(float),
            df["Latitude"].to_numpy(float),
        )
        band = r.read(1)
        nodata = r.nodata

        for i, (x, y) in enumerate(zip(xs, ys)):
            row, col = r.index(x, y)
            if not (0 <= row < r.height and 0 <= col < r.width):
                continue
            v = band[row, col]
            if not np.isfinite(v):
                continue
            if nodata is not None and np.isclose(v, nodata):
                continue
            if v <= -1e30:
                continue
            vals[i] = float(v)

    return vals


def rank_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUC with average-rank tie handling, equivalent to Mann-Whitney AUC."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(scores, method="average")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_variant(task: str, d: pd.DataFrame, folds, variant: str,
                cont: list[str], cat: list[str]) -> list[dict]:
    """Run all covariate-based models for one variant on fixed NNDM folds."""
    old_cont = list(M.CONTINUOUS)
    old_cat = list(M.CATEGORICAL)

    try:
        M.CONTINUOUS[:] = cont
        M.CATEGORICAL[:] = cat

        if task == "regression":
            model_names = ["elastic_net"] + list(M.regressors(C.SEED).keys())
        else:
            model_names = list(M.classifiers(C.SEED).keys())

        preds = {m: np.full(len(d), np.nan, dtype=float) for m in model_names}
        base = np.full(len(d), np.nan, dtype=float)

        for f, (te, tr) in enumerate(folds):
            Xtr = d.iloc[tr]
            Xte = d.iloc[te]

            pre = M.preprocessor().fit(Xtr)
            Ztr = pre.transform(Xtr)
            Zte = pre.transform(Xte)

            if task == "regression":
                ytr = Xtr["log10_Sc"].to_numpy(float)
                base[te] = float(ytr.mean())

                en = M.SpatialENet(inner_k=5, seed=C.SEED + f).fit(
                    Ztr,
                    ytr,
                    xy=Xtr[["x_km", "y_km"]].to_numpy(float),
                )
                preds["elastic_net"][te] = en.predict(Zte)

                regs = M.regressors(C.SEED + f)
                for name in M.regressors(C.SEED).keys():
                    preds[name][te] = clone(regs[name]).fit(Ztr, ytr).predict(Zte)

            else:
                ytr = Xtr["unsuccessful"].to_numpy(int)
                base[te] = float(ytr.mean())

                clfs = M.classifiers(C.SEED + f)
                for name in M.classifiers(C.SEED).keys():
                    preds[name][te] = (
                        clone(clfs[name]).fit(Ztr, ytr).predict_proba(Zte)[:, 1]
                    )

        if task == "regression":
            y = d["log10_Sc"].to_numpy(float)
        else:
            y = d["unsuccessful"].to_numpy(int)

        rows = []
        for name, p in preds.items():
            ok = np.isfinite(p) & np.isfinite(base)
            row = {
                "task": task,
                "variant": variant,
                "model": name,
                "n": int(ok.sum()),
                "n_continuous": len(cont),
                "n_categorical": len(cat),
            }

            if task == "regression":
                yy, pp, bb = y[ok], p[ok], base[ok]
                row["r2_cv"] = float(
                    1 - np.sum((yy - pp) ** 2) / np.sum((yy - bb) ** 2)
                )
                row["bss"] = np.nan
                row["auc_raw"] = np.nan
                row["auc_excess_risk"] = np.nan
            else:
                yy, pp, bb = y[ok], p[ok], base[ok]
                row["r2_cv"] = np.nan
                row["bss"] = float(
                    1 - np.mean((pp - yy) ** 2) / np.mean((bb - yy) ** 2)
                )
                row["auc_raw"] = float(roc_auc_score(yy, pp))
                row["auc_excess_risk"] = rank_auc(yy, pp - bb)

            rows.append(row)

        return rows

    finally:
        M.CONTINUOUS[:] = old_cont
        M.CATEGORICAL[:] = old_cat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--drainage",
        required=True,
        help="Path to Drainage.tif",
    )
    ap.add_argument(
        "--include-record-level",
        action="store_true",
        help="Write borehole-level drainage values locally. Do not use for public redistribution when provider conditions prohibit row-level data sharing.",
    )
    args = ap.parse_args()

    drainage_path = Path(args.drainage)
    if not drainage_path.exists():
        raise FileNotFoundError(drainage_path)

    out_dir = C.OUT_DIR / "step2" / "drainage_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = pickle.load(open(C.OUT_DIR / "step1" / "step1_tables.pkl", "rb"))
    full = t1["S3_borehole_covariates"].reset_index(drop=True).copy()
    full["dist_fault_km"] = full["dist_fault_km"].astype(float)
    full["unsuccessful"] = 1 - full["productive"].astype(int)

    drainage = sample_drainage(full, drainage_path)
    full["drainage_density"] = drainage

    # Save an audit table showing exactly which records are missing drainage.
    audit_cols = [
        "ID", "N", "locality", "Latitude", "Longitude",
        "productive", "common_support", "drainage_density"
    ]
    audit = full[audit_cols].copy()
    if args.include_record_level:
        audit.to_csv(out_dir / "drainage_values_224.csv", index=False)

    print(
        "Drainage:",
        f"valid={int(np.isfinite(drainage).sum())}",
        f"missing={int(np.isnan(drainage).sum())}",
        f"min={np.nanmin(drainage):.6f}",
        f"max={np.nanmax(drainage):.6f}",
        f"mean={np.nanmean(drainage):.6f}",
        f"sd={np.nanstd(drainage):.6f}",
        flush=True,
    )

    dom = np.load(C.OUT_DIR / "step2" / "prediction_domain_xy_km.npy")

    cont13 = list(M.CONTINUOUS)
    cat13 = list(M.CATEGORICAL)
    cont14 = cont13 + ["drainage_density"]

    all_rows = []
    meta_rows = []

    tasks = {
        "regression": full[full["productive"] == 1].reset_index(drop=True),
        "classification": full.reset_index(drop=True),
    }

    for task, d in tasks.items():
        xy = d[["x_km", "y_km"]].to_numpy(float)

        # Build NNDM folds once, then reuse *the exact same folds* for both variants.
        folds, info = VAL.nndm_loo(xy, dom, phi=None, min_train=0.5)

        meta_rows.append({
            "task": task,
            "n": len(d),
            "n_folds": len(folds),
            "drainage_missing": int(d["drainage_density"].isna().sum()),
            "drainage_valid": int(d["drainage_density"].notna().sum()),
            "phi": info.get("phi", np.nan) if isinstance(info, dict) else np.nan,
            "median_train_n": float(np.median([len(tr) for te, tr in folds])),
            "min_train_n": int(min(len(tr) for te, tr in folds)),
            "max_train_n": int(max(len(tr) for te, tr in folds)),
        })

        for variant, cont in [
            ("baseline_13", cont13),
            ("plus_drainage_14", cont14),
        ]:
            t0 = time.time()
            rows = run_variant(task, d, folds, variant, cont, cat13)
            all_rows.extend(rows)
            print(
                f"{task} {variant}: {time.time() - t0:.0f}s",
                flush=True,
            )

    summary = pd.DataFrame(all_rows)
    meta = pd.DataFrame(meta_rows)

    # Pairwise deltas: 14-covariate metric minus 13-covariate metric.
    pairs = []
    metrics = ["r2_cv", "bss", "auc_raw", "auc_excess_risk"]
    for (task, model), g in summary.groupby(["task", "model"]):
        a = g[g["variant"] == "baseline_13"]
        b = g[g["variant"] == "plus_drainage_14"]
        if len(a) != 1 or len(b) != 1:
            continue
        row = {"task": task, "model": model}
        for metric in metrics:
            va = a.iloc[0][metric]
            vb = b.iloc[0][metric]
            row[f"{metric}_13"] = va
            row[f"{metric}_14"] = vb
            row[f"delta_{metric}_14_minus_13"] = (
                float(vb - va) if pd.notna(va) and pd.notna(vb) else np.nan
            )
        pairs.append(row)

    delta = pd.DataFrame(pairs)

    summary.to_csv(out_dir / "drainage_sensitivity_summary.csv", index=False)
    delta.to_csv(out_dir / "drainage_sensitivity_deltas.csv", index=False)
    meta.to_csv(out_dir / "drainage_sensitivity_metadata.csv", index=False)

    xlsx = out_dir / "drainage_sensitivity.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="summary", index=False)
        delta.to_excel(xw, sheet_name="deltas_14_minus_13", index=False)
        meta.to_excel(xw, sheet_name="metadata", index=False)
        if args.include_record_level:
            audit.to_excel(xw, sheet_name="drainage_values_224", index=False)

    print("\n=== DELTAS: 14 minus 13 ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(delta.to_string(index=False))

    print("\nSaved:")
    print(out_dir / "drainage_sensitivity_summary.csv")
    print(out_dir / "drainage_sensitivity_deltas.csv")
    print(out_dir / "drainage_sensitivity_metadata.csv")
    print(xlsx)


if __name__ == "__main__":
    main()
