"""Borehole loading, projection and covariate extraction.

All covariates are re-extracted from the processed rasters so that the whole
chain from raster to predictor matrix is reproducible. The pixel containing the
borehole is read; when that pixel is empty, the mean (continuous) or the mode
(categorical) of valid pixels within FALLBACK_RADIUS_PIXELS is used and the
case is flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
import shapefile
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform, unary_union

from . import config as C


def load_boreholes() -> pd.DataFrame:
    d = pd.read_excel(C.BOREHOLE_XLSX, sheet_name=C.BOREHOLE_SHEET)
    d = d.rename(columns={"N°": "N", "Localité": "locality", "debit": "Q_m3h",
                          "Pt": "total_depth_m", "EA": "weathered_m", "ES": "basement_m",
                          "NS": "static_level_m", "AE1": "first_strike_m",
                          "T": "T_m2s", "K": "K_ms", "Qsp": "Sc_m3hm",
                          "Altitude": "altitude_record_m"})
    tr = Transformer.from_crs(C.CRS_GEOGRAPHIC, C.CRS_ANALYSIS, always_xy=True)
    x, y = tr.transform(d["Longitude"].values, d["Latitude"].values)
    d["x_km"] = np.asarray(x) / 1000.0
    d["y_km"] = np.asarray(y) / 1000.0
    d["productive"] = (d["Q_m3h"] > 0).astype(int)
    d["unsuccessful"] = 1 - d["productive"]
    d["common_support"] = (~d["N"].isin(C.EXCLUDED_FROM_COMMON_SUPPORT)).astype(int)
    pos = d["productive"] == 1
    d["log10_Sc"] = np.where(pos, np.log10(d["Sc_m3hm"].where(pos, np.nan)), np.nan)
    d["log10_T"] = np.where(pos, np.log10(d["T_m2s"].where(pos, np.nan)), np.nan)
    return d


def _sample_raster(path, lon, lat, categorical: bool):
    with rasterio.open(path) as r:
        tr = Transformer.from_crs(C.CRS_GEOGRAPHIC, r.crs, always_xy=True)
        xs, ys = tr.transform(lon, lat)
        arr = r.read(1)
        nod = r.nodata
        invalid = ~np.isfinite(arr) | (arr < -1e30)
        if nod is not None:
            invalid |= arr == nod
        vals, flags = [], []
        for x, y in zip(xs, ys):
            row, col = r.index(x, y)
            inside = 0 <= row < r.height and 0 <= col < r.width
            if inside and not invalid[row, col]:
                vals.append(float(arr[row, col])); flags.append("pixel")
                continue
            got = None
            for rad in range(1, C.FALLBACK_RADIUS_PIXELS + 1):
                r0, r1 = max(row - rad, 0), min(row + rad + 1, r.height)
                c0, c1 = max(col - rad, 0), min(col + rad + 1, r.width)
                if r0 >= r1 or c0 >= c1:
                    continue
                win = arr[r0:r1, c0:c1]
                ok = ~invalid[r0:r1, c0:c1]
                if ok.any():
                    v = win[ok].astype(float)
                    if categorical:
                        u, cnt = np.unique(v, return_counts=True)
                        got = float(u[np.argmax(cnt)])
                    else:
                        got = float(v.mean())
                    flags.append(f"fallback_{rad}px")
                    break
            if got is None:
                vals.append(np.nan); flags.append("missing")
            else:
                vals.append(got)
    return np.array(vals), flags


def extract_raster_covariates(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.DataFrame(index=d.index)
    log = pd.DataFrame(index=d.index)
    for col, (stem, kind) in C.RASTER_COVARIATES.items():
        v, f = _sample_raster(C.RASTER_DIR / f"{stem}.tif", d["Longitude"].values,
                              d["Latitude"].values, kind == "categorical")
        out[col] = v
        log[col] = f
    return out, log


def soil_and_fault(d: pd.DataFrame) -> pd.DataFrame:
    tr = Transformer.from_crs(C.CRS_GEOGRAPHIC, C.CRS_ANALYSIS, always_xy=True).transform
    pts = [transform(tr, Point(lo, la)) for lo, la in zip(d["Longitude"], d["Latitude"])]
    rs = shapefile.Reader(str(C.SOIL_SHP), encoding="utf-8")
    polys = [(transform(tr, shape(s.__geo_interface__)), rec["DOMSOI"], rec["FAOSOIL"])
             for s, rec in zip(rs.shapes(), rs.records())]
    dom, fao, how = [], [], []
    for p in pts:
        hit = [(g, a, b) for g, a, b in polys if g.contains(p)]
        if hit:
            dom.append(hit[0][1]); fao.append(hit[0][2]); how.append("inside")
        else:
            g, a, b = min(polys, key=lambda t: t[0].distance(p))
            dom.append(a); fao.append(b); how.append(f"nearest_{g.distance(p):.0f}m")
    rf = shapefile.Reader(str(C.FAULT_SHP))
    faults = unary_union([transform(tr, shape(s.__geo_interface__)) for s in rf.shapes()])
    return pd.DataFrame({"soil_dominant": dom, "soil_fao_unit": fao, "soil_assignment": how,
                         "dist_fault_km": [p.distance(faults) / 1000 for p in pts]},
                        index=d.index)


def lithology_names() -> dict:
    import struct
    # minimal DBF reader for the raster attribute table (no extra dependency)
    path = C.RASTER_DIR / "Geol_raster.tif.vat.dbf"
    with open(path, "rb") as fh:
        head = fh.read(32)
        nrec, hlen, rlen = struct.unpack("<xxxxIHH20x", head)
        fields = []
        while True:
            b = fh.read(32)
            if b[0] == 0x0D:
                break
            name = b[:11].split(b"\x00")[0].decode()
            fields.append((name, b[16]))
        fh.seek(hlen)
        rows = []
        for _ in range(nrec):
            rec = fh.read(rlen)[1:]
            pos, row = 0, {}
            for name, ln in fields:
                row[name] = rec[pos:pos + ln].decode("utf-8", "ignore").strip()
                pos += ln
            rows.append(row)
    return {int(float(r["Value"])): r["name"] for r in rows}


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = load_boreholes()
    cov, log = extract_raster_covariates(d)
    extra = soil_and_fault(d)
    full = pd.concat([d, cov, extra], axis=1)
    names = lithology_names()
    full["lithology_name"] = full["lithology_code"].map(lambda v: names.get(int(v)) if np.isfinite(v) else None)
    return full, log
