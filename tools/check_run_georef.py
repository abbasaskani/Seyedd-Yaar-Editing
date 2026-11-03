#!/usr/bin/env python3
"""Validate generated scientific BINs, timestamps, dimensions and coordinate contract.

Usage (after GitHub Action or local run):
    python tools/check_run_georef.py docs/latest

Prints a JSON report and exits 1 on structural errors. This cannot validate
native Copernicus coastline alignment without the original NetCDF inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("latest", nargs="?", default="docs/latest", type=Path)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    errors, warnings = [], []
    latest = args.latest
    index_file = latest / "meta_index.json"
    if not index_file.is_file():
        print(json.dumps({"ok": False, "errors": [f"Missing {index_file}"]}, indent=2))
        return 1
    index = json.loads(index_file.read_text(encoding="utf-8"))
    run = next((r for r in index["runs"] if r["run_id"] == index["latest_run_id"]), None)
    if run is None:
        errors.append("Latest run id not present in meta_index.runs")
        report = {"ok": False, "errors": errors}
        print(json.dumps(report, indent=2))
        return 1

    run_root = latest / run["path"]
    run_meta = json.loads((run_root / "meta.json").read_text(encoding="utf-8"))
    grid = run_meta["grid"]
    W,H = int(grid["width"]), int(grid["height"])
    n = W*H
    if W < 2 or H < 2:
        errors.append("Grid dimensions must both be >=2")
    for c in ("lat_min", "lat_max", "lon_min", "lon_max"):
        if not np.isfinite(float(grid[c])):
            errors.append(f"Missing finite grid {c}")
    tids = run_meta.get("available_time_ids") or run_meta.get("time_ids") or []
    if not tids:
        errors.append("No available timestamps in run meta")
    if tids != sorted(set(tids)):
        errors.append("Time IDs are not unique and chronological")
    species_report=[]
    for variant in run["variants"]:
        for species in run["species"]:
            sp_root=run_root / "variants" / variant / "species" / species
            if not (sp_root / "meta.json").is_file():
                errors.append(f"Missing species meta {sp_root}")
                continue
            sp_meta=json.loads((sp_root / "meta.json").read_text(encoding="utf-8"))
            if sp_meta["grid"] != grid:
                errors.append(f"Grid mismatch for {variant}/{species}")
            mask=np.fromfile(sp_root / "mask_u8.bin", dtype=np.uint8)
            if mask.size != n:
                errors.append(f"Invalid AOI mask length {mask.size} expected {n}: {sp_root}")
                continue
            if not np.isin(mask, [0,1]).all():
                errors.append(f"Unexpected AOI mask values in {sp_root}")
            template=sp_meta["paths"]["per_time"]
            summary={"variant":variant,"species":species,"time_count":len(tids),
                     "checks":[],"aoi_valid_cells":int(mask.sum())}
            for tid in tids:
                item={"time_id":tid}
                for key in ("pcatch_scoring","pcatch_ensemble","phab_scoring","pops","conf","qc_chl"):
                    rel=template.get(key)
                    if not rel:
                        warnings.append(f"Missing optional template for {key} in {variant}/{species}")
                        continue
                    file=run_root / rel.replace("{time}",tid)
                    if not file.is_file():
                        errors.append(f"Missing BIN: {file}")
                        continue
                    arr=np.fromfile(file,dtype=np.uint8 if key=="qc_chl" else np.float32)
                    if arr.size != n:
                        errors.append(f"Wrong BIN length for {file}: {arr.size}, expected {n}")
                        continue
                    if key in ("pcatch_ensemble","conf"):
                        valid=np.isfinite(arr) & (mask>0)
                        item[key+"_valid_cells"]=int(valid.sum())
                        if np.any((arr[valid] < -1e-5) | (arr[valid] > 1+1e-5)):
                            errors.append(f"Values outside [0,1] in {file}")
                    if key=="qc_chl" and not np.isin(arr,[0,1]).all():
                        errors.append(f"Invalid QC values in {file}")
                summary["checks"].append(item)
            species_report.append(summary)
    report={"ok":not errors,"run_id":run["run_id"],"grid":grid,
            "grid_convention":"EPSG:4326; centre samples; first row north",
            "species":species_report,"warnings":warnings,"errors":errors,
            "limitation":"Spatial agreement with real Socotra coastline requires original, georeferenced NetCDF layers and coastline reference."}
    output=json.dumps(report,ensure_ascii=False,indent=2)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(output,encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
