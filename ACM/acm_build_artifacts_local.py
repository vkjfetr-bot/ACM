#!/usr/bin/env python3
# acm_build_artifacts_local.py
# Changelog
# 2025-10-07: Local-files artifact builder: reads history CSV, builds contextual baselines + regimes, saves to acm_artifact.csv.

import argparse, json, numpy as np, pandas as pd
from acm_core_local_2 import (
    ensure_time_index, slice_time, resample_guard, enforce_tags,
    build_baseline, train_regimes, artifact_insert
)

# Usage:
# python acm_build_artifacts_local.py --equipment "FD-FAN-1" --csv "/mnt/data/FD FAN TRAINING DATA.csv" \
#   --tags "Power_kW,Temp_C,Pressure_bar,Vibration_mmps" --context hour --k 4 --version 1

def main():
    ap = argparse.ArgumentParser(description="ACM Artifact Builder (Local CSV)")
    ap.add_argument("--equipment", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--context", choices=["global","hour","weekday"], default="hour")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--version", type=int, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = ensure_time_index(df)
    if args.start or args.end: df = slice_time(df, args.start, args.end)
    df = resample_guard(df)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    enforce_tags(df, tags)

    trained_from, trained_to = df.index.min(), df.index.max()

    base = build_baseline(df, tags, kind=args.context)
    base_params = {
        "kind": base.kind,
        "med": {k: (float(v) if isinstance(v, (int,float,np.floating)) else dict(v)) for k,v in base.med.items()},
        "mad": {k: (float(v) if isinstance(v, (int,float,np.floating)) else dict(v)) for k,v in base.mad.items()},
        "p01": {k: (float(v) if isinstance(v, (int,float,np.floating)) else dict(v)) for k,v in base.p01.items()},
        "p99": {k: (float(v) if isinstance(v, (int,float,np.floating)) else dict(v)) for k,v in base.p99.items()},
        "tags": tags
    }

    regimes = train_regimes(df, tags, k=args.k)
    reg_params = {"feat_cols": regimes.feat_cols, "centers": regimes.centers.tolist(), "tags": tags}

    artifact_insert(args.equipment, "BASELINE", args.version, trained_from, trained_to, base_params, active=True)
    artifact_insert(args.equipment, "REGIME",   args.version, trained_from, trained_to, reg_params,  active=True)

    print(json.dumps({"status":"ok","equipment":args.equipment,"baseline_version":args.version,"regime_version":args.version}, ensure_ascii=False))

if __name__ == "__main__":
    main()
