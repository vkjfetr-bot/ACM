#!/usr/bin/env python3
# acm_build_artifacts.py
# Changelog
# 2025-10-06: Initial artifact builder (train) — builds contextual baselines & regimes from history CSV; writes to SQL artifacts.

import os, argparse, json, pandas as pd, numpy as np, datetime as dt
from acm_core import (ensure_time_index, slice_time, resample_guard, select_tags,
                      build_baseline, train_regimes, sql_connect, upsert_artifact)

# Usage:
#   set MSSQL_CNX="DRIVER={ODBC Driver 17 for SQL Server};SERVER=...;DATABASE=...;Trusted_Connection=yes;"
#   python acm_build_artifacts.py --equipment EAF-1 --csv history.csv --tags "Power_kW,Temp_C,Pressure_bar" --context hour --version 1

def main():
    ap = argparse.ArgumentParser(description="ACM Artifact Builder (Train)")
    ap.add_argument("--equipment", required=True)
    ap.add_argument("--csv", required=True, help="History CSV with Ts + provided tags")
    ap.add_argument("--tags", required=True, help="Comma-separated tag names (no autodiscovery)")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--context", choices=["global","hour","weekday"], default="hour")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--version", type=int, required=True, help="Artifact version to set active")
    args = ap.parse_args()

    cnx_str = os.getenv("MSSQL_CNX")
    if not cnx_str:
        raise SystemExit("Set MSSQL_CNX env var for SQL connection.")
    cnx = sql_connect(cnx_str)

    df = pd.read_csv(args.csv)
    df = ensure_time_index(df)
    if args.start or args.end:
        df = slice_time(df, args.start, args.end)
    df = resample_guard(df)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    select_tags(df, tags)

    trained_from, trained_to = df.index.min().to_pydatetime(), df.index.max().to_pydatetime()

    # Baseline
    base = build_baseline(df, tags, kind=args.context)
    base_params = {
        "kind": base.kind,
        "med": {k: (float(v) if isinstance(v, (int,float,np.floating)) else dict(v)) for k,v in base.med.items()},
        "mad": {k: (float(v) if isinstance(v, (int,float,np.floating)) else dict(v)) for k,v in base.mad.items()},
        "p01": {k: (float(v) if isinstance(v, (int,float,np.floating)) else dict(v)) for k,v in base.p01.items()},
        "p99": {k: (float(v) if isinstance(v, (int,float,np.floating)) else dict(v)) for k,v in base.p99.items()},
        "tags": tags
    }

    # Regimes
    regimes = train_regimes(df, tags, k=args.k)
    reg_params = {
        "feat_cols": regimes.feat_cols,
        "centers": regimes.centers.tolist(),
        "tags": tags
    }

    # Persist artifacts as Active=1 for this version
    upsert_artifact(cnx, args.equipment, "BASELINE", trained_from, trained_to, base_params, active=True, version=args.version)
    upsert_artifact(cnx, args.equipment, "REGIME",   trained_from, trained_to, reg_params,  active=True, version=args.version)
    print(json.dumps({"status":"ok","equipment":args.equipment,"baseline_version":args.version,"regime_version":args.version}))

if __name__ == "__main__":
    main()
