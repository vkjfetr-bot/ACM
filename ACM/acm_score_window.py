#!/usr/bin/env python3
# acm_score_window.py
# Changelog
# 2025-10-06: Initial scorer (test/infer) — scores last N-min window from CSV; loads active artifacts from SQL; writes run, events, health.

import os, argparse, json, pandas as pd, numpy as np, datetime as dt
from acm_core import (
    ensure_time_index, slice_time, resample_guard, select_tags,
    build_features, Baseline, RegimeModel, train_isoforest, iso_score, ensemble_score,
    assign_state, state_distance, psi, page_hinkley, quantile_cut, build_spans, health_index, health_label,
    sql_connect, get_active_artifacts, write_run_and_health, write_events
)

# Usage:
#   set MSSQL_CNX=...
#   python acm_score_window.py --equipment EAF-1 --csv live.csv --tags "Power_kW,Temp_C,Pressure_bar" --start 2025-10-06T09:00:00 --end 2025-10-06T09:10:00 --contam 0.02

def rebuild_baseline_from_params(params: dict) -> Baseline:
    kind = params["kind"]
    # Note: for contextual kinds, med/mad/p01/p99 may be dict-of-hour/weekday; we keep as is (mapping).
    return Baseline(kind, params["med"], params["mad"], params["p01"], params["p99"])

def rebuild_regime_from_params(params: dict) -> RegimeModel:
    return RegimeModel(feat_cols=params["feat_cols"], centers=np.array(params["centers"]))

def main():
    ap = argparse.ArgumentParser(description="ACM Scorer (Scheduled Window)")
    ap.add_argument("--equipment", required=True)
    ap.add_argument("--csv", required=True, help="Window CSV with Ts + provided tags")
    ap.add_argument("--tags", required=True, help="Comma-separated tag names (no autodiscovery)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--contam", type=float, default=0.02)
    ap.add_argument("--min_span", type=int, default=60)
    ap.add_argument("--merge_gap", type=int, default=30)
    ap.add_argument("--write_points", action="store_true", help="(Optional) write point scores (implement later)")
    args = ap.parse_args()

    cnx_str = os.getenv("MSSQL_CNX")
    if not cnx_str:
        raise SystemExit("Set MSSQL_CNX env var for SQL connection.")
    cnx = sql_connect(cnx_str)

    # Load active artifacts
    artifacts = get_active_artifacts(cnx, args.equipment)
    if "BASELINE" not in artifacts or "REGIME" not in artifacts:
        raise SystemExit(f"No active artifacts for {args.equipment}. Run acm_build_artifacts.py first.")

    base = rebuild_baseline_from_params(artifacts["BASELINE"]["params"])
    regime = rebuild_regime_from_params(artifacts["REGIME"]["params"])
    tags_art = artifacts["BASELINE"]["params"]["tags"]  # authoritative list from artifacts

    # Load window CSV
    df = pd.read_csv(args.csv)
    df = ensure_time_index(df)
    df = slice_time(df, args.start, args.end)
    df = resample_guard(df)

    # Enforce provided tags and artifact tags agreement
    tags_req = [t.strip() for t in args.tags.split(",") if t.strip()]
    select_tags(df, tags_req)
    if set(tags_req) != set(tags_art):
        # Strict alignment: tags at scoring must match artifact tags for consistent features
        raise SystemExit(f"Tags mismatch. Artifacts have {tags_art}, provided {tags_req}")

    # Feature space
    F = build_features(df, tags_req)

    # States & distance
    states = assign_state(F, regime)
    sdist  = state_distance(F, regime)

    # Drift checks (cheap)
    drift = {}
    try:
        for t in tags_req[:5]:
            drift[f"psi_{t}"] = psi(pd.to_numeric(df[t], errors="coerce").iloc[:len(df)//2],
                                    pd.to_numeric(df[t], errors="coerce").iloc[len(df)//2:])
        drift["page_hinkley_state_dist"] = page_hinkley(sdist)
    except Exception:
        drift = {}

    # Anomaly ensemble
    ifmods = train_isoforest(F, contamination=args.contam)
    iso_s  = iso_score(F, ifmods)
    score  = ensemble_score(df, tags_req, base, F, iso_s, sdist)

    # Threshold + spans
    q = quantile_cut(score, 1 - args.contam)
    hit = score >= q if isinstance(q, float) and not np.isnan(q) else pd.Series(False, index=score.index)
    spans = build_spans(hit, min_len_s=args.min_span, merge_gap_s=args.merge_gap)

    # Health
    hi = health_index(score, spans)
    hlabel = health_label(hi)

    # Persist run summary + health
    run_id = write_run_and_health(
        cnx, args.equipment,
        window_start=df.index.min().to_pydatetime(),
        window_end=df.index.max().to_pydatetime(),
        rows_proc=int(len(df)),
        tags_used=len(tags_req),
        anomaly_frac=float(hit.mean()),
        hi=hi, hlabel=hlabel,
        drift_json=drift,
        baseline_ver=int(artifacts["BASELINE"]["version"]),
        regime_ver=int(artifacts["REGIME"]["version"])
    )

    # Events
    # Top contributors (simple heuristic: highest |z_hist| columns in the window)
    top_tags = sorted(tags_req, key=lambda t: float(np.nanpercentile(np.abs((pd.to_numeric(df[t], errors="coerce") - np.nanmedian(pd.to_numeric(df[t], errors="coerce"))) / (np.nanmedian(np.abs(pd.to_numeric(df[t], errors="coerce") - np.nanmedian(pd.to_numeric(df[t], errors="coerce")))) or np.nan)), 95)), reverse=True)
    write_events(cnx, run_id, args.equipment, spans, score, states, top_tags)

    # Output for logs
    out = {
        "equipment": args.equipment,
        "rows": int(len(df)),
        "tags": tags_req,
        "anomaly_fraction": float(hit.mean()),
        "health_index": round(hi,2),
        "health_label": hlabel,
        "events": len(spans),
        "baseline_version": int(artifacts["BASELINE"]["version"]),
        "regime_version": int(artifacts["REGIME"]["version"])
    }
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
