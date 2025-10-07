#!/usr/bin/env python3
# acm_score_window_local.py
# Changelog
# 2025-10-07: Local-files scorer: reads window CSV, loads active artifacts from acm_artifact.csv, writes run/events/health CSVs.

import argparse, json, numpy as np, pandas as pd
from acm_core_local import (
    ensure_time_index, slice_time, resample_guard, enforce_tags,
    build_features, Baseline, RegimeModel, train_isoforest, iso_score, ensemble_score,
    assign_state, state_distance, psi, page_hinkley, quantile_cut, build_spans, health_index, health_label,
    artifacts_get_active, run_summary_insert, events_insert, asset_health_upsert
)

# Usage:
# python acm_score_window_local.py --equipment "FD-FAN-1" --csv "/mnt/data/FD FAN TEST DATA.csv" \
#   --tags "Power_kW,Temp_C,Pressure_bar,Vibration_mmps" --start 2025-10-07T10:00:00 --end 2025-10-07T10:15:00 --contam 0.02

def rebuild_baseline(params: dict) -> Baseline:
    return Baseline(params["kind"], params["med"], params["mad"], params["p01"], params["p99"])

def rebuild_regime(params: dict) -> RegimeModel:
    import numpy as np
    return RegimeModel(feat_cols=params["feat_cols"], centers=np.array(params["centers"]))

def main():
    ap = argparse.ArgumentParser(description="ACM Scorer (Local CSV)")
    ap.add_argument("--equipment", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--contam", type=float, default=0.02)
    ap.add_argument("--min_span", type=int, default=60)
    ap.add_argument("--merge_gap", type=int, default=30)
    args = ap.parse_args()

    arts = artifacts_get_active(args.equipment)
    if "BASELINE" not in arts or "REGIME" not in arts:
        raise SystemExit(f"No active artifacts for {args.equipment}. Run acm_build_artifacts_local.py first.")
    base = rebuild_baseline(arts["BASELINE"]["params"])
    regime = rebuild_regime(arts["REGIME"]["params"])
    tags_art = arts["BASELINE"]["params"]["tags"]

    df = pd.read_csv(args.csv)
    df = ensure_time_index(df)
    df = slice_time(df, args.start, args.end)
    df = resample_guard(df)

    tags_req = [t.strip() for t in args.tags.split(",") if t.strip()]
    enforce_tags(df, tags_req)
    if set(tags_req) != set(tags_art):
        raise SystemExit(f"Tags mismatch. Artifacts have {tags_art}, provided {tags_req}")

    F = build_features(df, tags_req)
    states = assign_state(F, regime)
    sdist  = state_distance(F, regime)

    drift = {}
    try:
        for t in tags_req[:5]:
            s = pd.to_numeric(df[t], errors="coerce")
            drift[f"psi_{t}"] = psi(s.iloc[:len(s)//2], s.iloc[len(s)//2:])
        drift["page_hinkley_state_dist"] = page_hinkley(sdist)
    except Exception:
        drift = {}

    ifmods = train_isoforest(F, contamination=args.contam)
    iso_s  = iso_score(F, ifmods)
    score  = ensemble_score(df, tags_req, base, F, iso_s, sdist)

    q = quantile_cut(score, 1 - args.contam)
    hit = score >= q if isinstance(q, float) and not np.isnan(q) else pd.Series(False, index=score.index)
    spans = build_spans(hit, min_len_s=args.min_span, merge_gap_s=args.merge_gap)

    hi = health_index(score, spans)
    hlabel = health_label(hi)
    last_peak = float(score.max()) if len(score) else None
    run_id = run_summary_insert(
        args.equipment,
        df.index.min(), df.index.max(),
        rows_proc=int(len(df)), tags_used=len(tags_req),
        anomaly_frac=float(hit.mean()), hi=hi, hlabel=hlabel,
        drift_json=drift,
        baseline_ver=int(arts["BASELINE"]["version"]),
        regime_ver=int(arts["REGIME"]["version"])
    )
    # naive top contributors by |z_hist| (window-level)
    contrib = []
    for t in tags_req:
        s = pd.to_numeric(df[t], errors="coerce")
        # robust z using window median/MAD (ok for quick attribution)
        m = float(np.nanmedian(s)); mad = float(np.nanmedian(np.abs(s - m))) or np.nan
        zh = (s - m) / mad
        contrib.append((t, float(np.nanpercentile(np.abs(zh), 95))))
    top_tags = [t for t,_ in sorted(contrib, key=lambda x: x[1], reverse=True)]

    events_insert(run_id, args.equipment, spans, score, states, top_tags)
    asset_health_upsert(args.equipment, df.index.min(), df.index.max(), hi, hlabel, float(hit.mean()),
                        int(arts["BASELINE"]["version"]), int(arts["REGIME"]["version"]),
                        last_event_peak=last_peak, drift_flag=bool(drift))

    print(json.dumps({
        "equipment": args.equipment,
        "rows": int(len(df)),
        "tags": tags_req,
        "anomaly_fraction": float(hit.mean()),
        "health_index": round(hi,2),
        "health_label": hlabel,
        "events": len(spans),
        "baseline_version": int(arts["BASELINE"]["version"]),
        "regime_version": int(arts["REGIME"]["version"])
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
