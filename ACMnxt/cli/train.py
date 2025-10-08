"""acmnxt train --csv --equip --out-dir

M2 implementation: IO + DQ + H1/H2 + fused + events.
Steps:
- Load CSV/XLSX
- Ensure DateTimeIndex (UTC)
- clean_time -> resample_numeric
- compute_dq -> write dq.csv and dq.png
- Fit PCA (H2) -> compute H1 and H2 scores
- Fuse -> build events (MAD)
- Write scores.csv and events.jsonl under <out-dir>/<equip_sanitized>/
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd

from acmnxt.io.loaders import read_table, ensure_datetime_index
from acmnxt.io.writers import write_csv
from acmnxt.core.dq import clean_time, resample_numeric, compute_dq
from acmnxt.vis.dq import plot_dq_heatmap
from acmnxt.core.h1_ar1 import score_h1
from acmnxt.core.h2_pca import fit_pca, score_h2
from acmnxt.core.fusion import fuse_scores
from acmnxt.core.events import build_events
from acmnxt.core.regimes import fit_assign_regimes
from acmnxt.vis.regimes import plot_regime_strip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--equip", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fast", action="store_true", help="Enable fast mode: subsample for PCA/regimes and quicker DQ plot")
    args = ap.parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    equip = args.equip
    equip_s = "".join(ch if ch.isalnum() else "_" for ch in equip)
    out = out_root / equip_s
    (out / "assets").mkdir(parents=True, exist_ok=True)
    print(f"[train:M2] load csv={args.csv} equip={equip} -> {out} fast={args.fast}")

    # Load and index
    df_raw = read_table(args.csv)
    df = ensure_datetime_index(df_raw, ts_col="Ts")

    # Clean and resample
    df = clean_time(df)
    df = resample_numeric(df, rule="1min")

    # Compute DQ
    metrics, dq_bad = compute_dq(df)

    # Write DQ artifacts
    write_csv(metrics, out / "dq.csv")
    dq_plot = out / "assets" / "dq.png"
    plot_dq_heatmap(df, dq_plot)
    print(f"[train:M2] wrote {out / 'dq.csv'} and {dq_plot}")

    # H1/H2 (impute NaNs for modeling)
    df_imp = df.ffill().bfill()
    if df_imp.isna().any().any():
        df_imp = df_imp.fillna(df_imp.mean(numeric_only=True))
    # Subsample for PCA fit if fast
    df_fit = df_imp
    if args.fast and len(df_imp) > 50000:
        df_fit = df_imp.sample(n=50000, random_state=42)
    arts = fit_pca(df_fit, n_components=min(5, max(1, min(df_fit.shape) - 1)), out_dir=out)
    h1 = score_h1(df_imp)
    h2 = score_h2(df_imp, artifacts=arts)

    # Fuse
    fused = fuse_scores({"H1_Forecast": h1, "H2_Recon": h2})
    fused = fused.rename("FusedScore")

    # Regimes (operating states) on imputed numeric data
    regimes = pd.Series(index=df_imp.index, dtype=int)
    try:
        max_rows = 20000 if args.fast else 100000
        regimes, _km = fit_assign_regimes(df_imp, max_rows=max_rows)
    except Exception:
        pass

    # Scores table including original numeric tags
    scores = df.copy()
    scores["FusedScore"] = fused
    scores["H1_Forecast"] = h1.reindex(scores.index)
    scores["H2_Recon"] = h2.reindex(scores.index)
    if not regimes.empty:
        scores["Regime"] = regimes.reindex(scores.index)
    scores_out = out / "scores.csv"
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    scores.reset_index(names=["Ts"]).to_csv(scores_out, index=False)

    # Regime strip image
    if "Regime" in scores.columns:
        plot_regime_strip(scores["Regime"], out / "images_native" / "regimes.png")

    # Events (indices t0/t1 relative to scores rows)
    ev_df = build_events(fused)
    events_path = out / "events.jsonl"
    if not ev_df.empty:
        # Map timestamps to integer indices
        ts_index = scores.index
        rows = []
        for _, r in ev_df.iterrows():
            # position indices
            t0_pos = int(ts_index.get_indexer([r["start"]], method="nearest")[0])
            t1_pos = int(ts_index.get_indexer([r["end"]], method="nearest")[0])
            rows.append({"event_id": int(r["id"]), "t0": t0_pos, "t1": t1_pos})
        with open(events_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    else:
        events_path.write_text("")

    print(f"[train:M2] wrote {scores_out} and {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
