"""acmnxt score --csv --equip --art-dir

Score-only path: reuse saved artifacts (e.g., PCA) to compute H1/H2,
fuse to FusedScore, detect events, and write scores.csv and events.jsonl
under <art-dir>/<equip_sanitized>/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from acmnxt.io.loaders import read_table, ensure_datetime_index
from acmnxt.io.writers import write_csv
from acmnxt.core.dq import clean_time, resample_numeric
from acmnxt.core.h1_ar1 import score_h1
from acmnxt.core.h2_pca import score_h2
from acmnxt.core.fusion import fuse_scores
from acmnxt.core.events import build_events


def main() -> int:
    ap = argparse.ArgumentParser(description="ACMnxt score-only")
    ap.add_argument("--csv", required=True, help="Input CSV/XLSX to score")
    ap.add_argument("--equip", required=True, help="Equipment name (used for folder)")
    ap.add_argument("--art-dir", required=True, help="Artifacts root directory (contains trained artifacts)")
    args = ap.parse_args()
    print(f"[score] csv={args.csv} equip={args.equip} art_dir={args.art_dir}")

    root = Path(args.art_dir)
    equip_s = "".join(ch if ch.isalnum() else "_" for ch in args.equip)
    out = root / equip_s
    out.mkdir(parents=True, exist_ok=True)

    # Load + prep
    df_raw = read_table(args.csv)
    df = ensure_datetime_index(df_raw, ts_col="Ts")
    df = clean_time(df)
    df = resample_numeric(df, rule="1min")

    # Impute for modeling
    df_imp = df.ffill().bfill()
    if df_imp.isna().any().any():
        df_imp = df_imp.fillna(df_imp.mean(numeric_only=True))

    # H1/H2 using existing PCA/scaler under out
    h1 = score_h1(df_imp)
    h2 = score_h2(df_imp, art_dir=out)
    fused = fuse_scores({"H1_Forecast": h1, "H2_Recon": h2}).rename("FusedScore")

    # Scores table
    scores = df.copy()
    scores["FusedScore"] = fused
    scores["H1_Forecast"] = h1.reindex(scores.index)
    scores["H2_Recon"] = h2.reindex(scores.index)
    scores_out = out / "scores.csv"
    scores.reset_index(names=["Ts"]).to_csv(scores_out, index=False)

    # Events
    ev_df = build_events(fused)
    events_path = out / "events.jsonl"
    if not ev_df.empty:
        ts_index = scores.index
        rows = []
        for _, r in ev_df.iterrows():
            t0_pos = int(ts_index.get_indexer([r["start"]], method="nearest")[0])
            t1_pos = int(ts_index.get_indexer([r["end"]], method="nearest")[0])
            rows.append({"event_id": int(r["id"]), "t0": t0_pos, "t1": t1_pos})
        with open(events_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    else:
        events_path.write_text("")

    print(f"[score] wrote {scores_out} and {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
