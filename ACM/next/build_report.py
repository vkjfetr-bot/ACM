"""
ACM Next — build-report CLI

Generates a self-contained HTML report and images into <art-dir>/<equip_sanitized>/.
This tool is standalone and does not modify the existing pipeline.
"""

from __future__ import annotations

import os
import argparse
import json
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

import viz_core as V
import viz_explain as X
import report_basic as R


def _read_scores(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_events(path: Optional[str]) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["event_id", "t0", "t1"])
    # support jsonl or json
    if path.lower().endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except Exception:
                    continue
        return pd.DataFrame(rows)
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            obj = obj.get("events", [])
        return pd.DataFrame(obj)


def _moving_ndt_threshold(series: pd.Series, win: int = 201, q: float = 0.95) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if win < 3 or win > len(s):
        win = max(3, min(len(s), 201))
    roll = s.rolling(win, center=True, min_periods=max(3, win // 4))
    return roll.quantile(q).fillna(method="bfill").fillna(method="ffill").fillna(s.quantile(q))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ACM Next: Build HTML report")
    sub = ap.add_subparsers(dest="cmd")

    br = sub.add_parser("build-report", help="Build HTML report and images")
    br.add_argument("--equip", required=True, help="Equipment name")
    br.add_argument("--art-dir", required=True, help="Artifacts root directory")
    br.add_argument("--top-tags", type=int, default=9)
    br.add_argument("--scores", required=True, help="Scores CSV/Parquet (time-index or Ts column)")
    br.add_argument("--events-json", default=None, help="Events JSON/JSONL")
    br.add_argument("--contrib", default=None, help="Contributions JSONL (optional)")
    br.add_argument("--attn", default=None, help="Attention NPZ (optional)")
    br.add_argument("--no-latent", action="store_true")
    br.add_argument("--no-attention", action="store_true")
    br.add_argument("--no-matrix", action="store_true")
    br.add_argument("--seed", type=int, default=42)

    args = ap.parse_args(argv)
    if args.cmd != "build-report":
        ap.print_help()
        return 2

    V.set_seed(args.seed)

    equip = args.equip
    equip_s = "".join(ch if ch.isalnum() else "_" for ch in equip)
    root = os.path.join(args.art_dir, equip_s)
    images_dir = V.ensure_dir(os.path.join(root, "images"))

    # Load scores
    scores = _read_scores(args.scores)
    if "Ts" in scores.columns:
        try:
            scores["Ts"] = pd.to_datetime(scores["Ts"], errors="coerce")
            scores = scores.set_index("Ts").sort_index()
        except Exception:
            pass
    scores = scores.reset_index(drop=False)

    # Pick tag columns (exclude derived knowns)
    derived = {"Regime", "FusedScore", "H1_Forecast", "H2_Recon", "H3_Contrast", "CorrBoost", "CPD", "ContextMask"}
    numcols = [c for c in scores.columns if c not in (derived | {"index", "Ts"}) and pd.api.types.is_numeric_dtype(scores[c])]
    # Keep top by variance
    if numcols:
        var = scores[numcols].var(numeric_only=True).sort_values(ascending=False)
        tags = [c for c in var.index.tolist()[: max(1, args.top_tags)] if c in numcols]
    else:
        tags = []

    # Snapshot files
    snapshot_csv = os.path.join(root, "snapshot.csv")
    scores.head(20).to_csv(snapshot_csv, index=False)
    # render a small HTML snippet as an image replacement (simple CSV link only)
    snippet_html = os.path.join(root, "snippet_table.html")
    with open(snippet_html, "w", encoding="utf-8") as f:
        f.write(scores.head(20).to_html(index=False))

    # Tag health
    dq = V.compute_dq(scores, tags)
    tag_health_png = os.path.join(images_dir, "tag_health.png")
    V.plot_tag_health(dq, tag_health_png)

    # Anomaly matrix (if tags exist)
    matrix_png = os.path.join(images_dir, "anomaly_matrix.png")
    if not args.no_matrix and tags:
        M = V.sanitize_array(scores[tags].to_numpy())
        V.plot_heatmap(M.T, title="Anomaly timeline (values by tag)", out_path=matrix_png, y_label="tags")

    # Threshold trace from FusedScore if present
    threshold_png = os.path.join(images_dir, "threshold_trace.png")
    if "FusedScore" in scores.columns:
        th = _moving_ndt_threshold(scores["FusedScore"], win=201, q=0.95)
        V.plot_threshold_trace(scores["FusedScore"], th, threshold_png, title="FusedScore vs NDT")

    # Events
    events = _read_events(args.events_json)
    event_imgs: List[str] = []
    waterfalls: List[str] = []
    if not events.empty and "FusedScore" in scores.columns:
        # For each event, plot FusedScore window
        for i, row in events.head(10).iterrows():
            eid = str(row.get("event_id", i))
            # pick window indices if available, else uniform slices
            t0 = int(row.get("t0", max(0, len(scores) // 4 - 100)))
            t1 = int(row.get("t1", min(len(scores) - 1, t0 + 200)))
            s = scores["FusedScore"].iloc[max(0, t0 - 50):min(len(scores), t1 + 50)]
            out = os.path.join(images_dir, f"event_{eid}.png")
            V.plot_trend(s.reset_index(drop=True), title=f"Event {eid}", out_path=out)
            event_imgs.append(os.path.relpath(out, root))

            # Waterfall placeholder from components if present
            comps = []
            for c in ("H1_Forecast", "H2_Recon", "H3_Contrast"):
                if c in scores.columns:
                    comps.append((c, float(scores[c].iloc[t0:t1].mean())))
            if comps:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
                ax.bar([c for c, _ in comps], [v for _, v in comps], color=["#60a5fa", "#34d399", "#fbbf24"][: len(comps)])
                ax.set_title(f"Waterfall components • event {eid}")
                wout = os.path.join(images_dir, f"waterfall_{eid}.png")
                V.save_fig(fig, wout)
                waterfalls.append(os.path.relpath(wout, root))

    # Contributions
    contrib_imgs: List[str] = []
    contribs = X.read_contrib_jsonl(args.contrib) if args.contrib else {}
    for eid, pairs in list(contribs.items())[:5]:
        out = os.path.join(images_dir, f"contrib_{eid}.png")
        X.plot_contrib_bars(eid, pairs, out)
        contrib_imgs.append(os.path.relpath(out, root))

    # Attention maps
    attn_temporal_rel = None
    attn_spatial_rel = None
    if not args.no_attention:
        t_out = os.path.join(images_dir, "temporal_attention.png")
        s_out = os.path.join(images_dir, "spatial_attention.png")
        X.plot_attention_maps(args.attn or "", t_out, s_out)
        attn_temporal_rel = os.path.relpath(t_out, root)
        attn_spatial_rel = os.path.relpath(s_out, root)

    # Latent space placeholder (PCA of numeric columns)
    latent_rel = None
    if not args.no_latent and len(numcols) >= 2:
        emb = scores[numcols].to_numpy()
        l_out = os.path.join(images_dir, "latent_space.png")
        X.plot_latent_space(emb, l_out, seed=args.seed)
        latent_rel = os.path.relpath(l_out, root)

    # Rolling corr (drift-like view)
    rolling_corr_rel = None
    if len(tags) >= 2:
        C = np.corrcoef(V.sanitize_array(scores[tags].to_numpy()).T)
        rc_out = os.path.join(images_dir, "rolling_corr_heatmap.png")
        V.plot_heatmap(C, title="Correlation heatmap (global)", out_path=rc_out, x_label="tags", y_label="tags")
        rolling_corr_rel = os.path.relpath(rc_out, root)

    # CSS
    css_path = os.path.join(os.path.dirname(__file__), "report_css.css")
    css_text = None
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_text = f.read()

    # Context for renderer
    ctx: Dict[str, Any] = {
        "title": f"ACM Report — {equip}",
        "equip": equip,
        "meta": {
            "seed": args.seed,
            "rows": len(scores),
            "tags_shown": ", ".join(tags[: min(5, len(tags))]) + ("…" if len(tags) > 5 else ""),
            "scores_file": args.scores,
            "events_file": args.events_json or "",
        },
        "paths": {
            "snapshot_rel": os.path.relpath(snapshot_csv, root),
            "snippet_rel": os.path.relpath(snippet_html, root),
            "tag_health_rel": os.path.relpath(tag_health_png, root) if os.path.exists(tag_health_png) else None,
            "matrix_rel": os.path.relpath(matrix_png, root) if os.path.exists(matrix_png) else None,
            "threshold_rel": os.path.relpath(threshold_png, root) if os.path.exists(threshold_png) else None,
            "event_imgs": event_imgs,
            "waterfalls": waterfalls,
            "contrib_imgs": contrib_imgs,
            "attn_temporal_rel": attn_temporal_rel,
            "attn_spatial_rel": attn_spatial_rel,
            "latent_rel": latent_rel,
            "rolling_corr_rel": rolling_corr_rel,
            "dq_table_rel": os.path.relpath(snapshot_csv, root),
        },
        "cfg_dump": json.dumps({"seed": args.seed, "top_tags": args.top_tags}, indent=2),
    }

    out_html = os.path.join(root, "report.html")
    R.render_report(ctx, out_html, root, css_text)
    print(f"[build-report] Wrote {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

