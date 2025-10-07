from __future__ import annotations

import os
import json
import argparse
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from .config_schema import load_config
from . import data_io, scoring_io, transforms, dq as dq_mod
from . import viz_matplotlib as V
from . import textgen
from .glossary import TERMS
from .html_writer import render


def _ndt(series: pd.Series, win: int = 201, q: float = 0.95) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if win < 3 or win > len(s):
        win = max(3, min(len(s), 201))
    roll = s.rolling(win, center=True, min_periods=max(3, win // 4))
    return roll.quantile(q).bfill().ffill().fillna(s.quantile(q))


def build(train_csv: str, test_csv: str, equip: str, out_dir: str, cfg_path: str | None) -> str:
    cfg = load_config(cfg_path)
    os.makedirs(out_dir, exist_ok=True)
    assets = os.path.join(out_dir, "assets")
    os.makedirs(assets, exist_ok=True)

    tr, te = data_io.load_train_test(train_csv, test_csv)
    df = transforms.align_and_clip(tr, te)

    # Use scored if provided; else synthesize from df
    scored = scoring_io.load_scored(df)

    threshold = _ndt(scored["FusedScore"], win=201, q=cfg.anomaly_thresholds.get("fused_p", 0.95))
    anom_mask = scored["FusedScore"].fillna(0.0) > threshold

    # KPIs (simple)
    episodes = transforms.episodeize(anom_mask, pd.Timedelta(minutes=5))
    kpis = {
        "episodes": len(episodes),
        "max_score": float(scored["FusedScore"].max() if not scored.empty else 0.0),
        "avg_daily_anomaly_minutes": int(anom_mask.resample("D").sum().mean() if isinstance(scored.index, pd.DatetimeIndex) else anom_mask.mean() * 24),
    }

    # Select tags
    numcols = scored.select_dtypes(include=[np.number]).columns.tolist()
    extras = {"FusedScore", "Regime"}
    tags = [c for c in numcols if c not in extras][: cfg.top_n_tags]

    # Plots
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    # Shaded timeline of fused score
    overview_png = os.path.join(assets, f"timeline_{ts}.png")
    V.plot_timeline_shaded(scored, threshold, anom_mask, overview_png)

    # Anomaly matrix across tags using simple rolling z-score per-tag
    matrix_png = os.path.join(assets, f"anom_matrix_{ts}.png")
    mask_df = None
    if tags:
        roll = scored[tags].rolling(201, center=True, min_periods=25)
        mu = roll.mean()
        sd = roll.std().replace(0, np.nan)
        z = (scored[tags] - mu) / (sd + 1e-9)
        mask_df = (z.abs() > 3.0)
        V.plot_anomaly_matrix(mask_df[tags], matrix_png)

    # Simple tags strip: for now render first tag trend
    # For now, save only the first tag trend
    tag_strip_png = os.path.join(assets, f"tags_{ts}.png")
    if tags:
        V.plot_tag_trend(scored, tags[0], anom_mask, tag_strip_png)
    else:
        V.plot_tag_trend(pd.DataFrame(index=scored.index), "", pd.Series(dtype=bool), tag_strip_png)

    drift_png = os.path.join(assets, f"drift_{ts}.png")
    if tags:
        V.plot_drift(scored, tags[0], baseline_win=201, outpath=drift_png)
    else:
        V.plot_drift(pd.DataFrame(index=scored.index), "", baseline_win=201, outpath=drift_png)

    dq_df = dq_mod.compute(scored[tags] if tags else scored)
    dq_png = os.path.join(assets, f"dq_{ts}.png")
    V.plot_dq_heatmap(dq_df, dq_png)

    # Correlations snapshot: naive split by anom mask
    df_norm = scored[tags][~anom_mask] if tags else pd.DataFrame()
    df_anom = scored[tags][anom_mask] if tags else pd.DataFrame()
    corr_prefix = os.path.join(assets, f"corr_{ts}")
    corr_n_png, corr_a_png = V.plot_corrs(df_norm, df_anom, tags or [], corr_prefix)

    # Episode strips (multi-panel) for up to 5 episodes
    episode_imgs = []
    for i, ep in enumerate(episodes[:5], start=1):
        ep_loc = {"id": f"E{i}", "t0": None, "t1": None}
        # best-effort integer t0/t1 from index positions using threshold crossings
        # If anom_mask is a Series aligned to scored, find spans again to map indices
        ep_png = os.path.join(assets, f"episode_{ts}_{i}.png")
        V.plot_episode_strip_multi(scored, ep_loc, tags[:3], ep_png)
        episode_imgs.append(os.path.relpath(ep_png, out_dir))

    # Text
    summary = textgen.summarize_overview(kpis)

    # Glossary
    glossary = TERMS

    # HTML + JSON
    title = f"ACM Report"
    html_name = f"ACM_Report_{equip.replace(' ', '_')}_{ts}.html"
    out_html = os.path.join(out_dir, html_name)
    # normalize episodes to JSON-safe
    eps_json = []
    for i, ep in enumerate(episodes, start=1):
        eps_json.append({
            "id": f"E{i}",
            "start": str(ep.get("start", "")),
            "end": str(ep.get("end", "")),
            "peak": float(scored["FusedScore"].max() if len(scored) else 0.0),
        })

    json_obj = {
        "equipment": equip,
        "period": {
            "start": str(scored.index.min()) if len(scored) else "",
            "end": str(scored.index.max()) if len(scored) else "",
        },
        "kpis": kpis,
        "episodes": eps_json,
        "dq": dq_df.to_dict(orient="records") if dq_df is not None else [],
    }
    ctx = {
        "title": title,
        "equipment": equip,
        "kpis": kpis,
        "images": {
            "overview": os.path.relpath(overview_png, out_dir),
            "anom_matrix": os.path.relpath(matrix_png, out_dir) if os.path.exists(matrix_png) else None,
            "tags_strip": os.path.relpath(tag_strip_png, out_dir),
            "drift": os.path.relpath(drift_png, out_dir),
            "dq_heatmap": os.path.relpath(dq_png, out_dir),
            "corr_normal": os.path.relpath(corr_n_png, out_dir),
            "corr_anom": os.path.relpath(corr_a_png, out_dir),
        },
        "episodes_imgs": episode_imgs,
        "dq_table_html": dq_df.head(20).to_html(index=False) if dq_df is not None else "",
        "glossary": glossary,
        "summary": summary,
        "json": json_obj,
        "json_text": json.dumps(json_obj, ensure_ascii=False),
    }

    render(ctx, out_html, assets)
    return out_html


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="acm.report.cli", description="Build ACM report (visualization-focused)")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build", help="Build report")
    b.add_argument("--train", required=True)
    b.add_argument("--test", required=True)
    b.add_argument("--equip", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--config", default=None, help="YAML config path")
    args = ap.parse_args(argv)
    if args.cmd != "build":
        ap.print_help()
        return 2
    out_html = build(args.train, args.test, args.equip, args.out, args.config)
    print(f"[acm.report] Wrote {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
