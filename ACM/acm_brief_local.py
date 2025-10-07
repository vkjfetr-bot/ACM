# acm_brief_local.py
# Builds a concise LLM brief (brief.json + brief.md) from ACM artifacts.
# - Tolerant to different event column names (t0/t1, ts_start/ts_end, etc.)
# - Safe when files/columns are missing
# - Focuses on tables and text (no cards, no styling)

import os
import json
import argparse
import datetime as dt
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


# ----------------------------
# Utilities & Normalizers
# ----------------------------

def utcnow_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


def safe_ts(x) -> pd.Timestamp:
    """Coerce to UTC timestamp; return NaT on failure."""
    try:
        return pd.to_datetime(x, errors="coerce", utc=True)
    except Exception:
        return pd.NaT


def load_csv_if_exists(path: str, **kwargs) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            df = pd.read_csv(path, **kwargs)
            return df
    except Exception:
        pass
    return None


def load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first existing column from candidates (case-insensitive)."""
    if df is None or df.empty:
        return None
    lc = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in lc:
            return lc[k.lower()]
    return None


def normalize_events(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Standardize event columns to: start, end, label, score (optional)."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["start", "end", "label", "score"])

    out = df.copy()
    # Choose columns (case-insensitive)
    start_col = pick_column(out, ["start", "t_start", "ts_start", "begin", "t0", "episode_start", "s"])
    end_col   = pick_column(out, ["end", "t_end", "ts_end", "finish", "t1", "episode_end", "e", "stop"])
    label_col = pick_column(out, ["label", "event", "type", "name", "tag", "category"])
    score_col = pick_column(out, ["score", "severity", "strength", "prob", "probability"])

    # Rename to canonical
    ren = {}
    if start_col and start_col != "start":
        ren[start_col] = "start"
    if end_col and end_col != "end":
        ren[end_col] = "end"
    if label_col and label_col != "label":
        ren[label_col] = "label"
    if score_col and score_col != "score":
        ren[score_col] = "score"
    if ren:
        out = out.rename(columns=ren)

    # Ensure datetimes for start/end
    for c in ["start", "end"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce", utc=True)

    # Keep only useful columns if present
    keep = [c for c in ["start", "end", "label", "score"] if c in out.columns]
    if keep:
        out = out[keep]
    else:
        out = pd.DataFrame(columns=["start", "end", "label", "score"])

    return out


def normalize_scored(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Standardize scored to have at least: ts, score (if possible)."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["ts", "score"])
    out = df.copy()

    ts_col = pick_column(out, ["ts", "time", "timestamp", "datetime"])
    if ts_col and ts_col != "ts":
        out = out.rename(columns={ts_col: "ts"})
    if "ts" in out.columns:
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce", utc=True)

    score_col = pick_column(out, ["score", "anomaly_score", "fused_score", "final_score"])
    if score_col and score_col != "score":
        out = out.rename(columns={score_col: "score"})

    # we won't drop other columns; but downstream will guard on existence
    return out


def normalize_drift(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Standardize drift dataframe to have ts and (optionally) metrics."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["ts"])
    out = df.copy()
    ts_col = pick_column(out, ["ts", "time", "timestamp", "datetime"])
    if ts_col and ts_col != "ts":
        out = out.rename(columns={ts_col: "ts"})
    if "ts" in out.columns:
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce", utc=True)
    return out


def read_any_events(art_dir: str) -> pd.DataFrame:
    """Try multiple filenames commonly used for events/episodes."""
    candidates = [
        "events.csv",
        "episodes.csv",
        "events_parsed.csv",
        "episodes_parsed.csv",
        "events.json",   # supports json list/dict formats (minimal)
        "episodes.json",
    ]
    for name in candidates:
        path = os.path.join(art_dir, name)
        if name.endswith(".csv"):
            df = load_csv_if_exists(path)
            if df is not None:
                return normalize_events(df)
        else:
            js = load_json_if_exists(path)
            if js is not None:
                try:
                    df = pd.DataFrame(js)
                    return normalize_events(df)
                except Exception:
                    pass
    return normalize_events(None)


def read_scored(art_dir: str) -> pd.DataFrame:
    candidates = [
        "scored.csv",
        "scores.csv",
        "inference_scored.csv",
        "acm_scored.csv",
    ]
    for name in candidates:
        path = os.path.join(art_dir, name)
        df = load_csv_if_exists(path)
        if df is not None:
            return normalize_scored(df)
    return normalize_scored(None)


def read_drift(art_dir: str) -> pd.DataFrame:
    candidates = [
        "drift.csv",
        "embedding_drift.csv",
        "pca_drift.csv",
    ]
    for name in candidates:
        path = os.path.join(art_dir, name)
        df = load_csv_if_exists(path)
        if df is not None:
            return normalize_drift(df)
    return normalize_drift(None)


# ----------------------------
# Headline & Metrics
# ----------------------------

def summarize_headline(scored: pd.DataFrame, events: pd.DataFrame, drift: pd.DataFrame) -> str:
    """Produce a one-line headline for the brief."""
    # Last anomaly change = max event start
    last_change = pd.NaT
    if "start" in events.columns and not events.empty:
        last_change = safe_ts(events["start"].max())

    # Current anomaly level (median of last N scores)
    curr_level = None
    if "ts" in scored.columns and "score" in scored.columns and not scored.empty:
        recent = scored.sort_values("ts").tail(300)  # recent window
        if not recent.empty:
            curr_level = float(np.nanmedian(recent["score"]))

    # Drift ping
    last_drift = pd.NaT
    if "ts" in drift.columns and not drift.empty:
        last_drift = safe_ts(drift["ts"].max())

    parts = []
    if pd.isna(last_change):
        parts.append("No recent anomaly episodes detected")
    else:
        parts.append(f"Last anomaly episode around {last_change.isoformat()}")

    if curr_level is not None and np.isfinite(curr_level):
        parts.append(f"current anomaly level ~ {curr_level:.2f}")

    if not pd.isna(last_drift):
        parts.append(f"last drift check {last_drift.isoformat()}")

    if not parts:
        return "No events or scores available."
    return "; ".join(parts) + "."


def summarize_events_table(events: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Compact table with most recent events."""
    if events.empty:
        return pd.DataFrame(columns=["start", "end", "label", "score"])
    tbl = events.copy()
    if "start" in tbl.columns:
        tbl = tbl.sort_values("start", ascending=False)
    if limit > 0:
        tbl = tbl.head(limit)
    # Format times as ISO without changing types in the DF returned (for JSON/md creation later)
    return tbl.reset_index(drop=True)


def summarize_score_stats(scored: pd.DataFrame) -> Dict[str, Any]:
    if scored.empty or "score" not in scored.columns:
        return {"count": 0}
    s = scored["score"].astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    return {
        "count": int(s.notna().sum()),
        "min": float(np.nanmin(s)) if s.notna().any() else None,
        "p25": float(np.nanpercentile(s, 25)) if s.notna().any() else None,
        "median": float(np.nanmedian(s)) if s.notna().any() else None,
        "p75": float(np.nanpercentile(s, 75)) if s.notna().any() else None,
        "max": float(np.nanmax(s)) if s.notna().any() else None,
    }


# ----------------------------
# Brief Builders
# ----------------------------

def build_brief(art_dir: str, equip: str) -> Dict[str, Any]:
    os.makedirs(art_dir, exist_ok=True)

    # Load inputs
    scored = read_scored(art_dir)
    events = read_any_events(art_dir)
    drift  = read_drift(art_dir)

    # Optional metadata
    meta = load_json_if_exists(os.path.join(art_dir, "meta.json")) or {}
    key_tags = load_json_if_exists(os.path.join(art_dir, "key_tags.json")) or {}

    # Summaries
    headline = summarize_headline(scored, events, drift)
    score_stats = summarize_score_stats(scored)
    events_tbl = summarize_events_table(events, limit=15)

    # Build JSON brief
    brief_json = {
        "generated_at_utc": utcnow_iso(),
        "equipment": equip,
        "headline": headline,
        "stats": {
            "scores": score_stats,
            "events_count": int(len(events_tbl)),
        },
        "meta": meta,
        "key_tags": key_tags,
        "events": events_tbl.to_dict(orient="records"),
    }

    # Build Markdown brief
    md_lines = []
    md_lines.append(f"# {equip} — ACM Brief")
    md_lines.append("")
    md_lines.append(f"_Generated (UTC): {brief_json['generated_at_utc']}_  ")
    md_lines.append("")
    md_lines.append(f"**Headline:** {headline}")
    md_lines.append("")
    md_lines.append("## Score Summary")
    if brief_json["stats"]["scores"].get("count", 0) == 0:
        md_lines.append("- No scores available.")
    else:
        s = brief_json["stats"]["scores"]
        md_lines.append(f"- Count: {s['count']}")
        md_lines.append(f"- Min / P25 / Median / P75 / Max: {s.get('min')} / {s.get('p25')} / {s.get('median')} / {s.get('p75')} / {s.get('max')}")
    md_lines.append("")

    md_lines.append("## Recent Events (up to 15)")
    if len(events_tbl) == 0:
        md_lines.append("_No events available._")
    else:
        # Emit as a simple Markdown table with available columns
        cols = [c for c in ["start", "end", "label", "score"] if c in events_tbl.columns]
        if not cols:
            md_lines.append("_Events exist but lack standard fields to tabulate._")
        else:
            md_lines.append("| " + " | ".join(cols) + " |")
            md_lines.append("|" + "|".join(["---"] * len(cols)) + "|")
            for _, r in events_tbl.iterrows():
                vals = []
                for c in cols:
                    v = r[c]
                    if isinstance(v, (pd.Timestamp, np.datetime64)):
                        try:
                            v = pd.to_datetime(v, utc=True).isoformat()
                        except Exception:
                            v = str(v)
                    vals.append("" if pd.isna(v) else str(v))
                md_lines.append("| " + " | ".join(vals) + " |")
    md_lines.append("")

    if key_tags:
        md_lines.append("## Key Tags")
        # show top-level keys only; values might be lists/weights etc.
        for k, v in key_tags.items():
            md_lines.append(f"- **{k}:** {v}")
        md_lines.append("")

    if meta:
        md_lines.append("## Meta")
        for k, v in meta.items():
            md_lines.append(f"- **{k}:** {v}")
        md_lines.append("")

    brief_md = "\n".join(md_lines)

    # Write outputs
    json_path = os.path.join(art_dir, "brief.json")
    md_path   = os.path.join(art_dir, "brief.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(brief_json, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(brief_md)

    # Return for logging/testing
    return {
        "json_path": json_path,
        "md_path": md_path,
        "headline": headline,
        "events_count": len(events_tbl),
        "scores_count": int(score_stats.get("count", 0)),
    }


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build ACM brief (brief.json + brief.md) from artifacts.")
    p.add_argument("--art_dir", required=True, help="Artifacts directory (e.g., C:\\...\\acm_artifacts)")
    p.add_argument("--equip", required=True, help="Equipment name for header.")
    return p.parse_args()


def main():
    args = parse_args()
    info = build_brief(args.art_dir, args.equip)
    print("== LLM Brief (brief.json + brief.md) ==")
    print(f"Headline: {info['headline']}")
    print(f"Events: {info['events_count']} | Scores: {info['scores_count']}")
    print(f"Wrote: {info['json_path']}")
    print(f"Wrote: {info['md_path']}")


if __name__ == "__main__":
    main()
