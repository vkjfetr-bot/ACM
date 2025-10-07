# acm_brief_local.py
# ------------------------------------------------------------
# Build compact "LLM briefing pack" summarizing ACM artifacts
# into human-friendly form for explanation generation.
# ------------------------------------------------------------

import os, json, argparse, datetime as dt
import pandas as pd
import numpy as np

def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {path}")

def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_ts(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

# ------------------------------------------------------------
# Summarizers
# ------------------------------------------------------------

def summarize_headline(scored, events, drift):
    if scored.empty:
        return {"health_state": "unknown", "primary_drivers": [], "confidence": 0.0}

    # recent health from fused_score
    recent = scored.tail(int(len(scored)*0.2))
    p95 = recent["fused_score"].quantile(0.95) if "fused_score" in recent else 0
    state = (
        "stable" if p95 < 0.5 else
        "watch" if p95 < 0.7 else
        "degrading" if p95 < 0.85 else
        "at_risk"
    )

    primary = []
    if "top_tags" in scored.columns:
        scored["top_tags"] = scored["top_tags"].fillna("").astype(str)
        for tags in scored["top_tags"].tail(20):
            for t in tags.split(","):
                if t and t not in primary:
                    primary.append(t)
    primary = primary[:5]

    last_change = safe_ts(events["start"].max()) if not events.empty else pd.NaT
    return {
        "health_state": state,
        "primary_drivers": primary,
        "confidence": float(np.clip(1 - abs(0.5 - p95), 0, 1)),
        "last_change_ts": last_change.isoformat() if pd.notna(last_change) else None
    }

def summarize_kpis(scored, events, masks):
    if scored.empty:
        return {}
    n = len(scored)
    uptime_pct = 1.0
    if not masks.empty and {"start", "end"}.issubset(masks.columns):
        total = (scored["Ts"].max() - scored["Ts"].min()).total_seconds()
        maint = sum([(safe_ts(r.end) - safe_ts(r.start)).total_seconds() for r in masks.itertuples()])
        uptime_pct = max(0, 1 - maint/total) if total>0 else 1
    ev_rate = len(events)/max(1, (n/ (24*60)))  # approx per day
    return {
        "uptime_pct": round(uptime_pct,3),
        "event_rate_per_day": round(ev_rate,3),
        "median_fused_score": float(scored["fused_score"].median() if "fused_score" in scored else 0),
        "p95_fused_score": float(scored["fused_score"].quantile(0.95) if "fused_score" in scored else 0),
        "highest_recent_event_score": float(events["peak_score"].max() if "peak_score" in events else 0)
    }

def summarize_regimes(regimes):
    if regimes.empty: return []
    out=[]
    for r in regimes.itertuples():
        out.append({
            "id": int(r.id) if "id" in regimes.columns else int(getattr(r,"Index",0)),
            "label": str(getattr(r,"label","Regime")),
            "support_pct": float(getattr(r,"support_pct",0)),
            "desc": str(getattr(r,"desc",""))
        })
    return out

def summarize_dq(dq):
    if dq.empty: return {"issues": [], "overall_grade": "N/A"}
    dq["grade"] = 1 - (0.5*dq.get("Flatline%",0)/100 + 0.4*dq.get("Dropout%",0)/100 + 0.1*dq.get("Spikes",0).clip(0,1))
    g = dq["grade"].mean()
    grade = "A" if g>0.9 else "B" if g>0.8 else "C"
    return {
        "issues": dq.head(10).to_dict(orient="records"),
        "overall_grade": grade
    }

def summarize_anomalies(events):
    if events.empty: return {"count":0,"recent_top":[]}
    events["start"] = pd.to_datetime(events["start"], errors="coerce")
    events = events.sort_values("start", ascending=False)
    top = events.head(3)
    out=[]
    for e in top.itertuples():
        out.append({
            "event_id": getattr(e,"event_id",None),
            "start": getattr(e,"start",None).isoformat() if pd.notna(getattr(e,"start",pd.NaT)) else None,
            "end": getattr(e,"end",None),
            "peak_score": float(getattr(e,"peak_score",0)),
            "regime_context": getattr(e,"regime_context",""),
            "tags_primary": getattr(e,"tags_primary",[]),
            "possible_causes": getattr(e,"possible_causes",[]),
            "confidence": float(getattr(e,"confidence",0.7))
        })
    return {"count": len(events), "recent_top": out}

def summarize_drift(drift_json):
    if not drift_json: return {}
    trend = drift_json.get("trend","unknown")
    cps = drift_json.get("change_points",[])
    return {"embeddings":{"trend":trend,"change_points":cps}}

def compose_scaffold(meta, headline, dq, anomalies, regimes):
    sentences=[]
    if headline.get("health_state"):
        sentences.append(
            f"The equipment is currently in **{headline['health_state']}** condition with confidence {headline['confidence']:.2f}."
        )
    if regimes:
        main = sorted(regimes, key=lambda x: x.get("support_pct",0), reverse=True)[0]
        sentences.append(f"It operated mostly in the **{main['label']}** regime ({main['support_pct']*100:.1f}%).")
    if anomalies["count"]:
        top = anomalies["recent_top"][0]
        sentences.append(f"Most recent anomaly (score {top['peak_score']:.2f}) occurred on {top['start']} in {top.get('regime_context','N/A')} context.")
    caveats=[]
    if dq["overall_grade"]!="A":
        caveats.append(f"Data quality grade {dq['overall_grade']} — interpret results with caution.")
    return {
        "plain_sentences": sentences,
        "caveats": caveats,
        "suggested_actions": [],
        "questions_for_sme": []
    }

# ------------------------------------------------------------
# Builders
# ------------------------------------------------------------

def build_brief(art_dir, equip):
    paths = lambda n: os.path.join(art_dir, n)
    scored = load_csv(paths("scored.csv"))
    if "Ts" in scored.columns: scored["Ts"] = pd.to_datetime(scored["Ts"])
    events = load_csv(paths("events.csv")) if os.path.exists(paths("events.csv")) else pd.DataFrame()
    if events.empty and os.path.exists(paths("events.jsonl")):
        events = pd.read_json(paths("events.jsonl"), lines=True)
    dq = load_csv(paths("dq.csv"))
    regimes = load_csv(paths("regimes.csv"))
    masks = load_csv(paths("masks.csv"))
    drift = load_json(paths("drift.json")) or load_json(paths("h3_embed.json"))
    run_log = []
    if os.path.exists(paths("run_log.jsonl")):
        with open(paths("run_log.jsonl"), "r") as f:
            run_log = [json.loads(x) for x in f if x.strip()]

    meta = {
        "equip": equip,
        "art_dir": art_dir,
        "run_id": dt.datetime.utcnow().isoformat()+"Z",
        "data_coverage": {
            "start": str(scored["Ts"].min()) if "Ts" in scored else None,
            "end": str(scored["Ts"].max()) if "Ts" in scored else None,
            "pct_present": 1.0
        },
        "units": {}
    }

    headline = summarize_headline(scored, events, drift)
    kpis = summarize_kpis(scored, events, masks)
    dqsum = summarize_dq(dq)
    regsum = summarize_regimes(regimes)
    anoms = summarize_anomalies(events)
    drift_summary = summarize_drift(drift)
    scaffold = compose_scaffold(meta, headline, dqsum, anoms, regsum)

    brief = {
        "meta": meta,
        "headline": headline,
        "kpis": kpis,
        "data_quality": dqsum,
        "regimes": regsum,
        "anomalies": anoms,
        "drift": drift_summary,
        "explanations_scaffold": scaffold,
        "provenance": {
            "files_used": [f for f in os.listdir(art_dir) if os.path.isfile(os.path.join(art_dir,f))]
        }
    }

    out_path = os.path.join(art_dir, "brief.json")
    jdump(brief, out_path)

    # also write markdown preview
    md = [
        f"# ACM Brief — {equip}",
        f"**Health:** {headline['health_state']} (confidence {headline['confidence']:.2f})",
        f"**Period:** {meta['data_coverage']['start']} → {meta['data_coverage']['end']}",
        f"**Top regime:** {regsum[0]['label'] if regsum else 'N/A'}",
        f"**Events detected:** {anoms['count']}",
        "",
        "## Key Sentences",
        *scaffold["plain_sentences"],
        "",
        "## Caveats",
        *scaffold["caveats"]
    ]
    with open(os.path.join(art_dir, "brief.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("[OK] Wrote brief.md")
    return brief

def build_prompt(brief_path):
    with open(brief_path,"r",encoding="utf-8") as f:
        brief=json.load(f)
    prompt={
        "system": "You are an industrial reliability analyst. Explain this summary in plain English for plant operators and managers. Be factual, avoid jargon, and propose 3-5 practical next steps.",
        "user": "Use the following briefing data to explain the equipment's condition. Do not invent data outside this briefing.",
        "context": brief,
        "constraints": {
            "forbidden": ["raw signal analysis"],
            "style": {"read_time_min":3,"max_sections":6}
        }
    }
    outp=os.path.join(os.path.dirname(brief_path),"llm_prompt.json")
    jdump(prompt,outp)
    return prompt

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd")

    b=sub.add_parser("build")
    b.add_argument("--art_dir", required=True)
    b.add_argument("--equip", required=True)

    p=sub.add_parser("prompt")
    p.add_argument("--brief", required=True)

    args=ap.parse_args()

    if args.cmd=="build":
        build_brief(args.art_dir, args.equip)
    elif args.cmd=="prompt":
        build_prompt(args.brief)
    else:
        ap.print_help()

if __name__=="__main__":
    main()
