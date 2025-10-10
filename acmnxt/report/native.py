from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

DERIVED = {"FusedScore","H1_Forecast","H2_Recon","Regime"}

def _save_fig(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def _fmt_ts(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts): return ""
    if isinstance(ts, pd.Timestamp):
        if ts.tz is not None: ts = ts.tz_convert(None)
        return ts.strftime("%d %b %Y %H:%M")
    return str(ts)

def _events_from_jsonl(path: Path, idx: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    if not path.exists(): return pd.DataFrame(columns=["event_id","t0","t1","Start","End","Duration_s","Peak"])
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except Exception: continue
    ev = pd.DataFrame(rows)
    if ev.empty: return ev
    def pos(i: int) -> Optional[pd.Timestamp]:
        return idx[min(max(int(i),0), len(idx)-1)]
    ev["Start"] = ev["t0"].map(pos)
    ev["End"] = ev["t1"].map(pos)
    return ev

def _plot_timeline(scores: pd.DataFrame, events: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(12,4))
    fs = pd.to_numeric(scores.get("FusedScore"), errors="coerce") if "FusedScore" in scores else None
    if fs is not None:
        q90 = float(fs.quantile(0.90)); q97 = float(fs.quantile(0.97))
        ax.axhspan(0,q90,color="#10b981",alpha=0.12)
        ax.axhspan(q90,q97,color="#f59e0b",alpha=0.12)
        ax.axhspan(q97,1.05,color="#ef4444",alpha=0.10)
        x = fs.index.tz_convert(None) if fs.index.tz is not None else fs.index
        ax.plot(x, fs.values, color="black", lw=1.5, label="Fused")
        thr = float(fs.quantile(0.95)); ax.axhline(thr,color="#ef4444",ls="--",lw=1)
        for _,r in events.iterrows():
            s,e = r.get("Start"), r.get("End");
            if pd.isna(s) or pd.isna(e): continue
            ax.axvspan(s.tz_convert(None) if s.tzinfo else s, e.tz_convert(None) if e.tzinfo else e, color="#ef4444", alpha=0.06)
    for name,col in [("H1_Forecast","#60a5fa"),("H2_Recon","#f59e0b")]:
        if name in scores:
            s = pd.to_numeric(scores[name], errors="coerce")
            x = s.index.tz_convert(None) if s.index.tz is not None else s.index
            ax.plot(x, s.values, lw=1.0, alpha=0.6, color=col, label=name.split('_')[0])
    ax.set_ylim(0,1.05); ax.set_ylabel("score"); ax.set_title("Anomaly Scores Over Time"); ax.grid(True,alpha=0.2); ax.legend(loc="upper left")
    _save_fig(fig,out)

def _plot_recent(scores: pd.DataFrame, events: pd.DataFrame, out: Path, hours: int=48):
    if not isinstance(scores.index, pd.DatetimeIndex) or scores.empty: return
    end = scores.index.max(); start = end - pd.Timedelta(hours=hours)
    sub = scores.loc[start:end]
    if sub.empty: sub = scores.tail(min(len(scores),500))
    sub = sub.resample("5min").median()
    fig, ax = plt.subplots(figsize=(12,3.5))
    fs = pd.to_numeric(sub.get("FusedScore"), errors="coerce") if "FusedScore" in sub else None
    if fs is not None:
        fs_s = fs.rolling(9,center=True,min_periods=1).median()
        q90 = float(fs_s.quantile(0.90)); q97 = float(fs_s.quantile(0.97))
        ax.axhspan(0,q90,color="#10b981",alpha=0.12); ax.axhspan(q90,q97,color="#f59e0b",alpha=0.12); ax.axhspan(q97,1.05,color="#ef4444",alpha=0.10)
        x = fs_s.index.tz_convert(None) if fs_s.index.tz is not None else fs_s.index
        ax.plot(x, fs_s.values, color="black", lw=1.4, label="Fused")
        thr = float(fs_s.quantile(0.95)); ax.axhline(thr,color="#ef4444",ls='--',lw=1)
        idx = fs_s[fs_s>thr].index; ax.scatter(idx.tz_convert(None) if idx.tz is not None else idx, fs_s.loc[idx].values,s=12,color="#ef4444")
    for name,col in [("H1_Forecast","#60a5fa"),("H2_Recon","#f59e0b")]:
        if name in sub:
            s = pd.to_numeric(sub[name], errors="coerce").rolling(9,center=True,min_periods=1).median()
            x = s.index.tz_convert(None) if s.index.tz is not None else s.index
            ax.plot(x, s.values, lw=1.0, alpha=0.5, color=col, label=name.split('_')[0])
    ax.set_ylim(0,1.05); ax.set_ylabel("score"); ax.set_title(f"Recent Health (last {hours}h)"); ax.grid(True,alpha=0.2); ax.legend(loc="upper left")
    _save_fig(fig,out)

def _plot_severity(fs: pd.Series, out: Path):
    if fs is None or len(fs)==0: return
    q90=float(fs.quantile(0.90)); q97=float(fs.quantile(0.97))
    low=(fs<q90).mean()*100; med=((fs>=q90)&(fs<q97)).mean()*100; high=(fs>=q97).mean()*100
    fig,ax=plt.subplots(figsize=(6,2.8)); ax.bar(["Low","Medium","High"],[low,med,high],color=["#10b981","#f59e0b","#ef4444"])
    ax.set_ylim(0,100); ax.set_ylabel("% time"); ax.set_title("Severity Occupancy"); ax.grid(True,axis='y',alpha=0.2)
    _save_fig(fig,out)

def build_native_report(art_dir: str|Path, equip: str, top_tags: int=6) -> Path:
    equip_s = "".join(ch if ch.isalnum() else "_" for ch in equip)
    root = Path(art_dir)/equip_s; imgs = root/"images_native"; imgs.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(root/"scores.csv"); scores["Ts"]=pd.to_datetime(scores["Ts"],utc=True,errors='coerce'); scores=scores.set_index("Ts").sort_index()
    tags = [c for c in scores.columns if c not in DERIVED and pd.api.types.is_numeric_dtype(scores[c])][:top_tags]
    events = _events_from_jsonl(root/"events.jsonl", scores.index)
    # health
    _plot_timeline(scores, events, imgs/"timeline.png")
    _plot_recent(scores, events, imgs/"timeline_recent.png")
    fs = pd.to_numeric(scores.get("FusedScore"), errors="coerce") if "FusedScore" in scores else None
    if fs is not None: _plot_severity(fs, imgs/"severity.png")
    # minimal HTML
    html = f"""
<!DOCTYPE html><html><head><meta charset='utf-8'/><title>ACMnxt Report - {equip_s}</title>
<style>body{{font-family:Arial;background:#0b1220;color:#e5e7eb}}main{{max-width:1200px;margin:auto;padding:16px}}.section{{background:#0f1621;border:1px solid #1f2a37;border-radius:10px;padding:14px 16px;margin:12px 0}}.img{{max-width:100%;height:auto;display:block;border:1px solid #334155;border-radius:8px}}</style></head><body><main>
<div class='section'><h2>Health Overview</h2>
<img class='img' src='images_native/timeline.png'/>
<img class='img' src='images_native/timeline_recent.png'/>
<img class='img' src='images_native/severity.png'/>
</div>
</main></body></html>
"""
    out = root/"report_native.html"; out.write_text(html,encoding='utf-8'); return out

