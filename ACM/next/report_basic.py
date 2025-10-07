"""
report_basic.py

Sectioned HTML report renderer for ACM next.
Renders anchors and links to PNGs saved under an assets/images directory.
"""

from __future__ import annotations

import os
from html import escape
from typing import Dict, Any, List, Optional


def _nav(sections: List[tuple[str, str]]) -> str:
    links = " ".join(f"<a href='#sec-{escape(k)}'>{escape(lbl)}</a>" for k, lbl in sections)
    return f"<nav class='section'>{links}</nav>"


def _wrap(title: str, body: str, css_text: Optional[str]) -> str:
    css_block = f"<style>{css_text}</style>" if css_text else ""
    safe_title = escape(title)
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'/>
  <title>{safe_title}</title>
  {css_block}
</head>
<body>
  <main>
    {body}
  </main>
</body>
</html>"""


def _img(path_rel: str, alt: str = "image") -> str:
    return f"<img class='img-wide' src='{escape(path_rel)}' alt='{escape(alt)}'/>"


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    thead = "<tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr>"
    body_rows = []
    for r in rows:
        body_rows.append("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>")
    return f"<table><thead>{thead}</thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_report(context: Dict[str, Any], out_html: str, assets_dir: str, css_text: Optional[str]) -> str:
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    title = context.get("title", "ACM Report")
    equip = context.get("equip", "equipment")
    sections = [
        ("overview", "Overview"),
        ("snapshot", "Data Snapshot"),
        ("health", "Tag Health"),
        ("timeline", "Anomaly Timeline"),
        ("events", "Event Details"),
        ("explain", "Explainability"),
        ("drift", "Drift & Dependence"),
        ("appendix", "Appendix"),
    ]

    parts: List[str] = []
    parts.append(_nav(sections))

    # Overview
    meta_rows = []
    meta = context.get("meta", {})
    for k, v in meta.items():
        meta_rows.append([k, v])
    parts.append(
        f"<section id='sec-overview' class='section'><h2>Overview — {escape(equip)}</h2>" +
        _table(["Key", "Value"], meta_rows) + "</section>"
    )

    # Snapshot
    snapshot_rel = context.get("paths", {}).get("snapshot_rel")
    snippet_rel = context.get("paths", {}).get("snippet_rel")
    parts.append(
        f"<section id='sec-snapshot' class='section'><h2>Data Snapshot</h2>"
        f"<div class='small'>First 20 rows — <a href='{escape(snapshot_rel or '')}'>snapshot.csv</a></div>"
        f"{_img(snippet_rel or '', 'snapshot table image') if snippet_rel else ''}"
        f"</section>"
    )

    # Tag health
    health_rel = context.get("paths", {}).get("tag_health_rel")
    parts.append(
        f"<section id='sec-health' class='section'><h2>Tag Health</h2>{_img(health_rel or '', 'tag health') if health_rel else ''}</section>"
    )

    # Timeline
    timeline_rel = context.get("paths", {}).get("matrix_rel")
    trace_rel = context.get("paths", {}).get("threshold_rel")
    parts.append(
        f"<section id='sec-timeline' class='section'><h2>Anomaly Timeline</h2>"
        f"{_img(timeline_rel or '', 'anomaly matrix') if timeline_rel else ''}"
        f"{_img(trace_rel or '', 'threshold trace') if trace_rel else ''}"
        f"</section>"
    )

    # Events
    ev_imgs = context.get("paths", {}).get("event_imgs", [])
    wf_imgs = context.get("paths", {}).get("waterfalls", [])
    contrib_imgs = context.get("paths", {}).get("contrib_imgs", [])
    ev_html = "".join(_img(p, os.path.basename(p)) for p in ev_imgs)
    wf_html = "".join(_img(p, os.path.basename(p)) for p in wf_imgs)
    cb_html = "".join(_img(p, os.path.basename(p)) for p in contrib_imgs)
    parts.append(
        f"<section id='sec-events' class='section'><h2>Event Details</h2>{ev_html}{wf_html}{cb_html}</section>"
    )

    # Explainability
    attn_t_rel = context.get("paths", {}).get("attn_temporal_rel")
    attn_s_rel = context.get("paths", {}).get("attn_spatial_rel")
    latent_rel = context.get("paths", {}).get("latent_rel")
    parts.append(
        f"<section id='sec-explain' class='section'><h2>Explainability</h2>"
        f"{_img(attn_t_rel or '', 'temporal attention') if attn_t_rel else ''}"
        f"{_img(attn_s_rel or '', 'spatial attention') if attn_s_rel else ''}"
        f"{_img(latent_rel or '', 'latent space') if latent_rel else ''}"
        f"</section>"
    )

    # Drift
    corr_rel = context.get("paths", {}).get("rolling_corr_rel")
    parts.append(
        f"<section id='sec-drift' class='section'><h2>Drift & Dependence</h2>"
        f"{_img(corr_rel or '', 'rolling corr') if corr_rel else ''}"
        f"</section>"
    )

    # Appendix
    dq_rel = context.get("paths", {}).get("dq_table_rel")
    cfg_dump = escape(context.get("cfg_dump", ""))
    parts.append(
        f"<section id='sec-appendix' class='section'><h2>Appendix</h2>"
        f"<pre class='small'>{cfg_dump}</pre>"
        f"</section>"
    )

    html = _wrap(title, "".join(parts), css_text)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html

