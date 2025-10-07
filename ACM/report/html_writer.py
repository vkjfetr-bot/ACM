from __future__ import annotations

import os
from html import escape
from typing import Dict


def render(context: Dict, out_html: str, assets_dir: str) -> None:
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    title = context.get("title", "ACM Report")
    equip = context.get("equipment", "Equipment")
    imgs = context.get("images", {})
    dq_table_html = context.get("dq_table_html", "")
    glossary = context.get("glossary", {})
    summary_lines = context.get("summary", [])
    json_text = context.get("json_text", "{}")

    def _img(path_rel: str, alt: str = "img") -> str:
        if not path_rel:
            return ""
        return f"<figure><img src='{escape(path_rel)}' style='max-width:100%;border:1px solid #334155;border-radius:8px;' alt='{escape(alt)}'/></figure>"

    glossary_html = "".join(f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>" for k, v in glossary.items())
    summary_html = "".join(f"<li>{escape(s)}</li>" for s in summary_lines)

    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang='en'>")
    parts.append("<head>")
    parts.append("  <meta charset='utf-8'/>")
    parts.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'/>")
    parts.append(f"  <title>{escape(title)} &mdash; {escape(equip)}</title>")
    parts.append("  <style>body{background:#0b1220;color:#e5e7eb;font:14px/1.4 system-ui,Segoe UI,Arial} main{max-width:1200px;margin:0 auto;padding:14px 16px}.section{background:#0f1621;border:1px solid #1f2a37;border-radius:10px;padding:14px 16px;margin:12px 0}table{border-collapse:collapse;width:100%} th,td{border:1px solid #1f2a37;padding:8px 10px;text-align:left} thead th{background:#111827}h1,h2,h3{margin:0 0 10px 0}</style>")
    parts.append(f"  <script type='application/json' id='report-json'>{escape(json_text)}</script>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("  <main>")
    parts.append(f"    <div class='section'><h1>{escape(title)} &mdash; {escape(equip)}</h1></div>")
    parts.append(f"    <div class='section'><h2>Executive Summary</h2><ul>{summary_html}</ul></div>")
    parts.append(f"    <div class='section'><h2>Overview (Anomaly Timeline)</h2>{_img(imgs.get('overview'))}</div>")
    if imgs.get('anom_matrix'):
        parts.append(f"    <div class='section'><h2>Anomaly Matrix</h2>{_img(imgs.get('anom_matrix'))}</div>")
    parts.append(f"    <div class='section'><h2>Top Tags</h2>{_img(imgs.get('tags_strip'))}</div>")
    parts.append(f"    <div class='section'><h2>Drift</h2>{_img(imgs.get('drift'))}</div>")
    parts.append(f"    <div class='section'><h2>Data Quality</h2>{_img(imgs.get('dq_heatmap'))}{dq_table_html}</div>")
    parts.append(f"    <div class='section'><h2>Correlations</h2>{_img(imgs.get('corr_normal'))}{_img(imgs.get('corr_anom'))}</div>")
    # Episodes
    eps = context.get('episodes_imgs', []) or []
    if eps:
        gallery = ''.join(_img(p) for p in eps)
        parts.append(f"    <div class='section'><h2>Episodes</h2>{gallery}</div>")
    parts.append(f"    <div class='section'><h2>Glossary</h2><table><thead><tr><th>Term</th><th>Meaning</th></tr></thead><tbody>{glossary_html}</tbody></table></div>")
    parts.append("  </main>")
    parts.append("</body>")
    parts.append("</html>")

    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

