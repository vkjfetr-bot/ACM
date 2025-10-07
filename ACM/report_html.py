# report_html.py
# Minimal HTML scaffolding and helpers (tables + charts only).
# Pairs with: report_css.get_css() and charts from report_charts.py.

from datetime import datetime
from report_css import get_css

def wrap_html(title: str, body: str) -> str:
    """Wrap whole document with <html> + inline CSS."""
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>{get_css()}</style>
</head>
<body>
  <main>
    <div class="section">
      <h1>{title}</h1>
      <div class="small">Generated: {datetime.now().isoformat()}</div>
    </div>
    {body}
  </main>
</body>
</html>"""

def kpi_grid(items):
    """items: list[(title, value)] → 5-column KPI grid."""
    cells = "".join(
        f"<div class='kpi'><div class='title'>{t}</div><div class='value'>{v}</div></div>"
        for t, v in items
    )
    return f"<div class='section'><div class='kpis'>{cells}</div></div>"

def section(title: str, inner_html: str) -> str:
    """Simple content section."""
    return f"<div class='section'><h2>{title}</h2>{inner_html}</div>"

def table(headers, rows):
    """Plain table. headers: list[str], rows: list[list[Any]]."""
    thead = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    tbody = "\n".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead>{thead}</thead><tbody>{tbody}</tbody></table>"

def chart(img_data_uri: str) -> str:
    """Embed a PNG base64 (from report_charts.*) as an <img>."""
    return f"<img class='chart' src='{img_data_uri}'/>"
