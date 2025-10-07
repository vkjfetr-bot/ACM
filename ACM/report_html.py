# report_html.py
# Minimal HTML scaffolding and helpers (tables + charts only).
# Pairs with: report_css.get_css() and charts from report_charts.py.

# report_html.py
# Minimal HTML scaffolding and helpers (tables + charts only).
# Pairs with: report_css.get_css() and charts from report_charts.py.

from datetime import datetime
from html import escape
from report_css import get_css

def wrap_html(title: str, body: str) -> str:
    """Wrap whole document with <html> + inline CSS."""
    safe_title = escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{safe_title}</title>
  <style>{get_css()}</style>
</head>
<body>
  <main>
    <div class="section">
      <h1>{safe_title}</h1>
      <div class="small">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    {body}
  </main>
</body>
</html>"""

def kpi_grid(items):
    """items: list[(title, value)] → 5-column KPI grid."""
    cells = "".join(
        f"<div class='kpi'><div class='title'>{escape(str(t))}</div>"
        f"<div class='value'>{escape(str(v))}</div></div>"
        for t, v in items
    )
    return f"<div class='section'><div class='kpis'>{cells}</div></div>"

def section(title: str, inner_html: str) -> str:
    """Simple content section. Note: inner_html is trusted/pre-escaped."""
    return f"<div class='section'><h2>{escape(title)}</h2>{inner_html}</div>"

def table(headers, rows, numeric_cols=None):
    """
    Plain table. headers: list[str], rows: list[list[Any]].
    numeric_cols: set of column indices to right-align (e.g., {2, 3}).
    """
    thead = "<tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr>"
    
    tbody_rows = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cls = ' class="number"' if numeric_cols and i in numeric_cols else ''
            cells.append(f"<td{cls}>{escape(str(cell))}</td>")
        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")
    
    tbody = "\n".join(tbody_rows)
    return f"<table><thead>{thead}</thead><tbody>{tbody}</tbody></table>"

def chart(img_data_uri: str, alt_text: str = "Chart") -> str:
    """Embed a PNG base64 (from report_charts.*) as an <img>."""
    return f"<img class='chart' src='{img_data_uri}' alt='{escape(alt_text)}'/>"

def term(text: str, definition: str = "") -> str:
    """Inline glossary term with optional title tooltip."""
    title_attr = f' title="{escape(definition)}"' if definition else ''
    return f"<span class='term'{title_attr}>{escape(text)}</span>"