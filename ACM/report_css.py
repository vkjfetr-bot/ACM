# report_css.py
# Minimal dark theme for a charts+tables-only report (no cards/JS).
# Keep selectors generic so HTML can stay simple.

def get_css() -> str:
    return """
/* Layout */
body { background:#0b0f14; color:#e8edf2; font-family:Segoe UI, Roboto, Arial, sans-serif; margin:0; }
main { max-width:1200px; margin:20px auto; padding:0 16px; }
h1,h2,h3 { color:#eef2f7; margin:0 0 10px 0; }
.section { background:#0f1621; padding:14px 16px; border-radius:10px; border:1px solid #1f2a37; margin:12px 0; }
.small { color:#a8b3c0; font-size:12px; }

/* KPI grid - with responsive breakpoints */
.kpis { display:grid; grid-template-columns: repeat(5, 1fr); gap:12px; }
@media (max-width: 900px) { .kpis { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 600px) { .kpis { grid-template-columns: repeat(2, 1fr); } }

.kpi { background:#0f1621; border:1px solid #1f2a37; border-radius:10px; padding:12px; }
.kpi .title { font-size:12px; color:#a8b3c0; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi .value { font-size:20px; font-weight:600; margin-top:4px; }

/* Tables - improved readability */
table { border-collapse:collapse; width:100%; max-width:1100px; font-size:14px; }
th, td { border:1px solid #1f2a37; padding:8px 12px; text-align:left; }
thead th { background:#111827; font-weight:600; color:#dbeafe; }
tbody tr:nth-child(even) td { background:#0d131d; }
tbody tr:hover td { background:#1a2332; } /* hover effect */
td.number { text-align:right; font-variant-numeric: tabular-nums; } /* right-align numbers */

/* Charts (PNG data URIs) */
img.chart { width:100%; max-width:100%; height:auto; border:1px solid #334155; border-radius:8px; display:block; }

/* Inline glossary terms */
.term { font-weight:600; color:#dbeafe; border-bottom:1px dotted #475569; cursor:help; }
"""