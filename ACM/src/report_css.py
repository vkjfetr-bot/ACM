# report_css.py
# Professional dark theme for analytical reports with charts and tables.

def get_css() -> str:
    return """
/* === Base & Layout === */
* { box-sizing: border-box; }
body { 
    background: #0a0e13; 
    color: #e4e9f0; 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', Arial, sans-serif; 
    margin: 0; 
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}
main { 
    max-width: 1200px; 
    margin: 24px auto; 
    padding: 0 20px; 
}

/* === Typography === */
h1 { 
    color: #f8fafc; 
    font-size: 28px; 
    font-weight: 600; 
    margin: 0 0 8px 0; 
    letter-spacing: -0.5px;
}
h2 { 
    color: #f1f5f9; 
    font-size: 20px; 
    font-weight: 600; 
    margin: 24px 0 12px 0; 
    letter-spacing: -0.3px;
}
h3 { 
    color: #e2e8f0; 
    font-size: 16px; 
    font-weight: 600; 
    margin: 16px 0 10px 0; 
}

/* === Sections === */
.section { 
    background: #0f1419; 
    padding: 20px 24px; 
    border-radius: 8px; 
    border: 1px solid #1a2332; 
    margin: 16px 0; 
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}
.small { 
    color: #94a3b8; 
    font-size: 13px; 
    font-weight: 400;
}

/* === KPI Grid === */
.kpis { 
    display: grid; 
    grid-template-columns: repeat(5, 1fr); 
    gap: 14px; 
    margin: 16px 0;
}
@media (max-width: 900px) { 
    .kpis { grid-template-columns: repeat(3, 1fr); } 
}
@media (max-width: 600px) { 
    .kpis { grid-template-columns: repeat(2, 1fr); } 
}

.kpi { 
    background: #0f1419; 
    border: 1px solid #1e293b; 
    border-radius: 6px; 
    padding: 16px; 
    transition: border-color 0.2s ease;
}
.kpi:hover { 
    border-color: #334155; 
}
.kpi .title { 
    font-size: 11px; 
    color: #94a3b8; 
    text-transform: uppercase; 
    letter-spacing: 0.8px; 
    font-weight: 500;
    margin-bottom: 6px;
}
.kpi .value { 
    font-size: 24px; 
    font-weight: 600; 
    color: #f8fafc;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
}
.kpi .change { 
    font-size: 12px; 
    margin-top: 4px; 
    font-weight: 500;
}
.kpi .change.positive { color: #34d399; }
.kpi .change.negative { color: #f87171; }

/* === Tables === */
table { 
    border-collapse: collapse; 
    width: 100%; 
    font-size: 14px; 
    background: #0f1419;
    border-radius: 6px;
    overflow: hidden;
}
thead th { 
    background: #151b24; 
    font-weight: 600; 
    color: #cbd5e1; 
    text-transform: uppercase;
    font-size: 12px;
    letter-spacing: 0.5px;
    position: sticky;
    top: 0;
    z-index: 10;
}
th, td { 
    border: 1px solid #1e293b; 
    padding: 12px 16px; 
    text-align: left; 
}
tbody tr { 
    transition: background-color 0.15s ease;
}
tbody tr:nth-child(odd) td { 
    background: #0a0f14; 
}
tbody tr:nth-child(even) td { 
    background: #0f1419; 
}
tbody tr:hover td { 
    background: #1a2332; 
}
td.number, th.number { 
    text-align: right; 
    font-variant-numeric: tabular-nums; 
}
td.positive { color: #34d399; font-weight: 500; }
td.negative { color: #f87171; font-weight: 500; }
td.neutral { color: #94a3b8; }

/* === Charts === */
.chart-container { 
    margin: 16px 0; 
    position: relative;
}
img.chart { 
    width: 100%; 
    max-width: 100%; 
    height: auto; 
    border: 1px solid #1e293b; 
    border-radius: 6px; 
    display: block; 
    background: #0a0f14;
}

/* === Inline Elements === */
.term { 
    font-weight: 500; 
    color: #93c5fd; 
    border-bottom: 1px dotted #475569; 
    cursor: help; 
}
.badge { 
    display: inline-block; 
    padding: 2px 8px; 
    border-radius: 4px; 
    font-size: 11px; 
    font-weight: 600; 
    text-transform: uppercase; 
    letter-spacing: 0.5px;
}
.badge.high { background: #1e3a5f; color: #93c5fd; }
.badge.medium { background: #2d3748; color: #cbd5e1; }
.badge.low { background: #1f2937; color: #94a3b8; }

/* === Utilities === */
.text-center { text-align: center; }
.text-right { text-align: right; }
.mt-1 { margin-top: 8px; }
.mt-2 { margin-top: 16px; }
.mb-1 { margin-bottom: 8px; }
.mb-2 { margin-bottom: 16px; }

/* === Print Styles === */
@media print {
    body { background: white; color: black; }
    .section { 
        border: 1px solid #ddd; 
        box-shadow: none; 
        page-break-inside: avoid; 
    }
    .kpi { border: 1px solid #ddd; }
    thead th { background: #f5f5f5; color: black; }
    tbody tr:nth-child(even) td { background: #fafafa; }
}

/* === Accessibility === */
a:focus, button:focus { 
    outline: 2px solid #60a5fa; 
    outline-offset: 2px; 
}
"""