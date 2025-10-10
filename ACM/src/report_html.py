"""
report_html.py

Optimized HTML scaffolding and helpers for ACM reports.
Provides efficient HTML generation with proper escaping and formatting.
"""

from datetime import datetime
from html import escape
from typing import List, Tuple, Optional, Set, Any
from report_css import get_css

# Constants
DEFAULT_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_CHART_ALT = "Chart"

# ---------- Document Structure ----------

def wrap_html(title: str, body: str) -> str:
    """
    Wrap document with HTML structure and inline CSS.
    
    Args:
        title: Page title (will be escaped)
        body: Pre-rendered HTML body content
    
    Returns:
        Complete HTML document as string
    """
    safe_title = escape(title)
    timestamp = datetime.now().strftime(DEFAULT_TIMESTAMP_FORMAT)
    
    return (
        f'<!DOCTYPE html>\n'
        f'<html lang="en">\n'
        f'<head>\n'
        f'  <meta charset="utf-8"/>\n'
        f'  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>\n'
        f'  <title>{safe_title}</title>\n'
        f'  <style>{get_css()}</style>\n'
        f'</head>\n'
        f'<body>\n'
        f'  <main>\n'
        f'    <div class="section">\n'
        f'      <h1>{safe_title}</h1>\n'
        f'      <div class="small">Generated: {timestamp}</div>\n'
        f'    </div>\n'
        f'    {body}\n'
        f'  </main>\n'
        f'</body>\n'
        f'</html>'
    )

# ---------- Content Components ----------

def kpi_grid(items: List[Tuple[str, Any]]) -> str:
    """
    Generate KPI grid display.
    
    Args:
        items: List of (title, value) tuples
    
    Returns:
        HTML string with KPI grid
    """
    if not items:
        return '<div class="section"><div class="kpis"></div></div>'
    
    # Pre-allocate list for better performance
    cells = []
    for title, value in items:
        safe_title = escape(str(title))
        safe_value = escape(str(value))
        cells.append(
            f'<div class="kpi">'
            f'<div class="title">{safe_title}</div>'
            f'<div class="value">{safe_value}</div>'
            f'</div>'
        )
    
    cells_html = ''.join(cells)
    return f'<div class="section"><div class="kpis">{cells_html}</div></div>'

def section(title: str, inner_html: str) -> str:
    """
    Create content section with title.
    
    Args:
        title: Section title (will be escaped)
        inner_html: Pre-rendered HTML content (trusted input)
    
    Returns:
        HTML section element
    """
    safe_title = escape(title)
    return f'<div class="section"><h2>{safe_title}</h2>{inner_html}</div>'

def table(
    headers: List[str], 
    rows: List[List[Any]], 
    numeric_cols: Optional[Set[int]] = None
) -> str:
    """
    Generate HTML table with optional numeric column alignment.
    
    Args:
        headers: Column headers
        rows: Table data rows
        numeric_cols: Set of column indices to right-align (0-indexed)
    
    Returns:
        HTML table element
    """
    if not headers:
        return '<table></table>'
    
    # Build header
    header_cells = [f'<th>{escape(str(h))}</th>' for h in headers]
    thead = f"<tr>{''.join(header_cells)}</tr>"
    
    # Build body rows
    if not rows:
        tbody = ''
    else:
        tbody_rows = []
        for row in rows:
            cells = []
            for col_idx, cell_value in enumerate(row):
                # Add numeric class if specified
                cls_attr = ' class="number"' if numeric_cols and col_idx in numeric_cols else ''
                safe_value = escape(str(cell_value))
                cells.append(f'<td{cls_attr}>{safe_value}</td>')
            
            tbody_rows.append(f"<tr>{''.join(cells)}</tr>")
        
        tbody = '\n'.join(tbody_rows)
    
    return f'<table>\n<thead>{thead}</thead>\n<tbody>{tbody}</tbody>\n</table>'

def chart(img_data_uri: str, alt_text: str = DEFAULT_CHART_ALT) -> str:
    """
    Embed base64-encoded image as chart.
    
    Args:
        img_data_uri: Base64 data URI for image (e.g., from matplotlib)
        alt_text: Alt text for accessibility
    
    Returns:
        HTML img element
    """
    safe_alt = escape(alt_text)
    # Note: img_data_uri should already be safe (base64), but we don't escape it
    # as it would break the data URI format
    return f'<img class="chart" src="{img_data_uri}" alt="{safe_alt}"/>'

def term(text: str, definition: str = "") -> str:
    """
    Create inline glossary term with optional tooltip.
    
    Args:
        text: Term text to display
        definition: Optional tooltip definition
    
    Returns:
        HTML span element with term styling
    """
    safe_text = escape(text)
    
    if definition:
        safe_def = escape(definition)
        return f'<span class="term" title="{safe_def}">{safe_text}</span>'
    
    return f'<span class="term">{safe_text}</span>'

# ---------- Utility Functions ----------

def empty_state(message: str = "No data available") -> str:
    """
    Generate empty state placeholder.
    
    Args:
        message: Message to display
    
    Returns:
        HTML div with empty state styling
    """
    safe_msg = escape(message)
    return f'<div class="small">{safe_msg}</div>'

def inline_code(code: str) -> str:
    """
    Format inline code snippet.
    
    Args:
        code: Code text to format
    
    Returns:
        HTML code element
    """
    safe_code = escape(code)
    return f'<code>{safe_code}</code>'

def build_list(items: List[str], ordered: bool = False) -> str:
    """
    Generate HTML list (ordered or unordered).
    
    Args:
        items: List items
        ordered: True for <ol>, False for <ul>
    
    Returns:
        HTML list element
    """
    if not items:
        return '<ul></ul>' if not ordered else '<ol></ol>'
    
    tag = 'ol' if ordered else 'ul'
    list_items = ''.join(f'<li>{escape(str(item))}</li>' for item in items)
    
    return f'<{tag}>{list_items}</{tag}>'

def alert(message: str, alert_type: str = "info") -> str:
    """
    Generate styled alert box.
    
    Args:
        message: Alert message
        alert_type: Type of alert (info, warning, error, success)
    
    Returns:
        HTML div with alert styling
    """
    safe_msg = escape(message)
    safe_type = escape(alert_type)
    return f'<div class="alert alert-{safe_type}">{safe_msg}</div>'

# ---------- Table Helpers ----------

def table_with_totals(
    headers: List[str],
    rows: List[List[Any]],
    total_cols: Optional[Set[int]] = None,
    numeric_cols: Optional[Set[int]] = None
) -> str:
    """
    Generate table with optional totals row.
    
    Args:
        headers: Column headers
        rows: Table data rows
        total_cols: Column indices to sum for totals row
        numeric_cols: Column indices to right-align
    
    Returns:
        HTML table with totals footer
    """
    if not headers or not rows:
        return table(headers, rows, numeric_cols)
    
    # Build main table
    header_cells = [f'<th>{escape(str(h))}</th>' for h in headers]
    thead = f"<tr>{''.join(header_cells)}</tr>"
    
    # Build body
    tbody_rows = []
    for row in rows:
        cells = []
        for col_idx, cell_value in enumerate(row):
            cls_attr = ' class="number"' if numeric_cols and col_idx in numeric_cols else ''
            cells.append(f'<td{cls_attr}>{escape(str(cell_value))}</td>')
        tbody_rows.append(f"<tr>{''.join(cells)}</tr>")
    
    tbody = '\n'.join(tbody_rows)
    
    # Build totals footer if specified
    if total_cols:
        totals = [''] * len(headers)
        totals[0] = 'Total'
        
        for col_idx in total_cols:
            try:
                col_values = [float(row[col_idx]) for row in rows if row[col_idx]]
                totals[col_idx] = f'{sum(col_values):,.2f}'
            except (ValueError, IndexError):
                totals[col_idx] = '—'
        
        footer_cells = []
        for col_idx, cell_value in enumerate(totals):
            cls_attr = ' class="number"' if numeric_cols and col_idx in numeric_cols else ''
            footer_cells.append(f'<td{cls_attr}><strong>{escape(str(cell_value))}</strong></td>')
        
        tfoot = f"<tr>{''.join(footer_cells)}</tr>"
        return f'<table>\n<thead>{thead}</thead>\n<tbody>{tbody}</tbody>\n<tfoot>{tfoot}</tfoot>\n</table>'
    
    return f'<table>\n<thead>{thead}</thead>\n<tbody>{tbody}</tbody>\n</table>'