"""Simple HTML Report Builder

Sections:
1. Run summary (config, timestamps)
2. Data quality table
3. Timeline plot
4. Top-N events table
5. Per-event visual panels
6. Glossary (AR1, PCA, Drift)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def build_html(markdown_text: str, out_path: str | Path) -> None:
    html = f"""
    <html>
      <head>
        <meta charset='utf-8' />
        <title>ACMnxt Report</title>
        <style>body {{ font-family: Arial, sans-serif; }}</style>
      </head>
      <body>
        {markdown_text}
      </body>
    </html>
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

