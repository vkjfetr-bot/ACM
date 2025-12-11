"""
Comprehensive ACM Database Schema Export

Generates a detailed markdown document with:
- Table schemas (columns, data types, nullability)
- Primary keys and indexes
- Row counts and data statistics
- Top 10 and bottom 10 records for each table
- Last update timestamp per table

Usage:
    python scripts/sql/export_comprehensive_schema.py --output docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.sql_client import SQLClient
from utils.logger import Console


def _fetch_tables(cur) -> list[str]:
    """Get all base tables in the dbo schema"""
    cur.execute(
        dedent(
            """
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'dbo'
              AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
            """
        )
    )
    return [row[0] for row in cur.fetchall()]


def _fetch_columns(cur, table: str) -> list[dict]:
    """Get column details for a specific table"""
    cur.execute(
        dedent(
            """
            SELECT COLUMN_NAME,
                   DATA_TYPE,
                   IS_NULLABLE,
                   COALESCE(CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION) AS LEN_PREC,
                   COLUMN_DEFAULT,
                   ORDINAL_POSITION
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'dbo'
              AND TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
            """
        ),
        (table,),
    )
    columns: list[dict] = []
    for name, dtype, nullable, len_prec, default, pos in cur.fetchall():
        columns.append(
            {
                "name": name,
                "data_type": dtype,
                "nullable": nullable == "YES",
                "len_prec": len_prec,
                "default": default,
                "ordinal": pos,
            }
        )
    return columns


def _fetch_primary_keys(cur) -> dict[str, list[str]]:
    """Get primary key columns for all tables"""
    cur.execute(
        dedent(
            """
            SELECT KU.TABLE_NAME,
                   KU.COLUMN_NAME,
                   KU.ORDINAL_POSITION
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS TC
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS KU
              ON TC.CONSTRAINT_NAME = KU.CONSTRAINT_NAME
             AND TC.TABLE_SCHEMA = KU.TABLE_SCHEMA
            WHERE TC.TABLE_SCHEMA = 'dbo'
              AND TC.CONSTRAINT_TYPE = 'PRIMARY KEY'
            ORDER BY KU.TABLE_NAME, KU.ORDINAL_POSITION
            """
        )
    )
    pk_map: dict[str, list[str]] = defaultdict(list)
    for table, column, _ in cur.fetchall():
        pk_map[table].append(column)
    return pk_map


def _get_table_stats(cur, table: str) -> dict[str, Any]:
    """Get row count and date range for a table"""
    stats = {"row_count": 0, "min_date": None, "max_date": None, "error": None}
    
    try:
        # Get row count
        cur.execute(f"SELECT COUNT(*) FROM dbo.[{table}]")
        stats["row_count"] = cur.fetchone()[0]
        
        # Try to find timestamp columns for date range
        timestamp_cols = []
        cur.execute(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'dbo'
              AND TABLE_NAME = ?
              AND DATA_TYPE IN ('datetime', 'datetime2', 'date')
            ORDER BY ORDINAL_POSITION
            """,
            (table,)
        )
        timestamp_cols = [row[0] for row in cur.fetchall()]
        
        if timestamp_cols and stats["row_count"] > 0:
            # Use first timestamp column for date range
            ts_col = timestamp_cols[0]
            cur.execute(f"SELECT MIN([{ts_col}]), MAX([{ts_col}]) FROM dbo.[{table}]")
            min_ts, max_ts = cur.fetchone()
            if min_ts:
                stats["min_date"] = min_ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(min_ts, 'strftime') else str(min_ts)
            if max_ts:
                stats["max_date"] = max_ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(max_ts, 'strftime') else str(max_ts)
    
    except Exception as e:
        stats["error"] = str(e)
    
    return stats


def _get_sample_rows(cur, table: str, columns: list[dict], top: bool = True) -> list[dict]:
    """Get top 10 or bottom 10 rows from a table"""
    if not columns:
        return []
    
    try:
        # Build column list (limit to first 10 columns if too many)
        col_names = [col["name"] for col in columns[:10]]
        col_list = ", ".join([f"[{c}]" for c in col_names])
        
        # Try to find an ordering column (timestamp, ID, etc.)
        order_col = None
        for col in columns:
            if any(keyword in col["name"].lower() for keyword in ["timestamp", "createdat", "id", "date", "time"]):
                order_col = col["name"]
                break
        
        if not order_col:
            # Use first column as fallback
            order_col = columns[0]["name"]
        
        # Build query
        direction = "DESC" if not top else "ASC"
        query = f"SELECT TOP 10 {col_list} FROM dbo.[{table}] ORDER BY [{order_col}] {direction}"
        
        cur.execute(query)
        rows = []
        for row in cur.fetchall():
            row_dict = {}
            for i, col_name in enumerate(col_names):
                value = row[i]
                # Convert to string for display
                if value is None:
                    row_dict[col_name] = "NULL"
                elif hasattr(value, 'strftime'):  # datetime
                    row_dict[col_name] = value.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(value, (bytes, bytearray)):
                    row_dict[col_name] = f"<binary {len(value)} bytes>"
                else:
                    str_val = str(value)
                    # Truncate long strings
                    if len(str_val) > 100:
                        row_dict[col_name] = str_val[:97] + "..."
                    else:
                        row_dict[col_name] = str_val
            rows.append(row_dict)
        
        return rows
    
    except Exception as e:
        Console.warn(f"Failed to get sample rows from {table}: {e}")
        return []


def _render_table_section(table: str, columns: list[dict], pk: list[str], stats: dict, 
                         top_rows: list[dict], bottom_rows: list[dict]) -> str:
    """Render markdown section for a single table"""
    lines = [f"## dbo.{table}", ""]
    
    # Metadata
    pk_str = ", ".join(pk) if pk else "No primary key"
    lines.append(f"**Primary Key:** {pk_str}  ")
    lines.append(f"**Row Count:** {stats['row_count']:,}  ")
    
    if stats.get('min_date') and stats.get('max_date'):
        lines.append(f"**Date Range:** {stats['min_date']} to {stats['max_date']}  ")
    
    if stats.get('error'):
        lines.append(f"**Error:** {stats['error']}  ")
    
    lines.append("")
    
    # Schema table
    lines.append("### Schema")
    lines.append("")
    lines.append("| Column | Data Type | Nullable | Length/Precision | Default |")
    lines.append("| --- | --- | --- | --- | --- |")
    
    for col in columns:
        nullable = "YES" if col["nullable"] else "NO"
        len_prec = col["len_prec"] if col["len_prec"] is not None else "—"
        default = col["default"] or "—"
        # Escape pipe characters in default values
        default = str(default).replace("|", "\\|")
        lines.append(
            f"| {col['name']} | {col['data_type']} | {nullable} | {len_prec} | {default} |"
        )
    
    lines.append("")
    
    # Top 10 rows
    if top_rows:
        lines.append("### Top 10 Records")
        lines.append("")
        
        # Build markdown table
        col_names = list(top_rows[0].keys())
        lines.append("| " + " | ".join(col_names) + " |")
        lines.append("| " + " | ".join(["---"] * len(col_names)) + " |")
        
        for row in top_rows:
            values = [str(row.get(col, "")).replace("|", "\\|") for col in col_names]
            lines.append("| " + " | ".join(values) + " |")
        
        lines.append("")
    
    # Bottom 10 rows
    if bottom_rows:
        lines.append("### Bottom 10 Records")
        lines.append("")
        
        # Build markdown table
        col_names = list(bottom_rows[0].keys())
        lines.append("| " + " | ".join(col_names) + " |")
        lines.append("| " + " | ".join(["---"] * len(col_names)) + " |")
        
        for row in bottom_rows:
            values = [str(row.get(col, "")).replace("|", "\\|") for col in col_names]
            lines.append("| " + " | ".join(values) + " |")
        
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)


def _render_markdown(schema_data: dict, generated_at: datetime) -> str:
    """Render complete markdown document"""
    header = dedent(
        f"""
        # ACM Comprehensive Database Schema Reference
        
        _Generated automatically on {generated_at.strftime('%Y-%m-%d %H:%M:%S')}_
        
        This document provides detailed information about all tables in the ACM database:
        - Schema (columns, data types, nullability, defaults)
        - Primary keys
        - Row counts and date ranges
        - Top 10 and bottom 10 records per table
        
        **Generation Command:**
        ```bash
        python scripts/sql/export_comprehensive_schema.py --output docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md
        ```
        
        ---
        
        ## Table of Contents
        """
    ).strip()
    
    # Build table of contents
    toc_lines = []
    for table_name in sorted(schema_data.keys()):
        toc_lines.append(f"- [dbo.{table_name}](#dbo{table_name.lower().replace('_', '')})")
    
    # Build summary table
    summary_lines = ["", "## Summary", "", "| Table | Columns | Rows | Primary Key |", "| --- | ---: | ---: | --- |"]
    for table_name in sorted(schema_data.keys()):
        data = schema_data[table_name]
        pk = ", ".join(data["pk"]) if data["pk"] else "—"
        col_count = len(data["columns"])
        row_count = data["stats"]["row_count"]
        summary_lines.append(f"| dbo.{table_name} | {col_count} | {row_count:,} | {pk} |")
    
    # Build detailed sections
    detail_sections = []
    for table_name in sorted(schema_data.keys()):
        data = schema_data[table_name]
        section = _render_table_section(
            table_name,
            data["columns"],
            data["pk"],
            data["stats"],
            data["top_rows"],
            data["bottom_rows"]
        )
        detail_sections.append(section)
    
    # Combine all parts
    return "\n\n".join([
        header,
        "\n".join(toc_lines),
        "\n".join(summary_lines),
        "---",
        "",
        "## Detailed Table Information",
        "",
        *detail_sections
    ])


def export_comprehensive_schema(ini_section: str, output_path: Path) -> None:
    """Export comprehensive schema with stats and sample data"""
    Console.info(f"Connecting to ACM database (section: {ini_section})...")
    client = SQLClient.from_ini(ini_section)
    conn = client.connect()
    
    try:
        cur = conn.cursor()
        
        # Get all tables
        Console.info("Fetching table list...")
        tables = _fetch_tables(cur)
        Console.info(f"Found {len(tables)} tables")
        
        # Get primary keys
        Console.info("Fetching primary keys...")
        pk_map = _fetch_primary_keys(cur)
        
        # Collect comprehensive data for each table
        schema_data = {}
        for i, table in enumerate(tables, 1):
            Console.info(f"[{i}/{len(tables)}] Processing table: {table}")
            
            columns = _fetch_columns(cur, table)
            stats = _get_table_stats(cur, table)
            
            # Only get sample rows if table has data
            top_rows = []
            bottom_rows = []
            if stats["row_count"] > 0:
                top_rows = _get_sample_rows(cur, table, columns, top=True)
                if stats["row_count"] > 10:
                    bottom_rows = _get_sample_rows(cur, table, columns, top=False)
            
            schema_data[table] = {
                "columns": columns,
                "pk": pk_map.get(table, []),
                "stats": stats,
                "top_rows": top_rows,
                "bottom_rows": bottom_rows
            }
        
    finally:
        conn.close()
    
    # Render markdown
    Console.info("Generating markdown document...")
    markdown = _render_markdown(schema_data, datetime.now())
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    
    Console.info(f"✓ Comprehensive schema exported to: {output_path}")
    Console.info(f"  Total tables: {len(schema_data)}")
    Console.info(f"  Document size: {len(markdown):,} characters")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export comprehensive ACM database schema with stats and sample data."
    )
    parser.add_argument(
        "--ini-section",
        default="acm",
        help="Section inside configs/sql_connection.ini to use (default: acm).",
    )
    parser.add_argument(
        "--output",
        default="docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md",
        help="Destination Markdown file (default: docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md).",
    )
    args = parser.parse_args()
    
    try:
        export_comprehensive_schema(args.ini_section, Path(args.output))
    except Exception as e:
        Console.error(f"Failed to export schema: {e}")
        import traceback
        Console.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
