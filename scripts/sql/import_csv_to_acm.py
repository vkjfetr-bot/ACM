#!/usr/bin/env python
"""
Quick CSV importer for ACM database.
Usage: python scripts/sql/import_csv_to_acm.py <csv_file> <equip_code> [--ts-col TS]

Examples:
    python scripts/sql/import_csv_to_acm.py "data/Wind Turbine Scada Dataset.csv" WIND_TURBINE
    python scripts/sql/import_csv_to_acm.py "data/my_data.csv" PUMP_1 --ts-col Timestamp
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import pyodbc
from core.observability import Console


def get_connection():
    """Get SQL connection from config file."""
    import configparser
    config = configparser.ConfigParser()
    config.read(Path(__file__).resolve().parents[2] / "configs" / "sql_connection.ini")
    
    server = config.get("acm", "server")
    database = config.get("acm", "database")
    driver = config.get("acm", "driver", fallback="ODBC Driver 18 for SQL Server")
    
    conn_str = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes;TrustServerCertificate=yes"
    return pyodbc.connect(conn_str)


def import_csv(csv_path: str, equip_code: str, ts_col: str = "TS", batch_size: int = 5000):
    """Import CSV to ACM database as {EQUIP_CODE}_Data table."""
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        Console.error(f"File not found: {csv_path}")
        return False
    
    Console.info(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    Console.info(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")
    
    # Validate timestamp column
    if ts_col not in df.columns:
        Console.error(f"Timestamp column '{ts_col}' not found. Available: {list(df.columns)}")
        return False
    
    # Parse timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], format="mixed", errors="coerce")
    invalid_ts = df[ts_col].isna().sum()
    if invalid_ts > 0:
        Console.warn(f"  {invalid_ts} rows with invalid timestamps will be dropped")
        df = df.dropna(subset=[ts_col])
    
    # Rename TS to EntryDateTime for ACM standard
    df = df.rename(columns={ts_col: "EntryDateTime"})
    
    # Connect to SQL
    conn = get_connection()
    cursor = conn.cursor()
    
    table_name = f"{equip_code}_Data"
    
    # Step 1: Add equipment to master table if not exists
    Console.info(f"Checking Equipment table for '{equip_code}'...")
    cursor.execute("SELECT EquipID FROM Equipment WHERE EquipCode = ?", (equip_code,))
    row = cursor.fetchone()
    
    if row:
        equip_id = row[0]
        Console.info(f"  Equipment exists: EquipID={equip_id}")
    else:
        # Insert new equipment (EquipID is IDENTITY - auto-generated)
        cursor.execute("""
            INSERT INTO Equipment (EquipCode, EquipName, Status, CreatedAtUTC)
            OUTPUT INSERTED.EquipID
            VALUES (?, ?, 1, GETUTCDATE())
        """, (equip_code, equip_code.replace("_", " ").title()))
        equip_id = cursor.fetchone()[0]
        conn.commit()
        Console.info(f"  Created equipment: EquipID={equip_id}, EquipCode={equip_code}")
    
    # Step 2: Create data table (drop if exists to handle schema changes)
    Console.info(f"Creating table {table_name}...")
    
    # Build column definitions from DataFrame
    col_defs = ["EntryDateTime DATETIME2 NOT NULL"]
    for col in df.columns:
        if col == "EntryDateTime":
            continue
        # Sanitize column name - replace spaces with underscores
        safe_col = col.replace(" ", "_").replace("-", "_")
        dtype = df[col].dtype
        if np.issubdtype(dtype, np.floating):
            col_defs.append(f"[{safe_col}] FLOAT NULL")
        elif np.issubdtype(dtype, np.integer):
            col_defs.append(f"[{safe_col}] BIGINT NULL")
        else:
            col_defs.append(f"[{safe_col}] NVARCHAR(255) NULL")
    
    # Rename DataFrame columns to match safe names
    df.columns = [c.replace(" ", "_").replace("-", "_") for c in df.columns]
    
    # Drop and recreate table
    cursor.execute(f"DROP TABLE IF EXISTS dbo.{table_name}")
    conn.commit()
    
    create_sql = f"""
    CREATE TABLE dbo.{table_name} (
        {', '.join(col_defs)}
    );
    CREATE CLUSTERED INDEX IX_{table_name}_Time ON dbo.{table_name}(EntryDateTime);
    """
    cursor.execute(create_sql)
    conn.commit()
    Console.info(f"  Table ready: {table_name}")
    
    # Step 2b: Register tags in ACM_TagEquipmentMap
    Console.info(f"Registering {len(df.columns) - 1} sensor tags...")
    for col in df.columns:
        if col == "EntryDateTime":
            continue
        safe_col = col.replace(" ", "_").replace("-", "_")
        # Check if tag already exists
        cursor.execute("""
            SELECT TagID FROM ACM_TagEquipmentMap 
            WHERE TagName = ? AND EquipID = ?
        """, (safe_col, equip_id))
        if cursor.fetchone() is None:
            cursor.execute("""
                INSERT INTO ACM_TagEquipmentMap 
                (TagName, EquipmentName, EquipID, TagDescription, TagType, IsActive, CreatedAt)
                VALUES (?, ?, ?, ?, 'Analog', 1, GETDATE())
            """, (safe_col, equip_code, equip_id, safe_col.replace("_", " ")))
    conn.commit()
    Console.info(f"  Tags registered in ACM_TagEquipmentMap")
    
    # Step 3: Insert data in batches
    Console.info(f"Inserting {len(df):,} rows...")
    
    columns = list(df.columns)
    placeholders = ", ".join(["?"] * len(columns))
    col_names = ", ".join([f"[{c}]" for c in columns])
    insert_sql = f"INSERT INTO dbo.{table_name} ({col_names}) VALUES ({placeholders})"
    
    # Convert DataFrame to list of tuples
    df = df.replace({np.nan: None})
    
    inserted = 0
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        rows = [tuple(row) for row in batch.values]
        cursor.executemany(insert_sql, rows)
        conn.commit()
        inserted += len(rows)
        pct = inserted / len(df) * 100
        print(f"\r  Progress: {inserted:,}/{len(df):,} ({pct:.1f}%)", end="", flush=True)
    
    print()  # newline after progress
    
    # Step 4: Verify
    cursor.execute(f"SELECT COUNT(*) FROM dbo.{table_name}")
    count = cursor.fetchone()[0]
    cursor.execute(f"SELECT MIN(EntryDateTime), MAX(EntryDateTime) FROM dbo.{table_name}")
    min_ts, max_ts = cursor.fetchone()
    
    Console.info(f"Import complete!")
    Console.info(f"  Table: {table_name}")
    Console.info(f"  Rows: {count:,}")
    Console.info(f"  Date range: {min_ts} to {max_ts}")
    Console.info(f"  EquipID: {equip_id}")
    
    cursor.close()
    conn.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Import CSV to ACM database")
    parser.add_argument("csv_file", help="Path to CSV file")
    parser.add_argument("equip_code", help="Equipment code (e.g., WIND_TURBINE)")
    parser.add_argument("--ts-col", default="TS", help="Timestamp column name (default: TS)")
    parser.add_argument("--batch-size", type=int, default=5000, help="Insert batch size")
    
    args = parser.parse_args()
    
    success = import_csv(args.csv_file, args.equip_code, args.ts_col, args.batch_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
