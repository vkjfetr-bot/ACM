"""
Load CSV data files into ACM_HistorianData table.
This script populates the SQL historian table from the CSV files in data/ folder.
"""
import pandas as pd
import pyodbc
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.sql_client import SQLClient
from utils.logger import Console


def load_csv_to_historian(equip_id: int, equip_code: str, csv_path: Path, sql_client: SQLClient):
    """Load a CSV file into ACM_HistorianData table."""
    Console.info(f"Loading {csv_path.name} for EquipID={equip_id} ({equip_code})")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    Console.info(f"  Read {len(df)} rows with columns: {list(df.columns)}")
    
    # Identify timestamp column
    ts_col = None
    for col in ['TS', 'Ts', 'Timestamp', 'DateTime']:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {csv_path.name}")
    
    # Parse timestamps robustly (mixed formats, day-first common in datasets)
    try:
        df[ts_col] = pd.to_datetime(df[ts_col], format='mixed', dayfirst=True, errors='raise')
    except Exception:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=True, errors='raise')
        except Exception:
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
            if df[ts_col].isna().mean() > 0.05:
                raise
            Console.warn("  Some timestamps could not be parsed; dropping NaT rows")
            df = df.dropna(subset=[ts_col])
    
    # Melt into long format: one row per (timestamp, tag, value)
    sensor_cols = [c for c in df.columns if c != ts_col]
    df_long = df.melt(id_vars=[ts_col], value_vars=sensor_cols, var_name='TagName', value_name='Value')
    df_long = df_long.rename(columns={ts_col: 'Timestamp'})
    df_long['EquipID'] = equip_id
    df_long['Quality'] = 192  # Good quality
    
    # Drop NaN values
    df_long = df_long.dropna(subset=['Value'])
    
    Console.info(f"  Melted to {len(df_long)} data points across {len(sensor_cols)} tags")
    
    # Check if data already exists
    check_sql = """
        SELECT COUNT(*) FROM dbo.ACM_HistorianData 
        WHERE EquipID = ? AND Timestamp >= ? AND Timestamp <= ?
    """
    cur = sql_client.cursor()
    try:
        cur.execute(check_sql, (equip_id, df_long['Timestamp'].min(), df_long['Timestamp'].max()))
        existing_count = cur.fetchone()[0]
    finally:
        cur.close()
    
    if existing_count > 0:
        Console.warn(f"  {existing_count} existing records found in time range. Clearing old data for EquipID={equip_id}...")
        delete_sql = """
            DELETE FROM dbo.ACM_HistorianData 
            WHERE EquipID = ? AND Timestamp >= ? AND Timestamp <= ?
        """
        sql_client.execute(delete_sql, equip_id, df_long['Timestamp'].min(), df_long['Timestamp'].max())
        Console.info(f"  Cleared {existing_count} old records")
    
    # Bulk insert via executemany for performance
    insert_sql = """
        INSERT INTO dbo.ACM_HistorianData (EquipID, TagName, Timestamp, Value, Quality, CreatedAt)
        VALUES (?, ?, ?, ?, ?, GETUTCDATE())
    """
    
    batch_size = 10000
    total_inserted = 0
    
    for i in range(0, len(df_long), batch_size):
        batch = df_long.iloc[i:i+batch_size]
        rows = [(int(row['EquipID']), str(row['TagName']), row['Timestamp'].to_pydatetime(), float(row['Value']), int(row['Quality'])) 
                for _, row in batch.iterrows()]
        
        inserted = sql_client.executemany(insert_sql, rows)
        total_inserted += inserted if isinstance(inserted, int) else len(rows)
        
        if (i + batch_size) % 50000 == 0:
            Console.info(f"    Progress: {total_inserted:,} / {len(df_long):,} rows inserted")
    
    Console.info(f"  ✓ Inserted {total_inserted:,} data points into ACM_HistorianData")


def main():
    """Load all CSV files into historian table."""
    Console.info("=" * 80)
    Console.info("Loading CSV data into ACM_HistorianData table")
    Console.info("=" * 80)
    
    # Connect to SQL
    sql_client = SQLClient.from_ini('acm').connect()
    Console.info("Connected to SQL Server")
    
    # Define equipment and their CSV files
    equipment = [
        (1, 'FD_FAN', [
            'data/FD_FAN_BASELINE_DATA.csv',
            'data/FD_FAN_BATCH_DATA.csv'
        ]),
        (2621, 'GAS_TURBINE', [
            'data/GAS_TURBINE_BASELINE_DATA.csv',
            'data/GAS_TURBINE_BATCH_DATA.csv'
        ])
    ]
    
    # Load each equipment's data
    total_files = 0
    for equip_id, equip_code, csv_files in equipment:
        Console.info(f"\n{'='*80}")
        Console.info(f"Processing {equip_code} (EquipID={equip_id})")
        Console.info(f"{'='*80}")
        
        for csv_path_str in csv_files:
            csv_path = Path(csv_path_str)
            if not csv_path.exists():
                Console.warn(f"  Skipping missing file: {csv_path}")
                continue
            
            try:
                load_csv_to_historian(equip_id, equip_code, csv_path, sql_client)
                total_files += 1
            except Exception as e:
                Console.error(f"  Failed to load {csv_path.name}: {e}")
    
    Console.info(f"\n{'='*80}")
    Console.info(f"✓ Completed loading {total_files} CSV files into ACM_HistorianData")
    Console.info(f"{'='*80}")
    
    # Show summary
    summary_sql = """
        SELECT 
            h.EquipID,
            e.EquipCode,
            COUNT(DISTINCT h.TagName) AS TagCount,
            COUNT(*) AS DataPointCount,
            MIN(h.Timestamp) AS EarliestTimestamp,
            MAX(h.Timestamp) AS LatestTimestamp
        FROM dbo.ACM_HistorianData h
        JOIN dbo.Equipment e ON h.EquipID = e.EquipID
        GROUP BY h.EquipID, e.EquipCode
        ORDER BY h.EquipID
    """
    cur = sql_client.cursor()
    try:
        cur.execute(summary_sql)
        rows = cur.fetchall()
    finally:
        cur.close()
    
    Console.info("\nHistorian Data Summary:")
    Console.info("-" * 80)
    for row in rows:
        Console.info(f"  {row[1]:15} (EquipID={row[0]:4}): {row[3]:,} points, {row[2]} tags, {row[4]} to {row[5]}")
    
    sql_client.close()


if __name__ == '__main__':
    main()
