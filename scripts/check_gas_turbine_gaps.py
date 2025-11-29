"""Check if GAS_TURBINE has data in the failing date range."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.sql_client import SQLClient
from utils.config_dict import ConfigDict

cfg = ConfigDict.from_csv("configs/config_table.csv", equip_id=1)
client = SQLClient(cfg)
client.connect()
cursor = client.cursor()

# Check overall data range
cursor.execute("SELECT MIN(EntryDateTime), MAX(EntryDateTime), COUNT(*) FROM GAS_TURBINE_Data")
row = cursor.fetchone()
print(f"\nGAS_TURBINE_Data overall: {row[2]:,} rows from {row[0]} to {row[1]}")

# Check the problematic date range
cursor.execute("""
    SELECT COUNT(*) 
    FROM GAS_TURBINE_Data 
    WHERE EntryDateTime >= '2024-01-16 00:00:00' AND EntryDateTime < '2024-01-19 00:00:00'
""")
count = cursor.fetchone()[0]
print(f"\nRows between 2024-01-16 and 2024-01-19: {count}")

if count > 0:
    cursor.execute("""
        SELECT MIN(EntryDateTime), MAX(EntryDateTime)
        FROM GAS_TURBINE_Data 
        WHERE EntryDateTime >= '2024-01-16 00:00:00' AND EntryDateTime < '2024-01-19 00:00:00'
    """)
    row = cursor.fetchone()
    print(f"  Range: {row[0]} to {row[1]}")

# Check what data gaps exist
cursor.execute("""
    SELECT 
        CAST(EntryDateTime AS DATE) as DataDate,
        COUNT(*) as RowCount,
        MIN(EntryDateTime) as FirstEntry,
        MAX(EntryDateTime) as LastEntry
    FROM GAS_TURBINE_Data
    WHERE EntryDateTime >= '2024-01-01' AND EntryDateTime < '2024-02-01'
    GROUP BY CAST(EntryDateTime AS DATE)
    ORDER BY DataDate
""")
print("\nJanuary 2024 data by day:")
rows = cursor.fetchall()
for row in rows:
    print(f"  {row[0]}: {row[1]:3} rows  ({row[2]} to {row[3]})")

if not rows:
    print("  NO DATA in January 2024!")

cursor.close()
client.close()
