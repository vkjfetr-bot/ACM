"""Check why OMR contributions table is so large."""
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

print("Sample rows from ACM_OMRContributionsLong:")
cursor.execute("SELECT TOP 10 * FROM ACM_OMRContributionsLong ORDER BY Timestamp DESC")
print("Columns:", [col[0] for col in cursor.description])
rows = cursor.fetchall()
for row in rows:
    print(row)

print("\nStatistics:")
cursor.execute("""
    SELECT 
        COUNT(DISTINCT Timestamp) as UniqueTimestamps,
        COUNT(DISTINCT SensorName) as UniqueSensors,
        COUNT(*) as TotalRows,
        MIN(Timestamp) as Earliest,
        MAX(Timestamp) as Latest
    FROM ACM_OMRContributionsLong
""")
stats = cursor.fetchone()
print(f"  Unique timestamps: {stats[0]:,}")
print(f"  Unique sensors: {stats[1]}")
print(f"  Total rows: {stats[2]:,}")
print(f"  Date range: {stats[3]} to {stats[4]}")
print(f"\n  Calculation: {stats[0]:,} timestamps × {stats[1]} sensors = {stats[0] * stats[1]:,} expected rows")
print(f"  Actual rows: {stats[2]:,}")
print(f"  Difference: {stats[2] - (stats[0] * stats[1]):,}")

cursor.close()
client.close()
