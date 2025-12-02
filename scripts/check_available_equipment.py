"""Check which equipment has data in SQL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sql_client import SQLClient
from utils.config_dict import ConfigDict

cfg = ConfigDict({
    'sql_connection': {
        'server': 'localhost\\B19CL3PCQLSERVER',
        'database': 'ACM',
        'trusted_connection': True
    }
})

sql = SQLClient(cfg)
sql.connect()

cur = sql.cursor()

print("\n=== Equipment with Data ===")
cur.execute("""
    SELECT 
        e.EquipID,
        e.EquipName,
        COUNT(DISTINCT sw.Timestamp) as DataPoints,
        MIN(sw.Timestamp) as EarliestData,
        MAX(sw.Timestamp) as LatestData
    FROM ACM_Equipment e
    LEFT JOIN ACM_Scores_Wide sw ON e.EquipID = sw.EquipID
    GROUP BY e.EquipID, e.EquipName
    ORDER BY DataPoints DESC
""")

for row in cur.fetchall():
    equip_id, equip_name, points, earliest, latest = row
    print(f"\nEquipID: {equip_id}")
    print(f"  Name: {equip_name}")
    print(f"  Data Points: {points:,}")
    if earliest and latest:
        print(f"  Date Range: {earliest} to {latest}")

print("\n=== Health Timeline Data ===")
cur.execute("""
    SELECT 
        EquipID,
        COUNT(*) as HealthPoints,
        MIN(Timestamp) as Earliest,
        MAX(Timestamp) as Latest
    FROM ACM_HealthTimeline
    GROUP BY EquipID
""")

for row in cur.fetchall():
    equip_id, points, earliest, latest = row
    print(f"EquipID {equip_id}: {points:,} health points ({earliest} to {latest})")

sql.close()
