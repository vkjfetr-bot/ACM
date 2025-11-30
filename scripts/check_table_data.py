"""Quick script to check ACM SQL table row counts and data availability."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.sql_client import SQLClient
from utils.config_dict import ConfigDict

def main():
    print("Connecting to SQL Server...")
    cfg = ConfigDict.from_csv("configs/config_table.csv", equip_id=1)
    client = SQLClient(cfg)
    client.connect()
    
    tables = [
        'ACM_HealthTimeline',
        'ACM_RegimeTimeline',
        'ACM_SensorNormalized_TS',
        'ACM_DriftEvents',
        'ACM_CulpritHistory',
        'ACM_DetectorContributions',
        'ACM_EpisodeMetrics',
        'ACM_OMRTimeline',
        'ACM_OMRContributionsLong',
        'ACM_OMR_Metrics',
        'ACM_OMR_TopContributors',
        'ACM_ForecastTimeline',
        'ACM_RUL_Timeline',
    ]
    
    cursor = client.cursor()
    
    print("\nTable Row Counts:")
    print("-" * 60)
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
            row = cursor.fetchone()
            count = row[0] if row else 0
            status = "✓ HAS DATA" if count > 0 else "✗ EMPTY"
            print(f"{table:40s} {count:>10,} {status}")
        except Exception as e:
            print(f"{table:40s} {'ERROR':>10s} (table may not exist)")
    
    print("\n" + "=" * 60)
    print("Equipment IDs in system:")
    try:
        cursor.execute("SELECT DISTINCT EquipID FROM ACM_HealthTimeline ORDER BY EquipID")
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                print(f"  - {row[0]}")
        else:
            print("  No equipment data found!")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nDate ranges:")
    try:
        cursor.execute("""
            SELECT 
                MIN(Timestamp) as earliest,
                MAX(Timestamp) as latest,
                COUNT(*) as records
            FROM ACM_HealthTimeline
        """)
        row = cursor.fetchone()
        if row and row[2] > 0:
            print(f"  Earliest: {row[0]}")
            print(f"  Latest:   {row[1]}")
            print(f"  Records:  {row[2]:,}")
        else:
            print("  No timestamp data available")
    except Exception as e:
        print(f"  Error: {e}")
    
    cursor.close()
    client.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
