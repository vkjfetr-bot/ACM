"""Real-time table population tracker for batch processing."""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.sql_client import SQLClient
from utils.config_dict import ConfigDict

def check_tables():
    cfg = ConfigDict.from_csv("configs/config_table.csv", equip_id=1)
    client = SQLClient(cfg)
    client.connect()
    cursor = client.cursor()

    print("\n" + "="*80)
    print("ACM BATCH PROCESSING - TABLE POPULATION STATUS")
    print("="*80)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check key tables by equipment
    equipment_tables = {
        'Core Analytics': [
            'ACM_HealthTimeline',
            'ACM_RegimeTimeline',
            'ACM_Episodes',
            'ACM_EpisodeMetrics',
        ],
        'OMR Features': [
            'ACM_OMRTimeline',
            'ACM_OMRContributionsLong',
            'ACM_OMR_Metrics',
            'ACM_OMR_TopContributors',
            'ACM_DetectorContributions',
        ],
        'Forecasting & RUL': [
            'ACM_RUL_TS',
            'ACM_RUL_Summary',
            'ACM_HealthForecast_TS',
            'ACM_MaintenanceRecommendation',
        ],
        'Sensors & Drift': [
            'ACM_SensorNormalized_TS',
            'ACM_DriftEvents',
            'ACM_CulpritHistory',
        ]
    }

    for category, tables in equipment_tables.items():
        print(f"\n{category}")
        print("-" * 80)
        for table in tables:
            try:
                # Get counts by equipment
                cursor.execute(f"""
                    SELECT 
                        COALESCE((SELECT EquipName FROM Equipment WHERE EquipID = t.EquipID), CAST(t.EquipID AS VARCHAR)) as Equipment,
                        COUNT(*) as RowCount
                    FROM {table} t
                    WHERE t.EquipID IN (1, 2621)
                    GROUP BY t.EquipID
                    ORDER BY t.EquipID
                """)
                rows = cursor.fetchall()
                if rows:
                    counts = {row[0]: row[1] for row in rows}
                    status = " | ".join([f"{eq}: {cnt:,}" for eq, cnt in counts.items()])
                    print(f"  {table:40s} {status}")
                else:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    total = cursor.fetchone()[0]
                    if total > 0:
                        print(f"  {table:40s} TOTAL: {total:,} (no EquipID breakdown)")
                    else:
                        print(f"  {table:40s} ✗ EMPTY")
            except Exception as e:
                print(f"  {table:40s} ✗ ERROR: {str(e)[:50]}")

    # Latest timestamps per equipment
    print("\n" + "="*80)
    print("LATEST DATA TIMESTAMPS")
    print("="*80)
    try:
        cursor.execute("""
            SELECT 
                COALESCE(e.EquipName, CAST(h.EquipID AS VARCHAR)) as Equipment,
                MAX(h.Timestamp) as LatestData,
                COUNT(*) as TotalRows
            FROM ACM_HealthTimeline h
            LEFT JOIN Equipment e ON h.EquipID = e.EquipID
            WHERE h.EquipID IN (1, 2621)
            GROUP BY h.EquipID, e.EquipName
            ORDER BY h.EquipID
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]:20s} Latest: {row[1]}  |  Total rows: {row[2]:,}")
    except Exception as e:
        print(f"  Error getting timestamps: {e}")

    cursor.close()
    client.close()
    print("\n" + "="*80)

if __name__ == "__main__":
    try:
        check_tables()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    except Exception as e:
        print(f"\nError: {e}")
