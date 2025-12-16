"""Monitor ACM batch progress and SQL data growth."""
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path('.').resolve()))
from core.sql_client import SQLClient

def check_progress():
    """Check and display current data counts."""
    sql = SQLClient.from_ini('acm').connect()
    cur = sql.cursor()
    
    tables = {
        'ACM_HealthTimeline': 'Health records',
        'ACM_SensorHotspots': 'Sensor hotspots',
        'ACM_RegimeTimeline': 'Regime records',
        'ACM_DefectTimeline': 'Defect records',
        'ACM_EpisodeMetrics': 'Episode metrics',
        'ACM_FailureForecast_TS': 'Failure forecasts',
        'ACM_RUL_Summary': 'RUL summaries'
    }
    
    print(f"\n{'='*60}")
    print(f"ACM Data Status - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    for tbl, desc in tables.items():
        cur.execute(f'SELECT COUNT(*) FROM dbo.{tbl} WHERE EquipID=1')
        count = cur.fetchone()[0]
        print(f"{desc:25} {count:>8}")
    
    # Get time range
    cur.execute('SELECT MIN(Timestamp), MAX(Timestamp) FROM dbo.ACM_HealthTimeline WHERE EquipID=1')
    row = cur.fetchone()
    if row and row[0]:
        print(f"\nData range: {row[0]} to {row[1]}")
    
    # Get recent run count
    cur.execute("SELECT COUNT(*) FROM dbo.ACM_Runs WHERE EquipID=1 AND ErrorMessage IS NULL AND CreatedAt > DATEADD(MINUTE, -10, GETDATE())")
    recent_runs = cur.fetchone()[0]
    print(f"Recent successful runs (last 10 min): {recent_runs}")
    
    cur.close()
    sql.close()
    print(f"{'='*60}\n")

if __name__ == '__main__':
    try:
        while True:
            check_progress()
            time.sleep(30)  # Check every 30 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
