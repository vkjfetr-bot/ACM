"""Check SQL run logs for forecasting and RUL execution."""
import sys
sys.path.insert(0, '.')

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

print("\n" + "="*80)
print("RUN LOGS ANALYSIS - Looking for Forecasting & RUL Execution")
print("="*80 + "\n")

# Check latest runs
cur.execute("""
    SELECT TOP 10 
        RunID,
        EquipID,
        StartTime,
        EndTime,
        Status,
        DATEDIFF(MINUTE, StartTime, EndTime) as DurationMin
    FROM ACM_Runs
    ORDER BY StartTime DESC
""")

print("=== LATEST 10 RUNS ===\n")
for row in cur.fetchall():
    print(f"  RunID: {row[0]}")
    print(f"  EquipID: {row[1]} | Status: {row[4]} | Duration: {row[5]} min")
    print(f"  {row[2]} → {row[3]}")
    print()

# Check for forecast/RUL related log entries
print("\n" + "="*80)
print("SEARCHING RUN LOGS FOR FORECAST/RUL/PCA KEYWORDS")
print("="*80 + "\n")

keywords = ['FORECAST', 'RUL', 'PCA', 'estimate_rul', 'enhanced_forecast', 'write_pca_metrics']

for keyword in keywords:
    cur.execute(f"""
        SELECT TOP 5
            LogLevel,
            Message,
            LogTime,
            RunID
        FROM ACM_RunLogs
        WHERE Message LIKE '%{keyword}%'
        ORDER BY LogTime DESC
    """)
    
    results = cur.fetchall()
    if results:
        print(f"\n--- Keyword: '{keyword}' ({len(results)} matches) ---")
        for row in results:
            print(f"  [{row[0]}] {row[2]} | RunID: {row[3]}")
            print(f"    {row[1][:150]}")
    else:
        print(f"\n--- Keyword: '{keyword}' --- NO MATCHES")

# Check what tables get populated per run
print("\n" + "="*80)
print("TABLE POPULATION PER RUN")
print("="*80 + "\n")

cur.execute("""
    SELECT TOP 1 RunID
    FROM ACM_Runs
    WHERE Status = 'completed'
    ORDER BY StartTime DESC
""")
latest_run = cur.fetchone()

if latest_run:
    run_id = latest_run[0]
    print(f"Latest completed run: {run_id}\n")
    
    # Check which tables have data for this run
    tables_to_check = [
        'ACM_Scores_Wide',
        'ACM_HealthTimeline',
        'ACM_Episodes',
        'ACM_PCA_Metrics',
        'ACM_HealthForecast_TS',
        'ACM_RUL_Summary',
        'ACM_FailureForecast_TS',
        'ACM_SensorForecast_TS'
    ]
    
    for table in tables_to_check:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE RunID = ?", (run_id,))
            count = cur.fetchone()[0]
            status = "✓" if count > 0 else "❌"
            print(f"  {status} {table}: {count} rows for RunID={run_id}")
        except Exception as e:
            print(f"  ❌ {table}: ERROR - {str(e)[:80]}")

# Check raw historian data availability
print("\n" + "="*80)
print("RAW HISTORIAN DATA COVERAGE")
print("="*80 + "\n")

try:
    cur.execute("""
        SELECT 
            MIN(Timestamp) as MinTime,
            MAX(Timestamp) as MaxTime,
            COUNT(DISTINCT Timestamp) as UniqueTimestamps,
            COUNT(*) as TotalRows,
            COUNT(DISTINCT TagName) as UniqueTags
        FROM ACM_HistorianData
        WHERE EquipID = 1
    """)
    
    result = cur.fetchone()
    if result and result[0]:
        print(f"  Equipment ID: 1")
        print(f"  Date Range: {result[0]} → {result[1]}")
        print(f"  Unique Timestamps: {result[2]:,}")
        print(f"  Total Rows: {result[3]:,}")
        print(f"  Unique Tags: {result[4]:,}")
    else:
        print("  ❌ No historian data found for EquipID=1")
except Exception as e:
    print(f"  ❌ ERROR: {e}")

# Compare historian vs analytics timestamps
print("\n" + "="*80)
print("TIMESTAMP ALIGNMENT: Raw Data vs Analytics")
print("="*80 + "\n")

try:
    cur.execute("""
        SELECT 
            'Historian' as Source,
            MIN(Timestamp) as MinTime,
            MAX(Timestamp) as MaxTime,
            COUNT(DISTINCT Timestamp) as UniqueTimestamps
        FROM ACM_HistorianData
        WHERE EquipID = 1
        
        UNION ALL
        
        SELECT 
            'Scores_Wide' as Source,
            MIN(Timestamp) as MinTime,
            MAX(Timestamp) as MaxTime,
            COUNT(*) as UniqueTimestamps
        FROM ACM_Scores_Wide
        WHERE EquipID = 1
        
        UNION ALL
        
        SELECT 
            'HealthTimeline' as Source,
            MIN(Timestamp) as MinTime,
            MAX(Timestamp) as MaxTime,
            COUNT(*) as UniqueTimestamps
        FROM ACM_HealthTimeline
        WHERE EquipID = 1
    """)
    
    for row in cur.fetchall():
        print(f"  {row[0]:15} | {row[1]} → {row[2]} | {row[3]:,} records")
        
except Exception as e:
    print(f"  ❌ ERROR: {e}")

sql.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80 + "\n")
