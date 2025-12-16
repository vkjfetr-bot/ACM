"""Analyze run failures and check for forecast/RUL execution."""
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
print("RUN FAILURE ANALYSIS")
print("="*80 + "\n")

# Get latest run
cur.execute("SELECT TOP 1 RunID, EquipID, StartedAt FROM ACM_Runs ORDER BY StartedAt DESC")
latest_run = cur.fetchone()

if latest_run:
    run_id, equip_id, started_at = latest_run
    print(f"Latest Run: {run_id}")
    print(f"Equipment: {equip_id}")
    print(f"Started: {started_at}\n")
    
    # Get all logs for this run
    print(f"{'='*80}")
    print(f"ALL LOGS FOR RUN {run_id}")
    print(f"{'='*80}\n")
    
    cur.execute("""
        SELECT LoggedAt, Level, Module, Message
        FROM ACM_RunLogs
        WHERE RunID = ?
        ORDER BY LoggedAt
    """, (str(run_id),))
    
    logs = cur.fetchall()
    if logs:
        for log in logs:
            level_prefix = {
                'ERROR': '[!]',
                'WARN': '[*]',
                'INFO': '[+]',
                'DEBUG': '[-]'
            }.get(log[1], '[ ]')
            
            print(f"{level_prefix} [{log[1]:5}] {log[0]} | {log[2] or 'N/A'}")
            print(f"  {log[3]}")
            print()
    else:
        print("  ❌ NO LOGS FOUND FOR THIS RUN")

# Check for successful runs with data
print(f"\n{'='*80}")
print("SUCCESSFUL RUNS WITH DATA")
print(f"{'='*80}\n")

cur.execute("""
    SELECT TOP 5 
        RunID, EquipID, StartedAt, CompletedAt,
        TrainRowCount, ScoreRowCount, EpisodeCount,
        HealthStatus
    FROM ACM_Runs
    WHERE TrainRowCount > 0 AND ScoreRowCount > 0
    ORDER BY StartedAt DESC
""")

successful_runs = cur.fetchall()
if successful_runs:
    for run in successful_runs:
        print(f"  RunID: {run[0]}")
        print(f"  Started: {run[2]} | Completed: {run[3]}")
        print(f"  Train: {run[4]} | Score: {run[5]} | Episodes: {run[6]} | Health: {run[7]}")
        print()
else:
    print("  ❌ NO SUCCESSFUL RUNS FOUND WITH DATA")

# Search for forecast/RUL keywords in all logs
print(f"\n{'='*80}")
print("SEARCHING FOR FORECAST/RUL/PCA IN ALL RUN LOGS")
print(f"{'='*80}\n")

keywords = ['FORECAST', 'RUL', 'PCA', 'estimate_rul', 'enhanced_forecast']

for keyword in keywords:
    cur.execute("""
        SELECT TOP 3
            RunID, LoggedAt, Level, Message
        FROM ACM_RunLogs
        WHERE Message LIKE ?
        ORDER BY LoggedAt DESC
    """, (f'%{keyword}%',))
    
    results = cur.fetchall()
    print(f"\n--- '{keyword}' ({len(results)} recent matches) ---")
    if results:
        for row in results:
            print(f"  [{row[2]}] {row[1]} | RunID: {row[0]}")
            print(f"    {row[3][:150]}")
    else:
        print(f"  ❌ NO MATCHES FOUND")

# Check raw historian data
print(f"\n{'='*80}")
print("RAW HISTORIAN DATA CHECK")
print(f"{'='*80}\n")

cur.execute("""
    SELECT 
        EquipID,
        MIN(Timestamp) as MinTime,
        MAX(Timestamp) as MaxTime,
        COUNT(*) as TotalRows,
        COUNT(DISTINCT Timestamp) as UniqueTimestamps
    FROM ACM_HistorianData
    GROUP BY EquipID
    ORDER BY EquipID
""")

historian_data = cur.fetchall()
if historian_data:
    for row in historian_data:
        print(f"  Equipment {row[0]}:")
        print(f"    Date Range: {row[1]} → {row[2]}")
        print(f"    Total Rows: {row[3]:,} | Unique Timestamps: {row[4]:,}")
        print()
else:
    print("  ❌ NO HISTORIAN DATA FOUND")

sql.close()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80 + "\n")
