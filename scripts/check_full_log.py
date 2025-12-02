"""Get FULL log sequence for latest run to find where it stops."""
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

# Get latest run
cur.execute("SELECT TOP 1 RunID FROM ACM_Runs ORDER BY StartedAt DESC")
latest_run = cur.fetchone()

if latest_run:
    run_id = str(latest_run[0])
    print(f"\n{'='*80}")
    print(f"COMPLETE LOG SEQUENCE FOR RUN: {run_id}")
    print(f"{'='*80}\n")
    
    cur.execute("""
        SELECT LoggedAt, Level, Module, Message
        FROM ACM_RunLogs
        WHERE RunID = ?
        ORDER BY LoggedAt
    """, (run_id,))
    
    logs = cur.fetchall()
    for idx, log in enumerate(logs, 1):
        level_mark = {
            'ERROR': '[!]',
            'WARN': '[*]',
            'WARNING': '[*]',
            'INFO': '[+]',
            'DEBUG': '[-]'
        }.get(log[1], '[ ]')
        
        print(f"{idx:4}. {level_mark} [{log[1]:7}] {log[0]} | {log[2] or 'N/A'}")
        print(f"      {log[3]}")
        print()
    
    print(f"\n{'='*80}")
    print(f"TOTAL LOGS: {len(logs)}")
    print(f"{'='*80}\n")
    
    # Check if we reach the forecasting section
    forecast_keywords = ['RUL', 'FORECAST', 'estimate_rul', 'enhanced_forecast']
    forecast_found = any(keyword.lower() in log[3].lower() for log in logs for keyword in forecast_keywords)
    
    if forecast_found:
        print("✓ Forecasting section WAS reached in this run")
    else:
        print("❌ Forecasting section was NOT reached - pipeline likely stopped early")
        
        # Find last message
        if logs:
            last_log = logs[-1]
            print(f"\nLast log entry:")
            print(f"  [{last_log[1]}] {last_log[0]}")
            print(f"  {last_log[3]}")

sql.close()
