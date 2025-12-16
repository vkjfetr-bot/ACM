
import sys
import os
from pathlib import Path
import pandas as pd

# Add core to path
sys.path.append(os.getcwd())

try:
    from core.sql_client import SQLClient
    from core.acm_main import _load_config
    
    config_path = Path("config.yaml")
    cfg = _load_config(config_path) if config_path.exists() else {"sql": {}}
    
    print("Connecting to SQL...")
    client = SQLClient(cfg)
    client.connect()
    
    query = """
    SELECT TOP 20 *
    FROM dbo.ACM_RunTimers
    ORDER BY CreatedAt DESC
    """
    
    with client.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        
    if not rows:
        print("No records found in ACM_RunTimers.")
    else:
        # Columns: TimerID, RunID, EquipID, BatchNum, Section, DurationSeconds, CreatedAt
        cols = ["TimerID", "RunID", "EquipID", "BatchNum", "Section", "DurationSeconds", "CreatedAt"]
        df = pd.DataFrame([tuple(r) for r in rows], columns=cols)
        print("\nRecent 20 Timer Records:")
        print(df.to_string())
        
        # Check for meaningful durations
        count_query = "SELECT COUNT(*) FROM dbo.ACM_RunTimers WHERE DurationSeconds > 10"
        with client.cursor() as cur:
            cur.execute(count_query)
            count = cur.fetchone()[0]
        print(f"\nTotal records with Duration > 10s: {count}")

except Exception as e:
    print(f"FAILURE: {e}")
