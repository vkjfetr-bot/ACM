
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
    
    # Check if DurationSeconds exists
    query_cols = "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='ACM_Runs'"
    with client.cursor() as cur:
        cur.execute(query_cols)
        cols = [r[0] for r in cur.fetchall()]
    print(f"ACM_Runs columns: {cols}")
    
    # Select run info
    query = """
    SELECT TOP 10 RunID, EquipID, StartedAt, CompletedAt, Outcome
    FROM dbo.ACM_Runs
    ORDER BY StartedAt DESC
    """
    
    with client.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        
    if not rows:
        print("No records found in ACM_Runs.")
    else:
        df = pd.DataFrame([tuple(r) for r in rows], columns=["RunID", "EquipID", "StartedAt", "CompletedAt", "Outcome"])
        print("\nRecent 10 Runs:")
        print(df.to_string())

except Exception as e:
    print(f"FAILURE: {e}")
