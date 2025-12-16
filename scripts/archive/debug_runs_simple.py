
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
    
    # Select distinct EquipID
    query = """
    SELECT DISTINCT EquipID, EquipName
    FROM dbo.ACM_Runs
    ORDER BY EquipID
    """
    
    with client.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        
    if not rows:
        print("No records found in ACM_Runs.")
    else:
        print("Valid EquipIDs found:")
        for r in rows:
            print(r)

except Exception as e:
    print(f"FAILURE: {e}")
