
import sys
import os
from pathlib import Path

# Add core to path
sys.path.append(os.getcwd())

try:
    from core.sql_client import SQLClient
    from core.acm_main import _load_config
    
    # Load config into a dummy args object or dict
    # _load_config expects a path
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("Config file not found, trying defaults or env vars...")
        cfg = {"sql": {}} 
    else:
        cfg = _load_config(config_path)

    print("Connecting to SQL...")
    client = SQLClient(cfg)
    client.connect()
    
    create_table_sql = """
    IF OBJECT_ID('dbo.ACM_RunTimers', 'U') IS NULL
    BEGIN
        CREATE TABLE dbo.ACM_RunTimers (
            TimerID INT IDENTITY(1,1) PRIMARY KEY,
            RunID VARCHAR(50) NOT NULL,
            EquipID INT NOT NULL,
            BatchNum INT DEFAULT 0,
            Section VARCHAR(100) NOT NULL,
            DurationSeconds FLOAT NOT NULL,
            CreatedAt DATETIME DEFAULT GETUTCDATE(),
            INDEX IX_ACM_RunTimers_RunID (RunID),
            INDEX IX_ACM_RunTimers_EquipID (EquipID)
        )
        PRINT 'Table ACM_RunTimers created.'
    END
    ELSE
    BEGIN
        PRINT 'Table ACM_RunTimers already exists.'
    END
    """
    
    with client.cursor() as cur:
        cur.execute(create_table_sql)
        client.conn.commit()
        
    print("SUCCESS: Table verification complete.")
    
except Exception as e:
    print(f"FAILURE: {e}")
