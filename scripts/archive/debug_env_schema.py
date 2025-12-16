
import sys
import os
from pathlib import Path

# Add core to path
sys.path.append(os.getcwd())

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    print("SUCCESS: ExponentialSmoothing imported.")
except ImportError as e:
    print(f"FAILURE: ExponentialSmoothing import failed: {e}")

try:
    from core.sql_client import SQLClient
    from core.acm_main import _load_config
    
    # Load config to get DB connection
    cfg = _load_config(Path("config.yaml"))
    client = SQLClient(cfg)
    client.connect()
    
    # Query schema for ACM_SensorForecast (or whatever the table name is)
    # Check for both "ACM_SensorForecast" and "ACM_SensorForecast_TS"
    tables = ["ACM_SensorForecast", "ACM_SensorForecast_TS"]
    
    for table in tables:
        print(f"\nChecking table: {table}")
        try:
            # Get columns using generic query (works on SQL Server)
            query = f"""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{table}'
            """
            with client.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
            
            if rows:
                cols = [r[0] for r in rows]
                print(f"Columns: {cols}")
                if "SensorName" in cols:
                    print("Status: 'SensorName' column EXISTS.")
                elif "Sensor" in cols:
                    print("Status: Found 'Sensor' column instead.")
                elif "SensorID" in cols:
                    print("Status: Found 'SensorID' column instead.")
                else:
                    print(f"Status: 'SensorName' NOT found. Available: {cols}")
            else:
                print("Status: Table not found in default schema.")
                
        except Exception as e:
            print(f"Error checking {table}: {e}")
            
except Exception as main_e:
    print(f"Setup failed: {main_e}")
