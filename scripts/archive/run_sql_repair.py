import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.sql_client import SQLClient
from core.observability import Console

def run_repair_script():
    try:
        client = SQLClient.from_ini('acm')
    except Exception as e:
        Console.error(f"Failed to initialize SQLClient: {e}")
        return

    if not client.connect():
        Console.error("Failed to connect to SQL Server")
        return

    script_path = project_root / "scripts" / "sql" / "repair_missing_tables.sql"
    if not script_path.exists():
        Console.error(f"Repair script not found at {script_path}")
        return

    try:
        with open(script_path, "r") as f:
            sql_script = f.read()

        # Split by GO
        batches = sql_script.split("GO")
        
        Console.info(f"Executing repair script: {script_path}")
        
        with client.conn.cursor() as cursor:
            for i, batch in enumerate(batches):
                batch = batch.strip()
                if not batch:
                    continue
                
                try:
                    cursor.execute(batch)
                    Console.info(f"Executed batch {i+1}/{len(batches)}")
                except Exception as e:
                    Console.error(f"Error executing batch {i+1}: {e}")
                    # Continue to next batch? Or stop? 
                    # Usually better to stop on DDL errors, but some might be "already exists" which we handle in SQL
                    # The SQL script uses IF NOT EXISTS, so it should be safe.
            
            client.conn.commit()
            Console.info("Repair script execution completed.")

    except Exception as e:
        Console.error(f"Failed to execute repair script: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    run_repair_script()
