import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.sql_client import SQLClient
from core.observability import Console

def run_sql_script(script_path):
    try:
        client = SQLClient.from_ini('acm')
        if not client.connect():
            Console.error("Failed to connect to SQL Server")
            return

        with open(script_path, 'r') as f:
            sql_script = f.read()

        # Split by GO
        batches = sql_script.split('GO')
        
        cursor = client.cursor()
        for batch in batches:
            if batch.strip():
                try:
                    cursor.execute(batch)
                    client.conn.commit()
                    print("Executed batch successfully.")
                except Exception as e:
                    print(f"Error executing batch: {e}")
                    # Continue to next batch
        
        client.close()
        Console.info("Script execution completed.")

    except Exception as e:
        Console.error(f"Execution failed: {e}")

if __name__ == "__main__":
    script_path = project_root / "scripts/sql/create_missing_csv_tables.sql"
    run_sql_script(script_path)
