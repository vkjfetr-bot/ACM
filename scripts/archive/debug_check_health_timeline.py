import pyodbc
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from core.sql_client import SQLClient

def check_health_timeline():
    print("Checking ACM_HealthTimeline...")
    
    try:
        # Use SQLClient to get connection
        client = SQLClient.from_ini("acm")
        client.connect()
        conn = client.conn
        
        if not conn:
            print("Error: Could not connect to SQL Server.")
            return

        cursor = conn.cursor()
        
        # Check total count
        cursor.execute("SELECT COUNT(*) FROM ACM_HealthTimeline")
        count = cursor.fetchone()[0]
        print(f"Total rows in ACM_HealthTimeline: {count}")
        
        # Check distinct EquipIDs
        cursor.execute("SELECT DISTINCT EquipID FROM ACM_HealthTimeline")
        equip_ids = [row[0] for row in cursor.fetchall()]
        print(f"Distinct EquipIDs in ACM_HealthTimeline: {equip_ids}")
        
        # Check specific EquipID 1
        cursor.execute("SELECT COUNT(*) FROM ACM_HealthTimeline WHERE EquipID = 1")
        count_1 = cursor.fetchone()[0]
        print(f"Rows for EquipID 1 (FD_FAN): {count_1}")

        # Check ACM_Runs for EquipID 1
        cursor.execute("SELECT COUNT(*) FROM ACM_Runs WHERE EquipID = 1")
        run_count = cursor.fetchone()[0]
        print(f"Runs for EquipID 1 (FD_FAN): {run_count}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_health_timeline()
