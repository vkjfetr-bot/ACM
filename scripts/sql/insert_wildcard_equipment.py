"""
Insert wildcard equipment (EquipID=0) for default config parameters.
"""
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyodbc

from utils.logger import Console


def _log(*args: Any, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False) -> None:
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message = f"{message}{end}"

    if not message:
        return

    lowered = message.lower()
    if "error" in lowered or "failed" in lowered:
        Console.error(message)
        return

    Console.info(message)


print = _log

def insert_wildcard_equipment():
    """Insert EquipID=0 as wildcard equipment for default config."""
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\B19CL3PCQLSERVER;"
        "DATABASE=ACM;"
        "Trusted_Connection=yes;"
    )
    
    try:
        print("[EQUIP] Connecting to ACM database...")
        conn = pyodbc.connect(conn_str, timeout=10)
        cursor = conn.cursor()
        
        # Check if wildcard equipment already exists
        cursor.execute("SELECT COUNT(*) FROM Equipment WHERE EquipID = 0")
        count = cursor.fetchone()[0]
        
        if count > 0:
            print(f"[EQUIP] Wildcard equipment (EquipID=0) already exists")
            cursor.close()
            conn.close()
            return
        
        # Insert wildcard equipment (IDENTITY_INSERT requires multiple statements)
        print(f"[EQUIP] Inserting wildcard equipment (EquipID=0)...")
        
        cursor.execute("SET IDENTITY_INSERT Equipment ON")
        
        cursor.execute("""
            INSERT INTO Equipment (EquipID, EquipCode, EquipName, Area, Unit, Status, CommissionDate, CreatedAtUTC)
            VALUES (0, '*', 'Default/Wildcard Config', 'Global', 'All Plants', 1, CAST('2025-01-01' AS DATETIME2), SYSUTCDATETIME())
        """)
        
        cursor.execute("SET IDENTITY_INSERT Equipment OFF")
        
        conn.commit()
        
        print(f"[EQUIP] Successfully inserted wildcard equipment (EquipID=0)")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"[EQUIP] Error inserting wildcard equipment: {e}")
        raise

if __name__ == "__main__":
    insert_wildcard_equipment()
