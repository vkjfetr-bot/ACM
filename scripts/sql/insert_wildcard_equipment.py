"""
Insert wildcard equipment (EquipID=0) for default config parameters.
"""
import sys
from pathlib import Path
<<<<<<< HEAD
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyodbc

from core.observability import Console


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

=======
import pyodbc

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.observability import Console

>>>>>>> 3d95a39f2dd1a1333531c7363d383cea730a3a74
def insert_wildcard_equipment():
    """Insert EquipID=0 as wildcard equipment for default config."""
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\B19CL3PCQLSERVER;"
        "DATABASE=ACM;"
        "Trusted_Connection=yes;"
    )
    
    try:
        Console.info("Connecting to ACM database...", component="EQUIP")
        conn = pyodbc.connect(conn_str, timeout=10)
        cursor = conn.cursor()
        
        # Check if wildcard equipment already exists
        cursor.execute("SELECT COUNT(*) FROM Equipment WHERE EquipID = 0")
        count = cursor.fetchone()[0]
        
        if count > 0:
            Console.info(f"[EQUIP] Wildcard equipment (EquipID=0) already exists", equip_id=0)
            cursor.close()
            conn.close()
            return
        
        # Insert wildcard equipment (IDENTITY_INSERT requires multiple statements)
        Console.info(f"[EQUIP] Inserting wildcard equipment (EquipID=0)...", equip_id=0)
        
        cursor.execute("SET IDENTITY_INSERT Equipment ON")
        
        cursor.execute("""
            INSERT INTO Equipment (EquipID, EquipCode, EquipName, Area, Unit, Status, CommissionDate, CreatedAtUTC)
            VALUES (0, '*', 'Default/Wildcard Config', 'Global', 'All Plants', 1, CAST('2025-01-01' AS DATETIME2), SYSUTCDATETIME())
        """)
        
        cursor.execute("SET IDENTITY_INSERT Equipment OFF")
        
        conn.commit()
        
        Console.ok(f"[EQUIP] Successfully inserted wildcard equipment (EquipID=0)", equip_id=0)
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        Console.error(f"[EQUIP] Error inserting wildcard equipment: {e}", error=str(e))
        raise

if __name__ == "__main__":
    insert_wildcard_equipment()
