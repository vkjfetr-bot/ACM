"""
Test config loading and SQL connection for dual-write mode
"""
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from utils.config_dict import ConfigDict
from core.sql_client import SQLClient
from core.observability import Console


_LOG_PREFIX_HANDLERS = (
    ("✗", Console.error),
    ("[ERROR]", Console.error),
    ("[ERR]", Console.error),
    ("[WARN]", Console.warn),
    ("[WARNING]", Console.warn),
    ("✓", Console.ok),
)


def _log(*args: Any, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False) -> None:
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message = f"{message}{end}"

    if not message:
        return

    if file is sys.stderr:
        Console.error(message)
        return

    trimmed = message.lstrip()
    for prefix, handler in _LOG_PREFIX_HANDLERS:
        if trimmed.startswith(prefix):
            handler(message)
            return

    Console.info(message)


print = _log

def test_config():
    _log("=" * 60)
    _log("Testing Config Loading")
    _log("=" * 60)
    
    # Load config for FD_FAN
    cfg = ConfigDict.from_csv(Path("configs/config_table.csv"), equip_id=1)
    
    # Check dual_mode flag
    dual_mode = cfg.get("output", {}).get("dual_mode", False)
    _log(f"✓ output.dual_mode = {dual_mode}")
    
    # Check SQL settings
    sql_enabled = cfg.get("sql", {}).get("enabled", False)
    _log(f"✓ sql.enabled = {sql_enabled}")
    
    sql_server = cfg.get("sql", {}).get("server", "N/A")
    sql_db = cfg.get("sql", {}).get("database", "N/A")
    _log(f"✓ SQL connection: {sql_server} / {sql_db}")
    
    _log("")
    return cfg

def test_sql_connection():
    _log("=" * 60)
    _log("Testing SQL Connection")
    _log("=" * 60)
    
    try:
        # Connect using INI file (Windows Auth)
        client = SQLClient.from_ini('acm')
        client.connect()
        _log("✓ SQL connection successful (Windows Auth via INI)")
        
        # Test query
        cur = client.cursor()
        cur.execute("SELECT COUNT(*) as equip_count FROM dbo.Equipment WHERE EquipID IN (1, 2621)")
        row = cur.fetchone()
        count = row[0] if row else 0
        cur.close()
        
        _log(f"✓ Equipment table: {count} records found (FD_FAN, GAS_TURBINE)")
        
        return True
    except Exception as e:
        _log(f"✗ SQL connection failed: {e}")
        return False

if __name__ == "__main__":
    config = test_config()
    sql_ok = test_sql_connection()
    
    _log("")
    _log("=" * 60)
    if sql_ok:
        _log("✓ All checks passed! Ready for dual-write mode.")
    else:
        _log("✗ SQL connection issues - dual-write will not work.")
    _log("=" * 60)
