"""
Test config loading and SQL connection for dual-write mode
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from utils.config_dict import ConfigDict
from core.sql_client import SQLClient

def test_config():
    print("=" * 60)
    print("Testing Config Loading")
    print("=" * 60)
    
    # Load config for FD_FAN
    cfg = ConfigDict.from_csv(Path("configs/config_table.csv"), equip_id=1)
    
    # Check dual_mode flag
    dual_mode = cfg.get("output", {}).get("dual_mode", False)
    print(f"✓ output.dual_mode = {dual_mode}")
    
    # Check SQL settings
    sql_enabled = cfg.get("sql", {}).get("enabled", False)
    print(f"✓ sql.enabled = {sql_enabled}")
    
    sql_server = cfg.get("sql", {}).get("server", "N/A")
    sql_db = cfg.get("sql", {}).get("database", "N/A")
    print(f"✓ SQL connection: {sql_server} / {sql_db}")
    
    print()
    return cfg

def test_sql_connection():
    print("=" * 60)
    print("Testing SQL Connection")
    print("=" * 60)
    
    try:
        # Connect using INI file (Windows Auth)
        client = SQLClient.from_ini('acm')
        client.connect()
        print("✓ SQL connection successful (Windows Auth via INI)")
        
        # Test query
        cur = client.cursor()
        cur.execute("SELECT COUNT(*) as equip_count FROM dbo.Equipment WHERE EquipID IN (1, 2621)")
        row = cur.fetchone()
        count = row[0] if row else 0
        cur.close()
        
        print(f"✓ Equipment table: {count} records found (FD_FAN, GAS_TURBINE)")
        
        return True
    except Exception as e:
        print(f"✗ SQL connection failed: {e}")
        return False

if __name__ == "__main__":
    config = test_config()
    sql_ok = test_sql_connection()
    
    print()
    print("=" * 60)
    if sql_ok:
        print("✓ All checks passed! Ready for dual-write mode.")
    else:
        print("✗ SQL connection issues - dual-write will not work.")
    print("=" * 60)
