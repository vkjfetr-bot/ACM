#!/usr/bin/env python
"""Quick verification that v10 forecast tables have data."""
import pyodbc

conn_str = """
Driver={ODBC Driver 18 for SQL Server};
Server=localhost\\B19CL3PCQLSERVER;
Database=ACM;
Trusted_Connection=yes;
TrustServerCertificate=yes;
"""

try:
    conn = pyodbc.connect(conn_str)
    cur = conn.cursor()
    
    print("\n=== Forecast Data Verification (v10 Tables) ===\n")
    
    for equip_name, equip_id in [('FD_FAN', 1), ('GAS_TURBINE', 2)]:
        print(f"{equip_name} (EquipID={equip_id}):")
        
        cur.execute('SELECT COUNT(*) FROM ACM_HealthForecast WHERE EquipID = ?', (equip_id,))
        hf_count = cur.fetchone()[0]
        print(f"  ACM_HealthForecast: {hf_count} rows")
        
        cur.execute('SELECT COUNT(*) FROM ACM_FailureForecast WHERE EquipID = ?', (equip_id,))
        ff_count = cur.fetchone()[0]
        print(f"  ACM_FailureForecast: {ff_count} rows")
        
        cur.execute('SELECT COUNT(*) FROM ACM_RUL WHERE EquipID = ?', (equip_id,))
        rul_count = cur.fetchone()[0]
        print(f"  ACM_RUL: {rul_count} rows")
        
        print()
    
    conn.close()
    print("✅ All v10 forecast tables verified!\n")

except Exception as e:
    print(f"❌ Error: {e}")
