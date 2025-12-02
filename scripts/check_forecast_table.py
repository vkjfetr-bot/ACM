"""Check if ACM_ForecastState table exists."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sql_client import SQLClient
from utils.config_dict import ConfigDict

cfg = ConfigDict({
    'sql_connection': {
        'server': 'localhost\\B19CL3PCQLSERVER',
        'database': 'ACM',
        'trusted_connection': True
    }
})

sql = SQLClient(cfg)
sql.connect()

cur = sql.cursor()
cur.execute("""
    SELECT COUNT(*) 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME = 'ACM_ForecastState'
""")
exists = cur.fetchone()[0]

print(f"ACM_ForecastState exists: {'YES' if exists > 0 else 'NO'}")

if exists:
    cur.execute("SELECT TOP 1 * FROM ACM_ForecastState")
    print(f"Columns: {[col[0] for col in cur.description]}")

sql.close()
