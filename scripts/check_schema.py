"""Quick schema check for ACM_Runs and ACM_RunLogs."""
import sys
sys.path.insert(0, '.')

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

# Check ACM_Runs schema
print("\n=== ACM_Runs Schema ===\n")
cur.execute("""
    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'ACM_Runs'
    ORDER BY ORDINAL_POSITION
""")
for row in cur.fetchall():
    print(f"  {row[0]:30} {row[1]:20} {'NULL' if row[2] == 'YES' else 'NOT NULL'}")

# Check ACM_RunLogs schema
print("\n=== ACM_RunLogs Schema ===\n")
cur.execute("""
    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'ACM_RunLogs'
    ORDER BY ORDINAL_POSITION
""")
for row in cur.fetchall():
    print(f"  {row[0]:30} {row[1]:20} {'NULL' if row[2] == 'YES' else 'NOT NULL'}")

# Sample recent runs
print("\n=== Recent Runs (Sample) ===\n")
cur.execute("SELECT TOP 5 * FROM ACM_Runs ORDER BY RunID DESC")
columns = [desc[0] for desc in cur.description]
print(f"Columns: {', '.join(columns)}\n")
for row in cur.fetchall():
    print(f"  {dict(zip(columns, row))}")

sql.close()
