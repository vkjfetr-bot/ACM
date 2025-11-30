"""Quick script to list all ACM tables in the database."""
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
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME LIKE 'ACM_%' OR TABLE_NAME = 'ModelRegistry'
    ORDER BY TABLE_NAME
""")

tables = [row[0] for row in cur.fetchall()]
print(f"\nFound {len(tables)} ACM tables in database:\n")
for t in tables:
    print(f"  ✓ {t}")

# Check row counts for key tables
print("\n=== KEY TABLE ROW COUNTS ===\n")
key_tables = ['ACM_Scores_Wide', 'ACM_Scores_Long', 'ACM_HealthTimeline', 'ACM_Episodes', 'ModelRegistry', 'ACM_Runs']
for table in key_tables:
    if table in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM dbo.[{table}]")
            count = cur.fetchone()[0]
            print(f"  {table}: {count:,} rows")
        except Exception as e:
            print(f"  {table}: ERROR - {e}")
    else:
        print(f"  {table}: TABLE DOES NOT EXIST")

sql.close()
