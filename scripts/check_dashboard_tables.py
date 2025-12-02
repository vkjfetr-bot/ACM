"""Check which dashboard tables exist and have data."""
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

# Get all ACM tables
cur.execute("""
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME LIKE 'ACM_%' OR TABLE_NAME = 'ModelRegistry'
    ORDER BY TABLE_NAME
""")
all_tables = [row[0] for row in cur.fetchall()]

print(f"\n=== ALL ACM TABLES ({len(all_tables)}) ===\n")
for t in all_tables:
    print(f"  {t}")

# Check specific dashboard tables
dashboard_tables = [
    'ACM_Scores_Wide',
    'ACM_Scores_Long', 
    'ACM_HealthTimeline',
    'ACM_Episodes',
    'ACM_PCA_Metrics',
    'ACM_HealthForecast_TS',
    'ACM_RUL_Summary',
    'ACM_FailureForecast_TS',
    'ACM_Runs'
]

print("\n=== DASHBOARD TABLE ROW COUNTS ===\n")
for table in dashboard_tables:
    if table in all_tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM dbo.[{table}]")
            count = cur.fetchone()[0]
            print(f"  {table}: {count:,} rows")
        except Exception as e:
            print(f"  {table}: ERROR - {e}")
    else:
        print(f"  {table}: ❌ TABLE DOES NOT EXIST")

# Check date ranges for tables with data
print("\n=== DATE RANGES FOR TIME-SERIES TABLES ===\n")
ts_tables = [
    ('ACM_Scores_Wide', 'Timestamp'),
    ('ACM_Scores_Long', 'Timestamp'),
    ('ACM_HealthTimeline', 'Timestamp'),
    ('ACM_HealthForecast_TS', 'Timestamp'),
    ('ACM_FailureForecast_TS', 'Timestamp')
]

for table, ts_col in ts_tables:
    if table in all_tables:
        try:
            cur.execute(f"SELECT MIN({ts_col}), MAX({ts_col}), COUNT(*) FROM dbo.[{table}]")
            result = cur.fetchone()
            if result[0]:
                print(f"  {table}:")
                print(f"    Min: {result[0]}")
                print(f"    Max: {result[1]}")
                print(f"    Rows: {result[2]:,}")
            else:
                print(f"  {table}: No data")
        except Exception as e:
            print(f"  {table}: ERROR - {e}")

sql.close()
