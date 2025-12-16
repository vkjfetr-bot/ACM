"""Quick data check - single snapshot."""
from core.sql_client import SQLClient

sql = SQLClient.from_ini('acm').connect()
cur = sql.cursor()

tables = ['ACM_HealthTimeline', 'ACM_SensorHotspots', 'ACM_RegimeTimeline', 'ACM_DefectTimeline']

print('FD_FAN Data Counts (EquipID=1):')
print('-' * 50)
for tbl in tables:
    cur.execute(f'SELECT COUNT(*) FROM dbo.{tbl} WHERE EquipID=1')
    count = cur.fetchone()[0]
    print(f'{tbl:30} {count:>8}')

cur.execute('SELECT MIN(Timestamp), MAX(Timestamp) FROM dbo.ACM_HealthTimeline WHERE EquipID=1')
row = cur.fetchone()
if row and row[0]:
    print(f'\nTimeline range: {row[0]} to {row[1]}')

cur.execute("SELECT COUNT(*) FROM dbo.ACM_Runs WHERE EquipID=1 AND ErrorMessage IS NULL")
total_runs = cur.fetchone()[0]
print(f'Total successful runs: {total_runs}')

cur.close()
sql.close()
