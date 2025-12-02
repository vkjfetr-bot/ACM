import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from core.sql_client import SQLClient

sql = SQLClient.from_ini('acm').connect()
cur = sql.cursor()

tables = ['ACM_HealthTimeline', 'ACM_SensorHotspots', 'ACM_RegimeTimeline', 'ACM_DefectTimeline', 'ACM_FailureForecast_TS']

print('Table counts for EquipID=1:')
print('-' * 50)
for tbl in tables:
    cur.execute(f'SELECT COUNT(*) FROM dbo.{tbl} WHERE EquipID=1')
    print(f'{tbl:30} {cur.fetchone()[0]:>8}')

cur.execute('SELECT MIN(Timestamp) as MinTS, MAX(Timestamp) as MaxTS FROM dbo.ACM_HealthTimeline WHERE EquipID=1')
row = cur.fetchone()
print(f'\nHealthTimeline range: {row[0]} to {row[1]}')

cur.close()
sql.close()
