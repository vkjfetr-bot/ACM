"""Compare data ranges across key tables to identify gaps."""
from core.sql_client import SQLClient

client = SQLClient.from_ini('acm')
client.connect()
cur = client.conn.cursor()

tables_info = [
    ('ACM_HealthTimeline', 'Timestamp'),
    ('ACM_RegimeTimeline', 'Timestamp'),
    ('ACM_DriftSeries', 'Timestamp'),
    ('ACM_HealthForecast_TS', 'Timestamp'),
    ('ACM_PCA_Metrics', 'Timestamp'),
    ('ACM_SensorAnomalyByPeriod', 'PeriodEnd'),
    ('ACM_RegimeOccupancy', 'Timestamp'),
]

print('=== Data Range Comparison ===\n')
for table_name, ts_col in tables_info:
    try:
        cur.execute(f"SELECT MIN({ts_col}), MAX({ts_col}), COUNT(*) FROM {table_name} WHERE EquipID=1")
        r = cur.fetchone()
        if r and r[2] > 0:
            print(f'{table_name}:')
            print(f'  {r[2]:6d} rows: {r[0]} to {r[1]}')
        else:
            print(f'{table_name}: NO DATA')
    except Exception as e:
        print(f'{table_name}: ERROR - {e}')
    print()

print('\n=== Checking Latest Run Stage ===')
cur.execute("SELECT TOP 1 RunID, Stage, StartTime, EndTime FROM ACM_Runs WHERE EquipID=1 ORDER BY StartTime DESC")
run = cur.fetchone()
if run:
    print(f'RunID: {run[0]}')
    print(f'Stage: {run[1]}')
    print(f'Start: {run[2]}')
    print(f'End: {run[3]}')

print('\n=== Checking Config for Drift Thresholds ===')
cur.execute("SELECT ConfigKey, ConfigValue FROM ACM_Config WHERE EquipID=1 AND ConfigKey LIKE '%drift%'")
drift_configs = cur.fetchall()
if drift_configs:
    print('Drift-related configs:')
    for k, v in drift_configs:
        print(f'  {k} = {v}')
else:
    print('No drift configs found in ACM_Config')

client.conn.close()
