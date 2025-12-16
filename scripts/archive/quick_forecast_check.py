"""Simple check for forecast logs."""
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

cur.execute('SELECT TOP 1 RunID FROM ACM_Runs ORDER BY StartedAt DESC')
run_id = str(cur.fetchone()[0])

cur.execute('SELECT COUNT(*) FROM ACM_RunLogs WHERE RunID = ?', (run_id,))
total = cur.fetchone()[0]
print(f'\nTotal logs for {run_id}: {total}')

cur.execute("""
    SELECT COUNT(*) FROM ACM_RunLogs 
    WHERE RunID = ? 
    AND (Message LIKE '%RUL%' OR Message LIKE '%FORECAST%' OR Message LIKE '%estimate_rul%')
""", (run_id,))
forecast_count = cur.fetchone()[0]
print(f'Forecast-related logs: {forecast_count}')
print(f'Forecasting reached: {"YES" if forecast_count > 0 else "NO"}\n')

# Get last 5 messages
cur.execute("""
    SELECT TOP 5 LoggedAt, Level, Message
    FROM ACM_RunLogs
    WHERE RunID = ?
    ORDER BY LoggedAt DESC
""", (run_id,))

print("Last 5 log messages:")
for row in cur.fetchall():
    print(f"  [{row[1]}] {row[0]}")
    print(f"    {row[2][:100]}")

sql.close()
