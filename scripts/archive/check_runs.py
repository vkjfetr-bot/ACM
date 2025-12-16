from core.sql_client import SQLClient

sql = SQLClient.from_ini('acm').connect()
cur = sql.cursor()

cur.execute('SELECT TOP 15 RunID, StartedAt, CompletedAt, TrainRowCount, ScoreRowCount, HealthStatus, ErrorMessage FROM dbo.ACM_Runs WHERE EquipID=1 ORDER BY CreatedAt DESC')
rows = cur.fetchall()

print('Recent ACM Runs:')
print('-'*100)
print(f"{'Timestamp':<20} {'Train':>6} {'Score':>6} {'Status':<15} {'Error':<40}")
print('-'*100)
for r in rows:
    err = r[6][:40] if r[6] else "OK"
    print(f"{str(r[1]):<20} {r[3]:>6} {r[4]:>6} {str(r[5]):<15} {err:<40}")

cur.close()
sql.close()
