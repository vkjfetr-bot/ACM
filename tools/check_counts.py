import pyodbc

tables = [
    'ACM_HealthTimeline',
    'ACM_HealthZoneByPeriod',
    'ACM_SensorHotspots',
    'ACM_RUL_Attribution',
    'ACM_RUL_Summary',
    'ACM_FailureForecast_TS',
]

conn = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost\\B19CL3PCQLSERVER;DATABASE=ACM;Trusted_Connection=yes;TrustServerCertificate=yes;')
cur = conn.cursor()
for t in tables:
    cur.execute(f'SELECT COUNT(*) FROM dbo.[{t}]')
    print(f"{t}: {cur.fetchone()[0]}")
cur.close(); conn.close()
