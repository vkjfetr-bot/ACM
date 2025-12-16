"""Check dashboard data and chart issues."""
from core.sql_client import SQLClient
import pandas as pd

client = SQLClient.from_ini('acm')
client.connect()
cur = client.conn.cursor()

print("\n=== ACM Tables ===")
cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME LIKE 'ACM%'")
tables = sorted([r[0] for r in cur.fetchall()])
for t in tables:
    print(f"  {t}")

print("\n=== Recent Health Timeline (last 20 rows) ===")
cur.execute("SELECT TOP 20 Timestamp, HealthIndex, HealthZone, FusedZ, RunID FROM ACM_HealthTimeline WHERE EquipID=1 ORDER BY Timestamp DESC")
cols = [d[0] for d in cur.description]
rows = cur.fetchall()
print(f"Columns: {cols}")
for r in rows:
    print(f"  {r[0]} | Health={r[1]:.2f} | Zone={r[2]} | FusedZ={r[3]:.3f}")

print("\n=== Checking for Drift/Alert data ===")
# Check if we have any drift-related columns
cur.execute("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'ACM_HealthTimeline' ORDER BY ORDINAL_POSITION")
ht_cols = [r[0] for r in cur.fetchall()]
print(f"ACM_HealthTimeline columns: {ht_cols}")

print("\n=== Checking Charts Tables ===")
chart_tables = [t for t in tables if 'Chart' in t]
for ct in chart_tables:
    cur.execute(f"SELECT COUNT(*) FROM {ct} WHERE EquipID=1")
    cnt = cur.fetchone()[0]
    print(f"{ct}: {cnt} rows for EquipID=1")
    if cnt > 0:
        cur.execute(f"SELECT TOP 3 * FROM {ct} WHERE EquipID=1")
        sample_cols = [d[0] for d in cur.description]
        print(f"  Columns: {sample_cols}")

print("\n=== Checking Regime Data ===")
regime_tables = [t for t in tables if 'Regime' in t]
for rt in regime_tables:
    cur.execute(f"SELECT COUNT(*) FROM {rt} WHERE EquipID=1")
    cnt = cur.fetchone()[0]
    print(f"{rt}: {cnt} rows for EquipID=1")
    if cnt > 0:
        cur.execute(f"SELECT MIN(Timestamp), MAX(Timestamp) FROM {rt} WHERE EquipID=1")
        minmax = cur.fetchone()
        print(f"  Date range: {minmax[0]} to {minmax[1]}")

print("\n=== Checking Health Data Range ===")
cur.execute("SELECT MIN(Timestamp), MAX(Timestamp), COUNT(*) FROM ACM_HealthTimeline WHERE EquipID=1")
ht_range = cur.fetchone()
print(f"Health Timeline: {ht_range[2]} rows from {ht_range[0]} to {ht_range[1]}")

print("\n=== Checking FD_FAN_Data Range ===")
cur.execute("SELECT MIN(EntryDateTime), MAX(EntryDateTime), COUNT(*) FROM FD_FAN_Data")
data_range = cur.fetchone()
print(f"FD_FAN_Data: {data_range[2]} rows from {data_range[0]} to {data_range[1]}")

client.conn.close()
