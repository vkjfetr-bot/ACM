"""Check what data exists in the historian tables."""
from core.sql_client import SQLClient

client = SQLClient.from_ini('acm')
client.connect()
cur = client.conn.cursor()

# Check for FD_FAN data
print("\n=== Checking FD_FAN_Data ===")
try:
    cur.execute("SELECT MIN(EntryDateTime), MAX(EntryDateTime), COUNT(*) FROM FD_FAN_Data")
    row = cur.fetchone()
    if row and row[2] > 0:
        print(f"Rows: {row[2]}")
        print(f"Date range: {row[0]} to {row[1]}")
    else:
        print("No data found")
except Exception as e:
    print(f"Error: {e}")

# Check daily distribution in October 2023
print("\n=== Daily Data Distribution (Oct 2023) ===")
cur.execute("""
    SELECT CAST(EntryDateTime AS DATE) as DateOnly, COUNT(*) as Cnt
    FROM FD_FAN_Data 
    WHERE EntryDateTime >= '2023-10-15' AND EntryDateTime < '2023-10-25'
    GROUP BY CAST(EntryDateTime AS DATE)
    ORDER BY DateOnly
""")
print("Date           Rows")
print("-" * 25)
for row in cur.fetchall():
    print(f"{row[0]}  {row[1]:5d}")

# Check what tables exist
print("\n=== Equipment Data Tables ===")
cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dbo' AND TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME")
all_tables = [r[0] for r in cur.fetchall()]
data_tables = [t for t in all_tables if '_Data' in t or 'Historian' in t]
print(f"Found {len(data_tables)} data tables:")
for t in data_tables:
    print(f"  - {t}")

client.conn.close()
