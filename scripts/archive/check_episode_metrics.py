import pyodbc

conn = pyodbc.connect(
    'DRIVER={SQL Server};'
    'SERVER=localhost\\B19CL3PCQLSERVER;'
    'DATABASE=ACM;'
    'Trusted_Connection=yes;'
)
cursor = conn.cursor()

print("=== ACM_EpisodeMetrics Table Structure ===")
cursor.execute("SELECT TOP 10 * FROM ACM_EpisodeMetrics WHERE EquipID=1 ORDER BY RunID DESC")
columns = [col[0] for col in cursor.description]
print(f"Columns: {columns}\n")

rows = cursor.fetchall()
print(f"Total rows returned: {len(rows)}\n")

for i, row in enumerate(rows):
    print(f"Row {i+1}:")
    for col, val in zip(columns, row):
        print(f"  {col}: {val}")
    print()

# Check distinct RunIDs
cursor.execute("SELECT COUNT(DISTINCT RunID) as UniqueRuns, COUNT(*) as TotalRows FROM ACM_EpisodeMetrics WHERE EquipID=1")
stats = cursor.fetchone()
print(f"Statistics: {stats[1]} total rows, {stats[0]} unique RunIDs")

conn.close()
