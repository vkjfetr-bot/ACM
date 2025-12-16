"""Quick script to check batch processing status in SQL."""
import pyodbc

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 18 for SQL Server};'
    'SERVER=localhost\\B19CL3PCQLSERVER;'
    'DATABASE=ACM;'
    'Trusted_Connection=yes;'
    'TrustServerCertificate=yes;'
)
cursor = conn.cursor()

print("\n=== ACM_Runs Schema ===")
cursor.execute("SELECT TOP 1 * FROM ACM_Runs")
cols = [col[0] for col in cursor.description]
print(f"Columns: {', '.join(cols)}")

print("\n=== Recent ACM Runs ===")
cursor.execute("""
    SELECT TOP 10 * 
    FROM ACM_Runs 
    ORDER BY CreatedAt DESC
""")
for row in cursor.fetchall():
    print(f"RunID: {row.RunID}, EquipID: {row.EquipID}, Stage: {row.Stage if hasattr(row, 'Stage') else 'N/A'}")

print("\n=== Latest Run Details ===")
cursor.execute("""
    SELECT TOP 1 
        RunID, StartedAt, CompletedAt, DurationSeconds, 
        TrainRowCount, ScoreRowCount, HealthStatus, ErrorMessage 
    FROM ACM_Runs 
    ORDER BY CreatedAt DESC
""")
row = cursor.fetchone()
if row:
    print(f"RunID: {row.RunID}")
    print(f"Started: {row.StartedAt}")
    print(f"Completed: {row.CompletedAt}")
    print(f"Duration: {row.DurationSeconds}s" if row.DurationSeconds else "Duration: N/A")
    print(f"Train Rows: {row.TrainRowCount}")
    print(f"Score Rows: {row.ScoreRowCount}")
    print(f"Health Status: {row.HealthStatus}")
    print(f"Error: {row.ErrorMessage if row.ErrorMessage else 'None'}")

print("\n=== Data Counts for EquipID=1 (FD_FAN) ===")
cursor.execute("SELECT COUNT(*) FROM ACM_Scores_Wide WHERE EquipID=1")
scores_count = cursor.fetchone()[0]
print(f"ACM_Scores_Wide rows: {scores_count}")

cursor.execute("SELECT COUNT(*) FROM ModelRegistry WHERE EquipID=1")
model_count = cursor.fetchone()[0]
print(f"ModelRegistry rows: {model_count}")

cursor.execute("""
    SELECT TOP 1 Timestamp 
    FROM ACM_Scores_Wide 
    WHERE EquipID=1 
    ORDER BY Timestamp DESC
""")
latest_score = cursor.fetchone()
if latest_score:
    print(f"Latest score timestamp: {latest_score[0]}")

conn.close()
print("\n✓ Status check complete\n")
