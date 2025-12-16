"""Analyze which ACM tables have RunID and should show only latest run data."""
from core.sql_client import SQLClient

client = SQLClient.from_ini('acm')
client.connect()
cur = client.conn.cursor()

print("="*70)
print("ANALYSIS: Tables That Should Show Latest Run Data Only")
print("="*70)

# Get all ACM tables
cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME LIKE 'ACM_%'")
all_tables = sorted([r[0] for r in cur.fetchall()])

# Check which tables have RunID
runid_tables = []
for table in all_tables:
    cur.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}' AND COLUMN_NAME = 'RunID'")
    if cur.fetchone():
        runid_tables.append(table)

print(f"\n1. Tables WITH RunID ({len(runid_tables)} tables):")
print("-" * 70)

categories = {
    'Snapshot/Latest Run Only': [],
    'Time Series/Historical': [],
    'Configuration/Lookup': []
}

# Categorize tables
for table in runid_tables:
    # Get row count and distinct RunID count
    cur.execute(f"SELECT COUNT(*), COUNT(DISTINCT RunID) FROM {table}")
    total_rows, distinct_runs = cur.fetchone()
    
    # Get column count
    cur.execute(f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}'")
    col_count = cur.fetchone()[0]
    
    # Check if it has Timestamp column
    cur.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}' AND COLUMN_NAME = 'Timestamp'")
    has_timestamp = cur.fetchone() is not None
    
    # Categorize based on name patterns and structure
    if any(keyword in table for keyword in ['Summary', 'Current', 'Now', 'Correlation', 'Calibration', 'Ranking', 'Hotspot', 'Stability', 'Occupancy', 'Transitions', 'Dwell']):
        categories['Snapshot/Latest Run Only'].append((table, total_rows, distinct_runs, has_timestamp))
    elif any(keyword in table for keyword in ['Timeline', 'Series', '_TS', 'History', 'Forecast']):
        categories['Time Series/Historical'].append((table, total_rows, distinct_runs, has_timestamp))
    else:
        categories['Configuration/Lookup'].append((table, total_rows, distinct_runs, has_timestamp))

# Print categorized results
print("\nA. SNAPSHOT/LATEST RUN ONLY (Should filter to latest RunID):")
print("   These tables contain aggregate/summary data per run")
print("-" * 70)
for table, rows, runs, has_ts in sorted(categories['Snapshot/Latest Run Only']):
    print(f"   {table:<45} {rows:>6} rows, {runs:>3} runs, TS={has_ts}")

print("\nB. TIME SERIES/HISTORICAL (May need latest RunID depending on use case):")
print("   These tables contain time-series data with timestamps")
print("-" * 70)
for table, rows, runs, has_ts in sorted(categories['Time Series/Historical']):
    print(f"   {table:<45} {rows:>6} rows, {runs:>3} runs, TS={has_ts}")

print("\nC. CONFIGURATION/LOOKUP:")
print("-" * 70)
for table, rows, runs, has_ts in sorted(categories['Configuration/Lookup']):
    print(f"   {table:<45} {rows:>6} rows, {runs:>3} runs, TS={has_ts}")

# Check specific panels that likely need latest run filtering
print("\n" + "="*70)
print("2. COMMON DASHBOARD PANELS THAT NEED LATEST RUN FILTER:")
print("="*70)

critical_tables = [
    'ACM_DetectorCorrelation',
    'ACM_CalibrationSummary', 
    'ACM_SensorRanking',
    'ACM_ContributionCurrent',
    'ACM_SensorHotspots',
    'ACM_DefectSummary',
    'ACM_RegimeOccupancy',
    'ACM_RegimeStability',
    'ACM_SinceWhen',
    'ACM_AlertAge',
    'ACM_FusionQuality',
    'ACM_RUL_Summary',
    'ACM_MaintenanceRecommendation'
]

for table in critical_tables:
    if table in runid_tables:
        cur.execute(f"SELECT COUNT(DISTINCT RunID) FROM {table} WHERE EquipID = 1")
        run_count = cur.fetchone()[0]
        
        cur.execute(f"SELECT TOP 1 RunID FROM {table} WHERE EquipID = 1 ORDER BY RunID DESC")
        latest_run = cur.fetchone()
        latest_run_id = latest_run[0] if latest_run else None
        
        if run_count > 1:
            print(f"\n⚠️  {table}:")
            print(f"    - Has {run_count} different RunIDs for EquipID=1")
            print(f"    - Latest RunID: {latest_run_id}")
            print(f"    - ❌ NEEDS: WHERE EquipID = $equipment AND RunID = (SELECT MAX(RunID)...)")
        else:
            print(f"\n✓  {table}: Only 1 RunID (OK)")

# Sample query for getting latest RunID
print("\n" + "="*70)
print("3. RECOMMENDED QUERY PATTERN FOR LATEST RUN DATA:")
print("="*70)
print("""
-- Pattern 1: Using subquery (most reliable)
SELECT *
FROM ACM_TableName
WHERE EquipID = $equipment 
  AND RunID = (
    SELECT TOP 1 RunID 
    FROM ACM_Runs 
    WHERE EquipID = $equipment 
    ORDER BY StartedAt DESC
  )

-- Pattern 2: Using window function (for aggregations)
WITH LatestRun AS (
    SELECT *, 
           ROW_NUMBER() OVER (PARTITION BY EquipID ORDER BY RunID DESC) as rn
    FROM ACM_TableName
    WHERE EquipID = $equipment
)
SELECT * FROM LatestRun WHERE rn = 1

-- Pattern 3: Join with latest run
SELECT t.*
FROM ACM_TableName t
INNER JOIN (
    SELECT EquipID, MAX(RunID) as LatestRunID
    FROM ACM_TableName
    WHERE EquipID = $equipment
    GROUP BY EquipID
) lr ON t.EquipID = lr.EquipID AND t.RunID = lr.LatestRunID
""")

client.conn.close()
