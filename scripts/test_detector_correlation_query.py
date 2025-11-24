"""Test the detector correlation query for Grafana heatmap."""
from core.sql_client import SQLClient

client = SQLClient.from_ini('acm')
client.connect()
cur = client.conn.cursor()

print("=== Testing Detector Correlation Query ===\n")

# Query as shown in Grafana screenshot
query = """
SELECT DetectorA as detector1, DetectorB as detector2, PearsonR as correlation
FROM ACM_DetectorCorrelation
WHERE EquipID = $equipment
"""

print("Query from Grafana:")
print(query)
print("\nExecuting with EquipID=1...\n")

cur.execute("""
    SELECT DetectorA as detector1, DetectorB as detector2, PearsonR as correlation
    FROM ACM_DetectorCorrelation
    WHERE EquipID = 1
    ORDER BY DetectorA, DetectorB
""")

rows = cur.fetchall()
print(f"Total rows returned: {len(rows)}")

if rows:
    print("\nSample data (first 15 rows):")
    print(f"{'Detector A':<15} | {'Detector B':<15} | Correlation")
    print("-" * 55)
    for r in rows[:15]:
        print(f"{r[0]:<15} | {r[1]:<15} | {r[2]:8.4f}")
    
    print("\nChecking for unique detectors:")
    cur.execute("""
        SELECT DISTINCT DetectorA 
        FROM ACM_DetectorCorrelation 
        WHERE EquipID = 1
        ORDER BY DetectorA
    """)
    detectors_a = [r[0] for r in cur.fetchall()]
    
    cur.execute("""
        SELECT DISTINCT DetectorB 
        FROM ACM_DetectorCorrelation 
        WHERE EquipID = 1
        ORDER BY DetectorB
    """)
    detectors_b = [r[0] for r in cur.fetchall()]
    
    print(f"\nUnique DetectorA values ({len(detectors_a)}): {detectors_a}")
    print(f"Unique DetectorB values ({len(detectors_b)}): {detectors_b}")
    
    # Check for unique RunIDs
    cur.execute("""
        SELECT RunID, COUNT(*) as row_count
        FROM ACM_DetectorCorrelation
        WHERE EquipID = 1
        GROUP BY RunID
        ORDER BY COUNT(*) DESC
    """)
    run_ids = cur.fetchall()
    print(f"\nNumber of different RunIDs: {len(run_ids)}")
    if run_ids:
        print(f"Most common RunID: {run_ids[0][0]} ({run_ids[0][1]} rows)")
    
    # Test aggregated query (what Grafana needs)
    print("\n" + "="*60)
    print("RECOMMENDED QUERY FOR GRAFANA HEATMAP:")
    print("="*60)
    recommended_query = """
SELECT 
    DetectorA as detector1, 
    DetectorB as detector2, 
    AVG(PearsonR) as correlation
FROM ACM_DetectorCorrelation
WHERE EquipID = $equipment
GROUP BY DetectorA, DetectorB
ORDER BY DetectorA, DetectorB
"""
    print(recommended_query)
    
    print("\nExecuting aggregated query:")
    cur.execute("""
        SELECT 
            DetectorA as detector1, 
            DetectorB as detector2, 
            AVG(PearsonR) as correlation
        FROM ACM_DetectorCorrelation
        WHERE EquipID = 1
        GROUP BY DetectorA, DetectorB
        ORDER BY DetectorA, DetectorB
    """)
    agg_rows = cur.fetchall()
    print(f"\nAggregated result: {len(agg_rows)} rows (one per detector pair)")
    print("\nAggregated data:")
    print(f"{'Detector A':<15} | {'Detector B':<15} | Correlation")
    print("-" * 55)
    for r in agg_rows[:20]:
        print(f"{r[0]:<15} | {r[1]:<15} | {r[2]:8.4f}")

else:
    print("ERROR: No data returned!")

client.conn.close()
