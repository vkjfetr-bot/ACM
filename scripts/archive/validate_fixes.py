"""Quick validation script for Mahalanobis and PCA fixes."""
import pyodbc
import pandas as pd

# Connect to database
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\B19CL3PCQLSERVER;"
    "DATABASE=ACM;"
    "Trusted_Connection=yes;"
)
conn = pyodbc.connect(conn_str)

print("=" * 60)
print("MAHALANOBIS SATURATION FIX VALIDATION")
print("=" * 60)

# Query Mahalanobis scores
mhal_query = """
SELECT 
    MIN(mhal_z) AS MinMhal, 
    MAX(mhal_z) AS MaxMhal, 
    AVG(mhal_z) AS AvgMhal, 
    STDEV(mhal_z) AS StdDev,
    COUNT(*) AS NumRows
FROM ACM_Scores_Wide 
WHERE EquipID=1 AND RunID='92F445B7-C9CB-4666-885C-B679D4F5DD3C'
"""
mhal_df = pd.read_sql(mhal_query, conn)
print(mhal_df.to_string(index=False))
print()

if mhal_df['StdDev'].iloc[0] > 1.0:
    print("✅ SUCCESS: Mahalanobis scores now vary (StdDev > 1.0)")
else:
    print("❌ ISSUE: Mahalanobis scores still saturated (low StdDev)")

print("\n" + "=" * 60)
print("PCA METRICS OUTPUT VALIDATION")
print("=" * 60)

# Query PCA metrics
pca_query = """
SELECT 
    ComponentName, 
    MetricType, 
    ROUND(Value, 4) AS Value 
FROM ACM_PCA_Metrics 
WHERE EquipID=1 
ORDER BY ComponentName, MetricType
"""
pca_df = pd.read_sql(pca_query, conn)
print(pca_df.to_string(index=False))
print()

if len(pca_df) > 0:
    print(f"✅ SUCCESS: PCA metrics populated ({len(pca_df)} rows)")
    variance_rows = pca_df[pca_df['MetricType'] == 'VarianceRatio']
    if len(variance_rows) > 0:
        print(f"   - {len(variance_rows)} components with variance ratios")
else:
    print("❌ ISSUE: PCA metrics table empty")

conn.close()
