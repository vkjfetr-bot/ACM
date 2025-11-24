"""Check forecast and chart data status."""
from core.sql_client import SQLClient

client = SQLClient.from_ini('acm')
client.connect()
cur = client.conn.cursor()

print("=== Forecast Tables Status ===")
forecast_tables = ['ACM_HealthForecast_TS', 'ACM_FailureForecast_TS', 
                   'ACM_EnhancedFailureProbability_TS', 'ACM_RUL_Summary']
for t in forecast_tables:
    cur.execute(f"SELECT COUNT(*) FROM {t} WHERE EquipID=1")
    cnt = cur.fetchone()[0]
    print(f"{t}: {cnt} rows")
    if cnt > 0:
        cur.execute(f"SELECT MIN(Timestamp), MAX(Timestamp) FROM {t} WHERE EquipID=1")
        minmax = cur.fetchone()
        print(f"  Range: {minmax[0]} to {minmax[1]}")

print("\n=== PCA Metrics ===")
cur.execute("SELECT COUNT(*) FROM ACM_PCA_Metrics WHERE EquipID=1")
pca_cnt = cur.fetchone()[0]
print(f"ACM_PCA_Metrics: {pca_cnt} rows")
if pca_cnt > 0:
    cur.execute("SELECT TOP 5 Timestamp, ExplainedVariance FROM ACM_PCA_Metrics WHERE EquipID=1 ORDER BY Timestamp DESC")
    print("Recent values:")
    for r in cur.fetchall():
        print(f"  {r[0]} | ExplainedVar={r[1]}")

print("\n=== Sensor Anomaly Rate ===")
cur.execute("SELECT COUNT(*) FROM ACM_SensorAnomalyByPeriod WHERE EquipID=1")
sar_cnt = cur.fetchone()[0]
print(f"ACM_SensorAnomalyByPeriod: {sar_cnt} rows")

print("\n=== Regime Data Timeline ===")
cur.execute("SELECT MIN(Timestamp), MAX(Timestamp), COUNT(*) FROM ACM_RegimeTimeline WHERE EquipID=1")
regime_range = cur.fetchone()
print(f"ACM_RegimeTimeline: {regime_range[2]} rows from {regime_range[0]} to {regime_range[1]}")

print("\n=== Health Data Timeline ===")
cur.execute("SELECT MIN(Timestamp), MAX(Timestamp), COUNT(*) FROM ACM_HealthTimeline WHERE EquipID=1")
health_range = cur.fetchone()[0], cur.fetchone()[1], cur.fetchone()[2]
cur.execute("SELECT MIN(Timestamp), MAX(Timestamp), COUNT(*) FROM ACM_HealthTimeline WHERE EquipID=1")
health_range = cur.fetchone()
print(f"ACM_HealthTimeline: {health_range[2]} rows from {health_range[0]} to {health_range[1]}")

print("\n=== Comparing Regime vs Health Timeline ===")
if regime_range[1] and health_range[1]:
    if regime_range[1] > health_range[1]:
        print(f"WARNING: Regime data extends beyond health data!")
        print(f"  Regime ends: {regime_range[1]}")
        print(f"  Health ends: {health_range[1]}")
        print(f"  Gap: {(regime_range[1] - health_range[1]).total_seconds() / 3600:.1f} hours")
    else:
        print("OK: Regime and health timelines are aligned")

print("\n=== Detector Correlation Matrix ===")
cur.execute("SELECT COUNT(*) FROM ACM_DetectorCorrelation WHERE EquipID=1")
det_cnt = cur.fetchone()[0]
print(f"ACM_DetectorCorrelation: {det_cnt} rows")

print("\n=== Regime Occupancy ===")
cur.execute("SELECT COUNT(*) FROM ACM_RegimeOccupancy WHERE EquipID=1")
occ_cnt = cur.fetchone()[0]
print(f"ACM_RegimeOccupancy: {occ_cnt} rows")

print("\n=== Recent Run Info ===")
cur.execute("SELECT TOP 1 RunID, Stage, StartTime, EndTime FROM ACM_Runs WHERE EquipID=1 ORDER BY StartTime DESC")
run_info = cur.fetchone()
if run_info:
    print(f"Latest Run: {run_info[0]}")
    print(f"  Stage: {run_info[1]}")
    print(f"  Start: {run_info[2]}")
    print(f"  End: {run_info[3]}")

client.conn.close()
