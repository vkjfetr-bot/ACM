"""Check drift/cusum z-score values to understand alerting."""
from core.sql_client import SQLClient
import numpy as np

client = SQLClient.from_ini('acm')
client.connect()
cur = client.conn.cursor()

print("=== Checking CUSUM_Z and DRIFT_Z values ===\n")

# Get recent values from Scores_Wide
cur.execute("""
    SELECT Timestamp, cusum_z, drift_z, fused
    FROM ACM_Scores_Wide 
    WHERE EquipID=1 
    AND Timestamp >= '2024-11-01 00:00:00'
    ORDER BY Timestamp DESC
""")
rows = cur.fetchall()

if rows:
    cusum_vals = [r[1] for r in rows if r[1] is not None]
    drift_vals = [r[2] for r in rows if r[2] is not None]
    fused_vals = [r[3] for r in rows if r[3] is not None]
    
    print(f"Rows analyzed: {len(rows)}")
    print(f"  CUSUM_Z non-null: {len(cusum_vals)}")
    print(f"  DRIFT_Z non-null: {len(drift_vals)}")
    print(f"  FUSED non-null: {len(fused_vals)}")
    
    if cusum_vals:
        print(f"\nCUSUM_Z statistics:")
        print(f"  Mean: {np.mean(cusum_vals):.3f}")
        print(f"  P50:  {np.percentile(cusum_vals, 50):.3f}")
        print(f"  P95:  {np.percentile(cusum_vals, 95):.3f}")
        print(f"  P99:  {np.percentile(cusum_vals, 99):.3f}")
        print(f"  Max:  {max(cusum_vals):.3f}")
    else:
        print("\nCUSUM_Z: NO DATA")
    
    if drift_vals:
        print(f"\nDRIFT_Z statistics:")
        print(f"  Mean: {np.mean(drift_vals):.3f}")
        print(f"  P50:  {np.percentile(drift_vals, 50):.3f}")
        print(f"  P95:  {np.percentile(drift_vals, 95):.3f}")
        print(f"  P99:  {np.percentile(drift_vals, 99):.3f}")
        print(f"  Max:  {max(drift_vals):.3f}")
    else:
        print("\nDRIFT_Z: NO DATA (COLUMN IS EMPTY)")
    
    print(f"\nFUSED statistics:")
    print(f"  Mean: {np.mean(fused_vals):.3f}")
    print(f"  P50:  {np.percentile(fused_vals, 50):.3f}")
    print(f"  P95:  {np.percentile(fused_vals, 95):.3f}")
    print(f"  P99:  {np.percentile(fused_vals, 99):.3f}")
    print(f"  Max:  {max(fused_vals):.3f}")
    
    print(f"\nRecent 10 values:")
    print("Timestamp            CUSUM_Z  DRIFT_Z  FUSED")
    print("-" * 55)
    for r in rows[:10]:
        print(f"{r[0]}  {r[1]:7.3f}  {r[2]:7.3f}  {r[3]:6.3f}")

else:
    print("No data found")

print("\n=== Drift Configuration ===")
print("Config thresholds:")
print("  hysteresis_on:  3.0  (turn ON drift alert)")
print("  hysteresis_off: 1.5  (turn OFF drift alert)")
print("  p95_threshold:  2.0  (legacy fallback)")

print("\n=== Analysis ===")
if rows and cusum_vals and fused_vals:
    cusum_p95 = np.percentile(cusum_vals, 95) if cusum_vals else 0
    drift_p95 = np.percentile(drift_vals, 95) if drift_vals else 0
    fused_p95 = np.percentile(fused_vals, 95) if fused_vals else 0
    
    print(f"CUSUM_Z P95 ({cusum_p95:.3f}) vs hysteresis_on (3.0): ", end="")
    if cusum_p95 > 3.0:
        print("SHOULD TRIGGER DRIFT ALERT")
    else:
        print("below threshold (OK)")
    
    print(f"DRIFT_Z P95 ({drift_p95:.3f}) vs hysteresis_on (3.0): ", end="")
    if drift_p95 > 3.0:
        print("SHOULD TRIGGER DRIFT ALERT")
    else:
        print("below threshold (OK)")
    
    print(f"FUSED P95 ({fused_p95:.3f}) - checking drift range (2.0-5.0): ", end="")
    if 2.0 <= fused_p95 <= 5.0:
        print("IN DRIFT RANGE")
    else:
        print("outside range")

client.conn.close()
