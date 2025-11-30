"""
Monitor Adaptive Threshold Batch Job Progress

Tracks the ACM_FD_FAN batch job and reports:
- Threshold metadata creation
- FusedZ distribution changes
- Alert rate comparison (adaptive vs hardcoded)
"""

import pyodbc
import time
from datetime import datetime
from pathlib import Path

SERVER = r"localhost\B19CL3PCQLSERVER"
DATABASE = "ACM"
EQUIP_ID = 1  # FD_FAN


def connect_sql():
    """Connect to SQL Server."""
    conn_str = f"DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes"
    return pyodbc.connect(conn_str)


def get_threshold_stats(conn):
    """Get current threshold metadata stats."""
    query = f"""
    SELECT 
        COUNT(*) as ThresholdCount,
        COUNT(CASE WHEN RegimeID IS NULL THEN 1 END) as GlobalCount,
        COUNT(CASE WHEN RegimeID IS NOT NULL THEN 1 END) as RegimeCount,
        MAX(CreatedAt) as LatestCreation,
        AVG(CASE WHEN ThresholdType = 'fused_alert_z' AND RegimeID IS NULL THEN ThresholdValue END) as AvgAlertThreshold
    FROM ACM_ThresholdMetadata
    WHERE EquipID = {EQUIP_ID} AND IsActive = 1
    """
    cursor = conn.cursor()
    cursor.execute(query)
    row = cursor.fetchone()
    cursor.close()
    
    return {
        'threshold_count': row[0] if row else 0,
        'global_count': row[1] if row else 0,
        'regime_count': row[2] if row else 0,
        'latest_creation': row[3] if row else None,
        'avg_alert_threshold': row[4] if row else None
    }


def get_health_stats(conn):
    """Get FusedZ distribution and alert statistics."""
    query = f"""
    WITH RecentData AS (
        SELECT TOP 5000
            FusedZ,
            CASE WHEN FusedZ >= 3.0 THEN 1 ELSE 0 END as HardcodedAlert,
            CASE 
                WHEN FusedZ >= (
                    SELECT TOP 1 ThresholdValue 
                    FROM ACM_ThresholdMetadata 
                    WHERE EquipID = {EQUIP_ID} 
                      AND ThresholdType = 'fused_alert_z' 
                      AND RegimeID IS NULL 
                      AND IsActive = 1
                    ORDER BY CreatedAt DESC
                ) THEN 1 
                ELSE 0 
            END as AdaptiveAlert
        FROM ACM_HealthTimeline
        WHERE EquipID = {EQUIP_ID}
          AND FusedZ IS NOT NULL
        ORDER BY Timestamp DESC
    )
    SELECT 
        COUNT(*) as TotalPoints,
        AVG(FusedZ) as AvgFusedZ,
        STDEV(FusedZ) as StdFusedZ,
        MAX(FusedZ) as MaxFusedZ,
        PERCENTILE_CONT(0.997) WITHIN GROUP (ORDER BY FusedZ) OVER () as P997,
        AVG(CAST(HardcodedAlert as FLOAT)) * 100 as HardcodedAlertPct,
        AVG(CAST(AdaptiveAlert as FLOAT)) * 100 as AdaptiveAlertPct
    FROM RecentData
    """
    cursor = conn.cursor()
    cursor.execute(query)
    row = cursor.fetchone()
    cursor.close()
    
    if not row:
        return {}
    
    return {
        'total_points': row[0],
        'avg_fused_z': row[1],
        'std_fused_z': row[2],
        'max_fused_z': row[3],
        'p997': row[4],
        'hardcoded_alert_pct': row[5],
        'adaptive_alert_pct': row[6]
    }


def print_status(threshold_stats, health_stats):
    """Print formatted status report."""
    print("\n" + "=" * 80)
    print(f"ADAPTIVE THRESHOLD MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    print("\n[THRESHOLD METADATA]")
    print(f"  Active Thresholds: {threshold_stats.get('threshold_count', 0)}")
    print(f"    - Global: {threshold_stats.get('global_count', 0)}")
    print(f"    - Per-Regime: {threshold_stats.get('regime_count', 0)}")
    if threshold_stats.get('latest_creation'):
        print(f"  Latest Update: {threshold_stats['latest_creation']}")
    if threshold_stats.get('avg_alert_threshold'):
        print(f"  Avg Alert Threshold: {threshold_stats['avg_alert_threshold']:.3f}")
    
    print("\n[HEALTH DISTRIBUTION] (Last 5000 points)")
    if health_stats:
        print(f"  Data Points: {health_stats.get('total_points', 0):,}")
        print(f"  FusedZ Mean: {health_stats.get('avg_fused_z', 0):.3f} ± {health_stats.get('std_fused_z', 0):.3f}")
        print(f"  FusedZ Max: {health_stats.get('max_fused_z', 0):.3f}")
        print(f"  P99.7 (3-sigma): {health_stats.get('p997', 0):.3f}")
    
    print("\n[ALERT RATE COMPARISON]")
    if health_stats:
        hardcoded_pct = health_stats.get('hardcoded_alert_pct', 0)
        adaptive_pct = health_stats.get('adaptive_alert_pct', 0)
        print(f"  Hardcoded (>=3.0): {hardcoded_pct:.2f}%")
        print(f"  Adaptive (data-driven): {adaptive_pct:.2f}%")
        if hardcoded_pct > 0 and adaptive_pct > 0:
            reduction = ((hardcoded_pct - adaptive_pct) / hardcoded_pct) * 100
            if reduction > 0:
                print(f"  ✅ False Alert Reduction: {reduction:.1f}%")
            else:
                print(f"  ⚠️  Alert Increase: {abs(reduction):.1f}%")
    
    print("\n" + "=" * 80)


def main():
    """Monitor batch job progress."""
    print("Starting Adaptive Threshold Monitor...")
    print(f"Equipment: FD_FAN (ID={EQUIP_ID})")
    print(f"Update Interval: 30 seconds")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        while True:
            try:
                conn = connect_sql()
                
                threshold_stats = get_threshold_stats(conn)
                health_stats = get_health_stats(conn)
                
                print_status(threshold_stats, health_stats)
                
                conn.close()
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
            
            time.sleep(30)  # Update every 30 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


if __name__ == "__main__":
    main()
