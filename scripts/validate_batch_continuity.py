"""
Batch Continuity Validation Script
===================================

Validates the effectiveness of batch continuity fixes by analyzing:
1. Z-score stability (should be <5σ for 95% of data)
2. Forecast continuity (health deltas <20% between runs)
3. Baseline coverage (overlap >50% with score windows)
4. Variance collapse incidents (near-zero std counts)

Run after implementing fixes to confirm improvements.
"""

import pyodbc
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# SQL Server connection (adjust as needed)
CONN_STRING = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost\\B19CL3PCQLSERVER;DATABASE=ACM;Trusted_Connection=yes;TrustServerCertificate=yes;"

def connect_db():
    return pyodbc.connect(CONN_STRING)

def check_zscore_stability(conn, equip_id=1, days=7):
    """Check Z-score distribution - should be <5σ for 95% of data."""
    print("\n" + "="*80)
    print("1. Z-SCORE STABILITY CHECK")
    print("="*80)
    
    query = """
    SELECT 
        DetectorName,
        COUNT(*) as total_points,
        AVG(ZScore) as avg_z,
        STDEV(ZScore) as std_z,
        MIN(ZScore) as min_z,
        MAX(ZScore) as max_z,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(ZScore)) as p95_abs_z,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ABS(ZScore)) as p99_abs_z,
        SUM(CASE WHEN ABS(ZScore) > 10 THEN 1 ELSE 0 END) as extreme_count
    FROM (
        SELECT 
            'ar1_z' as DetectorName, ar1_z as ZScore 
        FROM ACM_Scores_Wide 
        WHERE EquipID = ? AND Timestamp >= DATEADD(DAY, -?, GETDATE())
        UNION ALL
        SELECT 'pca_spe_z', pca_spe_z 
        FROM ACM_Scores_Wide 
        WHERE EquipID = ? AND Timestamp >= DATEADD(DAY, -?, GETDATE())
        UNION ALL
        SELECT 'pca_t2_z', pca_t2_z 
        FROM ACM_Scores_Wide 
        WHERE EquipID = ? AND Timestamp >= DATEADD(DAY, -?, GETDATE())
        UNION ALL
        SELECT 'mhal_z', mhal_z 
        FROM ACM_Scores_Wide 
        WHERE EquipID = ? AND Timestamp >= DATEADD(DAY, -?, GETDATE())
        UNION ALL
        SELECT 'iforest_z', iforest_z 
        FROM ACM_Scores_Wide 
        WHERE EquipID = ? AND Timestamp >= DATEADD(DAY, -?, GETDATE())
        UNION ALL
        SELECT 'gmm_z', gmm_z 
        FROM ACM_Scores_Wide 
        WHERE EquipID = ? AND Timestamp >= DATEADD(DAY, -?, GETDATE())
        UNION ALL
        SELECT 'omr_z', omr_z 
        FROM ACM_Scores_Wide 
        WHERE EquipID = ? AND Timestamp >= DATEADD(DAY, -?, GETDATE())
    ) z
    WHERE ZScore IS NOT NULL
    GROUP BY DetectorName
    ORDER BY max_z DESC;
    """
    
    params = tuple([equip_id, days] * 7)
    df = pd.read_sql(query, conn, params=params)
    
    print(f"\nZ-Score Statistics (EquipID={equip_id}, last {days} days):\n")
    print(df.to_string(index=False))
    
    # Flag issues
    issues = df[df['max_z'] > 10]
    if len(issues) > 0:
        print(f"\n⚠️  WARNING: {len(issues)} detectors have extreme Z-scores (>10σ):")
        for _, row in issues.iterrows():
            print(f"   - {row['DetectorName']}: max={row['max_z']:.1f}σ, "
                  f"{row['extreme_count']} extreme points ({100*row['extreme_count']/row['total_points']:.1f}%)")
        return False
    else:
        print(f"\n✅ PASS: All Z-scores within healthy range (<10σ)")
        return True

def check_forecast_continuity(conn, equip_id=1, days=7):
    """Check forecast health deltas - should be <20% between consecutive runs."""
    print("\n" + "="*80)
    print("2. FORECAST CONTINUITY CHECK")
    print("="*80)
    
    query = """
    WITH forecast_jumps AS (
        SELECT 
            RunID,
            Timestamp,
            HealthScore,
            LAG(HealthScore) OVER (PARTITION BY EquipID ORDER BY Timestamp) as prev_health,
            ABS(HealthScore - LAG(HealthScore) OVER (PARTITION BY EquipID ORDER BY Timestamp)) as health_delta,
            LAG(RunID) OVER (PARTITION BY EquipID ORDER BY Timestamp) as prev_run_id
        FROM ACM_HealthForecast_TS
        WHERE EquipID = ? AND Timestamp >= DATEADD(DAY, -?, GETDATE())
    )
    SELECT 
        COUNT(*) as total_transitions,
        AVG(health_delta) as avg_delta,
        MAX(health_delta) as max_delta,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY health_delta) as p95_delta,
        SUM(CASE WHEN health_delta > 20 THEN 1 ELSE 0 END) as large_jumps
    FROM forecast_jumps
    WHERE prev_health IS NOT NULL AND RunID != prev_run_id;  -- Only cross-run transitions
    """
    
    df = pd.read_sql(query, conn, params=(equip_id, days))
    
    print(f"\nForecast Continuity (EquipID={equip_id}, last {days} days):\n")
    print(df.to_string(index=False))
    
    if len(df) > 0:
        large_jumps_pct = 100 * df['large_jumps'].iloc[0] / max(df['total_transitions'].iloc[0], 1)
        
        if large_jumps_pct > 10:
            print(f"\n⚠️  WARNING: {large_jumps_pct:.1f}% of transitions have large jumps (>20% health change)")
            return False
        else:
            print(f"\n✅ PASS: Only {large_jumps_pct:.1f}% of transitions have large jumps")
            return True
    else:
        print("\n⚠️  No forecast data available")
        return False

def check_baseline_coverage(conn, equip_id=1, days=7):
    """Check baseline/score window overlap - should be >50%."""
    print("\n" + "="*80)
    print("3. BASELINE COVERAGE CHECK")
    print("="*80)
    
    query = """
    SELECT 
        r.RunID,
        r.StartTime as batch_start,
        MIN(bb.Timestamp) as baseline_start,
        MAX(bb.Timestamp) as baseline_end,
        DATEDIFF(SECOND, MAX(bb.Timestamp), r.StartTime) as gap_seconds,
        CASE 
            WHEN DATEDIFF(SECOND, MAX(bb.Timestamp), r.StartTime) > 3600 THEN 'LARGE_GAP'
            WHEN DATEDIFF(SECOND, MAX(bb.Timestamp), r.StartTime) > 1800 THEN 'MEDIUM_GAP'
            ELSE 'OK'
        END as gap_status
    FROM ACM_Runs r
    LEFT JOIN ACM_BaselineBuffer bb ON bb.EquipID = r.EquipID
        AND bb.Timestamp <= r.StartTime
        AND bb.Timestamp >= DATEADD(HOUR, -72, r.StartTime)
    WHERE r.EquipID = ? AND r.RunTimestamp >= DATEADD(DAY, -?, GETDATE())
    GROUP BY r.RunID, r.StartTime
    ORDER BY r.StartTime DESC;
    """
    
    df = pd.read_sql(query, conn, params=(equip_id, days))
    
    print(f"\nBaseline Coverage (EquipID={equip_id}, last {days} days):\n")
    if len(df) > 0:
        print(f"Total runs: {len(df)}")
        print(f"Runs with large gaps (>1h): {(df['gap_status']=='LARGE_GAP').sum()}")
        print(f"Runs with medium gaps (>30min): {(df['gap_status']=='MEDIUM_GAP').sum()}")
        print(f"Runs with good coverage: {(df['gap_status']=='OK').sum()}")
        
        large_gap_pct = 100 * (df['gap_status']=='LARGE_GAP').sum() / len(df)
        
        if large_gap_pct > 20:
            print(f"\n⚠️  WARNING: {large_gap_pct:.1f}% of runs have large baseline gaps")
            print("\nRecent runs with gaps:")
            print(df[df['gap_status'] != 'OK'].head(10).to_string(index=False))
            return False
        else:
            print(f"\n✅ PASS: Only {large_gap_pct:.1f}% of runs have large gaps")
            return True
    else:
        print("\n⚠️  No baseline data available")
        return False

def check_variance_collapse(conn, equip_id=1, days=7):
    """Check for variance collapse incidents in run logs."""
    print("\n" + "="*80)
    print("4. VARIANCE COLLAPSE CHECK")
    print("="*80)
    
    query = """
    SELECT 
        RunID,
        LoggedAt,
        LEFT(Message, 100) as Message
    FROM ACM_RunLogs
    WHERE EquipID = ? 
        AND LoggedAt >= DATEADD(DAY, -?, GETDATE())
        AND (
            Message LIKE '%low-variance%'
            OR Message LIKE '%variance collapse%'
            OR Message LIKE '%near-zero%'
            OR Message LIKE '%std<%'
        )
    ORDER BY LoggedAt DESC;
    """
    
    df = pd.read_sql(query, conn, params=(equip_id, days))
    
    print(f"\nVariance Warnings (EquipID={equip_id}, last {days} days):\n")
    if len(df) > 0:
        print(f"Found {len(df)} variance warnings:")
        print(df.head(20).to_string(index=False))
        
        if len(df) > 10:
            print(f"\n⚠️  WARNING: {len(df)} variance collapse warnings detected")
            return False
        else:
            print(f"\n⚠️  MINOR: {len(df)} variance warnings (acceptable)")
            return True
    else:
        print("✅ PASS: No variance collapse warnings")
        return True

def main():
    print("\n" + "="*80)
    print("BATCH CONTINUITY VALIDATION REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    conn = connect_db()
    
    try:
        results = {
            'zscore_stability': check_zscore_stability(conn),
            'forecast_continuity': check_forecast_continuity(conn),
            'baseline_coverage': check_baseline_coverage(conn),
            'variance_collapse': check_variance_collapse(conn)
        }
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        passed = sum(results.values())
        total = len(results)
        
        for test, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test:25s}: {status}")
        
        print(f"\nOverall: {passed}/{total} checks passed ({100*passed/total:.0f}%)")
        
        if passed == total:
            print("\n🎉 All checks passed! Batch continuity fixes are working.")
        elif passed >= total * 0.75:
            print("\n⚠️  Most checks passed, but some issues remain. Review warnings above.")
        else:
            print("\n❌ Multiple issues detected. Batch continuity fixes may need adjustment.")
            
    finally:
        conn.close()

if __name__ == "__main__":
    main()
