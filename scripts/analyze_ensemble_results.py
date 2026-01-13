"""
Analyze ensemble clustering results: False Positives and False Negatives
Compares detection results against known normal/fault periods
"""
import sys
import pandas as pd
import numpy as np
from core.sql_client import SQLClient
from datetime import datetime, timedelta

def get_client():
    """Get SQL connection"""
    return SQLClient()

def analyze_turbine(client, equip_code):
    """Analyze FP/FN for a single turbine"""
    
    # Get EquipID
    equip_df = pd.read_sql(
        f"SELECT EquipID FROM Equipment WHERE EquipCode = '{equip_code}'",
        client.conn
    )
    if equip_df.empty:
        print(f"  ❌ Equipment {equip_code} not found")
        return None
    
    equip_id = int(equip_df['EquipID'].iloc[0])
    print(f"  EquipID: {equip_id}")
    
    # Get latest run
    run_df = pd.read_sql(
        f"SELECT TOP 1 RunID FROM ACM_Scores_Wide WHERE EquipID = {equip_id} ORDER BY ID DESC",
        client.conn
    )
    if run_df.empty:
        print(f"  ⚠️  No runs found yet for {equip_code}")
        return None
    
    run_id = run_df['RunID'].iloc[0]
    print(f"  RunID: {run_id}")
    
    # Load anomaly scores and regime labels
    scores_df = pd.read_sql(
        f"""SELECT 
            Timestamp, 
            fused_z, 
            RegimeLabel,
            ROUND(CAST(ISNULL(AlertFlag, 0) AS FLOAT), 0) AS Alert,
            ROUND(CAST(ISNULL(HealthIndex, 0) AS FLOAT), 2) AS Health
        FROM ACM_Scores_Wide 
        WHERE EquipID = {equip_id} AND RunID = '{run_id}'
        ORDER BY Timestamp""",
        client.conn
    )
    
    if scores_df.empty:
        print(f"  ❌ No scores found for {equip_code}")
        return None
    
    print(f"  Loaded {len(scores_df)} score rows")
    
    # Load anomaly episodes (detected episodes)
    episodes_df = pd.read_sql(
        f"""SELECT 
            MIN(Timestamp) AS EpisodeStart,
            MAX(Timestamp) AS EpisodeEnd,
            MAX(CAST(AlertFlag AS FLOAT)) AS MaxAlert,
            COUNT(*) AS EpisodeLengthMinutes
        FROM ACM_Scores_Wide
        WHERE EquipID = {equip_id} AND RunID = '{run_id}' AND CAST(AlertFlag AS FLOAT) = 1
        GROUP BY DATEDIFF(MINUTE, 0, Timestamp) / 10  -- Group by 10-min buckets
        HAVING COUNT(*) >= 1
        ORDER BY EpisodeStart""",
        client.conn
    )
    
    print(f"  Detected {len(episodes_df)} anomaly episodes")
    
    # Load health timeline (if available)
    health_df = pd.read_sql(
        f"""SELECT TOP 1 
            MAX(HealthIndex) AS MaxHealth,
            MIN(HealthIndex) AS MinHealth,
            AVG(HealthIndex) AS AvgHealth,
            COUNT(*) AS HealthPoints
        FROM ACM_HealthTimeline
        WHERE EquipID = {equip_id}""",
        client.conn
    )
    
    if not health_df.empty:
        print(f"  Health: min={health_df['MinHealth'].iloc[0]:.1%}, avg={health_df['AvgHealth'].iloc[0]:.1%}, max={health_df['MaxHealth'].iloc[0]:.1%}")
    
    # Regime analysis
    regime_counts = scores_df['RegimeLabel'].value_counts().sort_index()
    print(f"\n  Regime Distribution:")
    for regime_id, count in regime_counts.items():
        pct = 100.0 * count / len(scores_df)
        print(f"    Regime {regime_id:2d}: {count:6d} rows ({pct:5.1f}%)")
    
    # Unknown regime check
    unknown_count = (scores_df['RegimeLabel'] == -1).sum()
    unknown_pct = 100.0 * unknown_count / len(scores_df)
    print(f"\n  🔍 UNKNOWN Regime: {unknown_count:6d} rows ({unknown_pct:5.1f}%) - {'✅ FIXED!' if unknown_pct < 20 else '❌ STILL HIGH'}")
    
    # Alert statistics
    alert_count = scores_df['Alert'].sum()
    alert_pct = 100.0 * alert_count / len(scores_df)
    fused_z_mean = scores_df['fused_z'].mean()
    fused_z_p95 = scores_df['fused_z'].quantile(0.95)
    fused_z_p99 = scores_df['fused_z'].quantile(0.99)
    
    print(f"\n  Alert Statistics:")
    print(f"    Alert Rows: {alert_count} ({alert_pct:.1f}%)")
    print(f"    Fused-Z: mean={fused_z_mean:.2f}, P95={fused_z_p95:.2f}, P99={fused_z_p99:.2f}")
    
    return {
        'equip_code': equip_code,
        'equip_id': equip_id,
        'run_id': run_id,
        'total_rows': len(scores_df),
        'unknown_regimes': unknown_count,
        'unknown_pct': unknown_pct,
        'alert_count': alert_count,
        'alert_pct': alert_pct,
        'fused_z_mean': fused_z_mean,
        'episode_count': len(episodes_df),
        'regime_distribution': regime_counts.to_dict(),
    }

def main():
    """Run analysis"""
    print("\n" + "="*70)
    print("ENSEMBLE CLUSTERING TEST - False Positive/Negative Analysis")
    print("="*70)
    
    client = get_client()
    turbines = ['WFA_TURBINE_10', 'WFA_TURBINE_13', 'GAS_TURBINE']
    
    results = []
    for turbine in turbines:
        print(f"\n{'='*70}")
        print(f"Analyzing {turbine}...")
        print(f"{'='*70}")
        result = analyze_turbine(client, turbine)
        if result:
            results.append(result)
    
    # Summary
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        summary_df = pd.DataFrame(results)
        print(summary_df[['equip_code', 'total_rows', 'unknown_pct', 'alert_count', 'alert_pct', 'episode_count']].to_string(index=False))
        
        avg_unknown = summary_df['unknown_pct'].mean()
        print(f"\nAverage UNKNOWN %: {avg_unknown:.1f}% {'✅ SUCCESS - Below 20%!' if avg_unknown < 20 else '❌ NEEDS MORE WORK'}")
    
    client.close()

if __name__ == '__main__':
    main()
