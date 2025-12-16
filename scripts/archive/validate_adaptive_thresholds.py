"""
Validate Adaptive Threshold Approach Against Historical Data

This script analyzes actual FusedZ distributions from ACM_HealthTimeline
to validate whether the hardcoded threshold of 3.0 is appropriate or if
data-driven thresholds would be more accurate.

Key Questions:
1. What is the actual P99.7 (3-sigma equivalent) of FusedZ per equipment?
2. How does it compare to the hardcoded 3.0?
3. Does it vary significantly across equipment/regimes?
4. What would be the impact of using adaptive thresholds?
"""

import pyodbc
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# SQL Server connection (adjust as needed)
SERVER = r"localhost\B19CL3PCQLSERVER"
DATABASE = "ACM"


def connect_sql():
    """Connect to SQL Server using Windows Authentication."""
    conn_str = f"DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes"
    return pyodbc.connect(conn_str)


def analyze_fused_distribution(conn, equip_id=None):
    """
    Analyze FusedZ distribution from ACM_HealthTimeline.
    
    Args:
        conn: SQL connection
        equip_id: Optional equipment ID filter (None = all equipment)
        
    Returns:
        DataFrame with distribution statistics per equipment
    """
    equip_filter = f"AND EquipID = {equip_id}" if equip_id else ""
    
    # Use subquery to calculate percentiles separately from aggregates
    query = f"""
    WITH Stats AS (
        SELECT 
            EquipID,
            COUNT(*) as SampleCount,
            AVG(FusedZ) as Mean_FusedZ,
            STDEV(FusedZ) as StdDev_FusedZ,
            MIN(FusedZ) as Min_FusedZ,
            MAX(FusedZ) as Max_FusedZ
        FROM ACM_HealthTimeline
        WHERE FusedZ IS NOT NULL 
          AND FusedZ BETWEEN 0 AND 50
          {equip_filter}
        GROUP BY EquipID
    ),
    Percentiles AS (
        SELECT DISTINCT
            EquipID,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY FusedZ) OVER (PARTITION BY EquipID) as P50_FusedZ,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY FusedZ) OVER (PARTITION BY EquipID) as P90_FusedZ,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY FusedZ) OVER (PARTITION BY EquipID) as P95_FusedZ,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY FusedZ) OVER (PARTITION BY EquipID) as P99_FusedZ,
            PERCENTILE_CONT(0.997) WITHIN GROUP (ORDER BY FusedZ) OVER (PARTITION BY EquipID) as P997_FusedZ
        FROM ACM_HealthTimeline
        WHERE FusedZ IS NOT NULL 
          AND FusedZ BETWEEN 0 AND 50
          {equip_filter}
    )
    SELECT 
        s.EquipID,
        s.SampleCount,
        s.Mean_FusedZ,
        s.StdDev_FusedZ,
        s.Min_FusedZ,
        s.Max_FusedZ,
        p.P50_FusedZ,
        p.P90_FusedZ,
        p.P95_FusedZ,
        p.P99_FusedZ,
        p.P997_FusedZ
    FROM Stats s
    INNER JOIN Percentiles p ON s.EquipID = p.EquipID
    """
    
    df = pd.read_sql(query, conn)
    
    # Deduplicate (PERCENTILE_CONT creates one row per input row)
    df = df.drop_duplicates(subset=['EquipID'])
    
    return df


def analyze_per_regime_distribution(conn, equip_id):
    """
    Analyze FusedZ distribution per regime for a specific equipment.
    
    Args:
        conn: SQL connection
        equip_id: Equipment ID
        
    Returns:
        DataFrame with distribution statistics per regime
    """
    # Use CTE to calculate stats and percentiles separately
    query = f"""
    WITH RegimeStats AS (
        SELECT 
            rt.EquipID,
            rt.RegimeID,
            COUNT(*) as SampleCount,
            AVG(ht.FusedZ) as Mean_FusedZ,
            STDEV(ht.FusedZ) as StdDev_FusedZ
        FROM ACM_RegimeTimeline rt
        INNER JOIN ACM_HealthTimeline ht 
            ON rt.EquipID = ht.EquipID 
            AND rt.Timestamp = ht.Timestamp
        WHERE rt.EquipID = {equip_id}
          AND ht.FusedZ IS NOT NULL
          AND ht.FusedZ BETWEEN 0 AND 50
          AND rt.RegimeID IS NOT NULL
        GROUP BY rt.EquipID, rt.RegimeID
    ),
    RegimePercentiles AS (
        SELECT DISTINCT
            rt.EquipID,
            rt.RegimeID,
            PERCENTILE_CONT(0.997) WITHIN GROUP (ORDER BY ht.FusedZ) OVER (PARTITION BY rt.RegimeID) as P997_FusedZ
        FROM ACM_RegimeTimeline rt
        INNER JOIN ACM_HealthTimeline ht 
            ON rt.EquipID = ht.EquipID 
            AND rt.Timestamp = ht.Timestamp
        WHERE rt.EquipID = {equip_id}
          AND ht.FusedZ IS NOT NULL
          AND ht.FusedZ BETWEEN 0 AND 50
          AND rt.RegimeID IS NOT NULL
    )
    SELECT 
        rs.EquipID,
        rs.RegimeID,
        rs.SampleCount,
        rs.Mean_FusedZ,
        rs.StdDev_FusedZ,
        rp.P997_FusedZ
    FROM RegimeStats rs
    INNER JOIN RegimePercentiles rp 
        ON rs.EquipID = rp.EquipID 
        AND rs.RegimeID = rp.RegimeID
    """
    
    df = pd.read_sql(query, conn)
    
    # Deduplicate
    df = df.drop_duplicates(subset=['EquipID', 'RegimeID'])
    
    return df


def calculate_threshold_impact(df):
    """
    Calculate impact of using adaptive vs hardcoded thresholds.
    
    Args:
        df: DataFrame with FusedZ distribution stats
        
    Returns:
        DataFrame with threshold comparison
    """
    df['Hardcoded_Threshold'] = 3.0
    df['Adaptive_Threshold'] = df['P997_FusedZ']
    df['Threshold_Delta'] = df['Adaptive_Threshold'] - df['Hardcoded_Threshold']
    df['Threshold_Delta_Pct'] = (df['Threshold_Delta'] / df['Hardcoded_Threshold']) * 100
    
    # Estimate false positive rate if using hardcoded 3.0
    # Assume samples above P99.7 are true positives (0.3%)
    # If adaptive threshold != 3.0, hardcoded 3.0 will have different FP rate
    df['Estimated_Alert_Rate_Hardcoded'] = (
        (df['Max_FusedZ'] > 3.0).astype(float) * 
        np.clip((3.0 - df['P997_FusedZ']) / (df['Max_FusedZ'] - df['P997_FusedZ'] + 1e-6), 0, 1)
    )
    
    # Classification
    df['Threshold_Assessment'] = pd.cut(
        df['Threshold_Delta_Pct'],
        bins=[-np.inf, -20, -5, 5, 20, np.inf],
        labels=['Too Sensitive', 'Slightly Sensitive', 'Appropriate', 'Slightly Lenient', 'Too Lenient']
    )
    
    return df


def main():
    """Run validation analysis."""
    print("=" * 80)
    print("ADAPTIVE THRESHOLD VALIDATION")
    print("=" * 80)
    print()
    
    # Connect to SQL Server
    print(f"Connecting to SQL Server: {SERVER}/{DATABASE}...")
    try:
        conn = connect_sql()
        print("Connected successfully.")
        print()
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Please ensure:")
        print("  1. SQL Server is running")
        print("  2. ACM database exists")
        print("  3. ACM_HealthTimeline table has data")
        return
    
    try:
        # Analysis 1: Global distribution per equipment
        print("[1/3] Analyzing FusedZ distribution per equipment...")
        global_df = analyze_fused_distribution(conn)
        
        if global_df.empty:
            print("No data found in ACM_HealthTimeline. Run acm_main.py first.")
            return
        
        print(f"Found {len(global_df)} equipment with FusedZ data")
        print()
        
        # Calculate impact
        global_df = calculate_threshold_impact(global_df)
        
        # Display results
        print("GLOBAL THRESHOLD ANALYSIS")
        print("-" * 80)
        print(global_df[[
            'EquipID', 'SampleCount', 'Mean_FusedZ', 'StdDev_FusedZ',
            'P997_FusedZ', 'Threshold_Delta', 'Threshold_Delta_Pct',
            'Threshold_Assessment'
        ]].to_string(index=False))
        print()
        
        # Analysis 2: Per-regime distribution for each equipment
        print("[2/3] Analyzing per-regime thresholds...")
        per_regime_results = []
        for equip_id in global_df['EquipID'].unique():
            try:
                regime_df = analyze_per_regime_distribution(conn, equip_id)
                if not regime_df.empty:
                    regime_df = calculate_threshold_impact(regime_df)
                    per_regime_results.append(regime_df)
            except Exception as e:
                print(f"Warning: Failed to analyze regimes for EquipID={equip_id}: {e}")
        
        if per_regime_results:
            regime_combined = pd.concat(per_regime_results, ignore_index=True)
            print("PER-REGIME THRESHOLD ANALYSIS")
            print("-" * 80)
            print(regime_combined[[
                'EquipID', 'RegimeID', 'SampleCount', 'Mean_FusedZ',
                'P997_FusedZ', 'Threshold_Delta', 'Threshold_Assessment'
            ]].to_string(index=False))
            print()
        else:
            print("No per-regime data found.")
            regime_combined = pd.DataFrame()
            print()
        
        # Analysis 3: Summary statistics
        print("[3/3] Summary Statistics")
        print("-" * 80)
        print(f"Total Equipment Analyzed: {len(global_df)}")
        print(f"Average P99.7 (3-sigma): {global_df['P997_FusedZ'].mean():.3f}")
        print(f"Hardcoded Threshold: 3.0")
        print(f"Average Delta: {global_df['Threshold_Delta'].mean():.3f} ({global_df['Threshold_Delta_Pct'].mean():.1f}%)")
        print()
        
        print("Threshold Assessment Distribution:")
        assessment_counts = global_df['Threshold_Assessment'].value_counts()
        for assessment, count in assessment_counts.items():
            pct = (count / len(global_df)) * 100
            print(f"  {assessment}: {count} ({pct:.1f}%)")
        print()
        
        # Key findings
        print("KEY FINDINGS")
        print("-" * 80)
        
        too_sensitive = global_df[global_df['Threshold_Assessment'].isin(['Too Sensitive', 'Slightly Sensitive'])]
        too_lenient = global_df[global_df['Threshold_Assessment'].isin(['Too Lenient', 'Slightly Lenient'])]
        
        if len(too_sensitive) > 0:
            print(f"⚠️  {len(too_sensitive)} equipment have thresholds that are TOO SENSITIVE:")
            print(f"    Hardcoded 3.0 triggers alerts more often than 99.7% confidence suggests.")
            print(f"    Equipment IDs: {too_sensitive['EquipID'].tolist()}")
            print()
        
        if len(too_lenient) > 0:
            print(f"⚠️  {len(too_lenient)} equipment have thresholds that are TOO LENIENT:")
            print(f"    Hardcoded 3.0 misses alerts that should trigger at 99.7% confidence.")
            print(f"    Equipment IDs: {too_lenient['EquipID'].tolist()}")
            print()
        
        appropriate = global_df[global_df['Threshold_Assessment'] == 'Appropriate']
        if len(appropriate) > 0:
            print(f"✅ {len(appropriate)} equipment have appropriate thresholds (within ±5%).")
            print()
        
        # Save results
        output_dir = Path("artifacts")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        global_path = output_dir / f"threshold_validation_global_{timestamp}.csv"
        global_df.to_csv(global_path, index=False)
        print(f"Saved global analysis: {global_path}")
        
        if not regime_combined.empty:
            regime_path = output_dir / f"threshold_validation_regime_{timestamp}.csv"
            regime_combined.to_csv(regime_path, index=False)
            print(f"Saved per-regime analysis: {regime_path}")
        
        print()
        print("=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
        print()
        print("CONCLUSION:")
        print(f"  - Average adaptive threshold would be {global_df['P997_FusedZ'].mean():.3f} vs hardcoded 3.0")
        print(f"  - This represents a {global_df['Threshold_Delta_Pct'].mean():.1f}% difference")
        print(f"  - {len(too_sensitive) + len(too_lenient)}/{len(global_df)} equipment would benefit from adaptive thresholds")
        print()
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
