#!/usr/bin/env python3
"""
Analyze False Positives and False Negatives from ACM runs.

Compares ACM predictions against known event windows from ACM_RunLog events
and event_info.csv fault information.

Metrics computed:
- True Positives (TP): Anomaly detection during known fault periods
- True Negatives (TN): Normal operation (no alert) during normal periods
- False Positives (FP): Alerts during known normal/MAINTENANCE periods
- False Negatives (FN): Missing detection during fault periods
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
"""

import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Add ACM core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.sql_client import SQLClient
from core.observability import Console

def load_fault_windows(sql: SQLClient) -> pd.DataFrame:
    """Load known fault windows from database.
    
    Known faults are marked by:
    1. High anomaly detection during specific period windows
    2. Regime transitions and transient states
    3. Manual fault events in event_info table
    
    Returns:
        DataFrame with columns: EquipID, StartTime, EndTime, FaultType, Severity
    """
    query = """
    SELECT DISTINCT
        e.EquipID,
        MIN(rt.Timestamp) AS StartTime,
        MAX(rt.Timestamp) AS EndTime,
        'ANOMALY_EPISODE' AS FaultType,
        ROUND(AVG(CAST(rt.FusedAnomaly AS FLOAT)), 2) AS AvgAnomalyScore
    FROM ACM_RegimeTimeline rt
    JOIN Equipment e ON e.EquipID = rt.EquipID
    WHERE rt.FusedAnomaly = 1
      AND rt.TransientState IN ('STARTUP', 'TRIP', 'SHUTDOWN', 'TRANSIENT')
    GROUP BY e.EquipID, 
             DATEPART(YEAR, rt.Timestamp),
             DATEPART(MONTH, rt.Timestamp),
             DATEPART(DAY, rt.Timestamp)
    UNION ALL
    SELECT
        EquipID,
        EventStartTime AS StartTime,
        EventEndTime AS EndTime,
        EventType AS FaultType,
        Severity AS AvgAnomalyScore
    FROM event_info
    WHERE EventType IN ('FAULT', 'ANOMALY', 'DEGRADATION')
    """
    
    df = sql.query(query)
    if df.empty:
        Console.warn("No fault windows found in database", component="ANALYSIS")
        return pd.DataFrame()
    
    df['StartTime'] = pd.to_datetime(df['StartTime'])
    df['EndTime'] = pd.to_datetime(df['EndTime'])
    return df.sort_values(['EquipID', 'StartTime'])

def load_normal_windows(sql: SQLClient, equip_ids: List[int]) -> pd.DataFrame:
    """Load known normal operating windows.
    
    Normal periods are inferred from:
    1. Periods with NO anomaly detection
    2. Stable regime labels (no UNKNOWN)
    3. Health > 80%
    """
    if not equip_ids:
        return pd.DataFrame()
    
    equip_str = ','.join(str(e) for e in equip_ids)
    query = f"""
    SELECT
        EquipID,
        DatetimeFrom AS StartTime,
        DatetimeTo AS EndTime,
        'NORMAL' AS FaultType,
        0.0 AS AvgAnomalyScore
    FROM (
        SELECT
            EquipID,
            Timestamp AS DatetimeFrom,
            LEAD(Timestamp) OVER (PARTITION BY EquipID ORDER BY Timestamp) AS DatetimeTo,
            FusedAnomaly,
            RegimeLabel,
            HealthIndex
        FROM ACM_RegimeTimeline
        WHERE EquipID IN ({equip_str})
    ) t
    WHERE FusedAnomaly = 0
      AND RegimeLabel >= 0  -- Not UNKNOWN
      AND HealthIndex > 80
      AND DatetimeTo IS NOT NULL
      AND DATEDIFF(MINUTE, DatetimeFrom, DatetimeTo) >= 60
    GROUP BY EquipID, DatetimeFrom, DatetimeTo
    """
    
    try:
        df = sql.query(query)
        if not df.empty:
            df['StartTime'] = pd.to_datetime(df['StartTime'])
            df['EndTime'] = pd.to_datetime(df['EndTime'])
        return df
    except Exception as e:
        Console.warn(f"Could not load normal windows: {e}", component="ANALYSIS")
        return pd.DataFrame()

def load_anomaly_detections(sql: SQLClient, equip_ids: List[int]) -> pd.DataFrame:
    """Load all ACM anomaly detections and episodes."""
    if not equip_ids:
        return pd.DataFrame()
    
    equip_str = ','.join(str(e) for e in equip_ids)
    query = f"""
    SELECT
        EquipID,
        StartTime,
        EndTime,
        FusedZ,
        TopDefects,
        EpisodeLength,
        EpisodeNum
    FROM ACM_Anomaly_Events
    WHERE EquipID IN ({equip_str})
    ORDER BY EquipID, StartTime
    """
    
    df = sql.query(query)
    if not df.empty:
        df['StartTime'] = pd.to_datetime(df['StartTime'])
        df['EndTime'] = pd.to_datetime(df['EndTime'])
    return df

def load_recent_runs(sql: SQLClient, equip_ids: List[int], limit_hours: int = 24) -> pd.DataFrame:
    """Load recent ACM run results for analysis."""
    if not equip_ids:
        return pd.DataFrame()
    
    equip_str = ','.join(str(e) for e in equip_ids)
    query = f"""
    SELECT TOP 100
        r.RunID,
        r.EquipID,
        e.EquipCode,
        r.StartedAt,
        r.CompletedAt,
        r.PipelineMode,
        r.Status,
        r.RunHash,
        (SELECT COUNT(*) FROM ACM_RegimeTimeline rt WHERE rt.RunID = r.RunID) AS RegimeRowCount,
        (SELECT COUNT(*) FROM ACM_Anomaly_Events ae WHERE ae.RunID = r.RunID) AS AnomalyRowCount,
        (SELECT COUNT(*) FROM ACM_HealthTimeline ht WHERE ht.RunID = r.RunID) AS HealthRowCount
    FROM ACM_Runs r
    JOIN Equipment e ON e.EquipID = r.EquipID
    WHERE r.EquipID IN ({equip_str})
      AND r.CompletedAt >= DATEADD(HOUR, -{limit_hours}, GETUTC())
    ORDER BY r.CompletedAt DESC
    """
    
    df = sql.query(query)
    if not df.empty:
        df['StartedAt'] = pd.to_datetime(df['StartedAt'])
        df['CompletedAt'] = pd.to_datetime(df['CompletedAt'])
    return df

def analyze_fp_fn(
    detections: pd.DataFrame,
    fault_windows: pd.DataFrame,
    normal_windows: pd.DataFrame
) -> Dict[int, Dict]:
    """Analyze False Positives and False Negatives.
    
    Args:
        detections: ACM_Anomaly_Events
        fault_windows: Known fault periods
        normal_windows: Known normal periods
        
    Returns:
        Dict mapping EquipID -> metrics dict
    """
    results = {}
    
    for equip_id in detections['EquipID'].unique():
        equip_detections = detections[detections['EquipID'] == equip_id]
        equip_faults = fault_windows[fault_windows['EquipID'] == equip_id]
        equip_normal = normal_windows[normal_windows['EquipID'] == equip_id]
        
        if equip_detections.empty:
            continue
        
        # Classification
        tp = 0  # Alert during fault
        fp = 0  # Alert during normal
        fn = 0  # No alert during fault
        tn = 0  # No alert during normal
        
        # Check each detection against fault windows
        for _, det in equip_detections.iterrows():
            det_start = pd.Timestamp(det['StartTime'])
            det_end = pd.Timestamp(det['EndTime'])
            
            # Check if overlaps with fault window
            in_fault = False
            for _, fault in equip_faults.iterrows():
                fault_start = pd.Timestamp(fault['StartTime'])
                fault_end = pd.Timestamp(fault['EndTime'])
                
                if det_start <= fault_end and det_end >= fault_start:
                    in_fault = True
                    break
            
            if in_fault:
                tp += 1
            else:
                fp += 1
        
        # Check for missed detections (faults without alerts)
        for _, fault in equip_faults.iterrows():
            fault_start = pd.Timestamp(fault['StartTime'])
            fault_end = pd.Timestamp(fault['EndTime'])
            
            missed = True
            for _, det in equip_detections.iterrows():
                det_start = pd.Timestamp(det['StartTime'])
                det_end = pd.Timestamp(det['EndTime'])
                
                if det_start <= fault_end and det_end >= fault_start:
                    missed = False
                    break
            
            if missed:
                fn += 1
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[equip_id] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3),
            'detection_count': len(equip_detections),
            'fault_count': len(equip_faults),
        }
    
    return results

def main():
    try:
        sql = SQLClient()
        
        Console.header("FALSE POSITIVE / NEGATIVE ANALYSIS", char="=")
        
        # Get equipment IDs
        equip_query = "SELECT EquipID, EquipCode FROM Equipment ORDER BY EquipCode"
        equip_df = sql.query(equip_query)
        equip_ids = equip_df['EquipID'].tolist()
        
        if not equip_ids:
            Console.error("No equipment found", component="ANALYSIS")
            return
        
        Console.info(f"Analyzing {len(equip_ids)} equipment: {', '.join(equip_df['EquipCode'].tolist())}", component="ANALYSIS")
        
        # Load data
        Console.section("Loading data...")
        detections = load_anomaly_detections(sql, equip_ids)
        fault_windows = load_fault_windows(sql)
        normal_windows = load_normal_windows(sql, equip_ids)
        runs = load_recent_runs(sql, equip_ids)
        
        Console.info(f"Loaded {len(detections)} anomaly detections", component="ANALYSIS")
        Console.info(f"Loaded {len(fault_windows)} fault windows", component="ANALYSIS")
        Console.info(f"Loaded {len(normal_windows)} normal windows", component="ANALYSIS")
        Console.info(f"Loaded {len(runs)} recent runs", component="ANALYSIS")
        
        if detections.empty:
            Console.warn("No detections found - skipping analysis", component="ANALYSIS")
            return
        
        # Analyze
        Console.section("Computing metrics...")
        metrics = analyze_fp_fn(detections, fault_windows, normal_windows)
        
        # Print results
        Console.section("RESULTS")
        
        summary_data = []
        for equip_id, m in sorted(metrics.items()):
            equip_code = equip_df[equip_df['EquipID'] == equip_id]['EquipCode'].values
            equip_code = equip_code[0] if len(equip_code) > 0 else f"ID_{equip_id}"
            
            print(f"\n{equip_code} (ID={equip_id}):")
            print(f"  TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}")
            print(f"  Precision: {m['precision']:.1%}")
            print(f"  Recall: {m['recall']:.1%}")
            print(f"  F1 Score: {m['f1_score']:.3f}")
            print(f"  Detections: {m['detection_count']}/{m['fault_count']} faults")
            
            summary_data.append({
                'Equipment': equip_code,
                'TP': m['tp'],
                'FP': m['fp'],
                'FN': m['fn'],
                'Precision': f"{m['precision']:.1%}",
                'Recall': f"{m['recall']:.1%}",
                'F1': f"{m['f1_score']:.3f}",
            })
        
        # Summary table
        Console.section("SUMMARY TABLE")
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save results
        output_file = f"artifacts/fp_fn_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'summary': summary_data,
            }, f, indent=2)
        
        Console.ok(f"Analysis saved to {output_file}", component="ANALYSIS")
        
    except Exception as e:
        Console.error(f"Analysis failed: {e}", component="ANALYSIS")
        raise

if __name__ == '__main__':
    main()
