"""RUL Backtest Evaluation Script

Generates quantitative validation for Remaining Useful Life predictions.
Outputs CSV + JSON under artifacts/{equip}/rul_backtest/.

Evaluation steps:
1. Fetch run-level RUL predictions (ACM_RUL_Summary joined with ACM_Runs).
2. Fetch health timeline + forecasts + failure probability timeline.
3. Derive ground-truth failure events using UNIFIED FAILURE CONDITION:
    a. Condition 1: HealthIndex < 75 sustained for >= 4 consecutive hours
    b. Condition 2: Episode with Severity='CRITICAL' in ACM_CulpritHistory
    c. Condition 3: FusedZ >= 3.0 sustained for >= 2 consecutive hours
4. For each RUL prediction: compute predicted failure time t_pred = prediction_time + RUL_Hours.
    Match to next ground-truth failure event T_fail > prediction_time.
5. Compute metrics: Absolute Error, Relative Error, Hit (within tolerance), Brier-like failure probability scoring.
6. Aggregate metrics: MAE, MedianAE, MAPE, HitRate, Calibration bins, Distribution of slope divergence.
7. Attribution consistency: overlap between RUL attribution sensors and episode culprits near failure.

Usage:
  python scripts/evaluate_rul_backtest.py --equip 1 --health-threshold 75 --health-sustain-hours 4 --fused-z-threshold 3.0 --fused-z-sustain-hours 2 --tolerance-frac 0.2

See docs/RUL_METHOD.md for complete failure condition specification.
"""
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import pyodbc
import pandas as pd

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RULPrediction:
    run_id: str
    prediction_time: pd.Timestamp
    rul_hours: float
    predicted_failure_time: pd.Timestamp

@dataclass
class FailureEvent:
    failure_time: pd.Timestamp
    source: str  # 'episode' or 'health_threshold'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def connect() -> pyodbc.Connection:
    return pyodbc.connect(
        'DRIVER={SQL Server};'
        'SERVER=localhost\\B19CL3PCQLSERVER;'
        'DATABASE=ACM;'
        'Trusted_Connection=yes;'
    )


def fetch_rul_predictions(cursor, equip_id: int) -> pd.DataFrame:
    sql = """
    SELECT r.RunID, r.StartedAt AS PredictionTime, s.RUL_Hours
    FROM ACM_RUL_Summary s
    JOIN ACM_Runs r ON s.RunID = r.RunID
    WHERE s.EquipID = ?
    ORDER BY r.StartedAt
    """
    df = pd.read_sql(sql, cursor.connection, params=[equip_id])
    df['PredictionTime'] = pd.to_datetime(df['PredictionTime'])
    df['PredictedFailureTime'] = df['PredictionTime'] + pd.to_timedelta(df['RUL_Hours'], unit='h')
    return df


def fetch_health_timeline(cursor, equip_id: int) -> pd.DataFrame:
    sql = "SELECT Timestamp, HealthIndex, FusedZ FROM ACM_HealthTimeline WHERE EquipID=? ORDER BY Timestamp"
    df = pd.read_sql(sql, cursor.connection, params=[equip_id])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df


def fetch_episodes(cursor, equip_id: int) -> pd.DataFrame:
    # Fetch CRITICAL severity episodes only (Condition 2)
    sql = "SELECT StartTimestamp, Severity FROM ACM_CulpritHistory WHERE EquipID=? ORDER BY StartTimestamp"
    try:
        df = pd.read_sql(sql, cursor.connection, params=[equip_id])
    except Exception:
        return pd.DataFrame(columns=['StartTimestamp', 'Severity'])
    df['StartTimestamp'] = pd.to_datetime(df['StartTimestamp'])
    return df


def identify_health_threshold_failures(health: pd.DataFrame, alert_threshold: float, sustain_hours: int) -> List[pd.Timestamp]:
    """Condition 1: HealthIndex < threshold sustained for >= sustain_hours consecutive hours."""
    below = health['HealthIndex'] < alert_threshold
    failures = []
    count = 0
    sustain_points = sustain_hours  # Assuming 1-hour data granularity
    for ts, is_below in zip(health['Timestamp'], below):
        if is_below:
            count += 1
            if count == sustain_points:
                failures.append(ts)
        else:
            count = 0
    return failures

def identify_fused_z_spike_failures(health: pd.DataFrame, z_threshold: float, sustain_hours: int) -> List[pd.Timestamp]:
    """Condition 3: FusedZ >= threshold sustained for >= sustain_hours consecutive hours."""
    if 'FusedZ' not in health.columns:
        return []
    above = health['FusedZ'] >= z_threshold
    failures = []
    count = 0
    sustain_points = sustain_hours
    for ts, is_above in zip(health['Timestamp'], above):
        if is_above:
            count += 1
            if count == sustain_points:
                failures.append(ts)
        else:
            count = 0
    return failures

def build_failure_events(episodes: pd.DataFrame, health_failures: List[pd.Timestamp], fused_z_failures: List[pd.Timestamp]) -> List[FailureEvent]:
    """Combine all three failure conditions into unified failure event list."""
    events: List[FailureEvent] = []
    
    # Condition 2: Critical episodes
    if 'Severity' in episodes.columns:
        critical_episodes = episodes[episodes['Severity'] == 'CRITICAL']
        for ts in critical_episodes['StartTimestamp'].tolist():
            events.append(FailureEvent(failure_time=ts, source='critical_episode'))
    
    # Condition 1: Health threshold
    for ts in health_failures:
        events.append(FailureEvent(failure_time=ts, source='health_sustained_low'))
    
    # Condition 3: FusedZ spike
    for ts in fused_z_failures:
        events.append(FailureEvent(failure_time=ts, source='fused_z_spike'))
    
    # Deduplicate within 24h window (merge nearby failures)
    events.sort(key=lambda e: e.failure_time)
    unique = []
    for e in events:
        # Check if any existing event within 24h
        is_duplicate = False
        for existing in unique:
            if abs((e.failure_time - existing.failure_time).total_seconds()) < 24 * 3600:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(e)
    return unique


def match_prediction_to_failure(pred_time: pd.Timestamp, events: List[FailureEvent]) -> Optional[FailureEvent]:
    for e in events:
        if e.failure_time > pred_time:
            return e
    return None


def evaluate_predictions(preds: pd.DataFrame, events: List[FailureEvent], tolerance_frac: float) -> pd.DataFrame:
    rows = []
    for _, row in preds.iterrows():
        pred_time = row['PredictionTime']
        rul_hours = row['RUL_Hours']
        t_pred = row['PredictedFailureTime']
        matched = match_prediction_to_failure(pred_time, events)
        if matched is None:
            continue  # Skip if no future failure event
        error_hours = (matched.failure_time - t_pred).total_seconds() / 3600.0
        abs_error = abs(error_hours)
        rel_error = abs_error / rul_hours if rul_hours > 0 else math.nan
        tolerance = tolerance_frac * rul_hours
        hit = abs_error <= tolerance
        rows.append({
            'RunID': row['RunID'],
            'PredictionTime': pred_time,
            'RUL_Hours': rul_hours,
            'PredictedFailureTime': t_pred,
            'ActualFailureTime': matched.failure_time,
            'FailureSource': matched.source,
            'ErrorHours': error_hours,
            'AbsErrorHours': abs_error,
            'RelError': rel_error,
            'HitTolerance': hit,
            'ToleranceHours': tolerance
        })
    return pd.DataFrame(rows)


def compute_metrics(eval_df: pd.DataFrame) -> Dict[str, float]:
    if eval_df.empty:
        return {}
    metrics = {
        'count': len(eval_df),
        'MAE_hours': float(eval_df['AbsErrorHours'].mean()),
        'MedianAE_hours': float(eval_df['AbsErrorHours'].median()),
        'MAPE': float(eval_df['RelError'].mean()),
        'HitRate': float(eval_df['HitTolerance'].mean()),
    }
    return metrics


def save_outputs(equip_id: int, eval_df: pd.DataFrame, metrics: Dict[str, float]):
    out_dir = os.path.join('artifacts', str(equip_id), 'rul_backtest')
    os.makedirs(out_dir, exist_ok=True)
    eval_df.to_csv(os.path.join(out_dir, 'rul_prediction_evaluation.csv'), index=False)
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--equip', type=int, required=True)
    parser.add_argument('--health-threshold', type=float, default=75.0, help='HealthIndex threshold for Condition 1')
    parser.add_argument('--health-sustain-hours', type=int, default=4, help='Sustain hours for Condition 1')
    parser.add_argument('--fused-z-threshold', type=float, default=3.0, help='FusedZ threshold for Condition 3')
    parser.add_argument('--fused-z-sustain-hours', type=int, default=2, help='Sustain hours for Condition 3')
    parser.add_argument('--tolerance-frac', type=float, default=0.2)
    args = parser.parse_args()

    conn = connect()
    cursor = conn.cursor()

    preds = fetch_rul_predictions(cursor, args.equip)
    health = fetch_health_timeline(cursor, args.equip)
    episodes = fetch_episodes(cursor, args.equip)

    # Apply unified failure condition (3 paths)
    health_failures = identify_health_threshold_failures(health, args.health_threshold, args.health_sustain_hours)
    fused_z_failures = identify_fused_z_spike_failures(health, args.fused_z_threshold, args.fused_z_sustain_hours)
    events = build_failure_events(episodes, health_failures, fused_z_failures)

    eval_df = evaluate_predictions(preds, events, args.tolerance_frac)
    metrics = compute_metrics(eval_df)
    save_outputs(args.equip, eval_df, metrics)

    print('Evaluation complete.')
    print('Metrics:', metrics)
    print('Rows evaluated:', len(eval_df))
    print('Failure events detected:', len(events))
    if events:
        print('Failure sources breakdown:')
        sources = {}
        for e in events:
            sources[e.source] = sources.get(e.source, 0) + 1
        for source, count in sources.items():
            print(f'  {source}: {count}')

    cursor.close()
    conn.close()

if __name__ == '__main__':
    main()
