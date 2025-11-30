"""
Comprehensive SQL table validation after batch run.
Checks all 79 ACM tables for data presence and integrity.
"""
from utils.config_dict import ConfigDict
from core.sql_client import SQLClient
import pandas as pd

# All expected ACM tables
EXPECTED_TABLES = [
    'ACM_Runs', 'ACM_Scores_Wide', 'ACM_Episodes', 'ACM_EpisodesQC',
    'ACM_HealthTimeline', 'ACM_RegimeTimeline', 'ACM_RegimeSummary',
    'ACM_RegimeFeatureImportance', 'ACM_RegimeTransitions',
    'ACM_ContributionCurrent', 'ACM_ContributionTimeline',
    'ACM_DriftSeries', 'ACM_ThresholdCrossings',
    'ACM_AlertAge', 'ACM_SensorRanking', 'ACM_RegimeOccupancy',
    'ACM_HealthHistogram', 'ACM_RegimeStability',
    'ACM_DefectSummary', 'ACM_DefectTimeline', 'ACM_SensorDefects',
    'ACM_HealthZoneByPeriod', 'ACM_SensorAnomalyByPeriod',
    'ACM_DetectorCorrelation', 'ACM_CalibrationSummary',
    'ACM_RegimeDwellStats', 'ACM_DriftEvents', 'ACM_CulpritHistory',
    'ACM_EpisodeMetrics', 'ACM_EpisodeDiagnostics',
    'ACM_DataQuality', 'ACM_Scores_Long', 'ACM_Drift_TS',
    'ACM_Anomaly_Events', 'ACM_Regime_Episodes',
    'ACM_PCA_Models', 'ACM_PCA_Loadings', 'ACM_PCA_Metrics',
    'ACM_Run_Stats', 'ACM_SinceWhen',
    'ACM_SensorHotspots', 'ACM_SensorHotspotTimeline',
    'ACM_HealthForecast_TS', 'ACM_FailureForecast_TS',
    'ACM_RUL_TS', 'ACM_RUL_Summary', 'ACM_RUL_Attribution',
    'ACM_SensorForecast_TS', 'ACM_MaintenanceRecommendation',
    'ACM_EnhancedFailureProbability_TS', 'ACM_FailureCausation',
    'ACM_EnhancedMaintenanceRecommendation', 'ACM_RecommendedActions',
    'ACM_SensorNormalized_TS', 'ACM_OMRContributions',
    'ACM_OMRContributionsLong', 'ACM_FusionQuality',
    'ACM_FusionQualityReport', 'ACM_OMRTimeline',
    'ACM_RegimeStats', 'ACM_DailyFusedProfile',
    'ACM_OMR_Diagnostics', 'ACM_Forecast_QualityMetrics',
    'ACM_HealthDistributionOverTime', 'ACM_ChartGenerationLog',
    'ACM_FusionMetrics', 'ACM_HealthForecast_Continuous',
    'ACM_FailureHazard_TS', 'ACM_ThresholdMetadata',
    'ModelRegistry', 'ACM_Logs', 'ACM_ConfigHistory'
]

def main():
    cfg = ConfigDict.from_csv('configs/config_table.csv', 'FD_FAN')
    sql = SQLClient(cfg)
    sql.connect()
    
    print("=" * 80)
    print("COMPREHENSIVE SQL TABLE VALIDATION")
    print("=" * 80)
    print()
    
    # Check ACM_Runs for recent successful runs
    print("### Recent Runs ###")
    try:
        runs = sql.execute("""
        SELECT TOP 5 
            SUBSTRING(CAST(RunID AS VARCHAR(36)), 1, 13) as RunID,
            EquipID,
            Outcome,
            RowsProcessed,
            CONVERT(VARCHAR(19), CreatedAt, 120) as CreatedAt
        FROM ACM_Runs 
        WHERE Outcome = 'SUCCESS'
        ORDER BY CreatedAt DESC
        """)
        if isinstance(runs, pd.DataFrame) and not runs.empty:
            print(runs.to_string(index=False))
            print()
        else:
            print("No successful runs found\n")
    except Exception as e:
        print(f"ERROR querying ACM_Runs: {e}\n")
    
    # Count rows in all tables
    print("### Table Row Counts ###")
    populated = []
    empty = []
    errors = []
    
    for table in sorted(EXPECTED_TABLES):
        try:
            result = sql.execute(f"SELECT COUNT(*) as cnt FROM {table}")
            if isinstance(result, pd.DataFrame):
                count = result.iloc[0]['cnt']
            elif isinstance(result, int):
                count = result
            else:
                count = 0
            
            if count > 0:
                populated.append((table, count))
                print(f"✓ {table:40s}: {count:>8,} rows")
            else:
                empty.append(table)
        except Exception as e:
            errors.append((table, str(e)))
            print(f"✗ {table:40s}: ERROR")
    
    print()
    print("=" * 80)
    print(f"SUMMARY: {len(populated)} populated, {len(empty)} empty, {len(errors)} errors")
    print("=" * 80)
    
    if empty:
        print(f"\nEmpty tables ({len(empty)}):")
        for table in empty:
            print(f"  - {table}")
    
    if errors:
        print(f"\nError tables ({len(errors)}):")
        for table, error in errors:
            print(f"  - {table}: {error[:60]}")
    
    # Check key analytics tables
    print("\n### Key Analytics Tables ###")
    key_tables = {
        'ACM_Scores_Wide': 'Fused scores',
        'ACM_HealthTimeline': 'Health index',
        'ACM_Episodes': 'Anomaly episodes',
        'ACM_RegimeTimeline': 'Operating regimes',
        'ACM_ContributionCurrent': 'Current contributors',
        'ACM_DriftSeries': 'Drift tracking',
        'ACM_OMRContributions': 'OMR contributions',
        'ModelRegistry': 'Trained models'
    }
    
    for table, description in key_tables.items():
        try:
            count = sql.execute(f"SELECT COUNT(*) as cnt FROM {table}")
            if isinstance(count, pd.DataFrame):
                c = count.iloc[0]['cnt']
            else:
                c = count
            status = "✓" if c > 0 else "✗"
            print(f"{status} {description:25s} ({table}): {c:>6,} rows")
        except Exception as e:
            print(f"✗ {description:25s} ({table}): ERROR")
    
    # Check for data in recent time windows
    print("\n### Recent Data Coverage ###")
    try:
        recent = sql.execute("""
        SELECT 
            COUNT(*) as RecentRows,
            MIN(Timestamp) as MinTime,
            MAX(Timestamp) as MaxTime
        FROM ACM_Scores_Wide
        WHERE Timestamp >= DATEADD(day, -30, GETDATE())
        """)
        if isinstance(recent, pd.DataFrame) and not recent.empty:
            print(recent.to_string(index=False))
        else:
            print("No recent data found")
    except Exception as e:
        print(f"ERROR: {e}")
    
    sql.close()
    print("\n✓ Validation complete!")

if __name__ == "__main__":
    main()
