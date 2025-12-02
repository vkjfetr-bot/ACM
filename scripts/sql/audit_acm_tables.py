"""
Audit script to verify all ACM_* tables from OutputManager.ALLOWED_TABLES exist in SQL.
"""
import pyodbc
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.sql_client import SQLClient

# Expected tables from OutputManager.ALLOWED_TABLES
EXPECTED_TABLES = {
    'ACM_Scores_Wide','ACM_Episodes','ACM_EpisodesQC',
    'ACM_HealthTimeline','ACM_RegimeTimeline',
    'ACM_RegimeSummary','ACM_RegimeFeatureImportance','ACM_RegimeTransitions',
    'ACM_ContributionCurrent','ACM_ContributionTimeline',
    'ACM_DriftSeries','ACM_ThresholdCrossings',
    'ACM_AlertAge','ACM_SensorRanking','ACM_RegimeOccupancy',
    'ACM_HealthHistogram','ACM_RegimeStability',
    'ACM_DefectSummary','ACM_DefectTimeline','ACM_SensorDefects',
    'ACM_HealthZoneByPeriod','ACM_SensorAnomalyByPeriod',
    'ACM_DetectorCorrelation','ACM_CalibrationSummary',
    'ACM_RegimeTransitions','ACM_RegimeDwellStats',
    'ACM_DriftEvents','ACM_CulpritHistory','ACM_EpisodeMetrics',
    'ACM_EpisodeDiagnostics',
    'ACM_DataQuality',
    'ACM_Scores_Long','ACM_Drift_TS',
    'ACM_Anomaly_Events','ACM_Regime_Episodes',
    'ACM_PCA_Models','ACM_PCA_Loadings','ACM_PCA_Metrics',
    'ACM_Run_Stats', 'ACM_SinceWhen',
    'ACM_SensorHotspots','ACM_SensorHotspotTimeline',
    'ACM_HealthForecast_TS','ACM_FailureForecast_TS',
    'ACM_RUL_TS','ACM_RUL_Summary','ACM_RUL_Attribution',
    'ACM_SensorForecast_TS','ACM_MaintenanceRecommendation',
    'ACM_EnhancedFailureProbability_TS','ACM_FailureCausation',
    'ACM_EnhancedMaintenanceRecommendation','ACM_RecommendedActions',
    'ACM_SensorNormalized_TS',
    'ACM_OMRContributions','ACM_OMRContributionsLong','ACM_FusionQuality','ACM_FusionQualityReport',
    'ACM_OMRTimeline','ACM_RegimeStats','ACM_DailyFusedProfile',
    'ACM_OMR_Diagnostics','ACM_Forecast_QualityMetrics',
    'ACM_HealthDistributionOverTime','ACM_ChartGenerationLog',
    'ACM_FusionMetrics',
    'ACM_HealthForecast_Continuous','ACM_FailureHazard_TS',
    'ACM_ThresholdMetadata'
}

def main():
    print("Auditing ACM_* tables in SQL Server...")
    
    try:
        sql = SQLClient.from_ini('acm')
        sql.connect()
        
        # Query all ACM_* tables
        cur = sql.cursor()
        cur.execute("""
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'dbo' 
              AND TABLE_NAME LIKE 'ACM_%'
            ORDER BY TABLE_NAME
        """)
        
        existing = {row[0] for row in cur.fetchall()}
        
        print(f"\nFound {len(existing)} ACM_* tables in database")
        print(f"Expected {len(EXPECTED_TABLES)} tables from ALLOWED_TABLES")
        
        missing = EXPECTED_TABLES - existing
        unexpected = existing - EXPECTED_TABLES
        
        if missing:
            print(f"\n❌ MISSING {len(missing)} TABLES:")
            for t in sorted(missing):
                print(f"  - {t}")
        else:
            print("\n✓ All expected tables exist")
        
        if unexpected:
            print(f"\n⚠ UNEXPECTED {len(unexpected)} TABLES (not in ALLOWED_TABLES):")
            for t in sorted(unexpected):
                print(f"  - {t}")
        
        sql.close()
        
        if missing:
            print("\n💡 Create missing tables using scripts in scripts/sql/")
            return 1
        else:
            print("\n✓ ACM table audit PASSED")
            return 0
            
    except Exception as e:
        print(f"❌ Audit failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
