import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.sql_client import SQLClient
from utils.logger import Console

def check_tables():
    try:
        client = SQLClient.from_ini('acm')
    except Exception as e:
        Console.error(f"Failed to initialize SQLClient: {e}")
        return

    if not client.connect():
        Console.error("Failed to connect to SQL Server")
        return

    tables_to_check = [
        "ACM_Scores_Wide",
        "ACM_Scores_Long",
        "ACM_PCA_Metrics",
        "ACM_PCA_Loadings",
        "ACM_PCA_Models",
        "ACM_Regime_Episodes",
        "ACM_EpisodeMetrics",
        "ACM_EpisodeDiagnostics",
        "ACM_DetectorCorrelation",
        "ACM_CalibrationSummary",
        "ACM_OMRContributions",
        "ACM_FusionQuality",
        "ACM_RegimeOccupancy",
        "ACM_RegimeTransitions",
        "ACM_RegimeDwellStats",
        "ACM_RegimeTimeline",
        "ACM_RegimeStability",
        "ACM_HealthTimeline",
        "ACM_HealthHistogram",
        "ACM_HealthZoneByPeriod",
        "ACM_ContributionCurrent",
        "ACM_ContributionTimeline",
        "ACM_DriftSeries",
        "ACM_DriftEvents",
        "ACM_ThresholdCrossings",
        "ACM_SinceWhen",
        "ACM_SensorRanking",
        "ACM_SensorHotspots",
        "ACM_SensorHotspotTimeline",
        "ACM_SensorDefects",
        "ACM_SensorAnomalyByPeriod",
        "ACM_SensorNormalized_TS",
        "ACM_DailyFusedProfile",
        "ACM_CulpritHistory",
        "ACM_DataQuality",
        "ACM_FusionMetrics",
        "ACM_ChartGenerationLog",
        "ACM_SensorForecast_TS"
    ]

    print(f"{'Table Name':<30} | {'Exists':<10} | {'Row Count':<10}")
    print("-" * 56)

    for table in tables_to_check:
        try:
            # Check existence in tables or views
            check_query = """
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = ?
            """
            cursor = client.cursor()
            cursor.execute(check_query, (table,))
            exists = cursor.fetchone()[0] > 0
            
            # If not found, check synonyms
            if not exists:
                check_synonym = "SELECT COUNT(*) FROM sys.synonyms WHERE name = ?"
                cursor.execute(check_synonym, (table,))
                exists = cursor.fetchone()[0] > 0
            
            row_count = "N/A"
            if exists:
                try:
                    count_query = f"SELECT COUNT(*) FROM dbo.{table}"
                    cursor.execute(count_query)
                    row_count = str(cursor.fetchone()[0])
                except Exception:
                    row_count = "Error"

            print(f"{table:<30} | {str(exists):<10} | {row_count:<10}")
            
        except Exception as e:
            print(f"{table:<30} | Error      | {e}")

    client.close()

if __name__ == "__main__":
    check_tables()
