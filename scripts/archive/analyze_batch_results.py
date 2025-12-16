import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.sql_client import SQLClient
from utils.logger import Console

def analyze_batch_results():
    try:
        client = SQLClient.from_ini('acm')
    except Exception as e:
        Console.error(f"Failed to initialize SQLClient: {e}")
        return

    if not client.connect():
        Console.error("Failed to connect to SQL Server")
        return

    try:
        # 1. Get Run Statistics
        query_runs = """
        SELECT 
            RunID, 
            EquipID, 
            StartedAt, 
            CompletedAt, 
            DurationSeconds, 
            HealthStatus, 
            AvgHealthIndex, 
            ErrorMessage
        FROM dbo.ACM_Runs 
        ORDER BY StartedAt DESC
        """
        runs = pd.read_sql(query_runs, client.conn)
        
        if runs.empty:
            Console.warn("No runs found in ACM_Runs.")
            return

        latest_runs = runs.head(10)
        Console.info("\n=== Latest 10 Runs ===")
        print(latest_runs.to_string())

        # 2. Analyze Row Counts per Run for Key Tables
        # We expect ~1440 rows per run if 1-minute sampling and 24h batch
        # Or ~8640 rows if 10-second sampling
        
        tables_to_check = [
            "ACM_HealthTimeline",
            "ACM_Scores_Long",
            "ACM_Drift_TS",
            "ACM_RegimeTimeline"
        ]
        
        Console.info("\n=== Row Counts per Run (Latest 5) ===")
        
        for run_id in latest_runs['RunID'].head(5):
            print(f"\nRun: {run_id}")
            for table in tables_to_check:
                try:
                    query = f"SELECT COUNT(*) as count FROM dbo.{table} WHERE RunID = ?"
                    cursor = client.cursor()
                    cursor.execute(query, (run_id,))
                    count = cursor.fetchone()[0]
                    print(f"  {table}: {count}")
                except Exception as e:
                    print(f"  {table}: Error ({e})")

        # 3. Data Validity Check (Sample Data)
        Console.info("\n=== Data Validity Check (Latest Run) ===")
        latest_run_id = latest_runs.iloc[0]['RunID']
        
        # Check HealthTimeline
        query_health = "SELECT TOP 5 * FROM dbo.ACM_HealthTimeline WHERE RunID = ? ORDER BY Timestamp DESC"
        health_sample = pd.read_sql(query_health, client.conn, params=[latest_run_id])
        print("\n-- ACM_HealthTimeline Sample --")
        print(health_sample.to_string())
        
        # Check Scores Long
        query_scores = "SELECT TOP 5 * FROM dbo.ACM_Scores_Long WHERE RunID = ? ORDER BY Timestamp DESC"
        scores_sample = pd.read_sql(query_scores, client.conn, params=[latest_run_id])
        print("\n-- ACM_Scores_Long Sample --")
        print(scores_sample.to_string())

        # Check Episodes
        query_episodes = "SELECT * FROM dbo.ACM_Episodes WHERE RunID = ?"
        episodes = pd.read_sql(query_episodes, client.conn, params=[latest_run_id])
        print(f"\n-- ACM_Episodes ({len(episodes)} rows) --")
        if not episodes.empty:
            print(episodes.head().to_string())

    except Exception as e:
        Console.error(f"Analysis failed: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    analyze_batch_results()
