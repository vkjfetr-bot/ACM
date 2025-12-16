import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from core.sql_client import SQLClient
from utils.logger import Console

def main():
    try:
        Console.info("Connecting to ACM database...")
        client = SQLClient.from_ini('acm').connect()
        
        # 1. Check ACM_Runs schema
        Console.info("Checking ACM_Runs schema...")
        query_schema = "SELECT TOP 1 * FROM dbo.ACM_Runs"
        runs_sample = pd.read_sql(query_schema, client.conn)
        print(f"Columns: {runs_sample.columns.tolist()}")
        
        # 1. Check latest runs (using available columns)
        Console.info("\n=== Latest 10 Runs ===")
        # Select columns that likely exist based on standard ACM schema, or just *
        query_runs = """
        SELECT TOP 10 *
        FROM dbo.ACM_Runs
        ORDER BY CreatedAt DESC
        """
        runs = pd.read_sql(query_runs, client.conn)
        print(runs.to_string())
        
        if runs.empty:
            Console.warn("No runs found in ACM_Runs.")
            return

        latest_run_id = runs.iloc[0]['RunID']
        Console.info(f"\nAnalyzing Latest Run: {latest_run_id}")
        
        # 2. Check ACM_RunLogs schema
        Console.info("Checking ACM_RunLogs schema...")
        query_schema_logs = "SELECT TOP 1 * FROM dbo.ACM_RunLogs"
        logs_sample = pd.read_sql(query_schema_logs, client.conn)
        print(f"Columns: {logs_sample.columns.tolist()}")

        # 2. Check logs for latest run (using available columns)
        Console.info("\n=== Errors/Warnings in Run Logs ===")
        # Select * for now to be safe
        query_logs = """
        SELECT *
        FROM dbo.ACM_RunLogs
        WHERE RunID = ? 
        ORDER BY LogTime
        """
        # Note: I'm guessing LogTime based on typical schemas, but I'll check the columns first
        # Actually, let's just select TOP 20 logs for that RunID without ordering if column unknown
        
        # Let's wait for schema check to be sure about columns
        # But to avoid another round trip, I'll try to select * and filter in pandas if needed
        query_logs = "SELECT * FROM dbo.ACM_RunLogs WHERE RunID = ?"
        logs = pd.read_sql(query_logs, client.conn, params=[latest_run_id])
        
        if logs.empty:
            print("No logs found for this run.")
        else:
            # Filter for errors/warnings if possible
            if 'Level' in logs.columns:
                errs = logs[logs['Level'].isin(['ERROR', 'WARNING', 'CRITICAL'])]
            elif 'LogLevel' in logs.columns:
                errs = logs[logs['LogLevel'].isin(['ERROR', 'WARNING', 'CRITICAL'])]
            else:
                errs = logs
            
            if errs.empty:
                print("No errors/warnings found.")
            else:
                print(errs.to_string())
            
        # 3. Check output tables row counts
        Console.info("\n=== Output Table Counts ===")
        tables = [
            "ACM_HealthTimeline",
            "ACM_SensorHotspots",
            "ACM_DefectTimeline",
            "ACM_HealthForecast_TS",
            "ACM_FailureForecast_TS",
            "ACM_RUL_Summary",
            "ACM_RUL_Attribution",
            "ACM_Scores_Long",
            "ACM_Anomaly_Events",
            "ACM_Regime_Episodes",
            "ACM_BaselineBuffer",
            "ACM_EpisodeCulprits",
            "ACM_EpisodeDiagnostics"
        ]
        
        for table in tables:
            if table == "ACM_BaselineBuffer":
                query_count = f"SELECT COUNT(*) as count FROM dbo.{table}"
                cursor = client.cursor()
                cursor.execute(query_count)
            else:
                query_count = f"SELECT COUNT(*) as count FROM dbo.{table} WHERE RunID = ?"
                cursor = client.cursor()
                cursor.execute(query_count, (latest_run_id,))
            
            count = cursor.fetchone()[0]
            print(f"{table}: {count} rows")
            
        # Check ACM_Scores_Long schema
        Console.info("\nChecking ACM_Scores_Long schema...")
        try:
            query_schema_scores = "SELECT TOP 1 * FROM dbo.ACM_Scores_Long"
            scores_sample = pd.read_sql(query_schema_scores, client.conn)
            print(f"Columns: {scores_sample.columns.tolist()}")
        except Exception as e:
            Console.error(f"Failed to read ACM_Scores_Long schema: {e}")
            
    except Exception as e:
        Console.error(f"Failed to analyze SQL status: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
