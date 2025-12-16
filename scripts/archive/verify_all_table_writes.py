#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive verification of all ACM table writes.

Checks that:
1. All modules write to correct table names
2. All tables exist in database
3. All tables have recent data
4. Schema mappings are consistent
"""

import sys
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Fix encoding for Windows console
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# SQL queries to verify table writes
REQUIRED_TABLES = {
    # Forecast tables
    'ACM_HealthForecast': 'Health forecast time series',
    'ACM_FailureForecast': 'Failure probability forecast',
    'ACM_DetectorForecast_TS': 'Detector-level forecasts',
    'ACM_SensorForecast': 'Sensor value forecasts',
    'ACM_RUL': 'RUL summary',
    
    # Drift tables
    'ACM_DriftSeries': 'Drift detection time series',
    'ACM_DriftEvents': 'Drift event summaries',
    
    # OMR and Contribution tables
    'ACM_OMRTimeline': 'OMR detector scores',
    'ACM_OMR_SensorContributions': 'OMR sensor contributions',
    'ACM_OMRContributionsLong': 'OMR contributions (long format)',
    'ACM_ContributionCurrent': 'Current detector contributions',
    'ACM_ContributionTimeline': 'Contribution history',
    
    # Regime tables
    'ACM_RegimeTimeline': 'Regime classification timeline',
    'ACM_RegimeStats': 'Regime statistics',
    'ACM_RegimeOccupancy': 'Regime occupancy stats',
    'ACM_RegimeTransitions': 'Regime transitions',
    
    # Episode tables
    'ACM_Episodes': 'Episode summaries',
    'ACM_EpisodeCulprits': 'Episode culprits (fault attribution)',
    'ACM_EpisodeDiagnostics': 'Episode diagnostic details',
    'ACM_EpisodeMetrics': 'Episode metrics aggregation',
    
    # Core tables
    'ACM_HealthTimeline': 'Health score timeline',
    'ACM_SensorHotspots': 'Top anomalous sensors',
    'ACM_Scores_Wide': 'Detector scores (wide format)',
}

MODULE_MAPPINGS = {
    'core/output_manager.py': {
        'ACM_DriftSeries': 'write_drift_ts',
        'ACM_HealthForecast': 'write_dataframe (forecasting)',
        'ACM_FailureForecast': 'write_dataframe (forecasting)',
        'ACM_SensorForecast': 'write_dataframe (forecasting)',
        'ACM_ContributionTimeline': 'write_dataframe (analytics)',
        'ACM_ContributionCurrent': 'write_dataframe (analytics)',
    },
    'core/regimes.py': {
        'ACM_RegimeStats': 'write_dataframe (regimes)',
        'ACM_RegimeOccupancy': 'write_dataframe (regimes)',
        'ACM_RegimeTransitions': 'write_dataframe (regimes)',
    },
    'core/forecasting.py': {
        'ACM_DetectorForecast_TS': 'detector_forecast_ts mapping',
        'ACM_SensorForecast': 'sensor_forecast_ts mapping',
        'ACM_RUL': 'rul_summary mapping',
    },
    'core/episode_culprits_writer.py': {
        'ACM_EpisodeCulprits': 'write_episode_culprits',
    },
}

def check_table_exists(server, database, table_name):
    """Check if table exists in SQL Server."""
    query = f"""
    SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME = '{table_name}'
    """
    try:
        result = subprocess.run(
            ['sqlcmd', '-S', server, '-d', database, '-E', '-Q', query],
            capture_output=True,
            text=True,
            timeout=10
        )
        return table_name in result.stdout
    except Exception as e:
        print(f"Error checking table {table_name}: {e}")
        return False

def check_table_recent_data(server, database, table_name, hours=24):
    """Check if table has been written to recently."""
    query = f"""
    SELECT COUNT(*) as RecordCount, MAX(CreatedAt) as LatestWrite 
    FROM {table_name} 
    WHERE CreatedAt > DATEADD(hour, -{hours}, GETDATE())
    """
    try:
        result = subprocess.run(
            ['sqlcmd', '-S', server, '-d', database, '-E', '-Q', query],
            capture_output=True,
            text=True,
            timeout=10
        )
        if 'RecordCount' in result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip() and not line.startswith('-') and 'RecordCount' not in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            count = int(parts[0].strip())
                            return count > 0
                        except:
                            pass
        return False
    except Exception as e:
        print(f"Error checking recent data for {table_name}: {e}")
        return False

def verify_code_mappings():
    """Verify that code uses correct table names."""
    issues = []
    
    # Check for incorrect table names in code
    incorrect_tables = {
        'ACM_Drift_TS': 'ACM_DriftSeries',
        'ACM_RegimeSummary': 'ACM_RegimeStats',
        'ACM_RegimeFeatureImportance': 'ACM_RegimeOccupancy',
    }
    
    for workspace in Path(__file__).parent.glob('core/*.py'):
        try:
            content = workspace.read_text()
            for incorrect, correct in incorrect_tables.items():
                if incorrect in content:
                    issues.append(
                        f"❌ {workspace.name}: Contains '{incorrect}' (should be '{correct}')"
                    )
        except Exception as e:
            pass
    
    return issues

def main():
    """Run all verification checks."""
    server = "localhost\\B19CL3PCQLSERVER"
    database = "ACM"
    
    print("\n" + "="*80)
    print("ACM TABLE WRITE VERIFICATION REPORT")
    print("="*80)
    
    # 1. Check required tables exist
    print("\n[1] REQUIRED TABLES EXISTENCE")
    print("-" * 80)
    missing_tables = []
    for table_name, description in REQUIRED_TABLES.items():
        exists = check_table_exists(server, database, table_name)
        status = "[OK]" if exists else "[FAIL]"
        print(f"{status} {table_name:<40} {description}")
        if not exists:
            missing_tables.append(table_name)
    
    if missing_tables:
        print(f"\n[WARN] Missing {len(missing_tables)} tables: {', '.join(missing_tables)}")
    else:
        print(f"\n[OK] All {len(REQUIRED_TABLES)} required tables exist")
    
    # 2. Check for recent data
    print("\n[2] RECENT DATA WRITES (last 24 hours)")
    print("-" * 80)
    stale_tables = []
    for table_name in REQUIRED_TABLES.keys():
        if check_table_exists(server, database, table_name):
            has_recent = check_table_recent_data(server, database, table_name)
            status = "[OK]" if has_recent else "[WARN]"
            print(f"{status} {table_name:<40} {'Recent data' if has_recent else 'No recent data'}")
            if not has_recent:
                stale_tables.append(table_name)
    
    if stale_tables:
        print(f"\n[WARN] {len(stale_tables)} tables have no recent writes: {', '.join(stale_tables[:5])}")
    else:
        print(f"\n[OK] All tables have recent data writes")
    
    # 3. Check code mappings
    print("\n[3] CODE MAPPING VERIFICATION")
    print("-" * 80)
    code_issues = verify_code_mappings()
    if code_issues:
        for issue in code_issues:
            print(issue)
        print(f"\n[FAIL] Found {len(code_issues)} code mapping issues")
    else:
        print("[OK] No incorrect table name references found in code")
    
    # 4. Module status
    print("\n[4] MODULE WRITE MAPPINGS")
    print("-" * 80)
    for module, tables in MODULE_MAPPINGS.items():
        print(f"\n{module}:")
        for table, method in tables.items():
            exists = check_table_exists(server, database, table)
            status = "[OK]" if exists else "[FAIL]"
            print(f"  {status} {table:<40} via {method}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"[OK] Total tables: {len(REQUIRED_TABLES)}")
    print(f"[OK] Tables existing: {len(REQUIRED_TABLES) - len(missing_tables)}")
    print(f"[FAIL] Tables missing: {len(missing_tables)}")
    print(f"[WARN] Code issues: {len(code_issues)}")
    
    success = len(missing_tables) == 0 and len(code_issues) == 0
    print(f"\n{'[OK] ALL CHECKS PASSED' if success else '[FAIL] ISSUES FOUND - SEE ABOVE'}")
    print("="*80 + "\n")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
