"""
SQL-50 Validation Script: End-to-End Pure SQL Operation
=======================================================

This script validates complete SQL-only operation by:
1. Running a full pipeline in SQL-only mode
2. Verifying no CSV/joblib files created (only charts allowed)
3. Checking all results are in SQL tables
4. Measuring performance (target: <15s per run)
5. Generating comprehensive validation report

Usage:
    python scripts/validate_sql_only_mode.py --equip FD_FAN
"""

import sys
from pathlib import Path
import time
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sql_client import SQLClient
from utils.config_dict import ConfigDict
from utils.logger import Console
import subprocess


def count_files_by_type(artifacts_dir: Path) -> dict:
    """Count files in artifacts directory by type."""
    counts = {
        'csv': 0,
        'joblib': 0,
        'json': 0,
        'png': 0,
        'html': 0,
        'other': 0,
        'total': 0
    }
    
    if not artifacts_dir.exists():
        return counts
    
    for file in artifacts_dir.rglob('*'):
        if file.is_file():
            counts['total'] += 1
            suffix = file.suffix.lower()
            
            if suffix == '.csv':
                counts['csv'] += 1
            elif suffix == '.joblib':
                counts['joblib'] += 1
            elif suffix == '.json':
                counts['json'] += 1
            elif suffix == '.png':
                counts['png'] += 1
            elif suffix == '.html':
                counts['html'] += 1
            else:
                counts['other'] += 1
    
    return counts


def get_sql_table_counts(sql_client: SQLClient, run_id: str) -> dict:
    """Get row counts from all ACM tables for the given run_id."""
    tables = [
        'ACM_Runs',
        'ACM_HealthTimeline',
        'ACM_ContributionTimeline',
        'ACM_SensorHotspots',
        'ACM_Episodes',
        'ACM_DefectSummary',
        'ACM_RegimeTimeline',
        'ACM_SensorForecast_TS',
        'ACM_HealthForecast_TS',
        'ACM_FailureForecast_TS',
        'ACM_RUL_TS',
        'ACM_RUL_Summary',
        'ACM_RUL_Attribution',
        'ACM_MaintenanceRecommendation',
        'ModelRegistry'
    ]
    
    counts = {}
    cursor = sql_client.conn.cursor()
    
    for table in tables:
        try:
            # ModelRegistry doesn't have RunID, use EquipID and latest version
            if table == 'ModelRegistry':
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE EquipID = 1 AND Version = (SELECT MAX(Version) FROM ModelRegistry WHERE EquipID = 1)")
            else:
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE RunID = ?", (run_id,))
            
            count = cursor.fetchone()[0]
            counts[table] = count
        except Exception as e:
            counts[table] = f"ERROR: {e}"
    
    return counts


def run_pipeline(equip: str) -> tuple:
    """Run ACM pipeline and return (success, run_id, duration)."""
    Console.info(f"[VALIDATE] Running pipeline for {equip} in SQL-only mode...")
    
    start_time = time.time()
    
    # Run pipeline
    cmd = [sys.executable, "-m", "core.acm_main", "--equip", equip]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    duration = time.time() - start_time
    
    # Extract run_id from output
    run_id = None
    for line in result.stdout.split('\n'):
        if 'RunID=' in line:
            # Extract UUID from line like "[RUN] Finalized RunID=DBEC14CC-3F66-4296-89C1-71858407C1A9"
            parts = line.split('RunID=')
            if len(parts) > 1:
                run_id = parts[1].split()[0].strip()
                break
    
    success = result.returncode == 0
    
    if not success:
        Console.error(f"[VALIDATE] Pipeline failed with exit code {result.returncode}")
        Console.error(f"[VALIDATE] STDERR: {result.stderr[-500:]}")  # Last 500 chars
    
    return success, run_id, duration


def main():
    parser = argparse.ArgumentParser(description='SQL-50: Validate SQL-only operation')
    parser.add_argument('--equip', default='FD_FAN', help='Equipment to test')
    args = parser.parse_args()
    
    Console.info("="*80)
    Console.info("SQL-50: End-to-End Pure SQL Operation Validation")
    Console.info("="*80)
    Console.info("")
    
    # Load config
    cfg = ConfigDict.from_csv("configs/config_table.csv", equip_id=1)
    
    # Verify SQL-only mode is enabled
    storage_backend = cfg.get("runtime", {}).get("storage_backend", "file")
    if storage_backend != "sql":
        Console.error(f"[VALIDATE] ✗ storage_backend={storage_backend}, expected 'sql'")
        Console.error(f"[VALIDATE] Please set storage_backend=sql in configs/config_table.csv")
        return 1
    
    Console.info(f"[VALIDATE] ✓ storage_backend=sql (SQL-only mode enabled)")
    Console.info("")
    
    # Get artifacts directory
    artifacts_dir = Path("artifacts") / args.equip
    
    # Count files BEFORE run
    Console.info("[VALIDATE] Counting files before pipeline run...")
    files_before = count_files_by_type(artifacts_dir)
    Console.info(f"[VALIDATE] Files before: {files_before['total']} total")
    Console.info(f"[VALIDATE]   - CSV: {files_before['csv']}, JOBLIB: {files_before['joblib']}, PNG: {files_before['png']}")
    Console.info("")
    
    # Run pipeline
    success, run_id, duration = run_pipeline(args.equip)
    
    if not success:
        Console.error("[VALIDATE] ✗ Pipeline execution failed")
        return 1
    
    Console.info(f"[VALIDATE] ✓ Pipeline completed successfully")
    Console.info(f"[VALIDATE] Duration: {duration:.2f}s (target: <15s)")
    Console.info(f"[VALIDATE] RunID: {run_id}")
    Console.info("")
    
    # Count files AFTER run
    Console.info("[VALIDATE] Counting files after pipeline run...")
    files_after = count_files_by_type(artifacts_dir)
    Console.info(f"[VALIDATE] Files after: {files_after['total']} total")
    Console.info(f"[VALIDATE]   - CSV: {files_after['csv']}, JOBLIB: {files_after['joblib']}, PNG: {files_after['png']}")
    Console.info("")
    
    # Calculate new files created
    new_files = {
        'csv': files_after['csv'] - files_before['csv'],
        'joblib': files_after['joblib'] - files_before['joblib'],
        'json': files_after['json'] - files_before['json'],
        'png': files_after['png'] - files_before['png'],
        'total': files_after['total'] - files_before['total']
    }
    
    Console.info("[VALIDATE] New files created:")
    Console.info(f"[VALIDATE]   - CSV: {new_files['csv']}")
    Console.info(f"[VALIDATE]   - JOBLIB: {new_files['joblib']}")
    Console.info(f"[VALIDATE]   - JSON: {new_files['json']}")
    Console.info(f"[VALIDATE]   - PNG: {new_files['png']}")
    Console.info(f"[VALIDATE]   - Total: {new_files['total']}")
    Console.info("")
    
    # Validation checks
    validation_passed = True
    
    # Check 1: No CSV files created
    if new_files['csv'] > 0:
        Console.error(f"[VALIDATE] ✗ Created {new_files['csv']} CSV files (expected 0)")
        validation_passed = False
    else:
        Console.info("[VALIDATE] ✓ No CSV files created")
    
    # Check 2: No JOBLIB files created
    if new_files['joblib'] > 0:
        Console.error(f"[VALIDATE] ✗ Created {new_files['joblib']} JOBLIB files (expected 0)")
        validation_passed = False
    else:
        Console.info("[VALIDATE] ✓ No JOBLIB files created")
    
    # Check 3: PNG files created (charts)
    if new_files['png'] > 0:
        Console.info(f"[VALIDATE] ✓ Created {new_files['png']} PNG chart files (expected behavior)")
    else:
        Console.warn("[VALIDATE] ⚠ No PNG charts created (chart generation may be disabled)")
    
    # Check 4: Performance target (<15s)
    if duration < 15.0:
        Console.info(f"[VALIDATE] ✓ Performance target met: {duration:.2f}s < 15s")
    else:
        Console.warn(f"[VALIDATE] ⚠ Performance target missed: {duration:.2f}s > 15s")
    
    Console.info("")
    
    # Check SQL tables
    if run_id:
        Console.info("[VALIDATE] Checking SQL table population...")
        sql_client = SQLClient(cfg, db_section="acm")
        sql_client.connect()
        
        table_counts = get_sql_table_counts(sql_client, run_id)
        
        critical_tables = [
            'ACM_Runs',
            'ACM_HealthTimeline',
            'ACM_ContributionTimeline',
            'ACM_SensorHotspots',
            'ModelRegistry'
        ]
        
        for table in critical_tables:
            count = table_counts.get(table, 0)
            if isinstance(count, str):  # Error
                Console.error(f"[VALIDATE] ✗ {table}: {count}")
                validation_passed = False
            elif count > 0:
                Console.info(f"[VALIDATE] ✓ {table}: {count} rows")
            else:
                Console.error(f"[VALIDATE] ✗ {table}: 0 rows (expected >0)")
                validation_passed = False
        
        Console.info("")
        Console.info("[VALIDATE] All SQL tables:")
        for table, count in sorted(table_counts.items()):
            if isinstance(count, str):
                Console.info(f"[VALIDATE]   {table}: {count}")
            else:
                Console.info(f"[VALIDATE]   {table}: {count:,} rows")
        
        sql_client.conn.close()
    else:
        Console.error("[VALIDATE] ✗ Could not extract RunID from pipeline output")
        validation_passed = False
    
    # Final verdict
    Console.info("")
    Console.info("="*80)
    if validation_passed:
        Console.info("✓ SQL-50 VALIDATION PASSED")
        Console.info("="*80)
        Console.info("")
        Console.info("Summary:")
        Console.info(f"  - No CSV/JOBLIB files created: ✓")
        Console.info(f"  - All data in SQL tables: ✓")
        Console.info(f"  - Performance: {duration:.2f}s {'✓' if duration < 15 else '⚠'}")
        Console.info(f"  - Charts generated: {new_files['png']} PNG files")
        Console.info("")
        return 0
    else:
        Console.error("✗ SQL-50 VALIDATION FAILED")
        Console.info("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
