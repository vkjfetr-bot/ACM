"""
RUL Engine Validation Script
============================

Validates the unified RUL engine (core.rul_engine) by testing:
1. Module imports and API surface
2. Configuration handling
3. Data loading from SQL
4. Core RUL computation
5. Output table generation

Usage:
    python scripts/validate_rul_engine.py --equip FD_FAN --run-id <run_id>
    python scripts/validate_rul_engine.py --equip GAS_TURBINE --test-mode

The script performs structural validation of the RUL engine without requiring
a full ACM pipeline run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import Console


def validate_imports() -> bool:
    """Validate that core.rul_engine can be imported and has expected API."""
    Console.info("[VALIDATE] Testing rul_engine imports...")
    
    try:
        from core import rul_engine
        Console.info("  ✓ core.rul_engine imported successfully")
    except ImportError as e:
        Console.error(f"  ✗ Failed to import core.rul_engine: {e}")
        return False
    
    # Check for main API function
    if not hasattr(rul_engine, 'run_rul'):
        Console.error("  ✗ rul_engine missing 'run_rul' function")
        return False
    Console.info("  ✓ rul_engine.run_rul exists")
    
    # Check for RULConfig
    if not hasattr(rul_engine, 'RULConfig'):
        Console.error("  ✗ rul_engine missing 'RULConfig' class")
        return False
    Console.info("  ✓ rul_engine.RULConfig exists")
    
    # Check for I/O functions
    for func_name in ['load_health_timeline', 'load_sensor_hotspots', 
                      'load_learning_state', 'save_learning_state']:
        if not hasattr(rul_engine, func_name):
            Console.error(f"  ✗ rul_engine missing '{func_name}' function")
            return False
    Console.info("  ✓ All I/O functions exist")
    
    # Check for core functions
    for func_name in ['compute_rul', 'compute_rul_multipath']:
        if not hasattr(rul_engine, func_name):
            Console.error(f"  ✗ rul_engine missing '{func_name}' function")
            return False
    Console.info("  ✓ All core computation functions exist")
    
    # Check for output builders
    for func_name in ['make_health_forecast_ts', 'make_failure_forecast_ts',
                      'make_rul_ts', 'make_rul_summary', 
                      'build_sensor_attribution', 'build_maintenance_recommendation']:
        if not hasattr(rul_engine, func_name):
            Console.error(f"  ✗ rul_engine missing '{func_name}' function")
            return False
    Console.info("  ✓ All output builder functions exist")
    
    Console.info("[VALIDATE] Import validation: PASSED")
    return True


def validate_config_handling() -> bool:
    """Validate RULConfig creation and normalization."""
    Console.info("[VALIDATE] Testing RULConfig handling...")
    
    try:
        from core.rul_engine import RULConfig
        
        # Test default config
        cfg = RULConfig()
        if cfg.health_threshold != 70.0:
            Console.error(f"  ✗ Default health_threshold should be 70.0, got {cfg.health_threshold}")
            return False
        Console.info("  ✓ Default RULConfig created successfully")
        
        # Test custom config
        cfg = RULConfig(
            health_threshold=65.0,
            max_forecast_hours=120.0,
            min_points=50
        )
        if cfg.health_threshold != 65.0:
            Console.error(f"  ✗ Custom health_threshold not set correctly")
            return False
        if cfg.max_forecast_hours != 120.0:
            Console.error(f"  ✗ Custom max_forecast_hours not set correctly")
            return False
        Console.info("  ✓ Custom RULConfig created successfully")
        
        Console.info("[VALIDATE] Config handling validation: PASSED")
        return True
        
    except Exception as e:
        Console.error(f"  ✗ Config validation failed: {e}")
        return False


def validate_data_structures() -> bool:
    """Validate learning state dataclasses."""
    Console.info("[VALIDATE] Testing data structures...")
    
    try:
        from core.rul_engine import ModelPerformanceMetrics, LearningState
        
        # Test ModelPerformanceMetrics
        metrics = ModelPerformanceMetrics(
            mae=1.5,
            rmse=2.0,
            bias=-0.5,
            recent_errors=[0.1, -0.2, 0.3],
            weight=0.33
        )
        if metrics.mae != 1.5:
            Console.error("  ✗ ModelPerformanceMetrics not initialized correctly")
            return False
        Console.info("  ✓ ModelPerformanceMetrics structure valid")
        
        # Test LearningState
        state = LearningState(
            equip_id=1,
            ar1_metrics=metrics,
            exp_metrics=metrics,
            weibull_metrics=metrics,
            calibration_factor=1.0,
            prediction_history=[]
        )
        if state.equip_id != 1:
            Console.error("  ✗ LearningState not initialized correctly")
            return False
        if state.ar1_metrics.mae != 1.5:
            Console.error("  ✗ LearningState metrics not set correctly")
            return False
        Console.info("  ✓ LearningState structure valid")
        
        Console.info("[VALIDATE] Data structure validation: PASSED")
        return True
        
    except Exception as e:
        Console.error(f"  ✗ Data structure validation failed: {e}")
        return False


def validate_with_sql(equip_id: int, run_id: str | None) -> bool:
    """Validate RUL engine with real SQL connection and data."""
    Console.info(f"[VALIDATE] Testing with SQL (EquipID={equip_id}, RunID={run_id})...")
    
    try:
        # Import SQL client
        from core.sql_client import SQLClient
        from core.output_manager import OutputManager
        from utils.config_dict import ConfigDict
        
        # Load config
        config_path = project_root / "configs" / "config_table.csv"
        config = ConfigDict(config_path)
        config.set_equipment("*")  # Start with global
        
        # Initialize SQL client
        sql_client = SQLClient()
        sql_client.connect()
        Console.info("  ✓ SQL connection established")
        
        # Initialize output manager
        output_mgr = OutputManager(
            equip_id=equip_id,
            run_id=run_id,
            sql_client=sql_client,
            enable_dual_write=False
        )
        Console.info("  ✓ OutputManager initialized")
        
        # Import and call run_rul
        from core.rul_engine import run_rul
        
        result = run_rul(
            equip_id=equip_id,
            run_id=run_id,
            config=config._data,  # Access underlying dict
            sql_client=sql_client,
            output_manager=output_mgr
        )
        
        # Validate result structure
        expected_tables = [
            'ACM_HealthForecast_TS',
            'ACM_FailureForecast_TS',
            'ACM_RUL_TS',
            'ACM_RUL_Summary',
            'ACM_RUL_Attribution',
            'ACM_MaintenanceRecommendation'
        ]
        
        for table_name in expected_tables:
            if table_name not in result:
                Console.warn(f"  ⚠ Missing output table: {table_name}")
            else:
                df = result[table_name]
                Console.info(f"  ✓ {table_name}: {len(df)} rows")
        
        sql_client.close()
        Console.info("[VALIDATE] SQL-based validation: PASSED")
        return True
        
    except Exception as e:
        Console.error(f"  ✗ SQL validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate unified RUL engine implementation"
    )
    parser.add_argument(
        "--equip",
        type=str,
        help="Equipment name (e.g., FD_FAN, GAS_TURBINE)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific RunID to validate (optional, uses latest if omitted)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run structural tests only (no SQL connection required)"
    )
    
    args = parser.parse_args()
    
    Console.info("="*60)
    Console.info("RUL Engine Validation Script")
    Console.info("="*60)
    
    # Run structural validation tests
    tests_passed = 0
    tests_total = 3
    
    if validate_imports():
        tests_passed += 1
    
    if validate_config_handling():
        tests_passed += 1
    
    if validate_data_structures():
        tests_passed += 1
    
    # Run SQL-based validation if requested
    if not args.test_mode:
        if not args.equip:
            Console.error("--equip required for SQL validation (or use --test-mode)")
            sys.exit(1)
        
        # Map equipment name to ID (simplified)
        equip_map = {
            'FD_FAN': 1,
            'GAS_TURBINE': 2,
            'TEST_EQUIP': 999
        }
        
        equip_id = equip_map.get(args.equip.upper())
        if equip_id is None:
            Console.error(f"Unknown equipment: {args.equip}")
            sys.exit(1)
        
        tests_total += 1
        if validate_with_sql(equip_id, args.run_id):
            tests_passed += 1
    
    # Print summary
    Console.info("="*60)
    Console.info(f"Validation Results: {tests_passed}/{tests_total} tests passed")
    Console.info("="*60)
    
    if tests_passed == tests_total:
        Console.info("✓ All validation tests PASSED")
        sys.exit(0)
    else:
        Console.error(f"✗ {tests_total - tests_passed} test(s) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
