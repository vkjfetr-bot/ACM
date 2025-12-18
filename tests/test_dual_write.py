"""
Test script for dual-write mode (Phase 2).

Tests that ACM can write to both files and SQL when dual_mode is enabled,
and gracefully falls back to file-only mode when SQL fails.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from core.sql_client import SQLClient
from core.observability import Console


def _require_storage_module():
    """Import core.storage or skip tests if unavailable."""
    try:
        import importlib

        return importlib.import_module("core.storage")
    except Exception as exc:
        Console.warn(f"Storage module unavailable: {exc}", component="TEST")
        pytest.skip(f"core.storage unavailable: {exc}")

def test_sql_connection():
    """Test basic SQL connectivity."""
    Console.info("Testing SQL connection...", component="TEST")
    result = None
    try:
        sql_client = SQLClient.from_ini('acm')
        sql_client.connect()
    except Exception as exc:
        Console.warn(f"SQL connection unavailable: {exc}", component="TEST")
        pytest.skip(f"SQL connection unavailable: {exc}")

    try:
        with sql_client.cursor() as cur:
            cur.execute("SELECT 1 AS test")
            result = cur.fetchone()
            Console.info(f"SQL connection OK: {result}", component="TEST")
    finally:
        try:
            sql_client.close()
        except Exception:
            pass

    assert result is not None
    assert result[0] == 1

def test_dual_mode_config():
    """Test that dual_mode config can be read from SQL."""
    Console.info("Testing dual_mode config read...", component="TEST")
    try:
        from utils.sql_config import get_equipment_config

        cfg = get_equipment_config(
            equipment_code="FD_FAN",
            use_sql=True,
            fallback_to_csv=True
        )
    except Exception as exc:
        Console.warn(f"Config read failed: {exc}", component="TEST")
        pytest.skip(f"SQL config unavailable: {exc}")

    dual_mode = cfg.get('output', {}).get('dual_mode', False)
    Console.info(f"dual_mode config: {dual_mode}", component="TEST")
    assert isinstance(dual_mode, bool)

def test_sql_health_check():
    """Test SQL health check with caching."""
    Console.info("Testing SQL health check...", component="TEST")
    storage = None
    try:
        storage = _require_storage_module()
        sql_client = SQLClient.from_ini('acm')
        sql_client.connect()
    except Exception as exc:
        Console.warn(f"Health check skipped: {exc}", component="TEST")
        pytest.skip(f"SQL connection unavailable: {exc}")

    assert storage is not None

    try:
        result1 = storage._check_sql_health(sql_client)
        Console.info(f"First health check: {result1}", component="TEST")

        result2 = storage._check_sql_health(sql_client)
        Console.info(f"Second health check (cached): {result2}", component="TEST")
    finally:
        try:
            sql_client.close()
        except Exception:
            pass

    result3 = storage._check_sql_health(None)
    Console.info(f"Health check with None client: {result3}", component="TEST")

    assert result1
    assert result2
    assert not result3

def test_storage_write_functions():
    """Test that storage write functions accept SQL parameters."""
    Console.info("Testing storage write function signatures...", component="TEST")
    storage = _require_storage_module()
    import pandas as pd
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        test_df = pd.DataFrame({
            'score': [1.0, 2.0, 3.0]
        }, index=pd.date_range('2025-01-01', periods=3, freq='1min'))

        result = storage.write_scores_csv(
            tmp_path,
            test_df,
            sql_client=None,
            run_id='test-run-id',
            equip_id=123
        )
        Console.info(f"write_scores_csv result: {result}", component="TEST")

        episodes_df = pd.DataFrame({
            'start_ts': ['2025-01-01 00:00:00'],
            'end_ts': ['2025-01-01 00:10:00'],
            'duration_hours': [0.167]
        })

        result2 = storage.write_episodes_csv(
            tmp_path,
            episodes_df,
            sql_client=None,
            run_id='test-run-id',
            equip_id=123
        )
        Console.info(f"write_episodes_csv result: {result2}", component="TEST")

    assert result is not None
    assert result2 is not None

def main():
    """Run all tests."""
    Console.info("=== Testing Dual-Write Mode (Phase 2) ===")
    
    tests = [
        ("SQL Connection", test_sql_connection),
        ("Dual Mode Config", test_dual_mode_config),
        ("SQL Health Check", test_sql_health_check),
        ("Storage Write Functions", test_storage_write_functions),
    ]
    
    results = []
    for test_name, test_func in tests:
        Console.info(f"\n--- Running: {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✓ PASS" if result else "✗ FAIL"
            Console.info(f"{test_name}: {status}", component="TEST")
        except Exception as e:
            results.append((test_name, False))
            Console.error(f"{test_name}: ✗ EXCEPTION - {e}", component="TEST")
    
    Console.info("\n=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    Console.info(f"Passed: {passed}/{total}")
    
    for test_name, result in results:
        status = "✓" if result else "✗"
        Console.info(f"  {status} {test_name}")
    
    if passed == total:
        Console.info("\n✓ All tests passed!")
        return 0
    else:
        Console.warn(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
