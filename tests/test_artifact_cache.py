"""
Test suite for FCST-15 and RUL-01: Artifact cache functionality
Tests OutputManager artifact cache for SQL-only mode support.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.output_manager import OutputManager


def test_artifact_cache_stores_on_write(tmp_path):
    """Test that write_dataframe caches the DataFrame."""
    output_manager = OutputManager(
        sql_client=None,
        run_id="test_run",
        equip_id=1,
        base_output_dir=tmp_path,
        sql_only_mode=False
    )
    
    # Create test data
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
        'Value': np.random.randn(10)
    })
    
    # Write the dataframe
    file_path = tmp_path / "test_data.csv"
    result = output_manager.write_dataframe(df, file_path)
    
    # Check that it was cached
    assert result['file_written'] is True
    cached_df = output_manager.get_cached_table("test_data.csv")
    assert cached_df is not None
    assert len(cached_df) == len(df)
    assert list(cached_df.columns) == list(df.columns)


def test_artifact_cache_returns_copy(tmp_path):
    """Test that get_cached_table returns a copy, not the original."""
    output_manager = OutputManager(
        sql_client=None,
        run_id="test_run",
        equip_id=1,
        base_output_dir=tmp_path,
        sql_only_mode=False
    )
    
    # Create and cache data
    df = pd.DataFrame({'Value': [1, 2, 3]})
    file_path = tmp_path / "test_data.csv"
    output_manager.write_dataframe(df, file_path)
    
    # Get cached data and modify it
    cached_df1 = output_manager.get_cached_table("test_data.csv")
    cached_df1.loc[0, 'Value'] = 999
    
    # Get cached data again - should not reflect the modification
    cached_df2 = output_manager.get_cached_table("test_data.csv")
    assert cached_df2.loc[0, 'Value'] != 999
    assert cached_df2.loc[0, 'Value'] == 1


def test_artifact_cache_missing_table(tmp_path):
    """Test that get_cached_table returns None for missing tables."""
    output_manager = OutputManager(
        sql_client=None,
        run_id="test_run",
        equip_id=1,
        base_output_dir=tmp_path
    )
    
    cached_df = output_manager.get_cached_table("nonexistent.csv")
    assert cached_df is None


def test_artifact_cache_list_tables(tmp_path):
    """Test that list_cached_tables returns all cached table names."""
    output_manager = OutputManager(
        sql_client=None,
        run_id="test_run",
        equip_id=1,
        base_output_dir=tmp_path,
        sql_only_mode=False
    )
    
    # Cache multiple tables
    for i in range(3):
        df = pd.DataFrame({'Value': [i]})
        file_path = tmp_path / f"table_{i}.csv"
        output_manager.write_dataframe(df, file_path)
    
    cached_tables = output_manager.list_cached_tables()
    assert len(cached_tables) == 3
    assert "table_0.csv" in cached_tables
    assert "table_1.csv" in cached_tables
    assert "table_2.csv" in cached_tables


def test_artifact_cache_clear(tmp_path):
    """Test that clear_artifact_cache removes all cached tables."""
    output_manager = OutputManager(
        sql_client=None,
        run_id="test_run",
        equip_id=1,
        base_output_dir=tmp_path,
        sql_only_mode=False
    )
    
    # Cache some data
    df = pd.DataFrame({'Value': [1, 2, 3]})
    file_path = tmp_path / "test_data.csv"
    output_manager.write_dataframe(df, file_path)
    
    assert len(output_manager.list_cached_tables()) == 1
    
    # Clear cache
    output_manager.clear_artifact_cache()
    
    assert len(output_manager.list_cached_tables()) == 0
    assert output_manager.get_cached_table("test_data.csv") is None


def test_sql_only_mode_caches_without_file_write(tmp_path):
    """Test that SQL-only mode still caches data even without file write."""
    output_manager = OutputManager(
        sql_client=None,  # No SQL client, but sql_only_mode=True
        run_id="test_run",
        equip_id=1,
        base_output_dir=tmp_path,
        sql_only_mode=True
    )
    
    # Create test data
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
        'Value': [1, 2, 3, 4, 5]
    })
    
    # Write the dataframe (no file should be created in SQL-only mode)
    file_path = tmp_path / "scores.csv"
    result = output_manager.write_dataframe(df, file_path)
    
    # File should not be written in SQL-only mode
    assert result['file_written'] is False
    assert not file_path.exists()
    
    # But data should still be cached for downstream modules
    cached_df = output_manager.get_cached_table("scores.csv")
    assert cached_df is not None
    assert len(cached_df) == 5
    assert list(cached_df['Value']) == [1, 2, 3, 4, 5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
