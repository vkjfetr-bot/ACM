"""
Tests for RegressionHarness (P5.6) in tests/regression_harness.py

Tests the regression harness for comparing ACM behavior against golden datasets.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path
from typing import Dict

from tests.regression_harness import (
    RegressionStatus,
    RegressionResult,
    GoldenDataset,
    RegressionSummary,
    RegressionHarness
)


# =============================================================================
# TEST: RegressionStatus Enum
# =============================================================================


class TestRegressionStatus:
    """Tests for RegressionStatus enum."""
    
    def test_values_exist(self):
        """Test all expected values exist."""
        assert RegressionStatus.PASSED.value == "PASSED"
        assert RegressionStatus.FAILED.value == "FAILED"
        assert RegressionStatus.SKIPPED.value == "SKIPPED"
        assert RegressionStatus.ERROR.value == "ERROR"
    
    def test_enum_count(self):
        """Test correct number of values."""
        assert len(RegressionStatus) == 4


# =============================================================================
# TEST: RegressionResult
# =============================================================================


class TestRegressionResult:
    """Tests for RegressionResult dataclass."""
    
    def test_basic_creation(self):
        """Test basic creation."""
        result = RegressionResult(
            test_name="ds1/metric1",
            passed=True,
            metric_name="metric1",
            golden_value=100.0,
            current_value=100.5,
            tolerance=0.05,
            deviation_pct=0.5
        )
        assert result.test_name == "ds1/metric1"
        assert result.passed is True
        assert result.status == RegressionStatus.PASSED
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = RegressionResult(
            test_name="test/metric",
            passed=False,
            metric_name="metric",
            golden_value=100.0,
            current_value=120.0,
            tolerance=0.05,
            deviation_pct=20.0,
            status=RegressionStatus.FAILED
        )
        d = result.to_dict()
        assert d["test_name"] == "test/metric"
        assert d["passed"] is False
        assert d["status"] == "FAILED"
        assert d["tolerance_pct"] == 5.0
        assert d["deviation_pct"] == 20.0
    
    def test_repr(self):
        """Test string representation."""
        result = RegressionResult(
            test_name="ds/metric",
            passed=True,
            metric_name="metric",
            golden_value=100.0,
            current_value=100.5,
            tolerance=0.05,
            deviation_pct=0.5
        )
        s = repr(result)
        assert "PASS" in s
        assert "ds/metric" in s


# =============================================================================
# TEST: GoldenDataset
# =============================================================================


class TestGoldenDataset:
    """Tests for GoldenDataset dataclass."""
    
    def test_basic_creation(self):
        """Test basic creation."""
        ds = GoldenDataset(
            name="test_dataset",
            equipment="FD_FAN",
            input_file=Path("test_input.csv"),
            expected_outputs={"metric1": 100.0}
        )
        assert ds.name == "test_dataset"
        assert ds.equipment == "FD_FAN"
        assert ds.expected_outputs["metric1"] == 100.0
    
    def test_get_tolerance_with_explicit(self):
        """Test get_tolerance with explicit value."""
        ds = GoldenDataset(
            name="test",
            equipment="FD_FAN",
            input_file=Path("test.csv"),
            expected_outputs={"metric1": 100.0},
            tolerance={"metric1": 0.1}
        )
        assert ds.get_tolerance("metric1") == 0.1
    
    def test_get_tolerance_default(self):
        """Test get_tolerance returns default for unknown metric."""
        ds = GoldenDataset(
            name="test",
            equipment="FD_FAN",
            input_file=Path("test.csv"),
            expected_outputs={"metric1": 100.0}
        )
        assert ds.get_tolerance("unknown_metric", default=0.07) == 0.07
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        ds = GoldenDataset(
            name="test",
            equipment="FD_FAN",
            input_file=Path("path/to/input.csv"),
            expected_outputs={"m1": 10.0, "m2": 20.0},
            tolerance={"m1": 0.05}
        )
        d = ds.to_dict()
        assert d["name"] == "test"
        assert d["equipment"] == "FD_FAN"
        assert d["input_file"] == "input.csv"
        assert d["expected_outputs"]["m1"] == 10.0


# =============================================================================
# TEST: RegressionSummary
# =============================================================================


class TestRegressionSummary:
    """Tests for RegressionSummary dataclass."""
    
    def test_empty_summary(self):
        """Test empty summary."""
        summary = RegressionSummary()
        assert summary.total_tests == 0
        assert summary.passed == 0
        assert summary.all_passed is True
        assert summary.success_rate == 100.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        summary = RegressionSummary(total_tests=10, passed=8, failed=2)
        assert summary.success_rate == 80.0
    
    def test_all_passed_with_failures(self):
        """Test all_passed is False when failures exist."""
        summary = RegressionSummary(total_tests=10, passed=9, failed=1)
        assert summary.all_passed is False
    
    def test_all_passed_with_errors(self):
        """Test all_passed is False when errors exist."""
        summary = RegressionSummary(total_tests=10, passed=9, errors=1)
        assert summary.all_passed is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = RegressionSummary(
            total_tests=5,
            passed=4,
            failed=1
        )
        d = summary.to_dict()
        assert d["total_tests"] == 5
        assert d["passed"] == 4
        assert d["failed"] == 1
        assert d["success_rate_pct"] == 80.0
        assert d["all_passed"] is False


# =============================================================================
# TEST: RegressionHarness Initialization
# =============================================================================


class TestRegressionHarnessInit:
    """Tests for RegressionHarness initialization."""
    
    def test_init_without_golden_dir(self):
        """Test initialization without golden_dir."""
        harness = RegressionHarness(golden_dir=None)
        assert harness.golden_dir is None
        assert harness.datasets == {}
        assert harness.default_tolerance == 0.05
    
    def test_init_with_nonexistent_dir(self):
        """Test initialization with directory that doesn't exist."""
        harness = RegressionHarness(golden_dir=Path("/nonexistent/path"))
        assert harness.datasets == {}
    
    def test_init_custom_tolerance(self):
        """Test initialization with custom default tolerance."""
        harness = RegressionHarness(golden_dir=None, default_tolerance=0.1)
        assert harness.default_tolerance == 0.1


# =============================================================================
# TEST: RegressionHarness.add_dataset()
# =============================================================================


class TestRegressionHarnessAddDataset:
    """Tests for add_dataset method."""
    
    def test_add_dataset(self):
        """Test adding a dataset."""
        harness = RegressionHarness(golden_dir=None)
        ds = GoldenDataset(
            name="test_ds",
            equipment="FD_FAN",
            input_file=Path("test.csv"),
            expected_outputs={"metric": 100.0}
        )
        harness.add_dataset(ds)
        assert "test_ds" in harness.datasets
        assert harness.datasets["test_ds"].equipment == "FD_FAN"


# =============================================================================
# TEST: RegressionHarness.run_regression()
# =============================================================================


class TestRegressionHarnessRunRegression:
    """Tests for run_regression method."""
    
    @pytest.fixture
    def temp_golden_dir(self, tmp_path):
        """Create temporary golden directory with test data."""
        # Create input file
        input_df = pd.DataFrame({
            "Timestamp": ["2024-01-15 10:00:00"],
            "Sensor1": [100.0],
            "Sensor2": [200.0]
        })
        input_file = tmp_path / "test_input.csv"
        input_df.to_csv(input_file, index=False)
        
        # Create manifest
        manifest = {
            "version": "1.0",
            "datasets": [
                {
                    "name": "test_baseline",
                    "equipment": "FD_FAN",
                    "input_file": "test_input.csv",
                    "expected_outputs": {
                        "mean_health": 95.0,
                        "episode_count": 0
                    },
                    "tolerance": {
                        "mean_health": 0.05,
                        "episode_count": 0.0
                    }
                }
            ]
        }
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        return tmp_path
    
    def test_run_regression_all_pass(self, temp_golden_dir):
        """Test regression with all tests passing."""
        harness = RegressionHarness(golden_dir=temp_golden_dir)
        
        def mock_pipeline(input_df: pd.DataFrame, equipment: str) -> Dict[str, float]:
            return {"mean_health": 95.0, "episode_count": 0}
        
        summary = harness.run_regression(pipeline_fn=mock_pipeline)
        
        assert summary.total_tests == 2
        assert summary.passed == 2
        assert summary.failed == 0
        assert summary.all_passed is True
    
    def test_run_regression_with_failure(self, temp_golden_dir):
        """Test regression with a failing test."""
        harness = RegressionHarness(golden_dir=temp_golden_dir)
        
        def mock_pipeline(input_df: pd.DataFrame, equipment: str) -> Dict[str, float]:
            return {"mean_health": 80.0, "episode_count": 0}  # 15% off
        
        summary = harness.run_regression(pipeline_fn=mock_pipeline)
        
        assert summary.failed == 1  # mean_health failed
        assert summary.passed == 1  # episode_count passed
        assert not summary.all_passed
    
    def test_run_regression_missing_metric(self, temp_golden_dir):
        """Test regression when pipeline is missing a metric."""
        harness = RegressionHarness(golden_dir=temp_golden_dir)
        
        def mock_pipeline(input_df: pd.DataFrame, equipment: str) -> Dict[str, float]:
            return {"mean_health": 95.0}  # Missing episode_count
        
        summary = harness.run_regression(pipeline_fn=mock_pipeline)
        
        # Missing metric should fail (NaN actual value)
        assert summary.failed >= 1
    
    def test_run_regression_specific_dataset(self, temp_golden_dir):
        """Test running regression on specific dataset."""
        harness = RegressionHarness(golden_dir=temp_golden_dir)
        
        def mock_pipeline(input_df: pd.DataFrame, equipment: str) -> Dict[str, float]:
            return {"mean_health": 95.0, "episode_count": 0}
        
        summary = harness.run_regression(
            pipeline_fn=mock_pipeline,
            dataset_name="test_baseline"
        )
        
        assert summary.total_tests == 2
        assert summary.all_passed
    
    def test_run_regression_unknown_dataset(self, temp_golden_dir):
        """Test running regression on unknown dataset."""
        harness = RegressionHarness(golden_dir=temp_golden_dir)
        
        def mock_pipeline(input_df: pd.DataFrame, equipment: str) -> Dict[str, float]:
            return {}
        
        summary = harness.run_regression(
            pipeline_fn=mock_pipeline,
            dataset_name="unknown_dataset"
        )
        
        assert summary.errors == 1
        assert "not found" in summary.results[0].error_message
    
    def test_run_regression_pipeline_error(self, temp_golden_dir):
        """Test handling of pipeline errors."""
        harness = RegressionHarness(golden_dir=temp_golden_dir)
        
        def failing_pipeline(input_df: pd.DataFrame, equipment: str) -> Dict[str, float]:
            raise ValueError("Pipeline failed")
        
        summary = harness.run_regression(
            pipeline_fn=failing_pipeline,
            continue_on_error=True
        )
        
        assert summary.errors >= 1


# =============================================================================
# TEST: RegressionHarness Tolerance Calculations
# =============================================================================


class TestRegressionHarnessTolerance:
    """Tests for tolerance calculations."""
    
    def test_within_tolerance(self):
        """Test value within tolerance passes."""
        harness = RegressionHarness(golden_dir=None)
        
        # Manually compare values
        result = harness._compare_values(
            test_name="test/metric",
            metric_name="metric",
            expected=100.0,
            actual=104.0,  # 4% deviation
            tolerance=0.05  # 5% tolerance
        )
        
        assert result.passed is True
        assert result.deviation_pct == pytest.approx(4.0, rel=0.01)
    
    def test_outside_tolerance(self):
        """Test value outside tolerance fails."""
        harness = RegressionHarness(golden_dir=None)
        
        result = harness._compare_values(
            test_name="test/metric",
            metric_name="metric",
            expected=100.0,
            actual=110.0,  # 10% deviation
            tolerance=0.05  # 5% tolerance
        )
        
        assert result.passed is False
        assert result.deviation_pct == pytest.approx(10.0, rel=0.01)
    
    def test_zero_expected_value(self):
        """Test handling of zero expected value."""
        harness = RegressionHarness(golden_dir=None)
        
        result = harness._compare_values(
            test_name="test/metric",
            metric_name="metric",
            expected=0.0,
            actual=0.01,  # Small deviation
            tolerance=0.05
        )
        
        # With expected=0, deviation = |actual| = 0.01
        assert result.deviation_pct == pytest.approx(1.0, rel=0.01)
    
    def test_nan_actual_value(self):
        """Test handling of NaN actual value."""
        harness = RegressionHarness(golden_dir=None)
        
        result = harness._compare_values(
            test_name="test/metric",
            metric_name="metric",
            expected=100.0,
            actual=np.nan,
            tolerance=0.05
        )
        
        assert result.passed is False
        assert result.status == RegressionStatus.FAILED
        assert "NaN" in result.error_message


# =============================================================================
# TEST: RegressionHarness.create_golden_dataset()
# =============================================================================


class TestRegressionHarnessCreateDataset:
    """Tests for create_golden_dataset method."""
    
    def test_create_golden_dataset(self, tmp_path):
        """Test creating a new golden dataset."""
        harness = RegressionHarness(golden_dir=tmp_path)
        
        input_df = pd.DataFrame({
            "Timestamp": ["2024-01-15 10:00:00"],
            "Sensor1": [100.0]
        })
        
        ds = harness.create_golden_dataset(
            name="new_dataset",
            equipment="GAS_TURBINE",
            input_df=input_df,
            outputs={"health": 90.0, "episodes": 1},
            tolerance={"health": 0.1}
        )
        
        assert ds.name == "new_dataset"
        assert ds.equipment == "GAS_TURBINE"
        assert "new_dataset" in harness.datasets
        
        # Check input file was created
        assert (tmp_path / "new_dataset_input.csv").exists()
        
        # Check manifest was created
        assert (tmp_path / "manifest.json").exists()
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)
        assert len(manifest["datasets"]) == 1
    
    def test_create_without_golden_dir_raises(self):
        """Test creating dataset without golden_dir raises error."""
        harness = RegressionHarness(golden_dir=None)
        
        with pytest.raises(ValueError, match="golden_dir must be set"):
            harness.create_golden_dataset(
                name="test",
                equipment="FD_FAN",
                input_df=pd.DataFrame(),
                outputs={}
            )


# =============================================================================
# TEST: RegressionHarness Reporting
# =============================================================================


class TestRegressionHarnessReporting:
    """Tests for reporting methods."""
    
    def test_get_failed_tests(self):
        """Test get_failed_tests method."""
        harness = RegressionHarness(golden_dir=None)
        
        results = [
            RegressionResult("a/m1", True, "m1", 100, 100, 0.05, 0, RegressionStatus.PASSED),
            RegressionResult("a/m2", False, "m2", 100, 120, 0.05, 20, RegressionStatus.FAILED),
            RegressionResult("a/m3", False, "m3", 100, 115, 0.05, 15, RegressionStatus.FAILED),
        ]
        summary = RegressionSummary(
            total_tests=3, passed=1, failed=2, results=results
        )
        
        failed = harness.get_failed_tests(summary)
        assert len(failed) == 2
        assert all(f.status == RegressionStatus.FAILED for f in failed)
    
    def test_format_report(self):
        """Test format_report method."""
        harness = RegressionHarness(golden_dir=None)
        
        results = [
            RegressionResult("ds/m1", True, "m1", 100, 100, 0.05, 0, RegressionStatus.PASSED),
            RegressionResult("ds/m2", False, "m2", 100, 120, 0.05, 20, RegressionStatus.FAILED),
        ]
        summary = RegressionSummary(
            total_tests=2, passed=1, failed=1, results=results
        )
        
        report = harness.format_report(summary)
        
        assert "REGRESSION TEST REPORT" in report
        assert "Total Tests: 2" in report
        assert "Passed: 1" in report
        assert "Failed: 1" in report
        assert "FAILURES:" in report


# =============================================================================
# TEST: Integration - Full Workflow
# =============================================================================


class TestRegressionHarnessIntegration:
    """Integration tests for full workflow."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow: create, load, run."""
        # Step 1: Create initial harness and golden dataset
        harness1 = RegressionHarness(golden_dir=tmp_path)
        
        input_df = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "Value": list(range(10))
        })
        
        harness1.create_golden_dataset(
            name="workflow_test",
            equipment="TEST_EQUIP",
            input_df=input_df,
            outputs={"sum_value": 45.0, "mean_value": 4.5},
            tolerance={"sum_value": 0.01, "mean_value": 0.01}
        )
        
        # Step 2: Create new harness and load datasets
        harness2 = RegressionHarness(golden_dir=tmp_path)
        assert "workflow_test" in harness2.datasets
        
        # Step 3: Run regression
        def simple_pipeline(df: pd.DataFrame, equipment: str) -> Dict[str, float]:
            return {
                "sum_value": float(df["Value"].sum()),
                "mean_value": float(df["Value"].mean())
            }
        
        summary = harness2.run_regression(pipeline_fn=simple_pipeline)
        
        assert summary.all_passed
        assert summary.total_tests == 2
        assert summary.passed == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
