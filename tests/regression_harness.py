"""
Regression Harness for ACM v11.0.0 (P5.6)

Compare ACM behavior against golden reference datasets to detect
unintended behavioral changes during refactoring.

Key Features:
- GoldenDataset: Reference datasets with expected outputs
- RegressionHarness: Run and compare against golden references
- RegressionResult: Detailed pass/fail results with deviations
- Manifest-based dataset management

Usage:
    harness = RegressionHarness(golden_dir=Path("tests/golden_data"))
    
    # Run all regression tests
    results = harness.run_regression(pipeline_fn=my_acm_pipeline)
    
    # Check results
    passed = all(r.passed for r in results)
    
    # Create new golden dataset
    harness.create_golden_dataset(
        name="fd_fan_baseline",
        equipment="FD_FAN",
        input_df=df,
        outputs={"mean_health_pct": 95.2, "episode_count": 0}
    )
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum
import json
import numpy as np
import pandas as pd
from datetime import datetime


class RegressionStatus(Enum):
    """Status of a regression test."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class RegressionResult:
    """Result of comparing current vs golden output.
    
    Attributes:
        test_name: Full test name (dataset/metric)
        passed: Whether the test passed
        metric_name: Name of the metric being compared
        golden_value: Expected value from golden dataset
        current_value: Actual value from current run
        tolerance: Allowed tolerance (0-1 for percentage)
        deviation_pct: Actual deviation as percentage
        status: Detailed status (PASSED, FAILED, SKIPPED, ERROR)
        error_message: Error message if status is ERROR
    """
    test_name: str
    passed: bool
    metric_name: str
    golden_value: float
    current_value: float
    tolerance: float
    deviation_pct: float
    status: RegressionStatus = RegressionStatus.PASSED
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "status": self.status.value,
            "metric_name": self.metric_name,
            "golden_value": round(self.golden_value, 4) if not np.isnan(self.golden_value) else None,
            "current_value": round(self.current_value, 4) if not np.isnan(self.current_value) else None,
            "tolerance_pct": round(self.tolerance * 100, 2),
            "deviation_pct": round(self.deviation_pct, 2),
            "error_message": self.error_message
        }
    
    def __repr__(self) -> str:
        status_icon = "PASS" if self.passed else "FAIL"
        return (
            f"[{status_icon}] {self.test_name}: "
            f"golden={self.golden_value:.3f}, "
            f"current={self.current_value:.3f}, "
            f"deviation={self.deviation_pct:.1f}%"
        )


@dataclass
class GoldenDataset:
    """Reference dataset for regression testing.
    
    Attributes:
        name: Unique dataset name
        equipment: Equipment type (FD_FAN, GAS_TURBINE, etc.)
        input_file: Path to input CSV file
        expected_outputs: Dictionary of metric names to expected values
        tolerance: Dictionary of metric names to tolerances (0-1)
        metadata: Optional metadata about the dataset
    """
    name: str
    equipment: str
    input_file: Path
    expected_outputs: Dict[str, float]
    tolerance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_tolerance(self, metric_name: str, default: float = 0.05) -> float:
        """Get tolerance for a metric, with default fallback."""
        return self.tolerance.get(metric_name, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for manifest."""
        return {
            "name": self.name,
            "equipment": self.equipment,
            "input_file": str(self.input_file.name) if isinstance(self.input_file, Path) else str(self.input_file),
            "expected_outputs": self.expected_outputs,
            "tolerance": self.tolerance,
            "metadata": self.metadata
        }


@dataclass
class RegressionSummary:
    """Summary of regression test run.
    
    Attributes:
        total_tests: Total number of tests run
        passed: Number of passed tests
        failed: Number of failed tests
        skipped: Number of skipped tests
        errors: Number of tests with errors
        results: List of all test results
        run_timestamp: When the tests were run
    """
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    results: List[RegressionResult] = field(default_factory=list)
    run_timestamp: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_tests == 0:
            return 100.0
        return (self.passed / self.total_tests) * 100
    
    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0 and self.errors == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "success_rate_pct": round(self.success_rate, 1),
            "all_passed": self.all_passed,
            "run_timestamp": str(self.run_timestamp) if self.run_timestamp else None,
            "results": [r.to_dict() for r in self.results]
        }
    
    def __repr__(self) -> str:
        return (
            f"RegressionSummary(total={self.total_tests}, "
            f"passed={self.passed}, failed={self.failed}, "
            f"success_rate={self.success_rate:.1f}%)"
        )


class RegressionHarness:
    """
    Compare ACM behavior against golden reference datasets.
    
    Detects unintended behavioral changes during refactoring by comparing
    current outputs against pre-recorded golden datasets.
    
    Usage:
        harness = RegressionHarness(golden_dir=Path("tests/golden_data"))
        
        # Load datasets from manifest
        harness.load_datasets()
        
        # Run regression tests
        summary = harness.run_regression(pipeline_fn=my_pipeline)
        
        if not summary.all_passed:
            print(f"Regression failures: {summary.failed}")
    
    Attributes:
        golden_dir: Directory containing golden datasets
        datasets: Dictionary of loaded golden datasets
        default_tolerance: Default tolerance for metrics without explicit tolerance
    """
    
    def __init__(
        self,
        golden_dir: Optional[Path] = None,
        default_tolerance: float = 0.05
    ):
        """
        Initialize RegressionHarness.
        
        Args:
            golden_dir: Directory containing golden datasets and manifest.json.
                       If None, no datasets are loaded.
            default_tolerance: Default tolerance (0-1) for metrics without
                              explicit tolerance (default 5%)
        """
        self.golden_dir = golden_dir
        self.default_tolerance = default_tolerance
        self.datasets: Dict[str, GoldenDataset] = {}
        
        if golden_dir is not None:
            self.load_datasets()
    
    def load_datasets(self) -> None:
        """Load all golden datasets from manifest.json."""
        if self.golden_dir is None:
            return
        
        manifest_path = self.golden_dir / "manifest.json"
        if not manifest_path.exists():
            return
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        for ds_config in manifest.get("datasets", []):
            ds = GoldenDataset(
                name=ds_config["name"],
                equipment=ds_config["equipment"],
                input_file=self.golden_dir / ds_config["input_file"],
                expected_outputs=ds_config.get("expected_outputs", {}),
                tolerance=ds_config.get("tolerance", {}),
                metadata=ds_config.get("metadata", {})
            )
            self.datasets[ds.name] = ds
    
    def add_dataset(self, dataset: GoldenDataset) -> None:
        """Add a golden dataset programmatically."""
        self.datasets[dataset.name] = dataset
    
    def run_regression(
        self,
        pipeline_fn: Callable[[pd.DataFrame, str], Dict[str, float]],
        dataset_name: Optional[str] = None,
        continue_on_error: bool = True
    ) -> RegressionSummary:
        """
        Run regression tests against golden datasets.
        
        Args:
            pipeline_fn: Function that takes (input_df, equipment) and returns
                        a dictionary of metric names to values.
            dataset_name: Specific dataset to test, or None for all datasets
            continue_on_error: Whether to continue on errors (default True)
        
        Returns:
            RegressionSummary with all test results
        """
        summary = RegressionSummary(run_timestamp=datetime.now())
        
        if dataset_name is not None:
            if dataset_name not in self.datasets:
                summary.errors = 1
                summary.total_tests = 1
                summary.results.append(RegressionResult(
                    test_name=f"{dataset_name}/unknown",
                    passed=False,
                    metric_name="unknown",
                    golden_value=np.nan,
                    current_value=np.nan,
                    tolerance=0.0,
                    deviation_pct=0.0,
                    status=RegressionStatus.ERROR,
                    error_message=f"Dataset '{dataset_name}' not found"
                ))
                return summary
            datasets_to_test = [self.datasets[dataset_name]]
        else:
            datasets_to_test = list(self.datasets.values())
        
        for ds in datasets_to_test:
            results = self._run_single_dataset(ds, pipeline_fn, continue_on_error)
            summary.results.extend(results)
        
        # Compute summary statistics
        summary.total_tests = len(summary.results)
        summary.passed = sum(1 for r in summary.results if r.status == RegressionStatus.PASSED)
        summary.failed = sum(1 for r in summary.results if r.status == RegressionStatus.FAILED)
        summary.skipped = sum(1 for r in summary.results if r.status == RegressionStatus.SKIPPED)
        summary.errors = sum(1 for r in summary.results if r.status == RegressionStatus.ERROR)
        
        return summary
    
    def _run_single_dataset(
        self,
        ds: GoldenDataset,
        pipeline_fn: Callable[[pd.DataFrame, str], Dict[str, float]],
        continue_on_error: bool
    ) -> List[RegressionResult]:
        """Run regression test for a single dataset."""
        results = []
        
        # Load input data
        try:
            if not ds.input_file.exists():
                return [RegressionResult(
                    test_name=f"{ds.name}/load",
                    passed=False,
                    metric_name="load",
                    golden_value=np.nan,
                    current_value=np.nan,
                    tolerance=0.0,
                    deviation_pct=0.0,
                    status=RegressionStatus.ERROR,
                    error_message=f"Input file not found: {ds.input_file}"
                )]
            
            input_df = pd.read_csv(ds.input_file)
        except Exception as e:
            return [RegressionResult(
                test_name=f"{ds.name}/load",
                passed=False,
                metric_name="load",
                golden_value=np.nan,
                current_value=np.nan,
                tolerance=0.0,
                deviation_pct=0.0,
                status=RegressionStatus.ERROR,
                error_message=f"Failed to load input: {e}"
            )]
        
        # Run pipeline
        try:
            outputs = pipeline_fn(input_df, ds.equipment)
        except Exception as e:
            if not continue_on_error:
                raise
            return [RegressionResult(
                test_name=f"{ds.name}/pipeline",
                passed=False,
                metric_name="pipeline",
                golden_value=np.nan,
                current_value=np.nan,
                tolerance=0.0,
                deviation_pct=0.0,
                status=RegressionStatus.ERROR,
                error_message=f"Pipeline error: {e}"
            )]
        
        # Compare each expected output
        for metric_name, expected_value in ds.expected_outputs.items():
            actual_value = outputs.get(metric_name, np.nan)
            tolerance = ds.get_tolerance(metric_name, self.default_tolerance)
            
            result = self._compare_values(
                test_name=f"{ds.name}/{metric_name}",
                metric_name=metric_name,
                expected=expected_value,
                actual=actual_value,
                tolerance=tolerance
            )
            results.append(result)
        
        return results
    
    def _compare_values(
        self,
        test_name: str,
        metric_name: str,
        expected: float,
        actual: float,
        tolerance: float
    ) -> RegressionResult:
        """Compare expected vs actual value and compute deviation."""
        # Handle NaN
        if np.isnan(actual):
            return RegressionResult(
                test_name=test_name,
                passed=False,
                metric_name=metric_name,
                golden_value=expected,
                current_value=actual,
                tolerance=tolerance,
                deviation_pct=100.0,
                status=RegressionStatus.FAILED,
                error_message="Actual value is NaN"
            )
        
        # Compute deviation
        if expected == 0:
            deviation = abs(actual)
        else:
            deviation = abs(actual - expected) / abs(expected)
        
        passed = deviation <= tolerance
        
        return RegressionResult(
            test_name=test_name,
            passed=passed,
            metric_name=metric_name,
            golden_value=expected,
            current_value=actual,
            tolerance=tolerance,
            deviation_pct=deviation * 100,
            status=RegressionStatus.PASSED if passed else RegressionStatus.FAILED
        )
    
    def create_golden_dataset(
        self,
        name: str,
        equipment: str,
        input_df: pd.DataFrame,
        outputs: Dict[str, float],
        tolerance: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GoldenDataset:
        """
        Create a new golden dataset from current run.
        
        Saves input data to CSV and updates manifest.json.
        
        Args:
            name: Unique dataset name
            equipment: Equipment type
            input_df: Input DataFrame to save
            outputs: Dictionary of metric names to expected values
            tolerance: Optional tolerance overrides
            metadata: Optional metadata
        
        Returns:
            Created GoldenDataset
        """
        if self.golden_dir is None:
            raise ValueError("golden_dir must be set to create datasets")
        
        # Ensure directory exists
        self.golden_dir.mkdir(parents=True, exist_ok=True)
        
        # Save input data
        input_filename = f"{name}_input.csv"
        input_path = self.golden_dir / input_filename
        input_df.to_csv(input_path, index=False)
        
        # Create dataset
        ds = GoldenDataset(
            name=name,
            equipment=equipment,
            input_file=input_path,
            expected_outputs=outputs,
            tolerance=tolerance or {},
            metadata=metadata or {"created_at": str(datetime.now())}
        )
        
        # Add to collection
        self.datasets[name] = ds
        
        # Update manifest
        self._save_manifest()
        
        return ds
    
    def update_golden_outputs(
        self,
        dataset_name: str,
        outputs: Dict[str, float]
    ) -> None:
        """Update expected outputs for an existing golden dataset."""
        if dataset_name not in self.datasets:
            raise KeyError(f"Dataset '{dataset_name}' not found")
        
        self.datasets[dataset_name].expected_outputs.update(outputs)
        self._save_manifest()
    
    def _save_manifest(self) -> None:
        """Save current datasets to manifest.json."""
        if self.golden_dir is None:
            return
        
        manifest = {
            "version": "1.0",
            "updated_at": str(datetime.now()),
            "datasets": [ds.to_dict() for ds in self.datasets.values()]
        }
        
        manifest_path = self.golden_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    
    def get_failed_tests(self, summary: RegressionSummary) -> List[RegressionResult]:
        """Get list of failed tests from summary."""
        return [r for r in summary.results if r.status == RegressionStatus.FAILED]
    
    def format_report(self, summary: RegressionSummary) -> str:
        """Format summary as human-readable report."""
        lines = [
            "=" * 60,
            "REGRESSION TEST REPORT",
            "=" * 60,
            f"Run Time: {summary.run_timestamp}",
            f"Total Tests: {summary.total_tests}",
            f"Passed: {summary.passed}",
            f"Failed: {summary.failed}",
            f"Errors: {summary.errors}",
            f"Success Rate: {summary.success_rate:.1f}%",
            "-" * 60,
        ]
        
        if summary.failed > 0:
            lines.append("FAILURES:")
            for r in summary.results:
                if r.status == RegressionStatus.FAILED:
                    lines.append(f"  {r}")
        
        if summary.errors > 0:
            lines.append("ERRORS:")
            for r in summary.results:
                if r.status == RegressionStatus.ERROR:
                    lines.append(f"  {r.test_name}: {r.error_message}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
