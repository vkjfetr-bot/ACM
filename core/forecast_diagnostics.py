"""
Forecast Diagnostics - metrics for forecast quality assessment.

v11.0.0 - Phase 4.5 Implementation

This module provides diagnostic metrics to assess the quality of
ACM forecasts (health, sensor, RUL). Key metrics include:

- Coverage: % of actual values within prediction intervals
- Sharpness: width of prediction intervals (narrower = better)
- Calibration: how well probabilities match actual frequencies
- Bias: systematic over/under prediction
- Skill Score: improvement over naive baseline
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np


@dataclass
class ForecastDiagnostics:
    """
    Diagnostic metrics for forecast quality assessment.
    
    These metrics allow operators and data scientists to understand
    how well ACM forecasts are performing.
    """
    
    # Coverage: % of actual values within prediction interval
    coverage_p80: float  # % within 10-90 percentile bounds (should be ~80%)
    coverage_p50: float  # % within 25-75 percentile bounds (should be ~50%)
    
    # Sharpness: width of prediction intervals (narrower = better, if accurate)
    sharpness_p80: float  # Average width of 80% interval (p90 - p10)
    sharpness_p50: float  # Average width of 50% interval (p75 - p25)
    
    # Calibration: how well probabilities match actual frequencies
    # 1.0 = perfectly calibrated, lower = poorly calibrated
    calibration_score: float
    
    # Bias: systematic over/under prediction
    # Positive = overpredicting, negative = underpredicting
    mean_bias: float
    
    # Skill score: improvement over naive baseline
    # 0 = same as baseline, 1 = perfect, negative = worse than baseline
    skill_score: float
    
    # Fields with defaults must come last
    median_bias: float = 0.0
    
    # MAE and RMSE for p50 predictions
    mae: float = 0.0
    rmse: float = 0.0
    
    # Count information
    n_forecasts: int = 0       # Total forecasts made
    n_validated: int = 0       # Forecasts with ground truth available
    
    @property
    def is_well_calibrated(self) -> bool:
        """Check if forecast is well-calibrated (coverage within expected range)."""
        if np.isnan(self.calibration_score):
            return False
        return self.calibration_score >= 0.7
    
    @property
    def validation_ratio(self) -> float:
        """Ratio of validated forecasts to total forecasts."""
        if self.n_forecasts == 0:
            return 0.0
        return self.n_validated / self.n_forecasts
    
    @property
    def has_skill(self) -> bool:
        """Check if forecast beats naive baseline."""
        if np.isnan(self.skill_score):
            return False
        return self.skill_score > 0.05  # 5% improvement threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SQL writes."""
        return {
            "CoverageP80": self.coverage_p80 if not np.isnan(self.coverage_p80) else None,
            "CoverageP50": self.coverage_p50 if not np.isnan(self.coverage_p50) else None,
            "SharpnessP80": self.sharpness_p80 if not np.isnan(self.sharpness_p80) else None,
            "SharpnessP50": self.sharpness_p50 if not np.isnan(self.sharpness_p50) else None,
            "CalibrationScore": self.calibration_score if not np.isnan(self.calibration_score) else None,
            "MeanBias": self.mean_bias if not np.isnan(self.mean_bias) else None,
            "MedianBias": self.median_bias if not np.isnan(self.median_bias) else None,
            "SkillScore": self.skill_score if not np.isnan(self.skill_score) else None,
            "MAE": self.mae if not np.isnan(self.mae) else None,
            "RMSE": self.rmse if not np.isnan(self.rmse) else None,
            "NForecastsMade": self.n_forecasts,
            "NForecastsValidated": self.n_validated,
        }
    
    @classmethod
    def empty(cls, n_forecasts: int = 0) -> "ForecastDiagnostics":
        """Create empty diagnostics (no validation data available)."""
        return cls(
            coverage_p80=np.nan,
            coverage_p50=np.nan,
            sharpness_p80=np.nan,
            sharpness_p50=np.nan,
            calibration_score=np.nan,
            mean_bias=np.nan,
            median_bias=np.nan,
            skill_score=np.nan,
            mae=np.nan,
            rmse=np.nan,
            n_forecasts=n_forecasts,
            n_validated=0,
        )
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.n_validated == 0:
            return f"No validated forecasts (made {self.n_forecasts} forecasts)"
        
        lines = [
            f"Forecast Diagnostics ({self.n_validated}/{self.n_forecasts} validated)",
            f"  Coverage P80: {self.coverage_p80:.1%} (target: 80%)",
            f"  Coverage P50: {self.coverage_p50:.1%} (target: 50%)",
            f"  Calibration:  {self.calibration_score:.2f}",
            f"  Skill Score:  {self.skill_score:.2f}",
            f"  Mean Bias:    {self.mean_bias:+.3f}",
            f"  MAE:          {self.mae:.3f}",
        ]
        return "\n".join(lines)


class ForecastValidator:
    """
    Validate forecast quality with diagnostic metrics.
    
    Compares forecasted values with actual observed values to compute
    coverage, calibration, sharpness, bias, and skill metrics.
    
    Usage:
        validator = ForecastValidator()
        diagnostics = validator.compute_diagnostics(forecasts_df, actuals_df)
        
        if not diagnostics.is_well_calibrated:
            print("Warning: Forecasts are not well calibrated")
    """
    
    def __init__(
        self,
        timestamp_col: str = "timestamp",
        actual_col: str = "actual_value",
    ):
        """
        Initialize validator.
        
        Args:
            timestamp_col: Column name for timestamps in both DataFrames
            actual_col: Column name for actual values in actuals DataFrame
        """
        self.timestamp_col = timestamp_col
        self.actual_col = actual_col
    
    def compute_diagnostics(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> ForecastDiagnostics:
        """
        Compute forecast diagnostics by comparing predictions to actuals.
        
        Args:
            forecasts: DataFrame with columns [timestamp, p10, p25, p50, p75, p90]
            actuals: DataFrame with columns [timestamp, actual_value]
        
        Returns:
            ForecastDiagnostics with all computed metrics
        """
        if forecasts.empty:
            return ForecastDiagnostics.empty()
        
        # Validate required columns
        required_forecast_cols = {"p10", "p25", "p50", "p75", "p90"}
        if not required_forecast_cols.issubset(forecasts.columns):
            missing = required_forecast_cols - set(forecasts.columns)
            raise ValueError(f"Forecasts missing columns: {missing}")
        
        if self.timestamp_col not in forecasts.columns:
            raise ValueError(f"Forecasts missing timestamp column: {self.timestamp_col}")
        
        if actuals.empty or self.actual_col not in actuals.columns:
            return ForecastDiagnostics.empty(n_forecasts=len(forecasts))
        
        # Merge on timestamp
        merged = pd.merge(
            forecasts,
            actuals[[self.timestamp_col, self.actual_col]],
            on=self.timestamp_col,
            how="inner"
        )
        
        if merged.empty:
            return ForecastDiagnostics.empty(n_forecasts=len(forecasts))
        
        # Extract values
        p10 = merged["p10"].values
        p25 = merged["p25"].values
        p50 = merged["p50"].values
        p75 = merged["p75"].values
        p90 = merged["p90"].values
        actual = merged[self.actual_col].values
        
        # --- Coverage: % within bounds ---
        in_p80 = (actual >= p10) & (actual <= p90)
        in_p50 = (actual >= p25) & (actual <= p75)
        coverage_p80 = np.mean(in_p80)
        coverage_p50 = np.mean(in_p50)
        
        # --- Sharpness: interval width ---
        sharpness_p80 = np.mean(p90 - p10)
        sharpness_p50 = np.mean(p75 - p25)
        
        # --- Calibration score ---
        # 80% interval should contain ~80% of points
        # Score = 1 - |actual_coverage - expected_coverage|
        calibration_p80 = 1.0 - abs(coverage_p80 - 0.80)
        calibration_p50 = 1.0 - abs(coverage_p50 - 0.50)
        calibration_score = (calibration_p80 + calibration_p50) / 2
        
        # --- Bias ---
        residuals = p50 - actual
        mean_bias = np.mean(residuals)
        median_bias = np.median(residuals)
        
        # --- Error metrics ---
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # --- Skill score vs naive (persistence) baseline ---
        # Naive forecast: next value = current value
        if len(actual) >= 2:
            naive_forecast = actual[:-1]  # Shifted by 1
            naive_actual = actual[1:]
            naive_error = np.mean(np.abs(naive_actual - naive_forecast))
            
            # Our forecast error
            forecast_error = mae
            
            # Skill = 1 - (our_error / naive_error)
            if naive_error > 1e-10:
                skill_score = 1.0 - (forecast_error / naive_error)
            else:
                skill_score = 0.0  # Perfect naive = no skill measurable
        else:
            skill_score = np.nan
        
        return ForecastDiagnostics(
            coverage_p80=coverage_p80,
            coverage_p50=coverage_p50,
            sharpness_p80=sharpness_p80,
            sharpness_p50=sharpness_p50,
            calibration_score=calibration_score,
            mean_bias=mean_bias,
            median_bias=median_bias,
            skill_score=skill_score,
            mae=mae,
            rmse=rmse,
            n_forecasts=len(forecasts),
            n_validated=len(merged),
        )
    
    def compute_per_horizon(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        horizon_col: str = "horizon_hours",
        horizons: Optional[List[float]] = None,
    ) -> Dict[float, ForecastDiagnostics]:
        """
        Compute diagnostics broken down by forecast horizon.
        
        Args:
            forecasts: DataFrame with forecast data and horizon column
            actuals: DataFrame with actual values
            horizon_col: Column indicating forecast horizon
            horizons: Specific horizons to compute (None = all unique horizons)
        
        Returns:
            Dict mapping horizon to ForecastDiagnostics
        """
        if horizon_col not in forecasts.columns:
            raise ValueError(f"Forecasts missing horizon column: {horizon_col}")
        
        if horizons is None:
            horizons = sorted(forecasts[horizon_col].unique())
        
        results = {}
        for h in horizons:
            forecast_subset = forecasts[forecasts[horizon_col] == h]
            results[h] = self.compute_diagnostics(forecast_subset, actuals)
        
        return results


class CalibrationAnalyzer:
    """
    Detailed calibration analysis for probabilistic forecasts.
    
    Provides reliability diagrams and calibration curves.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize analyzer.
        
        Args:
            n_bins: Number of bins for calibration analysis
        """
        self.n_bins = n_bins
    
    def compute_reliability_diagram(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        timestamp_col: str = "timestamp",
        actual_col: str = "actual_value",
    ) -> pd.DataFrame:
        """
        Compute reliability diagram data.
        
        Returns DataFrame with columns:
        - nominal_coverage: Expected coverage (0.1, 0.2, ..., 0.9)
        - actual_coverage: Observed coverage at that level
        - deviation: actual - nominal
        """
        # Early return for empty inputs
        if forecasts.empty or actuals.empty:
            return pd.DataFrame(columns=["nominal_coverage", "actual_coverage", "deviation"])
        
        # Check required columns exist
        if timestamp_col not in forecasts.columns or timestamp_col not in actuals.columns:
            return pd.DataFrame(columns=["nominal_coverage", "actual_coverage", "deviation"])
        if actual_col not in actuals.columns:
            return pd.DataFrame(columns=["nominal_coverage", "actual_coverage", "deviation"])
        
        # Merge forecasts and actuals
        merged = pd.merge(
            forecasts,
            actuals[[timestamp_col, actual_col]],
            on=timestamp_col,
            how="inner"
        )
        
        if merged.empty:
            return pd.DataFrame(columns=["nominal_coverage", "actual_coverage", "deviation"])
        
        # For each coverage level, compute actual coverage
        # We use the quantile columns if available
        results = []
        
        # Standard quantile pairs for coverage levels
        quantile_pairs = [
            (0.1, "p5", "p95"),   # If we had p5/p95
            (0.2, "p10", "p90"),  # 80% coverage
            (0.5, "p25", "p75"),  # 50% coverage
        ]
        
        # Check which columns exist
        if "p10" in merged.columns and "p90" in merged.columns:
            actual = merged[actual_col]
            in_80 = (actual >= merged["p10"]) & (actual <= merged["p90"])
            results.append({
                "nominal_coverage": 0.80,
                "actual_coverage": in_80.mean(),
                "deviation": in_80.mean() - 0.80
            })
        
        if "p25" in merged.columns and "p75" in merged.columns:
            actual = merged[actual_col]
            in_50 = (actual >= merged["p25"]) & (actual <= merged["p75"])
            results.append({
                "nominal_coverage": 0.50,
                "actual_coverage": in_50.mean(),
                "deviation": in_50.mean() - 0.50
            })
        
        return pd.DataFrame(results)
    
    def compute_pit_histogram(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        timestamp_col: str = "timestamp",
        actual_col: str = "actual_value",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Probability Integral Transform (PIT) histogram.
        
        For well-calibrated forecasts, PIT values should be uniform.
        
        Returns:
            (bin_centers, frequencies) for plotting histogram
        """
        merged = pd.merge(
            forecasts,
            actuals[[timestamp_col, actual_col]],
            on=timestamp_col,
            how="inner"
        )
        
        if merged.empty or len(merged) < 10:
            return np.array([]), np.array([])
        
        # Estimate PIT value for each point
        # PIT = CDF(actual) under forecast distribution
        # Approximate using quantiles
        pit_values = []
        
        for _, row in merged.iterrows():
            actual = row[actual_col]
            
            # Use linear interpolation between quantiles
            quantiles = [
                (0.10, row.get("p10", np.nan)),
                (0.25, row.get("p25", np.nan)),
                (0.50, row.get("p50", np.nan)),
                (0.75, row.get("p75", np.nan)),
                (0.90, row.get("p90", np.nan)),
            ]
            
            # Filter valid quantiles
            valid_q = [(p, v) for p, v in quantiles if not np.isnan(v)]
            if len(valid_q) < 2:
                continue
            
            probs, vals = zip(*valid_q)
            
            # Interpolate to find PIT value
            if actual <= min(vals):
                pit = min(probs)
            elif actual >= max(vals):
                pit = max(probs)
            else:
                pit = np.interp(actual, vals, probs)
            
            pit_values.append(pit)
        
        if len(pit_values) < 5:
            return np.array([]), np.array([])
        
        # Compute histogram
        bins = np.linspace(0, 1, self.n_bins + 1)
        hist, _ = np.histogram(pit_values, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        return bin_centers, hist


def compute_forecast_diagnostics_summary(
    health_forecasts: Optional[pd.DataFrame] = None,
    health_actuals: Optional[pd.DataFrame] = None,
    rul_forecasts: Optional[pd.DataFrame] = None,
    rul_actuals: Optional[pd.DataFrame] = None,
    sensor_forecasts: Optional[Dict[str, pd.DataFrame]] = None,
    sensor_actuals: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, ForecastDiagnostics]:
    """
    Compute diagnostics for all forecast types.
    
    Args:
        health_forecasts: Health forecasts DataFrame
        health_actuals: Health actuals DataFrame
        rul_forecasts: RUL forecasts DataFrame
        rul_actuals: RUL actuals DataFrame
        sensor_forecasts: Dict mapping sensor name to forecast DataFrame
        sensor_actuals: Dict mapping sensor name to actual DataFrame
    
    Returns:
        Dict mapping forecast type to ForecastDiagnostics
    """
    validator = ForecastValidator()
    results = {}
    
    # Health forecasts
    if health_forecasts is not None:
        actuals = health_actuals if health_actuals is not None else pd.DataFrame()
        results["health"] = validator.compute_diagnostics(health_forecasts, actuals)
    
    # RUL forecasts
    if rul_forecasts is not None:
        actuals = rul_actuals if rul_actuals is not None else pd.DataFrame()
        results["rul"] = validator.compute_diagnostics(rul_forecasts, actuals)
    
    # Sensor forecasts
    if sensor_forecasts is not None:
        for sensor_name, forecasts in sensor_forecasts.items():
            actuals = sensor_actuals.get(sensor_name, pd.DataFrame()) if sensor_actuals else pd.DataFrame()
            results[f"sensor_{sensor_name}"] = validator.compute_diagnostics(forecasts, actuals)
    
    return results
