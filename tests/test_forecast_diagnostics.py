"""
Tests for core/forecast_diagnostics.py - Forecast Diagnostics

v11.0.0 Phase 4.5 Tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.forecast_diagnostics import (
    ForecastDiagnostics,
    ForecastValidator,
    CalibrationAnalyzer,
    compute_forecast_diagnostics_summary,
)


# =============================================================================
# ForecastDiagnostics Tests
# =============================================================================

class TestForecastDiagnostics:
    """Tests for ForecastDiagnostics dataclass."""
    
    def test_well_calibrated(self):
        """Check is_well_calibrated property."""
        good = ForecastDiagnostics(
            coverage_p80=0.80,
            coverage_p50=0.50,
            sharpness_p80=10.0,
            sharpness_p50=5.0,
            calibration_score=0.85,
            mean_bias=0.0,
            skill_score=0.5,
            n_forecasts=100,
            n_validated=80,
        )
        assert good.is_well_calibrated is True
        
        poor = ForecastDiagnostics(
            coverage_p80=0.50,  # Way off target
            coverage_p50=0.30,
            sharpness_p80=10.0,
            sharpness_p50=5.0,
            calibration_score=0.5,  # Below 0.7
            mean_bias=5.0,
            skill_score=0.1,
            n_forecasts=100,
            n_validated=80,
        )
        assert poor.is_well_calibrated is False
    
    def test_validation_ratio(self):
        """Check validation_ratio calculation."""
        diag = ForecastDiagnostics(
            coverage_p80=0.80,
            coverage_p50=0.50,
            sharpness_p80=10.0,
            sharpness_p50=5.0,
            calibration_score=0.85,
            mean_bias=0.0,
            skill_score=0.5,
            n_forecasts=100,
            n_validated=75,
        )
        assert diag.validation_ratio == 0.75
    
    def test_validation_ratio_zero_forecasts(self):
        """validation_ratio handles zero forecasts."""
        diag = ForecastDiagnostics(
            coverage_p80=np.nan,
            coverage_p50=np.nan,
            sharpness_p80=np.nan,
            sharpness_p50=np.nan,
            calibration_score=np.nan,
            mean_bias=np.nan,
            skill_score=np.nan,
            n_forecasts=0,
            n_validated=0,
        )
        assert diag.validation_ratio == 0.0
    
    def test_has_skill(self):
        """Check has_skill property."""
        skilled = ForecastDiagnostics(
            coverage_p80=0.80,
            coverage_p50=0.50,
            sharpness_p80=10.0,
            sharpness_p50=5.0,
            calibration_score=0.85,
            mean_bias=0.0,
            skill_score=0.3,  # > 0.05
            n_forecasts=100,
            n_validated=80,
        )
        assert skilled.has_skill is True
        
        no_skill = ForecastDiagnostics(
            coverage_p80=0.80,
            coverage_p50=0.50,
            sharpness_p80=10.0,
            sharpness_p50=5.0,
            calibration_score=0.85,
            mean_bias=0.0,
            skill_score=0.01,  # < 0.05
            n_forecasts=100,
            n_validated=80,
        )
        assert no_skill.has_skill is False
    
    def test_to_dict(self):
        """Check to_dict conversion."""
        diag = ForecastDiagnostics(
            coverage_p80=0.82,
            coverage_p50=0.48,
            sharpness_p80=12.5,
            sharpness_p50=6.2,
            calibration_score=0.90,
            mean_bias=-0.5,
            median_bias=-0.3,
            skill_score=0.35,
            mae=2.1,
            rmse=3.5,
            n_forecasts=100,
            n_validated=95,
        )
        
        d = diag.to_dict()
        
        assert d["CoverageP80"] == 0.82
        assert d["CoverageP50"] == 0.48
        assert d["SharpnessP80"] == 12.5
        assert d["CalibrationScore"] == 0.90
        assert d["MeanBias"] == -0.5
        assert d["SkillScore"] == 0.35
        assert d["MAE"] == 2.1
        assert d["RMSE"] == 3.5
        assert d["NForecastsMade"] == 100
        assert d["NForecastsValidated"] == 95
    
    def test_to_dict_handles_nan(self):
        """to_dict converts NaN to None for SQL."""
        diag = ForecastDiagnostics.empty()
        d = diag.to_dict()
        
        assert d["CoverageP80"] is None
        assert d["CalibrationScore"] is None
        assert d["SkillScore"] is None
    
    def test_empty_factory(self):
        """empty() creates proper empty diagnostics."""
        diag = ForecastDiagnostics.empty(n_forecasts=50)
        
        assert np.isnan(diag.coverage_p80)
        assert np.isnan(diag.calibration_score)
        assert diag.n_forecasts == 50
        assert diag.n_validated == 0
    
    def test_summary(self):
        """summary() generates readable text."""
        diag = ForecastDiagnostics(
            coverage_p80=0.78,
            coverage_p50=0.52,
            sharpness_p80=10.0,
            sharpness_p50=5.0,
            calibration_score=0.88,
            mean_bias=0.1,
            skill_score=0.25,
            mae=1.5,
            rmse=2.0,
            n_forecasts=100,
            n_validated=90,
        )
        
        summary = diag.summary()
        assert "90/100" in summary
        assert "Coverage" in summary
        assert "Calibration" in summary


# =============================================================================
# ForecastValidator Tests
# =============================================================================

class TestForecastValidator:
    """Tests for ForecastValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Standard validator."""
        return ForecastValidator()
    
    @pytest.fixture
    def perfect_forecasts(self):
        """Forecasts that perfectly bracket actuals."""
        n = 100
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
        actuals = np.random.randn(n) * 10 + 50  # Mean 50, std 10
        
        forecasts = pd.DataFrame({
            "timestamp": timestamps,
            "p10": actuals - 15,  # Wide bounds
            "p25": actuals - 5,
            "p50": actuals,  # Perfect median
            "p75": actuals + 5,
            "p90": actuals + 15,
        })
        
        actuals_df = pd.DataFrame({
            "timestamp": timestamps,
            "actual_value": actuals,
        })
        
        return forecasts, actuals_df
    
    @pytest.fixture
    def biased_forecasts(self):
        """Forecasts with positive bias."""
        n = 100
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
        actuals = np.random.randn(n) * 10 + 50
        
        # Predictions consistently 5 units too high
        forecasts = pd.DataFrame({
            "timestamp": timestamps,
            "p10": actuals - 10,
            "p25": actuals,
            "p50": actuals + 5,  # Biased high
            "p75": actuals + 10,
            "p90": actuals + 20,
        })
        
        actuals_df = pd.DataFrame({
            "timestamp": timestamps,
            "actual_value": actuals,
        })
        
        return forecasts, actuals_df
    
    def test_perfect_forecasts(self, validator, perfect_forecasts):
        """Perfect forecasts have excellent metrics."""
        forecasts, actuals = perfect_forecasts
        diag = validator.compute_diagnostics(forecasts, actuals)
        
        # Coverage should be very high
        assert diag.coverage_p80 >= 0.95  # All points within wide bounds
        assert diag.coverage_p50 >= 0.8
        
        # Bias should be near zero
        assert abs(diag.mean_bias) < 1.0
        
        # MAE should be very low
        assert diag.mae < 1.0
        
        # Skill should be positive
        assert diag.skill_score > 0
    
    def test_biased_forecasts(self, validator, biased_forecasts):
        """Biased forecasts show positive mean bias."""
        forecasts, actuals = biased_forecasts
        diag = validator.compute_diagnostics(forecasts, actuals)
        
        # Should detect positive bias
        assert diag.mean_bias > 3.0  # We added 5, should detect most of it
    
    def test_empty_forecasts(self, validator):
        """Empty forecasts return empty diagnostics."""
        diag = validator.compute_diagnostics(pd.DataFrame(), pd.DataFrame())
        
        assert np.isnan(diag.coverage_p80)
        assert diag.n_forecasts == 0
        assert diag.n_validated == 0
    
    def test_no_matching_actuals(self, validator):
        """Forecasts with no matching actuals return empty diagnostics."""
        forecasts = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "p10": range(10),
            "p25": range(10),
            "p50": range(10),
            "p75": range(10),
            "p90": range(10),
        })
        
        # Actuals from different time range
        actuals = pd.DataFrame({
            "timestamp": pd.date_range("2024-06-01", periods=10, freq="1h"),
            "actual_value": range(10),
        })
        
        diag = validator.compute_diagnostics(forecasts, actuals)
        
        assert diag.n_forecasts == 10
        assert diag.n_validated == 0
    
    def test_partial_overlap(self, validator):
        """Handles partial overlap between forecasts and actuals."""
        timestamps = pd.date_range("2024-01-01", periods=100, freq="1h")
        
        forecasts = pd.DataFrame({
            "timestamp": timestamps,
            "p10": np.zeros(100),
            "p25": np.ones(100) * 25,
            "p50": np.ones(100) * 50,
            "p75": np.ones(100) * 75,
            "p90": np.ones(100) * 100,
        })
        
        # Only first 50 have actuals
        actuals = pd.DataFrame({
            "timestamp": timestamps[:50],
            "actual_value": np.ones(50) * 50,
        })
        
        diag = validator.compute_diagnostics(forecasts, actuals)
        
        assert diag.n_forecasts == 100
        assert diag.n_validated == 50
    
    def test_coverage_calculation(self, validator):
        """Verify coverage calculation."""
        n = 100
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
        
        # Set up so exactly 80% are in p80 range and 50% in p50 range
        actuals = np.array([i for i in range(n)])
        
        forecasts = pd.DataFrame({
            "timestamp": timestamps,
            "p10": np.where(actuals < 80, actuals - 1, actuals + 100),  # 80 in bounds
            "p25": np.where(actuals < 50, actuals - 1, actuals + 100),  # 50 in bounds
            "p50": actuals,
            "p75": np.where(actuals < 50, actuals + 1, actuals - 100),
            "p90": np.where(actuals < 80, actuals + 1, actuals - 100),
        })
        
        actuals_df = pd.DataFrame({
            "timestamp": timestamps,
            "actual_value": actuals,
        })
        
        diag = validator.compute_diagnostics(forecasts, actuals_df)
        
        assert diag.coverage_p80 == 0.80
        assert diag.coverage_p50 == 0.50
    
    def test_sharpness_calculation(self, validator):
        """Verify sharpness calculation."""
        n = 50
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
        
        forecasts = pd.DataFrame({
            "timestamp": timestamps,
            "p10": np.zeros(n),
            "p25": np.ones(n) * 10,
            "p50": np.ones(n) * 50,
            "p75": np.ones(n) * 30,  # p75 - p25 = 20
            "p90": np.ones(n) * 80,  # p90 - p10 = 80
        })
        
        actuals = pd.DataFrame({
            "timestamp": timestamps,
            "actual_value": np.ones(n) * 50,
        })
        
        diag = validator.compute_diagnostics(forecasts, actuals)
        
        assert diag.sharpness_p80 == 80.0  # p90 - p10
        assert diag.sharpness_p50 == 20.0  # p75 - p25
    
    def test_missing_columns_raises(self, validator):
        """Missing required columns raise ValueError."""
        forecasts = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "p10": [1, 2, 3],
            # Missing p25, p50, p75, p90
        })
        
        actuals = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "actual_value": [1, 2, 3],
        })
        
        with pytest.raises(ValueError, match="missing columns"):
            validator.compute_diagnostics(forecasts, actuals)


class TestForecastValidatorPerHorizon:
    """Tests for per-horizon diagnostics."""
    
    def test_per_horizon_diagnostics(self):
        """Compute diagnostics broken down by horizon."""
        n = 100
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
        
        # Create forecasts with different horizons
        forecasts = pd.DataFrame({
            "timestamp": timestamps,
            "horizon_hours": [1, 6, 24] * 33 + [1],
            "p10": np.zeros(n),
            "p25": np.ones(n) * 25,
            "p50": np.ones(n) * 50,
            "p75": np.ones(n) * 75,
            "p90": np.ones(n) * 100,
        })
        
        actuals = pd.DataFrame({
            "timestamp": timestamps,
            "actual_value": np.ones(n) * 50,
        })
        
        validator = ForecastValidator()
        results = validator.compute_per_horizon(forecasts, actuals)
        
        assert 1 in results
        assert 6 in results
        assert 24 in results
        
        # Each should have diagnostics
        for h, diag in results.items():
            assert diag.n_validated > 0


# =============================================================================
# CalibrationAnalyzer Tests
# =============================================================================

class TestCalibrationAnalyzer:
    """Tests for CalibrationAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        return CalibrationAnalyzer(n_bins=10)
    
    def test_reliability_diagram(self, analyzer):
        """Compute reliability diagram data."""
        n = 100
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
        actuals_vals = np.random.randn(n) * 10 + 50
        
        forecasts = pd.DataFrame({
            "timestamp": timestamps,
            "p10": actuals_vals - 15,
            "p25": actuals_vals - 5,
            "p50": actuals_vals,
            "p75": actuals_vals + 5,
            "p90": actuals_vals + 15,
        })
        
        actuals = pd.DataFrame({
            "timestamp": timestamps,
            "actual_value": actuals_vals,
        })
        
        diagram = analyzer.compute_reliability_diagram(forecasts, actuals)
        
        assert "nominal_coverage" in diagram.columns
        assert "actual_coverage" in diagram.columns
        assert "deviation" in diagram.columns
        assert len(diagram) >= 1
    
    def test_reliability_diagram_empty(self, analyzer):
        """Empty data returns empty diagram."""
        diagram = analyzer.compute_reliability_diagram(
            pd.DataFrame(),
            pd.DataFrame()
        )
        
        assert diagram.empty
    
    def test_pit_histogram(self, analyzer):
        """Compute PIT histogram."""
        n = 200
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
        actuals_vals = np.random.randn(n) * 10 + 50
        
        forecasts = pd.DataFrame({
            "timestamp": timestamps,
            "p10": actuals_vals - 15,
            "p25": actuals_vals - 5,
            "p50": actuals_vals,
            "p75": actuals_vals + 5,
            "p90": actuals_vals + 15,
        })
        
        actuals = pd.DataFrame({
            "timestamp": timestamps,
            "actual_value": actuals_vals,
        })
        
        bin_centers, frequencies = analyzer.compute_pit_histogram(forecasts, actuals)
        
        # For well-calibrated forecasts, histogram should be roughly uniform
        if len(bin_centers) > 0:
            assert len(bin_centers) == len(frequencies)
            assert len(bin_centers) == 10  # n_bins


# =============================================================================
# Summary Function Tests
# =============================================================================

class TestComputeForecastDiagnosticsSummary:
    """Tests for compute_forecast_diagnostics_summary function."""
    
    def test_health_forecasts_only(self):
        """Compute diagnostics for health forecasts only."""
        n = 50
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
        health = np.linspace(100, 50, n)
        
        health_forecasts = pd.DataFrame({
            "timestamp": timestamps,
            "p10": health - 5,
            "p25": health - 2,
            "p50": health,
            "p75": health + 2,
            "p90": health + 5,
        })
        
        health_actuals = pd.DataFrame({
            "timestamp": timestamps,
            "actual_value": health + np.random.randn(n),
        })
        
        results = compute_forecast_diagnostics_summary(
            health_forecasts=health_forecasts,
            health_actuals=health_actuals,
        )
        
        assert "health" in results
        assert results["health"].n_validated > 0
    
    def test_all_forecast_types(self):
        """Compute diagnostics for all forecast types."""
        n = 30
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
        
        # Create minimal valid forecasts
        base = np.ones(n) * 50
        forecasts = pd.DataFrame({
            "timestamp": timestamps,
            "p10": base - 10,
            "p25": base - 5,
            "p50": base,
            "p75": base + 5,
            "p90": base + 10,
        })
        
        actuals = pd.DataFrame({
            "timestamp": timestamps,
            "actual_value": base,
        })
        
        results = compute_forecast_diagnostics_summary(
            health_forecasts=forecasts.copy(),
            health_actuals=actuals.copy(),
            rul_forecasts=forecasts.copy(),
            rul_actuals=actuals.copy(),
            sensor_forecasts={"temp": forecasts.copy()},
            sensor_actuals={"temp": actuals.copy()},
        )
        
        assert "health" in results
        assert "rul" in results
        assert "sensor_temp" in results
