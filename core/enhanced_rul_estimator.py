"""
Enhanced RUL Estimator with Adaptive Learning
==============================================

Improvements:
- Ensemble of degradation models (AR, exponential, Weibull-inspired)
- Online learning from prediction errors
- Bayesian parameter updating
- Degradation rate analysis
- Confidence calibration based on historical accuracy

Timestamp Policy:
-----------------
All timestamps are treated as timezone-naive local time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json

import numpy as np
import pandas as pd
from math import erf, sqrt
from scipy.optimize import minimize_scalar
from scipy.stats import norm, weibull_min

from utils.logger import Console
from core.rul_common import norm_cdf, RULConfig


@dataclass
class ModelPerformanceMetrics:
    """Track model performance for adaptive learning."""
    model_name: str
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    mae: float = 0.0
    rmse: float = 0.0
    bias: float = 0.0

    def update(self, predicted: float, actual: float, timestamp: pd.Timestamp):
        """Add new prediction-actual pair and update metrics."""
        error = predicted - actual
        self.predictions.append(predicted)
        self.actuals.append(actual)
        self.errors.append(error)

        # Keep only recent history (last 100 points)
        if len(self.errors) > 100:
            self.predictions = self.predictions[-100:]
            self.actuals = self.actuals[-100:]
            self.errors = self.errors[-100:]

        # Recalculate metrics
        errors_arr = np.array(self.errors)
        self.mae = float(np.mean(np.abs(errors_arr)))
        self.rmse = float(np.sqrt(np.mean(errors_arr**2)))
        self.bias = float(np.mean(errors_arr))

    def get_adaptive_weight(self) -> float:
        """Calculate model weight based on recent performance (inverse error)."""
        if not self.errors or self.rmse == 0:
            return 1.0
        # Softmax-style weighting: better models get exponentially more weight
        return float(np.exp(-self.rmse / 10.0))


@dataclass
class LearningState:
    """Persistent state for online learning."""
    model_metrics: Dict[str, ModelPerformanceMetrics] = field(default_factory=dict)
    calibration_factor: float = 1.0  # Multiplicative factor for uncertainty
    last_updated: Optional[pd.Timestamp] = None
    prediction_history: List[Dict[str, Any]] = field(default_factory=list)

    def save(self, path: Path):
        """Persist learning state to disk."""
        state_dict = {
            "calibration_factor": self.calibration_factor,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "model_metrics": {
                name: {
                    "mae": m.mae,
                    "rmse": m.rmse,
                    "bias": m.bias,
                    "recent_errors": m.errors[-20:],  # Save recent errors only
                }
                for name, m in self.model_metrics.items()
            },
            "recent_predictions": self.prediction_history[-50:],  # Keep last 50
        }
        path.write_text(json.dumps(state_dict, indent=2, default=str))
        Console.info(f"[RUL-Learn] Saved learning state to {path}")

    @classmethod
    def load(cls, path: Path) -> "LearningState":
        """Load learning state from disk."""
        if not path.exists():
            return cls()

        try:
            state_dict = json.loads(path.read_text())
            state = cls()
            state.calibration_factor = state_dict.get("calibration_factor", 1.0)

            last_upd = state_dict.get("last_updated")
            if last_upd:
                state.last_updated = pd.Timestamp(last_upd)

            # Reconstruct model metrics
            for name, metrics in state_dict.get("model_metrics", {}).items():
                m = ModelPerformanceMetrics(model_name=name)
                m.mae = metrics.get("mae", 0.0)
                m.rmse = metrics.get("rmse", 0.0)
                m.bias = metrics.get("bias", 0.0)
                m.errors = metrics.get("recent_errors", [])
                state.model_metrics[name] = m

            state.prediction_history = state_dict.get("recent_predictions", [])
            Console.info(f"[RUL-Learn] Loaded learning state from {path}")
            return state
        except Exception as e:
            Console.warn(f"[RUL-Learn] Failed to load state: {e}, using fresh state")
            return cls()


class DegradationModel:
    """Base class for degradation models."""

    def __init__(self, name: str):
        self.name = name
        self.params = {}

    def fit(self, timestamps: pd.DatetimeIndex, health_values: np.ndarray) -> bool:
        """Fit model to historical data. Returns True if successful."""
        raise NotImplementedError

    def predict(self, future_timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        """Return (forecast_values, forecast_std)."""
        raise NotImplementedError


class AR1Model(DegradationModel):
    """Enhanced AR(1) with bias correction and adaptive parameters."""

    def fit(self, timestamps: pd.DatetimeIndex, health_values: np.ndarray) -> bool:
        y = health_values
        if len(y) < 10:
            return False

        # Center data
        self.mu = float(np.nanmean(y))
        yc = y - self.mu

        # AR(1) coefficient
        var_yc = float(np.var(yc))
        if var_yc < 1e-8:
            return False

        cov = float(np.dot(yc[1:], yc[:-1]))
        var = float(np.dot(yc[:-1], yc[:-1]))
        self.phi = np.clip(cov / (var + 1e-9), -0.99, 0.99)

        # Residual variance
        y_shift = np.concatenate([[self.mu], y[:-1]])
        pred = (y_shift - self.mu) * self.phi + self.mu
        resid = y - pred
        self.sigma = float(np.std(resid[1:])) if len(resid) > 1 else 1.0
        self.sigma = max(self.sigma, 0.1)  # Floor for numerical stability

        # Detect systematic bias (drift)
        recent_window = min(len(y), 30)
        t = np.arange(recent_window)
        y_recent = y[-recent_window:]
        if len(y_recent) > 3:
            drift = np.polyfit(t, y_recent, 1)[0]
            self.drift = float(drift)
        else:
            self.drift = 0.0

        self.last_value = float(y[-1])
        self.last_time = timestamps[-1]

        # Infer cadence
        deltas = np.diff(timestamps.values.astype("int64")) / 1e9
        self.step_sec = float(np.median(deltas))
        if self.step_sec <= 0:
            self.step_sec = 60.0

        return True

    def predict(self, future_timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        steps = len(future_timestamps)
        h = np.arange(1, steps + 1)

        # AR(1) forecast with drift correction
        if abs(self.phi) > 1e-9:
            phi_h = self.phi**h
            ar_component = phi_h * (self.last_value - self.mu)
        else:
            ar_component = np.zeros(steps)

        # Add linear drift for non-stationary series
        drift_component = self.drift * h * (self.step_sec / 3600.0)

        forecast = self.mu + ar_component + drift_component

        # Growing uncertainty
        if abs(self.phi) < 0.999:
            var_mult = (1 - self.phi ** (2 * h)) / (1 - self.phi**2 + 1e-9)
        else:
            var_mult = h

        var_mult = np.clip(var_mult, 1.0, 100.0)
        forecast_std = self.sigma * np.sqrt(var_mult)

        return forecast, forecast_std


class ExponentialDegradationModel(DegradationModel):
    """Exponential degradation: h(t) = h0 * exp(-λ*t) + offset."""

    def fit(self, timestamps: pd.DatetimeIndex, health_values: np.ndarray) -> bool:
        y = health_values
        if len(y) < 10:
            return False

        # Time in hours from start
        t = (timestamps - timestamps[0]).total_seconds().values / 3600.0

        # Fit exponential decay via log-linear regression on detrended data
        offset = float(np.min(y)) - 1.0  # Stabilize for log
        y_shifted = y - offset

        if np.any(y_shifted <= 0):
            return False

        log_y = np.log(y_shifted)

        # Linear regression on log scale
        A = np.vstack([t, np.ones(len(t))]).T
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
            self.lambda_ = float(-coeffs[0])  # Decay rate
            self.h0 = float(np.exp(coeffs[1]))
            self.offset = offset

            # Residual std in original scale
            pred_log = A @ coeffs
            pred = np.exp(pred_log) + offset
            self.sigma = float(np.std(y - pred))
            self.sigma = max(self.sigma, 0.1)

            self.last_time = timestamps[-1]
            return True
        except Exception:
            return False

    def predict(self, future_timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        t = (future_timestamps - self.last_time).total_seconds().values / 3600.0

        # Exponential projection
        forecast = self.h0 * np.exp(-self.lambda_ * t) + self.offset

        # Uncertainty grows with time
        forecast_std = self.sigma * np.sqrt(1 + 0.1 * t)  # Simple heuristic

        return forecast, forecast_std


class WeibullInspiredModel(DegradationModel):
    """Weibull-inspired degradation for non-linear failure acceleration."""

    def fit(self, timestamps: pd.DatetimeIndex, health_values: np.ndarray) -> bool:
        y = health_values
        if len(y) < 15:
            return False

        # Time in hours
        t = (timestamps - timestamps[0]).total_seconds().values / 3600.0

        # Fit power-law degradation: h(t) = h0 - k * t^β
        h0_est = float(y[0])

        def objective(beta):
            if beta <= 0:
                return 1e10
            try:
                t_beta = t**beta
                A = np.vstack([t_beta, np.ones(len(t))]).T
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                pred = A @ coeffs
                return np.sum((y - pred) ** 2)
            except Exception:
                return 1e10

        # Optimize beta (shape parameter)
        result = minimize_scalar(objective, bounds=(0.5, 3.0), method="bounded")

        if not result.success:
            return False

        self.beta = float(result.x)
        t_beta = t**self.beta
        A = np.vstack([t_beta, np.ones(len(t))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        self.k = float(-coeffs[0])  # Degradation rate (positive)
        self.h0 = float(coeffs[1])

        pred = A @ coeffs
        self.sigma = float(np.std(y - pred))
        self.sigma = max(self.sigma, 0.1)

        self.last_time = timestamps[-1]
        self.t_base = timestamps[0]

        return True

    def predict(self, future_timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        t = (future_timestamps - self.t_base).total_seconds().values / 3600.0

        forecast = self.h0 - self.k * (t**self.beta)

        # Uncertainty grows non-linearly
        t_rel = (future_timestamps - self.last_time).total_seconds().values / 3600.0
        forecast_std = self.sigma * np.sqrt(1 + 0.15 * t_rel)

        return forecast, forecast_std


def _ensemble_forecast(
    timestamps: pd.DatetimeIndex,
    health_values: np.ndarray,
    future_timestamps: pd.DatetimeIndex,
    learning_state: LearningState,
    cfg: RULConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Ensemble forecasting with adaptive weighting based on historical performance.

    Returns: (forecast, forecast_std, model_weights)
    """
    models = [
        AR1Model("AR1"),
        ExponentialDegradationModel("Exponential"),
        WeibullInspiredModel("Weibull"),
    ]

    fitted_models = []
    for model in models:
        try:
            if model.fit(timestamps, health_values):
                fitted_models.append(model)
                Console.info(f"[RUL-Ensemble] {model.name} fitted successfully")
            else:
                Console.warn(f"[RUL-Ensemble] {model.name} failed to fit")
        except Exception as e:
            Console.warn(f"[RUL-Ensemble] {model.name} exception: {e}")

    if not fitted_models:
        # Fallback to naive
        Console.warn("[RUL-Ensemble] No models fitted, using naive flat forecast")
        last_val = float(health_values[-1])
        forecast = np.full(len(future_timestamps), last_val)
        hist_std = float(np.std(health_values)) if len(health_values) > 1 else 1.0
        forecast_std = np.full(len(future_timestamps), max(hist_std, 1.0))
        return forecast, forecast_std, {}

    # Get predictions from each model
    predictions = []
    stds = []
    model_names = []

    for model in fitted_models:
        try:
            pred, std = model.predict(future_timestamps)
            predictions.append(pred)
            stds.append(std)
            model_names.append(model.name)
        except Exception as e:
            Console.warn(f"[RUL-Ensemble] {model.name} prediction failed: {e}")

    if not predictions:
        last_val = float(health_values[-1])
        forecast = np.full(len(future_timestamps), last_val)
        forecast_std = np.full(len(future_timestamps), 1.0)
        return forecast, forecast_std, {}

    predictions = np.array(predictions)
    stds = np.array(stds)

    # Calculate adaptive weights based on historical performance
    weights = np.ones(len(fitted_models))

    if cfg.enable_online_learning:
        for i, name in enumerate(model_names):
            if name in learning_state.model_metrics:
                w = learning_state.model_metrics[name].get_adaptive_weight()
                weights[i] = max(w, cfg.min_model_weight)

    # Normalize weights
    weights = weights / weights.sum()

    # Weighted ensemble
    forecast = np.average(predictions, axis=0, weights=weights)

    # Ensemble uncertainty: combine individual uncertainties + inter-model variance
    weighted_var = np.average(stds**2, axis=0, weights=weights)
    inter_model_var = np.var(predictions, axis=0)
    combined_var = weighted_var + inter_model_var
    forecast_std = np.sqrt(combined_var)

    # Apply calibration factor from learning
    forecast_std = forecast_std * learning_state.calibration_factor

    model_weights = dict(zip(model_names, weights))
    Console.info(f"[RUL-Ensemble] Model weights: {model_weights}")

    return forecast, forecast_std, model_weights


def _update_learning_state(
    learning_state: LearningState,
    timestamps: pd.DatetimeIndex,
    actual_health: np.ndarray,
    cfg: RULConfig,
):
    """
    Update learning state with new actual observations.
    Compare against historical predictions to improve model weights and calibration.
    """
    if not cfg.enable_online_learning:
        return

    # Match recent predictions with actuals
    for pred_record in learning_state.prediction_history[-20:]:  # Check recent predictions
        pred_time = pd.Timestamp(pred_record["timestamp"])
        pred_value = pred_record["predicted_health"]
        model_name = pred_record.get("model", "ensemble")

        # Find actual value close to prediction time (within 1 hour)
        time_diffs = np.abs((timestamps - pred_time).total_seconds())
        if time_diffs.min() < 3600:  # Within 1 hour
            idx = np.argmin(time_diffs)
            actual_value = float(actual_health[idx])

            # Update model metrics
            if model_name not in learning_state.model_metrics:
                learning_state.model_metrics[model_name] = ModelPerformanceMetrics(model_name)

            learning_state.model_metrics[model_name].update(pred_value, actual_value, pred_time)

            # Mark as processed
            pred_record["matched"] = True

    # Calibrate uncertainty based on recent errors
    all_errors = []
    for metrics in learning_state.model_metrics.values():
        all_errors.extend(metrics.errors)

    if len(all_errors) >= cfg.calibration_window:
        recent_errors = all_errors[-cfg.calibration_window:]
        empirical_std = float(np.std(recent_errors))
        expected_std = 1.0  # Normalized expectation

        # Adjust calibration factor with learning rate
        new_factor = empirical_std / (expected_std + 1e-6)
        learning_state.calibration_factor = (1 - cfg.learning_rate) * learning_state.calibration_factor + cfg.learning_rate * new_factor
        learning_state.calibration_factor = np.clip(learning_state.calibration_factor, 0.5, 3.0)

        Console.info(f"[RUL-Learn] Updated calibration factor: {learning_state.calibration_factor:.3f}")

    learning_state.last_updated = timestamps[-1]


def estimate_rul_and_failure(
    tables_dir: Path,
    equip_id: Optional[int],
    run_id: Optional[str],
    health_threshold: float = 70.0,
    cfg: Optional[RULConfig] = None,
    sql_client: Optional[Any] = None,
    output_manager: Optional[Any] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Enhanced RUL estimation with adaptive learning and ensemble forecasting.
    """
    cfg = cfg or RULConfig(health_threshold=health_threshold)

    # Load or initialize learning state
    learning_state_path = tables_dir / f"rul_learning_state_{equip_id or 'default'}.json"
    learning_state = LearningState.load(learning_state_path)

    # Cleanup old forecast data (SQL mode only)
    if sql_client is not None and equip_id is not None:
        try:
            import os

            keep_runs = max(1, min(int(os.getenv("ACM_FORECAST_RUNS_RETAIN", "2")), 50))
            cur = sql_client.cursor()

            for table in ["ACM_HealthForecast_TS", "ACM_FailureForecast_TS"]:
                cur.execute(
                    f"""
                    WITH RankedRuns AS (
                        SELECT DISTINCT RunID, 
                               ROW_NUMBER() OVER (ORDER BY MAX(CreatedAt) DESC) AS rn
                        FROM dbo.{table}
                        WHERE EquipID = ?
                        GROUP BY RunID
                    )
                    DELETE FROM dbo.{table}
                    WHERE EquipID = ? 
                      AND RunID IN (SELECT RunID FROM RankedRuns WHERE rn > ?)
                """,
                    (equip_id, equip_id, keep_runs),
                )

            if not sql_client.conn.autocommit:
                sql_client.conn.commit()
            Console.info(f"[RUL] Cleaned old forecast data for EquipID={equip_id}")
        except Exception as e:
            Console.warn(f"[RUL] Failed to cleanup old forecasts: {e}")

    # Load health timeline
    health_df = _load_health_timeline(
        tables_dir, sql_client=sql_client, equip_id=equip_id, run_id=run_id, output_manager=output_manager
    )

    if health_df is None or "HealthIndex" not in health_df.columns:
        Console.warn("[RUL] Health timeline not available; skipping RUL/forecast outputs.")
        return {}

    # Prepare health time series
    hi = health_df["HealthIndex"].astype(float).clip(lower=0.0, upper=100.0)
    # TIME-01: Ensure naive timestamps
    ts_idx = pd.to_datetime(health_df["Timestamp"], errors="coerce")
    if ts_idx.dt.tz is not None:
        ts_idx = ts_idx.dt.tz_localize(None)
    hi.index = pd.DatetimeIndex(ts_idx)
    hi = hi.sort_index()

    if hi.size < cfg.min_points:
        Console.warn(f"[RUL] Insufficient data points ({hi.size} < {cfg.min_points})")
        return {}

    # Update learning state with new observations
    _update_learning_state(learning_state, hi.index, hi.values, cfg)

    # Generate forecast using ensemble
    last_ts = hi.index[-1]
    step_sec = 3600.0  # 1-hour intervals
    max_steps = int(np.ceil(cfg.max_forecast_hours))
    future_idx = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=max_steps, freq="1h")

    forecast_values, forecast_std, model_weights = _ensemble_forecast(
        hi.index, hi.values, future_idx, learning_state, cfg
    )

    forecast = pd.Series(forecast_values, index=future_idx, name="ForecastHealth")
    forecast = forecast.clip(lower=0.0, upper=100.0)

    # Confidence intervals
    ci_k = 1.96
    ci_lower = (forecast - ci_k * forecast_std).clip(0, 100)
    ci_upper = (forecast + ci_k * forecast_std).clip(0, 100)

    # Failure probability
    thr = float(np.clip(health_threshold, 0.0, 100.0))
    z = (thr - forecast.values) / (forecast_std + 1e-9)
    failure_prob = np.clip(norm_cdf(z), 0.0, 1.0)
    # NOTE: Removed np.maximum.accumulate - it was forcing all probabilities to be identical
    # Failure probability should reflect the actual forecast distribution at each time point

    # RUL estimation
    below = forecast.values <= thr
    if below.any():
        first_idx = int(np.argmax(below))
        rul_hours = float(first_idx + 1)
        failure_time = forecast.index[first_idx]
    else:
        rul_hours = float(len(forecast))
        failure_time = forecast.index[-1]

    # Bounds from CI
    lower_cross = ci_lower.values <= thr
    upper_cross = ci_upper.values <= thr
    lower_bound_hours = float(np.argmax(lower_cross) + 1) if lower_cross.any() else rul_hours
    upper_bound_hours = float(np.argmax(upper_cross) + 1) if upper_cross.any() else rul_hours

    # Confidence based on ensemble agreement + calibration
    ci_width_norm = (ci_upper.values - ci_lower.values) / (thr + 1e-9)
    conf = float(np.clip(1.0 - np.nanmean(ci_width_norm) * learning_state.calibration_factor, 0.0, 1.0))

    # Store prediction for future learning
    learning_state.prediction_history.append(
        {
            "timestamp": failure_time.isoformat(),
            "predicted_health": float(forecast.values[first_idx] if below.any() else forecast.values[-1]),
            "rul_hours": rul_hours,
            "model": "ensemble",
            "model_weights": model_weights,
            "matched": False,
        }
    )

    # Save updated learning state
    learning_state.save(learning_state_path)

    # Build output DataFrames
    run_id_val = run_id or ""
    equip_id_val = int(equip_id) if equip_id is not None else None

    def _insert_ids(df: pd.DataFrame) -> pd.DataFrame:
        if run_id_val:
            df.insert(0, "RunID", run_id_val)
        if equip_id_val is not None:
            df.insert(1 if run_id_val else 0, "EquipID", equip_id_val)
        return df

    # Health forecast TS
    # CONTINUITY FIX: Include recent historical health data to bridge gap between history and forecast
    # This prevents stepped appearance in dashboard by providing smooth transition
    lookback_hours = 24.0  # Include last 24h of history
    cutoff_ts = forecast.index[0] - pd.Timedelta(hours=lookback_hours)
    recent_history = hi[hi.index >= cutoff_ts].copy()
    
    # Build forecast dataframe
    forecast_df = pd.DataFrame(
        {
            "Timestamp": forecast.index,
            "ForecastHealth": forecast.values,
            "CiLower": ci_lower.values,
            "CiUpper": ci_upper.values,
            "ForecastStd": forecast_std,
            "Method": "Ensemble_Adaptive",
            "ModelWeights": [str(model_weights)] * len(forecast),
        }
    )
    
    # Build historical dataframe for recent data
    if len(recent_history) > 0:
        # For historical data, use actual health value and narrow CI based on measurement uncertainty
        hist_uncertainty = 2.0  # Small fixed uncertainty for historical data
        history_df = pd.DataFrame({
            "Timestamp": recent_history.index,
            "ForecastHealth": recent_history.values,
            "CiLower": np.clip(recent_history.values - hist_uncertainty, 0.0, 100.0),
            "CiUpper": np.clip(recent_history.values + hist_uncertainty, 0.0, 100.0),
            "ForecastStd": hist_uncertainty,
            "Method": "Historical",
            "ModelWeights": "N/A",
        })
        # Concatenate history + forecast for seamless timeline
        health_forecast_df = pd.concat([history_df, forecast_df], ignore_index=True)
    else:
        health_forecast_df = forecast_df
    
    health_forecast_df = _insert_ids(health_forecast_df)

    # Failure probability TS
    failure_forecast_df = pd.DataFrame(
        {
            "Timestamp": forecast.index,
            "FailureProb": failure_prob,
            "ThresholdUsed": thr,
            "Method": "Ensemble_Adaptive",
        }
    )
    
    # CONTINUITY FIX: Include historical failure probability (near zero for healthy history)
    if len(recent_history) > 0:
        # For historical data, calculate failure probability based on actual health vs threshold
        z_hist = (thr - recent_history.values) / (hist_uncertainty + 1e-9)
        failure_prob_hist = np.clip(norm_cdf(z_hist), 0.0, 1.0)
        history_fail_df = pd.DataFrame({
            "Timestamp": recent_history.index,
            "FailureProb": failure_prob_hist,
            "ThresholdUsed": thr,
            "Method": "Historical",
        })
        failure_ts_df = pd.concat([history_fail_df, failure_forecast_df], ignore_index=True)
    else:
        failure_ts_df = failure_forecast_df
    
    failure_ts_df = _insert_ids(failure_ts_df)

    # RUL TS
    h_hours = np.arange(1, len(forecast) + 1, dtype=float)
    remaining_hours = np.maximum(rul_hours - h_hours, 0.0)
    rul_ts_df = pd.DataFrame(
        {
            "Timestamp": forecast.index,
            "RUL_Hours": remaining_hours,
            "LowerBound": np.maximum(lower_bound_hours - h_hours, 0.0),
            "UpperBound": np.maximum(upper_bound_hours - h_hours, 0.0),
            "Confidence": conf,
            "Method": "Ensemble_Adaptive",
            "CalibrationFactor": learning_state.calibration_factor,
        }
    )
    rul_ts_df = _insert_ids(rul_ts_df)

    # RUL summary
    now_ts = hi.index[-1]
    rul_summary_df = pd.DataFrame(
        [
            {
                "RUL_Hours": rul_hours,
                "LowerBound": lower_bound_hours,
                "UpperBound": upper_bound_hours,
                "Confidence": conf,
                "Method": "Ensemble_Adaptive",
                "LastUpdate": now_ts,
                "ModelWeights": str(model_weights),
                "CalibrationFactor": learning_state.calibration_factor,
            }
        ]
    )
    rul_summary_df = _insert_ids(rul_summary_df)

    # Sensor attribution
    attribution_df = _build_sensor_attribution(
        tables_dir=tables_dir,
        equip_id=equip_id_val,
        run_id=run_id_val,
        failure_time=failure_time,
        sql_client=sql_client,
    )

    # Maintenance recommendation
    maint_df = _build_maintenance_recommendation(
        forecast_index=forecast.index,
        failure_prob=failure_prob,
        cfg=cfg,
        run_id=run_id_val,
        equip_id=equip_id_val,
    )

    Console.info(
        f"[RUL] Estimated RUL: {rul_hours:.1f} hours (conf={conf:.2f}, cal_factor={learning_state.calibration_factor:.3f})"
    )

    return {
        "ACM_HealthForecast_TS": health_forecast_df,
        "ACM_FailureForecast_TS": failure_ts_df,
        "ACM_RUL_TS": rul_ts_df,
        "ACM_RUL_Summary": rul_summary_df,
        "ACM_RUL_Attribution": attribution_df,
        "ACM_MaintenanceRecommendation": maint_df,
    }


def _load_health_timeline(
    tables_dir: Path,
    sql_client: Optional[Any] = None,
    equip_id: Optional[int] = None,
    run_id: Optional[str] = None,
    output_manager: Optional[Any] = None,
) -> Optional[pd.DataFrame]:
    """
    Load health timeline for RUL estimation.
    Priority: artifact cache > SQL > CSV file.
    """
    # Try artifact cache first (SQL-only mode)
    if output_manager is not None:
        df = output_manager.get_cached_table("health_timeline.csv")
        if df is not None:
            Console.info(f"[RUL] Using cached health_timeline.csv ({len(df)} rows)")
            if "Timestamp" in df.columns:
                ts_col = "Timestamp"
            elif "timestamp" in df.columns:
                ts_col = "timestamp"
            else:
                ts_col = df.columns[0]
            # TIME-01: Ensure naive timestamps
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            if df[ts_col].dt.tz is not None:
                df[ts_col] = df[ts_col].dt.tz_localize(None)
            
            df = df.dropna(subset=[ts_col]).sort_values(ts_col)
            df = df.rename(columns={ts_col: "Timestamp"})
            return df

    # Try SQL
    if sql_client is not None and equip_id is not None and run_id:
        try:
            cur = sql_client.cursor()
            cur.execute(
                """
                SELECT Timestamp, HealthIndex
                FROM dbo.ACM_HealthTimeline
                WHERE EquipID = ? AND RunID = ?
                ORDER BY Timestamp
                """,
                (equip_id, run_id),
            )
            rows = cur.fetchall()
            cur.close()
            if rows:
                df = pd.DataFrame.from_records(rows, columns=["Timestamp", "HealthIndex"])
                # TIME-01: Ensure naive timestamps
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                if df["Timestamp"].dt.tz is not None:
                    df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
                
                df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
                Console.info(f"[RUL] Loaded {len(df)} health points from SQL")
                return df
        except Exception as e:
            Console.warn(f"[RUL] Failed to load from SQL: {e}")

    # Fallback to CSV
    p = tables_dir / "health_timeline.csv"
    if not p.exists():
        Console.warn(f"[RUL] health_timeline.csv not found")
        return None

    df = pd.read_csv(p)
    Console.info(f"[RUL] Loaded health_timeline.csv from file ({len(df)} rows)")
    if "Timestamp" in df.columns:
        ts_col = "Timestamp"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        ts_col = df.columns[0]
    # TIME-01: Ensure naive timestamps
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].dt.tz is not None:
        df[ts_col] = df[ts_col].dt.tz_localize(None)
        
    df = df.sort_values(ts_col).dropna(subset=[ts_col])
    df = df.rename(columns={ts_col: "Timestamp"})
    return df


def _build_sensor_attribution(
    tables_dir: Path,
    equip_id: Optional[int],
    run_id: str,
    failure_time: pd.Timestamp,
    sql_client: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Build sensor attribution at predicted failure time using current hotspots as proxy.
    """
    # Try SQL first
    if sql_client is not None and equip_id is not None and run_id:
        try:
            cur = sql_client.cursor()
            cur.execute(
                """
                SELECT TOP 10 SensorName, LatestAbsZ, AboveAlertCount
                FROM dbo.ACM_SensorHotspots
                WHERE EquipID = ? AND RunID = ?
                ORDER BY LatestAbsZ DESC
                """,
                (equip_id, run_id),
            )
            rows = cur.fetchall()
            cur.close()
            if rows:
                df = pd.DataFrame.from_records(rows, columns=["SensorName", "Z", "AlertCount"])
                df["Z"] = pd.to_numeric(df["Z"], errors="coerce").clip(lower=0.0)
                df = df.dropna(subset=["Z"])
                if df["Z"].sum() > 0:
                    df["FailureContribution"] = df["Z"] / df["Z"].sum()
                    df = df.sort_values("FailureContribution", ascending=False).head(10)
                else:
                    df = pd.DataFrame(columns=["SensorName", "FailureContribution", "Z", "AlertCount"])
            else:
                df = pd.DataFrame(columns=["SensorName", "FailureContribution", "Z", "AlertCount"])
        except Exception as e:
            Console.warn(f"[RUL] Failed to load sensor hotspots from SQL: {e}")
            df = pd.DataFrame(columns=["SensorName", "FailureContribution", "Z", "AlertCount"])
    else:
        # Fallback to CSV
        p = tables_dir / "sensor_hotspots.csv"
        if not p.exists():
            return pd.DataFrame(
                columns=[
                    "RunID",
                    "EquipID",
                    "FailureTime",
                    "SensorName",
                    "FailureContribution",
                    "ZScoreAtFailure",
                    "AlertCount",
                    "Comment",
                ]
            )
        df = pd.read_csv(p)
        if "SensorName" not in df.columns or "LatestAbsZ" not in df.columns:
            return pd.DataFrame(
                columns=[
                    "RunID",
                    "EquipID",
                    "FailureTime",
                    "SensorName",
                    "FailureContribution",
                    "ZScoreAtFailure",
                    "AlertCount",
                    "Comment",
                ]
            )
        df = df.rename(columns={"LatestAbsZ": "Z"})
        df["Z"] = df["Z"].astype(float).clip(lower=0.0)
        if df["Z"].sum() <= 0:
            return pd.DataFrame(
                columns=[
                    "RunID",
                    "EquipID",
                    "FailureTime",
                    "SensorName",
                    "FailureContribution",
                    "ZScoreAtFailure",
                    "AlertCount",
                    "Comment",
                ]
            )
        df["FailureContribution"] = df["Z"] / df["Z"].sum()
        df = df.sort_values("FailureContribution", ascending=False).head(10)

    result = pd.DataFrame(
        {
            "FailureTime": failure_time,
            "SensorName": df["SensorName"],
            "FailureContribution": df["FailureContribution"],
            "ZScoreAtFailure": df["Z"],
            "AlertCount": df.get("AboveAlertCount", pd.Series([None] * len(df))),
            "Comment": [None] * len(df),
        }
    )

    if run_id:
        result.insert(0, "RunID", run_id)
    if equip_id is not None:
        result.insert(1 if run_id else 0, "EquipID", equip_id)

    return result


def _build_maintenance_recommendation(
    forecast_index: pd.DatetimeIndex,
    failure_prob: np.ndarray,
    cfg: RULConfig,
    run_id: str,
    equip_id: Optional[int],
) -> pd.DataFrame:
    """
    Build maintenance window recommendation from failure probability curve.
    """
    if forecast_index.empty or failure_prob.size == 0:
        return pd.DataFrame(
            columns=[
                "RunID",
                "EquipID",
                "EarliestMaintenance",
                "PreferredWindowStart",
                "PreferredWindowEnd",
                "FailureProbAtWindowEnd",
                "Comment",
            ]
        )

    fp = np.asarray(failure_prob, dtype=float)
    times = forecast_index

    low_mask = fp >= float(cfg.maintenance_risk_low)
    high_mask = fp >= float(cfg.maintenance_risk_high)

    if low_mask.any():
        first_low_idx = int(np.argmax(low_mask))
    else:
        first_low_idx = fp.size - 1

    if high_mask.any():
        first_high_idx = int(np.argmax(high_mask))
    else:
        first_high_idx = fp.size - 1

    earliest_ts = times[first_low_idx]
    window_start = earliest_ts
    window_end = times[first_high_idx]
    prob_at_end = float(fp[first_high_idx])

    df = pd.DataFrame(
        [
            {
                "EarliestMaintenance": earliest_ts,
                "PreferredWindowStart": window_start,
                "PreferredWindowEnd": window_end,
                "FailureProbAtWindowEnd": prob_at_end,
                "Comment": None,
            }
        ]
    )

    if run_id:
        df.insert(0, "RunID", run_id)
    if equip_id is not None:
        df.insert(1 if run_id else 0, "EquipID", equip_id)

    return df


# ============================================================================
# Multi-Path RUL Derivation (FORECAST-STATE-04)
# ============================================================================

def compute_rul_multipath(
    health_forecast: pd.DataFrame,
    hazard_df: Optional[pd.DataFrame],
    anomaly_energy_df: Optional[pd.DataFrame],
    current_time: pd.Timestamp,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute RUL via three independent paths and take minimum.
    
    Path 1 (Trajectory): Health forecast crosses failure threshold
    Path 2 (Hazard): Cumulative failure probability reaches 0.5
    Path 3 (Energy): Cumulative anomaly energy exceeds calibrated threshold
    
    Args:
        health_forecast: DataFrame with [Timestamp, ForecastHealth, CI_Lower, CI_Upper]
        hazard_df: DataFrame with [Timestamp, FailureProb] (optional)
        anomaly_energy_df: DataFrame with [Timestamp, CumulativeEnergy] (optional)
        current_time: Current timestamp
        config: Configuration dict
    
    Returns:
        Dict with RUL breakdown and dominant path
    """
    forecast_cfg = config.get("forecasting", {})
    health_threshold = float(forecast_cfg.get("failure_threshold", 75.0))
    hazard_prob_threshold = float(forecast_cfg.get("hazard_failure_prob", 0.5))
    energy_fail_threshold = float(forecast_cfg.get("energy_fail_threshold", 1000.0))
    max_forecast_hours = float(forecast_cfg.get("max_forecast_hours", 168.0))
    
    # Path 1: Trajectory crossing
    rul_trajectory = np.inf
    if health_forecast is not None and not health_forecast.empty:
        trajectory_crossing = health_forecast[
            health_forecast["ForecastHealth"] <= health_threshold
        ]
        if not trajectory_crossing.empty:
            t1 = trajectory_crossing.iloc[0]["Timestamp"]
            rul_trajectory = (t1 - current_time).total_seconds() / 3600
            Console.info(f"[RUL-MULTIPATH] Trajectory crossing at {t1}, RUL={rul_trajectory:.1f}h")
    
    # Path 2: Hazard accumulation
    rul_hazard = np.inf
    if hazard_df is not None and not hazard_df.empty and "FailureProb" in hazard_df.columns:
        hazard_crossing = hazard_df[hazard_df["FailureProb"] >= hazard_prob_threshold]
        if not hazard_crossing.empty:
            t2 = hazard_crossing.iloc[0]["Timestamp"]
            rul_hazard = (t2 - current_time).total_seconds() / 3600
            Console.info(f"[RUL-MULTIPATH] Hazard crossing at {t2}, RUL={rul_hazard:.1f}h")
    
    # Path 3: Anomaly energy threshold
    rul_energy = np.inf
    if anomaly_energy_df is not None and not anomaly_energy_df.empty:
        energy_crossing = anomaly_energy_df[
            anomaly_energy_df["CumulativeEnergy"] >= energy_fail_threshold
        ]
        if not energy_crossing.empty:
            t3 = energy_crossing.iloc[0]["Timestamp"]
            rul_energy = (t3 - current_time).total_seconds() / 3600
            Console.info(f"[RUL-MULTIPATH] Energy crossing at {t3}, RUL={rul_energy:.1f}h")
    
    # Final RUL = minimum of all paths
    rul_final = min(rul_trajectory, rul_hazard, rul_energy)
    if rul_final == np.inf:
        rul_final = max_forecast_hours
        Console.info(f"[RUL-MULTIPATH] No crossing detected, using max horizon: {rul_final}h")
    
    # Confidence band from trajectory CI crossings
    confidence_band = 0.0
    if health_forecast is not None and not health_forecast.empty:
        ci_lower_crossing = health_forecast[
            health_forecast["CI_Lower"] <= health_threshold
        ]
        ci_upper_crossing = health_forecast[
            health_forecast["CI_Upper"] <= health_threshold
        ]
        
        if not ci_lower_crossing.empty and not ci_upper_crossing.empty:
            t_lower = (ci_lower_crossing.iloc[0]["Timestamp"] - current_time).total_seconds() / 3600
            t_upper = (ci_upper_crossing.iloc[0]["Timestamp"] - current_time).total_seconds() / 3600
            confidence_band = abs(t_upper - t_lower)
    
    # Determine dominant path
    if rul_final == rul_trajectory:
        dominant_path = "trajectory"
    elif rul_final == rul_hazard:
        dominant_path = "hazard"
    elif rul_final == rul_energy:
        dominant_path = "energy"
    else:
        dominant_path = "unknown"
    
    result = {
        "rul_trajectory_hours": rul_trajectory if rul_trajectory != np.inf else None,
        "rul_hazard_hours": rul_hazard if rul_hazard != np.inf else None,
        "rul_energy_hours": rul_energy if rul_energy != np.inf else None,
        "rul_final_hours": rul_final,
        "confidence_band_hours": confidence_band,
        "dominant_path": dominant_path
    }
    
    Console.info(
        f"[RUL-MULTIPATH] Final RUL={rul_final:.1f}h "
        f"(trajectory={result['rul_trajectory_hours']}, "
        f"hazard={result['rul_hazard_hours']}, "
        f"energy={result['rul_energy_hours']}, "
        f"dominant={dominant_path})"
    )
    
    return result
