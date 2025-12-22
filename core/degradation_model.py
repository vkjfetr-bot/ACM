"""
Degradation Models for Health Forecasting (v10.0.0)

Implements trend-based degradation models with uncertainty quantification.
Replaces duplicate logic from forecasting.py (lines 2095-2749).

Key Features:
- Abstract base class for pluggable degradation models
- Holt's linear trend with adaptive smoothing
- Uncertainty growth via residual analysis
- Incremental updates for online learning
- Research-backed parameter bounds

References:
- Holt (1957): Exponential smoothing with trend
- Hyndman & Athanasopoulos (2018): "Forecasting: Principles and Practice" - alpha/beta bounds
- Box & Jenkins (1970): Forecast uncertainty quantification
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import optimize

from core.observability import Console, Heartbeat


@dataclass
class DegradationForecast:
    """Degradation forecast with uncertainty bounds"""
    timestamps: pd.DatetimeIndex
    point_forecast: np.ndarray      # Point estimates
    lower_bound: np.ndarray          # Lower confidence bound
    upper_bound: np.ndarray          # Upper confidence bound
    std_error: float                 # Residual standard error
    level: float                     # Final level estimate
    trend: float                     # Final trend estimate (per hour)


class BaseDegradationModel(ABC):
    """
    Abstract base class for degradation models.
    
    Design Contract:
    - fit(): Train model on historical health data
    - predict(): Generate multi-step ahead forecasts with uncertainty
    - update_incremental(): Update model with new observations (online learning)
    - get_parameters(): Export model state for persistence
    - set_parameters(): Restore model state from saved parameters
    """
    
    @abstractmethod
    def fit(self, health_series: pd.Series) -> None:
        """
        Train model on historical health data.
        
        Args:
            health_series: Time-indexed health values
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        steps: int,
        dt_hours: float,
        confidence_level: float = 0.95
    ) -> DegradationForecast:
        """
        Generate multi-step ahead forecast with uncertainty bounds.
        
        Args:
            steps: Number of forecast steps
            dt_hours: Time interval per step (hours)
            confidence_level: Confidence level for bounds (default 0.95)
        
        Returns:
            DegradationForecast with timestamps, point forecast, and uncertainty bounds
        """
        pass
    
    @abstractmethod
    def update_incremental(self, new_observation: float) -> None:
        """
        Update model with new observation (online learning).
        
        Args:
            new_observation: Latest health value
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """Export model state for persistence"""
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, float]) -> None:
        """Restore model state from saved parameters"""
        pass


class LinearTrendModel(BaseDegradationModel):
    """
    Holt's linear trend model with adaptive exponential smoothing.
    
    Mathematical Formulation:
    - Level equation: L[t] = α*y[t] + (1-α)*(L[t-1] + T[t-1])
    - Trend equation: T[t] = β*(L[t] - L[t-1]) + (1-β)*T[t-1]
    - Forecast: ŷ[t+h] = L[t] + h*T[t]
    - Uncertainty: σ[h] = σ_residual * sqrt(1 + h*(α^2 + h*β^2/2))
    
    Features:
    - Adaptive alpha/beta via grid search (minimize MAE)
    - Trend clamping to prevent unrealistic projections
    - Flatline detection and handling
    - Robust to outliers via median imputation
    - Warm-start from previous model state
    
    References:
    - Holt (1957): Original exponential smoothing with trend
    - Hyndman & Athanasopoulos (2018): Alpha ∈ [0.05, 0.95], Beta ∈ [0.01, 0.30]
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        max_trend_per_hour: float = 5.0,
        flatline_epsilon: float = 1e-3,
        enable_adaptive: bool = True,
        min_samples_for_adaptive: int = 30
    ):
        """
        Initialize Holt's linear trend model.
        
        Args:
            alpha: Level smoothing parameter (0.05-0.95, per Hyndman & Athanasopoulos 2018)
            beta: Trend smoothing parameter (0.01-0.30, per Hyndman & Athanasopoulos 2018)
            max_trend_per_hour: Maximum allowable trend rate (health points/hour)
            flatline_epsilon: Threshold for detecting flat/constant signal
            enable_adaptive: Enable adaptive alpha/beta tuning
            min_samples_for_adaptive: Minimum samples required for adaptive tuning
        """
        self.alpha = np.clip(alpha, 0.05, 0.95)
        self.beta = np.clip(beta, 0.01, 0.30)
        self.max_trend_per_hour = max_trend_per_hour
        self.flatline_epsilon = flatline_epsilon
        self.enable_adaptive = enable_adaptive
        self.min_samples_for_adaptive = min_samples_for_adaptive
        
        # Model state (fitted parameters)
        self.level: float = 0.0
        self.trend: float = 0.0
        self.std_error: float = 1.0
        self.dt_hours: float = 1.0
        self.last_timestamp: Optional[pd.Timestamp] = None
        self.n_fitted: int = 0
        
        # Diagnostic metrics
        self.level_history: List[float] = []
        self.trend_history: List[float] = []
        self.residuals: List[float] = []
    
    def fit(self, health_series: pd.Series) -> None:
        """
        Fit Holt's linear trend model to historical health data.
        
        Process:
        1. Detect and handle outliers (robust median imputation)
        2. Compute time intervals (dt_hours)
        3. Detect flatline/low-variance signals
        4. Initialize level and trend
        5. Adaptive alpha/beta tuning (if enabled and sufficient samples)
        6. Forward pass: update level and trend with exponential smoothing
        7. Compute residual standard error for uncertainty quantification
        
        Args:
            health_series: Time-indexed health values (pd.Series with DatetimeIndex)
        """
        if health_series is None or len(health_series) == 0:
            Console.warn("Empty health series provided", component="DEGRADE")
            return
        
        # Prepare series (robust median imputation)
        health_values = health_series.copy().astype(float)
        health_values = self._detect_and_remove_outliers(health_values)
        median_val = float(np.nanmedian(health_values)) if len(health_values) > 0 else 0.0
        if not np.isfinite(median_val):
            median_val = 0.0
        health_values = health_values.fillna(median_val)
        
        n = len(health_values)
        self.n_fitted = n
        
        # Compute time intervals
        if isinstance(health_series.index, pd.DatetimeIndex):
            time_diffs = health_series.index.to_series().diff().dt.total_seconds() / 3600.0
            self.dt_hours = float(time_diffs.median()) if len(time_diffs) > 0 else 1.0
            if not np.isfinite(self.dt_hours) or self.dt_hours <= 0:
                self.dt_hours = 1.0
            self.last_timestamp = health_series.index[-1]
        else:
            self.dt_hours = 1.0
            self.last_timestamp = None
        
        # Detect flatline/low-variance series
        span = float(np.nanmax(health_values) - np.nanmin(health_values))
        variance = float(np.nanvar(health_values))
        is_flatline = (span < self.flatline_epsilon) or (variance < self.flatline_epsilon ** 2)
        
        # Initialize level and trend
        self.level = float(health_values.iloc[0])
        self.trend = (float(health_values.iloc[1]) - float(health_values.iloc[0])) / self.dt_hours if n > 1 else 0.0
        
        if is_flatline:
            self.trend = 0.0
            Console.info("Flatline detected; zeroing trend", component="DEGRADE")
        
        # Adaptive smoothing (if enabled and sufficient samples)
        if self.enable_adaptive and n >= self.min_samples_for_adaptive:
            try:
                self.alpha, self.beta = self._adaptive_smoothing(health_values)
                Console.info(f"Adaptive smoothing: alpha={self.alpha:.3f}, beta={self.beta:.3f}", component="DEGRADE")
            except Exception as e:
                Console.warn(f"Adaptive smoothing failed: {e}", component="DEGRADE", error_type=type(e).__name__, error=str(e)[:200])
        
        # Forward pass (Holt's equations)
        self.level_history = [self.level]
        self.trend_history = [self.trend]
        self.residuals = []
        
        for i in range(1, n):
            obs = float(health_values.iloc[i])
            prev_level = self.level
            prev_trend = self.trend
            
            # Level update
            self.level = self.alpha * obs + (1 - self.alpha) * (prev_level + prev_trend)
            
            # Trend update with clamping
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * prev_trend
            self.trend = np.clip(self.trend, -self.max_trend_per_hour * self.dt_hours, self.max_trend_per_hour * self.dt_hours)
            
            # Track history and residuals
            self.level_history.append(self.level)
            self.trend_history.append(self.trend)
            
            # One-step-ahead forecast error
            forecast_error = obs - (prev_level + prev_trend)
            self.residuals.append(forecast_error)
        
        # Zero trend if flatline
        if is_flatline:
            self.trend = 0.0
            self.trend_history = [0.0] * len(self.trend_history)
            self.residuals = [0.0]
        
        # Compute residual standard error
        self.std_error = float(np.std(self.residuals)) if len(self.residuals) > 0 else 1.0
        if not np.isfinite(self.std_error) or self.std_error < 1e-6:
            self.std_error = 1.0
        
        Console.info(
            f"Fitted: level={self.level:.2f}, trend={self.trend:.4f}/hr, "
            f"std_error={self.std_error:.2f}, n={n}",
            component="DEGRADE"
        )
    
    def predict(
        self,
        steps: int,
        dt_hours: Optional[float] = None,
        confidence_level: float = 0.95
    ) -> DegradationForecast:
        """
        Generate multi-step ahead forecast with properly widening uncertainty bounds.
        
        Forecast Equation (Holt's linear trend):
        - y_hat[t+h] = L[t] + h*T[t]
        
        Uncertainty Growth (per Hyndman & Athanasopoulos 2018, Chapter 8.1):
        The exact variance formula for Holt's method h-step-ahead forecast is:
        
        Var[e_t(h)] = sigma^2 * [1 + sum_{j=1}^{h-1}(alpha + alpha*beta*j)^2]
        
        Where:
        - sigma^2 = residual variance from training
        - alpha = level smoothing parameter  
        - beta = trend smoothing parameter
        - h = forecast horizon
        
        This creates proper "cone-shaped" prediction intervals that widen over time,
        reflecting increasing uncertainty in longer-range forecasts.
        
        Reference: Hyndman, R.J. & Athanasopoulos, G. (2018) Forecasting: 
        Principles and Practice, Chapter 8.1, otexts.com/fpp2/prediction-intervals.html
        
        Args:
            steps: Number of forecast steps
            dt_hours: Time interval per step (uses fitted dt_hours if None)
            confidence_level: Confidence level for bounds (default 0.95 = +/-1.96 sigma)
        
        Returns:
            DegradationForecast with timestamps, point forecast, and widening uncertainty bounds
        """
        if dt_hours is None:
            dt_hours = self.dt_hours
        
        # Generate forecast timestamps
        if self.last_timestamp is not None:
            forecast_timestamps = pd.date_range(
                start=self.last_timestamp + pd.Timedelta(hours=dt_hours),
                periods=steps,
                freq=pd.Timedelta(hours=dt_hours)
            )
        else:
            forecast_timestamps = pd.date_range(
                start=pd.Timestamp.now(),
                periods=steps,
                freq=pd.Timedelta(hours=dt_hours)
            )
        
        # Point forecast: y_hat[t+h] = L[t] + h*T[t]
        horizons = np.arange(1, steps + 1)
        point_forecast = self.level + horizons * self.trend
        
        # PROPER WIDENING PREDICTION INTERVALS (Hyndman & Athanasopoulos 2018)
        # Var[e_t(h)] = sigma^2 * [1 + sum_{j=1}^{h-1}(alpha + alpha*beta*j)^2]
        # This creates cone-shaped intervals that widen with forecast horizon
        #
        # VECTORIZED IMPLEMENTATION (v10.2.1):
        # For h=1..steps, we need cumsum_h = sum_{j=1}^{h-1} (alpha + alpha*beta*j)^2
        # Let c_j = alpha + alpha*beta*j = alpha*(1 + beta*j)
        # Then c_j^2 = alpha^2 * (1 + beta*j)^2
        # cumsum_h = sum_{j=1}^{h-1} alpha^2 * (1 + beta*j)^2
        # Use np.cumsum for O(steps) instead of O(steps^2)
        if steps > 0:
            j_vals = np.arange(1, steps)  # j = 1, 2, ..., steps-1
            coeffs_sq = (self.alpha * (1 + self.beta * j_vals)) ** 2
            cumsum_arr = np.cumsum(coeffs_sq)  # cumsum[k] = sum_{j=1}^{k+1} c_j^2
            # variance_multiplier[h-1] = 1 + sum_{j=1}^{h-1} c_j^2
            # For h=1: 1 + 0 = 1
            # For h=2: 1 + c_1^2 = 1 + cumsum_arr[0]
            # For h=k: 1 + cumsum_arr[k-2] (k >= 2)
            variance_multiplier = np.ones(steps)
            if steps > 1:
                variance_multiplier[1:] = 1.0 + cumsum_arr
        else:
            variance_multiplier = np.array([1.0])
        
        # Standard error at each horizon (widening cone)
        std_forecast = self.std_error * np.sqrt(variance_multiplier)
        
        # Confidence bounds (z-score for confidence level)
        from scipy import stats as sp_stats
        z_score = sp_stats.norm.ppf((1 + confidence_level) / 2.0)
        lower_bound = point_forecast - z_score * std_forecast
        upper_bound = point_forecast + z_score * std_forecast
        
        # CRITICAL: Clamp all forecast values to valid health range [0, 100]
        # Health index cannot exceed 100% or go below 0%
        point_forecast = np.clip(point_forecast, 0.0, 100.0)
        lower_bound = np.clip(lower_bound, 0.0, 100.0)
        upper_bound = np.clip(upper_bound, 0.0, 100.0)
        
        return DegradationForecast(
            timestamps=forecast_timestamps,
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            std_error=self.std_error,
            level=self.level,
            trend=self.trend
        )
    
    def update_incremental(self, new_observation: float) -> None:
        """
        Update model with new observation (online learning).
        
        Applies one iteration of Holt's equations to incorporate new data
        without full retraining. Useful for continuous model updates.
        
        Args:
            new_observation: Latest health value
        """
        if not np.isfinite(new_observation):
            return
        
        prev_level = self.level
        prev_trend = self.trend
        
        # Level update
        self.level = self.alpha * new_observation + (1 - self.alpha) * (prev_level + prev_trend)
        
        # Trend update with clamping
        self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * prev_trend
        self.trend = np.clip(self.trend, -self.max_trend_per_hour * self.dt_hours, self.max_trend_per_hour * self.dt_hours)
        
        # Update residual (one-step-ahead error)
        forecast_error = new_observation - (prev_level + prev_trend)
        self.residuals.append(forecast_error)
        
        # Update std_error (exponential moving average)
        if len(self.residuals) > 0:
            self.std_error = float(np.std(self.residuals[-100:]))  # Use last 100 residuals
            if not np.isfinite(self.std_error) or self.std_error < 1e-6:
                self.std_error = 1.0
        
        self.n_fitted += 1
    
    def get_parameters(self) -> Dict[str, float]:
        """Export model state for persistence to ACM_ForecastingState"""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "level": self.level,
            "trend": self.trend,
            "std_error": self.std_error,
            "dt_hours": self.dt_hours,
            "n_fitted": float(self.n_fitted)
        }
    
    def set_parameters(self, params: Dict[str, float]) -> None:
        """Restore model state from saved parameters"""
        self.alpha = params.get("alpha", 0.3)
        self.beta = params.get("beta", 0.1)
        self.level = params.get("level", 0.0)
        self.trend = params.get("trend", 0.0)
        self.std_error = params.get("std_error", 1.0)
        self.dt_hours = params.get("dt_hours", 1.0)
        self.n_fitted = int(params.get("n_fitted", 0))
        
        Console.info(
            f"Restored state: level={self.level:.2f}, trend={self.trend:.4f}/hr, "
            f"std_error={self.std_error:.2f}",
            component="DEGRADE"
        )
    
    def _detect_and_remove_outliers(self, series: pd.Series, n_std: float = 3.0) -> pd.Series:
        """Detect and remove outliers via z-score (n_std standard deviations)"""
        if len(series) < 10:
            return series  # Too few samples for robust outlier detection
        
        mean = series.mean()
        std = series.std()
        
        if std < 1e-6:
            return series  # Near-constant signal
        
        z_scores = np.abs((series - mean) / std)
        outliers = z_scores > n_std
        
        if outliers.sum() > 0:
            Console.info(f"Detected {outliers.sum()} outliers (z > {n_std})", component="DEGRADE")
            series = series.copy()
            series[outliers] = np.nan
        
        return series
    
    def _adaptive_smoothing(self, health_values: pd.Series) -> Tuple[float, float]:
        """
        Adaptive alpha/beta tuning via expanding-window time-series cross-validation.
        
        OPTIMIZED (v10.2.1):
        - Coarse-to-fine grid search: 4x4 initial grid, then refine around best
        - Max 10 CV folds (not n//20) for speed
        - Vectorized Holt's inner loop using numpy
        
        This implements proper time-series CV (not simple grid search) per
        Hyndman & Athanasopoulos (2018), Section 5.4: "Evaluating forecast accuracy".
        
        Grid bounds per Hyndman & Athanasopoulos (2018):
        - Alpha: [0.05, 0.95] (level smoothing)
        - Beta: [0.01, 0.30] (trend smoothing)
        
        Returns:
            (optimal_alpha, optimal_beta)
        """
        n = len(health_values)
        
        # CV parameters - cap folds for speed
        min_train_size = max(20, n // 4)  # At least 20 or 25% of data
        forecast_horizon = min(12, n // 10)  # Forecast horizon (cap at 12 steps)
        max_cv_folds = 10  # Cap CV folds for speed
        step_size = max(1, (n - min_train_size - forecast_horizon) // max_cv_folds)
        
        if n < min_train_size + forecast_horizon + 5:
            # Insufficient data for CV - fall back to simple grid search
            return self._simple_grid_search(health_values)
        
        # Convert to numpy for speed - ensure ndarray type
        health_arr: np.ndarray = np.asarray(health_values.values, dtype=float)
        
        # PHASE 1: Coarse grid search (4x4 = 16 combinations)
        alpha_grid = np.array([0.1, 0.3, 0.6, 0.9])
        beta_grid = np.array([0.02, 0.08, 0.15, 0.25])
        
        best_alpha = self.alpha
        best_beta = self.beta
        best_cv_error = float("inf")
        
        for alpha_candidate in alpha_grid:
            for beta_candidate in beta_grid:
                cv_error = self._compute_cv_error_vectorized(
                    health_arr, alpha_candidate, beta_candidate,
                    min_train_size, forecast_horizon, step_size, n
                )
                if cv_error < best_cv_error:
                    best_cv_error = cv_error
                    best_alpha = alpha_candidate
                    best_beta = beta_candidate
        
        # PHASE 2: Fine-tune around best (3x3 = 9 combinations)
        alpha_fine = np.clip([best_alpha - 0.1, best_alpha, best_alpha + 0.1], 0.05, 0.95)
        beta_fine = np.clip([best_beta - 0.05, best_beta, best_beta + 0.05], 0.01, 0.30)
        
        for alpha_candidate in alpha_fine:
            for beta_candidate in beta_fine:
                cv_error = self._compute_cv_error_vectorized(
                    health_arr, alpha_candidate, beta_candidate,
                    min_train_size, forecast_horizon, step_size, n
                )
                if cv_error < best_cv_error:
                    best_cv_error = cv_error
                    best_alpha = alpha_candidate
                    best_beta = beta_candidate
        
        return best_alpha, best_beta
    
    def _compute_cv_error_vectorized(
        self, 
        health_arr: np.ndarray, 
        alpha: float, 
        beta: float,
        min_train_size: int,
        forecast_horizon: int,
        step_size: int,
        n: int
    ) -> float:
        """Compute CV error using vectorized Holt's filter"""
        cv_errors = []
        trend_clamp = self.max_trend_per_hour * self.dt_hours
        
        for train_end in range(min_train_size, n - forecast_horizon, step_size):
            train_data = health_arr[:train_end]
            
            # Vectorized Holt's exponential smoothing
            level = train_data[0]
            trend = (train_data[1] - train_data[0]) / self.dt_hours if len(train_data) > 1 else 0.0
            
            # Process all observations (can't fully vectorize due to recursive nature)
            # But we use numpy scalar ops which are faster
            for i in range(1, len(train_data)):
                obs = train_data[i]
                prev_level = level
                prev_trend = trend
                level = alpha * obs + (1.0 - alpha) * (prev_level + prev_trend)
                trend = beta * (level - prev_level) + (1.0 - beta) * prev_trend
                if trend > trend_clamp:
                    trend = trend_clamp
                elif trend < -trend_clamp:
                    trend = -trend_clamp
            
            # Forecast h steps ahead
            forecast = level + forecast_horizon * trend
            actual = health_arr[train_end + forecast_horizon - 1]
            cv_errors.append(abs(forecast - actual))
        
        return float(np.mean(cv_errors)) if cv_errors else float("inf")
    
    def _simple_grid_search(self, health_values: pd.Series) -> Tuple[float, float]:
        """
        Fallback simple grid search when insufficient data for proper CV.
        Minimizes one-step-ahead MAE.
        """
        alpha_grid = np.linspace(0.05, 0.95, 10)
        beta_grid = np.linspace(0.01, 0.30, 10)
        
        best_alpha = self.alpha
        best_beta = self.beta
        best_mae = float("inf")
        
        for alpha_candidate in alpha_grid:
            for beta_candidate in beta_grid:
                mae = self._evaluate_smoothing_params(health_values, alpha_candidate, beta_candidate)
                
                if mae < best_mae:
                    best_mae = mae
                    best_alpha = alpha_candidate
                    best_beta = beta_candidate
        
        return best_alpha, best_beta
    
    def _evaluate_smoothing_params(self, health_values: pd.Series, alpha: float, beta: float) -> float:
        """Evaluate MAE for given alpha/beta parameters"""
        n = len(health_values)
        if n < 2:
            return float("inf")
        
        # Initialize
        level = float(health_values.iloc[0])
        trend = (float(health_values.iloc[1]) - float(health_values.iloc[0])) / self.dt_hours if n > 1 else 0.0
        
        errors = []
        
        # Forward pass
        for i in range(1, n):
            obs = float(health_values.iloc[i])
            prev_level = level
            prev_trend = trend
            
            # One-step-ahead forecast
            forecast = prev_level + prev_trend
            errors.append(abs(obs - forecast))
            
            # Update level and trend
            level = alpha * obs + (1 - alpha) * (prev_level + prev_trend)
            trend = beta * (level - prev_level) + (1 - beta) * prev_trend
            trend = np.clip(trend, -self.max_trend_per_hour * self.dt_hours, self.max_trend_per_hour * self.dt_hours)
        
        mae = float(np.mean(errors)) if len(errors) > 0 else float("inf")
        return mae
