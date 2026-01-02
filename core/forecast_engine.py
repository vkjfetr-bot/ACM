"""
Forecast Engine Orchestrator (v10.0.0)

Unified forecasting and RUL estimation engine replacing duplicate logic from:
- forecasting.py (2816-2943 lines)
- rul_engine.py (1914-2005 lines)

Orchestrates the complete forecasting workflow:
1. Load health timeline with quality checks
2. Load/restore persistent state with optimistic locking
3. Load adaptive configuration (equipment-specific overrides)
4. Check for auto-tuning trigger (data volume threshold)
5. Fit degradation model (Holt's exponential smoothing)
6. Generate health forecast with uncertainty
7. Estimate RUL via Monte Carlo simulation
8. Compute failure probabilities and statistics
9. Rank sensor attributions
10. Compute forecast quality metrics
11. Save all outputs to SQL via OutputManager
12. Update persistent state

v10.1.0 Enhancements:
- Regime-conditioned forecasting with per-regime degradation rates
- OMR/drift context integration for forecast confidence adjustment
- Per-regime RUL estimates and hazard rates
- Unified ACM_ForecastContext table for complete diagnostic context

References:
- All module-specific references in respective files
- ISO 13381-1:2015: Prognostics workflow standards

Future R&D Entrypoints (M15):
---------------------------------
TODO: FAULT_FAMILY_PREDICTION - Classify failure modes into fault families
      - Table: ACM_FaultFamilies (FaultID, EquipID, FaultName, SensorSignature, DetectionMethod)
      - Method: _predict_fault_family() using sensor attribution clusters
      - Integration: Add FaultFamily column to ACM_RUL after sensor attribution step
      
TODO: EPISODE_CLUSTERING - Group similar anomaly episodes for pattern mining
      - Table: ACM_BehaviourDeviation (EpisodeID, ClusterID, ClusterLabel, SimilarityScore)
      - Method: _cluster_episodes() using episode diagnostics features
      - Integration: Post-process ACM_EpisodeDiagnostics after forecasting complete
      
TODO: MULTIVARIATE_DEGRADATION - Extend LinearTrendModel to vector autoregression
      - Model: VARDegradationModel handling correlated sensor degradation
      - Method: _fit_var_model() for sensor-level multivariate forecasting
      - Integration: Replace LinearTrendModel when multi-sensor data available
      
TODO: CAUSAL_ATTRIBUTION - Pearl-style counterfactual sensor importance
      - Method: _compute_counterfactual_rul() in SensorAttributor
      - Compute RUL delta when each sensor is zeroed out
      - Replace heuristic attribution with causal importance scores
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from core.health_tracker import HealthTimeline, HealthQuality, DataSummary
from core.degradation_model import LinearTrendModel
from core.failure_probability import compute_failure_statistics, health_to_failure_probability
from core.rul_estimator import RULEstimator, RULEstimate
from core.sensor_attribution import SensorAttributor
from core.metrics import compute_comprehensive_metrics, log_metrics_summary, compute_forecast_diagnostics, log_forecast_diagnostics
from core.state_manager import StateManager, AdaptiveConfigManager, ForecastingState
from core.output_manager import OutputManager
from core.observability import Console, Span
# V11: Reliability gating for RUL predictions
from core.confidence import (
    ReliabilityStatus,
    compute_rul_confidence,
    check_rul_reliability,
)
# V11 CRITICAL-1: Only import MaturityState, NOT load_model_state_from_sql
# Model state is passed from acm_main to avoid race conditions
from core.model_lifecycle import MaturityState


# ========================================================================
# v10.1.0: Regime-Conditioned Forecasting Support
# ========================================================================

@dataclass
class RegimeStats:
    """Per-regime statistics for conditioned forecasting."""
    regime_label: int
    health_state: str  # healthy, suspect, critical
    degradation_rate: float  # Health units/hour in this regime (Theil-Sen robust estimate)
    degradation_rate_lower: float  # 95% CI lower bound (bootstrap)
    degradation_rate_upper: float  # 95% CI upper bound (bootstrap)
    degradation_r_squared: float  # R-squared of trend fit (0-1, higher = more reliable)
    health_mean: float
    health_std: float
    dwell_fraction: float  # % time spent in regime
    transition_count: int
    failure_threshold: float  # Regime-adjusted failure threshold
    sample_count: int


@dataclass  
class ForecastContext:
    """Unified context for regime-aware forecasting decisions."""
    current_regime: Optional[int]
    regime_confidence: float
    current_omr_z: Optional[float]
    omr_trend: str  # stable, increasing, decreasing
    omr_top_contributors: List[Dict[str, Any]]
    current_drift_z: Optional[float]
    drift_trend: str  # stable, increasing, decreasing
    health_trend: str  # improving, stable, degrading
    data_quality: float  # 0-1
    active_defects: int
    retraining_recommended: bool
    retraining_reason: Optional[str]


class ForecastEngine:
    """
    Unified forecasting and RUL estimation engine.
    
    Design Philosophy:
    - Single entry point for all forecasting operations
    - Replaces forecasting.estimate_rul() and rul_engine.run_rul()
    - Leverages 8 specialized modules for clean separation of concerns
    - SQL-only operation (no file mode)
    - Production-scale with optimistic locking and connection pooling
    
    Usage:
        engine = ForecastEngine(
            sql_client=sql_client,
            output_manager=output_mgr,
            equip_id=1,
            run_id="run_123"
        )
        
        results = engine.run_forecast()
        
        if results['success']:
            print(f"RUL P50: {results['rul_p50']} hours")
            print(f"Top sensors: {results['top_sensors']}")
    """
    
    def __init__(
        self,
        sql_client: Any,
        output_manager: OutputManager,
        equip_id: int,
        run_id: str,
        config: Optional[Dict[str, Any]] = None,
        model_state: Optional[Any] = None
    ):
        """
        Initialize forecast engine.
        
        Args:
            sql_client: Database connection (pyodbc)
            output_manager: OutputManager instance for SQL writes
            equip_id: Equipment ID
            run_id: ACM run identifier
            config: Configuration dictionary (optional, uses ACM_AdaptiveConfig if None)
            model_state: Pre-loaded ModelState from acm_main (avoids race condition with SQL)
        """
        self.sql_client = sql_client
        self.output_manager = output_manager
        self.equip_id = equip_id
        self.run_id = run_id
        self.config = config or {}
        self._model_state = model_state  # V11 CRITICAL-1: Use cached state to avoid race condition
        
        # Initialize managers
        self.state_mgr = StateManager(sql_client=sql_client)
        self.config_mgr = AdaptiveConfigManager(sql_client=sql_client)
        
        # Results storage
        self.results: Dict[str, Any] = {}
    
    def run_forecast(self) -> Dict[str, Any]:
        """
        Execute complete forecasting workflow.
        
        Workflow Steps:
        1. Load health timeline with quality checks
        2. Load/restore persistent state
        3. Load adaptive configuration
        4. Check auto-tuning trigger
        5. Fit degradation model
        6. Generate forecast and estimate RUL
        7. Rank sensor attributions
        8. Compute metrics
        9. Write outputs to SQL
        10. Update state
        
        Returns:
            Dictionary with keys:
            - 'success': bool
            - 'rul_p50': float (median RUL hours)
            - 'rul_p10': float (pessimistic bound)
            - 'rul_p90': float (optimistic bound)
            - 'top_sensors': str (top 3 sensor names)
            - 'data_quality': str (HealthQuality enum value)
            - 'tables_written': List[str] (SQL tables updated)
            - 'error': str (if success=False)
        """
        with Span("forecast.run", equip_id=self.equip_id):
            try:
                # Step 1: Load health timeline with DataSummary (M3)
                health_df, data_quality, data_summary = self._load_health_timeline()
                
                if health_df is None:
                    Console.warn("No health data available; skipping forecast",
                                 component="FORECAST", equip_id=self.equip_id, run_id=self.run_id)
                    return {
                        'success': False,
                        'error': 'No health data available',
                        'data_quality': 'NONE'
                    }
                
                # V11: Check model maturity BEFORE data quality gating
                # During COLDSTART/LEARNING, we allow sparse data and mark as NOT_RELIABLE
                maturity_state, _, _ = self._get_model_maturity_state()
                is_early_maturity = maturity_state in ('COLDSTART', 'LEARNING')
                
                # M3.2 + M14 + V11: Quality gating with maturity awareness
                # FLAT/NOISY are always blockers (can't forecast constant or erratic data)
                # SPARSE is allowed during early maturity (will be marked NOT_RELIABLE)
                if data_quality in [HealthQuality.FLAT, HealthQuality.NOISY]:
                    quality_messages = {
                        HealthQuality.FLAT: "Health data shows no variance - equipment may be in steady state",
                        HealthQuality.NOISY: "Health data has excessive noise - consider smoothing or filtering"
                    }
                    friendly_msg = quality_messages.get(data_quality, f"Data quality issue: {data_quality.value}")
                    Console.warn(f"{friendly_msg}; skipping forecast (not fatal)",
                                 component="FORECAST", equip_id=self.equip_id, run_id=self.run_id,
                                 quality=data_quality.value)
                    return {
                        'success': False,
                        'error': friendly_msg,
                        'data_quality': data_quality.value
                    }
                
                # V11: SPARSE data handling depends on maturity
                if data_quality == HealthQuality.SPARSE:
                    if is_early_maturity:
                        # During coldstart/learning, proceed with sparse data but warn
                        Console.warn(
                            f"SPARSE data during {maturity_state} phase - proceeding with limited confidence",
                            component="FORECAST", equip_id=self.equip_id, run_id=self.run_id,
                            quality=data_quality.value, n_samples=len(health_df) if health_df is not None else 0
                        )
                    else:
                        # If model is CONVERGED but data is sparse, this indicates a data pipeline issue
                        friendly_msg = "Insufficient data samples - need more historical data for reliable forecast"
                        Console.warn(f"{friendly_msg}; skipping forecast (not fatal)",
                                     component="FORECAST", equip_id=self.equip_id, run_id=self.run_id,
                                     quality=data_quality.value)
                        return {
                            'success': False,
                            'error': friendly_msg,
                            'data_quality': data_quality.value
                        }
                
                if data_quality == HealthQuality.GAPPY:
                    Console.warn("GAPPY data detected - proceeding with available data (historical replay mode)",
                                 component="FORECAST", equip_id=self.equip_id, run_id=self.run_id)
                
                # Step 2: Load persistent state
                state = self.state_mgr.load_state(self.equip_id)
                if state is None:
                    state = ForecastingState(equip_id=self.equip_id, state_version=0)
                
                # Step 3: Load adaptive configuration
                forecast_config = self._load_forecast_config()
                
                # M3.3: Use dt_hours from DataSummary (computed once, not recomputed)
                if data_summary and data_summary.dt_hours > 0:
                    forecast_config['dt_hours'] = data_summary.dt_hours
                
                # Step 4: Check auto-tuning trigger
                current_volume = len(health_df)
                state.data_volume_analyzed += current_volume
                should_tune = self.config_mgr.should_tune(self.equip_id, state.data_volume_analyzed)
                
                if should_tune:
                    Console.info(f"Auto-tuning triggered at DataVolume={state.data_volume_analyzed}",
                                 component="FORECAST", equip_id=self.equip_id, data_volume=state.data_volume_analyzed)
                    # TODO: Implement grid search tuning in future PR
                    # For now, use configured values
                
                # Step 5: Fit degradation model
                degradation_model = self._fit_degradation_model(health_df, forecast_config, state)
                
                # Step 6: Generate forecast and estimate RUL
                forecast_results = self._generate_forecast_and_rul(
                    health_df, degradation_model, forecast_config
                )
                
                # Step 6b: Compute forecast diagnostics (M9)
                diagnostics = compute_forecast_diagnostics(forecast_results, data_summary)
                log_forecast_diagnostics(diagnostics)
                
                # Step 7: Rank sensor attributions (M10)
                sensor_attributions = self._load_sensor_attributions()
                # Add to forecast_results for unified access
                attributor = SensorAttributor(sql_client=self.sql_client)
                top_sensors_str = attributor.format_top_n(sensor_attributions, n=3)
                forecast_results['sensor_attributions'] = sensor_attributions
                forecast_results['top_sensors'] = top_sensors_str
                
                # Step 8: Compute metrics (if we have historical forecasts to compare)
                # metrics = self._compute_metrics(forecast_results)
                
                # Step 9: Write outputs to SQL (M13: pass diagnostics for operator context)
                tables_written = self._write_outputs(
                    forecast_results, sensor_attributions, diagnostics, data_summary
                )
                
                # Step 9b (v10.1.0): Regime-conditioned forecasting
                # Only run if regime data is available and feature is enabled
                enable_regime_forecasting = bool(forecast_config.get('enable_regime_conditioned', True))
                if enable_regime_forecasting:
                    try:
                        regime_tables = self._run_regime_conditioned_forecasting(
                            health_df=health_df,
                            degradation_model=degradation_model,
                            forecast_config=forecast_config,
                            forecast_results=forecast_results
                        )
                        tables_written.extend(regime_tables)
                    except Exception as e:
                        # Non-fatal: regime forecasting is optional enhancement
                        Console.warn(f"Regime-conditioned forecasting skipped: {e}",
                                     component="FORECAST", equip_id=self.equip_id, run_id=self.run_id)
                
                # Step 10: Update state with diagnostics (M9)
                state.model_coefficients_json = degradation_model.get_parameters()
                state.updated_at = datetime.now()
                
                # Store diagnostics in state for persistence
                # Include health values in the diagnostics dict for state persistence
                diagnostics['last_health_value'] = float(health_df['HealthIndex'].iloc[-1])
                diagnostics['last_health_timestamp'] = health_df['Timestamp'].iloc[-1].isoformat() if hasattr(health_df['Timestamp'].iloc[-1], 'isoformat') else str(health_df['Timestamp'].iloc[-1])
                state.last_forecast_json = diagnostics
                state.recent_mae = diagnostics.get('forecast_std')  # Use forecast_std as proxy
                
                self.state_mgr.save_state(state)
                
                # Build success response
                return {
                    'success': True,
                    'rul_p50': forecast_results['rul_p50'],
                    'rul_p10': forecast_results['rul_p10'],
                    'rul_p90': forecast_results['rul_p90'],
                    'top_sensors': forecast_results['top_sensors'],
                    'data_quality': data_quality.value,
                    'tables_written': tables_written,
                    'state_version': state.state_version
                }
                
            except Exception as e:
                Console.error(f"Forecast failed: {e}",
                              component="FORECAST", equip_id=self.equip_id, run_id=self.run_id,
                              error_type=type(e).__name__, error_msg=str(e)[:500])
                return {
                    'success': False,
                    'error': str(e),
                    'data_quality': 'UNKNOWN'
                }
    
    def _load_health_timeline(self) -> tuple[Optional[pd.DataFrame], HealthQuality, Optional[DataSummary]]:
        """
        Load health timeline with quality assessment and data summary.
        
        Uses HealthTimeline with rolling window (M3.1) to avoid full-history overfitting.
        Returns DataSummary (M3.3) for efficient metadata access by ForecastEngine.
        
        Returns:
            Tuple of (DataFrame, HealthQuality, DataSummary)
        """
        # Get configurable rolling window (default 90 days = 2160 hours)
        history_window_hours = float(
            self.config_mgr.get_config(self.equip_id, 'history_window_hours', 2160.0)
        )
        
        tracker = HealthTimeline(
            sql_client=self.sql_client,
            equip_id=self.equip_id,
            run_id=self.run_id,
            output_manager=self.output_manager,
            min_train_samples=int(self.config_mgr.get_config(self.equip_id, 'min_train_samples', 200)),
            max_gap_hours=float(self.config_mgr.get_config(self.equip_id, 'max_gap_hours', 720.0)),
            history_window_hours=history_window_hours  # M3.1: Rolling window
        )
        
        health_df, quality = tracker.load_from_sql()
        
        # M3.3: Get data summary for ForecastEngine consumption
        data_summary = tracker.get_data_summary(health_df) if health_df is not None else None
        
        if quality != HealthQuality.OK and data_summary:
            Console.warn(
                f"Data quality issue: {data_summary.quality_reason}",
                component="FORECAST", equip_id=self.equip_id, quality=quality.value
            )
        
        if data_summary:
            Console.info(
                f"Data summary: n_samples={data_summary.n_samples}, "
                f"dt_hours={data_summary.dt_hours:.2f}, window={data_summary.window_hours:.0f}h",
                component="FORECAST", equip_id=self.equip_id, n_samples=data_summary.n_samples,
                dt_hours=data_summary.dt_hours, window_hours=data_summary.window_hours
            )
        
        return health_df, quality, data_summary
    
    def _load_forecast_config(self) -> Dict[str, Any]:
        """Load forecast configuration with equipment overrides"""
        config = self.config_mgr.get_all_configs(self.equip_id)
        
        # Apply any overrides from self.config
        forecast_section = self.config.get('forecasting', {})
        config.update(forecast_section)
        
        # M11: Ensure forecast horizon and resolution have defaults
        # forecast_horizon_hours: How far ahead to forecast (default 168h = 7 days)
        # forecast_resolution_hours: Step size for forecast points (defaults to data cadence)
        if 'forecast_horizon_hours' not in config:
            config['forecast_horizon_hours'] = 168.0
        if 'forecast_resolution_hours' not in config:
            config['forecast_resolution_hours'] = None  # Will use dt_hours from data
        
        # Alias for backward compatibility
        config['max_forecast_hours'] = config.get('max_forecast_hours', config['forecast_horizon_hours'])
        
        Console.info(
            f"Loaded forecast config: alpha={config.get('alpha', 0.3):.2f}, "
            f"beta={config.get('beta', 0.1):.2f}, "
            f"failure_threshold={config.get('failure_threshold', 70.0):.1f}, "
            f"horizon={config.get('forecast_horizon_hours', 168.0):.0f}h",
            component="FORECAST", equip_id=self.equip_id
        )
        
        return config
    
    def _fit_degradation_model(
        self,
        health_df: pd.DataFrame,
        forecast_config: Dict[str, Any],
        state: ForecastingState
    ) -> LinearTrendModel:
        """Fit degradation model with warm-start from previous state"""
        with Span("forecast.fit_degradation", n_samples=len(health_df)):
            # Create model with adaptive config
            model = LinearTrendModel(
                alpha=float(forecast_config.get('alpha', 0.3)),
                beta=float(forecast_config.get('beta', 0.1)),
                max_trend_per_hour=float(forecast_config.get('max_trend_per_hour', 5.0)),
                enable_adaptive=bool(forecast_config.get('enable_adaptive_smoothing', True))
            )
            
            # Warm-start from previous state if available
            if state.model_coefficients_json:
                try:
                    model.set_parameters(state.model_coefficients_json)
                    Console.info("Warm-started degradation model from previous state",
                                 component="FORECAST", equip_id=self.equip_id)
                except Exception as e:
                    Console.warn(f"Failed to warm-start model: {e}",
                                 component="FORECAST", equip_id=self.equip_id, error=str(e))
            
            # Prepare health series
            health_series = pd.Series(
                health_df['HealthIndex'].values,
                index=pd.to_datetime(health_df['Timestamp'])
            )
            
            # Fit model
            model.fit(health_series)
            
            return model
    
    def _generate_forecast_and_rul(
        self,
        health_df: pd.DataFrame,
        degradation_model: LinearTrendModel,
        forecast_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate health forecast and RUL estimate"""
        # Extract config values
        failure_threshold = float(forecast_config.get('failure_threshold', 70.0))
        confidence_level = float(forecast_config.get('confidence_min', 0.80))
        n_simulations = int(forecast_config.get('monte_carlo_simulations', 1000))
        
        # ADAPTIVE FORECAST HORIZON (v10.1.0, Issue #11)
        # Set horizon = max(7 days, 3 * estimated_RUL) to forecast into the "interesting" region
        # Fast-degrading equipment needs shorter horizons, slow degradation needs longer
        base_horizon_hours = float(forecast_config.get('forecast_horizon_hours', 168.0))
        
        # Estimate RUL from current health and trend
        current_health = float(health_df['HealthIndex'].iloc[-1]) if len(health_df) > 0 else 80.0
        trend = degradation_model.trend  # Health points per dt_hours
        
        if trend < -0.001:  # Degrading
            # Rough RUL estimate: (current_health - threshold) / |trend|
            estimated_rul_hours = abs((current_health - failure_threshold) / trend) * degradation_model.dt_hours
            # Adaptive horizon: max(base, 3 * estimated_RUL) but cap at 720h (30 days)
            adaptive_horizon = min(720.0, max(base_horizon_hours, 3.0 * estimated_rul_hours))
        else:
            # Not degrading - use base horizon
            adaptive_horizon = base_horizon_hours
        
        forecast_horizon_hours = adaptive_horizon
        max_forecast_hours = float(forecast_config.get('max_forecast_hours', forecast_horizon_hours))
        
        # M11: Forecast resolution - use configured value or fall back to data cadence
        # forecast_resolution_hours allows coarser output (e.g., hourly) than data cadence
        resolution_hours = forecast_config.get('forecast_resolution_hours')
        if resolution_hours is None:
            # M3.3: Use dt_hours from DataSummary if available, else from model
            dt_hours = float(forecast_config.get('dt_hours', degradation_model.dt_hours))
        else:
            dt_hours = float(resolution_hours)
        
        # Generate degradation forecast
        max_steps = int(max_forecast_hours / dt_hours)
        degradation_forecast = degradation_model.predict(
            steps=max_steps,
            dt_hours=dt_hours,
            confidence_level=confidence_level
        )
        
        # Compute failure probabilities
        failure_stats = compute_failure_statistics(
            health_forecast=degradation_forecast.point_forecast,
            failure_threshold=failure_threshold,
            health_std=degradation_forecast.std_error,
            dt_hours=dt_hours
        )
        
        # Estimate RUL via Monte Carlo
        current_health = float(health_df['HealthIndex'].iloc[-1])
        
        with Span("forecast.estimate_rul", current_health=int(current_health), n_simulations=n_simulations):
            rul_estimator = RULEstimator(
                degradation_model=degradation_model,
                failure_threshold=failure_threshold,
                n_simulations=n_simulations,
                confidence_level=confidence_level
            )
            
            rul_estimate = rul_estimator.estimate_rul(
                current_health=current_health,
                dt_hours=dt_hours,
                max_horizon_hours=max_forecast_hours
            )
        
        # Note: Sensor attributions are loaded separately in run_forecast() via _load_sensor_attributions()
        # to avoid duplicate SQL queries. The 'sensor_attributions' key is populated there.
        
        return {
            'current_health': current_health,
            'forecast_timestamps': degradation_forecast.timestamps,
            'forecast_values': degradation_forecast.point_forecast,
            'forecast_lower': degradation_forecast.lower_bound,
            'forecast_upper': degradation_forecast.upper_bound,
            'failure_probs': failure_stats['failure_probs'],
            'survival_probs': failure_stats['survival_probs'],
            'hazard_rates': failure_stats['hazard_rates'],
            'mttf_hours': failure_stats['mttf_hours'],
            'rul_p10': rul_estimate.p10_lower_bound,
            'rul_p50': rul_estimate.p50_median,
            'rul_p90': rul_estimate.p90_upper_bound,
            'rul_mean': rul_estimate.mean_rul,
            'rul_std': rul_estimate.std_rul,
            'failure_prob_horizon': rul_estimate.failure_probability
        }
    
    def _get_model_maturity_state(self) -> Tuple[str, int, float]:
        """
        Get model maturity state for RUL reliability gating.
        
        V11 Rule #10: RUL must be gated or suppressed when model not CONVERGED.
        
        V11 CRITICAL-1 FIX: Use ONLY the cached model_state from constructor.
        NEVER fallback to SQL - that creates a race condition where acm_main
        updates state but ForecastEngine reads stale SQL data.
        
        Returns:
            Tuple of (maturity_state, training_rows, training_days)
        """
        # V11 CRITICAL-1: Use ONLY cached state from constructor
        # If model_state is None, we're in COLDSTART (no models exist yet)
        model_state = self._model_state
        
        if model_state is None:
            # First run or no model exists - use conservative COLDSTART defaults
            return 'COLDSTART', 0, 0.0
        
        try:
            return (
                str(model_state.maturity.value),
                model_state.training_rows,
                model_state.training_days,
            )
        except Exception as e:
            Console.warn(f"Could not extract model maturity: {e}",
                         component="FORECAST", equip_id=self.equip_id)
            return 'LEARNING', 0, 0.0  # Conservative default
    
    def _load_sensor_attributions(self) -> list:
        """Load sensor attributions from ACM_SensorHotspots"""
        attributor = SensorAttributor(sql_client=self.sql_client)
        attributions = attributor.load_from_sql(self.equip_id, self.run_id)
        return attributions
    
    def _validate_forecast_timestamps(self, timestamps) -> list:
        """
        Validate and normalize forecast timestamps (M12).
        
        Ensures:
        - Timestamps are naive (no timezone info) - required for SQL Server datetime2
        - Timestamps are strictly increasing
        - Consistent step size (dt_hours) for proper time series alignment
        
        Args:
            timestamps: List or DatetimeIndex of datetime objects from forecast
            
        Returns:
            Validated list of naive datetime objects
        """
        # Handle empty/None cases - avoid pd.DatetimeIndex truth value ambiguity
        if timestamps is None:
            return []
        if hasattr(timestamps, '__len__') and len(timestamps) == 0:
            return []
        
        # Convert DatetimeIndex to list for iteration
        if isinstance(timestamps, pd.DatetimeIndex):
            timestamps = timestamps.tolist()
        
        validated = []
        prev_ts = None
        expected_delta = None
        
        for ts in timestamps:
            # Convert to naive datetime if needed (remove timezone)
            if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            
            # Ensure datetime type
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime().replace(tzinfo=None)
            
            # Check strictly increasing
            if prev_ts is not None:
                delta = ts - prev_ts
                if delta.total_seconds() <= 0:
                    Console.warn(f"Non-increasing timestamp detected: {prev_ts} -> {ts}",
                                 component="FORECAST", equip_id=self.equip_id)
                    continue  # Skip non-increasing timestamps
                
                # Track expected delta for consistency check
                if expected_delta is None:
                    expected_delta = delta
                elif abs((delta - expected_delta).total_seconds()) > 60:  # Allow 1 min tolerance
                    Console.warn(
                        f"Inconsistent timestamp spacing: expected {expected_delta}, got {delta}",
                        component="FORECAST", equip_id=self.equip_id
                    )
            
            validated.append(ts)
            prev_ts = ts
        
        return validated

    def _write_outputs(
        self,
        forecast_results: Dict[str, Any],
        sensor_attributions: list,
        diagnostics: Optional[Dict[str, Any]] = None,
        data_summary: Optional[Any] = None
    ) -> list[str]:
        """Write forecast outputs to SQL tables via OutputManager"""
        tables_written = []
        
        try:
            # M12: Validate timestamps before writing
            raw_timestamps = forecast_results['forecast_timestamps']
            validated_timestamps = self._validate_forecast_timestamps(raw_timestamps)
            
            if len(validated_timestamps) < len(raw_timestamps):
                Console.warn(
                    f"Filtered {len(raw_timestamps) - len(validated_timestamps)} invalid timestamps",
                    component="FORECAST", equip_id=self.equip_id,
                    raw_count=len(raw_timestamps), valid_count=len(validated_timestamps)
                )
            
            # Use validated timestamps for all forecast DataFrames
            forecast_values = forecast_results['forecast_values'][:len(validated_timestamps)]
            forecast_lower = forecast_results['forecast_lower'][:len(validated_timestamps)]
            forecast_upper = forecast_results['forecast_upper'][:len(validated_timestamps)]
            failure_probs = forecast_results['failure_probs'][:len(validated_timestamps)]
            survival_probs = forecast_results['survival_probs'][:len(validated_timestamps)]
            hazard_rates = forecast_results['hazard_rates'][:len(validated_timestamps)]
            
            # Get forecast std from diagnostics or compute from bounds
            forecast_std = float(forecast_results.get('forecast_std', 0.0))
            if forecast_std == 0 and len(forecast_upper) > 0 and len(forecast_lower) > 0:
                # Estimate std from CI bounds (approximately 2 std for 95% CI)
                forecast_std = float((forecast_upper[0] - forecast_lower[0]) / 4.0)
            
            # ACM_HealthForecast: Health forecast time series
            # Column names MUST match SQL table schema exactly
            df_health_forecast = pd.DataFrame({
                'EquipID': self.equip_id,
                'RunID': self.run_id,
                'Timestamp': validated_timestamps,
                'ForecastHealth': forecast_values,  # SQL column name (not HealthForecast)
                'CiLower': forecast_lower,          # SQL column name (not LowerBound)
                'CiUpper': forecast_upper,          # SQL column name (not UpperBound)
                'ForecastStd': forecast_std,
                'Method': 'ExponentialSmoothing',
                'CreatedAt': datetime.now()
            })
            
            self.output_manager.write_dataframe(
                df_health_forecast,
                artifact_name='acm_health_forecast',
                sql_table='ACM_HealthForecast',
                add_created_at=False
            )
            tables_written.append('ACM_HealthForecast')
            
            # ACM_FailureForecast: Failure probability time series
            df_failure_forecast = pd.DataFrame({
                'EquipID': self.equip_id,
                'RunID': self.run_id,
                'Timestamp': validated_timestamps,
                'FailureProb': failure_probs,
                'SurvivalProb': survival_probs,
                'HazardRate': hazard_rates,
                'CreatedAt': datetime.now()
            })
            
            self.output_manager.write_dataframe(
                df_failure_forecast,
                artifact_name='acm_failure_forecast',
                sql_table='ACM_FailureForecast',
                add_created_at=False
            )
            tables_written.append('ACM_FailureForecast')
            
            # ACM_RUL: RUL summary with sensor attributions
            top3_sensors = sensor_attributions[:3] if len(sensor_attributions) >= 3 else sensor_attributions
            
            # Calculate predicted failure time
            current_time = datetime.now()
            rul_hours = forecast_results['rul_p50']
            predicted_failure_time = current_time + timedelta(hours=float(rul_hours))
            
            # V11: Get model maturity state for reliability gating
            maturity_state, training_rows, training_days = self._get_model_maturity_state()
            
            # V11: Compute RUL confidence with reliability gate
            confidence, reliability_status, reliability_reason = compute_rul_confidence(
                p10=forecast_results['rul_p10'],
                p50=forecast_results['rul_p50'],
                p90=forecast_results['rul_p90'],
                maturity_state=maturity_state,
                training_rows=training_rows,
                training_days=training_days,
            )
            
            # Log reliability status
            if reliability_status != ReliabilityStatus.RELIABLE:
                Console.warn(f"RUL reliability: {reliability_status.value} - {reliability_reason}",
                             component="FORECAST", equip_id=self.equip_id,
                             status=reliability_status.value, maturity=maturity_state)
            
            # M13: Derive operator context fields
            current_health = forecast_results['current_health']
            
            # HealthLevel: GOOD/CAUTION/WARNING/CRITICAL based on health thresholds
            if current_health >= 85:
                health_level = 'GOOD'
            elif current_health >= 70:
                health_level = 'CAUTION'
            elif current_health >= 50:
                health_level = 'WARNING'
            else:
                health_level = 'CRITICAL'
            
            # TrendSlope: Derived from forecast values (health units per hour)
            forecast_values_arr = forecast_results.get('forecast_values', [])
            if len(forecast_values_arr) >= 2:
                # Simple linear slope from first to last forecast point
                delta_health = forecast_values_arr[-1] - forecast_values_arr[0]
                n_steps = len(forecast_values_arr)
                # Estimate hours based on validated timestamps
                if len(validated_timestamps) >= 2:
                    delta_time = (validated_timestamps[-1] - validated_timestamps[0]).total_seconds() / 3600
                    trend_slope = delta_health / max(delta_time, 1.0) if delta_time > 0 else 0.0
                else:
                    trend_slope = 0.0
            else:
                trend_slope = 0.0
            
            # DataQuality: From diagnostics or data_summary
            data_quality = 'UNKNOWN'
            if diagnostics and 'data_quality' in diagnostics:
                data_quality = diagnostics['data_quality']
            elif data_summary and hasattr(data_summary, 'quality'):
                data_quality = data_summary.quality.value if data_summary.quality else 'UNKNOWN'
            
            # ForecastStd: From diagnostics
            forecast_std = diagnostics.get('forecast_std', 0.0) if diagnostics else 0.0
            
            df_rul = pd.DataFrame({
                'EquipID': [self.equip_id],
                'RunID': [self.run_id],
                'RUL_Hours': [forecast_results['rul_p50']],  # FIX: Primary RUL estimate (P50)
                'P10_LowerBound': [forecast_results['rul_p10']],
                'P50_Median': [forecast_results['rul_p50']],
                'P90_UpperBound': [forecast_results['rul_p90']],
                'MeanRUL': [forecast_results['rul_mean']],
                'StdRUL': [forecast_results['rul_std']],
                'MTTF_Hours': [forecast_results['mttf_hours']],
                'FailureProbability': [forecast_results['failure_prob_horizon']],
                'CurrentHealth': [current_health],
                'Confidence': [float(confidence)],  # V11: Proper confidence from reliability gate
                'RUL_Status': [reliability_status.value],  # V11: RELIABLE, NOT_RELIABLE, LEARNING, INSUFFICIENT_DATA
                'FailureTime': [predicted_failure_time],  # FIX: Add predicted failure timestamp
                'NumSimulations': [1000],  # FIX: Monte Carlo simulation count
                'Method': ['Multipath'],  # FIX: Add forecasting method
                # M13: Operator context fields
                'HealthLevel': [health_level],
                'TrendSlope': [float(trend_slope)],
                'DataQuality': [data_quality],
                'ForecastStd': [float(forecast_std) if not np.isnan(forecast_std) else 0.0],
                # V11: Model maturity context
                'MaturityState': [maturity_state],
                # Sensor attributions
                'TopSensor1': [top3_sensors[0].sensor_name if len(top3_sensors) > 0 else None],
                'TopSensor1Contribution': [top3_sensors[0].failure_contribution if len(top3_sensors) > 0 else None],
                'TopSensor2': [top3_sensors[1].sensor_name if len(top3_sensors) > 1 else None],
                'TopSensor2Contribution': [top3_sensors[1].failure_contribution if len(top3_sensors) > 1 else None],
                'TopSensor3': [top3_sensors[2].sensor_name if len(top3_sensors) > 2 else None],
                'TopSensor3Contribution': [top3_sensors[2].failure_contribution if len(top3_sensors) > 2 else None],
                'CreatedAt': [datetime.now()]
            })
            
            self.output_manager.write_dataframe(
                df_rul,
                artifact_name='acm_rul_summary',
                sql_table='ACM_RUL',
                add_created_at=False
            )
            tables_written.append('ACM_RUL')
            
            Console.info(f"Wrote {len(tables_written)} forecast tables to SQL",
                         component="FORECAST", equip_id=self.equip_id, run_id=self.run_id,
                         tables=tables_written)
            
            # NEW: ACM_SensorForecast - Physical sensor forecasts
            sensor_forecast_df = self._generate_sensor_forecasts(sensor_attributions, forecast_results)
            has_sensor_data = sensor_forecast_df is not None and not sensor_forecast_df.empty
            
            if has_sensor_data and sensor_forecast_df is not None:
                self.output_manager.write_dataframe(
                    sensor_forecast_df,
                    artifact_name='acm_sensor_forecast',
                    sql_table='ACM_SensorForecast',
                    add_created_at=False
                )
                tables_written.append('ACM_SensorForecast')
                Console.info(f"Wrote sensor forecasts for {len(sensor_attributions)} sensors",
                             component="FORECAST", equip_id=self.equip_id, sensors=len(sensor_attributions))
            
            # v10.1.0: Multivariate forecasting with VAR model
            # Only run if sensor data exists (same table as _generate_sensor_forecasts)
            enable_multivariate = self.config_mgr.get_config(
                self.equip_id, 'forecasting.multivariate.enabled', True
            )
            if enable_multivariate and has_sensor_data and len(sensor_attributions) >= 2:
                try:
                    from core.multivariate_forecast import MultivariateSensorForecaster
                    var_max_lag = int(self.config_mgr.get_config(
                        self.equip_id, 'forecasting.multivariate.var_max_lag', 12
                    ))
                    correlation_window_hours = int(self.config_mgr.get_config(
                        self.equip_id, 'forecasting.multivariate.correlation_window_hours', 168
                    ))
                    
                    # Get sensor names from attributions
                    sensor_names = [s.sensor_name for s in sensor_attributions]
                    
                    mvar_forecaster = MultivariateSensorForecaster(
                        sql_client=self.sql_client,
                        equip_id=self.equip_id,
                        run_id=self.run_id,
                        lookback_hours=float(correlation_window_hours)
                    )
                    mvar_result = mvar_forecaster.forecast(
                        sensor_names=sensor_names,
                        horizon_hours=float(forecast_results.get('forecast_horizon', 168))
                    )
                    if mvar_result is not None and not mvar_result.forecast_df.empty:
                        # Write multivariate forecasts to SQL
                        self.output_manager.write_dataframe(
                            mvar_result.forecast_df,
                            artifact_name='acm_multivariate_forecast',
                            sql_table='ACM_MultivariateForecast',
                            add_created_at=True
                        )
                        tables_written.append('ACM_MultivariateForecast')
                        Console.info(
                            f"Multivariate (VAR) forecast complete: {len(sensor_names)} sensors, method={mvar_result.method}",
                            component="FORECAST", equip_id=self.equip_id, sensors=len(sensor_names),
                            method=mvar_result.method
                        )
                except ImportError as ie:
                    Console.warn(f"Multivariate forecasting module not available: {ie}",
                                 component="FORECAST", equip_id=self.equip_id)
                except Exception as e:
                    Console.warn(f"Multivariate forecasting failed (non-fatal): {e}",
                                 component="FORECAST", equip_id=self.equip_id, error=str(e))
            
        except Exception as e:
            Console.error(f"Failed to write forecast outputs: {e}",
                          component="FORECAST", equip_id=self.equip_id, run_id=self.run_id,
                          error_type=type(e).__name__, error_msg=str(e)[:500])
        
        return tables_written
    
    def _run_regime_conditioned_forecasting(
        self,
        health_df: pd.DataFrame,
        degradation_model: LinearTrendModel,
        forecast_config: Dict[str, Any],
        forecast_results: Dict[str, Any]
    ) -> List[str]:
        """
        Run regime-conditioned forecasting extension (v10.1.0).
        
        This method:
        1. Creates RegimeConditionedForecaster instance
        2. Loads forecast context (OMR, drift, regime state)
        3. Computes per-regime statistics
        4. Estimates RUL per regime with adjusted thresholds
        5. Writes outputs to ACM_RUL_ByRegime, ACM_RegimeHazard, ACM_ForecastContext
        
        Args:
            health_df: Health timeline DataFrame
            degradation_model: Fitted degradation model
            forecast_config: Configuration dictionary
            forecast_results: Results from standard forecasting
            
        Returns:
            List of tables written
        """
        tables_written = []
        
        # FAST EARLY-EXIT: Check if regime data exists before creating forecaster
        # This avoids wasting time in batch mode when no regime data is available
        try:
            check_query = """
                SELECT TOP 1 1 FROM ACM_RegimeTimeline WHERE EquipID = ?
            """
            with self.sql_client.get_cursor() as cur:
                cur.execute(check_query, (self.equip_id,))
                if not cur.fetchone():
                    # No regime data - return immediately without warnings
                    return tables_written
        except Exception:
            pass  # If check fails, proceed with normal flow
        
        try:
            # Create regime-conditioned forecaster
            conditioned = RegimeConditionedForecaster(
                sql_client=self.sql_client,
                output_manager=self.output_manager,
                equip_id=self.equip_id,
                run_id=self.run_id,
                config=forecast_config
            )
            
            # Load context (OMR, drift, regime state)
            context = conditioned.load_forecast_context()
            Console.info(
                f"Regime context: regime={context.current_regime}, "
                f"omr_z={context.current_omr_z}, drift_trend={context.drift_trend}",
                component="FORECAST", equip_id=self.equip_id, regime=context.current_regime,
                omr_z=context.current_omr_z, drift_trend=context.drift_trend
            )
            
            # Compute per-regime stats
            regime_stats = conditioned.compute_regime_stats(lookback_days=90)
            if not regime_stats:
                # Silent return - early check at function start should prevent this path
                return tables_written
            
            # Estimate RUL per regime
            current_health = forecast_results.get('current_health', 85.0)
            rul_results = conditioned.estimate_rul_by_regime(
                current_health=current_health,
                degradation_model=degradation_model,
                current_regime=context.current_regime,
                forecast_config=forecast_config
            )
            
            # Log regime-conditioned RUL
            conditioned_rul = rul_results.get('rul_conditioned')
            if conditioned_rul:
                Console.info(
                    f"Regime-conditioned RUL: P50={conditioned_rul.p50_median:.1f}h (regime={context.current_regime})",
                    component="FORECAST", equip_id=self.equip_id, rul_p50=conditioned_rul.p50_median,
                    regime=context.current_regime
                )
            
            # Write outputs to SQL
            tables_written = conditioned.write_regime_conditioned_outputs(
                rul_results=rul_results,
                forecast_context=context
            )
            
        except Exception as e:
            Console.error(f"Regime-conditioned forecasting failed: {e}",
                          component="FORECAST", equip_id=self.equip_id, run_id=self.run_id,
                          error_type=type(e).__name__, error_msg=str(e)[:500])
        
        return tables_written
    
    def _generate_sensor_forecasts(
        self,
        sensor_attributions: list,
        forecast_results: Dict[str, Any],
        forecast_horizon_hours: float = 168.0
    ) -> Optional[pd.DataFrame]:
        """
        Generate sensor-level time series forecasts.
        
        This implements sensor-level forecasting using exponential smoothing with trend.
        For each high-contribution sensor, we:
        1. Load recent sensor readings from ACM_SensorNormalized_TS
        2. Fit exponential smoothing model (Holt's method)
        3. Generate 7-day forecast with confidence intervals
        4. Detect trend direction (increasing/decreasing/stable)
        
        Args:
            sensor_attributions: List of SensorAttribution objects with top sensors
            forecast_results: Dict containing current forecast metadata
            forecast_horizon_hours: Hours to forecast ahead (default 168 = 7 days)
            
        Returns:
            DataFrame with columns: EquipID, RunID, Timestamp, SensorName, ForecastValue,
                                   CI_Lower, CI_Upper, TrendDirection, Method, CreatedAt
            None if no sensor data available or forecasting fails
        """
        try:
            # FAST EARLY-EXIT: Check if ACM_SensorNormalized_TS has ANY data for this equipment
            # This avoids wasting 10-20 seconds on empty table queries in batch mode
            try:
                check_query = """
                    SELECT TOP 1 1 FROM ACM_SensorNormalized_TS WHERE EquipID = ?
                """
                with self.sql_client.get_cursor() as cur:
                    cur.execute(check_query, (self.equip_id,))
                    if not cur.fetchone():
                        # No data exists - return immediately without warnings
                        return None
            except Exception:
                pass  # If check fails, proceed with normal flow
            
            # Try to import statsmodels, but fall back to simple moving average if unavailable
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                USE_EXPONENTIAL_SMOOTHING = True
            except ImportError:
                Console.warn("statsmodels not available, using simple trend forecasting",
                             component="FORECAST", equip_id=self.equip_id)
                USE_EXPONENTIAL_SMOOTHING = False
            
            from scipy import stats
            
            # Only forecast top N sensors to keep table size manageable
            max_sensors = min(10, len(sensor_attributions))
            top_sensors = sensor_attributions[:max_sensors]
            
            if not top_sensors:
                Console.warn("No sensor attributions available for forecasting",
                             component="FORECAST", equip_id=self.equip_id)
                return None
            
            # Load recent sensor data from ACM_SensorNormalized_TS (contains SensorName, NormalizedValue)
            # Use data-relative lookback to handle batch/historical runs (not just datetime.now())
            lookback_hours = 720  # 30 days of history
            
            # First, get the max timestamp from the sensor data to anchor our lookback
            try:
                with self.sql_client.get_cursor() as cur:
                    cur.execute("""
                        SELECT MAX(Timestamp) FROM ACM_SensorNormalized_TS WHERE EquipID = ?
                    """, (self.equip_id,))
                    max_ts_row = cur.fetchone()
                    if max_ts_row and max_ts_row[0]:
                        data_anchor = pd.to_datetime(max_ts_row[0])
                    else:
                        data_anchor = datetime.now()
            except Exception:
                data_anchor = datetime.now()
            
            cutoff_time = data_anchor - timedelta(hours=lookback_hours)
            
            query = """
            SELECT 
                sn.Timestamp,
                sn.SensorName,
                sn.NormalizedValue
            FROM ACM_SensorNormalized_TS sn
            WHERE sn.EquipID = ?
              AND sn.Timestamp >= ?
              AND sn.SensorName IN ({placeholders})
            ORDER BY sn.Timestamp ASC
            """.format(placeholders=','.join(['?'] * len(top_sensors)))
            
            sensor_names = [s.sensor_name for s in top_sensors]
            
            # Debug: Log query parameters
            Console.debug(f"Sensor forecast query: equip={self.equip_id}, cutoff={cutoff_time}, sensors={sensor_names[:3]}...",
                         component="FORECAST")
            
            cursor = self.sql_client.cursor()
            cursor.execute(query, [self.equip_id, cutoff_time] + sensor_names)
            rows = cursor.fetchall()
            cursor.close()
            
            Console.debug(f"Sensor forecast query returned {len(rows)} rows",
                         component="FORECAST")
            
            if not rows:
                # Debug: Log why we're returning None
                Console.warn(f"No sensor history found for forecasting (cutoff={cutoff_time}, sensors={len(sensor_names)})",
                            component="FORECAST", equip_id=self.equip_id)
                return None
            
            # Convert to DataFrame - use NormalizedValue (stored as 'Score') for forecasting
            sensor_history = pd.DataFrame(
                [{'Timestamp': r[0], 'SensorName': r[1], 'Score': r[2]} for r in rows]
            )
            sensor_history['Timestamp'] = pd.to_datetime(sensor_history['Timestamp'])
            
            # Generate forecasts per sensor
            forecast_records = []
            dt_hours = 1.0  # Hourly intervals for sensor forecasts
            n_steps = int(forecast_horizon_hours / dt_hours)
            
            for sensor_attr in top_sensors:
                sensor_name = sensor_attr.sensor_name
                sensor_data = sensor_history[sensor_history['SensorName'] == sensor_name].copy()
                
                if len(sensor_data) < 24:  # Need at least 24 hours
                    Console.warn(f"Insufficient data for sensor: {sensor_name} ({len(sensor_data)} points)",
                                 component="FORECAST", equip_id=self.equip_id, sensor=sensor_name,
                                 points=len(sensor_data))
                    continue
                
                # Prepare time series
                sensor_data = sensor_data.sort_values('Timestamp')
                sensor_data = sensor_data.set_index('Timestamp')
                series = sensor_data['Score']
                
                # Remove any NaNs
                series = series.dropna()
                
                if len(series) < 24:
                    continue
                
                try:
                    if USE_EXPONENTIAL_SMOOTHING:
                        # V11 FIX: Set explicit frequency to suppress statsmodels warning
                        # "No frequency information was provided, so inferred frequency X will be used"
                        # Infer frequency from data cadence or use 30min as default (ACM standard)
                        if series.index.freq is None:
                            inferred_freq = pd.infer_freq(series.index)
                            if inferred_freq:
                                series = series.asfreq(inferred_freq)
                            else:
                                # Fallback: calculate median time delta and round to nearest standard freq
                                time_diffs = series.index.to_series().diff().dropna()
                                if len(time_diffs) > 0:
                                    median_diff = time_diffs.median()
                                    # Round to nearest standard frequency (30min, 1h, etc.)
                                    if median_diff <= pd.Timedelta(minutes=45):
                                        series = series.asfreq('30min')
                                    elif median_diff <= pd.Timedelta(hours=1.5):
                                        series = series.asfreq('1h')
                                    else:
                                        series = series.asfreq('1h')  # Default fallback
                        
                        # Fit exponential smoothing with trend (Holt's method)
                        model = ExponentialSmoothing(
                            series,
                            trend='add',
                            seasonal=None,
                            initialization_method='estimated'
                        )
                        fitted_model = model.fit(
                            smoothing_level=0.3,
                            smoothing_trend=0.1,
                            optimized=False
                        )
                        
                        # Generate forecast
                        forecast = fitted_model.forecast(steps=n_steps)
                        
                        # Calculate confidence intervals using residual std (robust MAD)
                        # v11.1.2: Use MAD * 1.4826 instead of std for robustness
                        residuals = series - fitted_model.fittedvalues
                        residual_median = float(residuals.median())
                        residual_mad = float((residuals - residual_median).abs().median())
                        residual_std = residual_mad * 1.4826  # Scale MAD to be consistent with std
                    else:
                        # Simple fallback: linear trend with moving average smoothing
                        # Calculate trend using last 168 hours (7 days)
                        recent_window = min(168, len(series))
                        recent_series = series.tail(recent_window)
                        
                        # Fit linear trend
                        x = np.arange(len(recent_series))
                        y = recent_series.values
                        slope, intercept = np.polyfit(x, y, 1)
                        
                        # Calculate residual std for confidence intervals (robust MAD)
                        # v11.1.2: Use MAD * 1.4826 instead of std for robustness
                        trend_line = slope * x + intercept
                        residuals = y - trend_line
                        residual_median = float(np.median(residuals))
                        residual_mad = float(np.median(np.abs(residuals - residual_median)))
                        residual_std = residual_mad * 1.4826  # Scale MAD to be consistent with std
                        
                        # Generate forecast using linear extrapolation
                        last_x = len(recent_series) - 1
                        forecast_x = np.arange(last_x + 1, last_x + 1 + n_steps)
                        forecast_values = slope * forecast_x + intercept
                        
                        # Create pandas Series for consistency with statsmodels output
                        forecast_times = [series.index[-1] + timedelta(hours=dt_hours * (i + 1)) for i in range(n_steps)]
                        forecast = pd.Series(forecast_values, index=forecast_times)
                    
                    # 95% confidence interval
                    z_score = 1.96
                    forecast_lower = forecast - (z_score * residual_std)
                    forecast_upper = forecast + (z_score * residual_std)
                    
                    # Detect trend direction using Mann-Kendall test (v10.1.0)
                    # Mann-Kendall is robust to outliers and doesn't assume linearity
                    # Reference: Mann (1945), Kendall (1975)
                    recent_window = series.tail(168)  # Last week
                    if len(recent_window) >= 24:
                        trend_direction = self._mann_kendall_trend(recent_window.values)
                    else:
                        trend_direction = 'Unknown'
                    
                    # Build forecast records - VECTORIZED for performance
                    last_timestamp = series.index[-1]
                    method_name = 'SimpleLinearTrend' if not USE_EXPONENTIAL_SMOOTHING else 'ExponentialSmoothing'
                    now_ts = datetime.now()
                    
                    # Create timestamps array
                    forecast_timestamps = [last_timestamp + timedelta(hours=dt_hours * (i + 1)) for i in range(n_steps)]
                    
                    # Build DataFrame directly instead of appending dicts
                    sensor_forecast_df = pd.DataFrame({
                        'EquipID': self.equip_id,
                        'RunID': self.run_id,
                        'Timestamp': forecast_timestamps,
                        'SensorName': sensor_name,
                        'ForecastValue': forecast.values[:n_steps],
                        'CiLower': forecast_lower.values[:n_steps],
                        'CiUpper': forecast_upper.values[:n_steps],
                        'ForecastStd': residual_std,
                        'Method': method_name,
                        'RegimeLabel': 0,
                        'CreatedAt': now_ts
                    })
                    forecast_records.extend(sensor_forecast_df.to_dict('records'))
                    
                except Exception as e:
                    Console.warn(f"Failed to forecast sensor {sensor_name}: {e}",
                                 component="FORECAST", equip_id=self.equip_id, sensor=sensor_name,
                                 error=str(e))
                    continue
            
            if not forecast_records:
                Console.warn("No sensor forecasts generated",
                             component="FORECAST", equip_id=self.equip_id)
                return None
            
            df = pd.DataFrame(forecast_records)
            Console.info(
                f"Generated {len(df)} sensor forecast points for {df['SensorName'].nunique()} sensors over {forecast_horizon_hours:.0f}h",
                component="FORECAST", equip_id=self.equip_id, points=len(df),
                sensors=df['SensorName'].nunique(), horizon_hours=forecast_horizon_hours
            )
            return df
            
        except Exception as e:
            Console.error(f"Sensor forecasting failed: {e}",
                          component="FORECAST", equip_id=self.equip_id, run_id=self.run_id,
                          error_type=type(e).__name__, error_msg=str(e)[:500])
            return None

    def _mann_kendall_trend(
        self,
        y: np.ndarray,
        threshold_tau: float = 0.1,
        alpha: float = 0.05
    ) -> str:
        """
        Detect monotonic trend using Mann-Kendall test.
        
        Mann-Kendall is non-parametric and robust to:
        - Non-normal distributions
        - Outliers  
        - Missing values
        - Serial correlation (with variance correction)
        
        The test computes Kendall's tau correlation between data and time.
        
        Reference:
        - Mann (1945): Nonparametric tests against trend
        - Kendall (1975): Rank Correlation Methods
        
        Args:
            y: Time series values (chronological order)
            threshold_tau: Minimum |tau| for practical significance (default 0.1)
            alpha: Significance level (default 0.05)
        
        Returns:
            'Increasing', 'Decreasing', 'Stable', or 'Unknown'
        """
        n = len(y)
        if n < 8:
            return 'Unknown'
        
        try:
            from scipy.stats import kendalltau
            
            x = np.arange(n)
            result = kendalltau(x, y)
            tau = float(result.statistic) if hasattr(result, 'statistic') else float(result[0])
            p_value = float(result.pvalue) if hasattr(result, 'pvalue') else float(result[1])
            
            # Check both statistical AND practical significance
            if p_value < alpha and abs(tau) > threshold_tau:
                return 'Increasing' if tau > 0 else 'Decreasing'
            else:
                return 'Stable'
                
        except Exception:
            return 'Unknown'


# ========================================================================
# v10.1.0: Regime-Conditioned Forecasting Extension
# ========================================================================

class RegimeConditionedForecaster:
    """
    Extension to ForecastEngine providing regime-aware forecasting.
    
    Key Features:
    - Per-regime degradation rates computed from historical data
    - Regime-adjusted failure thresholds (critical regimes get lower thresholds)
    - OMR/drift context integration for confidence adjustment
    - Hazard rate computation per regime
    - Unified forecast context logging
    
    Usage:
        conditioned = RegimeConditionedForecaster(
            sql_client=sql_client,
            output_manager=output_mgr,
            equip_id=1,
            run_id="run_123"
        )
        
        # Load context and compute per-regime stats
        context = conditioned.load_forecast_context()
        regime_stats = conditioned.compute_regime_stats()
        
        # Estimate RUL per regime
        rul_by_regime = conditioned.estimate_rul_by_regime(
            current_health=85.0,
            current_regime=1
        )
    """
    
    def __init__(
        self,
        sql_client: Any,
        output_manager: OutputManager,
        equip_id: int,
        run_id: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.sql_client = sql_client
        self.output_manager = output_manager
        self.equip_id = equip_id
        self.run_id = run_id
        self.config = config or {}
        
        # Cache for regime stats
        self._regime_stats: Optional[Dict[int, RegimeStats]] = None
        self._forecast_context: Optional[ForecastContext] = None
    
    def load_forecast_context(self) -> ForecastContext:
        """
        Load unified forecast context including regime, OMR, and drift state.
        
        Returns:
            ForecastContext with all diagnostic information
        """
        if self._forecast_context is not None:
            return self._forecast_context
        
        # Load OMR/drift context via OutputManager helper
        omr_drift = self.output_manager.load_omr_drift_context(
            self.equip_id, lookback_hours=24
        )
        
        # Load current regime from most recent RegimeTimeline
        current_regime = self._load_current_regime()
        regime_confidence = self._compute_regime_confidence()
        
        # Detect health trend from recent HealthTimeline
        health_trend = self._detect_health_trend()
        
        # Count active defects from SensorDefects
        active_defects = self._count_active_defects()
        
        # Determine if retraining is recommended
        retraining_recommended, retraining_reason = self._check_retraining_needed(omr_drift)
        
        # Estimate data quality from recent metrics
        data_quality = self._estimate_data_quality()
        
        self._forecast_context = ForecastContext(
            current_regime=current_regime,
            regime_confidence=regime_confidence,
            current_omr_z=omr_drift.get('omr_z'),
            omr_trend=omr_drift.get('omr_trend', 'unknown'),
            omr_top_contributors=omr_drift.get('top_contributors', []),
            current_drift_z=omr_drift.get('drift_z'),
            drift_trend=omr_drift.get('drift_trend', 'unknown'),
            health_trend=health_trend,
            data_quality=data_quality,
            active_defects=active_defects,
            retraining_recommended=retraining_recommended,
            retraining_reason=retraining_reason
        )
        
        return self._forecast_context
    
    def compute_regime_stats(self, lookback_days: int = 90) -> Dict[int, RegimeStats]:
        """
        Compute per-regime statistics for conditioned forecasting.
        
        Analyzes historical data to compute:
        - Degradation rate per regime (health units per hour)
        - Health mean/std per regime
        - Dwell time fraction per regime
        - Regime-adjusted failure thresholds
        
        Args:
            lookback_days: Days of history to analyze
            
        Returns:
            Dict mapping regime_label to RegimeStats
        """
        if self._regime_stats is not None:
            return self._regime_stats
        
        try:
            # Query historical health + regime data
            query = """
                SELECT 
                    rt.RegimeLabel,
                    ht.Timestamp,
                    ht.HealthIndex,
                    ht.FusedZ
                FROM ACM_HealthTimeline ht
                JOIN ACM_RegimeTimeline rt 
                    ON ht.EquipID = rt.EquipID AND ht.Timestamp = rt.Timestamp
                WHERE ht.EquipID = ?
                  AND ht.Timestamp >= DATEADD(day, -?, GETDATE())
                ORDER BY ht.Timestamp ASC
            """
            
            with self.sql_client.get_cursor() as cur:
                cur.execute(query, (self.equip_id, lookback_days))
                rows = cur.fetchall()
            
            if not rows:
                # Silent return - early check in _run_regime_conditioned_forecasting should catch this
                self._regime_stats = {}
                return self._regime_stats
            
            # Build DataFrame for analysis
            df = pd.DataFrame(rows, columns=['RegimeLabel', 'Timestamp', 'HealthIndex', 'FusedZ'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values('Timestamp')
            
            # Compute per-regime stats
            self._regime_stats = {}
            base_failure_threshold = float(self.config.get('failure_threshold', 70.0))
            
            for regime_label in df['RegimeLabel'].unique():
                regime_df = df[df['RegimeLabel'] == regime_label].copy()
                
                if len(regime_df) < 10:
                    continue
                
                # Compute degradation rate using Theil-Sen robust regression with bootstrap CI
                # This replaces the naive median-of-rates which amplifies noise (v10.1.0)
                regime_df = regime_df.sort_values('Timestamp')
                health_values = regime_df['HealthIndex'].values
                
                # Compute dt_hours from median time difference
                time_diffs = regime_df['Timestamp'].diff().dt.total_seconds() / 3600
                dt_hours = float(time_diffs.median()) if len(time_diffs) > 1 else 1.0
                if not np.isfinite(dt_hours) or dt_hours <= 0:
                    dt_hours = 1.0
                
                if len(health_values) >= 10:
                    from core.failure_probability import bootstrap_degradation_rate
                    rate_result = bootstrap_degradation_rate(
                        health_values, dt_hours=dt_hours, n_bootstrap=200, confidence_level=0.95
                    )
                    degradation_rate = rate_result['rate']
                    degradation_rate_lower = rate_result['rate_lower']
                    degradation_rate_upper = rate_result['rate_upper']
                    degradation_r_squared = rate_result['r_squared']
                else:
                    # Fallback for small samples
                    degradation_rate = 0.0
                    degradation_rate_lower = 0.0
                    degradation_rate_upper = 0.0
                    degradation_r_squared = 0.0
                
                # Health statistics using ROBUST estimators
                # v11.1.2: Use median and MAD instead of mean/std
                health_median = float(regime_df['HealthIndex'].median())
                health_mad = float((regime_df['HealthIndex'] - health_median).abs().median())
                health_mean = health_median  # For backward compatibility, use median as "mean"
                health_std = health_mad * 1.4826  # Scale MAD to be consistent with std
                
                # Dwell fraction
                total_points = len(df)
                dwell_fraction = float(len(regime_df) / total_points) if total_points > 0 else 0.0
                
                # Transition count (entries to this regime)
                regime_changes = df['RegimeLabel'].diff().fillna(0)
                transition_count = int((regime_changes != 0).sum())
                
                # Health state classification
                fused_median = float(regime_df['FusedZ'].median()) if 'FusedZ' in regime_df.columns else 0.0
                if fused_median >= 3.0:
                    health_state = 'critical'
                elif fused_median >= 1.5:
                    health_state = 'suspect'
                else:
                    health_state = 'healthy'
                
                # Regime-adjusted failure threshold using Weibull survival model (v10.1.0)
                # Instead of ad-hoc 5/10 point adjustments, compute threshold that gives
                # equivalent risk across regimes based on degradation rate and uncertainty
                # Reference: ISO 13381-1:2015 - Prognostics risk-based thresholds
                from core.failure_probability import WeibullHazardModel
                
                # Base risk: probability of failure within 24h at base threshold
                weibull = WeibullHazardModel(shape=2.0, scale=168.0)
                base_risk_24h = 0.05  # Target 5% failure probability in 24h as "critical"
                
                # Adjust threshold based on degradation rate and confidence
                # Faster degradation = lower threshold (earlier warning)
                # Higher uncertainty (lower R^2) = lower threshold (more conservative)
                rate_factor = min(abs(degradation_rate) * 2.0, 10.0)  # Cap at 10 points
                uncertainty_factor = (1.0 - degradation_r_squared) * 5.0  # Up to 5 points for uncertain estimates
                
                # Health state still provides categorical adjustment
                state_adjustment = {
                    'healthy': 0.0,
                    'suspect': 2.5,
                    'critical': 5.0
                }
                
                total_adjustment = state_adjustment.get(health_state, 0.0) + rate_factor * 0.5 + uncertainty_factor * 0.5
                failure_threshold = max(base_failure_threshold - total_adjustment, 50.0)  # Floor at 50
                
                self._regime_stats[int(regime_label)] = RegimeStats(
                    regime_label=int(regime_label),
                    health_state=health_state,
                    degradation_rate=degradation_rate,
                    degradation_rate_lower=degradation_rate_lower,
                    degradation_rate_upper=degradation_rate_upper,
                    degradation_r_squared=degradation_r_squared,
                    health_mean=health_mean,
                    health_std=health_std,
                    dwell_fraction=dwell_fraction,
                    transition_count=transition_count,
                    failure_threshold=failure_threshold,
                    sample_count=len(regime_df)
                )
            
            Console.info(f"Computed stats for {len(self._regime_stats)} regimes",
                         component="FORECAST", equip_id=self.equip_id, regimes=len(self._regime_stats))
            return self._regime_stats
            
        except Exception as e:
            Console.error(f"Failed to compute regime stats: {e}",
                          component="FORECAST", equip_id=self.equip_id, run_id=self.run_id,
                          error_type=type(e).__name__, error_msg=str(e)[:500])
            self._regime_stats = {}
            return self._regime_stats
    
    def estimate_rul_by_regime(
        self,
        current_health: float,
        degradation_model: LinearTrendModel,
        current_regime: Optional[int] = None,
        forecast_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Estimate RUL with regime-conditioned adjustments.
        
        Uses per-regime degradation rates and failure thresholds to provide
        more accurate RUL estimates based on current operating mode.
        
        Args:
            current_health: Current health index
            degradation_model: Fitted degradation model
            current_regime: Current regime label (loads from SQL if None)
            forecast_config: Configuration overrides
            
        Returns:
            Dict with:
            - 'rul_global': RULEstimate (standard, non-regime)
            - 'rul_by_regime': Dict[int, Dict] per-regime RUL estimates
            - 'rul_conditioned': RULEstimate using current regime
            - 'regime_hazards': DataFrame for ACM_RegimeHazard
        """
        config = forecast_config or self.config
        regime_stats = self.compute_regime_stats()
        
        if current_regime is None:
            current_regime = self._load_current_regime()
        
        # Global (unconditional) RUL estimate
        base_threshold = float(config.get('failure_threshold', 70.0))
        n_simulations = int(config.get('monte_carlo_simulations', 1000))
        confidence_level = float(config.get('confidence_min', 0.80))
        max_horizon = float(config.get('max_forecast_hours', 720.0))
        dt_hours = float(config.get('dt_hours', 1.0))
        
        global_estimator = RULEstimator(
            degradation_model=degradation_model,
            failure_threshold=base_threshold,
            n_simulations=n_simulations,
            confidence_level=confidence_level
        )
        
        rul_global = global_estimator.estimate_rul(
            current_health=current_health,
            dt_hours=dt_hours,
            max_horizon_hours=max_horizon
        )
        
        # Per-regime RUL estimates
        rul_by_regime = {}
        for regime_label, stats in regime_stats.items():
            # Create regime-adjusted estimator with empirical noise (v10.1.0)
            # Use R-squared from bootstrap fit - low R^2 = high noise
            # This replaces arbitrary noise_factor=1+rate*10
            noise_from_fit = 1.0 + (1.0 - stats.degradation_r_squared) * 3.0  # Scale by fit quality
            regime_estimator = RULEstimator(
                degradation_model=degradation_model,
                failure_threshold=stats.failure_threshold,
                n_simulations=n_simulations // 2,  # Fewer sims per regime for speed
                confidence_level=confidence_level,
                noise_factor=noise_from_fit  # Empirical noise from fit quality
            )
            
            regime_rul = regime_estimator.estimate_rul(
                current_health=current_health,
                dt_hours=dt_hours,
                max_horizon_hours=max_horizon
            )
            
            rul_by_regime[regime_label] = {
                'RUL_Hours': regime_rul.p50_median,
                'P10_LowerBound': regime_rul.p10_lower_bound,
                'P50_Median': regime_rul.p50_median,
                'P90_UpperBound': regime_rul.p90_upper_bound,
                'DegradationRate': stats.degradation_rate,
                'Confidence': regime_rul.confidence_level * stats.dwell_fraction,
                'Method': 'RegimeConditioned',
                'SampleCount': stats.sample_count,
                'FailureThreshold': stats.failure_threshold
            }
        
        # Conditioned RUL using current regime
        rul_conditioned = rul_global  # Default to global
        if current_regime is not None and current_regime in regime_stats:
            stats = regime_stats[current_regime]
            # Use empirical noise from fit quality (v10.1.0)
            noise_from_fit = 1.0 + (1.0 - stats.degradation_r_squared) * 2.0
            conditioned_estimator = RULEstimator(
                degradation_model=degradation_model,
                failure_threshold=stats.failure_threshold,
                n_simulations=n_simulations,
                confidence_level=confidence_level,
                noise_factor=noise_from_fit  # Empirical noise
            )
            rul_conditioned = conditioned_estimator.estimate_rul(
                current_health=current_health,
                dt_hours=dt_hours,
                max_horizon_hours=max_horizon
            )
        
        # Compute regime hazards for time series output
        regime_hazards = self._compute_regime_hazards(
            current_health=current_health,
            degradation_model=degradation_model,
            regime_stats=regime_stats,
            max_horizon=max_horizon,
            dt_hours=dt_hours
        )
        
        return {
            'rul_global': rul_global,
            'rul_by_regime': rul_by_regime,
            'rul_conditioned': rul_conditioned,
            'regime_hazards': regime_hazards,
            'current_regime': current_regime
        }
    
    def write_regime_conditioned_outputs(
        self,
        rul_results: Dict[str, Any],
        forecast_context: Optional[ForecastContext] = None
    ) -> List[str]:
        """
        Write regime-conditioned forecast outputs to SQL.
        
        Args:
            rul_results: Output from estimate_rul_by_regime()
            forecast_context: ForecastContext (loads if None)
            
        Returns:
            List of tables written
        """
        tables_written = []
        
        try:
            # ACM_RUL_ByRegime
            rul_by_regime = rul_results.get('rul_by_regime', {})
            if rul_by_regime:
                df_rul = pd.DataFrame([
                    {'RegimeLabel': regime, **stats}
                    for regime, stats in rul_by_regime.items()
                ])
                self.output_manager.write_rul_by_regime(
                    self.equip_id, self.run_id, df_rul
                )
                tables_written.append('ACM_RUL_ByRegime')
            
            # ACM_RegimeHazard
            regime_hazards = rul_results.get('regime_hazards')
            if regime_hazards is not None and not regime_hazards.empty:
                self.output_manager.write_regime_hazard(
                    self.equip_id, self.run_id, regime_hazards
                )
                tables_written.append('ACM_RegimeHazard')
            
            # ACM_ForecastContext
            if forecast_context is None:
                forecast_context = self.load_forecast_context()
            
            context_df = pd.DataFrame([{
                'Timestamp': datetime.now(),
                'ForecastHorizon_Hours': float(self.config.get('max_forecast_hours', 720.0)),
                'CurrentHealth': rul_results['rul_conditioned'].p50_median if hasattr(rul_results.get('rul_conditioned'), 'p50_median') else 0.0,
                'CurrentRegime': forecast_context.current_regime,
                'RegimeConfidence': forecast_context.regime_confidence,
                'CurrentOMR_Z': forecast_context.current_omr_z,
                'OMR_Contribution': forecast_context.omr_top_contributors[0]['contribution'] if forecast_context.omr_top_contributors else None,
                'CurrentDrift_Z': forecast_context.current_drift_z,
                'DriftTrend': forecast_context.drift_trend,
                'FusedZ': None,  # Would need to load from health timeline
                'HealthTrend': forecast_context.health_trend,
                'DataQuality': forecast_context.data_quality,
                'ModelConfidence': forecast_context.regime_confidence,
                'ActiveDefects': forecast_context.active_defects,
                'TopContributor': forecast_context.omr_top_contributors[0]['sensor'] if forecast_context.omr_top_contributors else None,
                'Notes': forecast_context.retraining_reason
            }])
            
            self.output_manager.write_forecast_context(
                self.equip_id, self.run_id, context_df
            )
            tables_written.append('ACM_ForecastContext')
            
            Console.info(f"Wrote {len(tables_written)} regime-conditioned tables",
                         component="FORECAST", equip_id=self.equip_id, tables=tables_written)
        
        except Exception as e:
            Console.error(f"Failed to write regime outputs: {e}",
                          component="FORECAST", equip_id=self.equip_id, run_id=self.run_id,
                          error_type=type(e).__name__, error_msg=str(e)[:500])
        
        return tables_written
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _load_current_regime(self) -> Optional[int]:
        """Load most recent regime label from ACM_RegimeTimeline."""
        if self.sql_client is None:
            return None
        
        try:
            query = """
                SELECT TOP 1 RegimeLabel 
                FROM ACM_RegimeTimeline 
                WHERE EquipID = ? 
                ORDER BY Timestamp DESC
            """
            with self.sql_client.get_cursor() as cur:
                cur.execute(query, (self.equip_id,))
                row = cur.fetchone()
                return int(row[0]) if row else None
        except Exception:
            return None
    
    def _compute_regime_confidence(self) -> float:
        """
        Compute confidence in current regime assignment using classification entropy.
        
        This replaces the simple "fraction in dominant regime" with proper
        classification confidence based on regime label distribution entropy.
        
        Lower entropy = higher confidence (more certainty about regime).
        
        Formula: Confidence = 1 - (H / H_max)
        Where H = -sum(p_i * log(p_i)) and H_max = log(n_regimes)
        
        Reference: Shannon (1948) entropy for classification uncertainty
        """
        if self.sql_client is None:
            return 0.5
        
        try:
            query = """
                SELECT RegimeLabel, COUNT(*) as Cnt
                FROM ACM_RegimeTimeline
                WHERE EquipID = ?
                  AND Timestamp >= DATEADD(hour, -24, GETDATE())
                GROUP BY RegimeLabel
            """
            with self.sql_client.get_cursor() as cur:
                cur.execute(query, (self.equip_id,))
                rows = cur.fetchall()
            
            if not rows:
                return 0.5
            
            total = sum(r[1] for r in rows)
            n_regimes = len(rows)
            
            if total == 0 or n_regimes <= 1:
                return 1.0 if n_regimes == 1 else 0.5
            
            # Compute normalized entropy
            probs = np.array([r[1] / total for r in rows])
            probs = probs[probs > 0]  # Remove zeros to avoid log(0)
            
            entropy = -np.sum(probs * np.log(probs))
            max_entropy = np.log(n_regimes)  # Uniform distribution
            
            # Confidence = 1 - normalized_entropy
            if max_entropy > 0:
                confidence = 1.0 - (entropy / max_entropy)
            else:
                confidence = 1.0
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception:
            return 0.5
    
    def _detect_health_trend(self) -> str:
        """Detect health trend from recent HealthTimeline."""
        if self.sql_client is None:
            return 'unknown'
        
        try:
            query = """
                SELECT HealthIndex, Timestamp
                FROM ACM_HealthTimeline
                WHERE EquipID = ?
                  AND Timestamp >= DATEADD(hour, -168, GETDATE())
                ORDER BY Timestamp ASC
            """
            with self.sql_client.get_cursor() as cur:
                cur.execute(query, (self.equip_id,))
                rows = cur.fetchall()
            
            if len(rows) < 10:
                return 'unknown'
            
            # Use Mann-Kendall test for trend detection (v10.1.0)
            y = np.array([r[0] for r in rows])
            return self._mann_kendall_trend(y, threshold_tau=0.1)
                
        except Exception:
            return 'unknown'
    
    def _mann_kendall_trend(
        self,
        y: np.ndarray,
        threshold_tau: float = 0.1,
        alpha: float = 0.05
    ) -> str:
        """Detect monotonic trend using Mann-Kendall test (see ForecastEngine for full docs)."""
        n = len(y)
        if n < 8:
            return 'Unknown'
        try:
            from scipy.stats import kendalltau
            x = np.arange(n)
            result = kendalltau(x, y)
            tau = float(result.statistic) if hasattr(result, 'statistic') else float(result[0])
            p_value = float(result.pvalue) if hasattr(result, 'pvalue') else float(result[1])
            if p_value < alpha and abs(tau) > threshold_tau:
                return 'improving' if tau > 0 else 'degrading'
            return 'Stable'
        except Exception:
            return 'Unknown'
    
    def _count_active_defects(self) -> int:
        """Count active sensor defects."""
        if self.sql_client is None:
            return 0
        
        try:
            query = """
                SELECT COUNT(*) 
                FROM ACM_SensorDefects 
                WHERE EquipID = ? AND ActiveDefect = 1
            """
            with self.sql_client.get_cursor() as cur:
                cur.execute(query, (self.equip_id,))
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return 0
    
    def _check_retraining_needed(self, omr_drift: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if model retraining is recommended based on drift indicators."""
        reasons = []
        
        # High drift suggests distribution shift
        drift_z = omr_drift.get('drift_z')
        if drift_z is not None and drift_z > 3.0:
            reasons.append(f"High drift detected (Z={drift_z:.2f})")
        
        # Increasing drift trend
        if omr_drift.get('drift_trend') == 'increasing':
            reasons.append("Drift trend increasing")
        
        # High OMR with increasing trend
        omr_z = omr_drift.get('omr_z')
        if omr_z is not None and omr_z > 4.0 and omr_drift.get('omr_trend') == 'increasing':
            reasons.append(f"OMR elevated and increasing (Z={omr_z:.2f})")
        
        if reasons:
            return True, '; '.join(reasons)
        return False, None
    
    def _estimate_data_quality(self) -> float:
        """Estimate data quality score 0-1."""
        # Based on recent data gaps and completeness
        if self.sql_client is None:
            return 0.5
        
        try:
            query = """
                SELECT COUNT(*) as Cnt,
                       COUNT(CASE WHEN HealthIndex IS NULL THEN 1 END) as NullCnt
                FROM ACM_HealthTimeline
                WHERE EquipID = ?
                  AND Timestamp >= DATEADD(day, -7, GETDATE())
            """
            with self.sql_client.get_cursor() as cur:
                cur.execute(query, (self.equip_id,))
                row = cur.fetchone()
            
            if row and row[0] > 0:
                null_ratio = row[1] / row[0]
                return max(0.0, 1.0 - null_ratio * 2)  # Penalize nulls
            return 0.5
            
        except Exception:
            return 0.5
    
    def _compute_regime_hazards(
        self,
        current_health: float,
        degradation_model: LinearTrendModel,
        regime_stats: Dict[int, RegimeStats],
        max_horizon: float,
        dt_hours: float
    ) -> pd.DataFrame:
        """
        Compute hazard rates per regime over forecast horizon.
        
        Returns DataFrame for ACM_RegimeHazard with columns:
        - RegimeLabel, Timestamp, HazardRate, SurvivalProb, CumulativeHazard,
          FailureProb, HealthAtTime
        """
        records = []
        base_time = datetime.now()
        
        # Generate baseline forecast
        max_steps = int(max_horizon / dt_hours)
        forecast = degradation_model.predict(steps=max_steps, dt_hours=dt_hours)
        
        # Import Weibull hazard model for proper survival analysis (v10.1.0)
        from core.failure_probability import WeibullHazardModel
        
        for regime_label, stats in regime_stats.items():
            # Initialize Weibull model for this regime
            # Fit from historical health data if available, else use defaults
            weibull = WeibullHazardModel(
                shape=2.0,  # Typical wear-out shape
                scale=max(168.0, abs(1.0 / max(stats.degradation_rate, 1e-6)))  # Estimate scale from rate
            )
            
            # If we have enough forecast data, fit Weibull parameters
            if len(forecast.point_forecast) >= 10:
                weibull.fit_from_degradation(
                    forecast.point_forecast[:min(100, len(forecast.point_forecast))],
                    failure_threshold=stats.failure_threshold,
                    dt_hours=dt_hours
                )
            
            for i in range(0, max_steps, max(1, max_steps // 50)):  # Sample 50 points
                forecast_time = base_time + timedelta(hours=i * dt_hours)
                health_at_time = forecast.point_forecast[i] if i < len(forecast.point_forecast) else stats.health_mean
                
                # Use proper Weibull hazard model (v10.1.0)
                # Time is hours from now
                t = max(1.0, i * dt_hours)
                hazard_rate = float(weibull.hazard_rate(np.array([t]))[0])
                
                # Get Weibull survival/failure probabilities directly
                survival_prob = float(weibull.survival_probability(np.array([t]))[0])
                failure_prob = float(weibull.failure_probability(np.array([t]))[0])
                
                # Cumulative hazard: -ln(S(t))
                cumulative_hazard = -np.log(max(survival_prob, 1e-10))
                
                records.append({
                    'RegimeLabel': regime_label,
                    'Timestamp': forecast_time,
                    'HazardRate': float(hazard_rate),
                    'SurvivalProb': float(survival_prob),
                    'CumulativeHazard': float(cumulative_hazard),
                    'FailureProb': float(failure_prob),
                    'HealthAtTime': float(health_at_time)
                })
        
        return pd.DataFrame(records) if records else pd.DataFrame()
