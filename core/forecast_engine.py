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

References:
- All module-specific references in respective files
- ISO 13381-1:2015: Prognostics workflow standards
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from core.health_tracker import HealthTimeline, HealthQuality
from core.degradation_model import LinearTrendModel
from core.failure_probability import compute_failure_statistics, health_to_failure_probability
from core.rul_estimator import RULEstimator
from core.sensor_attribution import SensorAttributor
from core.metrics import compute_comprehensive_metrics, log_metrics_summary
from core.state_manager import StateManager, AdaptiveConfigManager, ForecastingState
from core.output_manager import OutputManager
from utils.logger import Console


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
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize forecast engine.
        
        Args:
            sql_client: Database connection (pyodbc)
            output_manager: OutputManager instance for SQL writes
            equip_id: Equipment ID
            run_id: ACM run identifier
            config: Configuration dictionary (optional, uses ACM_AdaptiveConfig if None)
        """
        self.sql_client = sql_client
        self.output_manager = output_manager
        self.equip_id = equip_id
        self.run_id = run_id
        self.config = config or {}
        
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
        try:
            # Step 1: Load health timeline
            health_df, data_quality = self._load_health_timeline()
            
            if health_df is None or data_quality != HealthQuality.OK:
                Console.warn(f"[ForecastEngine] Poor data quality: {data_quality.value}; skipping forecast")
                return {
                    'success': False,
                    'error': f'Poor data quality: {data_quality.value}',
                    'data_quality': data_quality.value
                }
            
            # Step 2: Load persistent state
            state = self.state_mgr.load_state(self.equip_id)
            if state is None:
                state = ForecastingState(equip_id=self.equip_id, state_version=0)
            
            # Step 3: Load adaptive configuration
            forecast_config = self._load_forecast_config()
            
            # Step 4: Check auto-tuning trigger
            current_volume = len(health_df)
            state.data_volume += current_volume
            should_tune = self.config_mgr.should_tune(self.equip_id, state.data_volume)
            
            if should_tune:
                Console.info(f"[ForecastEngine] Auto-tuning triggered at DataVolume={state.data_volume}")
                # TODO: Implement grid search tuning in future PR
                # For now, use configured values
            
            # Step 5: Fit degradation model
            degradation_model = self._fit_degradation_model(health_df, forecast_config, state)
            
            # Step 6: Generate forecast and estimate RUL
            forecast_results = self._generate_forecast_and_rul(
                health_df, degradation_model, forecast_config
            )
            
            # Step 7: Rank sensor attributions
            sensor_attributions = self._load_sensor_attributions()
            
            # Step 8: Compute metrics (if we have historical forecasts to compare)
            # metrics = self._compute_metrics(forecast_results)
            
            # Step 9: Write outputs to SQL
            tables_written = self._write_outputs(forecast_results, sensor_attributions)
            
            # Step 10: Update state
            state.model_params = degradation_model.get_parameters()
            state.last_health_value = float(health_df['HealthIndex'].iloc[-1])
            state.last_health_timestamp = health_df['Timestamp'].iloc[-1]
            state.updated_at = datetime.now()
            
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
            Console.error(f"[ForecastEngine] Forecast failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'data_quality': 'UNKNOWN'
            }
    
    def _load_health_timeline(self) -> tuple[Optional[pd.DataFrame], HealthQuality]:
        """Load health timeline with quality assessment"""
        tracker = HealthTimeline(
            sql_client=self.sql_client,
            equip_id=self.equip_id,
            run_id=self.run_id,
            output_manager=self.output_manager,
            min_train_samples=int(self.config_mgr.get_config(self.equip_id, 'min_train_samples', 200)),
            max_gap_hours=float(self.config_mgr.get_config(self.equip_id, 'max_gap_hours', 6.0))
        )
        
        health_df, quality = tracker.load_from_sql()
        
        if quality != HealthQuality.OK:
            stats = tracker.get_statistics(health_df) if health_df is not None else None
            if stats:
                Console.warn(
                    f"[ForecastEngine] Data quality issue: {stats.quality_reason}"
                )
        
        return health_df, quality
    
    def _load_forecast_config(self) -> Dict[str, Any]:
        """Load forecast configuration with equipment overrides"""
        config = self.config_mgr.get_all_configs(self.equip_id)
        
        # Apply any overrides from self.config
        forecast_section = self.config.get('forecasting', {})
        config.update(forecast_section)
        
        Console.info(
            f"[ForecastEngine] Loaded config: alpha={config.get('alpha', 0.3):.2f}, "
            f"beta={config.get('beta', 0.1):.2f}, "
            f"failure_threshold={config.get('failure_threshold', 70.0):.1f}"
        )
        
        return config
    
    def _fit_degradation_model(
        self,
        health_df: pd.DataFrame,
        forecast_config: Dict[str, Any],
        state: ForecastingState
    ) -> LinearTrendModel:
        """Fit degradation model with warm-start from previous state"""
        # Create model with adaptive config
        model = LinearTrendModel(
            alpha=float(forecast_config.get('alpha', 0.3)),
            beta=float(forecast_config.get('beta', 0.1)),
            max_trend_per_hour=float(forecast_config.get('max_trend_per_hour', 5.0)),
            enable_adaptive=bool(forecast_config.get('enable_adaptive_smoothing', True))
        )
        
        # Warm-start from previous state if available
        if state.model_params:
            try:
                model.set_parameters(state.model_params)
                Console.info("[ForecastEngine] Warm-started model from previous state")
            except Exception as e:
                Console.warn(f"[ForecastEngine] Failed to warm-start model: {e}")
        
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
        max_forecast_hours = float(forecast_config.get('max_forecast_hours', 168.0))
        confidence_level = float(forecast_config.get('confidence_min', 0.80))
        n_simulations = int(forecast_config.get('monte_carlo_simulations', 1000))
        dt_hours = degradation_model.dt_hours
        
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
        
        # Load sensor attributions
        attributor = SensorAttributor(sql_client=self.sql_client)
        sensor_attrs = attributor.load_from_sql(self.equip_id, self.run_id)
        top_sensors_str = attributor.format_top_n(sensor_attrs, n=3)
        
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
            'failure_prob_horizon': rul_estimate.failure_probability,
            'sensor_attributions': sensor_attrs,
            'top_sensors': top_sensors_str
        }
    
    def _load_sensor_attributions(self) -> list:
        """Load sensor attributions from ACM_SensorHotspots"""
        attributor = SensorAttributor(sql_client=self.sql_client)
        attributions = attributor.load_from_sql(self.equip_id, self.run_id)
        return attributions
    
    def _write_outputs(
        self,
        forecast_results: Dict[str, Any],
        sensor_attributions: list
    ) -> list[str]:
        """Write forecast outputs to SQL tables via OutputManager"""
        tables_written = []
        
        try:
            # ACM_HealthForecast: Health forecast time series
            df_health_forecast = pd.DataFrame({
                'EquipID': self.equip_id,
                'RunID': self.run_id,
                'Timestamp': forecast_results['forecast_timestamps'],
                'HealthForecast': forecast_results['forecast_values'],
                'LowerBound': forecast_results['forecast_lower'],
                'UpperBound': forecast_results['forecast_upper'],
                'CreatedAt': datetime.now()
            })
            
            # Create temp CSV path for OutputManager signature
            from pathlib import Path
            temp_dir = Path("artifacts") / f"equip_{self.equip_id}" / "forecast"
            temp_dir.mkdir(parents=True, exist_ok=True)
            health_path = temp_dir / f"health_forecast_{self.run_id}.csv"
            
            self.output_manager.write_dataframe(
                df_health_forecast,
                file_path=health_path,
                sql_table='ACM_HealthForecast',
                add_created_at=False
            )
            tables_written.append('ACM_HealthForecast')
            
            # ACM_FailureForecast: Failure probability time series
            df_failure_forecast = pd.DataFrame({
                'EquipID': self.equip_id,
                'RunID': self.run_id,
                'Timestamp': forecast_results['forecast_timestamps'],
                'FailureProb': forecast_results['failure_probs'],
                'SurvivalProb': forecast_results['survival_probs'],
                'HazardRate': forecast_results['hazard_rates'],
                'CreatedAt': datetime.now()
            })
            
            failure_path = temp_dir / f"failure_forecast_{self.run_id}.csv"
            self.output_manager.write_dataframe(
                df_failure_forecast,
                file_path=failure_path,
                sql_table='ACM_FailureForecast',
                add_created_at=False
            )
            tables_written.append('ACM_FailureForecast')
            
            # ACM_RUL: RUL summary with sensor attributions
            top3_sensors = sensor_attributions[:3] if len(sensor_attributions) >= 3 else sensor_attributions
            
            df_rul = pd.DataFrame({
                'EquipID': [self.equip_id],
                'RunID': [self.run_id],
                'P10_LowerBound': [forecast_results['rul_p10']],
                'P50_Median': [forecast_results['rul_p50']],
                'P90_UpperBound': [forecast_results['rul_p90']],
                'MeanRUL': [forecast_results['rul_mean']],
                'StdRUL': [forecast_results['rul_std']],
                'MTTF_Hours': [forecast_results['mttf_hours']],
                'FailureProbability': [forecast_results['failure_prob_horizon']],
                'CurrentHealth': [forecast_results['current_health']],
                'TopSensor1': [top3_sensors[0].sensor_name if len(top3_sensors) > 0 else None],
                'TopSensor1Contribution': [top3_sensors[0].failure_contribution if len(top3_sensors) > 0 else None],
                'TopSensor2': [top3_sensors[1].sensor_name if len(top3_sensors) > 1 else None],
                'TopSensor2Contribution': [top3_sensors[1].failure_contribution if len(top3_sensors) > 1 else None],
                'TopSensor3': [top3_sensors[2].sensor_name if len(top3_sensors) > 2 else None],
                'TopSensor3Contribution': [top3_sensors[2].failure_contribution if len(top3_sensors) > 2 else None],
                'CreatedAt': [datetime.now()]
            })
            
            rul_path = temp_dir / f"rul_summary_{self.run_id}.csv"
            self.output_manager.write_dataframe(
                df_rul,
                file_path=rul_path,
                sql_table='ACM_RUL',
                add_created_at=False
            )
            tables_written.append('ACM_RUL')
            
            Console.info(f"[ForecastEngine] Wrote {len(tables_written)} forecast tables to SQL")
            
        except Exception as e:
            Console.error(f"[ForecastEngine] Failed to write outputs: {e}")
        
        return tables_written
