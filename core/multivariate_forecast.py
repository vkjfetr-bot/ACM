# core/multivariate_forecast.py
"""
Multivariate Time Series Forecasting Module (v10.1.0)

Implements Vector Autoregression (VAR) and correlation-aware forecasting
for sensor-level predictions that respect inter-sensor dependencies.

Key Features:
- VAR model for correlated sensor forecasting
- Sensor correlation matrix computation and tracking
- Lead-lag relationship detection
- Granger causality analysis for causal sensor identification

References:
- ISO 13381-1:2015: Condition monitoring and prognostics
- Lütkepohl, H. (2005): New Introduction to Multiple Time Series Analysis
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from core.observability import Console, Heartbeat


@dataclass
class SensorCorrelation:
    """Correlation between two sensors with lag information."""
    sensor_a: str
    sensor_b: str
    correlation: float  # Pearson correlation
    optimal_lag: int  # Lag at which correlation is maximized
    granger_pvalue: Optional[float]  # p-value for Granger causality test
    lead_sensor: Optional[str]  # Which sensor leads (if causal)


@dataclass
class MultivariateForecaseResult:
    """Result of multivariate sensor forecasting."""
    forecast_df: pd.DataFrame  # Timestamp, SensorName, ForecastValue, CI_Lower, CI_Upper
    correlation_matrix: pd.DataFrame  # Sensor x Sensor correlation matrix
    lead_lag_relationships: List[SensorCorrelation]
    method: str  # 'VAR', 'CorrelatedEWM', 'IndependentEWM'
    var_order: int  # Lag order for VAR model
    diagnostics: Dict[str, Any]


class MultivariateSensorForecaster:
    """
    Forecaster that models sensor correlations for joint predictions.
    
    Unlike univariate forecasting (each sensor independent), this:
    1. Computes correlation matrix between sensors
    2. Identifies lead-lag relationships (sensor A leads sensor B by N hours)
    3. Uses VAR model if statsmodels available, else correlated EWM
    4. Propagates forecast uncertainty through correlation structure
    
    Usage:
        forecaster = MultivariateSensorForecaster(
            sql_client=sql_client,
            equip_id=1,
            lookback_hours=720
        )
        result = forecaster.forecast(
            sensor_names=['temp1', 'temp2', 'pressure'],
            horizon_hours=168
        )
    """
    
    def __init__(
        self,
        sql_client: Any,
        equip_id: int,
        run_id: str,
        lookback_hours: float = 720.0,
        min_samples: int = 100
    ):
        self.sql_client = sql_client
        self.equip_id = equip_id
        self.run_id = run_id
        self.lookback_hours = lookback_hours
        self.min_samples = min_samples
        
        # Cached data
        self._sensor_data: Optional[pd.DataFrame] = None
        self._correlation_matrix: Optional[pd.DataFrame] = None
    
    def load_sensor_history(self, sensor_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Load sensor history from ACM_SensorNormalized_TS.
        
        Returns wide-format DataFrame with Timestamp index and sensor columns.
        """
        if self._sensor_data is not None:
            return self._sensor_data
        
        if not sensor_names:
            return None
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.lookback_hours)
            placeholders = ','.join(['?'] * len(sensor_names))
            
            query = f"""
            SELECT 
                Timestamp,
                SensorName,
                ZScore
            FROM ACM_SensorNormalized_TS
            WHERE EquipID = ?
              AND Timestamp >= ?
              AND SensorName IN ({placeholders})
            ORDER BY Timestamp ASC
            """
            
            with self.sql_client.get_cursor() as cur:
                cur.execute(query, [self.equip_id, cutoff_time] + sensor_names)
                rows = cur.fetchall()
            
            if not rows:
                Console.warn("[MultivariateForecast] No sensor data found")
                return None
            
            # Convert to long format first
            df_long = pd.DataFrame(rows, columns=['Timestamp', 'SensorName', 'ZScore'])
            df_long['Timestamp'] = pd.to_datetime(df_long['Timestamp'])
            df_long['ZScore'] = pd.to_numeric(df_long['ZScore'], errors='coerce')
            
            # Pivot to wide format (sensors as columns)
            df_wide = df_long.pivot_table(
                index='Timestamp',
                columns='SensorName',
                values='ZScore',
                aggfunc='mean'
            )
            
            # Forward-fill small gaps, then drop remaining NaN rows
            df_wide = df_wide.ffill(limit=3).dropna()
            
            if len(df_wide) < self.min_samples:
                Console.warn(f"[MultivariateForecast] Insufficient data: {len(df_wide)} < {self.min_samples}")
                return None
            
            self._sensor_data = df_wide
            Console.info(f"[MultivariateForecast] Loaded {len(df_wide)} samples for {len(df_wide.columns)} sensors")
            return df_wide
            
        except Exception as e:
            Console.error(f"[MultivariateForecast] Failed to load sensor history: {e}")
            return None
    
    def compute_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix between all sensors."""
        if self._correlation_matrix is not None:
            return self._correlation_matrix
        
        corr = df.corr(method='pearson')
        self._correlation_matrix = corr
        return corr
    
    def detect_lead_lag_relationships(
        self,
        df: pd.DataFrame,
        max_lag: int = 24
    ) -> List[SensorCorrelation]:
        """
        Detect lead-lag relationships using cross-correlation.
        
        Returns list of SensorCorrelation with optimal lag and Granger causality.
        """
        relationships = []
        sensors = list(df.columns)
        
        for i, sensor_a in enumerate(sensors):
            for sensor_b in sensors[i+1:]:
                try:
                    series_a = df[sensor_a].values
                    series_b = df[sensor_b].values
                    
                    # Cross-correlation at different lags
                    best_corr = 0.0
                    best_lag = 0
                    
                    for lag in range(-max_lag, max_lag + 1):
                        if lag < 0:
                            corr = np.corrcoef(series_a[-lag:], series_b[:lag])[0, 1]
                        elif lag > 0:
                            corr = np.corrcoef(series_a[:-lag], series_b[lag:])[0, 1]
                        else:
                            corr = np.corrcoef(series_a, series_b)[0, 1]
                        
                        if np.isfinite(corr) and abs(corr) > abs(best_corr):
                            best_corr = corr
                            best_lag = lag
                    
                    # Determine lead sensor based on lag sign
                    lead_sensor = sensor_a if best_lag > 0 else (sensor_b if best_lag < 0 else None)
                    
                    # Granger causality test (if statsmodels available)
                    granger_pvalue = None
                    try:
                        from statsmodels.tsa.stattools import grangercausalitytests
                        if abs(best_lag) > 0:
                            test_data = pd.DataFrame({sensor_a: series_a, sensor_b: series_b})
                            result = grangercausalitytests(test_data[[sensor_b, sensor_a]], maxlag=min(4, abs(best_lag)), verbose=False)
                            # Get p-value from F-test at optimal lag
                            test_lag = min(max(1, abs(best_lag)), 4)
                            granger_pvalue = result[test_lag][0]['ssr_ftest'][1]
                    except Exception:
                        pass  # Granger test optional
                    
                    relationships.append(SensorCorrelation(
                        sensor_a=sensor_a,
                        sensor_b=sensor_b,
                        correlation=best_corr,
                        optimal_lag=best_lag,
                        granger_pvalue=granger_pvalue,
                        lead_sensor=lead_sensor
                    ))
                    
                except Exception as e:
                    Console.warn(f"[MultivariateForecast] Lead-lag detection failed for {sensor_a}/{sensor_b}: {e}")
        
        # Sort by absolute correlation (strongest relationships first)
        relationships.sort(key=lambda r: abs(r.correlation), reverse=True)
        return relationships
    
    def forecast_var(
        self,
        df: pd.DataFrame,
        horizon_hours: float,
        dt_hours: float = 1.0
    ) -> Optional[pd.DataFrame]:
        """
        Forecast using Vector Autoregression (VAR) model.
        
        VAR captures linear dependencies among multiple time series,
        producing forecasts that respect inter-sensor correlations.
        """
        try:
            from statsmodels.tsa.api import VAR
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            Console.warn("[MultivariateForecast] statsmodels.tsa.VAR not available")
            return None
        
        n_steps = int(horizon_hours / dt_hours)
        sensors = list(df.columns)
        
        try:
            # Check stationarity and difference if needed
            df_stationary = df.copy()
            differenced = False
            
            for col in sensors:
                try:
                    adf_result = adfuller(df[col].dropna(), autolag='AIC')
                    if adf_result[1] > 0.05:  # Non-stationary
                        df_stationary[col] = df[col].diff().dropna()
                        differenced = True
                except Exception:
                    pass
            
            if differenced:
                df_stationary = df_stationary.dropna()
            
            # Fit VAR model with automatic lag selection
            model = VAR(df_stationary)
            
            # Use AIC to select optimal lag order (max 12 for hourly data)
            max_lags = min(12, len(df_stationary) // 10)
            lag_order_results = model.select_order(maxlags=max_lags)
            optimal_order = lag_order_results.aic
            optimal_order = max(1, min(optimal_order, max_lags))
            
            # Fit with optimal order
            var_result = model.fit(optimal_order)
            
            Console.info(f"[MultivariateForecast] VAR({optimal_order}) fitted with AIC={var_result.aic:.2f}")
            
            # Generate forecast
            forecast = var_result.forecast(df_stationary.values[-optimal_order:], steps=n_steps)
            
            # If differenced, integrate back
            if differenced:
                last_values = df.iloc[-1].values
                forecast_integrated = np.zeros_like(forecast)
                forecast_integrated[0] = last_values + forecast[0]
                for i in range(1, n_steps):
                    forecast_integrated[i] = forecast_integrated[i-1] + forecast[i]
                forecast = forecast_integrated
            
            # Build forecast DataFrame
            last_timestamp = df.index[-1]
            timestamps = [last_timestamp + timedelta(hours=dt_hours * (i + 1)) for i in range(n_steps)]
            
            forecast_df = pd.DataFrame(forecast, columns=sensors, index=timestamps)
            
            # Compute confidence intervals from VAR covariance
            # Approximate: use forecast error variance
            forecast_errors = var_result.forecast_interval(
                df_stationary.values[-optimal_order:],
                steps=n_steps,
                alpha=0.05
            )
            lower = pd.DataFrame(forecast_errors[1], columns=sensors, index=timestamps)
            upper = pd.DataFrame(forecast_errors[2], columns=sensors, index=timestamps)
            
            # Melt to long format for SQL storage
            records = []
            for ts in timestamps:
                for sensor in sensors:
                    records.append({
                        'Timestamp': ts,
                        'SensorName': sensor,
                        'ForecastValue': float(forecast_df.loc[ts, sensor]),
                        'CiLower': float(lower.loc[ts, sensor]),
                        'CiUpper': float(upper.loc[ts, sensor]),
                        'Method': f'VAR({optimal_order})',
                        'EquipID': self.equip_id,
                        'RunID': self.run_id
                    })
            
            return pd.DataFrame(records)
            
        except Exception as e:
            Console.error(f"[MultivariateForecast] VAR forecasting failed: {e}")
            return None
    
    def forecast_correlated_ewm(
        self,
        df: pd.DataFrame,
        horizon_hours: float,
        dt_hours: float = 1.0,
        alpha: float = 0.3
    ) -> pd.DataFrame:
        """
        Fallback: Correlated Exponential Weighted Moving Average.
        
        Unlike pure EWM, this adjusts each sensor's forecast based on
        correlated sensors' recent behavior.
        """
        n_steps = int(horizon_hours / dt_hours)
        sensors = list(df.columns)
        corr_matrix = self.compute_correlation_matrix(df)
        
        records = []
        last_timestamp = df.index[-1]
        
        for sensor in sensors:
            series = df[sensor]
            
            # Compute EWM trend
            ewm = series.ewm(alpha=alpha, adjust=False).mean()
            ewm_std = series.ewm(alpha=alpha, adjust=False).std()
            
            current_value = ewm.iloc[-1]
            current_std = ewm_std.iloc[-1] if ewm_std.iloc[-1] > 0 else series.std()
            
            # Trend from last 24 hours
            recent = series.tail(24)
            if len(recent) >= 2:
                x = np.arange(len(recent))
                slope, _ = np.polyfit(x, recent.values, 1)
            else:
                slope = 0.0
            
            # Adjust trend based on correlated sensors
            adjustment = 0.0
            for other_sensor in sensors:
                if other_sensor != sensor:
                    corr = corr_matrix.loc[sensor, other_sensor]
                    if abs(corr) > 0.5:  # Only use strong correlations
                        other_recent = df[other_sensor].tail(24)
                        if len(other_recent) >= 2:
                            x = np.arange(len(other_recent))
                            other_slope, _ = np.polyfit(x, other_recent.values, 1)
                            # If correlated sensor is trending differently, pull forecast
                            adjustment += corr * (other_slope - slope) * 0.3
            
            # Generate forecast with correlation-adjusted trend
            adjusted_slope = slope + adjustment
            
            for i in range(n_steps):
                hours_ahead = dt_hours * (i + 1)
                forecast_value = current_value + (adjusted_slope * hours_ahead)
                
                # Confidence interval widens with time
                ci_width = 1.96 * current_std * np.sqrt(1 + hours_ahead / 24)
                
                records.append({
                    'Timestamp': last_timestamp + timedelta(hours=hours_ahead),
                    'SensorName': sensor,
                    'ForecastValue': float(forecast_value),
                    'CiLower': float(forecast_value - ci_width),
                    'CiUpper': float(forecast_value + ci_width),
                    'Method': 'CorrelatedEWM',
                    'EquipID': self.equip_id,
                    'RunID': self.run_id
                })
        
        return pd.DataFrame(records)
    
    def forecast(
        self,
        sensor_names: List[str],
        horizon_hours: float = 168.0,
        prefer_var: bool = True
    ) -> Optional[MultivariateForecaseResult]:
        """
        Main forecasting method - chooses best available approach.
        
        Priority:
        1. VAR model (if statsmodels available and sufficient data)
        2. Correlated EWM (if VAR fails or unavailable)
        3. None (if insufficient data)
        """
        # Load data
        df = self.load_sensor_history(sensor_names)
        if df is None:
            return None
        
        # Filter to requested sensors only
        available_sensors = [s for s in sensor_names if s in df.columns]
        if len(available_sensors) < 2:
            Console.warn("[MultivariateForecast] Need at least 2 sensors for multivariate forecasting")
            return None
        
        df = df[available_sensors]
        
        # Compute correlations and lead-lag
        corr_matrix = self.compute_correlation_matrix(df)
        lead_lag = self.detect_lead_lag_relationships(df, max_lag=24)
        
        # Log strong correlations
        strong_corrs = [(r.sensor_a, r.sensor_b, r.correlation) 
                       for r in lead_lag if abs(r.correlation) > 0.7]
        if strong_corrs:
            Console.info(f"[MultivariateForecast] Strong correlations: {strong_corrs[:3]}")
        
        # Try VAR first
        forecast_df = None
        method = 'None'
        var_order = 0
        
        if prefer_var:
            forecast_df = self.forecast_var(df, horizon_hours)
            if forecast_df is not None:
                method = forecast_df['Method'].iloc[0] if len(forecast_df) > 0 else 'VAR'
                # Extract order from method string like 'VAR(3)'
                if 'VAR(' in method:
                    try:
                        var_order = int(method.split('(')[1].split(')')[0])
                    except:
                        var_order = 1
        
        # Fallback to correlated EWM
        if forecast_df is None:
            forecast_df = self.forecast_correlated_ewm(df, horizon_hours)
            method = 'CorrelatedEWM'
        
        if forecast_df is None or len(forecast_df) == 0:
            return None
        
        return MultivariateForecaseResult(
            forecast_df=forecast_df,
            correlation_matrix=corr_matrix,
            lead_lag_relationships=lead_lag,
            method=method,
            var_order=var_order,
            diagnostics={
                'n_sensors': len(available_sensors),
                'n_samples': len(df),
                'horizon_hours': horizon_hours,
                'strong_correlations': len([r for r in lead_lag if abs(r.correlation) > 0.7]),
                'causal_relationships': len([r for r in lead_lag if r.granger_pvalue and r.granger_pvalue < 0.05])
            }
        )


def save_sensor_correlations(
    sql_client: Any,
    equip_id: int,
    run_id: str,
    correlations: List[SensorCorrelation]
) -> int:
    """
    Save sensor correlation matrix to ACM_SensorCorrelations table.
    
    Returns number of rows written.
    """
    if not correlations:
        return 0
    
    try:
        records = []
        for corr in correlations:
            records.append({
                'EquipID': equip_id,
                'RunID': run_id,
                'SensorA': corr.sensor_a,
                'SensorB': corr.sensor_b,
                'Correlation': corr.correlation,
                'OptimalLag': corr.optimal_lag,
                'GrangerPValue': corr.granger_pvalue,
                'LeadSensor': corr.lead_sensor,
                'CreatedAt': datetime.now()
            })
        
        df = pd.DataFrame(records)
        
        # Use bulk insert if available
        with sql_client.get_cursor() as cur:
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO ACM_SensorCorrelations 
                    (EquipID, RunID, SensorA, SensorB, Correlation, OptimalLag, GrangerPValue, LeadSensor, CreatedAt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['EquipID'], row['RunID'], row['SensorA'], row['SensorB'],
                    row['Correlation'], row['OptimalLag'], row['GrangerPValue'],
                    row['LeadSensor'], row['CreatedAt']
                ))
            sql_client.connection.commit()
        
        Console.info(f"[MultivariateForecast] Saved {len(records)} sensor correlations")
        return len(records)
        
    except Exception as e:
        Console.error(f"[MultivariateForecast] Failed to save correlations: {e}")
        return 0
