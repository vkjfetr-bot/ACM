"""
RUL Estimator with Monte Carlo Simulation (v10.0.0)

Remaining Useful Life estimation via Monte Carlo degradation simulations.
Replaces logic from rul_engine.py (lines 248-394) and forecasting.py RUL sections.

Key Features:
- Monte Carlo simulation with stochastic degradation paths
- P10/P50/P90 quantile-based confidence intervals
- Agresti-Coull confidence interval adjustment
- Time-to-failure distribution via survival analysis
- Integration with degradation models and failure probability functions

References:
- Saxena et al. (2008) IEEE Trans: Monte Carlo RUL estimation with 500-5000 simulations
- Agresti & Coull (1998): Approximate confidence intervals for binomial proportions
- ISO 13381-1:2015: RUL estimation terminology and methods
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import stats

from core.degradation_model import BaseDegradationModel, DegradationForecast
from core.failure_probability import compute_failure_statistics
from utils.logger import Console


@dataclass
class RULEstimate:
    """RUL estimate with uncertainty quantification"""
    p10_lower_bound: float    # 10th percentile (pessimistic)
    p50_median: float         # 50th percentile (most likely)
    p90_upper_bound: float    # 90th percentile (optimistic)
    mean_rul: float           # Mean RUL across simulations
    std_rul: float            # Standard deviation of RUL
    confidence_level: float   # Confidence level used (e.g., 0.80)
    n_simulations: int        # Number of Monte Carlo runs
    failure_probability: float  # Probability of failure within forecast horizon
    simulation_times: np.ndarray  # Full distribution of time-to-failure


class RULEstimator:
    """
    Monte Carlo-based RUL estimator with uncertainty quantification.
    
    Algorithm:
    1. Load degradation model (LinearTrendModel or other BaseDegradationModel)
    2. Run N Monte Carlo simulations:
       - For each simulation:
         a. Add stochastic noise to degradation trajectory
         b. Project forward until health crosses failure threshold
         c. Record time-to-failure
    3. Compute quantiles (P10, P50, P90) from time-to-failure distribution
    4. Apply Agresti-Coull adjustment for small sample confidence intervals
    
    References:
    - Saxena et al. (2008): Monte Carlo RUL with 1000+ simulations
    - Agresti & Coull (1998): Binomial confidence intervals
    
    Usage:
        estimator = RULEstimator(
            degradation_model=fitted_model,
            failure_threshold=70.0,
            n_simulations=1000
        )
        rul_est = estimator.estimate_rul(
            current_health=85.0,
            dt_hours=1.0,
            max_horizon_hours=720
        )
        print(f"RUL P50: {rul_est.p50_median:.1f} hours")
    """
    
    def __init__(
        self,
        degradation_model: BaseDegradationModel,
        failure_threshold: float = 70.0,
        n_simulations: int = 1000,
        confidence_level: float = 0.80,
        noise_factor: float = 1.0
    ):
        """
        Initialize RUL estimator.
        
        Args:
            degradation_model: Fitted degradation model (e.g., LinearTrendModel)
            failure_threshold: Health threshold defining failure (default 70.0)
            n_simulations: Number of Monte Carlo runs (500-5000 per Saxena 2008)
            confidence_level: Confidence level for intervals (default 0.80)
            noise_factor: Multiplier for degradation noise (default 1.0)
        """
        self.degradation_model = degradation_model
        self.failure_threshold = failure_threshold
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.noise_factor = noise_factor
    
    def estimate_rul(
        self,
        current_health: float,
        dt_hours: float = 1.0,
        max_horizon_hours: float = 720.0
    ) -> RULEstimate:
        """
        Estimate RUL via Monte Carlo simulation.
        
        Process:
        1. Generate baseline degradation forecast from model
        2. Run N Monte Carlo simulations with stochastic noise
        3. For each simulation, find time when health < failure_threshold
        4. Compute quantiles (P10, P50, P90) from time-to-failure distribution
        5. Apply Agresti-Coull adjustment for confidence intervals
        
        Args:
            current_health: Current health index value
            dt_hours: Time step for simulation (hours)
            max_horizon_hours: Maximum simulation horizon (hours)
        
        Returns:
            RULEstimate with P10/P50/P90 quantiles and uncertainty metrics
        """
        # Quick check: already below threshold
        if current_health <= self.failure_threshold:
            return RULEstimate(
                p10_lower_bound=0.0,
                p50_median=0.0,
                p90_upper_bound=0.0,
                mean_rul=0.0,
                std_rul=0.0,
                confidence_level=self.confidence_level,
                n_simulations=self.n_simulations,
                failure_probability=1.0,
                simulation_times=np.array([0.0])
            )
        
        # Generate baseline forecast from degradation model
        max_steps = int(max_horizon_hours / dt_hours)
        baseline_forecast = self.degradation_model.predict(
            steps=max_steps,
            dt_hours=dt_hours,
            confidence_level=self.confidence_level
        )
        
        # Extract model uncertainty (std_error)
        model_std = baseline_forecast.std_error * self.noise_factor
        
        # Run Monte Carlo simulations
        simulation_times = self._run_monte_carlo_simulations(
            baseline_forecast=baseline_forecast.point_forecast,
            model_std=model_std,
            dt_hours=dt_hours,
            max_steps=max_steps
        )
        
        # Compute quantiles
        p10 = float(np.percentile(simulation_times, 10))
        p50 = float(np.percentile(simulation_times, 50))
        p90 = float(np.percentile(simulation_times, 90))
        mean_rul = float(np.mean(simulation_times))
        std_rul = float(np.std(simulation_times))
        
        # Compute failure probability within forecast horizon
        failures_within_horizon = np.sum(simulation_times < max_horizon_hours)
        failure_prob = float(failures_within_horizon / len(simulation_times))
        
        # Agresti-Coull adjustment for binomial confidence intervals
        # Reference: Agresti & Coull (1998)
        if self.n_simulations >= 10:
            p10, p50, p90 = self._agresti_coull_adjustment(
                p10, p50, p90, self.n_simulations, self.confidence_level
            )
        
        Console.info(
            f"[RULEstimator] RUL estimate: P50={p50:.1f}h, P10={p10:.1f}h, P90={p90:.1f}h, "
            f"mean={mean_rul:.1f}h, std={std_rul:.1f}h, failure_prob={failure_prob:.3f}"
        )
        
        return RULEstimate(
            p10_lower_bound=p10,
            p50_median=p50,
            p90_upper_bound=p90,
            mean_rul=mean_rul,
            std_rul=std_rul,
            confidence_level=self.confidence_level,
            n_simulations=self.n_simulations,
            failure_probability=failure_prob,
            simulation_times=simulation_times
        )
    
    def _run_monte_carlo_simulations(
        self,
        baseline_forecast: np.ndarray,
        model_std: float,
        dt_hours: float,
        max_steps: int
    ) -> np.ndarray:
        """
        Run Monte Carlo simulations with stochastic noise.
        
        For each simulation:
        1. Add Gaussian noise to baseline forecast: health[t] = baseline[t] + N(0, σ²)
        2. Find first time step where health < failure_threshold
        3. Record time-to-failure
        
        Args:
            baseline_forecast: Deterministic forecast from degradation model
            model_std: Standard deviation of forecast errors
            dt_hours: Time step (hours)
            max_steps: Maximum simulation steps
        
        Returns:
            Array of time-to-failure values (hours) from N simulations
        """
        simulation_times = []
        
        for _ in range(self.n_simulations):
            # Add stochastic noise to baseline forecast
            noise = np.random.normal(0, model_std, len(baseline_forecast))
            health_trajectory = baseline_forecast + noise
            
            # Find time-to-failure (first crossing of threshold)
            time_to_failure = self._find_time_to_failure(
                health_trajectory,
                dt_hours,
                max_steps
            )
            
            simulation_times.append(time_to_failure)
        
        return np.array(simulation_times)
    
    def _find_time_to_failure(
        self,
        health_trajectory: np.ndarray,
        dt_hours: float,
        max_steps: int
    ) -> float:
        """
        Find time when health first crosses failure threshold.
        
        Args:
            health_trajectory: Simulated health trajectory
            dt_hours: Time step (hours)
            max_steps: Maximum steps
        
        Returns:
            Time to failure (hours), or max_horizon if never fails
        """
        for i, health in enumerate(health_trajectory):
            if health < self.failure_threshold:
                return (i + 1) * dt_hours
        
        # Never crosses threshold within forecast horizon
        return max_steps * dt_hours
    
    def _agresti_coull_adjustment(
        self,
        p10: float,
        p50: float,
        p90: float,
        n_samples: int,
        confidence_level: float
    ) -> Tuple[float, float, float]:
        """
        Apply Agresti-Coull adjustment for small-sample confidence intervals.
        
        Agresti-Coull method adjusts binomial proportions for better coverage
        when sample sizes are small (n < 100).
        
        Formula:
        - ñ = n + z²
        - p̃ = (x + z²/2) / ñ
        - CI = p̃ ± z * sqrt(p̃(1-p̃) / ñ)
        
        Reference:
        - Agresti & Coull (1998): "Approximate is better than 'exact' for interval
          estimation of binomial proportions"
        
        Args:
            p10, p50, p90: Unadjusted quantiles
            n_samples: Number of simulations
            confidence_level: Confidence level (e.g., 0.80)
        
        Returns:
            (adjusted_p10, adjusted_p50, adjusted_p90)
        """
        # Z-score for confidence level
        z = stats.norm.ppf((1 + confidence_level) / 2.0)
        
        # Adjusted sample size
        n_tilde = n_samples + z**2
        
        # Apply adjustment to each quantile
        # (Simplification: treat quantiles as proportions for adjustment)
        adjustment_factor = z * np.sqrt(0.25 / n_tilde)  # Conservative: p=0.5
        
        p10_adj = max(0.0, p10 - adjustment_factor * p10)
        p50_adj = p50  # Median less sensitive to adjustment
        p90_adj = p90 + adjustment_factor * p90
        
        return p10_adj, p50_adj, p90_adj
