"""
Health Timeline Loader and Quality Assessment (v10.0.0)

Replaces scattered health loading logic from rul_engine.py and forecasting.py.
Provides unified interface for loading health data with comprehensive quality checks.

Key Features:
- SQL-first loading from ACM_HealthTimeline
- 5-level quality assessment (OK, SPARSE, GAPPY, FLAT, NOISY)
- Regime shift detection via Kolmogorov-Smirnov test
- Robust statistics and gap analysis
- Research-backed thresholds

References:
- Box & Jenkins (1970): Time series diagnostics
- Cleveland (1979): Robust statistics for time series
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import Console


class HealthQuality(Enum):
    """Data quality classification based on statistical properties"""
    OK = "OK"              # Sufficient data, good coverage, reasonable variance
    SPARSE = "SPARSE"      # Insufficient samples (< min_train_samples)
    GAPPY = "GAPPY"        # Large time gaps (> max_gap_hours)
    FLAT = "FLAT"          # Near-zero variance (std < min_std_dev)
    NOISY = "NOISY"        # Excessive variance (std > max_std_dev)
    MISSING = "MISSING"    # No data available


@dataclass
class HealthStatistics:
    """Comprehensive health timeline statistics"""
    n_samples: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    iqr: float
    max_gap_hours: float
    mean_gap_hours: float
    total_duration_hours: float
    quality: HealthQuality
    quality_reason: str


class HealthTimeline:
    """
    Loads and validates health timeline data with quality assessment.
    
    Design Philosophy:
    - SQL-first (ACM_HealthTimeline) with OutputManager cache fallback
    - Fail-fast quality checks before expensive forecasting operations
    - Regime shift detection to trigger model retraining
    - Configurable thresholds from ACM_AdaptiveConfig
    
    Usage:
        tracker = HealthTimeline(
            sql_client=sql_client,
            equip_id=1,
            run_id="run_123",
            output_manager=output_mgr,
            min_train_samples=200,
            max_gap_hours=6.0
        )
        df, quality = tracker.load_from_sql()
        if quality != HealthQuality.OK:
            Console.warn(f"Poor data quality: {quality.value}")
            return
        
        stats = tracker.get_statistics(df)
        shift_detected = tracker.detect_regime_shift(df, prev_health_df)
    """
    
    def __init__(
        self,
        sql_client: Optional[Any],
        equip_id: int,
        run_id: str,
        output_manager: Optional[Any] = None,
        min_train_samples: int = 200,
        max_gap_hours: float = 6.0,
        min_std_dev: float = 0.01,
        max_std_dev: float = 50.0,
        max_timeline_rows: int = 10000,
        downsample_freq: str = "15min"
    ):
        """
        Initialize health timeline loader.
        
        Args:
            sql_client: Database connection (pyodbc)
            equip_id: Equipment ID from Equipment table
            run_id: ACM run identifier
            output_manager: OutputManager instance for cache access
            min_train_samples: Minimum rows for SPARSE check (default 200)
            max_gap_hours: Maximum time gap for GAPPY check (default 6.0 hours)
            min_std_dev: Minimum std for FLAT check (default 0.01)
            max_std_dev: Maximum std for NOISY check (default 50.0)
            max_timeline_rows: Downsample threshold (default 10000)
            downsample_freq: Resample frequency when over limit (default "15min")
        """
        self.sql_client = sql_client
        self.equip_id = equip_id
        self.run_id = run_id
        self.output_manager = output_manager
        self.min_train_samples = min_train_samples
        self.max_gap_hours = max_gap_hours
        self.min_std_dev = min_std_dev
        self.max_std_dev = max_std_dev
        self.max_timeline_rows = max_timeline_rows
        self.downsample_freq = downsample_freq
    
    def load_from_sql(self) -> Tuple[Optional[pd.DataFrame], HealthQuality]:
        """
        Load health timeline with SQL-first priority.
        
        Priority:
        1. OutputManager cache (in-memory artifact)
        2. SQL query (ACM_HealthTimeline)
        NO CSV FALLBACK (v10.0.0 is SQL-only).
        
        Returns:
            (DataFrame, HealthQuality): Health data with Timestamp, HealthIndex, FusedZ columns
                                       and quality assessment
        """
        # Try cache first (fast path)
        if self.output_manager is not None:
            df = self.output_manager.get_cached_table("health_timeline.csv")
            if df is not None:
                Console.info(f"[HealthTracker] Using cached health_timeline ({len(df)} rows)")
                df = self._normalize_columns(df)
                df = self._apply_row_limit(df)
                quality = self.quality_check(df)
                return df, quality
        
        # SQL path
        if self.sql_client is None:
            Console.warn("[HealthTracker] No SQL client provided; cannot load health timeline")
            return None, HealthQuality.MISSING
        
        try:
            cur = self.sql_client.cursor()
            cur.execute(
                """
                SELECT Timestamp, HealthIndex, FusedZ
                FROM dbo.ACM_HealthTimeline
                WHERE EquipID = ? AND RunID = ?
                ORDER BY Timestamp
                """,
                (self.equip_id, self.run_id),
            )
            rows = cur.fetchall()
            cur.close()
            
            if not rows:
                Console.warn(f"[HealthTracker] No health timeline found for EquipID={self.equip_id}, RunID={self.run_id}")
                return None, HealthQuality.MISSING
            
            df = pd.DataFrame.from_records(rows, columns=["Timestamp", "HealthIndex", "FusedZ"])
            Console.info(f"[HealthTracker] Loaded {len(df)} health points from SQL")
            
            df = self._normalize_columns(df)
            df = self._apply_row_limit(df)
            quality = self.quality_check(df)
            
            return df, quality
            
        except Exception as e:
            Console.warn(f"[HealthTracker] Failed to load health timeline from SQL: {e}")
            return None, HealthQuality.MISSING
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and timestamps to standard format"""
        if df is None or df.empty:
            return df
        
        # Find timestamp column (case-insensitive)
        ts_col = None
        for col in ["Timestamp", "timestamp", "ts", "time"]:
            if col in df.columns:
                ts_col = col
                break
        if ts_col is None:
            ts_col = df.columns[0]  # Assume first column
        
        # Convert to datetime and strip timezone (ACM uses local-naive)
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        if df[ts_col].dt.tz is not None:
            df[ts_col] = df[ts_col].dt.tz_localize(None)
        
        # Sort and drop invalid timestamps
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
        df = df.rename(columns={ts_col: "Timestamp"})
        
        return df
    
    def _apply_row_limit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downsample if timeline exceeds maximum rows"""
        if df is None or len(df) <= self.max_timeline_rows:
            return df
        
        Console.warn(
            f"[HealthTracker] Health timeline has {len(df)} rows (limit={self.max_timeline_rows}); "
            f"downsampling to {self.downsample_freq}"
        )
        
        df = df.set_index("Timestamp").resample(self.downsample_freq).mean().dropna().reset_index()
        return df
    
    def quality_check(self, df: pd.DataFrame) -> HealthQuality:
        """
        Assess data quality and return classification.
        
        Quality checks (fail-fast order):
        1. MISSING: No data
        2. SPARSE: Fewer than min_train_samples
        3. GAPPY: Time gaps exceed max_gap_hours
        4. FLAT: Std dev below min_std_dev (near-constant signal)
        5. NOISY: Std dev above max_std_dev (extreme variance)
        6. OK: Passes all checks
        
        Args:
            df: Health DataFrame with Timestamp, HealthIndex columns
        
        Returns:
            HealthQuality enum value
        """
        if df is None or df.empty:
            return HealthQuality.MISSING
        
        # Check 1: Sufficient samples
        if len(df) < self.min_train_samples:
            return HealthQuality.SPARSE
        
        # Check 2: Time gaps (detect missing data periods)
        if "Timestamp" in df.columns:
            time_diffs = df["Timestamp"].diff().dt.total_seconds() / 3600.0  # hours
            max_gap = time_diffs.max()
            if max_gap > self.max_gap_hours:
                return HealthQuality.GAPPY
        
        # Check 3: Signal variance (flat vs noisy)
        if "HealthIndex" in df.columns:
            std = df["HealthIndex"].std()
            if std < self.min_std_dev:
                return HealthQuality.FLAT
            if std > self.max_std_dev:
                return HealthQuality.NOISY
        
        return HealthQuality.OK
    
    def get_statistics(self, df: pd.DataFrame) -> HealthStatistics:
        """
        Compute comprehensive health timeline statistics.
        
        Includes:
        - Distributional stats (mean, std, quantiles)
        - Gap analysis (max gap, mean gap)
        - Duration metrics
        - Quality assessment with reason
        
        Args:
            df: Health DataFrame with Timestamp, HealthIndex columns
        
        Returns:
            HealthStatistics dataclass with all metrics
        """
        if df is None or df.empty:
            return HealthStatistics(
                n_samples=0, mean=0.0, std=0.0, min=0.0, max=0.0,
                median=0.0, q25=0.0, q75=0.0, iqr=0.0,
                max_gap_hours=0.0, mean_gap_hours=0.0, total_duration_hours=0.0,
                quality=HealthQuality.MISSING, quality_reason="No data available"
            )
        
        health = df["HealthIndex"] if "HealthIndex" in df.columns else df.iloc[:, 1]
        
        # Distributional statistics
        n_samples = len(df)
        mean = float(health.mean())
        std = float(health.std())
        min_val = float(health.min())
        max_val = float(health.max())
        median = float(health.median())
        q25 = float(health.quantile(0.25))
        q75 = float(health.quantile(0.75))
        iqr = q75 - q25
        
        # Gap analysis
        if "Timestamp" in df.columns and len(df) > 1:
            time_diffs = df["Timestamp"].diff().dt.total_seconds() / 3600.0
            max_gap_hours = float(time_diffs.max())
            mean_gap_hours = float(time_diffs.mean())
            total_duration_hours = (df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]).total_seconds() / 3600.0
        else:
            max_gap_hours = 0.0
            mean_gap_hours = 0.0
            total_duration_hours = 0.0
        
        # Quality assessment with reason
        quality = self.quality_check(df)
        quality_reason = self._get_quality_reason(df, quality, std, max_gap_hours)
        
        return HealthStatistics(
            n_samples=n_samples,
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            median=median,
            q25=q25,
            q75=q75,
            iqr=iqr,
            max_gap_hours=max_gap_hours,
            mean_gap_hours=mean_gap_hours,
            total_duration_hours=total_duration_hours,
            quality=quality,
            quality_reason=quality_reason
        )
    
    def _get_quality_reason(self, df: pd.DataFrame, quality: HealthQuality, std: float, max_gap: float) -> str:
        """Generate human-readable quality reason"""
        if quality == HealthQuality.MISSING:
            return "No data available"
        elif quality == HealthQuality.SPARSE:
            return f"Only {len(df)} samples (need {self.min_train_samples})"
        elif quality == HealthQuality.GAPPY:
            return f"Max gap {max_gap:.1f} hours (threshold {self.max_gap_hours} hours)"
        elif quality == HealthQuality.FLAT:
            return f"Std dev {std:.4f} (threshold {self.min_std_dev})"
        elif quality == HealthQuality.NOISY:
            return f"Std dev {std:.2f} (threshold {self.max_std_dev})"
        else:
            return f"{len(df)} samples, std={std:.2f}, max_gap={max_gap:.1f}h"
    
    def detect_regime_shift(
        self,
        current_df: pd.DataFrame,
        previous_df: pd.DataFrame,
        alpha: float = 0.05
    ) -> bool:
        """
        Detect regime shift between current and previous health data.
        
        Uses two-sample Kolmogorov-Smirnov test to detect distribution changes.
        Triggers model retraining when operational regime changes significantly.
        
        References:
        - Kolmogorov (1933): Distribution-free two-sample test
        - Massey (1951): KS test for goodness of fit
        
        Args:
            current_df: Current health timeline
            previous_df: Previous health timeline (from last model fit)
            alpha: Significance level (default 0.05)
        
        Returns:
            True if regime shift detected (p-value < alpha), False otherwise
        """
        if current_df is None or previous_df is None:
            return False
        
        if current_df.empty or previous_df.empty:
            return False
        
        if "HealthIndex" not in current_df.columns or "HealthIndex" not in previous_df.columns:
            return False
        
        try:
            current_health = current_df["HealthIndex"].dropna().values
            previous_health = previous_df["HealthIndex"].dropna().values
            
            if len(current_health) < 10 or len(previous_health) < 10:
                return False  # Insufficient data for reliable test
            
            # Two-sample KS test
            statistic, p_value = stats.ks_2samp(current_health, previous_health)
            
            shift_detected = p_value < alpha
            
            if shift_detected:
                Console.info(
                    f"[HealthTracker] Regime shift detected: KS statistic={statistic:.3f}, "
                    f"p-value={p_value:.4f} (threshold={alpha})"
                )
            
            return shift_detected
            
        except Exception as e:
            Console.warn(f"[HealthTracker] Regime shift detection failed: {e}")
            return False
