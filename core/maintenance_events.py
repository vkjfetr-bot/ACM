"""
Maintenance Event Handling for ACM v11.0.0

Detects maintenance/recalibration events from sensor data patterns:
- Step changes (sudden value jumps)
- Gaps (missing data periods)
- Recalibration signatures (pattern changes)

Phase 1.4 Implementation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from core.observability import Console


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MaintenanceEvent:
    """
    Represents a detected maintenance or recalibration event.
    
    Attributes:
        timestamp: When the event occurred
        event_type: Type of event (GAP, STEP_CHANGE, RECALIBRATION, RESET)
        affected_sensors: List of sensors affected by the event
        magnitude: Size/severity of the event
        metadata: Additional event-specific information
    """
    timestamp: datetime
    event_type: str  # "GAP", "STEP_CHANGE", "RECALIBRATION", "RESET"
    affected_sensors: List[str]
    magnitude: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Valid event types
    VALID_TYPES = {"GAP", "STEP_CHANGE", "RECALIBRATION", "RESET", "SENSOR_FAILURE"}
    
    def __post_init__(self):
        if self.event_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid event_type '{self.event_type}'. "
                f"Must be one of: {self.VALID_TYPES}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SQL persistence."""
        return {
            "EventTime": self.timestamp,
            "EventType": self.event_type,
            "AffectedSensors": ",".join(self.affected_sensors) if self.affected_sensors else None,
            "Magnitude": self.magnitude,
            "Metadata": str(self.metadata) if self.metadata else None,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MaintenanceEvent":
        """Create from dictionary."""
        sensors = d.get("AffectedSensors", "")
        sensor_list = sensors.split(",") if sensors else []
        
        return cls(
            timestamp=d["EventTime"],
            event_type=d["EventType"],
            affected_sensors=sensor_list,
            magnitude=d.get("Magnitude", 0.0),
            metadata=d.get("Metadata", {}),
        )


@dataclass
class BaselineSegment:
    """
    A contiguous segment of data suitable for baseline training.
    
    Attributes:
        start: Segment start timestamp
        end: Segment end timestamp
        n_rows: Number of data points in segment
        is_valid: Whether segment meets minimum requirements
        reason: Why segment is invalid (if applicable)
    """
    start: datetime
    end: datetime
    n_rows: int
    is_valid: bool = True
    reason: str = ""
    
    @property
    def duration_hours(self) -> float:
        """Segment duration in hours."""
        return (self.end - self.start).total_seconds() / 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "n_rows": self.n_rows,
            "duration_hours": self.duration_hours,
            "is_valid": self.is_valid,
            "reason": self.reason,
        }


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MaintenanceDetectionConfig:
    """Configuration for maintenance event detection."""
    
    # Gap detection
    gap_hours: float = 24.0  # Minimum gap to consider as maintenance
    
    # Step change detection
    step_threshold_sigma: float = 3.0  # Standard deviations for step detection
    min_step_magnitude: float = 0.1  # Minimum absolute change
    
    # Recalibration detection
    variance_change_threshold: float = 2.0  # Variance ratio for recalibration
    mean_shift_threshold: float = 2.0  # Standard deviations for mean shift
    
    # Baseline segmentation
    min_segment_hours: float = 48.0  # Minimum segment duration
    min_segment_rows: int = 100  # Minimum rows per segment
    
    # Sensitivity
    lookback_window: int = 50  # Rows for baseline statistics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gap_hours": self.gap_hours,
            "step_threshold_sigma": self.step_threshold_sigma,
            "min_step_magnitude": self.min_step_magnitude,
            "variance_change_threshold": self.variance_change_threshold,
            "mean_shift_threshold": self.mean_shift_threshold,
            "min_segment_hours": self.min_segment_hours,
            "min_segment_rows": self.min_segment_rows,
            "lookback_window": self.lookback_window,
        }


# =============================================================================
# Main Handler
# =============================================================================

class MaintenanceEventHandler:
    """
    Detects maintenance and recalibration events from sensor data.
    
    This handler identifies:
    1. Data gaps (missing data periods > gap_hours)
    2. Step changes (sudden value jumps in individual sensors)
    3. Recalibration signatures (variance or mean changes)
    4. Sensor failures (sudden flat-lining or NaN sequences)
    
    Example:
        handler = MaintenanceEventHandler()
        events = handler.detect_events(df, sensor_cols=["temp", "pressure"])
        segments = handler.segment_baseline(df, events)
    """
    
    def __init__(self, config: Optional[MaintenanceDetectionConfig] = None):
        self.config = config or MaintenanceDetectionConfig()
    
    # -------------------------------------------------------------------------
    # Main Detection Methods
    # -------------------------------------------------------------------------
    
    def detect_events(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        timestamp_col: str = "Timestamp"
    ) -> List[MaintenanceEvent]:
        """
        Detect all maintenance events in the data.
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor column names to analyze
            timestamp_col: Name of timestamp column
            
        Returns:
            List of MaintenanceEvent objects, sorted by timestamp
        """
        if df.empty:
            return []
        
        if timestamp_col not in df.columns:
            Console.warn(f"Missing timestamp column '{timestamp_col}'")
            return []
        
        events: List[MaintenanceEvent] = []
        
        # Detect gaps
        gap_events = self._detect_gaps(df, timestamp_col)
        events.extend(gap_events)
        
        # Detect step changes per sensor
        for col in sensor_cols:
            if col not in df.columns:
                continue
            
            step_events = self._detect_step_changes(df, col, timestamp_col)
            events.extend(step_events)
            
            # Detect sensor failures
            failure_events = self._detect_sensor_failures(df, col, timestamp_col)
            events.extend(failure_events)
        
        # Detect recalibration patterns
        recal_events = self._detect_recalibration(df, sensor_cols, timestamp_col)
        events.extend(recal_events)
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        Console.info(
            f"Detected {len(events)} maintenance events",
            gaps=len([e for e in events if e.event_type == "GAP"]),
            steps=len([e for e in events if e.event_type == "STEP_CHANGE"]),
            recals=len([e for e in events if e.event_type == "RECALIBRATION"]),
        )
        
        return events
    
    def segment_baseline(
        self,
        df: pd.DataFrame,
        events: List[MaintenanceEvent],
        timestamp_col: str = "Timestamp"
    ) -> List[BaselineSegment]:
        """
        Split data into segments separated by maintenance events.
        
        Each segment represents a period of consistent sensor behavior,
        suitable for baseline training.
        
        Args:
            df: DataFrame with sensor data
            events: List of detected maintenance events
            timestamp_col: Name of timestamp column
            
        Returns:
            List of BaselineSegment objects
        """
        if df.empty:
            return []
        
        if timestamp_col not in df.columns:
            return []
        
        segments: List[BaselineSegment] = []
        
        # Get event timestamps
        event_times = sorted([e.timestamp for e in events])
        
        # Add data boundaries
        data_start = df[timestamp_col].min()
        data_end = df[timestamp_col].max()
        
        # Create segment boundaries
        boundaries = [data_start] + event_times + [data_end]
        
        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]
            
            # Count rows in segment
            mask = (df[timestamp_col] >= seg_start) & (df[timestamp_col] < seg_end)
            n_rows = mask.sum()
            
            # Validate segment
            is_valid = True
            reason = ""
            
            duration_hours = (seg_end - seg_start).total_seconds() / 3600
            
            if duration_hours < self.config.min_segment_hours:
                is_valid = False
                reason = f"Duration {duration_hours:.1f}h < {self.config.min_segment_hours}h"
            elif n_rows < self.config.min_segment_rows:
                is_valid = False
                reason = f"Rows {n_rows} < {self.config.min_segment_rows}"
            
            segments.append(BaselineSegment(
                start=seg_start,
                end=seg_end,
                n_rows=n_rows,
                is_valid=is_valid,
                reason=reason,
            ))
        
        n_valid = sum(1 for s in segments if s.is_valid)
        Console.info(
            f"Created {len(segments)} baseline segments ({n_valid} valid)"
        )
        
        return segments
    
    def get_valid_baseline_windows(
        self,
        segments: List[BaselineSegment]
    ) -> List[Tuple[datetime, datetime]]:
        """
        Get list of (start, end) tuples for valid baseline windows.
        
        Args:
            segments: List of BaselineSegment objects
            
        Returns:
            List of (start, end) datetime tuples for valid segments
        """
        return [(s.start, s.end) for s in segments if s.is_valid]
    
    # -------------------------------------------------------------------------
    # Detection Helpers
    # -------------------------------------------------------------------------
    
    def _detect_gaps(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> List[MaintenanceEvent]:
        """Detect data gaps exceeding threshold."""
        events = []
        
        timestamps = pd.to_datetime(df[timestamp_col])
        if len(timestamps) < 2:
            return events
        
        # Calculate time differences in hours
        gaps = timestamps.diff().dt.total_seconds() / 3600
        
        # Find gaps exceeding threshold
        gap_mask = gaps > self.config.gap_hours
        gap_indices = gaps[gap_mask].index
        
        for idx in gap_indices:
            gap_duration = gaps[idx]
            events.append(MaintenanceEvent(
                timestamp=df.loc[idx, timestamp_col],
                event_type="GAP",
                affected_sensors=[],  # Affects all sensors
                magnitude=gap_duration,
                metadata={"gap_hours": gap_duration},
            ))
        
        return events
    
    def _detect_step_changes(
        self,
        df: pd.DataFrame,
        sensor_col: str,
        timestamp_col: str
    ) -> List[MaintenanceEvent]:
        """Detect step changes in a single sensor."""
        events = []
        
        series = df[sensor_col].dropna()
        if len(series) < self.config.lookback_window:
            return events
        
        # Calculate differences
        diff = series.diff().abs()
        
        # Calculate rolling standard deviation for threshold
        rolling_std = series.rolling(
            window=self.config.lookback_window,
            min_periods=10
        ).std()
        
        # Threshold: step_threshold_sigma * rolling_std
        threshold = np.maximum(
            rolling_std * self.config.step_threshold_sigma,
            self.config.min_step_magnitude
        )
        
        # Find step changes
        step_mask = diff > threshold
        step_indices = diff[step_mask].index
        
        for idx in step_indices:
            if idx not in df.index:
                continue
            
            events.append(MaintenanceEvent(
                timestamp=df.loc[idx, timestamp_col],
                event_type="STEP_CHANGE",
                affected_sensors=[sensor_col],
                magnitude=float(diff[idx]),
                metadata={
                    "sensor": sensor_col,
                    "threshold": float(threshold.loc[idx]) if idx in threshold.index else 0,
                },
            ))
        
        return events
    
    def _detect_sensor_failures(
        self,
        df: pd.DataFrame,
        sensor_col: str,
        timestamp_col: str
    ) -> List[MaintenanceEvent]:
        """Detect sensor failures (flat-lining, NaN sequences)."""
        events = []
        
        series = df[sensor_col]
        
        # Detect flat-lining (constant values)
        rolling_var = series.rolling(
            window=self.config.lookback_window,
            min_periods=10
        ).var()
        
        # Very low variance indicates flat-lining
        flat_threshold = 1e-10
        flat_mask = rolling_var < flat_threshold
        
        # Find transitions to flat-lining
        flat_starts = flat_mask & ~flat_mask.shift(1, fill_value=False)
        
        for idx in flat_starts[flat_starts].index:
            if idx not in df.index:
                continue
            
            events.append(MaintenanceEvent(
                timestamp=df.loc[idx, timestamp_col],
                event_type="SENSOR_FAILURE",
                affected_sensors=[sensor_col],
                magnitude=0.0,
                metadata={"failure_type": "flat_line", "sensor": sensor_col},
            ))
        
        return events
    
    def _detect_recalibration(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        timestamp_col: str
    ) -> List[MaintenanceEvent]:
        """Detect recalibration patterns affecting multiple sensors."""
        events = []
        
        if len(sensor_cols) < 2:
            return events
        
        # Look for simultaneous changes across multiple sensors
        window = self.config.lookback_window
        
        # Get available sensors
        available_cols = [c for c in sensor_cols if c in df.columns]
        if len(available_cols) < 2:
            return events
        
        # Calculate rolling statistics for each sensor
        mean_shifts = pd.DataFrame(index=df.index)
        var_changes = pd.DataFrame(index=df.index)
        
        for col in available_cols:
            series = df[col].dropna()
            if len(series) < window * 2:
                continue
            
            # Rolling mean
            rolling_mean = series.rolling(window=window, min_periods=10).mean()
            rolling_std = series.rolling(window=window, min_periods=10).std()
            
            # Mean shift detection
            mean_diff = rolling_mean.diff(window)
            mean_shifts[col] = (mean_diff.abs() / rolling_std.shift(window)).fillna(0)
            
            # Variance change detection
            rolling_var = series.rolling(window=window, min_periods=10).var()
            var_ratio = rolling_var / rolling_var.shift(window)
            var_changes[col] = var_ratio.fillna(1)
        
        if mean_shifts.empty or var_changes.empty:
            return events
        
        # Find points where multiple sensors show simultaneous changes
        n_sensors_with_shift = (mean_shifts > self.config.mean_shift_threshold).sum(axis=1)
        n_sensors_with_var_change = (
            (var_changes > self.config.variance_change_threshold) |
            (var_changes < 1 / self.config.variance_change_threshold)
        ).sum(axis=1)
        
        # Recalibration: >= 50% of sensors affected
        recal_threshold = len(available_cols) * 0.5
        recal_mask = (n_sensors_with_shift >= recal_threshold) | (n_sensors_with_var_change >= recal_threshold)
        
        # Group consecutive True values
        recal_groups = (recal_mask != recal_mask.shift()).cumsum()
        
        for group_id in recal_groups[recal_mask].unique():
            group_mask = (recal_groups == group_id) & recal_mask
            group_idx = group_mask[group_mask].index[0]
            
            if group_idx not in df.index:
                continue
            
            # Find affected sensors
            affected = []
            for col in available_cols:
                if col in mean_shifts.columns and group_idx in mean_shifts.index:
                    if mean_shifts.loc[group_idx, col] > self.config.mean_shift_threshold:
                        affected.append(col)
            
            events.append(MaintenanceEvent(
                timestamp=df.loc[group_idx, timestamp_col],
                event_type="RECALIBRATION",
                affected_sensors=affected,
                magnitude=float(n_sensors_with_shift.loc[group_idx]),
                metadata={
                    "n_sensors_affected": len(affected),
                    "total_sensors": len(available_cols),
                },
            ))
        
        return events
    
    # -------------------------------------------------------------------------
    # Analysis Methods
    # -------------------------------------------------------------------------
    
    def summarize_events(self, events: List[MaintenanceEvent]) -> Dict[str, Any]:
        """Generate summary statistics for detected events."""
        if not events:
            return {
                "total_events": 0,
                "by_type": {},
                "affected_sensors": [],
                "date_range": None,
            }
        
        by_type: Dict[str, int] = {}
        affected_sensors: set = set()
        
        for event in events:
            by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
            affected_sensors.update(event.affected_sensors)
        
        return {
            "total_events": len(events),
            "by_type": by_type,
            "affected_sensors": sorted(affected_sensors),
            "date_range": {
                "first": min(e.timestamp for e in events),
                "last": max(e.timestamp for e in events),
            },
        }


# =============================================================================
# Factory Functions
# =============================================================================

def detect_maintenance_events(
    df: pd.DataFrame,
    sensor_cols: List[str],
    gap_hours: float = 24.0,
    step_threshold: float = 3.0,
    timestamp_col: str = "Timestamp"
) -> List[MaintenanceEvent]:
    """
    Convenience function to detect maintenance events.
    
    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor column names
        gap_hours: Minimum gap duration to detect
        step_threshold: Sigma threshold for step detection
        timestamp_col: Name of timestamp column
        
    Returns:
        List of MaintenanceEvent objects
    """
    config = MaintenanceDetectionConfig(
        gap_hours=gap_hours,
        step_threshold_sigma=step_threshold,
    )
    handler = MaintenanceEventHandler(config)
    return handler.detect_events(df, sensor_cols, timestamp_col)


def get_baseline_segments(
    df: pd.DataFrame,
    sensor_cols: List[str],
    timestamp_col: str = "Timestamp",
    min_hours: float = 48.0,
    min_rows: int = 100
) -> List[Tuple[datetime, datetime]]:
    """
    Get valid baseline windows from data.
    
    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor column names
        timestamp_col: Name of timestamp column
        min_hours: Minimum segment duration
        min_rows: Minimum rows per segment
        
    Returns:
        List of (start, end) datetime tuples
    """
    config = MaintenanceDetectionConfig(
        min_segment_hours=min_hours,
        min_segment_rows=min_rows,
    )
    handler = MaintenanceEventHandler(config)
    events = handler.detect_events(df, sensor_cols, timestamp_col)
    segments = handler.segment_baseline(df, events, timestamp_col)
    return handler.get_valid_baseline_windows(segments)
