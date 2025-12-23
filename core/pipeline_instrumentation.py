"""
Pipeline Stage Instrumentation for ACM v11.0.0

Provides timing and metrics collection for each pipeline stage.
Integrates with observability stack (OpenTelemetry traces, Prometheus metrics).

Phase 1.5 Implementation
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Generator
from datetime import datetime
import time

from core.observability import Console, Span


# Stub for Metrics class - observability.py doesn't have a Metrics class yet
# This will be implemented when full Prometheus metrics integration is added
class Metrics:
    """
    Stub Metrics class for compatibility.
    
    Will be replaced with actual Prometheus metrics integration.
    """
    
    @staticmethod
    def counter(name: str, value: float = 1.0, **labels) -> None:
        """Record counter metric (stub)."""
        pass
    
    @staticmethod
    def gauge(name: str, value: float, **labels) -> None:
        """Record gauge metric (stub)."""
        pass
    
    @staticmethod
    def histogram(name: str, value: float, **labels) -> None:
        """Record histogram metric (stub)."""
        pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StageMetrics:
    """
    Metrics collected for a pipeline stage.
    
    Attributes:
        stage: Stage name
        duration_ms: Execution time in milliseconds
        rows_in: Input row count
        rows_out: Output row count
        features_count: Number of features processed
        metadata: Additional stage-specific data
    """
    stage: str
    duration_ms: float
    rows_in: int = 0
    rows_out: int = 0
    features_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SQL persistence."""
        return {
            "Stage": self.stage,
            "DurationMs": self.duration_ms,
            "RowsIn": self.rows_in,
            "RowsOut": self.rows_out,
            "FeaturesCount": self.features_count,
            "Metadata": str(self.metadata) if self.metadata else None,
        }


@dataclass
class PipelineRun:
    """
    Aggregated metrics for a complete pipeline run.
    
    Tracks all stages and provides summary statistics.
    """
    run_id: Optional[int] = None
    equip_id: Optional[int] = None
    equipment_name: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    stages: List[StageMetrics] = field(default_factory=list)
    status: str = "IN_PROGRESS"
    error_message: Optional[str] = None
    
    @property
    def total_duration_ms(self) -> float:
        """Total duration across all stages."""
        return sum(s.duration_ms for s in self.stages)
    
    @property
    def total_rows_processed(self) -> int:
        """Maximum rows processed."""
        return max((s.rows_out for s in self.stages), default=0)
    
    def add_stage(self, metrics: StageMetrics) -> None:
        """Add stage metrics."""
        self.stages.append(metrics)
    
    def mark_complete(self, status: str = "SUCCESS") -> None:
        """Mark run as complete."""
        self.end_time = datetime.now()
        self.status = status
    
    def mark_failed(self, error: str) -> None:
        """Mark run as failed."""
        self.end_time = datetime.now()
        self.status = "FAILED"
        self.error_message = error
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary."""
        return {
            "run_id": self.run_id,
            "equip_id": self.equip_id,
            "equipment_name": self.equipment_name,
            "status": self.status,
            "total_duration_ms": self.total_duration_ms,
            "total_rows": self.total_rows_processed,
            "n_stages": len(self.stages),
            "stage_breakdown": {s.stage: s.duration_ms for s in self.stages},
        }


# =============================================================================
# Stage Timer Context Manager
# =============================================================================

@contextmanager
def stage_timer(
    stage_name: str,
    equipment_name: str = "",
    run_id: Optional[int] = None,
    equip_id: Optional[int] = None,
    emit_metrics: bool = True,
    emit_traces: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for timing and instrumenting pipeline stages.
    
    Collects timing metrics and optionally emits to observability stack.
    
    Args:
        stage_name: Name of the pipeline stage
        equipment_name: Equipment being processed
        run_id: Current run ID
        equip_id: Equipment ID
        emit_metrics: Whether to emit Prometheus metrics
        emit_traces: Whether to emit OpenTelemetry traces
        
    Yields:
        Dictionary for collecting stage-specific metrics (rows, features, etc.)
        
    Example:
        with stage_timer("data_load", equipment_name="FD_FAN") as metrics:
            df = load_data(...)
            metrics["rows"] = len(df)
    """
    start = time.perf_counter()
    
    # Metrics dict for caller to populate
    metrics: Dict[str, Any] = {
        "stage": stage_name,
        "equipment": equipment_name,
        "run_id": run_id,
        "equip_id": equip_id,
        "rows_in": 0,
        "rows_out": 0,
        "features_count": 0,
    }
    
    error_occurred = False
    error_message = ""
    
    try:
        if emit_traces:
            with Span(f"pipeline.{stage_name}", category="pipeline"):
                yield metrics
        else:
            yield metrics
    except Exception as e:
        error_occurred = True
        error_message = str(e)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics["duration_ms"] = elapsed_ms
        
        # Emit metrics
        if emit_metrics:
            _emit_stage_metrics(stage_name, metrics, error_occurred)
        
        # Log timing
        if error_occurred:
            Console.warn(
                f"Stage {stage_name} failed after {elapsed_ms:.1f}ms",
                error=error_message
            )
        else:
            Console.info(
                f"Stage {stage_name} completed in {elapsed_ms:.1f}ms",
                rows_in=metrics.get("rows_in", 0),
                rows_out=metrics.get("rows_out", 0),
            )


def _emit_stage_metrics(
    stage_name: str,
    metrics: Dict[str, Any],
    error: bool = False
) -> None:
    """Emit stage metrics to observability stack."""
    equipment = metrics.get("equipment", "unknown")
    
    # Duration metric
    Metrics.time(
        f"acm.pipeline.{stage_name}.duration_ms",
        metrics.get("duration_ms", 0),
        equipment=equipment
    )
    
    # Row counts
    if metrics.get("rows_in", 0) > 0:
        Metrics.count(
            f"acm.pipeline.{stage_name}.rows_in",
            metrics["rows_in"],
            equipment=equipment
        )
    
    if metrics.get("rows_out", 0) > 0:
        Metrics.count(
            f"acm.pipeline.{stage_name}.rows_out",
            metrics["rows_out"],
            equipment=equipment
        )
    
    # Feature count
    if metrics.get("features_count", 0) > 0:
        Metrics.count(
            f"acm.pipeline.{stage_name}.features",
            metrics["features_count"],
            equipment=equipment
        )
    
    # Error counter
    if error:
        Metrics.count(
            f"acm.pipeline.{stage_name}.errors",
            1,
            equipment=equipment
        )


# =============================================================================
# Pipeline Tracker
# =============================================================================

class PipelineTracker:
    """
    Tracks metrics across an entire pipeline run.
    
    Example:
        tracker = PipelineTracker(equipment_name="FD_FAN", run_id=123)
        
        with tracker.stage("data_load") as m:
            df = load_data()
            m["rows_out"] = len(df)
        
        with tracker.stage("feature_compute") as m:
            features = compute_features(df)
            m["rows_out"] = len(features)
            m["features_count"] = len(features.columns)
        
        summary = tracker.finish()
    """
    
    def __init__(
        self,
        equipment_name: str = "",
        equip_id: Optional[int] = None,
        run_id: Optional[int] = None,
        emit_metrics: bool = True,
        emit_traces: bool = True,
    ):
        self.equipment_name = equipment_name
        self.equip_id = equip_id
        self.run_id = run_id
        self.emit_metrics = emit_metrics
        self.emit_traces = emit_traces
        
        self._run = PipelineRun(
            run_id=run_id,
            equip_id=equip_id,
            equipment_name=equipment_name,
            start_time=datetime.now(),
        )
        self._current_stage: Optional[str] = None
    
    @contextmanager
    def stage(self, stage_name: str) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager for a pipeline stage.
        
        Args:
            stage_name: Name of the stage
            
        Yields:
            Metrics dictionary for stage data collection
        """
        self._current_stage = stage_name
        
        with stage_timer(
            stage_name=stage_name,
            equipment_name=self.equipment_name,
            run_id=self.run_id,
            equip_id=self.equip_id,
            emit_metrics=self.emit_metrics,
            emit_traces=self.emit_traces,
        ) as metrics:
            yield metrics
        
        # Record stage metrics
        stage_metrics = StageMetrics(
            stage=stage_name,
            duration_ms=metrics.get("duration_ms", 0),
            rows_in=metrics.get("rows_in", 0),
            rows_out=metrics.get("rows_out", 0),
            features_count=metrics.get("features_count", 0),
            metadata={k: v for k, v in metrics.items() 
                     if k not in ("stage", "equipment", "run_id", "equip_id",
                                  "rows_in", "rows_out", "features_count", "duration_ms")}
        )
        self._run.add_stage(stage_metrics)
        self._current_stage = None
    
    def finish(self, status: str = "SUCCESS") -> PipelineRun:
        """
        Mark pipeline run as complete.
        
        Args:
            status: Final status (SUCCESS, FAILED, NOOP)
            
        Returns:
            Complete PipelineRun object
        """
        self._run.mark_complete(status)
        
        # Emit total run metrics
        if self.emit_metrics:
            Metrics.time(
                "acm.pipeline.total.duration_ms",
                self._run.total_duration_ms,
                equipment=self.equipment_name
            )
            Metrics.count(
                "acm.pipeline.runs",
                1,
                equipment=self.equipment_name,
                status=status
            )
        
        Console.ok(
            f"Pipeline completed: {status}",
            equipment=self.equipment_name,
            total_ms=f"{self._run.total_duration_ms:.1f}",
            stages=len(self._run.stages),
        )
        
        return self._run
    
    def fail(self, error: str) -> PipelineRun:
        """
        Mark pipeline run as failed.
        
        Args:
            error: Error message
            
        Returns:
            Failed PipelineRun object
        """
        self._run.mark_failed(error)
        
        if self.emit_metrics:
            Metrics.count(
                "acm.pipeline.failures",
                1,
                equipment=self.equipment_name
            )
        
        Console.error(
            f"Pipeline failed: {error}",
            equipment=self.equipment_name,
            stage=self._current_stage or "unknown",
        )
        
        return self._run
    
    def get_stages(self) -> List[StageMetrics]:
        """Get all recorded stage metrics."""
        return self._run.stages
    
    def get_summary(self) -> Dict[str, Any]:
        """Get run summary."""
        return self._run.summary()


# =============================================================================
# Stage Definitions
# =============================================================================

class PipelineStages:
    """Standard pipeline stage names."""
    
    # Data loading
    DATA_LOAD = "data_load"
    DATA_VALIDATE = "data_validate"
    
    # Feature engineering
    FEATURE_COMPUTE = "feature_compute"
    FEATURE_NORMALIZE = "feature_normalize"
    
    # Regime detection
    REGIME_DETECT = "regime_detect"
    REGIME_ASSIGN = "regime_assign"
    
    # Detector scoring
    DETECTOR_AR1 = "detector_ar1"
    DETECTOR_PCA = "detector_pca"
    DETECTOR_IFOREST = "detector_iforest"
    DETECTOR_GMM = "detector_gmm"
    DETECTOR_OMR = "detector_omr"
    DETECTOR_CORR = "detector_correlation"
    
    # Fusion and health
    FUSION = "fusion"
    HEALTH_COMPUTE = "health_compute"
    
    # Episode detection
    EPISODE_DETECT = "episode_detect"
    EPISODE_DIAGNOSE = "episode_diagnose"
    
    # Forecasting
    FORECAST_HEALTH = "forecast_health"
    FORECAST_SENSOR = "forecast_sensor"
    RUL_ESTIMATE = "rul_estimate"
    
    # Output
    SQL_WRITE = "sql_write"
    CLEANUP = "cleanup"
    
    @classmethod
    def all_stages(cls) -> List[str]:
        """Get all stage names."""
        return [
            v for k, v in vars(cls).items()
            if not k.startswith("_") and isinstance(v, str) and k != "all_stages"
        ]


# =============================================================================
# Convenience Functions
# =============================================================================

def create_tracker(
    equipment_name: str,
    equip_id: Optional[int] = None,
    run_id: Optional[int] = None
) -> PipelineTracker:
    """
    Create a pipeline tracker with standard settings.
    
    Args:
        equipment_name: Name of equipment being processed
        equip_id: Equipment ID
        run_id: Run ID
        
    Returns:
        Configured PipelineTracker
    """
    return PipelineTracker(
        equipment_name=equipment_name,
        equip_id=equip_id,
        run_id=run_id,
        emit_metrics=True,
        emit_traces=True,
    )


def time_operation(
    operation_name: str,
    equipment_name: str = ""
) -> contextmanager:
    """
    Simple timing decorator for individual operations.
    
    Args:
        operation_name: Name of the operation
        equipment_name: Equipment name for tagging
        
    Returns:
        Context manager for timing
    """
    return stage_timer(
        stage_name=operation_name,
        equipment_name=equipment_name,
        emit_traces=False,  # Just metrics, no trace
    )
