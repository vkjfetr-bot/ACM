"""
ACM Observability Module - Consolidated instrumentation for traces, metrics, logs, and profiling.

This module provides a unified interface for all observability concerns:
- Traces: OpenTelemetry SDK with OTLP export to Grafana Tempo
- Metrics: OpenTelemetry SDK with OTLP export to Grafana Mimir/Prometheus
- Logs: structlog with JSON output, trace correlation, SQL sink
- Profiling: Grafana Pyroscope integration for continuous flamegraphs

Usage:
    from core.observability import init_observability, get_tracer, get_meter, get_logger
    
    # Initialize at application startup
    init_observability(
        service_name="acm-batch",
        otlp_endpoint="http://localhost:4318",  # Grafana Alloy
        pyroscope_endpoint="http://localhost:4040",  # Optional
    )
    
    # Get instrumentation handles
    tracer = get_tracer()
    meter = get_meter()
    log = get_logger()
    
    # Use in code
    @tracer.start_as_current_span("process_batch")
    def process_batch(equipment: str, df):
        log.info("batch_started", equipment=equipment, rows=len(df))
        # ...

Environment Variables:
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (default: http://localhost:4318)
    OTEL_SERVICE_NAME: Service name for traces/metrics (default: acm)
    OTEL_SDK_DISABLED: Disable OpenTelemetry entirely (default: false)
    OTEL_TRACES_SAMPLER_ARG: Trace sampling ratio (default: 1.0 = 100%)
    ACM_PYROSCOPE_ENDPOINT: Pyroscope server endpoint (optional)
    ACM_LOG_FORMAT: Log output format - json or console (default: json)
    ACM_LOG_LEVEL: Minimum log level (default: INFO)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  ACM Python Application                                      │
    │  ┌─────────────────┐ ┌─────────────────┐ ┌────────────────┐ │
    │  │ OTel SDK        │ │ structlog       │ │ Pyroscope SDK  │ │
    │  │ (traces+metrics)│ │ (JSON logs)     │ │ (profiling)    │ │
    │  └────────┬────────┘ └────────┬────────┘ └───────┬────────┘ │
    └───────────┼───────────────────┼──────────────────┼──────────┘
                │ OTLP              │ SQL + stdout     │ HTTP
                ▼                   ▼                  ▼
    ┌───────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  Grafana Alloy    │  │ ACM_RunLogs SQL │  │ Pyroscope       │
    │  (collector)      │  │ + Loki          │  │ Server          │
    └─────────┬─────────┘  └─────────────────┘  └─────────────────┘
              │
              ▼
    ┌─────────────────┐
    │ Tempo (traces)  │
    │ Mimir (metrics) │
    └─────────────────┘
"""
from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

# Optional imports with graceful fallbacks
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    metrics = None

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    OTEL_EXPORTERS_AVAILABLE = True
except ImportError:
    OTEL_EXPORTERS_AVAILABLE = False

try:
    import pyroscope
    PYROSCOPE_AVAILABLE = True
except ImportError:
    PYROSCOPE_AVAILABLE = False
    pyroscope = None

# =============================================================================
# LOG LEVELS
# =============================================================================

class LogLevel(IntEnum):
    """Log levels matching Python stdlib."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ObservabilityConfig:
    """Configuration for observability stack."""
    # Service identification
    service_name: str = "acm"
    service_version: str = "10.3.0"
    environment: str = "development"
    
    # OpenTelemetry
    otlp_endpoint: Optional[str] = None
    enable_tracing: bool = True
    enable_metrics: bool = True
    trace_sample_rate: float = 1.0  # 1.0 = 100%, 0.1 = 10%
    metrics_export_interval_ms: int = 60000  # 1 minute
    
    # Profiling
    pyroscope_endpoint: Optional[str] = None
    enable_profiling: bool = False
    profiling_sample_rate: int = 100  # Hz
    
    # Logging
    log_format: str = "json"  # "json" or "console"
    log_level: str = "INFO"
    enable_sql_sink: bool = True
    
    # SQL sink config (for BatchedSqlLogSink)
    sql_client: Any = None
    run_id: Optional[str] = None
    equip_id: Optional[int] = None
    
    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Load configuration from environment variables."""
        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "acm"),
            service_version=os.getenv("ACM_VERSION", "10.2.0"),
            environment=os.getenv("ACM_ENVIRONMENT", "development"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            enable_tracing=os.getenv("OTEL_SDK_DISABLED", "false").lower() != "true",
            enable_metrics=os.getenv("OTEL_SDK_DISABLED", "false").lower() != "true",
            trace_sample_rate=float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0")),
            pyroscope_endpoint=os.getenv("ACM_PYROSCOPE_ENDPOINT"),
            enable_profiling=bool(os.getenv("ACM_PYROSCOPE_ENDPOINT")),
            log_format=os.getenv("ACM_LOG_FORMAT", "json"),
            log_level=os.getenv("ACM_LOG_LEVEL", "INFO"),
        )


# =============================================================================
# GLOBAL STATE
# =============================================================================

_initialized = False
_config: Optional[ObservabilityConfig] = None
_tracer_provider: Optional[Any] = None
_meter_provider: Optional[Any] = None
_sql_sink: Optional[Any] = None
_structlog_logger: Optional[Any] = None

# Thread-local context for run/equipment/batch info
_context = threading.local()


# =============================================================================
# STRUCTLOG PROCESSORS
# =============================================================================

def _add_otel_context(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add OpenTelemetry trace/span IDs to log records for correlation."""
    if OTEL_AVAILABLE and trace:
        span = trace.get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def _add_acm_context(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add ACM-specific context (run_id, equip_id, batch info)."""
    run_id = getattr(_context, "run_id", None)
    equip_id = getattr(_context, "equip_id", None)
    batch_num = getattr(_context, "batch_num", None)
    batch_total = getattr(_context, "batch_total", None)
    
    if run_id:
        event_dict["run_id"] = run_id
    if equip_id:
        event_dict["equip_id"] = equip_id
    if batch_num is not None:
        event_dict["batch"] = batch_num + 1  # 1-indexed for display
        if batch_total is not None:
            event_dict["batch_total"] = batch_total
    
    return event_dict


def _add_category_prefix(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add category prefix to message for console output compatibility."""
    category = event_dict.pop("category", None)
    if category:
        event_dict["category"] = category
        # Optionally prefix message for console readability
        if _config and _config.log_format == "console":
            event_dict["event"] = f"[{category}] {event_dict.get('event', '')}"
    return event_dict


# =============================================================================
# SQL LOG SINK (bridges to BatchedSqlLogSink)
# =============================================================================

class StructlogSqlSink:
    """Bridges structlog to BatchedSqlLogSink for SQL persistence."""
    
    def __init__(self, sql_sink: Any):
        self._sql_sink = sql_sink
    
    def __call__(self, logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process log record and send to SQL sink."""
        if self._sql_sink is None:
            # Fall through to next processor
            return event_dict
        
        # Extract fields for SQL
        try:
            self._sql_sink.log(
                level=event_dict.get("level", "INFO").upper(),
                message=event_dict.get("event", ""),
                module=event_dict.get("logger", None),
                event_type=event_dict.get("category", None),
                step_name=event_dict.get("step_name", None),
                duration_ms=event_dict.get("duration_ms", None),
                row_count=event_dict.get("row_count", None),
                col_count=event_dict.get("col_count", None),
                context=event_dict,
            )
        except Exception:
            pass  # Don't let logging errors crash the application
        
        return event_dict


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_observability(
    service_name: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    pyroscope_endpoint: Optional[str] = None,
    sql_client: Any = None,
    run_id: Optional[str] = None,
    equip_id: Optional[int] = None,
    config: Optional[ObservabilityConfig] = None,
    **kwargs,
) -> None:
    """
    Initialize the observability stack.
    
    Call this once at application startup. Safe to call multiple times
    (subsequent calls update context but don't reinitialize providers).
    
    Args:
        service_name: Service name for traces/metrics (default: from env or "acm")
        otlp_endpoint: OTLP collector endpoint (default: from env)
        pyroscope_endpoint: Pyroscope server endpoint (optional)
        sql_client: SQL client for log persistence (optional)
        run_id: Current run UUID for log context
        equip_id: Equipment ID for log context
        config: Full configuration object (overrides other args)
        **kwargs: Additional config overrides
    """
    global _initialized, _config, _tracer_provider, _meter_provider, _sql_sink, _structlog_logger
    
    # Build config
    if config is None:
        config = ObservabilityConfig.from_env()
    
    # Override with explicit args
    if service_name:
        config.service_name = service_name
    if otlp_endpoint:
        config.otlp_endpoint = otlp_endpoint
    if pyroscope_endpoint:
        config.pyroscope_endpoint = pyroscope_endpoint
        config.enable_profiling = True
    if sql_client:
        config.sql_client = sql_client
    if run_id:
        config.run_id = run_id
    if equip_id is not None:
        config.equip_id = equip_id
    
    # Apply kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    _config = config
    
    # Update thread-local context
    set_context(run_id=run_id, equip_id=equip_id)
    
    # Only initialize providers once
    if _initialized:
        return
    
    # Initialize OpenTelemetry
    if OTEL_AVAILABLE and OTEL_EXPORTERS_AVAILABLE and config.otlp_endpoint:
        resource = Resource.create({
            SERVICE_NAME: config.service_name,
            "service.version": config.service_version,
            "deployment.environment": config.environment,
        })
        
        # Traces
        if config.enable_tracing:
            _tracer_provider = TracerProvider(resource=resource)
            _tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=f"{config.otlp_endpoint}/v1/traces")
                )
            )
            trace.set_tracer_provider(_tracer_provider)
        
        # Metrics
        if config.enable_metrics:
            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=f"{config.otlp_endpoint}/v1/metrics"),
                export_interval_millis=config.metrics_export_interval_ms,
            )
            _meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(_meter_provider)
    
    # Initialize Pyroscope
    if PYROSCOPE_AVAILABLE and config.enable_profiling and config.pyroscope_endpoint:
        pyroscope.configure(
            application_name=config.service_name,
            server_address=config.pyroscope_endpoint,
            sample_rate=config.profiling_sample_rate,
            detect_subprocesses=False,
            oncpu=True,
            gil_only=True,
            tags={
                "environment": config.environment,
                "version": config.service_version,
            }
        )
    
    # Initialize SQL sink
    if config.enable_sql_sink and config.sql_client:
        try:
            from core.sql_logger_v2 import BatchedSqlLogSink
            _sql_sink = BatchedSqlLogSink(
                sql_client=config.sql_client,
                run_id=config.run_id,
                equip_id=config.equip_id,
            )
        except ImportError:
            _sql_sink = None
    
    # Initialize structlog
    if STRUCTLOG_AVAILABLE:
        processors: List[Any] = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _add_otel_context,
            _add_acm_context,
            _add_category_prefix,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        # Add SQL sink if available
        if _sql_sink:
            processors.append(StructlogSqlSink(_sql_sink))
        
        # Output format
        if config.log_format == "console":
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(structlog.processors.JSONRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Set stdlib logging level
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, config.log_level.upper(), logging.INFO),
        )
    
    _initialized = True
    atexit.register(shutdown)


def shutdown() -> None:
    """Flush and shutdown all observability providers."""
    global _tracer_provider, _meter_provider, _sql_sink
    
    if _tracer_provider:
        try:
            _tracer_provider.shutdown()
        except Exception:
            pass
    
    if _meter_provider:
        try:
            _meter_provider.shutdown()
        except Exception:
            pass
    
    if _sql_sink:
        try:
            _sql_sink.close()
        except Exception:
            pass


# =============================================================================
# CONTEXT MANAGEMENT
# =============================================================================

def set_context(
    run_id: Optional[str] = None,
    equip_id: Optional[int] = None,
    batch_num: Optional[int] = None,
    batch_total: Optional[int] = None,
) -> None:
    """
    Set thread-local context for all subsequent logs/traces.
    
    Args:
        run_id: Current run UUID
        equip_id: Equipment ID from SQL
        batch_num: Current batch number (0-indexed internally)
        batch_total: Total number of batches in this run
    """
    if run_id is not None:
        _context.run_id = run_id
    if equip_id is not None:
        _context.equip_id = equip_id
    if batch_num is not None:
        _context.batch_num = batch_num
    if batch_total is not None:
        _context.batch_total = batch_total
    
    # Update SQL sink context if available
    if _sql_sink:
        if run_id is not None:
            _sql_sink.run_id = run_id
        if equip_id is not None:
            _sql_sink.equip_id = equip_id


def get_run_id() -> Optional[str]:
    """Get current run ID from context."""
    return getattr(_context, "run_id", None)


def get_equip_id() -> Optional[int]:
    """Get current equipment ID from context."""
    return getattr(_context, "equip_id", None)


# =============================================================================
# INSTRUMENTATION GETTERS
# =============================================================================

class NoOpTracer:
    """No-op tracer for when OpenTelemetry is not available."""
    
    def start_as_current_span(self, name: str, **kwargs):
        @contextmanager
        def noop_span():
            yield NoOpSpan()
        return noop_span()
    
    def start_span(self, name: str, **kwargs):
        return NoOpSpan()


class NoOpSpan:
    """No-op span for when OpenTelemetry is not available."""
    
    def set_attribute(self, key: str, value: Any) -> None:
        pass
    
    def set_status(self, status: Any) -> None:
        pass
    
    def record_exception(self, exception: Exception) -> None:
        pass
    
    def is_recording(self) -> bool:
        return False
    
    def get_span_context(self):
        return None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class NoOpMeter:
    """No-op meter for when OpenTelemetry is not available."""
    
    def create_counter(self, name: str, **kwargs):
        return NoOpCounter()
    
    def create_histogram(self, name: str, **kwargs):
        return NoOpHistogram()
    
    def create_up_down_counter(self, name: str, **kwargs):
        return NoOpCounter()
    
    def create_gauge(self, name: str, **kwargs):
        return NoOpGauge()


class NoOpCounter:
    def add(self, value: float, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass


class NoOpHistogram:
    def record(self, value: float, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass


class NoOpGauge:
    def set(self, value: float, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass


def get_tracer(name: str = "acm") -> Any:
    """
    Get an OpenTelemetry tracer.
    
    Returns a no-op tracer if OpenTelemetry is not available or not initialized.
    """
    if OTEL_AVAILABLE and trace:
        return trace.get_tracer(name)
    return NoOpTracer()


def get_meter(name: str = "acm") -> Any:
    """
    Get an OpenTelemetry meter.
    
    Returns a no-op meter if OpenTelemetry is not available or not initialized.
    """
    if OTEL_AVAILABLE and metrics:
        return metrics.get_meter(name)
    return NoOpMeter()


def get_logger(name: str = "acm") -> Any:
    """
    Get a structured logger.
    
    Returns a structlog logger if available, otherwise a stdlib logger.
    """
    if STRUCTLOG_AVAILABLE and structlog:
        return structlog.get_logger(name)
    return logging.getLogger(name)


# =============================================================================
# CONVENIENCE DECORATORS
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


def traced(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator to trace a function.
    
    Args:
        name: Span name (default: function name)
        attributes: Static attributes to add to span
    
    Example:
        @traced("process_batch", attributes={"component": "detector"})
        def process_batch(equipment: str, df):
            ...
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if OTEL_AVAILABLE and hasattr(span, 'record_exception'):
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper  # type: ignore
    return decorator


def timed(
    step_name: Optional[str] = None,
    log_result: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to time a function and log performance.
    
    Args:
        step_name: Step name for logs (default: function name)
        log_result: Whether to log timing on completion
    
    Example:
        @timed("detector.fit")
        def fit_detector(df):
            ...
    """
    def decorator(func: F) -> F:
        name = step_name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            log = get_logger()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                if log_result:
                    duration_ms = (time.perf_counter() - start) * 1000
                    log.info(
                        "step_completed",
                        category="PERF",
                        step_name=name,
                        duration_ms=duration_ms,
                    )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                log.error(
                    "step_failed",
                    category="PERF",
                    step_name=name,
                    duration_ms=duration_ms,
                    error=str(e),
                )
                raise
        
        return wrapper  # type: ignore
    return decorator


# =============================================================================
# PROFILING CONTEXT MANAGER
# =============================================================================

@contextmanager
def profile_section(tags: Optional[Dict[str, str]] = None):
    """
    Context manager for profiling a code section with Pyroscope.
    
    Args:
        tags: Additional tags to add to the profile
    
    Example:
        with profile_section({"equipment": "FD_FAN", "detector": "omr"}):
            run_detector()
    """
    if PYROSCOPE_AVAILABLE and pyroscope and _config and _config.enable_profiling:
        with pyroscope.tag_wrapper(tags or {}):
            yield
    else:
        yield


# =============================================================================
# METRICS HELPERS
# =============================================================================

# Pre-defined metrics for ACM (lazy initialization)
_batch_counter: Optional[Any] = None
_batch_duration: Optional[Any] = None
_rows_processed: Optional[Any] = None
_detector_duration: Optional[Any] = None
_health_score: Optional[Any] = None


def _ensure_metrics():
    """Lazily initialize common ACM metrics."""
    global _batch_counter, _batch_duration, _rows_processed, _detector_duration, _health_score
    
    if _batch_counter is not None:
        return
    
    meter = get_meter()
    
    _batch_counter = meter.create_counter(
        "acm.batches.processed",
        description="Number of batches processed",
    )
    
    _batch_duration = meter.create_histogram(
        "acm.batch.duration_seconds",
        description="Batch processing duration in seconds",
    )
    
    _rows_processed = meter.create_counter(
        "acm.rows.processed",
        description="Total rows processed",
    )
    
    _detector_duration = meter.create_histogram(
        "acm.detector.duration_seconds",
        description="Detector execution duration in seconds",
    )
    
    _health_score = meter.create_gauge(
        "acm.health.score",
        description="Equipment health score (0-100)",
    )


def record_batch_processed(
    equipment: str,
    duration_seconds: float,
    rows: int,
    status: str = "success",
) -> None:
    """Record batch processing metrics."""
    _ensure_metrics()
    attrs = {"equipment": equipment, "status": status}
    _batch_counter.add(1, attrs)
    _batch_duration.record(duration_seconds, attrs)
    _rows_processed.add(rows, {"equipment": equipment})


def record_detector_duration(
    detector: str,
    equipment: str,
    duration_seconds: float,
) -> None:
    """Record detector execution time."""
    _ensure_metrics()
    _detector_duration.record(
        duration_seconds,
        {"detector": detector, "equipment": equipment}
    )


def record_health_score(equipment: str, score: float) -> None:
    """Record current health score."""
    _ensure_metrics()
    _health_score.set(score, {"equipment": equipment})


# =============================================================================
# CATEGORY-SPECIFIC LOGGING (backwards compatibility with ACMLog)
# =============================================================================

class ACMLogger:
    """
    Category-aware logger for ACM.
    
    Provides methods for each ACM log category (RUN, CFG, DATA, etc.)
    while using structlog under the hood.
    """
    
    def __init__(self, name: str = "acm"):
        self._name = name
    
    def _log(self, level: str, category: str, message: str, **kwargs) -> None:
        log = get_logger(self._name)
        method = getattr(log, level.lower(), log.info)
        method(message, category=category, **kwargs)
    
    def run(self, message: str, level: str = "info", **kwargs) -> None:
        """Log pipeline lifecycle events."""
        self._log(level, "RUN", message, **kwargs)
    
    def cfg(self, message: str, level: str = "info", **kwargs) -> None:
        """Log configuration events."""
        self._log(level, "CFG", message, **kwargs)
    
    def data(self, message: str, level: str = "info", **kwargs) -> None:
        """Log data loading and validation events."""
        self._log(level, "DATA", message, **kwargs)
    
    def feat(self, message: str, level: str = "info", **kwargs) -> None:
        """Log feature engineering events."""
        self._log(level, "FEAT", message, **kwargs)
    
    def model(self, message: str, level: str = "info", **kwargs) -> None:
        """Log model training/caching events."""
        self._log(level, "MODEL", message, **kwargs)
    
    def score(self, message: str, level: str = "info", **kwargs) -> None:
        """Log scoring and detection events."""
        self._log(level, "SCORE", message, **kwargs)
    
    def fuse(self, message: str, level: str = "info", **kwargs) -> None:
        """Log fusion and episode events."""
        self._log(level, "FUSE", message, **kwargs)
    
    def regime(self, message: str, level: str = "info", **kwargs) -> None:
        """Log regime detection events."""
        self._log(level, "REGIME", message, **kwargs)
    
    def episode(self, message: str, level: str = "info", **kwargs) -> None:
        """Log episode detection events."""
        self._log(level, "EPISODE", message, **kwargs)
    
    def output(self, message: str, level: str = "info", **kwargs) -> None:
        """Log output write events."""
        self._log(level, "OUTPUT", message, **kwargs)
    
    def perf(self, step_name: str, duration_ms: float, **kwargs) -> None:
        """Log performance metrics."""
        if duration_ms >= 1000:
            dur_str = f"{duration_ms/1000:.2f}s"
        else:
            dur_str = f"{duration_ms:.1f}ms"
        
        message = f"{step_name:<25} {dur_str}"
        if "row_count" in kwargs:
            message += f" ({kwargs['row_count']:,} rows)"
        
        self._log("info", "PERF", message, step_name=step_name, duration_ms=duration_ms, **kwargs)
    
    def health(self, message: str, level: str = "info", **kwargs) -> None:
        """Log health tracking events."""
        self._log(level, "HEALTH", message, **kwargs)
    
    def rul(self, message: str, level: str = "info", **kwargs) -> None:
        """Log RUL estimation events."""
        self._log(level, "RUL", message, **kwargs)
    
    def forecast(self, message: str, level: str = "info", **kwargs) -> None:
        """Log forecasting events."""
        self._log(level, "FORECAST", message, **kwargs)
    
    def sql(self, message: str, level: str = "info", **kwargs) -> None:
        """Log SQL connection/query events."""
        self._log(level, "SQL", message, **kwargs)
    
    def threshold(self, message: str, level: str = "info", **kwargs) -> None:
        """Log threshold calculation events."""
        self._log(level, "THRESHOLD", message, **kwargs)
    
    def coldstart(self, message: str, level: str = "info", **kwargs) -> None:
        """Log coldstart events."""
        self._log(level, "COLDSTART", message, **kwargs)


# Global ACM logger instance
acm_log = ACMLogger()


# =============================================================================
# LOG CLEANUP
# =============================================================================

def cleanup_old_logs(
    sql_client: Any,
    retention_days: int = 30,
    table_name: str = "ACM_RunLogs",
) -> int:
    """
    Delete old log records from SQL table.
    
    Args:
        sql_client: SQL client with cursor() method
        retention_days: Keep logs newer than this many days
        table_name: Table to clean up
    
    Returns:
        Number of rows deleted
    """
    try:
        cur = sql_client.cursor()
        cur.execute(f"""
            DELETE FROM dbo.{table_name}
            WHERE LoggedAt < DATEADD(DAY, -{retention_days}, GETUTCDATE())
        """)
        deleted = cur.rowcount
        cur.close()
        
        if hasattr(sql_client, "conn"):
            sql_client.conn.commit()
        
        log = get_logger()
        log.info(
            "log_cleanup_completed",
            category="MAINT",
            table=table_name,
            retention_days=retention_days,
            rows_deleted=deleted,
        )
        
        return deleted
    except Exception as e:
        log = get_logger()
        log.warning(
            "log_cleanup_failed",
            category="MAINT",
            table=table_name,
            error=str(e),
        )
        return 0


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Initialization
    "init_observability",
    "shutdown",
    "ObservabilityConfig",
    
    # Context
    "set_context",
    "get_run_id",
    "get_equip_id",
    
    # Instrumentation getters
    "get_tracer",
    "get_meter",
    "get_logger",
    
    # Decorators
    "traced",
    "timed",
    
    # Profiling
    "profile_section",
    
    # Metrics helpers
    "record_batch_processed",
    "record_detector_duration",
    "record_health_score",
    
    # Category logger (backwards compat)
    "ACMLogger",
    "acm_log",
    
    # Cleanup
    "cleanup_old_logs",
    
    # Log levels
    "LogLevel",
]
