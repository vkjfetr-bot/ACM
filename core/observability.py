"""
ACM Unified Observability v4.0

Built on standard libraries: structlog + rich + OpenTelemetry.

COMPONENTS:
- structlog: Structured logging with processors
- rich: Colorful console output, progress indicators
- OpenTelemetry: Traces to Tempo, Metrics to Prometheus, Logs to Loki

API:
    from core.observability import log, Console, Span, traced, Progress, init

    # Initialize at startup (optional - works without init for basic logging)
    init(equipment="FD_FAN", equip_id=1, run_id="abc-123")

    # Logging - uses structlog
    log.info("Loaded data", rows=5000)
    log.warning("Low variance", sensors=3)
    log.error("SQL failed", table="ACM_Scores")

    # Console - backwards compatible wrapper
    Console.info("[DATA] Loaded 5000 rows")
    Console.warn("Warning message")

    # Progress - uses rich.progress (replaces Heartbeat)
    with Progress("Loading data") as p:
        for i in range(100):
            # work
            p.advance()

    # Spans - OpenTelemetry traces
    with Span("fit.pca"):
        model.fit(X)

    @traced("score.gmm")
    def score_gmm(X):
        return gmm.score(X)
"""
from __future__ import annotations

import atexit
import functools
import logging
import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Generator, Optional, TypeVar

# =============================================================================
# STRUCTLOG + COLORAMA SETUP (Rich doesn't work reliably when piped)
# =============================================================================

import structlog
import colorama
from colorama import Fore, Back, Style

# Initialize colorama - strip=False ensures colors even when piped
colorama.init(autoreset=True, strip=False)

# Color definitions for different log elements
class _Colors:
    """Color constants for console output."""
    # Timestamp colors
    DATE = Fore.YELLOW  # Gold-ish
    TIME = Fore.CYAN    # Blue-ish
    # Level colors
    INFO = Fore.CYAN + Style.BRIGHT
    WARN = Fore.YELLOW + Style.BRIGHT
    ERROR = Fore.RED + Style.BRIGHT
    OK = Fore.GREEN + Style.BRIGHT
    DEBUG = Style.DIM
    STATUS = Fore.MAGENTA + Style.BRIGHT  # Console-only status (purple/magenta)
    # Message
    MSG = Fore.WHITE
    RESET = Style.RESET_ALL

# Configure structlog
def _configure_structlog():
    """Configure structlog with console output."""
    
    # Processors for structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(colors=True),
    ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

_configure_structlog()

# Global logger
log = structlog.get_logger("acm")


# =============================================================================
# OPENTELEMETRY (OPTIONAL)
# =============================================================================

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, AggregationTemporality
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.trace import Status, StatusCode, SpanKind
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    otel_trace = None
    otel_metrics = None
    StatusCode = None
    Status = None
    AggregationTemporality = None

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    OTEL_EXPORTERS_AVAILABLE = True
except ImportError:
    OTEL_EXPORTERS_AVAILABLE = False

try:
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    OTEL_LOGS_AVAILABLE = True
except ImportError:
    OTEL_LOGS_AVAILABLE = False


# =============================================================================
# PYROSCOPE (CONTINUOUS PROFILING)
# =============================================================================

try:
    import pyroscope
    PYROSCOPE_AVAILABLE = True
except ImportError:
    pyroscope = None
    PYROSCOPE_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# OTLP HTTP endpoint (Alloy/Collector on port 4318)
DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"
# Loki native push endpoint
DEFAULT_LOKI_ENDPOINT = "http://localhost:3100"
# Prometheus remote-write endpoint  
DEFAULT_PROMETHEUS_ENDPOINT = "http://localhost:9090"
# Pyroscope profiling endpoint
DEFAULT_PYROSCOPE_ENDPOINT = "http://localhost:4040"

class _Config:
    """Runtime configuration."""
    service_name: str = "acm-pipeline"
    service_version: str = "10.3.0"
    otlp_endpoint: str = DEFAULT_OTLP_ENDPOINT
    loki_endpoint: str = DEFAULT_LOKI_ENDPOINT
    prometheus_endpoint: str = DEFAULT_PROMETHEUS_ENDPOINT
    pyroscope_endpoint: str = DEFAULT_PYROSCOPE_ENDPOINT
    equipment: str = ""
    equip_id: int = 0
    run_id: str = ""
    batch_num: int = 0
    batch_total: int = 0

_config = _Config()
_tracer: Optional[Any] = None
_meter: Optional[Any] = None
_loki_pusher: Optional["_LokiPusher"] = None
_pyroscope_enabled: bool = False
_initialized: bool = False
_shutdown_called: bool = False
_sql_sink: Optional["_SqlLogSink"] = None
_metrics: Dict[str, Any] = {}
_init_lock = threading.Lock()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _check_endpoint_reachable(endpoint: str, timeout: float = 1.0) -> bool:
    """Check if an HTTP endpoint is reachable (quick connectivity test)."""
    import socket
    from urllib.parse import urlparse
    
    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


# =============================================================================
# INITIALIZATION
# =============================================================================

def init(
    equipment: str = "",
    equip_id: int = 0,
    run_id: str = "",
    sql_client: Optional[Any] = None,
    service_name: str = "acm-pipeline",
    otlp_endpoint: str = DEFAULT_OTLP_ENDPOINT,
    loki_endpoint: str = DEFAULT_LOKI_ENDPOINT,
    pyroscope_endpoint: str = DEFAULT_PYROSCOPE_ENDPOINT,
    # Legacy param - maps to otlp_endpoint
    tempo_endpoint: Optional[str] = None,
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    enable_loki: bool = True,
    enable_profiling: bool = True,
) -> None:
    """Initialize observability stack.
    
    Args:
        equipment: Equipment name (e.g., "FD_FAN")
        equip_id: Equipment ID in database
        run_id: Unique run identifier
        sql_client: Optional SQLClient for ACM_RunLogs sink
        service_name: OpenTelemetry service name
        tempo_endpoint: Tempo OTLP endpoint (default: http://localhost:4321)
        loki_endpoint: Loki push endpoint (default: http://localhost:3100)
        pyroscope_endpoint: Pyroscope endpoint (default: http://localhost:4040)
        enable_tracing: Enable trace export to Tempo
        enable_metrics: Enable metric export via OTEL
        enable_profiling: Enable continuous profiling with Pyroscope
        enable_loki: Enable log push to Loki
    """
    global _initialized, _tracer, _meter, _sql_sink, _loki_pusher, _config, _metrics
    
    # Handle legacy tempo_endpoint param
    if tempo_endpoint is not None:
        otlp_endpoint = tempo_endpoint
    
    with _init_lock:
        if _initialized:
            return
        
        # Update config
        _config.service_name = service_name
        _config.otlp_endpoint = otlp_endpoint
        _config.loki_endpoint = loki_endpoint
        _config.equipment = equipment
        _config.equip_id = equip_id
        _config.run_id = run_id
        
        # Bind context to structlog
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            equipment=equipment,
            equip_id=equip_id,
            run_id=run_id,
        )
        
        # SQL log sink
        if sql_client is not None:
            _sql_sink = _SqlLogSink(sql_client, run_id, equip_id)
        
        # Loki log pusher (native Loki API, not OTLP)
        if enable_loki:
            _loki_pusher = _LokiPusher(
                endpoint=f"{loki_endpoint}/loki/api/v1/push",
                labels={
                    "app": "acm",
                    "service": service_name, 
                    "equipment": equipment or "unknown"
                },
            )
            if _loki_pusher._connected:
                Console.ok(f"[OTEL] Loki logs -> {loki_endpoint}")
            else:
                Console.warn(f"[OTEL] Loki not connected at {loki_endpoint}")
        
        # Pyroscope continuous profiling
        global _pyroscope_enabled
        if enable_profiling and PYROSCOPE_AVAILABLE:
            try:
                pyroscope_reachable = _check_endpoint_reachable(pyroscope_endpoint)
                if pyroscope_reachable:
                    pyroscope.configure(
                        application_name=service_name,
                        server_address=pyroscope_endpoint,
                        tags={
                            "equipment": equipment or "unknown",
                            "equip_id": str(equip_id),
                            "run_id": run_id or "unknown",
                        },
                    )
                    _pyroscope_enabled = True
                    _config.pyroscope_endpoint = pyroscope_endpoint
                    Console.ok(f"[OTEL] Profiling -> {pyroscope_endpoint}")
                else:
                    Console.warn(f"[OTEL] Pyroscope not reachable at {pyroscope_endpoint} - profiling disabled")
            except Exception as e:
                Console.warn(f"[OTEL] Pyroscope setup failed: {e}")
        elif enable_profiling and not PYROSCOPE_AVAILABLE:
            Console.warn("[OTEL] pyroscope-io not installed - profiling disabled (pip install pyroscope-io)")
        
        # OpenTelemetry setup for tracing
        if not OTEL_AVAILABLE or not OTEL_EXPORTERS_AVAILABLE:
            _initialized = True
            return
        
        # Pre-check OTLP endpoint connectivity to avoid noisy export errors
        otlp_reachable = _check_endpoint_reachable(otlp_endpoint)
        if not otlp_reachable:
            Console.warn(f"[OTEL] OTLP endpoint not reachable at {otlp_endpoint} - tracing/metrics disabled")
            _initialized = True
            return
        
        resource = Resource.create({SERVICE_NAME: service_name})
        
        # Tracing via OTLP
        if enable_tracing:
            trace_provider = TracerProvider(resource=resource)
            trace_provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
                )
            )
            otel_trace.set_tracer_provider(trace_provider)
            _tracer = otel_trace.get_tracer(service_name)
            Console.ok(f"[OTEL] Traces -> {otlp_endpoint}/v1/traces")
        
        # Metrics via OTLP
        if enable_metrics:
            try:
                # Import instrument types for temporality mapping
                from opentelemetry.sdk.metrics import Counter, Histogram, UpDownCounter, ObservableCounter, ObservableUpDownCounter, ObservableGauge
                
                # Use CUMULATIVE temporality for Prometheus compatibility
                # Delta temporality (default) doesn't work with Prometheus
                cumulative_temporality = {
                    Counter: AggregationTemporality.CUMULATIVE,
                    Histogram: AggregationTemporality.CUMULATIVE,
                    UpDownCounter: AggregationTemporality.CUMULATIVE,
                    ObservableCounter: AggregationTemporality.CUMULATIVE,
                    ObservableUpDownCounter: AggregationTemporality.CUMULATIVE,
                    ObservableGauge: AggregationTemporality.CUMULATIVE,
                }
                
                metric_exporter = OTLPMetricExporter(
                    endpoint=f"{otlp_endpoint}/v1/metrics",
                    preferred_temporality=cumulative_temporality,
                )
                metric_reader = PeriodicExportingMetricReader(
                    metric_exporter,
                    export_interval_millis=10000,  # Export every 10s for faster feedback
                )
                meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                otel_metrics.set_meter_provider(meter_provider)
                _meter = otel_metrics.get_meter(service_name)
                
                # ===== TIMING METRICS =====
                _metrics["stage_duration"] = _meter.create_histogram(
                    "acm_stage_duration_seconds",
                    description="Duration of pipeline stages (hierarchical: fit.pca, score.gmm, etc.)",
                    unit="s",
                )
                _metrics["run_duration"] = _meter.create_histogram(
                    "acm_run_duration_seconds",
                    description="Total run duration",
                    unit="s",
                )
                
                # ===== COUNTER METRICS =====
                _metrics["runs"] = _meter.create_counter(
                    "acm_runs_total",
                    description="Run outcomes by status (OK/FAIL/NOOP)",
                )
                _metrics["batches"] = _meter.create_counter(
                    "acm_batches_total",
                    description="Total batches processed",
                )
                _metrics["rows_processed"] = _meter.create_counter(
                    "acm_rows_processed_total",
                    description="Total rows processed",
                )
                _metrics["sql_ops"] = _meter.create_counter(
                    "acm_sql_ops_total",
                    description="SQL operations by table",
                )
                _metrics["coldstarts"] = _meter.create_counter(
                    "acm_coldstarts_total",
                    description="Coldstart completions",
                )
                _metrics["episodes"] = _meter.create_counter(
                    "acm_episodes_total",
                    description="Anomaly episodes detected",
                )
                _metrics["errors"] = _meter.create_counter(
                    "acm_errors_total",
                    description="Errors by type",
                )
                _metrics["model_refits"] = _meter.create_counter(
                    "acm_model_refits_total",
                    description="Model refit/retrain events",
                )
                
                # ===== GAUGE METRICS (current values) =====
                _metrics["health_score"] = _meter.create_gauge(
                    "acm_health_score",
                    description="Current equipment health score (0-100)",
                )
                _metrics["rul_hours"] = _meter.create_gauge(
                    "acm_rul_hours",
                    description="Remaining useful life in hours",
                )
                _metrics["active_defects"] = _meter.create_gauge(
                    "acm_active_defects",
                    description="Number of active defects",
                )
                _metrics["fused_z"] = _meter.create_gauge(
                    "acm_fused_z_score",
                    description="Current fused anomaly z-score",
                )
                _metrics["detector_z"] = _meter.create_gauge(
                    "acm_detector_z_score",
                    description="Per-detector z-scores (ar1, pca_spe, pca_t2, iforest, gmm, omr)",
                )
                _metrics["regime"] = _meter.create_gauge(
                    "acm_current_regime",
                    description="Current operating regime ID",
                )
                _metrics["data_quality"] = _meter.create_gauge(
                    "acm_data_quality_score",
                    description="Data quality score (0-100)",
                )
                
                Console.ok(f"Metrics -> {otlp_endpoint}/v1/metrics", component="OTEL")
            except Exception as e:
                Console.warn(f"[OTEL] Metrics setup failed: {e}")
        
        _initialized = True
        atexit.register(shutdown)


def shutdown() -> None:
    """Flush and shutdown all providers."""
    global _sql_sink, _loki_pusher, _shutdown_called, _pyroscope_enabled
    
    # Prevent double shutdown (atexit may call again)
    if _shutdown_called:
        return
    _shutdown_called = True
    
    # Shutdown Pyroscope profiling
    if _pyroscope_enabled and PYROSCOPE_AVAILABLE and pyroscope is not None:
        try:
            pyroscope.shutdown()
        except Exception:
            pass  # Best effort
        _pyroscope_enabled = False
    
    # Flush and shutdown OTEL metric provider to ensure final metrics are exported
    if OTEL_AVAILABLE and otel_metrics is not None:
        try:
            provider = otel_metrics.get_meter_provider()
            if hasattr(provider, 'force_flush'):
                provider.force_flush(timeout_millis=10000)
            if hasattr(provider, 'shutdown'):
                provider.shutdown()
        except Exception:
            pass  # Best effort flush
    
    # Flush and shutdown OTEL trace provider
    if OTEL_AVAILABLE and otel_trace is not None:
        try:
            provider = otel_trace.get_tracer_provider()
            if hasattr(provider, 'force_flush'):
                provider.force_flush(timeout_millis=10000)
            if hasattr(provider, 'shutdown'):
                provider.shutdown()
        except Exception:
            pass  # Best effort flush
    
    if _sql_sink:
        _sql_sink.close()
        _sql_sink = None
    if _loki_pusher:
        _loki_pusher.close()
        _loki_pusher = None


def set_context(
    equipment: Optional[str] = None,
    equip_id: Optional[int] = None,
    run_id: Optional[str] = None,
    batch_num: Optional[int] = None,
    batch_total: Optional[int] = None,
) -> None:
    """Update context for all subsequent logs."""
    if equipment is not None:
        _config.equipment = equipment
    if equip_id is not None:
        _config.equip_id = equip_id
    if run_id is not None:
        _config.run_id = run_id
    if batch_num is not None:
        _config.batch_num = batch_num
    if batch_total is not None:
        _config.batch_total = batch_total
    
    # Update structlog context
    structlog.contextvars.bind_contextvars(
        equipment=_config.equipment,
        equip_id=_config.equip_id,
        run_id=_config.run_id,
        batch_num=_config.batch_num,
        batch_total=_config.batch_total,
    )


# =============================================================================
# CONSOLE - Backwards Compatible Wrapper using Colorama
# =============================================================================

class Console:
    """
    Unified logging with structured records.
    
    Each log call creates a single LogRecord that is:
    1. Rendered to console with colors and formatting
    2. Sent to Loki with proper labels (no regex extraction needed)
    
    Usage:
        Console.info("Loading data", component="DATA", rows=5000)
        Console.warn("Low variance", component="MODEL")
        Console.error("SQL failed", component="SQL", table="ACM_Scores")
    
    The `component` parameter becomes:
    - Console: [INFO] [DATA] Loading data
    - Loki label: {component="data", level="info"} "Loading data"
    """
    
    @staticmethod
    def _format_timestamp() -> tuple:
        """Format current timestamp as (date, time) tuple."""
        now = datetime.now()
        return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    
    @staticmethod
    def _render_console(level: str, level_color: str, message: str, component: Optional[str] = None) -> None:
        """Render a log record to console with colors."""
        date, time_str = Console._format_timestamp()
        
        # Build the console line
        timestamp = f"{_Colors.DATE}[{date}{_Colors.RESET} {_Colors.TIME}{time_str}]{_Colors.RESET}"
        level_tag = f"{level_color}[{level}]{_Colors.RESET}"
        
        if component:
            comp_tag = f"{_Colors.INFO}[{component.upper()}]{_Colors.RESET} "
        else:
            comp_tag = ""
        
        print(f"{timestamp} {level_tag} {comp_tag}{_Colors.MSG}{message}{_Colors.RESET}")
    
    @staticmethod
    def _send_to_loki(level: str, message: str, component: Optional[str] = None, **kwargs) -> None:
        """Send structured log to Loki with proper labels.
        
        Automatically filters out formatting-only messages (separators, blank lines)
        to prevent log pollution.
        """
        if not _loki_pusher:
            return
        # Filter out formatting-only lines (separators, banners)
        stripped = message.strip()
        if not stripped or all(c in '=-_*#~' for c in stripped):
            return  # Skip pure separator/decoration lines
        _loki_pusher.log(level, message, component=component, **kwargs)
    
    @staticmethod
    def debug(message: str, component: Optional[str] = None, **kwargs) -> None:
        """Debug message. Only shown in console, low priority in Loki."""
        Console._render_console("DEBUG", _Colors.DEBUG, message, component)
        Console._send_to_loki("debug", message, component, **kwargs)
    
    @staticmethod
    def info(message: str, component: Optional[str] = None, skip_loki: bool = False, **kwargs) -> None:
        """Info message. Standard operational logging."""
        Console._render_console("INFO", _Colors.INFO, message, component)
        if not skip_loki:
            Console._send_to_loki("info", message, component, **kwargs)
    
    @staticmethod
    def warn(message: str, component: Optional[str] = None, **kwargs) -> None:
        """Warning message. Something unexpected but not fatal."""
        Console._render_console("WARN", _Colors.WARN, message, component)
        Console._send_to_loki("warning", message, component, **kwargs)
    
    warning = warn
    
    @staticmethod
    def error(message: str, component: Optional[str] = None, **kwargs) -> None:
        """Error message. Something failed."""
        Console._render_console("ERROR", _Colors.ERROR, message, component)
        Console._send_to_loki("error", message, component, **kwargs)
    
    @staticmethod
    def ok(message: str, component: Optional[str] = None, **kwargs) -> None:
        """Success message (green). Logs as level=info with tag=success to Loki."""
        Console._render_console("SUCCESS", _Colors.OK, message, component)
        kwargs.pop("level", None)  # Avoid conflict
        Console._send_to_loki("info", message, component, tag="success", **kwargs)
    
    @staticmethod
    def status(message: str) -> None:
        """Console-only status message (magenta). Does NOT push to Loki.
        
        Use for progress indicators, section headers, decorative separators,
        and operational messages that would pollute log analysis.
        
        Examples:
            Console.status("Processing Equipment: FD_FAN")
            Console.status("="*60)  # Section divider
        """
        date, time_str = Console._format_timestamp()
        print(f"{_Colors.DATE}[{date}{_Colors.RESET} {_Colors.TIME}{time_str}]{_Colors.RESET} {_Colors.STATUS}>>>{_Colors.RESET} {_Colors.MSG}{message}{_Colors.RESET}")
        # Intentionally NO Loki push - console only
    
    @staticmethod
    def header(title: str, char: str = "=", width: int = 60) -> None:
        """Print a section header box. Console-only, no Loki.
        
        Example:
            Console.header("Processing Equipment: FD_FAN")
            
        Output:
            >>> ============================================================
            >>> Processing Equipment: FD_FAN
            >>> ============================================================
        """
        Console.status(char * width)
        Console.status(title)
        Console.status(char * width)
    
    @staticmethod  
    def section(title: str) -> None:
        """Print a lighter section marker. Console-only, no Loki.
        
        Example:
            Console.section("Starting coldstart")
            
        Output:
            >>> --- Starting coldstart ---
        """
        Console.status(f"--- {title} ---")


# =============================================================================
# PROGRESS / HEARTBEAT - Complete No-Op
# =============================================================================

class Progress:
    """
    No-op progress indicator. All methods are stubs for backward compatibility.
    """
    
    def __init__(self, description: str = "", *args, **kwargs):
        pass
    
    def __enter__(self) -> "Progress":
        return self
    
    def __exit__(self, *args) -> None:
        pass
    
    def start(self) -> "Progress":
        return self
    
    def stop(self) -> None:
        pass
    
    def advance(self, amount: float = 1) -> None:
        pass
    
    def track(self, iterable, total: Optional[int] = None):
        yield from iterable


# Backwards compatibility alias
Heartbeat = Progress


# =============================================================================
# SPANS - OpenTelemetry Tracing
# =============================================================================

# Span kind mapping for colorful traces in Tempo
# Different span kinds get different colors in the trace view
_SPAN_KIND_MAP = {
    # CLIENT (blue): External data access, I/O operations
    "load_data": "CLIENT",
    "load": "CLIENT",
    "sql": "CLIENT",
    "persist": "CLIENT",
    "write": "CLIENT",
    # INTERNAL (green): Core processing and algorithms
    "fit": "INTERNAL",
    "score": "INTERNAL",
    "features": "INTERNAL",
    "models": "INTERNAL",
    "calibrate": "INTERNAL",
    "fusion": "INTERNAL",
    "regimes": "INTERNAL",
    "forecast": "INTERNAL",
    "train": "INTERNAL",
    "compute": "INTERNAL",
    # SERVER (purple): Entry points and control flow
    "outputs": "SERVER",
    "startup": "SERVER",
    "acm": "SERVER",
    # PRODUCER (orange): Data generation and preparation
    "data": "PRODUCER",
    "baseline": "PRODUCER",
}


class Span:
    """
    Context manager for OpenTelemetry spans.
    
    Usage:
        with Span("fit.pca"):
            model.fit(X)
    
    Spans are colored in Tempo based on span kind:
    - CLIENT (blue): External calls (SQL, file I/O)
    - INTERNAL (green): Internal processing
    - SERVER (purple): Entry points
    - PRODUCER (orange): Data generation
    """
    
    def __init__(self, name: str, **attributes):
        self.name = name
        self.attributes = attributes
        self._span: Optional[Any] = None
        self._context_token: Optional[Any] = None
        self._start_time = 0.0
    
    def _get_span_kind(self) -> Any:
        """Determine span kind based on span name prefix."""
        if not OTEL_AVAILABLE:
            return None
        # Get the first part of hierarchical name (e.g., "fit" from "fit.pca")
        prefix = self.name.split(".")[0]
        kind_str = _SPAN_KIND_MAP.get(prefix, "INTERNAL")
        return getattr(SpanKind, kind_str, SpanKind.INTERNAL)
    
    def __enter__(self) -> "Span":
        self._start_time = time.perf_counter()
        
        if _tracer is not None:
            span_kind = self._get_span_kind()
            # Include equipment in span name for easy identification in Tempo
            # e.g., "fit.pca" -> "fit.pca:FD_FAN"
            equip_suffix = f":{_config.equipment}" if _config.equipment else ""
            span_display_name = f"{self.name}{equip_suffix}"
            self._span = _tracer.start_span(span_display_name, kind=span_kind)
            self._context_token = otel_trace.use_span(self._span, end_on_exit=False)
            self._context_token.__enter__()
            
            # Add standard attributes
            if _config.equipment:
                self._span.set_attribute("acm.equipment", _config.equipment)
            if _config.equip_id:
                self._span.set_attribute("acm.equip_id", _config.equip_id)
            if _config.run_id:
                self._span.set_attribute("acm.run_id", _config.run_id)
            
            # Add span category for easier filtering
            self._span.set_attribute("acm.category", self.name.split(".")[0])
            
            # Add custom attributes
            for key, value in self.attributes.items():
                self._span.set_attribute(f"acm.{key}", value)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.perf_counter() - self._start_time
        
        # Record metric with full hierarchical stage name
        if _meter and "stage_duration" in _metrics:
            # Split into parent/child for filtering (e.g., "fit.pca" -> parent="fit", stage="fit.pca")
            parts = self.name.split(".")
            parent = parts[0] if parts else self.name
            _metrics["stage_duration"].record(
                elapsed,
                {
                    "stage": self.name,  # Full hierarchical name
                    "parent": parent,     # Top-level category
                    "equipment": _config.equipment or "unknown"
                }
            )
            
            # Also push structured timer log to Loki
            if _loki_pusher:
                _loki_pusher.log(
                    "info",
                    f"{self.name} completed in {elapsed:.3f}s",
                    log_type="timer",
                    section=self.name,
                    duration_s=round(elapsed, 6),
                    parent=parent
                )
        
        # End span
        if self._span is not None:
            if exc_val is not None:
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self._span.record_exception(exc_val)
            else:
                self._span.set_status(Status(StatusCode.OK))
            
            self._span.end()
            if self._context_token:
                self._context_token.__exit__(exc_type, exc_val, exc_tb)
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Add attribute to current span."""
        if self._span is not None:
            self._span.set_attribute(key, value)


def traced(name: str):
    """Decorator to trace a function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Span(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# METRICS
# =============================================================================

def record_batch(equipment: str, rows: int, duration_s: float) -> None:
    """Record batch processing metrics."""
    if _meter:
        if "batches" in _metrics:
            _metrics["batches"].add(1, {"equipment": equipment})
        if "rows_processed" in _metrics:
            _metrics["rows_processed"].add(rows, {"equipment": equipment})
        if "run_duration" in _metrics:
            _metrics["run_duration"].record(duration_s, {"equipment": equipment, "type": "batch"})
    if _loki_pusher:
        _loki_pusher.log("info", f"Batch completed: {rows} rows in {duration_s:.2f}s", equipment=equipment, rows=rows, duration_s=duration_s)


def record_run(equipment: str, outcome: str, duration_s: float) -> None:
    """Record run outcome metrics."""
    if _meter:
        if "runs" in _metrics:
            _metrics["runs"].add(1, {"equipment": equipment, "outcome": outcome})
        if "run_duration" in _metrics:
            _metrics["run_duration"].record(duration_s, {"equipment": equipment, "outcome": outcome})
    if _loki_pusher:
        _loki_pusher.log("info", f"Run {outcome}: {duration_s:.2f}s", equipment=equipment, outcome=outcome, duration_s=duration_s)


def record_batch_processed(equipment: str, rows: int = 0, duration_seconds: float = 0.0, **kwargs) -> None:
    """Record batch rows processed."""
    if _meter and "rows_processed" in _metrics:
        _metrics["rows_processed"].add(rows, {"equipment": equipment})
    if _loki_pusher:
        _loki_pusher.log("info", f"Batch processed: {rows} rows in {duration_seconds:.1f}s", equipment=equipment, rows=rows, duration_seconds=duration_seconds)


def record_health(equipment: str, health: float) -> None:
    """Record health score metric."""
    if _meter and "health_score" in _metrics:
        _metrics["health_score"].set(health, {"equipment": equipment})
    if _loki_pusher:
        _loki_pusher.log("info", f"Health: {health:.1f}%", equipment=equipment, health=health)


def record_health_score(equipment: str, health: float) -> None:
    """Alias for record_health."""
    record_health(equipment, health)


def record_rul(equipment: str, rul_hours: float, p10: float = 0, p50: float = 0, p90: float = 0) -> None:
    """Record RUL prediction with confidence bounds."""
    if _meter and "rul_hours" in _metrics:
        _metrics["rul_hours"].set(rul_hours, {"equipment": equipment, "percentile": "mean"})
        if p10 > 0:
            _metrics["rul_hours"].set(p10, {"equipment": equipment, "percentile": "p10"})
        if p50 > 0:
            _metrics["rul_hours"].set(p50, {"equipment": equipment, "percentile": "p50"})
        if p90 > 0:
            _metrics["rul_hours"].set(p90, {"equipment": equipment, "percentile": "p90"})
    if _loki_pusher:
        _loki_pusher.log("info", f"RUL: {rul_hours:.1f}h", equipment=equipment, rul_hours=rul_hours, p10=p10, p50=p50, p90=p90)


def record_active_defects(equipment: str, count: int) -> None:
    """Record active defect count."""
    if _meter and "active_defects" in _metrics:
        _metrics["active_defects"].set(count, {"equipment": equipment})
    if _loki_pusher:
        _loki_pusher.log("info", f"Active defects: {count}", equipment=equipment, active_defects=count)


def record_episode(equipment: str, count: int = 1, episode_id: str = "", severity: str = "warning") -> None:
    """Record episode event(s)."""
    if _meter and "episodes" in _metrics:
        _metrics["episodes"].add(count, {"equipment": equipment, "severity": severity})
    if _loki_pusher:
        _loki_pusher.log("info", f"Episode: {count} detected ({severity})", equipment=equipment, episode_id=episode_id, severity=severity, count=count)


def record_error(equipment: str, error: str, error_type: str = "unknown") -> None:
    """Record error event."""
    if _meter and "errors" in _metrics:
        _metrics["errors"].add(1, {"equipment": equipment, "error_type": error_type})
    if _loki_pusher:
        _loki_pusher.log("error", f"Error: {error}", equipment=equipment, error=error, error_type=error_type)


def record_coldstart(equipment: str, status: str = "complete") -> None:
    """Record coldstart status."""
    if _meter and "coldstarts" in _metrics and status == "complete":
        _metrics["coldstarts"].add(1, {"equipment": equipment})
    if _loki_pusher:
        _loki_pusher.log("info", f"Coldstart: {status}", equipment=equipment, coldstart_status=status)


def record_sql_op(table: str = "", operation: str = "", rows: int = 0, 
                  equipment: str = "", duration_ms: float = 0.0) -> None:
    """Record SQL operation metrics."""
    if _meter and "sql_ops" in _metrics:
        _metrics["sql_ops"].add(1, {"table": table, "operation": operation, "equipment": equipment})
    if _loki_pusher:
        _loki_pusher.log("debug", f"SQL: {operation} {table} ({rows} rows, {duration_ms:.1f}ms)", 
                        table=table, operation=operation, rows=rows, equipment=equipment, duration_ms=duration_ms)


def record_detector_scores(equipment: str, scores: dict) -> None:
    """Record per-detector z-scores.
    
    Args:
        equipment: Equipment name
        scores: Dict of detector_name -> z_score, e.g.:
            {"ar1_z": 2.5, "pca_spe_z": 1.2, "fused_z": 3.1}
    """
    if _meter:
        # Record fused score
        if "fused_z" in _metrics and "fused_z" in scores:
            _metrics["fused_z"].set(float(scores["fused_z"]), {"equipment": equipment})
        
        # Record individual detector scores
        if "detector_z" in _metrics:
            for detector in ["ar1_z", "pca_spe_z", "pca_t2_z", "iforest_z", "gmm_z", "omr_z"]:
                if detector in scores:
                    _metrics["detector_z"].set(
                        float(scores[detector]), 
                        {"equipment": equipment, "detector": detector.replace("_z", "")}
                    )
    
    if _loki_pusher:
        fused = scores.get("fused_z", 0)
        _loki_pusher.log("info", f"Detector scores: fused_z={fused:.2f}", 
                        equipment=equipment, component="detector", **{k: round(v, 3) for k, v in scores.items()})


def record_regime(equipment: str, regime_id: int, regime_label: str = "") -> None:
    """Record current operating regime."""
    if _meter and "regime" in _metrics:
        _metrics["regime"].set(regime_id, {"equipment": equipment, "label": regime_label})
    if _loki_pusher:
        _loki_pusher.log("info", f"Regime: {regime_id} ({regime_label})", 
                        equipment=equipment, component="regime", regime_id=regime_id, regime_label=regime_label)


def record_data_quality(equipment: str, quality_score: float, missing_pct: float = 0.0, 
                        outlier_pct: float = 0.0, sensors_dropped: int = 0) -> None:
    """Record data quality metrics."""
    if _meter and "data_quality" in _metrics:
        _metrics["data_quality"].set(quality_score, {"equipment": equipment})
    if _loki_pusher:
        _loki_pusher.log("info", f"Data quality: {quality_score:.1f}%", 
                        equipment=equipment, component="data",
                        quality_score=quality_score, missing_pct=missing_pct, 
                        outlier_pct=outlier_pct, sensors_dropped=sensors_dropped)


def record_model_refit(equipment: str, reason: str = "", detector: str = "") -> None:
    """Record model refit/retrain event."""
    if _meter and "model_refits" in _metrics:
        _metrics["model_refits"].add(1, {"equipment": equipment, "reason": reason, "detector": detector})
    if _loki_pusher:
        _loki_pusher.log("info", f"Model refit: {detector} ({reason})", 
                        equipment=equipment, component="model", reason=reason, detector=detector)


def log_timer(section: str, duration_s: float, pct: float = 0.0, 
              parent: str = "", total_s: float = 0.0) -> None:
    """Log timer section with structured fields for Loki.
    
    Args:
        section: Timer section name (e.g., 'models.persistence.load')
        duration_s: Duration in seconds
        pct: Percentage of parent time (optional)
        parent: Parent section name (optional)
        total_s: Total run time for percentage calculation (optional)
    """
    if _loki_pusher:
        # Format message with percentage if available
        if pct > 0:
            msg = f"{section}: {duration_s:.3f}s ({pct:.1f}%)"
        else:
            msg = f"{section}: {duration_s:.3f}s"
        
        _loki_pusher.log(
            "info", 
            msg,
            component="timer",  # Use component for Loki label filtering
            log_type="timer",
            section=section,
            parent=parent if parent else "root"
        )


def get_tracer():
    """Get the OpenTelemetry tracer."""
    return _tracer


def get_meter():
    """Get the OpenTelemetry meter."""
    return _meter


# OTEL availability flag
OTEL_AVAILABLE = OTEL_AVAILABLE if "OTEL_AVAILABLE" in dir() else False


# =============================================================================
# LOKI LOG PUSHER (Native Loki API)
# =============================================================================

import json
import urllib.request
import urllib.error

class _LokiPusher:
    """Push logs to Loki using native push API (not OTLP).
    
    Loki uses LABELS for efficient filtering and the log LINE for the message.
    Labels should contain: level, component (from [BRACKETS]), equip_id, etc.
    Loki uses LABELS for efficient filtering and the log LINE for the message.
    Labels are passed as parameters from Console methods - no regex extraction needed.
    
    Example output in Grafana:
        {app="acm", level="info", component="fuse", equip_id="1"} Computing final fusion...
    
    Label structure:
        - app: "acm" (static)
        - service: service name (static)
        - equipment: equipment name (static)
        - level: info/warning/error/debug (per-log)
        - component: fuse/data/model/sql etc. (per-log, from caller)
        - equip_id: equipment ID as string (per-log)
        - run_id: run identifier (per-log, if set)
        - tag: optional extra tag like "success" (per-log)
    """
    
    def __init__(self, endpoint: str, labels: Dict[str, str], batch_size: int = 20):
        self._endpoint = endpoint
        self._base_labels = labels  # Static labels: app, service, equipment
        self._batch_size = batch_size
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._connected = False
        
        # Test connection
        try:
            req = urllib.request.Request(
                endpoint.replace("/loki/api/v1/push", "/ready"),
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                self._connected = resp.status == 200
        except Exception:
            self._connected = False
        
        if self._connected:
            # Background flush thread
            self._thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._thread.start()
    
    def log(self, level: str, message: str, component: Optional[str] = None, **context) -> None:
        """Queue a structured log record for Loki.
        
        Args:
            level: Log level (info, warning, error, debug)
            message: Clean log message (no [COMPONENT] prefix needed)
            component: Component name (e.g., "DATA", "MODEL", "FUSE")
            **context: Additional labels (e.g., tag="success")
        
        The message is sent as-is. Component becomes a Loki label.
        No regex extraction - component is passed explicitly from Console methods.
        """
        if not self._connected:
            return
        
        # Loki expects nanosecond timestamps
        ts_ns = str(int(time.time() * 1_000_000_000))
        
        # Build dynamic labels (merged with base labels)
        # Note: Loki labels must be strings
        labels = {
            **self._base_labels,
            "level": level,
            "component": (component or "general").lower(),
            "equip_id": str(_config.equip_id) if _config.equip_id else "0",
        }
        
        # Add optional context as labels (must be strings)
        # Handle known label fields from context
        if context.get("tag"):
            labels["tag"] = str(context.pop("tag"))
        if context.get("log_type"):
            labels["log_type"] = str(context.pop("log_type"))
        if context.get("section"):
            labels["section"] = str(context.pop("section"))
        if context.get("parent"):
            labels["parent"] = str(context.pop("parent"))
        if _config.run_id:
            labels["run_id"] = _config.run_id
        
        # Queue the entry: (timestamp, labels_dict, message)
        self._queue.put((ts_ns, labels, message))
    
    def _flush_loop(self) -> None:
        """Background thread that flushes logs to Loki."""
        while not self._stop.is_set():
            self._flush_batch()
            time.sleep(2.0)
        self._flush_batch()  # Final flush
    
    def _flush_batch(self) -> None:
        """Flush queued logs to Loki.
        
        Since each log can have different labels (level, component), we need to
        group them by label set. Loki requires all entries in a stream to have
        the same labels.
        
        New payload format (proper Loki structure):
        {
            "streams": [
                {"stream": {"app":"acm", "level":"info", "component":"fuse"}, "values": [[ts, "msg1"], [ts, "msg2"]]},
                {"stream": {"app":"acm", "level":"error", "component":"sql"}, "values": [[ts, "error msg"]]}
            ]
        }
        """
        batch = []
        while len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        
        if not batch:
            return
        
        # Group by label set (convert dict to frozenset for hashing)
        # Each entry is (ts_ns, labels_dict, message)
        streams_map = {}  # type: ignore
        for ts_ns, labels, message in batch:
            label_key = frozenset(labels.items())
            if label_key not in streams_map:
                streams_map[label_key] = {"labels": labels, "values": []}
            streams_map[label_key]["values"].append([ts_ns, message])
        
        # Build Loki push payload with multiple streams
        streams_list = []
        for stream_data in streams_map.values():
            streams_list.append({
                "stream": stream_data["labels"],
                "values": stream_data["values"]
            })
        payload = {"streams": streams_list}
        
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                pass  # 204 No Content on success
        except urllib.error.HTTPError as e:
            # Log HTTP errors once (don't spam)
            if not hasattr(self, '_http_error_logged'):
                print(f"[LOKI] Push failed: {e.code} {e.reason}")
                self._http_error_logged = True
        except Exception:
            pass  # Don't crash on other failures
    
    def _flush_all(self) -> None:
        """Drain the queue completely."""
        while True:
            try:
                self._flush_batch()
                if self._queue.empty():
                    break
            except Exception:
                break
    
    def close(self) -> None:
        """Stop background thread and flush remaining logs."""
        self._stop.set()
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        # Final synchronous flush of any remaining logs
        self._flush_all()


# =============================================================================
# SQL LOG SINK
# =============================================================================

class _SqlLogSink:
    """Batched SQL log sink for ACM_RunLogs table."""
    
    def __init__(self, sql_client, run_id: str, equip_id: int, batch_size: int = 50):
        self._sql_client = sql_client
        self._run_id = run_id
        self._equip_id = equip_id
        self._batch_size = batch_size
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._written = 0
        
        # Background flush thread
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
    
    def log(self, level: str, message: str, **context) -> None:
        """Queue a log record."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message[:4000],
            "context": context,
        }
        self._queue.put(record)
    
    def _flush_loop(self) -> None:
        """Background thread that flushes logs to SQL."""
        while not self._stop.is_set():
            self._flush_batch()
            time.sleep(2.0)
        self._flush_batch()  # Final flush
    
    def _flush_batch(self) -> None:
        """Flush queued logs to SQL."""
        batch = []
        while len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        
        if not batch:
            return
        
        try:
            with self._sql_client.cursor() as cur:
                for record in batch:
                    cur.execute(
                        """INSERT INTO ACM_RunLogs 
                           (RunID, EquipID, LoggedAt, Level, Message, Context)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            self._run_id,
                            self._equip_id,
                            record["timestamp"],
                            record["level"],
                            record["message"],
                            str(record.get("context", {})),
                        )
                    )
            self._written += len(batch)
        except Exception:
            pass  # Don't crash on log failures
    
    def close(self) -> None:
        """Stop background thread and flush remaining logs."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "init",
    "shutdown",
    "set_context",
    "log",
    "Console",
    "Progress",
    "Heartbeat",  # Alias for Progress
    "Span",
    "traced",
    "record_batch",
    "record_batch_processed",
    "record_run",
    "record_health",
    "record_health_score",
    "record_rul",
    "record_active_defects",
    "record_episode",
    "record_error",
    "record_coldstart",
    "record_sql_op",
    "record_detector_scores",
    "record_regime",
    "record_data_quality",
    "record_model_refit",
    "log_timer",
    "get_tracer",
    "get_meter",
    "OTEL_AVAILABLE",
]
