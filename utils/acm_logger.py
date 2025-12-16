"""ACM Standardized Logging System v1.0

This module provides a unified, standardized logging interface for ACM.

## Design Principles

1. **Single Source of Truth**: All logging goes through ACMLog class
2. **Structured Data**: Logs carry typed metadata, not just strings  
3. **Consistent Prefixes**: Predefined categories ensure uniform formatting
4. **Level Discipline**: Clear mapping of severity to log levels
5. **SQL-Ready**: Records flow seamlessly to BatchedSqlLogSink

## Log Categories (Prefixes)

| Category   | Description                      | Examples                           |
|------------|----------------------------------|-----------------------------------|
| RUN        | Pipeline lifecycle events        | Start/end run, batch window       |
| CFG        | Configuration loading/updates    | Load from SQL, param changes      |
| DATA       | Data loading and validation      | Historian load, row counts        |
| FEAT       | Feature engineering              | Build features, imputation        |
| MODEL      | Model training/caching           | Fit, cache load, retrain          |
| SCORE      | Scoring and detection            | Detector scores, anomalies        |
| FUSE       | Fusion and episodes              | Weight tuning, episode detection  |
| OUTPUT     | Output writes                    | SQL/CSV writes, table counts      |
| PERF       | Performance metrics              | Timing, throughput                |
| HEALTH     | Health tracking                  | Health index, RUL estimates       |

### Detector Sub-Categories
| Category   | Description                      |
|------------|----------------------------------|
| DET.OMR    | Overall Model Residual detector  |
| DET.AR1    | AR(1) baseline detector          |
| DET.PCA    | PCA subspace detector            |
| DET.MHAL   | Mahalanobis distance detector    |
| DET.GMM    | GMM clustering detector          |
| DET.IFOR   | Isolation Forest detector        |

## Usage

```python
from utils.acm_logger import ACMLog

# Basic logging
ACMLog.run("Started batch processing", run_id="abc-123")
ACMLog.cfg("Loaded config from SQL", equip_id=5010)
ACMLog.data("Retrieved rows from historian", row_count=500, duration_ms=150.5)

# Detector-specific logging
ACMLog.det("OMR", "Fitted model", model_type="ridge", r2=0.95)
ACMLog.det("AR1", "Column warnings", near_constant=5, clamped=10)

# Error handling
ACMLog.error("MODEL", "Failed to load cache", exc=e)
ACMLog.warn("DATA", "Low-variance sensors detected", count=3)

# Performance timing
ACMLog.perf("features.build", duration_ms=450.3, row_count=1000)
```

## Level Guidelines

- **DEBUG**: Detailed diagnostic info (usually disabled in production)
- **INFO**: Normal operational messages (milestones, successful operations)
- **WARNING**: Recoverable issues, degraded operation, fallbacks
- **ERROR**: Failures that impact results but don't crash
- **CRITICAL**: Unrecoverable errors, pipeline abort
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

# Import the base logger
from utils.logger import _logger, Console, LogLevel


class Category(str, Enum):
    """Standardized log categories for ACM."""
    # Pipeline lifecycle
    RUN = "RUN"
    CFG = "CFG"
    
    # Data pipeline
    DATA = "DATA"
    FEAT = "FEAT"
    
    # Model lifecycle
    MODEL = "MODEL"
    CACHE = "CACHE"
    
    # Detection and fusion
    SCORE = "SCORE"
    FUSE = "FUSE"
    REGIME = "REGIME"
    EPISODE = "EPISODE"
    
    # Detectors (use det() method with detector name)
    DET = "DET"
    
    # Output and performance
    OUTPUT = "OUTPUT"
    PERF = "PERF"
    
    # Health and forecasting
    HEALTH = "HEALTH"
    RUL = "RUL"
    FORECAST = "FORECAST"
    
    # Infrastructure
    SQL = "SQL"
    LOG = "LOG"
    
    # Threshold and tuning
    THRESHOLD = "THRESHOLD"
    TUNE = "TUNE"
    ADAPTIVE = "ADAPTIVE"
    
    # Coldstart
    COLDSTART = "COLDSTART"
    BASELINE = "BASELINE"


# Detector names for det() logging
DETECTOR_NAMES = {"OMR", "AR1", "PCA", "MHAL", "GMM", "IFOR", "CORR", "CUSUM"}


@dataclass
class ACMLogRecord:
    """Structured ACM log record with typed fields.
    
    This record carries all metadata needed for:
    - Console display (formatted message)
    - SQL storage (typed columns)
    - Analysis and querying
    """
    timestamp: datetime
    level: str
    category: str
    message: str
    
    # Optional typed fields for SQL storage
    run_id: Optional[str] = None
    equip_id: Optional[int] = None
    stage: Optional[str] = None
    step_name: Optional[str] = None
    duration_ms: Optional[float] = None
    row_count: Optional[int] = None
    col_count: Optional[int] = None
    
    # Data pipeline fields
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    baseline_start: Optional[datetime] = None
    baseline_end: Optional[datetime] = None
    data_quality_metric: Optional[str] = None
    data_quality_value: Optional[float] = None
    
    # Model fields  
    model_type: Optional[str] = None
    model_metric: Optional[str] = None
    model_value: Optional[float] = None
    
    # Detector fields
    detector_name: Optional[str] = None
    detector_metric: Optional[str] = None
    detector_value: Optional[float] = None
    
    # Error handling
    error_type: Optional[str] = None
    error_detail: Optional[str] = None
    
    # Arbitrary context (stored as JSON)
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


class ACMLog:
    """Unified logging interface for ACM.
    
    All methods are static for easy access across modules.
    Messages are formatted consistently and routed to both
    console and SQL sink (if attached).
    """
    
    # Thread-local storage for context (run_id, equip_id)
    _context = threading.local()
    
    @classmethod
    def set_context(cls, run_id: Optional[str] = None, equip_id: Optional[int] = None) -> None:
        """Set thread-local context for all subsequent logs."""
        if run_id is not None:
            cls._context.run_id = run_id
        if equip_id is not None:
            cls._context.equip_id = equip_id
    
    @classmethod
    def get_run_id(cls) -> Optional[str]:
        return getattr(cls._context, 'run_id', None)
    
    @classmethod
    def get_equip_id(cls) -> Optional[int]:
        return getattr(cls._context, 'equip_id', None)
    
    # =========================================================================
    # CORE LOGGING METHODS
    # =========================================================================
    
    @staticmethod
    def _log(
        level: str,
        category: str,
        message: str,
        skip_sql: bool = False,
        **kwargs
    ) -> None:
        """Internal logging method.
        
        Formats message with category prefix and routes to Console + SQL sink.
        """
        # Format message with category prefix
        formatted_msg = f"[{category}] {message}"
        
        # Route to appropriate Console method
        if level == "DEBUG":
            Console.debug(formatted_msg, **kwargs)
        elif level == "INFO":
            Console.info(formatted_msg, skip_sql=skip_sql, **kwargs)
        elif level == "WARNING":
            Console.warn(formatted_msg, **kwargs)
        elif level == "ERROR":
            Console.error(formatted_msg, **kwargs)
        elif level == "CRITICAL":
            Console.critical(formatted_msg, **kwargs)
        else:
            Console.info(formatted_msg, skip_sql=skip_sql, **kwargs)
    
    # =========================================================================
    # CATEGORY-SPECIFIC METHODS (most common)
    # =========================================================================
    
    @classmethod
    def run(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log pipeline lifecycle events."""
        cls._log(level, Category.RUN.value, message, **kwargs)
    
    @classmethod
    def cfg(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log configuration events."""
        cls._log(level, Category.CFG.value, message, **kwargs)
    
    @classmethod
    def data(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log data loading and validation events."""
        cls._log(level, Category.DATA.value, message, **kwargs)
    
    @classmethod
    def feat(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log feature engineering events."""
        cls._log(level, Category.FEAT.value, message, **kwargs)
    
    @classmethod
    def model(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log model training/caching events."""
        cls._log(level, Category.MODEL.value, message, **kwargs)
    
    @classmethod
    def score(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log scoring and detection events."""
        cls._log(level, Category.SCORE.value, message, **kwargs)
    
    @classmethod
    def fuse(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log fusion and episode events."""
        cls._log(level, Category.FUSE.value, message, **kwargs)
    
    @classmethod
    def regime(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log regime detection events."""
        cls._log(level, Category.REGIME.value, message, **kwargs)
    
    @classmethod
    def episode(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log episode detection events."""
        cls._log(level, Category.EPISODE.value, message, **kwargs)
    
    @classmethod
    def output(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log output write events."""
        cls._log(level, Category.OUTPUT.value, message, **kwargs)
    
    @classmethod
    def perf(cls, step_name: str, duration_ms: float, **kwargs) -> None:
        """Log performance metrics.
        
        Args:
            step_name: Name of the step (e.g., "features.build", "train.fit")
            duration_ms: Duration in milliseconds
            **kwargs: Additional context (row_count, col_count, etc.)
        """
        # Format duration nicely
        if duration_ms >= 1000:
            dur_str = f"{duration_ms/1000:.2f}s"
        else:
            dur_str = f"{duration_ms:.1f}ms"
        
        message = f"{step_name:<25} {dur_str}"
        if "row_count" in kwargs:
            message += f" ({kwargs['row_count']:,} rows)"
        
        cls._log("INFO", "PERF", message, step_name=step_name, duration_ms=duration_ms, **kwargs)
    
    @classmethod
    def health(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log health tracking events."""
        cls._log(level, Category.HEALTH.value, message, **kwargs)
    
    @classmethod
    def rul(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log RUL estimation events."""
        cls._log(level, Category.RUL.value, message, **kwargs)
    
    @classmethod
    def forecast(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log forecasting events."""
        cls._log(level, Category.FORECAST.value, message, **kwargs)
    
    @classmethod
    def sql(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log SQL connection/query events."""
        cls._log(level, Category.SQL.value, message, **kwargs)
    
    @classmethod
    def threshold(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log threshold calculation events."""
        cls._log(level, Category.THRESHOLD.value, message, **kwargs)
    
    @classmethod
    def tune(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log auto-tuning events."""
        cls._log(level, Category.TUNE.value, message, **kwargs)
    
    @classmethod
    def adaptive(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log adaptive parameter events."""
        cls._log(level, Category.ADAPTIVE.value, message, **kwargs)
    
    @classmethod
    def coldstart(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log coldstart events."""
        cls._log(level, Category.COLDSTART.value, message, **kwargs)
    
    @classmethod
    def baseline(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log baseline events."""
        cls._log(level, Category.BASELINE.value, message, **kwargs)
    
    # =========================================================================
    # DETECTOR LOGGING
    # =========================================================================
    
    @classmethod
    def det(cls, detector: str, message: str, level: str = "INFO", **kwargs) -> None:
        """Log detector-specific events.
        
        Args:
            detector: Detector name (OMR, AR1, PCA, MHAL, GMM, IFOR, CORR, CUSUM)
            message: Log message
            level: Log level (default: INFO)
            **kwargs: Additional context
        """
        detector_upper = detector.upper()
        if detector_upper not in DETECTOR_NAMES:
            # Allow unknown detector names but standardize format
            pass
        
        # Use detector name as category: [OMR], [AR1], etc.
        cls._log(level, detector_upper, message, detector_name=detector_upper, **kwargs)
    
    # Convenience methods for each detector
    @classmethod
    def omr(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log OMR detector events."""
        cls.det("OMR", message, level, **kwargs)
    
    @classmethod
    def ar1(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log AR1 detector events."""
        cls.det("AR1", message, level, **kwargs)
    
    @classmethod
    def pca(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log PCA detector events."""
        cls.det("PCA", message, level, **kwargs)
    
    @classmethod
    def mhal(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log Mahalanobis detector events."""
        cls.det("MHAL", message, level, **kwargs)
    
    @classmethod
    def gmm(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log GMM detector events."""
        cls.det("GMM", message, level, **kwargs)
    
    @classmethod
    def ifor(cls, message: str, level: str = "INFO", **kwargs) -> None:
        """Log Isolation Forest detector events."""
        cls.det("IFOR", message, level, **kwargs)
    
    # =========================================================================
    # SEVERITY SHORTCUTS
    # =========================================================================
    
    @classmethod
    def debug(cls, category: str, message: str, **kwargs) -> None:
        """Log debug message."""
        cls._log("DEBUG", category, message, **kwargs)
    
    @classmethod
    def info(cls, category: str, message: str, **kwargs) -> None:
        """Log info message."""
        cls._log("INFO", category, message, **kwargs)
    
    @classmethod
    def warn(cls, category: str, message: str, **kwargs) -> None:
        """Log warning message."""
        cls._log("WARNING", category, message, **kwargs)
    
    @classmethod
    def error(cls, category: str, message: str, exc: Optional[Exception] = None, **kwargs) -> None:
        """Log error message, optionally including exception details."""
        if exc is not None:
            kwargs['error_type'] = type(exc).__name__
            kwargs['error_detail'] = str(exc)
            message = f"{message}: {exc}"
        cls._log("ERROR", category, message, **kwargs)
    
    @classmethod
    def critical(cls, category: str, message: str, exc: Optional[Exception] = None, **kwargs) -> None:
        """Log critical message."""
        if exc is not None:
            kwargs['error_type'] = type(exc).__name__
            kwargs['error_detail'] = str(exc)
            message = f"{message}: {exc}"
        cls._log("CRITICAL", category, message, **kwargs)
    
    # =========================================================================
    # PROGRESS INDICATORS
    # =========================================================================
    
    @classmethod
    def start(cls, category: str, message: str, **kwargs) -> None:
        """Log start of an operation (uses [..] prefix)."""
        formatted_msg = f"[..] {message} ..."
        Console.info(formatted_msg, skip_sql=True, **kwargs)
    
    @classmethod
    def done(cls, category: str, message: str, duration_s: Optional[float] = None, **kwargs) -> None:
        """Log completion of an operation (uses [OK] prefix)."""
        if duration_s is not None:
            formatted_msg = f"[OK] {message} done in {duration_s:.2f}s"
        else:
            formatted_msg = f"[OK] {message}"
        Console.info(formatted_msg, skip_sql=True, **kwargs)
    
    # =========================================================================
    # TIMING CONTEXT MANAGER
    # =========================================================================
    
    class Timer:
        """Context manager for timing operations."""
        
        def __init__(self, category: str, step_name: str, **kwargs):
            self.category = category
            self.step_name = step_name
            self.kwargs = kwargs
            self._start = None
        
        def __enter__(self):
            self._start = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = (time.perf_counter() - self._start) * 1000
            ACMLog.perf(self.step_name, duration_ms, **self.kwargs)
            return False
    
    @classmethod
    def timer(cls, category: str, step_name: str, **kwargs) -> "ACMLog.Timer":
        """Create a timing context manager.
        
        Usage:
            with ACMLog.timer("FEAT", "features.build", row_count=1000):
                # ... operation ...
        """
        return cls.Timer(category, step_name, **kwargs)


# =========================================================================
# MIGRATION HELPERS
# =========================================================================

def migrate_console_to_acmlog(old_format: str) -> str:
    """Helper to identify what ACMLog method to use for existing Console.xxx calls.
    
    This is a documentation helper, not runtime code.
    
    Examples:
        Console.info("[CFG] Loaded config")     -> ACMLog.cfg("Loaded config")
        Console.warn("[DATA] Low variance")     -> ACMLog.warn("DATA", "Low variance")
        Console.info("[AR1] Fitted model")      -> ACMLog.ar1("Fitted model")
        Console.info("[OMR] Diagnostics")       -> ACMLog.omr("Diagnostics")
        Console.info("[TIMER] step 0.5s")       -> ACMLog.perf("step", 500)
    """
    return """
    MIGRATION GUIDE
    ===============
    
    Old Pattern                           -> New Pattern
    ------------                             -----------
    Console.info("[CFG] msg")             -> ACMLog.cfg("msg")
    Console.info("[DATA] msg")            -> ACMLog.data("msg")
    Console.info("[FEAT] msg")            -> ACMLog.feat("msg")
    Console.info("[MODEL] msg")           -> ACMLog.model("msg")
    Console.info("[RUN] msg")             -> ACMLog.run("msg")
    Console.info("[OUTPUT] msg")          -> ACMLog.output("msg")
    Console.info("[THRESHOLD] msg")       -> ACMLog.threshold("msg")
    Console.info("[REGIME] msg")          -> ACMLog.regime("msg")
    Console.info("[FUSE] msg")            -> ACMLog.fuse("msg")
    Console.info("[COLDSTART] msg")       -> ACMLog.coldstart("msg")
    Console.info("[BASELINE] msg")        -> ACMLog.baseline("msg")
    Console.info("[HEALTH] msg")          -> ACMLog.health("msg")
    
    # Warnings
    Console.warn("[DATA] msg")            -> ACMLog.warn("DATA", "msg")
    Console.warn("[MODEL] msg")           -> ACMLog.warn("MODEL", "msg")
    
    # Errors
    Console.error("[SQL] msg")            -> ACMLog.error("SQL", "msg")
    
    # Detectors
    Console.info("[OMR] msg")             -> ACMLog.omr("msg")
    Console.info("[AR1] msg")             -> ACMLog.ar1("msg")
    Console.info("[PCA] msg")             -> ACMLog.pca("msg")
    Console.info("[MHAL] msg")            -> ACMLog.mhal("msg")
    Console.info("[GMM] msg")             -> ACMLog.gmm("msg")
    Console.warn("[AR1] msg")             -> ACMLog.ar1("msg", level="WARNING")
    
    # Timing
    Console.info("[TIMER] step 0.5s")     -> ACMLog.perf("step", 500)
    
    # Progress
    Console.info("[..] Loading...")       -> ACMLog.start("DATA", "Loading")
    Console.info("[OK] Done in 0.5s")     -> ACMLog.done("DATA", "Loading", 0.5)
    """


# Export main class
__all__ = ["ACMLog", "Category", "DETECTOR_NAMES", "ACMLogRecord"]
