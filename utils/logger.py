# utils/logger.py
"""Enhanced logging system for ACM V8.

Provides a unified logging interface with support for:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Structured logging (text and JSON formats)
- Configurable output (stdout, stderr, file)
- Context metadata
- ASCII-only mode for compatibility
- Environment-based configuration
- Heartbeat progress indicator
"""
from __future__ import annotations
import inspect
import json
import sys
import time
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, Optional, TextIO, Literal, List
from enum import IntEnum


class LogLevel(IntEnum):
    """Log levels in ascending order of severity."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogFormat(IntEnum):
    """Output format for log messages."""
    TEXT = 1
    JSON = 2


class Logger:
    """Enhanced logger with configurable levels, formats, and outputs."""
    
    def __init__(self):
        self._level = self._get_level_from_env()
        self._format = self._get_format_from_env()
        self._file: Optional[TextIO] = None
        self._file_path: Optional[Path] = None
        self._ascii_only = self._get_ascii_only_from_env()
        self._setup_file_output()
        self._module_levels: Dict[str, LogLevel] = {}
        self._sinks: List[Callable[[Dict[str, Any]], None]] = []
        self._sink_lock = threading.Lock()
    
    def _get_level_from_env(self) -> LogLevel:
        """Get log level from LOG_LEVEL environment variable."""
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level_map = {
            "DEBUG": LogLevel.DEBUG,
            "INFO": LogLevel.INFO,
            "WARNING": LogLevel.WARNING,
            "WARN": LogLevel.WARNING,
            "ERROR": LogLevel.ERROR,
            "CRITICAL": LogLevel.CRITICAL,
        }
        return level_map.get(level_str, LogLevel.INFO)
    
    def _get_format_from_env(self) -> LogFormat:
        """Get log format from LOG_FORMAT environment variable."""
        format_str = os.getenv("LOG_FORMAT", "text").lower()
        if format_str == "json":
            return LogFormat.JSON
        return LogFormat.TEXT
    
    def _get_ascii_only_from_env(self) -> bool:
        """Get ASCII-only mode from LOG_ASCII_ONLY environment variable."""
        ascii_str = os.getenv("LOG_ASCII_ONLY", "false").lower()
        return ascii_str in ("true", "1", "yes", "on")
    
    def _setup_file_output(self):
        """Setup file output if LOG_FILE is specified."""
        file_path = os.getenv("LOG_FILE")
        if file_path:
            try:
                self._file_path = Path(file_path)
                self._file_path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self._file_path.open("a", encoding="utf-8")
            except Exception as e:
                # Fallback: warn to stderr but continue
                print(f"[WARNING] Could not open log file {file_path}: {e}", file=sys.stderr)
                self._file = None
    
    def set_level(self, level: str | LogLevel) -> None:
        """Set the minimum log level."""
        if isinstance(level, str):
            level_map = {
                "DEBUG": LogLevel.DEBUG,
                "INFO": LogLevel.INFO,
                "WARNING": LogLevel.WARNING,
                "WARN": LogLevel.WARNING,
                "ERROR": LogLevel.ERROR,
                "CRITICAL": LogLevel.CRITICAL,
            }
            self._level = level_map.get(level.upper(), LogLevel.INFO)
        else:
            self._level = level
    
    def set_format(self, fmt: str | LogFormat) -> None:
        """Set the output format."""
        if isinstance(fmt, str):
            fmt = LogFormat.JSON if fmt.lower() == "json" else LogFormat.TEXT
        self._format = fmt
    
    def set_output(self, file_path: Optional[Path]) -> None:
        """Set the file output path."""
        if self._file:
            self._file.close()
            self._file = None
        
        if file_path:
            try:
                self._file_path = file_path
                self._file_path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self._file_path.open("a", encoding="utf-8")
            except Exception as e:
                print(f"[WARNING] Could not open log file {file_path}: {e}", file=sys.stderr)
    
    @property
    def ascii_only(self) -> bool:
        return self._ascii_only

    def clear_module_levels(self) -> None:
        """Remove all module-specific level overrides."""
        self._module_levels.clear()

    def set_module_level(self, module: str, level: str | LogLevel) -> None:
        """Set log level for a specific module."""
        if isinstance(level, str):
            level_map = {
                "DEBUG": LogLevel.DEBUG,
                "INFO": LogLevel.INFO,
                "WARNING": LogLevel.WARNING,
                "WARN": LogLevel.WARNING,
                "ERROR": LogLevel.ERROR,
                "CRITICAL": LogLevel.CRITICAL,
            }
            level = level_map.get(level.upper(), LogLevel.INFO)
        self._module_levels[module] = level

    def add_sink(self, sink: Callable[[Dict[str, Any]], None]) -> None:
        """Register an additional sink for structured log records."""
        with self._sink_lock:
            self._sinks.append(sink)

    def remove_sink(self, sink: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a previously registered sink."""
        with self._sink_lock:
            self._sinks = [s for s in self._sinks if s is not sink]

    def _format_message(self, level: LogLevel, msg: str, context: Dict[str, Any], ts: datetime) -> str:
        """Format a log message according to the current format."""
        if self._format == LogFormat.JSON:
            record = {
                "timestamp": ts.isoformat(),
                "level": level.name,
                "message": msg,
            }
            if context:
                record["context"] = context
            return json.dumps(record, ensure_ascii=True)
        else:
            # Text format
            timestamp = ts.strftime("%Y-%m-%d %H:%M:%S")
            context_str = ""
            if context:
                context_str = " " + " ".join(f"{k}={v}" for k, v in context.items())
            return f"[{timestamp}] [{level.name}] {msg}{context_str}"

    def _log(self, level: LogLevel, msg: str, **context: Any) -> None:
        """Internal logging method."""
        ctx = dict(context) if context else {}
        module_name = ctx.get("module")
        if module_name is None:
            module_name = self._infer_module_name()
            ctx["module"] = module_name

        effective_level = self._module_levels.get(module_name, self._level)
        if level < effective_level:
            return

        timestamp = datetime.utcnow()
        formatted = self._format_message(level, msg, ctx, timestamp)
        
        # Determine output stream
        stream = sys.stderr if level >= LogLevel.ERROR else sys.stdout
        
        # Write to console
        print(formatted, file=stream)
        
        # Write to file if configured
        if self._file:
            try:
                self._file.write(formatted + "\n")
                self._file.flush()
            except Exception:
                pass  # Silent failure to avoid recursion

        sinks = None
        if self._sinks:
            with self._sink_lock:
                sinks = list(self._sinks)
        if sinks:
            record = {
                "timestamp": timestamp.isoformat(),
                "level": level.name,
                "message": msg,
                "module": module_name,
                "context": ctx,
            }
            for sink in sinks:
                try:
                    sink(record)
                except Exception:
                    pass

    def _infer_module_name(self) -> str:
        """Best-effort inference of the caller's module name."""
        frame = inspect.currentframe()
        if not frame:
            return "__main__"
        try:
            caller = frame
            # Skip frames: _log -> Console.<method> -> caller
            for _ in range(3):
                if caller.f_back:
                    caller = caller.f_back
                else:
                    break
            return caller.f_globals.get("__name__", "__main__")
        finally:
            del frame
    
    def debug(self, msg: str, **context: Any) -> None:
        """Log a debug message."""
        self._log(LogLevel.DEBUG, msg, **context)
    
    def info(self, msg: str, **context: Any) -> None:
        """Log an info message."""
        self._log(LogLevel.INFO, msg, **context)
    
    def warn(self, msg: str, **context: Any) -> None:
        """Log a warning message."""
        self._log(LogLevel.WARNING, msg, **context)
    
    def warning(self, msg: str, **context: Any) -> None:
        """Alias for warn()."""
        self.warn(msg, **context)
    
    def error(self, msg: str, **context: Any) -> None:
        """Log an error message."""
        self._log(LogLevel.ERROR, msg, **context)
    
    def critical(self, msg: str, **context: Any) -> None:
        """Log a critical message."""
        self._log(LogLevel.CRITICAL, msg, **context)
    
    def ok(self, msg: str, **context: Any) -> None:
        """Log a success message (info level)."""
        self.info(msg, **context)
    
    @property
    def ascii_only(self) -> bool:
        """Return whether ASCII-only mode is enabled."""
        return self._ascii_only
    
    def __del__(self):
        """Close file handle on destruction."""
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass


# Global logger instance
_logger = Logger()


class Console:
    """Unified console logger interface - wraps the global Logger instance.
    
    This maintains backward compatibility with existing code while providing
    access to enhanced logging features.
    """
    
    @staticmethod
    def debug(msg: str, **context: Any) -> None:
        """Log a debug message."""
        _logger.debug(msg, **context)
    
    @staticmethod
    def info(msg: str, **context: Any) -> None:
        """Log an info message."""
        _logger.info(msg, **context)
    
    @staticmethod
    def ok(msg: str, **context: Any) -> None:
        """Log a success message."""
        _logger.ok(msg, **context)
    
    @staticmethod
    def warn(msg: str, **context: Any) -> None:
        """Log a warning message."""
        _logger.warn(msg, **context)
    
    @staticmethod
    def warning(msg: str, **context: Any) -> None:
        """Alias for warn()."""
        _logger.warning(msg, **context)
    
    @staticmethod
    def error(msg: str, **context: Any) -> None:
        """Log an error message."""
        _logger.error(msg, **context)
    
    @staticmethod
    def critical(msg: str, **context: Any) -> None:
        """Log a critical message."""
        _logger.critical(msg, **context)
    
    @staticmethod
    def set_level(level: str | LogLevel) -> None:
        """Set the minimum log level."""
        _logger.set_level(level)
    
    @staticmethod
    def set_format(fmt: str | LogFormat) -> None:
        """Set the output format."""
        _logger.set_format(fmt)
    
    @staticmethod
    def set_output(file_path: Optional[Path]) -> None:
        """Set the file output path."""
        _logger.set_output(file_path)
    
    @staticmethod
    def ascii_only() -> bool:
        """Return whether ASCII-only mode is enabled."""
        return _logger.ascii_only
    
    @staticmethod
    def set_module_level(module: str, level: str | LogLevel) -> None:
        """Set log level for a specific module."""
        _logger.set_module_level(module, level)
    
    @staticmethod
    def clear_module_levels() -> None:
        """Clear all module-specific level overrides."""
        _logger.clear_module_levels()
    
    @staticmethod
    def add_sink(sink: Callable[[Dict[str, Any]], None]) -> None:
        """Register a structured log sink."""
        _logger.add_sink(sink)
    
    @staticmethod
    def remove_sink(sink: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a structured log sink."""
        _logger.remove_sink(sink)


class Heartbeat:
    """Progress indicator with periodic updates.
    
    Shows real-time progress during long-running operations with:
    - Animated spinner (ASCII-only or Unicode based on LOG_ASCII_ONLY)
    - Elapsed time
    - ETA hint
    - Next step hint
    
    Usage:
        hb = Heartbeat("Loading data", next_hint="parsing", eta_hint=30).start()
        # ... long operation ...
        hb.stop()
    
    Environment variables:
        - ACM_HEARTBEAT: Enable/disable heartbeat (default: true)
        - LOG_ASCII_ONLY: Use ASCII spinner instead of Unicode (default: false)
    """
    
    # Unicode braille spinner (smooth animation)
    SPINNER_UNICODE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    # ASCII spinner (compatible with all terminals)
    SPINNER_ASCII = ["-", "\\", "|", "/"]
    
    def __init__(
        self,
        label: str,
        next_hint: Optional[str] = None,
        eta_hint: Optional[float] = None,
        interval: float = 2.0
    ):
        """Initialize heartbeat.
        
        Args:
            label: Description of the operation
            next_hint: Description of the next step
            eta_hint: Estimated time in seconds
            interval: Update interval in seconds
        """
        self.label = label
        self.next_hint = next_hint
        self.eta_hint = eta_hint
        self.interval = interval
        self._stop = threading.Event()
        self._t0 = time.perf_counter()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._started = False
        
        # Check if heartbeat is enabled
        self._enabled = os.getenv("ACM_HEARTBEAT", "true").lower() not in ("false", "0", "no", "off")
        
        # Select spinner based on ASCII-only mode
        self._spinner = self.SPINNER_ASCII if _logger.ascii_only else self.SPINNER_UNICODE
    
    def start(self) -> "Heartbeat":
        """Start the heartbeat."""
        if not self._enabled:
            return self
        
        Console.info(f"[..] {self.label} ...")
        self._started = True
        self._thr.start()
        return self
    
    def stop(self) -> None:
        """Stop the heartbeat and log completion."""
        if not self._enabled:
            return
        
        self._stop.set()
        if self._started and self._thr.is_alive():
            self._thr.join(timeout=0.1)
        
        took = time.perf_counter() - self._t0
        Console.ok(f"[OK] {self.label} done in {took:.2f}s")
    
    def _run(self) -> None:
        """Background thread that updates progress."""
        i = 0
        while not self._stop.wait(self.interval):
            spent = time.perf_counter() - self._t0
            eta = f" | ~{max(0.0, self.eta_hint - spent):0.0f}s left" if self.eta_hint else ""
            nxt = f" | next: {self.next_hint}" if self.next_hint else ""
            
            spinner_char = self._spinner[i % len(self._spinner)]
            Console.info(f"[..] {spinner_char} {self.label}{eta}{nxt}")
            i += 1


def jsonl_logger(path: Path) -> Callable[[Dict[str, Any]], None]:
    """
    Append JSON lines to `path` with a simple writer:
      {"event":"START","block":"FEATURES","t":...}
      {"event":"END","block":"FEATURES","t":...,"dt_s":...}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a", encoding="utf-8")
    def write(obj: Dict[str, Any]) -> None:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()
    return write
