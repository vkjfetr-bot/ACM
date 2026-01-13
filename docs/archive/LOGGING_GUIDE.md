# ACM V8 - Enhanced Logging Guide

**Last Updated:** 2025-11-15  
**Version:** 2.0

---

## Overview

ACM V8 uses an enhanced unified logging system that provides:

- **Structured logging** with multiple log levels
- **Configurable output** (console, file, JSON format)
- **Context metadata** support for detailed diagnostics
- **Heartbeat progress indicator** for long-running operations
- **Timer integration** for performance monitoring
- **Environment-based configuration** for easy deployment

---

## Quick Start

### Basic Usage

```python
from utils.logger import Console

# Simple messages
Console.info("Processing started")
Console.warn("Low memory detected")
Console.error("Connection failed")

# With context metadata
Console.info("Data loaded", rows=1000, columns=42)
Console.error("Query failed", table="ACM_Scores", code=500)
```

### CLI Overrides

```bash
python -m core.acm_main \
  --equip FD_FAN \
  --log-level DEBUG \
  --log-format json \
  --log-file artifacts/logs/fd_fan.log \
  --log-module-level core.sql_batch_runner=WARNING
```

Use these switches to override environment/config values without editing YAML.

### Configuration Block

```yaml
logging:
  level: INFO
  format: text
  file: artifacts/logs/acm.log
  module_levels:
    core.acm_main: INFO
    scripts.sql_batch_runner: WARNING
  enable_sql_sink: true
```

Place this block in SQL/YAML config to set defaults for every run.

### Progress Indicator

```python
from utils.logger import Heartbeat

# Long-running operation with progress updates
hb = Heartbeat("Loading data", next_hint="parsing", eta_hint=30).start()
# ... do work ...
hb.stop()
```

### Performance Timing

```python
from utils.timer import Timer

T = Timer()

# Custom log entry
T.log("batch_processed", count=1000, status="ok")
```

---

## Log Levels

ACM V8 supports standard log levels in ascending severity:

| Level | Method | Use For | Output Stream |
|-------|--------|---------|---------------|
| DEBUG | `Console.debug()` | Detailed diagnostic info | stdout |
| INFO | `Console.info()` | General informational messages | stdout |
| WARNING | `Console.warn()` | Warning messages, non-critical issues | stdout |
| ERROR | `Console.error()` | Error conditions that need attention | stderr |
| CRITICAL | `Console.critical()` | Critical failures requiring immediate action | stderr |

### Additional Methods

- `Console.ok()` - Alias for `info()`, used for success messages
- `Console.warning()` - Alias for `warn()`

---

## Configuration

### Environment Variables

All logging behavior can be controlled via environment variables:

#### `LOG_LEVEL`

Set the minimum log level to display.

```bash
export LOG_LEVEL=INFO        # Default - show INFO and above
export LOG_LEVEL=WARNING     # Only warnings and errors
export LOG_LEVEL=DEBUG       # Show everything (verbose)
```

**Valid values:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

#### `LOG_FORMAT`

Control output format.

```bash
export LOG_FORMAT=text       # Default - human-readable text
export LOG_FORMAT=json       # Structured JSON for automation
```

**Text format example:**
```
[2025-11-15 17:30:45] [INFO] Processing started rows=1000
```

**JSON format example:**
```json
{"timestamp": "2025-11-15T17:30:45.123456", "level": "INFO", "message": "Processing started", "context": {"rows": 1000}}
```

#### `LOG_FILE`

Write logs to a file in addition to console.

```bash
export LOG_FILE=/var/log/acm/acm.log
```

Logs will be appended to this file. The directory will be created if it doesn't exist.

#### `LOG_ASCII_ONLY`

Force ASCII-only output (no Unicode characters).

```bash
export LOG_ASCII_ONLY=true   # Use ASCII spinner in Heartbeat
export LOG_ASCII_ONLY=false  # Default - Unicode spinner OK
```

This affects the Heartbeat spinner:
- **ASCII mode:** `-`, `\`, `|`, `/`
- **Unicode mode:** `⠋`, `⠙`, `⠹`, `⠸`, `⠼`, `⠴`, `⠦`, `⠧`, `⠇`, `⠏` (braille)

#### `ACM_HEARTBEAT`

Enable or disable heartbeat progress indicators.

```bash
export ACM_HEARTBEAT=true    # Default - show progress
export ACM_HEARTBEAT=false   # Suppress progress indicators
```

#### `ACM_TIMINGS`

Enable or disable performance timing output.

```bash
export ACM_TIMINGS=1         # Default - show timings
export ACM_TIMINGS=0         # Suppress timing output
```

---

## Programmatic Configuration

You can also configure the logger programmatically:

```python
from utils.logger import Console

# Set log level
Console.set_level("DEBUG")
Console.set_level("WARNING")

# Set output format
Console.set_format("json")
Console.set_format("text")

# Set file output
from pathlib import Path
Console.set_output(Path("/var/log/acm/acm.log"))
Console.set_output(None)  # Disable file output
```

---

## Heartbeat Progress Indicator

The `Heartbeat` class provides real-time progress updates during long-running operations.

### Basic Usage

```python
from utils.logger import Heartbeat

hb = Heartbeat("Loading CSV files").start()
# ... do work ...
hb.stop()
```

**Output:**
```
[..] Loading CSV files ...
[..] - Loading CSV files
[..] \ Loading CSV files
[..] | Loading CSV files
[OK] Loading CSV files done in 12.34s
```

### With Hints

```python
hb = Heartbeat(
    "Processing batch",
    next_hint="validation",  # What comes after this step
    eta_hint=45.0            # Expected duration in seconds
).start()
# ... do work ...
hb.stop()
```

**Output:**
```
[..] Processing batch ...
[..] - Processing batch | ~40s left | next: validation
[..] \ Processing batch | ~35s left | next: validation
[OK] Processing batch done in 48.23s
```

### Environment Control

```bash
# Disable heartbeat entirely
export ACM_HEARTBEAT=false

# Use ASCII-only spinner
export LOG_ASCII_ONLY=true
```

---

## Timer Integration

The `Timer` utility tracks performance of pipeline stages.

### Basic Usage

```python
from utils.timer import Timer

T = Timer()

with T.section("data_loading"):
    # Load data
    pass

with T.section("feature_engineering"):
    # Engineer features
    pass

# Timer automatically logs a summary via Console on exit
```

**Output:**
```
[TIMER] data_loading           5.234s
[TIMER] feature_engineering    12.456s
[TIMER] -- Summary ------------------------------
[TIMER] feature_engineering    12.456s ( 70.4%)
[TIMER] data_loading            5.234s ( 29.6%)
[TIMER] total_run              17.690s
```

### Custom Log Entries

```python
T.log("checkpoint", phase="train", samples=5000)
```

**Output:**
```
[TIMER] checkpoint           phase=train samples=5000
```

### Decorator Usage

```python
@T.wrap("my_function")
def expensive_operation():
    # ... do work ...
    pass

expensive_operation()  # Automatically timed
```

### JSON Mode

When `LOG_FORMAT=json`, Timer outputs structured data:

```json
{"timer": "data_loading", "duration_s": 5.234, "event": "section_end"}
{"event": "timer_summary", "total_duration_s": 17.690, "sections": [...]}
```

## SQL Run Logs (ACM_RunLogs)

When ACM runs in SQL-only or dual mode it streams every Console entry into
`dbo.ACM_RunLogs` with both the raw message and parsed fields for easier filtering
and dashboarding.

| Column | Description |
| ------ | ----------- |
| `RunID` | Current ACM run identifier |
| `EquipID` | Numeric equipment ID |
| `LoggedAt` | UTC timestamp emitted by the logger |
| `Level` | Log level (INFO/WARNING/ERROR/...) |
| `Module` | Originating module (auto-inferred when not provided) |
| `EventType` | Tag captured from leading brackets (e.g., `[TIMER]`, `[FEAT]`) or `context.event_type` |
| `Stage` | Pipeline stage/component (`context.stage` / `pipeline_stage`) |
| `StepName` | Operation name (from context or derived from message body) |
| `DurationMs` | Duration in milliseconds (context or parsed from message suffix like `0.12s`) |
| `RowCount` / `ColCount` | Shape hints parsed from context or message (e.g., `n_rows=300`, `n_cols=9`) |
| `WindowSize` | Rolling window size when provided (`window=16`) |
| `BatchStart` / `BatchEnd` | Batch window bounds (datetime) |
| `BaselineStart` / `BaselineEnd` | Baseline window bounds (datetime) |
| `DataQualityMetric` / `DataQualityValue` | Data-quality metric name/value when provided |
| `LeakageFlag` | Boolean leakage indicator when provided |
| `ParamsJson` | Flattened key/value pairs extracted from the message and context extras |
| `Message` | Text body (truncated to 4000 chars) |
| `Context` | JSON payload with the original context |

SQL sink is always enabled in SQL mode. Example:

```bash
python -m core.acm_main --equip FD_FAN
```

---

## Best Practices

### 1. Use Appropriate Log Levels

```python
# ✓ Good
Console.debug("Cache hit", key="sensor_123")
Console.info("Processing batch 5 of 10")
Console.warn("Memory usage high", percent=90)
Console.error("Failed to connect to database", host="localhost")
Console.critical("System shutdown required, disk full")

# ✗ Avoid
Console.info("Cache hit")  # Too detailed for INFO - use DEBUG
Console.error("Processing batch 5 of 10")  # Not an error - use INFO
```

### 2. Add Context Metadata

```python
# ✓ Good - includes useful context
Console.info("Model trained", 
             model="PCA", 
             samples=1000, 
             components=5, 
             accuracy=0.95)

# ✗ Avoid - missing context
Console.info("Model trained")
```

### 3. Use Consistent Prefixes

```python
# ✓ Good - module prefix for filtering
Console.info("[OMR] Model fitted", features=10)
Console.info("[DRIFT] Threshold exceeded", value=3.2)

# ✗ Avoid - inconsistent or missing prefixes
Console.info("OMR model fitted")
Console.info("drift detected")
```

### 4. Use Heartbeat for Long Operations

```python
# ✓ Good - user knows what's happening
hb = Heartbeat("Loading 100GB dataset", eta_hint=120).start()
data = load_large_file()
hb.stop()

# ✗ Avoid - silent operation confuses users
data = load_large_file()  # No feedback for 2 minutes
```

### 5. Time Performance-Critical Sections

```python
# ✓ Good - track performance
with T.section("feature_engineering"):
    features = engineer_features(data)

# ✗ Avoid - no visibility into bottlenecks
features = engineer_features(data)
```

---

## Production Deployment

### Recommended Configuration

For production deployments, use these settings:

```bash
# /etc/acm/logging.env
export LOG_LEVEL=INFO
export LOG_FORMAT=json
export LOG_FILE=/var/log/acm/acm-$(date +%Y%m%d).log
export LOG_ASCII_ONLY=true
export ACM_HEARTBEAT=false
export ACM_TIMINGS=1
```

### Log Rotation

Use `logrotate` to manage log files:

```bash
# /etc/logrotate.d/acm
/var/log/acm/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 acm acm
    sharedscripts
}
```

### Monitoring Integration

JSON format integrates easily with log aggregation tools:

```bash
# Ship to Elasticsearch
filebeat -c filebeat-acm.yml

# Ship to Splunk
./splunk add monitor /var/log/acm/

# Ship to CloudWatch
aws logs put-log-events --log-group acm --log-stream acm-$(hostname)
```

---

## Troubleshooting

### Issue: No log output

**Check:**
1. `LOG_LEVEL` is not too restrictive
2. Logger import is successful
3. Verify console output redirection

```bash
# Verbose mode
export LOG_LEVEL=DEBUG
python -m core.acm_main --equip FD_FAN
```

### Issue: Unicode characters broken

**Solution:** Enable ASCII-only mode

```bash
export LOG_ASCII_ONLY=true
```

### Issue: Log file not created

**Check:**
1. Directory permissions
2. Disk space
3. Path is absolute

```bash
# Create directory manually
mkdir -p /var/log/acm
chmod 755 /var/log/acm

# Test write permission
touch /var/log/acm/test.log
```

### Issue: Too much output

**Solution:** Increase log level or disable heartbeat

```bash
export LOG_LEVEL=WARNING
export ACM_HEARTBEAT=false
export ACM_TIMINGS=0
```

---

## Migration from Old Logging

### Old Code (print statements)

```python
# Old
print("[INFO] Processing data")
print(f"[OMR] Fitted model: {n_samples} samples")
print(f"WARNING: Low memory")
```

### New Code (Console logger)

```python
# New
Console.info("[INFO] Processing data")
Console.info("[OMR] Fitted model", samples=n_samples)
Console.warn("Low memory")
```

### Old Code (inline Console class)

```python
# Old - fallback definition
try:
    from utils.logger import Console
except ImportError:
    class Console:
        @staticmethod
        def info(msg): print(f"[INFO] {msg}")
```

### New Code (no fallback needed)

```python
# New - fail fast if logger unavailable
from utils.logger import Console
# No fallback needed - logger is always available
```

---

## API Reference

### Console Class

#### Methods

- `Console.debug(msg: str, **context)` - Log debug message
- `Console.info(msg: str, **context)` - Log info message
- `Console.ok(msg: str, **context)` - Log success message (alias for info)
- `Console.warn(msg: str, **context)` - Log warning message
- `Console.warning(msg: str, **context)` - Alias for warn
- `Console.error(msg: str, **context)` - Log error message
- `Console.critical(msg: str, **context)` - Log critical message
- `Console.set_level(level: str | LogLevel)` - Set minimum log level
- `Console.set_format(fmt: str | LogFormat)` - Set output format
- `Console.set_output(file_path: Path | None)` - Set file output
- `Console.ascii_only() -> bool` - Check if ASCII-only mode is enabled

### Heartbeat Class

#### Constructor

```python
Heartbeat(
    label: str,                    # Operation description
    next_hint: str | None = None,  # Next step hint
    eta_hint: float | None = None, # Estimated time in seconds
    interval: float = 2.0          # Update interval in seconds
)
```

#### Methods

- `start() -> Heartbeat` - Start the heartbeat
- `stop()` - Stop the heartbeat and log completion

### Timer Class

#### Constructor

```python
Timer(enable: bool | None = None)  # None = read from ACM_TIMINGS env var
```

#### Methods

- `section(name: str)` - Context manager for timing a section
- `end(name: str) -> float` - Manually end a section and return duration
- `log(name: str, **kv)` - Log a custom timing entry
- `wrap(name: str)` - Decorator for timing a function

---

## Examples

### Complete Pipeline Example

```python
from utils.logger import Console, Heartbeat
from utils.timer import Timer

def run_pipeline(equip_name: str):
    """Run ACM pipeline with comprehensive logging."""
    T = Timer()
    
    Console.info("[PIPELINE] Starting", equipment=equip_name)
    
    # Stage 1: Load data
    with T.section("data_loading"):
        hb = Heartbeat("Loading historian data", eta_hint=30).start()
        data = load_data(equip_name)
        hb.stop()
        Console.info("[DATA] Loaded", rows=len(data), columns=len(data.columns))
    
    # Stage 2: Engineer features
    with T.section("feature_engineering"):
        Console.info("[FEATURES] Computing rolling statistics")
        features = engineer_features(data)
        Console.ok("[FEATURES] Complete", features=len(features.columns))
    
    # Stage 3: Train models
    with T.section("model_training"):
        Console.info("[MODELS] Training detectors")
        models = train_models(features)
        Console.ok("[MODELS] Trained", count=len(models))
    
    # Stage 4: Score
    with T.section("scoring"):
        hb = Heartbeat("Scoring data", next_hint="save results", eta_hint=15).start()
        scores = score_data(features, models)
        hb.stop()
        Console.info("[SCORES] Generated", rows=len(scores))
    
    Console.ok("[PIPELINE] Complete", equipment=equip_name, duration=T.totals)

if __name__ == "__main__":
    run_pipeline("FD_FAN")
```

### Error Handling Example

```python
from utils.logger import Console

def safe_operation():
    """Operation with comprehensive error handling."""
    try:
        Console.info("[OP] Starting operation", phase="init")
        
        # Critical section
        result = risky_function()
        
        Console.ok("[OP] Success", result=result)
        return result
        
    except ValueError as e:
        Console.error("[OP] Invalid input", error=str(e))
        raise
        
    except ConnectionError as e:
        Console.critical("[OP] Connection lost", host="localhost", error=str(e))
        raise
        
    except Exception as e:
        Console.error("[OP] Unexpected error", error=str(e), type=type(e).__name__)
        raise
```

---

## Future Enhancements

Planned features for future releases:

1. **Log Aggregation**: Direct integration with Elasticsearch, Splunk, CloudWatch
2. **Trace IDs**: Correlation IDs for distributed request tracking
3. **Alert Thresholds**: Automatic escalation based on error frequency
4. **Log Sampling**: Reduce volume in high-frequency scenarios
5. **Custom Handlers**: Plugin system for custom log destinations

---

**End of Logging Guide**
