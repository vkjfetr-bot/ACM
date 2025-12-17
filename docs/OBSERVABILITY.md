# ACM Observability Stack

This document describes the consolidated observability architecture for ACM v10.2.x+.

## Overview

ACM uses a modern observability stack built on open standards:

| Signal | Tool | Backend | Purpose |
|--------|------|---------|---------|
| **Traces** | OpenTelemetry SDK | Grafana Tempo | Distributed tracing, request flow |
| **Metrics** | OpenTelemetry SDK | Grafana Mimir | Performance metrics, counters |
| **Logs** | structlog | Grafana Loki + SQL | Structured JSON logs |
| **Profiling** | Grafana Pyroscope | Pyroscope | Continuous CPU/memory flamegraphs |

## Architecture

```
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
          │
          ▼
┌─────────────────────┐
│     Grafana         │
│  (visualization)    │
└─────────────────────┘
```

## Installation

### Base (Required)
```bash
pip install structlog>=24.0
```

### Full Telemetry (Optional)
```bash
# OpenTelemetry for traces + metrics
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

# Continuous profiling
pip install pyroscope-io

# Or install everything via extras
pip install -e ".[observability]"
```

### Development Profiling
```bash
pip install py-spy scalene memray
```

## Quick Start

### Basic Usage (Logs Only)
```python
from core.observability import init_observability, get_logger, acm_log

# Initialize at startup
init_observability(service_name="acm-batch")

# Get structured logger
log = get_logger()
log.info("batch_started", equipment="FD_FAN", rows=1500)

# Or use category-aware logger (backwards compatible with ACMLog)
acm_log.run("Pipeline started")
acm_log.data("Loaded 1500 rows", row_count=1500)
acm_log.perf("detector.fit", duration_ms=234.5)
```

### Full Telemetry
```python
from core.observability import (
    init_observability, 
    get_tracer, 
    get_meter, 
    get_logger,
    traced,
    timed,
    set_context,
)

# Initialize with OTLP endpoint
init_observability(
    service_name="acm-batch",
    otlp_endpoint="http://localhost:4318",  # Grafana Alloy
    pyroscope_endpoint="http://localhost:4040",  # Optional
)

# Set run context
set_context(run_id="abc-123", equip_id=42, batch_num=0, batch_total=5)

# Get instrumentation handles
tracer = get_tracer()
meter = get_meter()
log = get_logger()

# Create custom metrics
batch_counter = meter.create_counter("acm.custom.counter")

# Use decorator for tracing
@traced("process_batch")
@timed("batch.process")
def process_batch(df):
    log.info("processing", rows=len(df))
    # ... work ...
    batch_counter.add(1, {"status": "success"})
```

### Profiling with Pyroscope
```python
from core.observability import profile_section

# Profile specific code sections
with profile_section({"equipment": "FD_FAN", "detector": "omr"}):
    run_detector()
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_SERVICE_NAME` | `acm` | Service name for traces/metrics |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | none | OTLP collector endpoint |
| `OTEL_SDK_DISABLED` | `false` | Disable OpenTelemetry entirely |
| `OTEL_TRACES_SAMPLER_ARG` | `1.0` | Trace sampling ratio (0.1 = 10%) |
| `ACM_PYROSCOPE_ENDPOINT` | none | Pyroscope server endpoint |
| `ACM_LOG_FORMAT` | `json` | Log format: `json` or `console` |
| `ACM_LOG_LEVEL` | `INFO` | Minimum log level |

## Log Categories

The `acm_log` object provides category-aware logging:

| Category | Method | Purpose |
|----------|--------|---------|
| RUN | `acm_log.run()` | Pipeline lifecycle |
| CFG | `acm_log.cfg()` | Configuration |
| DATA | `acm_log.data()` | Data loading/validation |
| FEAT | `acm_log.feat()` | Feature engineering |
| MODEL | `acm_log.model()` | Model training/caching |
| SCORE | `acm_log.score()` | Scoring/detection |
| FUSE | `acm_log.fuse()` | Fusion/episodes |
| PERF | `acm_log.perf()` | Performance metrics |
| HEALTH | `acm_log.health()` | Health tracking |
| RUL | `acm_log.rul()` | RUL estimation |
| FORECAST | `acm_log.forecast()` | Forecasting |
| SQL | `acm_log.sql()` | SQL operations |

## Built-in Metrics

Use the helper functions for common ACM metrics:

```python
from core.observability import (
    record_batch_processed,
    record_detector_duration,
    record_health_score,
)

# Record batch completion
record_batch_processed(
    equipment="FD_FAN",
    duration_seconds=12.5,
    rows=1500,
    status="success"
)

# Record detector timing
record_detector_duration(
    detector="omr",
    equipment="FD_FAN", 
    duration_seconds=2.3
)

# Record health score
record_health_score(equipment="FD_FAN", score=87.5)
```

## Log Cleanup

Clean up old logs from SQL:

```python
from core.observability import cleanup_old_logs
from core.sql_client import SQLClient

sql = SQLClient.from_config()
deleted = cleanup_old_logs(sql, retention_days=30)
print(f"Deleted {deleted} old log records")
```

## Grafana Stack Setup (Docker)

### docker-compose.yml
```yaml
version: '3.8'
services:
  alloy:
    image: grafana/alloy:latest
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
    volumes:
      - ./config.alloy:/etc/alloy/config.alloy
    command: run /etc/alloy/config.alloy

  tempo:
    image: grafana/tempo:latest
    ports:
      - "3200:3200"

  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"

  pyroscope:
    image: grafana/pyroscope:latest
    ports:
      - "4040:4040"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
```

### config.alloy (Grafana Alloy)
```hcl
// Receive OTLP from Python apps
otelcol.receiver.otlp "default" {
  grpc { endpoint = "0.0.0.0:4317" }
  http { endpoint = "0.0.0.0:4318" }

  output {
    traces  = [otelcol.processor.batch.default.input]
    metrics = [otelcol.processor.batch.default.input]
  }
}

otelcol.processor.batch "default" {
  output {
    traces  = [otelcol.exporter.otlp.tempo.input]
    metrics = [otelcol.exporter.prometheus.mimir.input]
  }
}

otelcol.exporter.otlp "tempo" {
  client { endpoint = "tempo:4317" }
}

otelcol.exporter.prometheus "mimir" {
  forward_to = [prometheus.remote_write.default.receiver]
}

prometheus.remote_write "default" {
  endpoint {
    url = "http://mimir:9009/api/v1/push"
  }
}
```

## Migration from Old Logging

### Before (Multiple Files)
```python
# Old way - multiple imports
from utils.logger import Console
from utils.acm_logger import ACMLog
from core.sql_logger_v2 import BatchedSqlLogSink

Console.info("message")
ACMLog.run("message")
```

### After (Consolidated)
```python
# New way - single module
from core.observability import init_observability, get_logger, acm_log

init_observability(sql_client=sql, run_id=run_id)

log = get_logger()
log.info("message")

acm_log.run("message")  # Same API, uses structlog under the hood
```

## File Consolidation

| Old File | Status | Replacement |
|----------|--------|-------------|
| `utils/logger.py` | Keep (base) | Still used internally |
| `utils/acm_logger.py` | Deprecated | `core/observability.ACMLogger` |
| `core/sql_logger.py` | **Delete** | `core/sql_logger_v2.py` |
| `core/sql_logger_v2.py` | Keep | SQL sink for logs |
| `core/resource_monitor.py` | Keep | Complements Pyroscope |

## Performance Considerations

1. **Sampling**: In production, set `OTEL_TRACES_SAMPLER_ARG=0.1` (10% sampling)
2. **Batch Export**: Uses `BatchSpanProcessor` by default (non-blocking)
3. **Metrics Interval**: Exports every 60s by default
4. **Profiling Rate**: Pyroscope samples at 100Hz by default

## Troubleshooting

### OpenTelemetry not sending data
```bash
# Check OTLP endpoint is reachable
curl -v http://localhost:4318/v1/traces

# Enable debug logging
export OTEL_LOG_LEVEL=debug
```

### structlog not producing JSON
```bash
# Force JSON format
export ACM_LOG_FORMAT=json
```

### Pyroscope not profiling
```python
# Verify pyroscope is available
import pyroscope
print(pyroscope.__version__)
```
