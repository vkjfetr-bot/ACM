# ACM Observability Stack - Installation Guide

## Overview

This directory contains configuration files and installation scripts for the complete Grafana observability stack:

| Component | Purpose | Port |
|-----------|---------|------|
| **Tempo** | Distributed tracing backend | 3200 (API), 4317/4318 (OTLP) |
| **Pyroscope** | Continuous profiling | 4040 |
| **Loki** | Log aggregation | 3100 |
| **Alloy** | OTLP collector/router (optional) | 4317/4318 |
| **Grafana** | Visualization (already installed) | 3000 |

## Quick Start (Docker Compose) - RECOMMENDED

**Note:** Tempo and Pyroscope do not provide native Windows binaries. Docker is the recommended deployment method.

### Prerequisites

- **Docker Desktop** for Windows (https://www.docker.com/products/docker-desktop/)
- **Grafana** v10.0+ (already installed for ACM dashboards)

### Step 1: Start the Stack

```powershell
cd C:\Users\bhadk\Documents\ACM V8 SQL\ACM\install\observability
docker-compose up -d
```

This starts:
- **acm-tempo** - Distributed tracing (ports 3200, 4317, 4318)
- **acm-pyroscope** - Continuous profiling (port 4040)
- **acm-loki** - Log aggregation (port 3100)

### Step 2: Verify Services

```powershell
# Check all containers are healthy
docker ps

# Test endpoints
Invoke-WebRequest -Uri "http://localhost:3200/ready" -UseBasicParsing  # Tempo
Invoke-WebRequest -Uri "http://localhost:4040/ready" -UseBasicParsing  # Pyroscope
Invoke-WebRequest -Uri "http://localhost:3100/ready" -UseBasicParsing  # Loki
```

### Step 3: Configure Grafana Datasources

Run the configuration script (requires Grafana admin password):

```powershell
.\configure-grafana-datasources.ps1 -Password "your-grafana-password"
```

Or manually add in Grafana UI (http://localhost:3000 → Configuration → Data Sources):

| Datasource | Type | URL |
|------------|------|-----|
| Tempo | Tempo | http://localhost:3200 |
| Pyroscope | Pyroscope | http://localhost:4040 |
| Loki | Loki | http://localhost:3100 |

### Step 4: Test with ACM

```powershell
$env:OTEL_EXPORTER_OTLP_ENDPOINT = "http://localhost:4318"
$env:OTEL_SERVICE_NAME = "acm-pipeline"
python -m core.acm_main --equip WFA_TURBINE_10 --start-time 2022-10-09T08:40:00 --end-time 2022-10-12T08:40:00
```

## Managing the Stack

```powershell
# Stop the stack
docker-compose down

# View logs
docker-compose logs -f tempo
docker-compose logs -f pyroscope
docker-compose logs -f loki

# Restart a service
docker-compose restart tempo

# Remove all data (clean start)
docker-compose down -v
```

## WSL/Linux Alternative

If you prefer native binaries via WSL:

```bash
# In WSL terminal
cd /mnt/c/Users/<you>/grafana-stack

# Download Linux binaries
curl -LO https://github.com/grafana/tempo/releases/download/v2.9.0/tempo_2.9.0_linux_amd64.tar.gz
curl -LO https://github.com/grafana/pyroscope/releases/download/v1.16.0/pyroscope_1.16.0_linux_amd64.tar.gz

# Extract
tar -xzf tempo_2.9.0_linux_amd64.tar.gz
tar -xzf pyroscope_1.16.0_linux_amd64.tar.gz

# Run
./tempo -config.file=tempo.yaml &
./pyroscope -config.file=pyroscope.yaml &
```

## Configuration Files

| File | Purpose |
|------|---------|
| `tempo.yaml` | Tempo config - OTLP receiver, local storage |
| `pyroscope.yaml` | Pyroscope config - profile storage |
| `alloy.config` | Alloy config - routes OTLP to Tempo/Loki |
| `install-observability-stack.ps1` | PowerShell installer |

## Grafana Datasource Setup

After starting the stack, add these datasources in Grafana:

1. **Tempo** (Traces)
   - Type: Tempo
   - URL: `http://localhost:3200`
   
2. **Pyroscope** (Profiles)
   - Type: Pyroscope
   - URL: `http://localhost:4040`
   
3. **Loki** (Logs)
   - Type: Loki
   - URL: `http://localhost:3100`

## Python Configuration

Set these environment variables for ACM:

```powershell
# PowerShell
$env:OTEL_EXPORTER_OTLP_ENDPOINT = "http://localhost:4318"
$env:OTEL_SERVICE_NAME = "acm-pipeline"
$env:ACM_PYROSCOPE_ENDPOINT = "http://localhost:4040"
$env:ACM_OBSERVABILITY_ENABLED = "true"
```

Or in Python:

```python
from core.observability import init_observability

init_observability(
    service_name="acm-pipeline",
    otlp_endpoint="http://localhost:4318",
    pyroscope_endpoint="http://localhost:4040",
    log_level="INFO"
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ACM Python App                           │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │  structlog    │  │ OpenTelemetry │  │  pyroscope-io │       │
│  │  (logs)       │  │  (traces)     │  │  (profiles)   │       │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘       │
└──────────┼──────────────────┼──────────────────┼───────────────┘
           │                  │                  │
           │ OTLP/HTTP        │ OTLP/HTTP        │ HTTP POST
           ▼                  ▼                  ▼
    ┌──────────────────────────────────┐   ┌───────────────┐
    │         Grafana Alloy            │   │   Pyroscope   │
    │  (OTLP collector - port 4318)    │   │  (port 4040)  │
    └──────────┬───────────────┬───────┘   └───────────────┘
               │               │
        ┌──────┴──────┐  ┌─────┴──────┐
        ▼             ▼  ▼            │
   ┌─────────┐   ┌─────────┐          │
   │  Tempo  │   │  Loki   │          │
   │ (3200)  │   │ (3100)  │          │
   └────┬────┘   └────┬────┘          │
        │             │               │
        └──────┬──────┴───────────────┘
               │
               ▼
        ┌──────────────┐
        │   Grafana    │
        │   (3000)     │
        └──────────────┘
```

## Troubleshooting

### Tempo not receiving traces

1. Check Tempo is running: `curl http://localhost:3200/ready`
2. Check OTLP port: `netstat -an | findstr 4317`
3. Test trace ingestion:
   ```python
   from opentelemetry import trace
   tracer = trace.get_tracer("test")
   with tracer.start_as_current_span("test-span"):
       print("Trace sent!")
   ```

### Pyroscope not receiving profiles

1. Check Pyroscope is running: `curl http://localhost:4040/ready`
2. Verify pyroscope-io is installed: `pip install pyroscope-io`
   - Note: On Windows, this requires Rust. Use `pip install py-spy` as alternative.

### Logs not appearing in Loki

1. Check Loki is running: `curl http://localhost:3100/ready`
2. Check Alloy is forwarding: Look at Alloy logs
3. Verify structlog is configured with OTLP exporter

## SQL Metrics via MSSQL Exporter

Alloy scrapes ACM operational metrics directly from SQL Server tables. This provides database-level visibility that complements the Python app metrics.

### Installation

1. Copy the SQL metrics config:
   ```powershell
   Copy-Item "install\observability\acm_sql_metrics.yaml" "C:\Program Files\GrafanaLabs\Alloy\acm_sql_metrics.yaml"
   ```

2. Copy the updated Alloy config:
   ```powershell
   Copy-Item "install\observability\config.alloy" "C:\Program Files\GrafanaLabs\Alloy\config.alloy"
   ```

3. Restart Alloy:
   ```powershell
   Restart-Service Alloy
   ```

### Available SQL Metrics

| Metric | Labels | Description |
|--------|--------|-------------|
| `acm_sql_runs_total` | equipment, outcome | Total runs by equipment/outcome |
| `acm_sql_runs_last_24h` | equipment | Runs in last 24 hours |
| `acm_sql_run_duration_seconds` | equipment | Average run duration |
| `acm_sql_last_run_timestamp` | equipment | Unix timestamp of last run |
| `acm_sql_scores_rows_total` | equipment | Rows in ACM_Scores_Wide |
| `acm_sql_scores_latest_timestamp` | equipment | Most recent score timestamp |
| `acm_sql_health_timeline_rows` | equipment | Rows in ACM_HealthTimeline |
| `acm_sql_anomaly_events_total` | equipment, severity | Anomaly events by severity |
| `acm_sql_active_defects` | equipment | Currently active defects |
| `acm_sql_episodes_total` | equipment | Total diagnostic episodes |
| `acm_sql_episode_duration_hours` | equipment | Total episode duration |
| `acm_sql_rul_hours` | equipment, method | Latest RUL prediction |
| `acm_sql_health_current` | equipment | Current health score (0-100) |
| `acm_sql_models_count` | equipment, model_type | Models in registry |
| `acm_sql_model_size_bytes` | equipment | Model blob sizes |
| `acm_sql_logs_by_level` | equipment, log_level | Log entries by level |
| `acm_sql_errors_last_24h` | equipment | Errors in last 24 hours |
| `acm_sql_table_rows` | table_name | Row count per ACM table |
| `acm_sql_table_size_mb` | table_name | Table size in MB |
| `acm_sql_database_size_mb` | - | Total database size |
| `acm_sql_active_connections` | - | Active DB connections |

### Windows Performance Counters

Alloy also collects Windows system metrics:

| Metric Prefix | Description |
|---------------|-------------|
| `windows_cpu_*` | CPU usage per core |
| `windows_memory_*` | Memory usage |
| `windows_logical_disk_*` | Disk I/O |
| `windows_process_*` | Per-process stats (python/acm) |
| `windows_system_*` | System uptime, context switches |

## Port Summary

| Port | Service | Protocol |
|------|---------|----------|
| 3000 | Grafana UI | HTTP |
| 3100 | Loki API | HTTP |
| 3200 | Tempo API | HTTP |
| 4040 | Pyroscope API/UI | HTTP |
| 4317 | OTLP gRPC | gRPC |
| 4318 | OTLP HTTP | HTTP |
| 9095 | Tempo gRPC | gRPC |
