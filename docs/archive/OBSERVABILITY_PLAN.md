# ACM Comprehensive Observability Plan v10.3.0

## Overview

This document defines the complete observability strategy for ACM (Automated Condition Monitoring). The goal is to have **full visibility** into every operation ACM performs, enabling:
- Performance optimization
- Root cause analysis
- Capacity planning
- Alerting on degradation

---

## 1. TRACES (OpenTelemetry Spans)

### 1.1 Root Span Structure

```
acm.run (root)
├── acm.startup
│   ├── acm.config.load
│   └── acm.sql.connect
├── acm.data
│   ├── acm.data.load_historian
│   ├── acm.data.split_train_score
│   ├── acm.data.parse_timestamps
│   └── acm.data.quality_check
├── acm.features
│   ├── acm.features.compute_fill_values
│   ├── acm.features.compute_train
│   ├── acm.features.compute_score
│   ├── acm.features.normalize
│   └── acm.features.impute
├── acm.models
│   ├── acm.models.load_cache
│   ├── acm.fit
│   │   ├── acm.fit.ar1
│   │   ├── acm.fit.pca
│   │   ├── acm.fit.iforest
│   │   ├── acm.fit.gmm
│   │   └── acm.fit.omr
│   ├── acm.score
│   │   ├── acm.score.ar1
│   │   ├── acm.score.pca
│   │   ├── acm.score.iforest
│   │   ├── acm.score.gmm
│   │   └── acm.score.omr
│   └── acm.models.save
├── acm.calibrate
│   └── acm.calibrate.self_tune
├── acm.fusion
│   ├── acm.fusion.compute_weights
│   ├── acm.fusion.fuse_scores
│   └── acm.fusion.detect_episodes
├── acm.regimes
│   ├── acm.regimes.cluster
│   ├── acm.regimes.label
│   └── acm.regimes.transient_detection
├── acm.thresholds
│   └── acm.thresholds.adaptive
├── acm.persist
│   ├── acm.persist.scores_wide
│   ├── acm.persist.episodes
│   ├── acm.persist.health_timeline
│   └── acm.persist.[table_name] (for each of 35+ tables)
├── acm.analytics
│   └── acm.analytics.generate
├── acm.forecast
│   ├── acm.forecast.load_history
│   ├── acm.forecast.fit_model
│   ├── acm.forecast.predict_rul
│   └── acm.forecast.write_results
└── acm.cleanup
    ├── acm.sql.finalize
    └── acm.metrics.flush
```

### 1.2 Span Attributes (Required)

| Attribute | Type | Description |
|-----------|------|-------------|
| `acm.equipment` | string | Equipment name (e.g., "WFA_TURBINE_10") |
| `acm.equip_id` | int | Equipment ID from SQL |
| `acm.run_id` | string | UUID of current run |
| `acm.batch_num` | int | Batch number (1-indexed) |
| `acm.batch_total` | int | Total batches in job |
| `acm.row_count` | int | Rows processed in operation |
| `acm.outcome` | string | "OK", "NOOP", "FAIL" |

### 1.3 Span Attributes (Per-Operation)

**Data Loading:**
- `acm.data.source`: "sql_historian" | "file"
- `acm.data.start_time`: ISO timestamp
- `acm.data.end_time`: ISO timestamp
- `acm.data.rows_loaded`: int

**Detectors:**
- `acm.detector.name`: "ar1" | "pca" | "iforest" | "gmm" | "omr"
- `acm.detector.operation`: "fit" | "score"
- `acm.detector.n_features`: int
- `acm.detector.n_samples`: int

**SQL Writes:**
- `acm.sql.table`: table name
- `acm.sql.rows_written`: int
- `acm.sql.duration_ms`: float

**Forecasting:**
- `acm.forecast.rul_p50`: float (hours)
- `acm.forecast.rul_p10`: float
- `acm.forecast.rul_p90`: float
- `acm.forecast.confidence`: float

---

## 2. METRICS (Prometheus/OpenTelemetry)

### 2.1 Counters

| Metric Name | Labels | Description |
|-------------|--------|-------------|
| `acm_batches_total` | equipment, status | Total batches processed |
| `acm_rows_processed_total` | equipment, operation | Rows processed (load/write) |
| `acm_episodes_detected_total` | equipment, severity | Episodes by severity |
| `acm_sql_queries_total` | table, operation | SQL operations count |
| `acm_sql_rows_written_total` | table | Rows written per table |
| `acm_model_fits_total` | detector | Model training count |
| `acm_errors_total` | equipment, error_type | Error counts |
| `acm_coldstarts_total` | equipment | Coldstart completions |

### 2.2 Histograms (Durations)

| Metric Name | Labels | Buckets (seconds) |
|-------------|--------|-------------------|
| `acm_batch_duration_seconds` | equipment, status | 10, 30, 60, 120, 300, 600 |
| `acm_data_load_seconds` | equipment | 1, 5, 10, 30, 60 |
| `acm_feature_compute_seconds` | equipment | 1, 5, 10, 30 |
| `acm_detector_fit_seconds` | detector | 1, 5, 10, 30, 60, 120 |
| `acm_detector_score_seconds` | detector | 0.1, 0.5, 1, 5, 10 |
| `acm_fusion_seconds` | equipment | 0.1, 0.5, 1, 5 |
| `acm_sql_write_seconds` | table | 0.1, 1, 5, 10, 30, 60 |
| `acm_forecast_seconds` | equipment | 1, 5, 10, 30 |

### 2.3 Gauges

| Metric Name | Labels | Description |
|-------------|--------|-------------|
| `acm_health_score` | equipment | Current health (0-100) |
| `acm_rul_hours` | equipment, percentile | RUL P10/P50/P90 |
| `acm_active_defects` | equipment | Number of active defects |
| `acm_episode_count` | equipment | Open episodes |
| `acm_model_version` | equipment, detector | Model version number |
| `acm_threshold_alert` | equipment | Alert threshold value |
| `acm_threshold_warn` | equipment | Warning threshold value |
| `acm_queue_size` | queue_name | Pending work items |

### 2.4 SQL Server Metrics

| Metric Name | Labels | Description |
|-------------|--------|-------------|
| `acm_sql_connection_pool_size` | | Active connections |
| `acm_sql_connection_pool_available` | | Available connections |
| `acm_sql_query_duration_seconds` | query_type | Query execution time |
| `acm_sql_deadlocks_total` | | Deadlock count |
| `acm_sql_table_row_count` | table | Rows in ACM tables |
| `acm_sql_table_size_bytes` | table | Table size |

### 2.5 System Metrics (via Prometheus Node Exporter or Alloy)

| Metric Name | Description |
|-------------|-------------|
| `process_cpu_seconds_total` | CPU usage |
| `process_resident_memory_bytes` | Memory usage |
| `python_gc_objects_collected_total` | Garbage collection |
| `process_open_fds` | File descriptors |

---

## 3. LOGS (Structured JSON via structlog)

### 3.1 Log Format

```json
{
  "timestamp": "2025-12-17T10:15:00.123Z",
  "level": "INFO",
  "event": "Batch processing started",
  "logger": "core.acm_main",
  "trace_id": "abc123...",
  "span_id": "def456...",
  "run_id": "159cf1aa-6654-46b7-90c1-8c2c11102a49",
  "equip_id": 5010,
  "equipment": "WFA_TURBINE_10",
  "batch": 1,
  "batch_total": 5,
  "category": "BATCH",
  "extra_field": "value"
}
```

### 3.2 Log Levels and Categories

| Level | Category | When to Use |
|-------|----------|-------------|
| DEBUG | DATA | Raw data details, intermediate values |
| DEBUG | FEAT | Feature computation details |
| DEBUG | MODEL | Model internals, weights |
| INFO | RUN | Run start/end, configuration |
| INFO | DATA | Row counts, time ranges |
| INFO | TIMER | Performance timings |
| INFO | OUTPUT | SQL writes, file outputs |
| WARNING | DATA | Low variance, missing data |
| WARNING | THRESHOLD | Threshold crossings |
| WARNING | QUALITY | Clustering quality issues |
| ERROR | SQL | Database errors |
| ERROR | MODEL | Model training failures |
| ERROR | RUN | Fatal run errors |

### 3.3 Required Log Fields

**Always Include:**
- `trace_id`, `span_id` (from OTEL context)
- `run_id`, `equip_id`, `equipment`
- `batch`, `batch_total` (if batch mode)
- `category` (operation type)

**Operation-Specific:**
- `row_count`, `duration_ms`, `table_name`
- `detector`, `n_samples`, `n_features`
- `error_type`, `error_message`, `stack_trace`

---

## 4. PROFILING (Pyroscope)

### 4.1 Profile Types

| Profile Type | When to Use |
|--------------|-------------|
| `cpu` | Always on - identifies CPU hotspots |
| `alloc_objects` | Memory allocation tracking |
| `alloc_space` | Memory size tracking |
| `inuse_objects` | Objects in memory |
| `inuse_space` | Memory in use |
| `goroutines` | Not applicable (Python) |

### 4.2 Profiling Strategy

**Always Profile:**
- GMM fitting (known CPU intensive)
- PCA fitting (large matrices)
- Feature computation (Polars/pandas)
- SQL batch writes

**Conditional Profiling:**
- Enable via `ACM_PYROSCOPE_ENDPOINT` env var
- Disable in high-frequency streaming mode
- Enable for batch processing

### 4.3 Tags for Pyroscope

```python
pyroscope.configure(
    tags={
        "equipment": equipment_name,
        "operation": "batch",  # or "coldstart", "streaming"
        "detector": "all",
    }
)
```

---

## 5. SQL SERVER OBSERVABILITY

### 5.1 Query Instrumentation

Wrap all SQL operations:
```python
with tracer.start_as_current_span("acm.sql.query") as span:
    span.set_attribute("acm.sql.table", table_name)
    span.set_attribute("acm.sql.operation", "INSERT")
    start = time.perf_counter()
    result = execute_query(sql)
    span.set_attribute("acm.sql.duration_ms", (time.perf_counter() - start) * 1000)
    span.set_attribute("acm.sql.rows_affected", result.rowcount)
```

### 5.2 Connection Pool Monitoring

```python
# Emit metrics every batch
sql_connection_pool_size.set(pool.size)
sql_connection_pool_available.set(pool.available)
sql_connection_pool_waiting.set(pool.waiting)
```

### 5.3 Table Statistics (Periodic)

```sql
-- Run weekly/daily to track growth
SELECT 
    t.NAME AS TableName,
    p.rows AS RowCount,
    SUM(a.total_pages) * 8 AS TotalSpaceKB
FROM sys.tables t
INNER JOIN sys.indexes i ON t.OBJECT_ID = i.object_id
INNER JOIN sys.partitions p ON i.object_id = p.OBJECT_ID AND i.index_id = p.index_id
INNER JOIN sys.allocation_units a ON p.partition_id = a.container_id
WHERE t.NAME LIKE 'ACM_%'
GROUP BY t.Name, p.Rows
ORDER BY TotalSpaceKB DESC
```

---

## 6. DASHBOARDS

### 6.1 ACM Operations Overview

**Panels:**
1. Batches processed (time series)
2. Batch duration distribution (histogram)
3. Success/Failure rate (stat)
4. Rows processed (counter)
5. Active equipment (table)
6. Recent errors (logs panel)

### 6.2 Equipment Health Dashboard

**Per-Equipment:**
1. Health score gauge (0-100)
2. Health timeline (time series)
3. RUL prediction (stat + trend)
4. Active defects (list)
5. Episode timeline (annotations)
6. Top contributing sensors (bar chart)

### 6.3 Performance Analysis

**Panels:**
1. Duration by step (pie chart from traces)
2. Slowest operations (table from traces)
3. SQL write times (histogram)
4. Detector fit/score times (stacked bar)
5. Memory usage (time series)
6. CPU flame graph (Pyroscope embed)

### 6.4 Alerting Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| Batch Failed | `acm_errors_total > 0` | Critical |
| Slow Batch | `acm_batch_duration_seconds > 600` | Warning |
| Health Critical | `acm_health_score < 30` | Critical |
| RUL Low | `acm_rul_hours{percentile="p50"} < 72` | Warning |
| Connection Pool Exhausted | `acm_sql_connection_pool_available < 2` | Critical |
| Model Stale | `time() - acm_model_last_updated > 86400` | Warning |

---

## 7. IMPLEMENTATION PRIORITY

### Phase 1: Foundation ✅ COMPLETED
- [x] Basic tracing (root span `acm.run`)
- [x] Basic metrics (`acm_batches_processed_total`, `acm_batch_duration_seconds`)
- [x] Logs to Loki via OTEL bridge (Console → Python logging → OTLP → Alloy → Loki)
- [x] Trace context (trace_id, span_id) in log records
- [x] Pyroscope profiling working

### Phase 2: Comprehensive Tracing (IN PROGRESS)
- [ ] Add spans for each detector fit/score
- [ ] Add spans for each SQL table write
- [ ] Add spans for forecasting pipeline
- [ ] Link spans with parent-child relationships

### Phase 3: Full Metrics
- [ ] Add all counters/histograms/gauges from Section 2
- [ ] SQL Server metrics via pyodbc instrumentation
- [ ] System metrics via Alloy

### Phase 4: Dashboards and Alerts
- [ ] Create Grafana dashboards
- [ ] Set up alerting rules
- [ ] Create runbooks for alerts

---

## 8. CODE CHANGES REQUIRED

### 8.1 Replace Console with Logger

```python
# Before
Console.info("[DATA] Loaded 53591 rows")

# After
from core.observability import get_logger
log = get_logger(__name__)
log.info("data_loaded", category="DATA", row_count=53591)
```

### 8.2 Add Tracing Decorator

```python
from core.observability import traced

@traced("acm.fit.pca")
def fit_pca(train_data):
    # Automatically creates span with timing
    ...
```

### 8.3 Add Metrics Recording

```python
from core.observability import record_detector_duration, record_batch_processed

# After detector completes
record_detector_duration("pca", equipment, elapsed_seconds)

# After batch completes
record_batch_processed(equipment, total_duration, rows, "ok")
```

---

## 9. ENVIRONMENT VARIABLES

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | Alloy collector |
| `OTEL_SERVICE_NAME` | `acm-pipeline` | Service name |
| `ACM_LOG_LEVEL` | `INFO` | Minimum log level |
| `ACM_LOG_FORMAT` | `json` | `json` or `console` |
| `ACM_PYROSCOPE_ENDPOINT` | (none) | Pyroscope server |
| `ACM_METRICS_EXPORT_INTERVAL_MS` | `60000` | Metrics push interval |

---

## 10. NEXT STEPS

1. **Immediate**: Fix logging to use Python stdlib and export to Loki
2. **This Week**: Add spans for all detector operations
3. **This Month**: Complete metrics coverage
4. **Ongoing**: Build dashboards iteratively based on needs
