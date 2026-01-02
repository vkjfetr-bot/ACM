# ACM Operational Metrics Inventory

> **Philosophy**: These metrics monitor ACM itself as a production system - not asset health outcomes.
> We need visibility into EVERY operation that happens behind the scenes.

**Version**: 10.3.0  
**Last Updated**: 2025-01-27

---

## Metrics Categories

| Category | Purpose | Priority |
|----------|---------|----------|
| Pipeline Execution | Track run lifecycle, timing, outcomes | P0 |
| Data Operations | Data loading, validation, row counts | P0 |
| SQL Operations | Query timing, connection health | P0 |
| Model Operations | Load, save, train, predict timing | P0 |
| Detector Operations | Per-detector execution metrics | P1 |
| Batch Runner | Subprocess management, parallelism | P1 |
| System Resources | CPU, memory, I/O during runs | P2 |

---

## P0: Pipeline Execution Metrics

### Existing Metrics

| Metric | Type | Labels | Status | Location |
|--------|------|--------|--------|----------|
| `acm_runs_total` | Counter | `equipment`, `outcome` | ✅ Implemented | `core/observability.py` |
| `acm_run_duration_seconds` | Histogram | `equipment` | ✅ Implemented | `core/observability.py` |
| `acm_errors_total` | Counter | `equipment`, `error_type` | ✅ Implemented | `core/observability.py` |
| `acm_data_points_processed_total` | Counter | `equipment` | ✅ Implemented | `core/observability.py` |

### Needed Metrics

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `acm_coldstart_runs_total` | Counter | `equipment` | Track coldstart vs normal runs |
| `acm_noop_runs_total` | Counter | `equipment`, `reason` | Why did we skip processing? |
| `acm_pipeline_stage_duration_seconds` | Histogram | `equipment`, `stage` | Time per stage (load, detect, fuse, forecast) |

---

## P0: Data Operations Metrics

### Needed Metrics

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `acm_data_load_duration_seconds` | Histogram | `equipment`, `source` | How long to load data from SQL |
| `acm_data_load_rows_total` | Counter | `equipment` | Rows loaded per run |
| `acm_data_validation_failures_total` | Counter | `equipment`, `check` | Schema/null/range validation failures |
| `acm_data_time_range_seconds` | Gauge | `equipment` | Time span of loaded data window |
| `acm_last_processed_timestamp` | Gauge | `equipment` | Epoch of most recent processed data point |

---

## P0: SQL Operations Metrics

### Needed Metrics

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `acm_sql_query_duration_seconds` | Histogram | `operation`, `table` | Query/insert/update timing |
| `acm_sql_connection_acquire_seconds` | Histogram | - | Connection pool timing |
| `acm_sql_rows_written_total` | Counter | `table` | Rows written per table |
| `acm_sql_errors_total` | Counter | `operation`, `error_type` | SQL failures by type |
| `acm_sql_retry_total` | Counter | `operation` | Connection retries |

---

## P0: Model Operations Metrics

### Needed Metrics

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `acm_model_load_duration_seconds` | Histogram | `equipment`, `model_type` | Model deserialization time |
| `acm_model_save_duration_seconds` | Histogram | `equipment`, `model_type` | Model serialization time |
| `acm_model_train_duration_seconds` | Histogram | `equipment`, `detector` | Training time per detector |
| `acm_model_predict_duration_seconds` | Histogram | `equipment`, `detector` | Inference time per detector |
| `acm_model_size_bytes` | Gauge | `equipment`, `model_type` | Model blob size |
| `acm_model_cache_hit_total` | Counter | `equipment` | Model cache hits |
| `acm_model_cache_miss_total` | Counter | `equipment` | Model cache misses (reload required) |

---

## P1: Detector Operations Metrics

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `acm_detector_duration_seconds` | Histogram | `equipment`, `detector` | Per-detector execution time |
| `acm_detector_anomalies_total` | Counter | `equipment`, `detector` | Anomalies detected per detector |
| `acm_detector_score_distribution` | Histogram | `equipment`, `detector` | Score distribution (calibration check) |
| `acm_detector_trained_features` | Gauge | `equipment`, `detector` | Number of features used |

---

## P1: Batch Runner Metrics

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `acm_batch_equipment_total` | Counter | `batch_id` | Equipment count in batch |
| `acm_batch_duration_seconds` | Histogram | - | Total batch duration |
| `acm_batch_parallel_workers` | Gauge | - | Current parallelism level |
| `acm_subprocess_spawn_total` | Counter | `equipment`, `outcome` | Subprocess outcomes |
| `acm_subprocess_duration_seconds` | Histogram | `equipment` | Per-equipment subprocess time |
| `acm_batch_queue_depth` | Gauge | - | Equipment waiting to be processed |

---

## P2: System Resource Metrics

Collected via Grafana Alloy Windows exporter:

| Metric | Type | Source | Purpose |
|--------|------|--------|---------|
| `windows_cpu_time_total` | Counter | Alloy | CPU usage |
| `windows_cs_physical_memory_bytes` | Gauge | Alloy | Memory usage |
| `windows_logical_disk_*` | Various | Alloy | Disk I/O |
| `windows_process_*` | Various | Alloy | Per-process stats (python/acm) |

Configuration: `install/observability/config.alloy`

---

## Implementation Locations

### core/observability.py
Central metrics registry. Add all new metrics here.

```python
# Existing
acm_runs_total = Counter("acm_runs_total", "Total ACM runs", ["equipment", "outcome"])
acm_run_duration_seconds = Histogram("acm_run_duration_seconds", "Run duration", ["equipment"])
acm_errors_total = Counter("acm_errors_total", "Total errors", ["equipment", "error_type"])
acm_data_points_processed_total = Counter("acm_data_points_processed_total", "Rows processed", ["equipment"])

# Add P0 metrics here
```

### core/sql_client.py
Instrument with query timing:

```python
def _execute_with_metrics(self, operation: str, table: str, func):
    with acm_sql_query_duration_seconds.labels(operation=operation, table=table).time():
        return func()
```

### core/model_persistence.py
Instrument model load/save:

```python
def load_model(...):
    with acm_model_load_duration_seconds.labels(equipment=equip, model_type=mtype).time():
        # existing load logic
```

### scripts/sql_batch_runner.py
Already has `record_run()` and `record_error()` calls. Add batch-level metrics.

---

## Grafana Dashboard Panels

### acm_observability.json

| Panel | Metric | Status |
|-------|--------|--------|
| Total Runs (24h) | `acm_runs_total` | ✅ Fixed |
| Successful Runs | `acm_runs_total{outcome="SUCCESS"}` | ✅ Fixed |
| Failed Runs | `acm_runs_total{outcome="FAIL"}` | ✅ Fixed |
| Errors (24h) | `acm_errors_total` | ✅ Fixed |
| Run Duration | `acm_run_duration_seconds` | ✅ Exists |
| Data Points | `acm_data_points_processed_total` | ✅ Exists |

### Needed Panels

| Panel | Metric | Priority |
|-------|--------|----------|
| SQL Query Latency | `acm_sql_query_duration_seconds` | P0 |
| Data Load Time | `acm_data_load_duration_seconds` | P0 |
| Model Load Time | `acm_model_load_duration_seconds` | P0 |
| Rows Loaded | `acm_data_load_rows_total` | P0 |
| NOOP Reasons | `acm_noop_runs_total` | P1 |

---

## Tracing Integration

All P0 metrics should be recorded within trace spans:

```python
with Span("acm.pipeline.load_data", equipment=equip) as span:
    start = time.time()
    data = load_data()
    duration = time.time() - start
    
    acm_data_load_duration_seconds.labels(equipment=equip, source="sql").observe(duration)
    acm_data_load_rows_total.labels(equipment=equip).inc(len(data))
    span.set_attribute("rows", len(data))
```

---

## Implementation Roadmap

### Phase 1: SQL Timing (This Week)
1. Add `acm_sql_query_duration_seconds` to `core/sql_client.py`
2. Wrap `execute()`, `fetch_all()`, `execute_many()` with timing
3. Add Grafana panel for SQL latency percentiles

### Phase 2: Data Operations (Next)
1. Add `acm_data_load_*` metrics to `core/acm_main.py`
2. Track rows loaded, time ranges, validation failures
3. Expose last processed timestamp as gauge

### Phase 3: Model Operations
1. Add model timing to `core/model_persistence.py`
2. Track cache hits/misses if caching implemented
3. Monitor model blob sizes

### Phase 4: Detector Breakdown
1. Per-detector timing in respective modules
2. Anomaly counts per detector
3. Feature count tracking

---

## Alerting Rules (Future)

```yaml
# Prometheus alerting rules
groups:
  - name: acm-operations
    rules:
      - alert: ACMHighFailureRate
        expr: rate(acm_runs_total{outcome="FAIL"}[1h]) / rate(acm_runs_total[1h]) > 0.1
        for: 5m
        annotations:
          summary: "ACM failure rate above 10%"
          
      - alert: ACMSlowSQLQueries
        expr: histogram_quantile(0.95, rate(acm_sql_query_duration_seconds_bucket[5m])) > 5
        for: 5m
        annotations:
          summary: "95th percentile SQL query time > 5s"
          
      - alert: ACMNoRuns
        expr: increase(acm_runs_total[1h]) == 0
        for: 1h
        annotations:
          summary: "No ACM runs in the last hour"
```

---

## Quick Reference

### Recording a successful run
```python
from core.observability import record_run, record_data_points
record_run(equipment, "SUCCESS", duration_seconds)
record_data_points(equipment, row_count)
```

### Recording a failure
```python
from core.observability import record_run, record_error
record_run(equipment, "FAIL", duration_seconds)
record_error(equipment, "DataLoadError")
```

### Recording SQL timing (to be implemented)
```python
from core.observability import record_sql_query
with record_sql_query("SELECT", "ACM_Scores_Wide"):
    cursor.execute(query)
```
