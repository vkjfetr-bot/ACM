# ACM Observability Dashboard

**Comprehensive observability using Prometheus, Tempo, and Loki**

This dashboard provides a complete view of ACM pipeline health, performance, and behavior using the LGTM stack (Loki, Grafana, Tempo, Mimir/Prometheus).

## Dashboard File

`grafana_dashboards/acm_observability.json`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ACM Pipeline                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  core/observability.py                                    │   │
│  │  ├─ Console.info/warn/error → Loki                       │   │
│  │  ├─ record_* functions → Prometheus                      │   │
│  │  └─ Span context managers → Tempo                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Grafana Alloy                                │
│  ├─ OTLP Receiver (4318) → Traces, Metrics                     │
│  └─ Routes to Tempo, Prometheus/Mimir                          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│     Loki      │   │    Tempo      │   │  Prometheus   │
│    (Logs)     │   │   (Traces)    │   │   (Metrics)   │
│  :3100        │   │   :3200       │   │   :9090       │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌───────────────┐
                    │   Grafana     │
                    │    :3000      │
                    └───────────────┘
```

---

## Dashboard Sections

### 🎯 System Health Overview
| Panel | Data Source | Description |
|-------|-------------|-------------|
| Total Runs (24h) | Prometheus | Count of all ACM runs |
| Successful Runs | Prometheus | Runs with outcome=OK |
| Failed Runs | Prometheus | Runs with outcome=FAIL |
| Rows Processed | Prometheus | Total data rows processed |
| Errors (24h) | Prometheus | Error count by type |
| Episodes Detected | Prometheus | Anomaly episodes found |

### ⚡ Equipment Health Metrics (Prometheus)
| Panel | Metric | Description |
|-------|--------|-------------|
| Health Score Over Time | `acm_health_score` | Equipment health 0-100% |
| Remaining Useful Life | `acm_rul_hours` | RUL with P10/P50/P90 bounds |
| Runs by Outcome | `acm_runs_total` | Stacked bar by OK/FAIL/NOOP |
| Episodes by Severity | `acm_episodes_total` | Warning vs info episodes |
| Active Defects | `acm_active_defects` | Current defect count gauge |

### ⏱️ Pipeline Performance (Prometheus)
| Panel | Metric | Description |
|-------|--------|-------------|
| Run Duration Over Time | `acm_run_duration_seconds` | P50/P95 duration histogram |
| Stage Duration by Phase | `acm_stage_duration_seconds` | Parent stage breakdown |
| Average Duration by Stage | `acm_stage_duration_seconds` | Top 10 slowest stages |
| SQL Operations by Table | `acm_sql_ops_total` | Writes by table name |

### 🔍 Distributed Traces (Tempo)
| Panel | Query | Description |
|-------|-------|-------------|
| Recent Traces | TraceQL service.name="acm-pipeline" | Full trace list |
| Trace Duration Scatter | TraceQL | Duration distribution |
| Span Duration by Category | TraceQL | Hierarchical span breakdown |

### 📋 Logs & Events (Loki)
| Panel | Query | Description |
|-------|-------|-------------|
| All Logs (Filtered) | `{app="acm"}` | Full log stream with filters |
| Errors Only | `{app="acm", level="error"}` | Critical failures |
| Warnings Only | `{app="acm", level="warning"}` | Non-fatal issues |
| Log Volume by Level | `count_over_time()` | Histogram by severity |
| Log Volume by Component | `count_over_time()` | Histogram by module |
| Log Volume by Equipment | `count_over_time()` | Histogram by asset |

### ⏱️ Timer Breakdown (Loki)
| Panel | Query | Description |
|-------|-------|-------------|
| Timer Events | `log_type="timer"` | Section duration logs |
| Success Events | `tag="success"` | Completion confirmations |

### 🔧 Component Deep Dive (Loki)
Individual log panels for each ACM component:
- **DATA** - Data loading and validation
- **MODEL** - Model training and persistence
- **FUSE** - Multi-detector fusion
- **SQL** - Database operations
- **FORECAST** - Health/RUL forecasting
- **REGIME** - Operating regime detection

### 🔬 Detector Analytics (Loki)
Logs specific to each anomaly detector:
- **AR1** - Autoregressive detector
- **PCA** - PCA-SPE and PCA-T² detectors
- **OMR** - Overall Model Residual detector

### 🎛️ Coldstart & Model Lifecycle
| Panel | Source | Description |
|-------|--------|-------------|
| Coldstart Completions | Prometheus | Training completions |
| Coldstart Logs | Loki | Detailed coldstart events |

---

## Prometheus Metrics Reference

### Counters
| Metric | Labels | Description |
|--------|--------|-------------|
| `acm_runs_total` | equipment, outcome | Run completions by status |
| `acm_batches_total` | equipment | Batch processing count |
| `acm_rows_processed_total` | equipment | Data rows processed |
| `acm_sql_ops_total` | table, operation, equipment | SQL operation count |
| `acm_coldstarts_total` | equipment | Coldstart completions |
| `acm_episodes_total` | equipment, severity | Anomaly episodes detected |
| `acm_errors_total` | equipment, error_type | Errors by category |
| `acm_model_refits_total` | equipment, reason, detector | Model refit events |

### Gauges
| Metric | Labels | Description |
|--------|--------|-------------|
| `acm_health_score` | equipment | Current health 0-100% |
| `acm_rul_hours` | equipment, percentile | RUL with confidence |
| `acm_active_defects` | equipment | Active defect count |
| `acm_fused_z_score` | equipment | Fused anomaly z-score |
| `acm_detector_z_score` | equipment, detector | Per-detector z-scores |
| `acm_current_regime` | equipment, label | Current operating regime |
| `acm_data_quality_score` | equipment | Data quality percentage |

### Histograms
| Metric | Labels | Description |
|--------|--------|-------------|
| `acm_run_duration_seconds` | equipment, outcome | Total run duration |
| `acm_stage_duration_seconds` | stage, parent, equipment | Stage-level timing |

---

## Loki Labels Reference

All ACM logs use these labels for efficient filtering:

| Label | Values | Description |
|-------|--------|-------------|
| `app` | "acm" | Application identifier |
| `service` | "acm-pipeline" | Service name |
| `equipment` | FD_FAN, GAS_TURBINE, etc. | Equipment being processed |
| `level` | info, warning, error, debug | Log severity |
| `component` | data, model, fuse, sql, etc. | Pipeline component |
| `equip_id` | "1", "2621", etc. | Database equipment ID |
| `run_id` | UUID | Unique run identifier |
| `tag` | success, timer | Optional event type |

### Example LogQL Queries

```logql
# All errors for FD_FAN
{app="acm", level="error", equipment="FD_FAN"}

# MODEL component warnings
{app="acm", level="warning", component="model"}

# Timer events with duration parsing
{app="acm"} | json | log_type="timer" | duration_s > 5

# Full-text search
{app="acm"} |= "SQL failed"

# Component breakdown for specific run
{app="acm", run_id="abc-123"} | component != ""
```

---

## Tempo Trace Structure

ACM traces follow a hierarchical structure with color-coded span kinds:

### Span Categories
| Category | Color | Examples |
|----------|-------|----------|
| CLIENT (blue) | External I/O | load_data, sql.write, persist |
| INTERNAL (green) | Processing | fit.pca, score.gmm, fusion |
| SERVER (purple) | Entry points | acm.run, outputs |
| PRODUCER (orange) | Data generation | data.prep, baseline |

### Trace Attributes
| Attribute | Description |
|-----------|-------------|
| `acm.equipment` | Equipment name |
| `acm.equip_id` | Equipment database ID |
| `acm.run_id` | Unique run identifier |
| `acm.category` | Span category (fit, score, etc.) |

### Example TraceQL Queries

```traceql
# All traces for equipment
{resource.service.name="acm-pipeline" && span.acm.equipment="FD_FAN"}

# Slow spans (>10s)
{resource.service.name="acm-pipeline"} | duration > 10s

# Failed traces
{resource.service.name="acm-pipeline" && status=error}

# Specific run
{span.acm.run_id="abc-123-def"}
```

---

## Variables

| Variable | Type | Description |
|----------|------|-------------|
| `prometheus_ds` | Datasource | Prometheus data source |
| `tempo_ds` | Datasource | Tempo data source |
| `loki_ds` | Datasource | Loki data source |
| `equipment` | Multi-select | Filter by equipment name |
| `component` | Multi-select | Filter by component |
| `log_filter` | Textbox | Full-text log search |

---

## Prerequisites

### Data Sources Required
1. **Prometheus** - For metrics
2. **Tempo** - For distributed traces
3. **Loki** - For logs

### ACM Configuration
Initialize observability in your ACM run:

```python
from core.observability import init

init(
    equipment="FD_FAN",
    equip_id=1,
    run_id="unique-run-id",
    enable_tracing=True,
    enable_metrics=True,
    enable_loki=True,
)
```

### Environment Variables
```bash
# OTLP endpoint (traces + metrics via Alloy)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Loki push endpoint
export LOKI_URL=http://localhost:3100
```

---

## Observability API Reference

The following record functions in `core/observability.py` emit metrics/logs:

### Core Metrics Functions
| Function | Parameters | Description |
|----------|------------|-------------|
| `record_batch(equipment)` | equipment | Record batch start |
| `record_batch_processed(equipment, rows)` | equipment, rows | Record batch completion |
| `record_run(equipment, outcome, duration)` | equipment, outcome, duration_s | Record run completion |
| `record_health_score(equipment, score)` | equipment, score | Set current health (0-100) |
| `record_rul(equipment, rul_hours, percentile, confidence)` | equipment, rul_hours, percentile, confidence | Set RUL prediction |
| `record_active_defects(equipment, count)` | equipment, count | Set active defect count |
| `record_episode(equipment, count, severity)` | equipment, count, severity | Record anomaly episodes |
| `record_error(equipment, error_type)` | equipment, error_type | Record error event |
| `record_coldstart(equipment, status)` | equipment, status | Record coldstart event |
| `record_sql_op(table, operation, rows, equipment)` | table, operation, rows, equipment, duration_ms | Record SQL operation |

### New Detector/Regime/Quality Functions
| Function | Parameters | Description |
|----------|------------|-------------|
| `record_detector_scores(equipment, scores)` | equipment, scores dict | Record all detector z-scores |
| `record_regime(equipment, regime_id, label)` | equipment, regime_id, regime_label | Record current operating regime |
| `record_data_quality(equipment, quality_score)` | equipment, quality_score, missing_pct, outlier_pct, sensors_dropped | Record data quality metrics |
| `record_model_refit(equipment, reason, detector)` | equipment, reason, detector | Record model retrain event |
| `log_timer(section, duration_s, pct)` | section, duration_s, pct, parent, total_s | Log timer section to Loki |

---

## Installation

1. In Grafana, go to **Dashboards > Import**
2. Click **Upload dashboard JSON file**
3. Select `grafana_dashboards/acm_observability.json`
4. Configure data source mappings
5. Click **Import**

---

## Troubleshooting

### No Metrics Appearing
- Verify Prometheus is receiving metrics: `http://localhost:9090/targets`
- Check OTLP endpoint is accessible: `curl http://localhost:4318/v1/metrics`
- Ensure `init()` was called with `enable_metrics=True`

### No Traces
- Verify Tempo is running: `http://localhost:3200/ready`
- Check OTLP trace export: `curl http://localhost:4318/v1/traces`
- Ensure `init()` was called with `enable_tracing=True`

### No Logs in Loki
- Check Loki ready: `curl http://localhost:3100/ready`
- Verify push endpoint: `http://localhost:3100/loki/api/v1/push`
- Check `Console.info()` calls have `component=` parameter

### Labels Not Filterable
- Ensure using new Console API: `Console.info("msg", component="DATA")`
- Verify Loki is receiving labels (check stream metadata)
- Labels are lowercase in Loki: `component="data"` not `component="DATA"`
