# ACM Tracing Audit and Improvement Plan

**Date**: 2025-12-20  
**Version**: v10.3.0  
**Status**: Design Document

---

## Executive Summary

This document audits the current state of OpenTelemetry tracing in ACM and proposes improvements to:
1. **Color-code traces** by major feature/module for visual clarity in Grafana Tempo
2. **Ensure proper correlation** between metrics, logs, traces, and profiles
3. **Standardize naming conventions** for consistency and filterability
4. **Fill gaps** where important pipeline phases lack span instrumentation

---

## Current State Analysis

### 1. Existing Span Coverage

#### A. Detector Modules (Core Algorithm Spans)

| Module | Fit Span | Score Span | Location |
|--------|----------|------------|----------|
| AR1 | `fit.ar1` | `score.ar1` | `core/ar1_detector.py` |
| PCA | `fit.pca` | `score.pca` | `core/correlation.py` |
| IForest | `fit.iforest` | `score.iforest` | `core/outliers.py` |
| GMM | `fit.gmm` | `score.gmm` | `core/outliers.py` |
| OMR | `fit.omr` | `score.omr` | `core/omr.py` |

**Observations**:
- ✅ Consistent naming pattern: `fit.{detector}` and `score.{detector}`
- ✅ Attributes include `n_samples`, `n_features`
- ⚠️ All use span kind `INTERNAL` (green) - could differentiate fit vs score

#### B. Pipeline Orchestration Spans (Timer Sections)

The main pipeline (`core/acm_main.py`) uses `Timer` class which creates spans internally:

| Phase | Timer Section | Purpose |
|-------|---------------|---------|
| Startup | `startup` | Initialization, config loading |
| Data Loading | `load_data` | SQL data retrieval |
| Baseline | `baseline.seed` | Coldstart baseline creation |
| Data Quality | `data.guardrails.data_quality` | Data validation |
| Features | `features.build`, `features.compute_train`, `features.compute_score` | Feature engineering |
| Feature Hash | `features.hash` | Deterministic hash for caching |
| Model Loading | `models.persistence.load` | Load from SQL ModelRegistry |
| Model Fitting | `train.detector_fit`, `fit.{detector}` | Train all detectors |
| Model Scoring | `score.detector_score`, `score.{detector}` | Score all detectors |
| Regimes | `regimes.label` | Regime clustering |
| Calibration | `calibrate` | Detector calibration |
| Fusion | `fusion`, `fusion.auto_tune` | Multi-detector fusion |
| Thresholds | `thresholds.adaptive` | Adaptive threshold calculation |
| Transients | `regimes.transient_detection` | Transient state detection |
| Drift | `drift` | Concept drift monitoring |
| Persistence | `persist`, `persist.write_scores`, etc. | Write results to SQL |
| Analytics | `outputs.comprehensive_analytics` | Generate analytics tables |
| Forecasting | `outputs.forecasting` | RUL/health forecasting |
| SQL Writes | `sql.batch_writes`, `sql.scores.write`, etc. | Batched SQL operations |

**Observations**:
- ✅ Comprehensive coverage of major phases
- ⚠️ Inconsistent hierarchy: some use dots (e.g., `features.build`), some don't (e.g., `startup`)
- ⚠️ Deep nesting in some areas (e.g., `data.guardrails.data_quality`) but flat in others
- ⚠️ No explicit span kinds - all default to `INTERNAL`

#### C. Specialized Spans

| Span | Module | Purpose |
|------|--------|---------|
| `fusion.tune_weights` | `core/fuse.py` | Auto-tune detector weights |
| `fusion.combine` | `core/fuse.py` | Combine detector scores |
| `features.compute` | `core/fast_features.py` | Fast feature computation |
| `regimes.fit` | `core/regimes.py` | Regime clustering |

---

### 2. Current Span Kind Mapping

From `core/observability.py` line 894-919:

```python
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
```

**Span Kind Logic**:
- Span kind is determined by the **first part** of hierarchical name (before first dot)
- Example: `fit.pca` → prefix is `fit` → maps to `INTERNAL` (green)
- Example: `sql.batch_writes` → prefix is `sql` → maps to `CLIENT` (blue)

**Current Color Distribution**:
- 🔵 **Blue (CLIENT)**: 10 prefixes - I/O operations
- 🟢 **Green (INTERNAL)**: 40+ prefixes - Core algorithms (most common)
- 🟣 **Purple (SERVER)**: 3 prefixes - Entry points
- 🟠 **Orange (PRODUCER)**: 2 prefixes - Data preparation

**Problem**: Most spans are green (INTERNAL), making it hard to visually distinguish phases.

---

### 3. Correlation Mechanisms

#### A. Trace-to-Logs Correlation ✅

**Status**: FULLY IMPLEMENTED

**Mechanism** (from `core/observability.py:1717-1729`):
```python
# Loki labels automatically include trace context
if OTEL_AVAILABLE and otel_trace is not None:
    current_span = otel_trace.get_current_span()
    if current_span and current_span.is_recording():
        span_ctx = current_span.get_span_context()
        if span_ctx and span_ctx.trace_id:
            labels["trace_id"] = format(span_ctx.trace_id, '032x')
        if span_ctx and span_ctx.span_id:
            labels["span_id"] = format(span_ctx.span_id, '016x')
```

**Labels Added to Loki**:
- `trace_id`: 32-character hex string (Tempo format)
- `span_id`: 16-character hex string

**Query Example**:
```logql
{app="acm", trace_id="abc123..."} | json
```

**Grafana Integration**:
- Tempo → Logs: Click span, see "Logs for this span"
- Loki → Traces: Click log, see "View trace"

---

#### B. Trace-to-Profiles Correlation ✅

**Status**: FULLY IMPLEMENTED

**Mechanism** (from `core/observability.py:1030-1034`):
```python
# Set trace context in Pyroscope for profile-to-trace correlation
if _pyroscope_pusher is not None:
    span_context = self._span.get_span_context()
    if span_context.is_valid:
        trace_id = format(span_context.trace_id, '032x')
        span_id = format(span_context.span_id, '016x')
        _pyroscope_pusher.set_trace_context(trace_id, span_id)
```

**Labels Added to Pyroscope**:
- `trace_id`: Active trace ID (when profiling inside a span)
- `span_id`: Active span ID
- `service_name`: "acm-pipeline" (standard Grafana label)
- `equipment`: Equipment name
- `equip_id`: Equipment database ID
- `run_id`: Run identifier

**Query Example**:
```promql
{service_name="acm-pipeline", trace_id="abc123..."}
```

**Grafana Integration**:
- Tempo → Profiles: Click span, see "Profiles" link (tracesToProfiles config)
- Pyroscope → Traces: Click profile sample, see trace

---

#### C. Trace-to-Metrics Correlation ✅

**Status**: FULLY IMPLEMENTED

**Mechanism**:
- All metrics include `equipment`, `run_id` labels
- Span attributes include `acm.equipment`, `acm.run_id`
- Metrics are recorded during span execution

**Example Metrics with Correlation Labels**:
```python
# From observability.py:1069-1076
_metrics["stage_duration"].record(elapsed, {
    "stage": self.name,        # Full hierarchical name
    "parent": parent,           # Top-level category
    "equipment": equipment,     # Correlation key
    "run_id": run_id           # Correlation key
})
```

**Query Example** (Prometheus):
```promql
acm_stage_duration_seconds{equipment="FD_FAN", run_id="abc-123"}
```

**Grafana Integration**:
- Tempo → Metrics: Manual correlation via `equipment` or `run_id` in Prometheus queries
- Dashboard panels can filter by `equipment` variable

---

#### D. Service Name Consistency ✅

**Status**: FULLY CONSISTENT

**Service Name**: `"acm-pipeline"` everywhere

**Usage**:
1. **Traces** (OTLP resource): `Resource.create({SERVICE_NAME: service_name})`
2. **Metrics** (OTLP resource): Same resource object
3. **Logs** (Loki labels): `{"service": service_name}`
4. **Profiles** (Pyroscope tags): `{"service_name": service_name}`

**Standard Label**: `service_name` is used for Grafana's `tracesToProfiles` feature.

---

### 4. Gap Analysis

#### A. Missing Spans for Key Pipeline Phases

| Phase | Current Coverage | Gap |
|-------|------------------|-----|
| SQL Connection | ❌ No span | Need `sql.connect` span |
| Config Loading | ⚠️ Via Timer only | Need explicit `config.load` span |
| Data Validation | ⚠️ Via Timer only | Need `data.validate` span |
| Episode Detection | ❌ No span | Need `fusion.episodes` span |
| Threshold Calculation | ⚠️ Via Timer only | Need `thresholds.calculate` span |
| Model Persistence (Save) | ⚠️ Via Timer only | Need `models.save` span |
| Forecast Engine | ❌ No span in forecast_engine.py | Need spans in ForecastEngine |

**Recommendation**: Add explicit `Span` wrappers in these modules, not just Timer sections.

---

#### B. Inconsistent Timer vs Span Naming

**Example 1**: Features
- Timer: `features.build`, `features.compute_train`, `features.compute_score`
- Span: `features.compute` (in fast_features.py)
- **Issue**: Timer uses different names than actual Span

**Example 2**: Model Fitting
- Timer: `train.detector_fit` → `fit.{detector}`
- Span: `fit.{detector}` (in detector classes)
- **Issue**: Timer adds extra `train.` prefix, Span doesn't

**Example 3**: SQL Operations
- Timer: `sql.batch_writes`, `sql.scores.write`
- Span: None (no explicit Span in output_manager.py)
- **Issue**: Timer has spans, but no actual Span instrumentation

**Recommendation**: Make Timer sections match Span names, or add Spans where Timer exists.

---

#### C. Span Attribute Completeness

**Current Attributes**:
```python
# From Span.__enter__ (observability.py:1014-1026)
self._span.set_attribute("acm.equipment", _config.equipment)
self._span.set_attribute("acm.equip_id", _config.equip_id)
self._span.set_attribute("acm.run_id", _config.run_id)
self._span.set_attribute("acm.category", self.name.split(".")[0])
# Custom attributes from caller
self._span.set_attribute(f"acm.{key}", value)
```

**Missing Attributes** (for better filtering):
- `acm.batch_num`: Current batch number (for batch runs)
- `acm.batch_total`: Total batches (for progress tracking)
- `acm.phase`: High-level phase (startup, fit, score, persist, forecast)
- `acm.outcome`: Result status (success, fail, noop)

---

## Proposed Improvements

### 1. Enhanced Span Kind Mapping (Color-Coding Strategy)

**Goal**: Use all 5 span kinds to visually distinguish pipeline phases in Tempo.

#### Proposed Mapping

```python
# New mapping with explicit color strategy
_SPAN_KIND_MAP = {
    # 🔵 CLIENT (blue): External I/O - data in/out
    "load": "CLIENT",
    "sql": "CLIENT",
    "persist": "CLIENT",
    "write": "CLIENT",
    "read": "CLIENT",
    "fetch": "CLIENT",
    
    # 🟢 INTERNAL (green): Core algorithms - processing
    "fit": "INTERNAL",
    "score": "INTERNAL",
    "compute": "INTERNAL",
    "calibrate": "INTERNAL",
    "regimes": "INTERNAL",
    "drift": "INTERNAL",
    "hash": "INTERNAL",
    "normalize": "INTERNAL",
    "impute": "INTERNAL",
    
    # 🟣 SERVER (purple): High-level orchestration - entry/exit
    "startup": "SERVER",
    "outputs": "SERVER",
    "finalize": "SERVER",
    "shutdown": "SERVER",
    "pipeline": "SERVER",
    
    # 🟠 PRODUCER (orange): Data generation - creation
    "features": "PRODUCER",
    "baseline": "PRODUCER",
    "data": "PRODUCER",
    "forecast": "PRODUCER",
    "analytics": "PRODUCER",
    
    # 🟡 CONSUMER (yellow): Aggregation/fusion - consumption
    "fusion": "CONSUMER",
    "thresholds": "CONSUMER",
    "episodes": "CONSUMER",
    "culprits": "CONSUMER",
}
```

**Color Distribution** (after changes):
- 🔵 **Blue (CLIENT)**: 6 prefixes - All I/O operations (20% of spans)
- 🟢 **Green (INTERNAL)**: 9 prefixes - Core algorithms (30% of spans)
- 🟣 **Purple (SERVER)**: 5 prefixes - Orchestration (10% of spans)
- 🟠 **Orange (PRODUCER)**: 5 prefixes - Data creation (20% of spans)
- 🟡 **Yellow (CONSUMER)**: 4 prefixes - Aggregation (20% of spans)

**Visual Impact**: Traces will show clear color-coded phases instead of mostly green.

---

### 2. Hierarchical Naming Convention

**Goal**: Consistent, filterable span names with clear parent-child relationships.

#### Naming Rules

1. **Format**: `{category}.{subcategory}.{operation}`
   - Example: `features.compute.train`
   - Example: `models.persistence.save`

2. **Maximum Depth**: 3 levels (category.subcategory.operation)
   - Too deep: `data.guardrails.data_quality.missing_values` ❌
   - Good: `data.validate.missing` ✅

3. **Category Names** (top-level):
   - `startup` - Initialization
   - `load` - Data loading
   - `features` - Feature engineering
   - `models` - Model management
   - `fit` - Model training
   - `score` - Model scoring
   - `regimes` - Regime clustering
   - `calibrate` - Calibration
   - `fusion` - Multi-detector fusion
   - `thresholds` - Threshold calculation
   - `episodes` - Episode detection
   - `drift` - Drift monitoring
   - `forecast` - RUL/health forecasting
   - `analytics` - Analytics generation
   - `persist` - Data persistence
   - `finalize` - Cleanup/shutdown

4. **Suffix Conventions**:
   - `.train` / `.score` - Training vs scoring phases
   - `.load` / `.save` - I/O operations
   - `.compute` / `.calculate` - Computation
   - `.validate` / `.check` - Validation

#### Example Refactoring

**Before**:
```python
# Inconsistent naming
with T.section("train.detector_fit"):
    with T.section("fit.ar1"): ...
    with T.section("fit.pca"): ...
with T.section("score.detector_score"):
    with T.section("score.ar1"): ...
with T.section("outputs.comprehensive_analytics"): ...
```

**After**:
```python
# Consistent hierarchical naming
with Span("models.fit"):
    with Span("fit.ar1"): ...
    with Span("fit.pca"): ...
with Span("models.score"):
    with Span("score.ar1"): ...
with Span("analytics.generate"): ...
```

---

### 3. Standard Span Attributes

**Goal**: Consistent attributes for filtering and correlation.

#### Required Attributes (added automatically by Span class)

```python
# Always present
"acm.equipment": str       # Equipment name
"acm.equip_id": int        # Equipment database ID
"acm.run_id": str          # Run identifier (UUID)
"acm.category": str        # Top-level category (from span name)
"acm.service": str         # Always "acm-pipeline"

# Phase tracking (NEW)
"acm.phase": str           # High-level phase (startup/fit/score/persist/forecast)
"acm.batch_num": int       # Batch number (for batch runs)
"acm.batch_total": int     # Total batches
```

#### Optional Attributes (caller-provided)

```python
# Data attributes
"acm.n_samples": int       # Number of samples
"acm.n_features": int      # Number of features
"acm.n_detectors": int     # Number of detectors

# Model attributes
"acm.detector": str        # Detector name (ar1, pca, iforest, gmm, omr)
"acm.model_version": int   # Model version from ModelRegistry

# Result attributes
"acm.outcome": str         # success, fail, noop, skip
"acm.error_type": str      # Exception class name (if failed)

# Resource attributes (auto-captured if track_resources=True)
"acm.mem_mb": float        # Memory at end (MB)
"acm.mem_delta_mb": float  # Memory change (MB)
"acm.cpu_pct": float       # CPU usage (%)
"acm.duration_s": float    # Duration (seconds)
```

---

### 4. Span Instrumentation Gaps - Fill Plan

#### A. Add Spans to Forecast Engine

**File**: `core/forecast_engine.py`

```python
# Add spans around key methods
class ForecastEngine:
    def run(self, ...):
        with Span("forecast.run", n_samples=len(health_df)):
            ...
    
    def _fit_degradation(self, ...):
        with Span("forecast.fit_degradation", n_samples=len(health_df)):
            ...
    
    def _estimate_rul(self, ...):
        with Span("forecast.estimate_rul"):
            ...
```

#### B. Add Spans to Output Manager

**File**: `core/output_manager.py`

```python
# Add spans for SQL operations
def _execute_insert(self, table: str, df: pd.DataFrame):
    with Span("sql.insert", table=table, n_rows=len(df)):
        ...

def _execute_bulk_insert(self, operations: List[...]):
    with Span("sql.bulk_insert", n_operations=len(operations)):
        ...
```

#### C. Add Spans to Episode Detection

**File**: `core/fuse.py`

```python
# Add span for episode detection
def detect_episodes(...):
    with Span("fusion.detect_episodes", n_samples=len(scores_wide)):
        ...
```

#### D. Add Top-Level Pipeline Span

**File**: `core/acm_main.py`

```python
# Wrap entire main() in a top-level span
def main(args):
    with Span("pipeline.run", equipment=args.equip):
        # All pipeline logic
        ...
```

---

### 5. Correlation Enhancements

#### A. Grafana Dashboard Links

**File**: `install/observability/dashboards/acm_observability.json`

Add exemplar links:
```json
{
  "datasource": "Tempo",
  "expr": "{service_name=\"acm-pipeline\"}",
  "traceQuery": {
    "query": "resource.service.name=\"acm-pipeline\" && span.attributes.acm.equipment=\"${equipment}\""
  }
}
```

#### B. Loki to Tempo Link Configuration

**File**: `install/observability/provisioning/datasources/datasources.yaml`

```yaml
datasources:
  - name: Loki
    type: loki
    jsonData:
      derivedFields:
        - name: TraceID
          matcherRegex: '"trace_id":"([0-9a-f]{32})"'
          url: '$${__value.raw}'
          datasourceUid: tempo
```

#### C. Tempo to Pyroscope Link

**File**: `install/observability/provisioning/datasources/datasources.yaml`

```yaml
datasources:
  - name: Tempo
    type: tempo
    jsonData:
      tracesToProfiles:
        datasourceUid: pyroscope
        tags:
          - key: service_name
            value: service_name
          - key: trace_id
            value: trace_id
```

---

## Implementation Roadmap

### Phase 1: Foundation (Low Risk)

1. **Update `_SPAN_KIND_MAP`** in `core/observability.py`
   - Add CONSUMER kind
   - Reorganize categories for better color distribution
   - **Files**: `core/observability.py`
   - **Risk**: Low (only affects visualization)

2. **Add Standard Attributes** to Span class
   - Add `acm.phase`, `acm.batch_num`, `acm.batch_total`
   - **Files**: `core/observability.py`
   - **Risk**: Low (additive change)

3. **Update Documentation**
   - Document new naming convention
   - Update OBSERVABILITY.md with span examples
   - **Files**: `docs/OBSERVABILITY.md`
   - **Risk**: None

### Phase 2: Instrumentation (Medium Risk)

4. **Add Spans to Forecast Engine**
   - `forecast.run`, `forecast.fit_degradation`, `forecast.estimate_rul`
   - **Files**: `core/forecast_engine.py`
   - **Risk**: Medium (new instrumentation)

5. **Add Spans to Output Manager**
   - `sql.insert`, `sql.bulk_insert`, `sql.batch_commit`
   - **Files**: `core/output_manager.py`
   - **Risk**: Medium (affects SQL timing)

6. **Add Spans to Fusion**
   - `fusion.detect_episodes`, `fusion.cusum`
   - **Files**: `core/fuse.py`
   - **Risk**: Low (isolated)

### Phase 3: Refactoring (Higher Risk)

7. **Align Timer and Span Names** in `acm_main.py`
   - Rename Timer sections to match proposed convention
   - **Files**: `core/acm_main.py`
   - **Risk**: High (touches main pipeline)

8. **Add Top-Level Pipeline Span**
   - Wrap `main()` in `pipeline.run` span
   - **Files**: `core/acm_main.py`
   - **Risk**: Medium (affects all traces)

### Phase 4: Grafana Integration

9. **Update Grafana Datasource Links**
   - Configure derived fields for trace correlation
   - **Files**: `install/observability/provisioning/datasources/datasources.yaml`
   - **Risk**: Low (Grafana config only)

10. **Update Dashboards**
    - Add trace panels to ACM Observability dashboard
    - Add exemplar links from metrics to traces
    - **Files**: `install/observability/dashboards/*.json`
    - **Risk**: Low (visualization only)

---

## Testing Plan

### 1. Unit Tests

**File**: `tests/test_observability_spans.py` (new)

```python
def test_span_kind_mapping():
    """Verify span kinds map correctly to colors."""
    assert get_span_kind("fit.pca") == SpanKind.INTERNAL
    assert get_span_kind("sql.insert") == SpanKind.CLIENT
    assert get_span_kind("fusion.combine") == SpanKind.CONSUMER
    assert get_span_kind("features.compute") == SpanKind.PRODUCER
    assert get_span_kind("startup") == SpanKind.SERVER

def test_span_attributes():
    """Verify required attributes are set."""
    with Span("test.span") as span:
        assert span._span.attributes["acm.category"] == "test"
        assert span._span.attributes["acm.service"] == "acm-pipeline"
```

### 2. Integration Tests

**Test Scenario**: Run batch job and verify trace hierarchy

```python
def test_trace_hierarchy():
    """Verify parent-child span relationships."""
    # Run single batch
    run_acm_batch(equipment="TEST_EQUIP", tick_minutes=1440)
    
    # Query Tempo for traces
    traces = tempo_client.query(service_name="acm-pipeline")
    
    # Verify hierarchy
    assert traces[0].root_span.name == "pipeline.run"
    assert "load.data" in [s.name for s in traces[0].spans]
    assert "fit.pca" in [s.name for s in traces[0].spans]
```

### 3. Correlation Tests

**Test Scenario**: Verify log-to-trace correlation

```python
def test_log_trace_correlation():
    """Verify logs contain trace_id and span_id."""
    with Span("test.correlation") as span:
        Console.info("Test message", component="TEST")
        
    # Query Loki for the log
    logs = loki_client.query('{component="test"}')
    
    # Verify trace context
    assert logs[0].labels["trace_id"] == span.trace_id
    assert logs[0].labels["span_id"] == span.span_id
```

### 4. Manual Verification (Grafana)

1. **Start observability stack**:
   ```powershell
   cd install/observability
   docker compose up -d
   ```

2. **Run ACM batch**:
   ```powershell
   python scripts/sql_batch_runner.py --equip TEST_EQUIP --tick-minutes 1440 --max-ticks 1
   ```

3. **Open Grafana Tempo** (http://localhost:3000/explore → Tempo):
   - Query: `{service.name="acm-pipeline"}`
   - Verify spans are color-coded by phase
   - Click on span → verify attributes
   - Click "Logs for this span" → verify Loki correlation
   - Click "Profiles" → verify Pyroscope correlation

4. **Open Grafana Loki** (http://localhost:3000/explore → Loki):
   - Query: `{app="acm"} | json`
   - Click on log → verify "View trace" link works

5. **Open Grafana Pyroscope** (http://localhost:3000/explore → Pyroscope):
   - Select profile type: `process_cpu:cpu:nanoseconds`
   - Filter: `{service_name="acm-pipeline"}`
   - Verify trace_id label is present

---

## Expected Outcomes

### Before (Current State)

**Tempo Trace View**:
- Mostly green spans (INTERNAL)
- Hard to distinguish phases
- Some gaps in coverage

**Example Trace**:
```
┌─ startup (purple)
├─ load_data (blue)
├─ fit.ar1 (green)
├─ fit.pca (green)
├─ fit.iforest (green)
├─ fit.gmm (green)
├─ fit.omr (green)
├─ score.ar1 (green)
├─ score.pca (green)
├─ (many more green spans)
└─ outputs (purple)
```

**Issues**:
- All algorithm spans are green → hard to see structure
- No clear separation between fit, score, fusion phases
- Missing spans for forecast, analytics

### After (Improved State)

**Tempo Trace View**:
- Color-coded by phase type
- Clear visual hierarchy
- Complete coverage

**Example Trace**:
```
┌─ pipeline.run (purple - SERVER)
  ├─ startup (purple - SERVER)
  ├─ load.data (blue - CLIENT)
  ├─ features.compute (orange - PRODUCER)
  ├─ models.fit (purple - SERVER)
  │  ├─ fit.ar1 (green - INTERNAL)
  │  ├─ fit.pca (green - INTERNAL)
  │  ├─ fit.iforest (green - INTERNAL)
  │  ├─ fit.gmm (green - INTERNAL)
  │  └─ fit.omr (green - INTERNAL)
  ├─ models.score (purple - SERVER)
  │  ├─ score.ar1 (green - INTERNAL)
  │  ├─ score.pca (green - INTERNAL)
  │  ├─ score.iforest (green - INTERNAL)
  │  ├─ score.gmm (green - INTERNAL)
  │  └─ score.omr (green - INTERNAL)
  ├─ regimes.fit (green - INTERNAL)
  ├─ fusion.combine (yellow - CONSUMER)
  ├─ fusion.detect_episodes (yellow - CONSUMER)
  ├─ thresholds.calculate (yellow - CONSUMER)
  ├─ forecast.run (orange - PRODUCER)
  │  ├─ forecast.fit_degradation (green - INTERNAL)
  │  └─ forecast.estimate_rul (green - INTERNAL)
  ├─ analytics.generate (orange - PRODUCER)
  ├─ persist.scores (blue - CLIENT)
  ├─ persist.episodes (blue - CLIENT)
  └─ finalize (purple - SERVER)
```

**Benefits**:
- 🔵 Blue (CLIENT): All I/O operations - easy to identify bottlenecks
- 🟢 Green (INTERNAL): Algorithm execution - performance tuning
- 🟣 Purple (SERVER): Orchestration - overall flow
- 🟠 Orange (PRODUCER): Data generation - feature/forecast creation
- 🟡 Yellow (CONSUMER): Aggregation - fusion/threshold logic

---

## Correlation Examples

### 1. Trace → Logs

**Scenario**: Find logs for a specific fit.pca span

1. Open Tempo, find trace with slow `fit.pca` span
2. Click span → "Logs for this span"
3. Grafana opens Loki with query:
   ```logql
   {app="acm", trace_id="<trace_id>", span_id="<span_id>"}
   ```
4. See all logs emitted during that span

### 2. Logs → Trace

**Scenario**: Find trace for an error log

1. Open Loki, search for errors:
   ```logql
   {app="acm"} |= "ERROR" | json
   ```
2. Click log line → "View trace"
3. Grafana opens Tempo with query:
   ```
   <trace_id from log>
   ```
4. See full trace context around the error

### 3. Trace → Profiles

**Scenario**: Find CPU hotspot for slow fit.gmm

1. Open Tempo, find trace with slow `fit.gmm` span
2. Click span → "Profiles"
3. Grafana opens Pyroscope filtered to:
   ```promql
   {service_name="acm-pipeline", trace_id="<trace_id>"}
   ```
4. See CPU flamegraph for that exact time range

### 4. Metrics → Traces

**Scenario**: Investigate high `acm_stage_duration_seconds{stage="fit.pca"}`

1. Open Prometheus/Metrics dashboard
2. See spike in `acm_stage_duration_seconds{stage="fit.pca"}`
3. Click exemplar (if enabled) or manually query Tempo:
   ```
   {service.name="acm-pipeline"} && span.attributes.acm.category="fit" && span.name="fit.pca"
   ```
4. Find traces during spike, analyze what went wrong

---

## Appendix: Complete Span Inventory

### Current Span Count

**Total Unique Span Names**: ~60

**By Category**:
- Detector fit/score: 10 (ar1, pca, iforest, gmm, omr x 2)
- Features: 8
- Models: 12
- Fusion: 3
- SQL/Persistence: 15
- Analytics/Outputs: 8
- Regimes: 4

### Proposed New Spans

**To Add**: ~15 new spans

1. `pipeline.run` - Top-level span
2. `config.load` - Config loading
3. `sql.connect` - SQL connection
4. `data.validate` - Data validation
5. `forecast.run` - Forecast orchestrator
6. `forecast.fit_degradation` - Degradation model
7. `forecast.estimate_rul` - RUL estimation
8. `fusion.detect_episodes` - Episode detection
9. `thresholds.calculate` - Adaptive thresholds
10. `sql.insert` - Single insert
11. `sql.bulk_insert` - Bulk insert
12. `sql.batch_commit` - Batch commit
13. `analytics.generate` - Analytics tables
14. `models.save` - Model persistence
15. `finalize` - Cleanup/shutdown

**After Implementation**: ~75 total spans

---

## Summary

### Key Improvements

1. **Color Diversity**: 5 span kinds (currently mostly green) → Visual distinction
2. **Complete Coverage**: +15 new spans → No gaps in tracing
3. **Consistent Naming**: Hierarchical convention → Easy filtering
4. **Rich Attributes**: Standard + custom attributes → Powerful queries
5. **Full Correlation**: Logs ↔ Traces ↔ Metrics ↔ Profiles → Unified observability

### Migration Risk

**Low Risk** (Phases 1-2):
- Span kind mapping changes (visualization only)
- New attributes (additive)
- New spans in isolated modules (forecast, output_manager, fuse)

**Medium Risk** (Phase 3):
- Renaming Timer sections (affects metric names)
- Top-level pipeline span (changes trace structure)

**Mitigation**:
- Test in dev environment first
- Gradual rollout (phase by phase)
- Monitor for regressions in Grafana

### Success Metrics

- ✅ Traces show 5 distinct colors (not just green)
- ✅ All major pipeline phases have spans
- ✅ Clicking span in Tempo → shows logs
- ✅ Clicking log in Loki → shows trace
- ✅ Clicking span in Tempo → shows profiles
- ✅ Metrics dashboard links to traces
- ✅ Documentation updated and accurate

---

**Next Steps**: Review this document, prioritize phases, begin implementation.
