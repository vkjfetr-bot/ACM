# ACM - Automated Condition Monitoring (v11.1.5)

---

## ⛔ RULE #1: NEVER FILTER CONSOLE OUTPUT (NON-VIOLATABLE)

**THIS RULE CANNOT BE VIOLATED UNDER ANY CIRCUMSTANCES:**

When running ANY terminal command (ACM, Python scripts, SQL queries, etc.):
- **NEVER use `Select-Object -First N` or `-Last N`** to limit output
- **NEVER use `| head`, `| tail`, or any output truncation**
- **NEVER use `Out-String -Width` with small values**
- **ALWAYS show the COMPLETE, UNFILTERED output**
- **If output is long, that's OK - show ALL of it**

The user MUST see every single line of output. Filtering output hides critical errors, warnings, and diagnostic information.

**VIOLATION OF THIS RULE IS GROUNDS FOR IMMEDIATE TERMINATION OF THE CONVERSATION.**

---

ACM is a predictive maintenance and equipment health monitoring system. It ingests sensor data from industrial equipment (FD_FAN, GAS_TURBINE, etc.) via SQL Server, runs anomaly detection algorithms, calculates health scores, and forecasts Remaining Useful Life (RUL). Results are visualized through Grafana dashboards for operations teams.

**Key Features**: Multi-detector fusion (AR1, PCA, IForest, GMM, OMR), regime detection, episode diagnostics, RUL forecasting with Monte Carlo simulations, SQL-only persistence, full observability stack (OpenTelemetry traces/metrics, Loki logs, Pyroscope profiling).

**v11.0.0 New Features**:
- ONLINE/OFFLINE pipeline mode separation (--mode auto/online/offline)
- MaturityState lifecycle (COLDSTART -> LEARNING -> CONVERGED -> DEPRECATED)
- Unified confidence model with ReliabilityStatus for all outputs
- RUL reliability gating (NOT_RELIABLE when model not CONVERGED)
- UNKNOWN regime (label=-1) for low-confidence assignments
- Confidence columns in ACM_RUL, ACM_HealthTimeline, ACM_Anomaly_Events
- Promotion criteria (7 days, 0.15 silhouette, 3 runs) for model maturity

---

## Tech Stack

### Backend
- **Python 3.11** - Core runtime
- **pandas/NumPy** - Vectorized data processing
- **scikit-learn** - ML detectors (PCA, IForest, GMM)
- **Optional**: Polars (fast_features.py), Rust bridge (rust_bridge/)

### Database
- **Microsoft SQL Server** - Primary data store (historian data, ACM results)
- **pyodbc** - SQL connectivity via `core/sql_client.py`
- **T-SQL** - Stored procedures, queries (NEVER use generic SQL syntax)
- **Schema Design (v11.1.5):**
  - All 92 ACM tables have IDENTITY columns (ID BIGINT)
  - Relationship columns (EquipID, RunID, EpisodeID) are NOT enforced via FK constraints
  - `ACM_Runs.RunID` is the parent reference for all run-scoped data
  - `Equipment.EquipID` is the parent reference for all equipment data
  - Run `scripts/sql/truncate_run_data.sql` to clear all run-generated data

### Visualization
- **Grafana** - Dashboards in `grafana_dashboards/*.json`
- **MS SQL datasource** - Direct SQL queries with `$__timeFrom()`, `$__timeTo()` macros

### Observability Stack (v11.0.0 - Docker-based)
- **Docker Compose** - Complete stack in `install/observability/docker-compose.yaml`
- **Grafana** - Dashboard UI on port 3000 (admin/admin), auto-provisioned datasources
- **Grafana Alloy** - OTLP collector on ports 4317 (gRPC), 4318 (HTTP)
- **OpenTelemetry** - Distributed tracing (Tempo) and metrics (Prometheus)
- **Loki** - Structured log aggregation on port 3100
- **Pyroscope** - Continuous profiling on port 4040 (requires `pip install yappi`)
- **Unified API** - `core/observability.py` provides Console, Span, Metrics classes
- **Dashboards** - `install/observability/dashboards/` (auto-provisioned to Grafana)

### Configuration
- **configs/config_table.csv** - Equipment-specific settings (cascading: `*` global, then equipment rows)
- **configs/sql_connection.ini** - SQL Server credentials (gitignored, local-only)

---

## Project Structure

```
ACM/
+-- .github/              # Copilot instructions (this file)
+-- configs/              # config_table.csv, sql_connection.ini
+-- core/                 # Main codebase (see Module Relationships below)
|   +-- acm_main.py       # Pipeline orchestrator (6000+ lines)
|   +-- output_manager.py # All SQL writes (respects ALLOWED_TABLES)
|   +-- sql_client.py     # SQL Server connectivity
|   +-- observability.py  # Console, Span, Metrics classes
|   +-- fast_features.py  # Feature engineering (pandas/Polars)
|   +-- fuse.py           # Multi-detector fusion
|   +-- regimes.py        # Operating regime detection
|   +-- forecast_engine.py # RUL and health forecasting
|   +-- model_persistence.py  # SQL ModelRegistry storage
|   +-- model_lifecycle.py    # MaturityState, PromotionCriteria
|   +-- confidence.py         # Unified confidence model
|   +-- pipeline_types.py     # DataContract, PipelineMode
|   +-- seasonality.py        # Diurnal/weekly pattern detection
+-- scripts/              # Operational scripts
|   +-- sql_batch_runner.py   # Production batch processing
|   +-- sql/              # SQL utilities, schema export
+-- docs/                 # All documentation (NEVER create docs in root)
|   +-- archive/          # Historical/archived docs
|   +-- sql/              # COMPREHENSIVE_SCHEMA_REFERENCE.md
+-- grafana_dashboards/   # Grafana JSON dashboards
+-- tests/                # pytest test suites
+-- utils/                # config_dict.py, version.py
+-- artifacts/            # gitignored - runtime outputs
+-- data/                 # gitignored - sample CSVs for file-mode
+-- logs/                 # gitignored - runtime logs
```

---

## Script Relationships & Entry Points

### Entry Points Hierarchy

```
1. scripts/sql_batch_runner.py (PRODUCTION - Primary entry point)
   |
   +-> core/acm_main.py::run_acm() (called via subprocess)
       |
       +-> All pipeline phases (see Pipeline Phase Sequence below)

2. python -m core.acm_main --equip EQUIPMENT (TESTING - Single run)
   |
   +-> core/acm_main.py::run_acm() (direct call)
       |
       +-> All pipeline phases

3. core/acm.py (ALTERNATIVE - Mode-aware router)
   |
   +-> Parses --mode (auto/online/offline)
   +-> Detects mode based on cached models if auto
   +-> Calls core/acm_main.py::run_acm() with mode
```

### Script Relationships

```
sql_batch_runner.py
├── Purpose: Continuous batch processing, coldstart management, multi-equipment
├── Calls: core/acm_main.py via subprocess (python -m core.acm_main)
├── Manages: Coldstart state, batch windows, resume from last run
├── SQL Tables: Reads ACM_ColdstartState, writes ACM_Runs
└── Arguments:
    --equip FD_FAN GAS_TURBINE  # Multiple equipment
    --tick-minutes 1440          # Batch window size
    --max-workers 2              # Parallel equipment processing
    --start-from-beginning       # Full reset (coldstart)
    --resume                     # Continue from last run
    --max-batches 1              # Limit batches (testing)

core/acm_main.py
├── Purpose: Single pipeline run (train/score/forecast)
├── Imports: All core modules (see Module Call Sequence)
├── Manages: Model training, scoring, persistence
└── Arguments:
    --equip FD_FAN               # Single equipment
    --start-time "2024-01-01T00:00:00"
    --end-time "2024-01-31T23:59:59"
    --mode offline|online|auto   # Pipeline mode

scripts/sql/verify_acm_connection.py
├── Purpose: Test SQL Server connectivity
├── Calls: core/sql_client.SQLClient
└── Output: Connection test result

scripts/sql/export_comprehensive_schema.py
├── Purpose: Export SQL schema to markdown
├── Calls: SQL INFORMATION_SCHEMA
└── Output: docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md

scripts/sql/populate_acm_config.py
├── Purpose: Sync config_table.csv to SQL ACM_Config
├── Reads: configs/config_table.csv
└── Writes: SQL ACM_Config table
```

---

## Pipeline Phase Sequence (acm_main.py)

The main pipeline executes in this order. Each phase corresponds to a timed section in the output:

```
PHASE 1: INITIALIZATION (startup)
├── Parse CLI arguments (--equip, --start-time, --end-time, --mode)
├── Load config from SQL (ConfigDict)
├── Determine PipelineMode (ONLINE/OFFLINE/AUTO)
├── Initialize OutputManager with SQL client
└── Create RunID for this execution

PHASE 2: DATA CONTRACT VALIDATION (data.contract)
├── DataContract.validate(raw_data)
├── Check sensor coverage (min 70% required)
├── Write ACM_DataContractValidation
└── Fail fast if validation fails

PHASE 3: DATA LOADING (load_data)
├── Load historian data from SQL (stored procedure)
├── Apply coldstart split (60% train / 40% score)
├── Validate timestamp column and cadence
└── Output: train DataFrame, score DataFrame

PHASE 4: BASELINE SEEDING (baseline.seed)
├── Load baseline from ACM_BaselineBuffer
├── Check for overlap with score data
└── Apply baseline for normalization

PHASE 5: SEASONALITY DETECTION (seasonality.detect)
├── SeasonalityHandler.detect_patterns()
├── Detect DAILY/WEEKLY cycles using FFT
├── Apply seasonal adjustment if enabled (v11)
└── Write ACM_SeasonalPatterns

PHASE 6: DATA QUALITY GUARDRAILS (data.guardrails)
├── Check train/score overlap
├── Validate variance and coverage
├── Write ACM_DataQuality
└── Output quality metrics

PHASE 7: FEATURE ENGINEERING (features.build + features.impute)
├── fast_features.compute_all_features()
├── Build rolling stats, lag features, z-scores
├── Impute missing values from train medians
├── Compute feature hash for caching
└── Output: Feature matrices (train_features, score_features)

PHASE 8: MODEL LOADING/TRAINING (train.detector_fit)
├── Check for cached models in ModelRegistry
├── If OFFLINE or models missing:
│   ├── Fit AR1 detector (ar1_detector.py)
│   ├── Fit PCA detector (pca via sklearn)
│   ├── Fit IForest detector (sklearn.ensemble)
│   ├── Fit GMM detector (sklearn.mixture)
│   └── Fit OMR detector (omr.py)
├── If ONLINE: Load all detectors from cache
└── Output: Trained detector objects

PHASE 10: DETECTOR SCORING (score.detector_score)
├── Score all detectors on score data
├── Compute z-scores per detector
├── Output: scores_wide DataFrame with detector columns
└── Columns: ar1_z, pca_spe_z, pca_t2_z, iforest_z, gmm_z, omr_z

PHASE 11: REGIME LABELING (regimes.label)
├── regimes.label() with regime context
├── Auto-k selection (silhouette score)
├── K-Means clustering on raw sensor values
├── UNKNOWN regime (-1) for low-confidence assignments
├── Write ACM_RegimeDefinitions
└── Output: Regime labels per row

PHASE 12: MODEL PERSISTENCE (models.persistence.save)
├── Save all models to SQL ModelRegistry
├── Increment model version
└── Write metadata to ACM_ModelHistory

PHASE 13: MODEL LIFECYCLE (v11)
├── load_model_state_from_sql()
├── Update model state with run metrics
├── Check promotion criteria (LEARNING -> CONVERGED)
├── Write ACM_ActiveModels
└── Output: MaturityState (COLDSTART/LEARNING/CONVERGED/DEPRECATED)

PHASE 14: CALIBRATION (calibrate)
├── Score TRAIN data for calibration baseline
├── Compute adaptive clip_z from P99
├── Self-tune thresholds for target FP rate
└── Write ACM_Thresholds

PHASE 15: DETECTOR FUSION (fusion.auto_tune + fusion)
├── Auto-tune detector weights (episode separability)
├── Compute fused_z (weighted combination)
├── CUSUM parameter tuning (k_sigma, h_sigma)
├── Detect anomaly episodes
└── Output: fused_alert, episode markers

PHASE 16: ADAPTIVE THRESHOLDS (thresholds.adaptive)
├── Calculate per-regime thresholds
├── Global thresholds: alert=3.0, warn=1.5
└── Write to SQL

PHASE 17: TRANSIENT DETECTION (regimes.transient_detection)
├── Detect state transitions (startup, trip, steady)
├── Label transient periods
└── Output: Transient state per row

PHASE 18: DRIFT MONITORING (drift)
├── Compute drift metrics (CUSUM trend)
└── Classify: STABLE, DRIFTING, FAULT

PHASE 19: OUTPUT GENERATION (persist.*)
├── write_scores_wide() -> ACM_Scores_Wide
├── write_anomaly_events() -> ACM_Anomaly_Events
├── write_detector_correlation() -> ACM_DetectorCorrelation
├── write_sensor_correlation() -> ACM_SensorCorrelations
├── write_sensor_normalized_ts() -> ACM_SensorNormalized_TS
└── write_seasonal_patterns() -> ACM_SeasonalPatterns

PHASE 20: ANALYTICS GENERATION (outputs.comprehensive_analytics)
├── _generate_health_timeline() -> ACM_HealthTimeline
├── _generate_regime_timeline() -> ACM_RegimeTimeline
├── _generate_sensor_defects() -> ACM_SensorDefects
├── _generate_sensor_hotspots() -> ACM_SensorHotspots
└── Compute confidence values (v11)

PHASE 21: FORECASTING (outputs.forecasting)
├── ForecastEngine.run_forecast()
│   ├── Load health history from ACM_HealthTimeline
│   ├── Fit degradation model (Holt-Winters)
│   ├── Generate health forecast -> ACM_HealthForecast
│   ├── Generate failure forecast -> ACM_FailureForecast
│   ├── Compute RUL with Monte Carlo -> ACM_RUL
│   ├── Compute confidence and reliability (v11)
│   └── Generate sensor forecasts -> ACM_SensorForecast
└── Write forecast tables

PHASE 22: RUN FINALIZATION (sql.run_stats)
├── Write PCA loadings -> ACM_PCA_Loadings
├── Write run statistics -> ACM_Run_Stats
├── Write run metadata -> ACM_Runs
└── Commit all pending SQL writes
```

---

## Module Dependency Graph

```
sql_batch_runner.py
    └── subprocess calls: core/acm_main.py

core/acm_main.py (MAIN ORCHESTRATOR)
    ├── utils/config_dict.py (ConfigDict)
    ├── core/sql_client.py (SQLClient)
    ├── core/output_manager.py (OutputManager)
    ├── core/observability.py (Console, Span, Metrics, T)
    ├── core/pipeline_types.py (DataContract, PipelineMode)
    ├── core/fast_features.py (compute_all_features)
    ├── core/ar1_detector.py (AR1Detector)
    ├── core/omr.py (OMRDetector)
    ├── core/regimes.py (label, detect_transient_states)
    ├── core/fuse.py (compute_fusion, detect_episodes)
    ├── core/adaptive_thresholds.py (calculate_thresholds)
    ├── core/drift.py (compute_drift_metrics)
    ├── core/model_persistence.py (save_models, load_models)
    ├── core/model_lifecycle.py (ModelState, promote_model)
    ├── core/confidence.py (compute_*_confidence)
    ├── core/seasonality.py (SeasonalityHandler)
    ├── core/forecast_engine.py (ForecastEngine)
    └── core/health_tracker.py (HealthTracker)

core/output_manager.py
    ├── core/sql_client.py (SQLClient)
    ├── core/observability.py (Console)
    └── core/confidence.py (compute_*_confidence)

core/forecast_engine.py
    ├── core/sql_client.py (SQLClient)
    ├── core/degradation_model.py (fit_degradation)
    ├── core/rul_estimator.py (estimate_rul)
    ├── core/confidence.py (compute_rul_confidence)
    ├── core/model_lifecycle.py (load_model_state_from_sql)
    └── core/health_tracker.py (HealthTracker)

core/regimes.py
    ├── sklearn.cluster (KMeans)
    ├── sklearn.metrics (silhouette_score)
    └── core/observability.py (Console)
```

---

## Coding Guidelines

### Python Style
- Python 3.11, ~100-char lines, type hints always
- Vectorized pandas/NumPy preferred over loops
- Use existing lint/type tooling (`ruff`, `mypy`)
- No emojis in code, comments, tests, or output

### SQL/T-SQL Rules
- ALWAYS use Microsoft SQL Server T-SQL syntax
- Use `sqlcmd -S "server\instance" -d database -E -Q "..."` for queries
- Use T-SQL functions: `DATEADD`, `DATEDIFF`, `TOP`, `CAST`, `ROUND`, `COALESCE`
- NEVER use: `LIMIT` (use `TOP`), `DATE_TRUNC` (use `DATEADD/DATEDIFF`), `FORMAT()` for time series
- **NEVER use reserved words as aliases**: `End`, `RowCount`, `Count`, `Date`, `Time`, `Order`, `Group`
- **Safe aliases**: `EndTimeStr`, `TotalRows`, `TotalCount`, `DateValue`, `TimeValue`, `OrderNum`, `GroupName`

### PowerShell 5.1 Rules
- Use `;` for command chaining (NOT `&&`)
- Use `Select-Object -Last 20` (NOT `tail -n 20`)
- Script params: `-Parameter value`; Python args: `--arg value`

### CRITICAL: Python Command-Line Execution in PowerShell (Windows)
**NEVER use inline Python with complex strings in PowerShell `python -c "..."` commands:**

❌ **FORBIDDEN PATTERNS:**
```powershell
# NEVER: f-strings with nested quotes break PowerShell parsing
python -c "import x; conn = x.connect(f'DRIVER={{{var}}}')"

# NEVER: Multi-line strings in -c flag
python -c "
import module
code here
"

# NEVER: Dictionary/config access in f-strings
python -c "s = config['acm']; print(f'x={s[\"key\"]}')"
```

✅ **ALWAYS USE THESE PATTERNS:**
```powershell
# ✓ Create a temporary .py script file instead
$code = @'
import module
# Multi-line code here
'@
$code | Out-File -Encoding UTF8 temp_script.py
python temp_script.py
Remove-Item temp_script.py

# ✓ Use scripts/ folder for any non-trivial Python
# Create scripts/verify_something.py, then:
python scripts/verify_something.py

# ✓ For simple one-liners, avoid f-strings entirely
python -c "import sys; print(sys.version)"
```

**WHY THIS MATTERS:**
- PowerShell interprets `{`, `}`, `$`, `"`, `'`, `\` as special characters
- F-strings with nested quotes create unescapable parsing conflicts
- Multi-line strings in `-c` fail due to PowerShell's line continuation rules
- **Solution: Always create a .py file for anything beyond trivial one-liners**

### Logging & Observability Rules (v10.3.0)
- **ALWAYS use `Console` class** from `core.observability` for all logging
- **NEVER use `print()` or legacy loggers** (`utils/logger.py`, `utils/acm_logger.py` are deleted)
- Console methods:
  - `Console.info(msg, **kwargs)` - General info (goes to Loki)
  - `Console.warn(msg, **kwargs)` - Warnings (goes to Loki)
  - `Console.error(msg, **kwargs)` - Errors (goes to Loki)
  - `Console.ok(msg, **kwargs)` - Success messages (goes to Loki)
  - `Console.status(msg)` - Console-only output (NO Loki)
  - `Console.header(title, char="=")` - Section headers (NO Loki)
  - `Console.section(title)` - Lighter separators (NO Loki)
- **Separator lines MUST use `Console.status/header/section`** to avoid leaking to Loki
- **Timer output** uses `Console.section/status` (console-only, not logged)

### Grafana Dashboard Rules
- Time series: Return raw `DATETIME` columns (not `FORMAT()` strings)
- Order: `ORDER BY time ASC` for time series
- Time filter: `WHERE Timestamp BETWEEN $__timeFrom() AND $__timeTo()`
- spanNulls: Use threshold in ms (e.g., `3600000`) not `true/false`

---

## Resources and Commands

### Primary Entry Points
```powershell
# Single equipment run
python -m core.acm_main --equip GAS_TURBINE

# Batch processing
python scripts/sql_batch_runner.py --equip FD_FAN GAS_TURBINE --tick-minutes 1440 --max-workers 2 --start-from-beginning
```

### SQL Tools
```powershell
# Verify SQL connection
python scripts/sql/verify_acm_connection.py

# Export schema (SOLE AUTHORITATIVE TOOL for SQL inspection)
python scripts/sql/export_comprehensive_schema.py --output docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md

# Sync config to SQL
python scripts/sql/populate_acm_config.py

# Create/update dashboard views (13 views for Grafana)
sqlcmd -S "localhost\INSTANCE" -d ACM -E -i "scripts/sql/create_acm_views.sql"

# Data Retention Cleanup (run regularly to prevent DB bloat)
sqlcmd -S "localhost\INSTANCE" -d ACM -E -Q "EXEC dbo.usp_ACM_DataRetention @DryRun = 1"  -- Preview
sqlcmd -S "localhost\INSTANCE" -d ACM -E -Q "EXEC dbo.usp_ACM_DataRetention @DryRun = 0"  -- Execute
```

### Dashboard Views (v11.1.5)
ACM provides 13 pre-built SQL views optimized for Grafana dashboards:

| View | Purpose | Grafana Panel Type |
|------|---------|-------------------|
| vw_ACM_CurrentHealth | Current health per equipment | Stat, Gauge |
| vw_ACM_HealthHistory | Health time series | Time series |
| vw_ACM_ActiveDefects | Active sensor defects | Table |
| vw_ACM_TopSensorContributors | Top anomaly contributors | Bar chart |
| vw_ACM_RecentEpisodes | Recent anomaly episodes | Table, Timeline |
| vw_ACM_RULSummary | Remaining useful life | Stat, Table |
| vw_ACM_DetectorScores | Detector scores time series | Time series |
| vw_ACM_RegimeAnalysis | Operating regime analysis | Pie, Bar chart |
| vw_ACM_DriftStatus | Drift detection status | Status indicators |
| vw_ACM_RunHistory | ACM run history | Table |
| vw_ACM_EquipmentOverview | Complete equipment summary | Table (main dashboard) |
| vw_ACM_SensorForecasts | Sensor value forecasts | Time series |
| vw_ACM_HealthForecasts | Health trajectory forecasts | Time series |

**Usage**: `SELECT * FROM vw_ACM_EquipmentOverview WHERE EquipCode = 'FD_FAN'`

### Data Retention Policy (v11.0.3+)
ACM implements automatic data retention to prevent unbounded database growth:

| Table | Retention | Rationale |
|-------|-----------|-----------|
| ACM_SensorNormalized_TS | 30 days | Only needed for sensor forecasting lookback |
| ACM_SensorCorrelations | Latest run/equip | Only current correlations are useful |
| ACM_Scores_Wide | 90 days | Core analytics time series |
| ACM_HealthTimeline | 90 days | Health trending dashboards |
| ACM_RegimeTimeline | 90 days | Regime analysis |
| ACM_RunTimers | 30 days | Performance metrics |
| ACM_RunLogs | 14 days | Debugging only |
| ACM_PCA_Loadings | Last 5 runs/equip | Keep recent model history |
| ACM_FeatureDropLog | 30 days | Feature engineering diagnostics |

**Schedule**: Run `usp_ACM_DataRetention` daily via SQL Agent job or scheduled task.

### Observability Stack (Docker-based)
```powershell
# Start complete observability stack (Grafana, Alloy, Tempo, Loki, Prometheus, Pyroscope)
cd install/observability; docker compose up -d

# Verify all containers are healthy
docker ps --format "table {{.Names}}\t{{.Status}}"

# Expected containers:
# acm-grafana      (port 3000) - Dashboard UI, admin/admin
# acm-alloy        (port 4317, 4318) - OTLP collector
# acm-tempo        (port 3200) - Traces
# acm-loki         (port 3100) - Logs
# acm-prometheus   (port 9090) - Metrics
# acm-pyroscope    (port 4040) - Profiling

# Access Grafana with auto-provisioned datasources and dashboards
# Open http://localhost:3000 (admin/admin)
# Dashboards in ACM folder: ACM Behavior, ACM Observability

# Stop the stack
docker compose down

# Clean restart (removes all data)
docker compose down -v; docker compose up -d
```

### Enable Profiling
```powershell
# Install yappi for CPU profiling (pure Python, no Rust required)
pip install yappi

# Verify in ACM output:
# [SUCCESS] [OTEL] Profiling -> http://localhost:4040
```

### Testing
```powershell
pytest tests/test_fast_features.py
pytest tests/test_dual_write.py
pytest tests/test_progress_tracking.py
pytest tests/test_observability.py
```

---

## Critical Rules (MUST FOLLOW)

### NON-VIOLATABLE TESTING RULES
**CRITICAL: These rules are ABSOLUTE and may NEVER be violated:**

1. **ONLY WAY TO TEST ACM IS BATCH MODE**
   - The ONLY way to test ACM functionality is to run it in batch mode with `python scripts/sql_batch_runner.py`
   - NEVER create single-use diagnostic scripts to test ACM functionality
   - Run full batches to test the entire data flow end-to-end

2. **DIAGNOSIS IS THROUGH LOGGING ONLY**
   - Problems are diagnosed by reading ACM_RunLogs in SQL or console output
   - Add proper logging at the point where issues occur
   - NEVER create standalone scripts to "check" or "validate" ACM behavior

3. **ALLOWED TESTING SCRIPTS**
   - `python -c "import module; print('OK')"` - Test imports only
   - `sqlcmd -S ... -Q "SELECT ..."` - Read data from SQL tables
   - That's it. No other diagnostic scripts allowed.

4. **FORECASTING MUST ALWAYS WORK**
   - Forecasting must work in ALL modes: batch, single-run, streaming
   - NEVER skip forecasting because "data is missing" - populate the data
   - All required tables (ACM_SensorNormalized_TS, ACM_RegimeTimeline, etc.) MUST be populated

### Documentation Policy
- **NEVER** create new markdown/HTML files unless explicitly requested
- **ALL** documentation goes in `docs/` folder, archived docs in `docs/archive/`
- **NEVER** create files in root directory (keep root clean)
- Reference existing docs instead of creating new ones
- Report status/analysis inline in responses, not as new documents

### Root Directory Cleanliness
- Root contains ONLY: `README.md`, `pyproject.toml`, `.gitignore`, and folders
- NEVER create: markdown files, HTML files, log files, temp files, equipment folders in root
- Artifacts go to `artifacts/` (gitignored), logs to `logs/` (gitignored)

### SQL-Only Mode
- ACM uses SQL-mode exclusively via `core/sql_client.SQLClient`
- DO NOT USE FILE MODE EVER
- All writes go through `core/output_manager.OutputManager` (respects `ALLOWED_TABLES`)
- Model persistence uses SQL `ModelRegistry` table (not filesystem)

### Time Handling
- Timestamps are local-naive (no UTC conversions)
- Use `_to_naive*` helpers in output_manager.py and acm_main.py

### Version Management
- Version stored in `utils/version.py` (currently v10.3.0)
- Increment only when explicitly requested
- Follow semantic versioning (MAJOR.MINOR.PATCH)

### Config Sync Discipline
- When `configs/config_table.csv` changes, run `python scripts/sql/populate_acm_config.py` to sync `ACM_Config` in SQL
- Keep `ConfigDict` dotted-path semantics intact

---

## Key Knowledge Base Documents

| Document | Purpose |
|----------|---------||
| `README.md` | Product overview, setup, running ACM |
| `docs/ACM_SYSTEM_OVERVIEW.md` | Architecture, module map, data flow |
| `docs/OBSERVABILITY.md` | Observability stack (traces, metrics, logs, profiling) |
| `docs/SOURCE_CONTROL_PRACTICES.md` | Git workflow, branching, releases |
| `docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md` | Authoritative SQL table definitions |
| `utils/version.py` | Current version and release notes |
| `docs/OMR_DETECTOR.md` | Overall Model Residual detector |
| `docs/COLDSTART_MODE.md` | Cold-start strategy for sparse data |
| `docs/EQUIPMENT_IMPORT_PROCEDURE.md` | How to add new equipment |
| `install/observability/README.md` | Observability stack installation |

---

## Source Control

- Branch for all work: `feature/<topic>` or `fix/<topic>`
- Avoid pushing directly to `main`
- Rebase onto `main` before merging PRs
- Clear, imperative commits (e.g., "Add batch-mode env guards")
- Run tests before merging; never merge with failing checks
- Never commit: artifacts, logs, credentials (respect `.gitignore`)

---

## Active Detectors (v10.2.0)

| Detector | Column Prefix | Purpose |
|----------|---------------|---------|
| AR1 | `ar1_z` | Autoregressive residual |
| PCA-SPE | `pca_spe_z` | Squared prediction error |
| PCA-T2 | `pca_t2_z` | Hotelling T-squared |
| IForest | `iforest_z` | Isolation forest anomaly |
| GMM | `gmm_z` | Gaussian mixture model |
| OMR | `omr_z` | Overall model residual |

**Removed Detectors**:
- `mhal_z` (Mahalanobis): Removed - redundant with PCA-T2
- `river_hst_z` (River HST): Removed - not implemented, streaming detector not needed

---

## Analytical Correctness Rules (v11.1.4+)

**CRITICAL: These statistical principles must be followed in ALL ML/analytics code.**

### Rule 1: Data Flow Traceability
When transforming data, update the SOURCE variable, not just derivatives:
```python
# ❌ BUG: train_numeric updated but train (used downstream) unchanged
train_numeric = adjusted_data

# ✅ FIX: Update the actual source used by downstream functions
for col in adjusted_cols:
    train[col] = adjusted_data[col].values
```

### Rule 2: Robust Statistics (Median/MAD, not Mean/Std)
```python
# ❌ BUG: Mean/std corrupted by outliers
mu, sd = np.mean(x), np.std(x)

# ✅ FIX: Median/MAD robust to 50% contamination
mu = np.nanmedian(x)
sd = np.nanmedian(np.abs(x - mu)) * 1.4826  # MAD → σ conversion
```
**Constant**: MAD to σ = **1.4826** = 1/Φ⁻¹(0.75)

### Rule 3: Correlation-Aware Fusion
Before fusing multiple signals, check pairwise correlation:
```python
# ❌ BUG: Naive sum double-counts correlated info
fused = w1 * signal1 + w2 * signal2  # If corr=0.8, combined gets 2x influence

# ✅ FIX: Discount correlated pairs
if abs(corr) > 0.5:
    discount = min(0.3, (abs(corr) - 0.5) * 0.5)
    w1 *= (1 - discount)
    w2 *= (1 - discount)
```

### Rule 4: Level Shift Detection for Trend Models
Maintenance resets create positive health jumps that corrupt trend fits:
```python
# ❌ BUG: Fitting entire history including maintenance resets
model.fit(health_series)  # Jump from 40% → 95% corrupts trend

# ✅ FIX: Use only post-maintenance data (jumps > 15%)
diffs = health_series.diff()
last_jump_idx = (diffs > 15.0).iloc[::-1].idxmax()
model.fit(health_series[last_jump_idx:])
```

### Rule 5: State Passthrough via Constructor
Pipeline state must flow to ALL consumers:
```python
# ❌ BUG: State computed but not passed
model_state = load_model_state()
engine = ForecastEngine(sql_client=client)  # model_state missing!

# ✅ FIX: Pass via constructor
engine = ForecastEngine(sql_client=client, model_state=model_state)
```

### Rule 6: Scope-Level Variable Initialization
Initialize all variables BEFORE conditional logic:
```python
# ❌ BUG: Variable used before init if exception occurs
if condition:
    my_var = compute_value()  # May not execute
result = my_var * 2  # UnboundLocalError

# ✅ FIX: Initialize at function scope top
my_var: float = 0.0  # Safe default
if condition:
    my_var = compute_value()
result = my_var * 2  # Always safe
```

### Statistical Constants Reference
| Constant | Value | Usage |
|----------|-------|-------|
| MAD to σ | 1.4826 | `std = mad * 1.4826` |
| Correlation threshold | 0.5 | Discount pairs with \|r\| > 0.5 |
| Health jump threshold | 15% | Positive jumps > 15% = maintenance |
| HDBSCAN min_cluster | 5% of n | `max(10, n // 20)` |

---

## Troubleshooting & Common Fixes

### Data Loading Issues
**Problem**: Batch runs return NOOP despite data existing in SQL tables.
**Root Cause**: `core/output_manager.py::_load_data_from_sql()` passing wrong parameter to stored procedure.
**Solution**: Ensure stored procedure call uses `@EquipmentName` parameter (not `@EquipID`):
```python
cur.execute(
    "EXEC dbo.usp_ACM_GetHistorianData_TEMP @StartTime=?, @EndTime=?, @EquipmentName=?",
    (start_utc, end_utc, equipment_name)
)
```

### SQL Column Mismatch Errors
**Problem**: SQL writes fail with "Cannot insert NULL into column" errors.
**Solution**: Always include all required NOT NULL columns:
- `ACM_HealthForecast_TS`: requires `Method` (NVARCHAR)
- `ACM_FailureForecast_TS`: requires `Method`, `ThresholdUsed` (FLOAT)
- `ACM_SensorForecast_TS`: requires `Method` (NVARCHAR)

### Coldstart Minimum Rows
**Problem**: NOOP with "Insufficient data" warnings.
**Root Cause**: Coldstart requires 200+ rows by default.
**Solution**: Use 5-10 day batch windows for sparse data.

### SQL Server Connection
**File**: `configs/sql_connection.ini`
```ini
[acm]
server = localhost\INSTANCENAME
database = ACM
trusted_connection = yes
driver = ODBC Driver 18 for SQL Server
TrustServerCertificate = yes
```

### Common SQL Query Patterns
```sql
-- Check table row counts
SELECT COUNT(*) FROM ACM_Scores_Wide WHERE EquipID=1

-- Get recent run logs
SELECT TOP 20 LoggedAt, Level, Message FROM ACM_RunLogs ORDER BY LoggedAt DESC

-- Find data date ranges
SELECT MIN(EntryDateTime), MAX(EntryDateTime), COUNT(*) FROM FD_FAN_Data

-- Check equipment IDs
SELECT EquipID, EquipCode, EquipName FROM Equipment
```

---

## CRITICAL ACM Table Schema Knowledge (MUST REMEMBER)

### ACM_RUL Table Columns (v10.0.0+)
**CORRECT Column Names** (NEVER use old names):
- `P10_LowerBound` (NOT `LowerBound`)
- `P50_Median` (median RUL)
- `P90_UpperBound` (NOT `UpperBound`)
- `RUL_Hours`, `Confidence`, `Method`, `FailureTime`
- `TopSensor1`, `TopSensor2`, `TopSensor3` (culprit sensors)

### ACM_Anomaly_Events Time Series Query
**CRITICAL**: For Grafana time_series format, MUST use DATETIME not VARCHAR:
```sql
-- CORRECT: Returns datetime for time column
SELECT DATEADD(HOUR, DATEDIFF(HOUR, 0, StartTime), 0) AS time,
       COUNT(*) AS value, 'Events' AS metric
FROM ACM_Anomaly_Events
WHERE EquipID = $equipment
  AND StartTime BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY DATEADD(HOUR, DATEDIFF(HOUR, 0, StartTime), 0)
ORDER BY time ASC
```

### ACM_EpisodeDiagnostics and ACM_EpisodeMetrics
- `ACM_EpisodeDiagnostics.duration_h` (per-episode duration in hours)
- `ACM_EpisodeMetrics.TotalDurationHours` (aggregated across run)

---

## Python Best Practices

### String Formatting
```python
# CORRECT: Single quotes inside f-string for SQL
sql = f"SELECT * FROM Table WHERE Name = '{value}'"

# WRONG: Mixing quotes causes errors
sql = f'SELECT * FROM Table WHERE Name = "{value}"'
```

---

## PowerShell 5.1 Command Patterns

```powershell
# Use backtick for line continuation
sqlcmd -S "localhost\INSTANCE" `
       -d ACM `
       -E `
       -Q "SELECT * FROM ACM_Runs"

# Use semicolon for command chaining (NOT &&)
cd C:\path\to\dir; python script.py; echo "Done"

# Script parameters use -Name syntax
.\script.ps1 -Equipment "FD_FAN" -StartDate "2024-01-01"

# Python args use --name syntax
python -m core.acm_main --equip FD_FAN --start-time "2024-01-01T00:00:00"
```

---

## T-SQL Best Practices (Microsoft SQL Server)

### Datetime Handling
```sql
-- Use DATEADD/DATEDIFF for time rounding (not DATE_TRUNC)
SELECT DATEADD(HOUR, DATEDIFF(HOUR, 0, Timestamp), 0) AS HourStart
FROM ACM_HealthTimeline

-- Time range filters for Grafana
WHERE Timestamp BETWEEN $__timeFrom() AND $__timeTo()
```

### Aggregation
```sql
-- TOP N (not LIMIT)
SELECT TOP 10 * FROM ACM_RUL ORDER BY RUL_Hours ASC

-- COALESCE for null handling
SELECT COALESCE(SUM(TotalEpisodes), 0) AS Total
FROM ACM_EpisodeMetrics WHERE EquipID = 1
```

---

## Grafana Dashboard Best Practices

### Time Series Panel Configuration
**ALWAYS set spanNulls to disconnect on gaps** (threshold in ms, not true/false):
```json
{
  "custom": {
    "spanNulls": 3600000,  // Disconnect if gap > 1 hour
    "lineInterpolation": "smooth"
  }
}
```

### SQL Query Guidelines for Grafana
1. **Always add time range filters**: `WHERE Timestamp BETWEEN $__timeFrom() AND $__timeTo()`
2. **Use proper datetime columns** (not FORMAT strings)
3. **Order by time ASC** for time series (DESC causes rendering issues)
4. **Use metric column** for multiple series

### Default Time Range
ACM dashboards should default to 5 years: `"from": "now-5y"`

### Grafana Datasource tracesToMetrics/tracesToLogs/tracesToProfiles Tag Mapping
**CRITICAL: Understand how Grafana maps span attributes to query variables**

ACM span attributes use `acm.` prefix (e.g., `acm.equipment`, `acm.run_id`)
Prometheus/Loki labels do NOT have prefix (e.g., `equipment`, `run_id`)

The `tags` config in datasources.yaml maps:
- `key`: Span attribute name to extract FROM
- `value`: Query variable name to use AS

```yaml
# CORRECT:
tags:
  - key: acm.equipment      # Span attribute
    value: equipment        # Query variable name
queries:
  - query: 'metric{equipment="${__span.tags.equipment}"}'  # Use 'value' name

# WRONG - NEVER DO THIS:
tags:
  - key: acm.equipment
    value: ''               # Empty value breaks it!
queries:
  - query: 'metric{equipment="${__span.tags["acm.equipment"]}"}'  # WRONG syntax
```

---

## Common Mistakes to AVOID

| Category | Wrong | Correct |
|----------|-------|---------|
| SQL columns | `ACM_RUL.LowerBound` | `ACM_RUL.P10_LowerBound` |
| SQL columns | `ACM_RUL.UpperBound` | `ACM_RUL.P90_UpperBound` |
| SQL reserved words | `AS End` | `AS EndTimeStr` (avoid reserved words) |
| SQL reserved words | `AS RowCount` | `AS TotalRows` (RowCount conflicts) |
| SQL column names | `StartTime` on ACM_Runs | `StartedAt` (check schema first!) |
| SQL aliases | `COUNT(*) as RowCount` | `COUNT(*) as TotalRows` |
| Time series | `FORMAT(time, 'yyyy-MM-dd')` | Return raw `DATETIME` |
| Time series | `ORDER BY time DESC` | `ORDER BY time ASC` |
| PowerShell | `command1 && command2` | `command1; command2` |
| PowerShell | `tail -n 20` | `Select-Object -Last 20` |
| PowerShell | `python -c "f'{var}'"` | Create `.py` file instead |
| Python inline | Multi-line in `-c` flag | **ALWAYS create script file** |
| Python inline | F-strings with `{}` or quotes | **ALWAYS create script file** |
| Grafana | `"spanNulls": true` | `"spanNulls": 3600000` |
| RUL queries | `ORDER BY RUL_Hours ASC` | `ORDER BY CreatedAt DESC` |
| Grafana tags | `value: ''` with `${__span.tags["key"]}` | `value: alias` with `${__span.tags.alias}` |
| Grafana tags | `${__span.tags["acm.equipment"]}` | Map `key: acm.equipment, value: equipment` then use `${__span.tags.equipment}` |

---

## Output Manager & Table Integrity (v11.0.3+)

### CRITICAL: Never Create Write Methods Without Calling Them

When adding a new table or write method to `output_manager.py`:
1. **ALWAYS wire it up** in `acm_main.py` at the appropriate pipeline phase
2. **Test with a batch run** to verify rows are written
3. **Document in audit** - see `docs/sql/ACM_TABLE_AUDIT_V11.md`

### Table Write Location Reference

| Table | Write Method | Pipeline Phase | Location in acm_main.py |
|-------|--------------|----------------|-------------------------|
| ACM_Scores_Wide | `write_scores()` | persist | ~5530 |
| ACM_HealthTimeline | (via comprehensive_analytics) | outputs | ~5650+ |
| ACM_RegimeTimeline | (via comprehensive_analytics) | outputs | ~5650+ |
| ACM_CalibrationSummary | `write_calibration_summary()` | calibrate | ~4955 |
| ACM_RegimeOccupancy | `write_regime_occupancy()` | regimes.label | ~4530 |
| ACM_RegimeTransitions | `write_regime_transitions()` | regimes.label | ~4545 |
| ACM_RegimePromotionLog | `write_regime_promotion_log()` | models.lifecycle | ~4780 |
| ACM_DriftController | `write_drift_controller()` | drift | ~5365 |
| ACM_ContributionTimeline | `write_contribution_timeline()` | contribution.timeline | ~5510 |

### Audit Process for Empty Tables

When a table in ALLOWED_TABLES has 0 rows:
1. **Check if write method exists** in output_manager.py
2. **Search for method calls** in acm_main.py (grep for method name)
3. If NOT called → **FIX by adding call** at appropriate pipeline phase
4. If called but 0 rows → Check if data conditions are met (e.g., no drift detected)

### Timestamp Column Standards

| Column Name | Purpose | When to Use |
|-------------|---------|-------------|
| `CreatedAt` | Record insertion time | All non-time-series tables |
| `ModifiedAt` | Record update time | Tables supporting UPSERT |
| `Timestamp` | Measurement time | Time-series data only |

**Avoid**: ValidatedAt, CalculatedAt, DroppedAt, LoggedAt (rename to CreatedAt)

---

## RUL Prediction Validation Rules (MUST FOLLOW)

### CRITICAL: RUL Prediction Reliability Requirements
**RUL predictions showing imminent failure (<24 hours) MUST be validated:**

1. **Confidence Bounds Check**:
   - P10_LowerBound, P50_Median, P90_UpperBound must NOT be NULL
   - ALWAYS filter: `WHERE (P10_LowerBound IS NOT NULL OR P50_Median IS NOT NULL)`

2. **Health State Correlation**:
   - Imminent failure (<24h) requires Health < 50% OR HealthZone = 'CRITICAL'
   - If Health > 80%, RUL < 24h is likely FALSE POSITIVE

3. **Active Defect Validation**:
   - Single moderate defect (CurrentZ < 5) does NOT justify imminent failure
   - Require MULTIPLE ActiveDefect=1 OR at least one detector with CurrentZ > 8

4. **Query Ordering - Use Most Recent, Not Worst-Case**:
   - ALWAYS: `ORDER BY CreatedAt DESC` (most recent prediction)
   - NEVER: `ORDER BY RUL_Hours ASC` (selects worst-case from all history)

### Example Valid RUL Query (Grafana)
```sql
SELECT TOP 1
    Method,
    ROUND(RUL_Hours, 1) AS 'RUL (h)',
    ROUND(P10_LowerBound, 1) AS 'P10',
    ROUND(P50_Median, 1) AS 'P50',
    ROUND(P90_UpperBound, 1) AS 'P90',
    ROUND(Confidence, 3) AS 'Confidence'
FROM ACM_RUL
WHERE EquipID = $equipment
    AND (P10_LowerBound IS NOT NULL OR P50_Median IS NOT NULL)
ORDER BY CreatedAt DESC
```

---

## Learning from Errors

When encountering errors:
1. **Check table schema**: `SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='TableName'`
2. **Verify column exists** before using in queries
3. **Test queries in sqlcmd** before adding to dashboard
4. **Validate RUL predictions** against health state
5. **Update this file** with new schema learnings immediately

---

## Core Contracts (Internal Reference)

- **Config**: `utils/config_dict.ConfigDict` loads cascading `configs/config_table.csv`
- **Output manager**: all writes go through `core/output_manager.OutputManager`
- **Time policy**: timestamps are local-naive; use `_to_naive*` helpers
- **Performance**: `core/fast_features.py` supports pandas + optional Polars
- **Rust bridge**: optional accelerator in `rust_bridge/`; Python path remains primary
