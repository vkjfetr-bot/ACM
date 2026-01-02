# ACM V11 - System Handbook

This handbook is a complete, implementation-level walkthrough of ACM V11 for new maintainers. It covers the end-to-end data flow, the role of every module, configuration surfaces, and the reasoning behind each major decision so that a new engineer can operate, extend, and hand off the system confidently.

**Current Version:** v11.0.0 - Production Release (Typed Contracts & Maturity Lifecycle)

### V11.0.0 Production Release (Dec 27, 2025)
All V11 features are now production-ready and verified:
- **DataContract Validation**: 3 validation records in ACM_DataContractValidation
- **Seasonality Detection**: 7 daily patterns detected in ACM_SeasonalPatterns
- **Asset Profiles**: 1 profile in ACM_AssetProfiles for cold-start transfer
- **Regime Definitions**: 4 versioned regimes in ACM_RegimeDefinitions
- **Active Models**: 1 model pointer in ACM_ActiveModels
- **Feature Drop Logging**: 9 low-variance features logged to ACM_FeatureDropLog
- **Refactoring Complete**: 43 helper functions extracted from acm_main.py

### V11 Architecture (Implemented)
- **DataContract Validation**: Entry-point validation via `core/pipeline_types.py` ensures data quality before processing
- **MaturityState Lifecycle**: Regime models have explicit states (INITIALIZING → LEARNING → CONVERGED → DEPRECATED)
- **FeatureMatrix Schema**: Standardized feature representation with schema enforcement in `core/feature_matrix.py`
- **DetectorProtocol ABC**: Unified detector interface in `core/detector_protocol.py` for all anomaly detectors
- **Seasonality Detection**: Diurnal/weekly pattern detection and adjustment in `core/seasonality.py`
- **Asset Similarity**: Cold-start transfer learning using similar equipment in `core/asset_similarity.py`
- **SQL Performance**: Deprecated ACM_Scores_Long (~44K rows/batch savings), batched DELETEs
- **Grafana Dashboards**: 9 production dashboards including comprehensive equipment health monitoring

### v10 Deltas (Dec 2025)
- **v10.3.0**: Consolidated observability stack with unified `core/observability.py`:
  - Removed legacy loggers: `utils/logger.py`, `utils/acm_logger.py`, `core/sql_logger.py`, `core/sql_logger_v2.py`
  - Unified Console API: `Console.info/warn/error/ok/status/header` for all logging
  - OpenTelemetry integration: Traces to Tempo, metrics to Prometheus, logs to Loki
  - Grafana Pyroscope: Continuous profiling for performance analysis
  - Timer metrics: `utils/timer.py` emits OTEL spans and Prometheus histograms
  - New dashboards: `acm_observability.json`, `acm_performance_monitor.json`
  - Install scripts: `install/observability/` with Docker Compose stack
- **v10.2.0**: Mahalanobis detector deprecated - mathematically redundant with PCA-T² (both compute Mahalanobis distance). PCA-T² is numerically stable in orthogonal space. Simplified to 6 active detectors.
- Scripts cleanup: Single-purpose analysis/check/debug scripts archived to `scripts/archive/`. Schema updater remains: `scripts/sql/export_comprehensive_schema.py`.
- Forecast/RUL refactor complete: `core/forecast_engine.py` is primary.
- Schema reference: `docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md` (authoritative ACM table/column definitions).

### v10.0.0 Major Changes from v9.0.0
- **Continuous Forecasting Architecture**: Exponential temporal blending eliminates per-batch forecast duplication; single continuous health forecast line per equipment with 12-hour decay window
- **State Persistence & Versioning**: `ForecastState` class with version tracking (v807→v813 validated) stored in `ACM_ForecastState` table; audit trail with RunID + BatchNum
- **Hazard-Based RUL**: Converts health forecasts to failure hazard rates with EWMA smoothing; survival probability curves `S(t) = exp(-∫ lambda_smooth)` with Monte Carlo confidence bounds (P10/P50/P90)
- **Time-Series Tables**: New `ACM_HealthForecast_Continuous` (merged forecasts) and `ACM_FailureHazard_TS` (smoothed hazard) tables for Grafana visualization
- **Multi-Signal Evolution**: All signals (drift CUSUM, regime MiniBatchKMeans, 7+ detectors, adaptive thresholds) evolve correctly across batches with 28 pairwise detector correlations
- **Production Validation**: Comprehensive analytical robustness report (`docs/CONTINUOUS_LEARNING_ROBUSTNESS.md`) with 14-component checklist (all ✅ PASS)
- **Critical Bug Fix**: Added Method='GaussianTail' to hazard DataFrame preventing SQL write failures for `ACM_FailureForecast` table

### v9.0.0 Major Changes from v8.2.0
- **Detector Label Standardization**: Fixed `extract_dominant_sensor()` to preserve full detector labels ("Multivariate Outlier (PCA-T²)") instead of truncating to sensor names or codes
- **Database Hygiene**: Removed migration backup tables (PCA_Components_BACKUP, RunLog_BACKUP, Runs_BACKUP) and 6 unused feature tables (Drift_TS, Enhanced*, Forecast_QualityMetrics)
- **Equipment Data Integrity**: Standardized equipment names across all runs; all 26 runs now reference consistent equipment codes
- **Run Completion**: All runs have valid CompletedAt timestamps; incomplete runs marked with NOOP status and zero duration
- **Stored Procedure Fix**: usp_ACM_FinalizeRun now correctly references ACM_Runs table with proper column mappings
- **Comprehensive Testing**: Added 30+ Python unit tests and 8 SQL validation checks covering all P0 fixes
- **Version Management**: Implemented semantic versioning with v9.0.0 tag and proper release management practices

---

## 1) Mental Model (Top-Level Flow)

```
        +--------------+    +-----------------+    +----------------+    +--------------------+
        | Ingestion    |    | Feature Builder |    | Detector Heads |    | Fusion & Episodes  |
        | (CSV / SQL)  | -> | (fast_features) | -> | (PCA/IF/GMM/   | -> |  (fuse)            |
        | acm_main     |    |                 |    |  AR1/OMR)      |    |                    |
        +--------------+    +-----------------+    +----------------+    +--------------------+
               |                       |                       |                      |
               v                       v                       v                      v
        +--------------+    +-----------------+    +----------------+    +--------------------+
        | Regimes      |    | Calibration     |    | Drift          |    | Outputs & SQL      |
        | (regimes)    |    | (z-scores, per- |    | (cusum)        |    | (OutputManager)    |
        +--------------+    | regime/adaptive)|    +----------------+    +--------------------+
                                               \                                        |
                                                \-> Forecast/RUL (forecasting, rul_*) <-/
```
```

* **acm_main.py** is the orchestrator: it loads config, ingests data, cleans/guards, builds features, fits and scores detectors, fuses, detects episodes, computes drift/regimes, writes analytics, and finalizes (including SQL logging and metadata).
* **Modes:** file mode (CSV artifacts), SQL mode (historian + stored procedures + SQL sinks), dual-write (file + SQL).
* **Artifacts:** `artifacts/{EQUIP}/run_<ts>/` per-run tables/charts + `artifacts/{EQUIP}/models/` for cached detectors and forecast/regime state.

---

## Codebase Map (Deep Index)

- **Top-level runtime**
  - `core/acm_main.py`: Orchestrator CLI entry; parses config, routes file-vs-SQL data load, runs detectors/fusion/episodes/regimes/drift, writes outputs, and finalizes SQL run metadata.
  - `core/output_manager.py`: Unified IO hub; CSV + SQL loading (`load_data`, `_load_data_from_sql`), analytics writers, batching, and table allowlist/field defaults.
  - `core/model_persistence.py`: Model registry + joblib cache manager; SQL-only/dual-write handling, manifest management, and load/save semantics.
  - `core/forecasting.py`: Enhanced forecasting and regime-aware projections; handles config ingestion and detector alignment.
  - `core/rul_engine.py`: RUL computation pipeline; persistence, run metadata extraction, and health/failure timelines.
  - `core/regimes.py`, `core/drift.py`, `core/fuse.py`, `core/fast_features.py`, `core/outliers.py`, `core/correlation.py`: Detector heads and feature plumbing used by `acm_main`.
  - `core/sql_client.py`: Thin pyodbc wrapper used by SQL mode (SP calls, retries).
  - `core/smart_coldstart.py`: Coldstart retry/orchestration when SQL historian is sparse.
  - `core/observability.py`: **Unified observability module** (v10.3.0) providing:
    - `Console` class: Structured logging with `.info()/.warn()/.error()/.ok()/.status()/.header()`
    - `Span` class: OpenTelemetry trace spans with automatic context propagation
    - `Metrics` class: Prometheus counters, histograms, gauges
    - `log_timer()`: Structured timer logs for Grafana visualization
    - Automatic Loki integration via structlog + Grafana Alloy
    - Pyroscope profiling hooks for performance analysis
    - See `docs/OBSERVABILITY.md` for full API reference
  - `utils/timer.py`: Scoped timing with OTEL span integration; uses `Console.section/status` for console-only output.
  - Feature builder implementation detail: `core/fast_features.py` prefers Polars over pandas by default. The threshold `fusion.features.polars_threshold` is set to 10 to aggressively route feature computations through Polars for performance.

- **Persistence & SQL assets**
  - `scripts/sql/` (numbered migrations + helpers):
    - `49_create_equipment_data_tables.sql`, `51_create_historian_sp_temp.sql`: Equipment data tables and historian SP used by SQL mode.
    - `52_fix_start_run_sp.sql` + later migrations: `usp_ACM_StartRun` window handling, coldstart fixes.
    - `50_create_tag_equipment_map.sql`, `53_validate_all_tables.sql`, `54_create_latest_run_views.sql`: Tag mapping, table validation, latest-run dedup views.
    - `load_equipment_data_to_sql.py`, `load_historian_from_csv.py`: CSV → SQL loaders (equipment tables vs unified historian).
    - `populate_acm_config.py`: Syncs `configs/config_table.csv` into `ACM_Config`.
    - `migrations/008_fix_start_run_data_range.sql`: Start-run window fix (OUTPUT params).
  - `configs/sql_connection.ini*`: DSN/auth for `SQLClient`.

- **Batch automation & runners**
  - `scripts/sql_batch_runner.py`: Continuous SQL batch engine; coldstart retries, window walking, progress tracking, output QA, and env toggles (`ACM_BATCH_MODE`, `ACM_FORCE_SQL_MODE`).
  - `scripts/run/run_sql_batch.ps1`: PowerShell wrapper for `sql_batch_runner` (tick, resume, parallel workers).
  - `scripts/run/run_file_mode.ps1`: File-mode convenience wrapper (CSV paths).
  - `scripts/run_data_range_batches.ps1`, `scripts/run_all_batches.ps1`: Helpers for specific batch ranges/runs.

- **Monitoring, analysis, and QA scripts**
  - `scripts/analyze_*`, `tools/*.py`, `check_*.py`: Post-run analysis, counts, schema checks, dashboard/data sanity, and truncation tools.
  - `scripts/sql/test_sql_mode_loading.py`: Sanity test for historian SP load path.
  - `scripts/update_grafana_dashboards.ps1`: Bulk dashboard query updates.

- **Documentation & status**
  - `docs/ACM_SYSTEM_OVERVIEW.md`: This handbook; now includes the deep map.
  - SQL mode/batch docs: `docs/SQL_BATCH_RUNNER.md`, `docs/SQL_BATCH_QUICK_REF.md`, `docs/SQL-44_IMPLEMENTATION.md`, `docs/BATCH_MODE_WAVE_PATTERN_FIX.md`, `docs/SQL_INTEGRATION_PLAN.md`.
  - Architecture/guides: `docs/CONTINUOUS_LEARNING.md`, `docs/CHANGELOG.md`, `docs/ACM_SYSTEM_OVERVIEW.md`.

- **Configuration & artifacts**
  - `configs/config_table.csv`: Canonical config table (mirrored into `ACM_Config`).
  - `artifacts/`: Per-run outputs/models (file mode) and cached baselines; SQL mode minimizes usage.
  - `code_tags.tsv`: Lightweight symbol index (name/kind/path:line) generated for quick lookup (ctags fallback).

---

## 2) Runtime Modes & Entry Points

**CLI:** `python -m core.acm_main --equip <EQUIP> [--train-csv ... --score-csv ...] [--config ...] [--clear-cache] [--log-level ...]`

**SQL batch automation:** `python scripts/sql_batch_runner.py --equip FD_FAN GAS_TURBINE --tick-minutes 1440 --max-workers 2 --start-from-beginning`  
Uses SQL historian tables and calls `usp_ACM_StartRun`/`usp_ACM_FinalizeRun`. Handles cold-start retries and progress tracking (`.sql_batch_progress.json`). This command sets `ACM_BATCH_MODE`/`ACM_BATCH_NUM`; avoid manipulating them elsewhere.

**File mode helper (diagnostics-only):** `powershell ./scripts/run/run_file_mode.ps1` (wraps acm_main with CSV defaults).

Mode decision and migration:
- SQL mode is the target/default; file mode exists for local diagnostics only.
- Avoid adding new file/SQL branching; collapse to the SQL-first path where practical.
- `--equip` selects the config row (SQL `ACM_Config` or `configs/config_table.csv` fallback).
- `ACM_BATCH_MODE=1` toggles batch-run semantics (continuous learning hooks).

### Quick Actions

- Resume a paused batch run for a single equipment:
  - `python scripts/sql_batch_runner.py --equip FD_FAN --tick-minutes 1440 --resume`
- Sync local config changes into SQL `ACM_Config` (ensures `fusion.features.polars_threshold=10` is reflected):
  - `python scripts/sql/populate_acm_config.py`

---

## 3) Configuration Surfaces

### Primary table (`configs/config_table.csv` or SQL `ACM_Config`)

**Current State (v10.1.0):** 238 parameters across 10 categories, cleaned of duplicates and disabled features (Dec 2025 cleanup: removed river.* streaming features, merged redundant detectors.* into models.*, resolved conflicting values).

**Key Configuration Categories:**

1. **Data Ingestion (`data.*`)**
   - `train_csv`, `score_csv`, `data_dir`: File paths for baseline/batch CSVs (file mode)
   - `timestamp_col`: Column name for timestamps (default: `EntryDateTime`)
   - `sampling_secs`: Data cadence in seconds (1800 = 30-min intervals)
   - `max_rows`: Maximum rows per batch (100000)
   - `min_train_samples`: Minimum samples for training (200; K-means n_clusters=6 requires statistical validity)

2. **Feature Engineering (`features.*`)**
   - `window`: Rolling window size (16); must match process dynamics to capture oscillations without oversmoothing
   - `fft_bands`: Spectral frequency bins for FFT decomposition
   - `polars_threshold`: Row count to trigger Polars acceleration (5000)

3. **Detectors & Models (`models.*`)**
   - `pca.*`: PCA configuration (n_components=5, randomized SVD, incremental=False)
   - `ar1.*`: AR1 detector (window=256, alpha=0.05, z_cap=8.0)
   - `iforest.*`: Isolation Forest (n_estimators=100, contamination=0.01); tighter contamination lowers false positives
   - `gmm.*`: Gaussian Mixture (k_min=2, k_max=3, BIC search enabled); higher `pca.n_components` trades explainability vs. residual sensitivity
   - `mahl.regularization`: Mahalanobis regularization (0.1 global, 1.0 for FD_FAN) to prevent ill-conditioning
   - `omr.*`: Overall Model Residual (auto model selection, n_components=5, min_samples=100)
   - `use_cache`: Enable ModelVersionManager caching (True in SQL mode)
   - `auto_retrain.*`: Automatic retraining triggers (max_anomaly_rate=0.25, max_drift_score=2.0, max_model_age_hours=720)

4. **Fusion & Weights (`fusion.*`)**
   - `weights.*`: Detector contributions (ar1_z=0.2, iforest_z=0.2, gmm_z=0.1, pca_spe_z=0.2, mhal_z=0.2, omr_z=0.10)
   - `per_regime`: Per-regime fusion enabled (True)
   - `auto_tune.*`: Adaptive weight tuning (enabled, learning_rate=0.3, temperature=1.5, method=episode_separability)

5. **Episodes & Anomaly Detection (`episodes.*`)**
   - `cpd.k_sigma`: K-sigma threshold for change-point detection (2.0 global, 4.0 for FD_FAN/GAS_TURBINE)
   - `cpd.h_sigma`: H-sigma threshold for episode boundaries (12.0)
   - `min_len`, `gap_merge`: Episode merging logic (min_len=3, gap_merge=5)
   - `cpd.auto_tune.*`: Barrier auto-tuning (k_factor=0.8, h_factor=1.2)

6. **Thresholds (`thresholds.*`)**
   - `q`: Quantile threshold (0.98 for calibration)
   - `alert`, `warn`: Thresholds for health zones (0.85, 0.7)
   - `self_tune.*`: Self-tuning (enabled, target_fp_rate=0.001, max_clip_z=100.0)
   - `adaptive.*`: Per-regime adaptive thresholds (enabled, method=quantile, confidence=0.997, per_regime=True)

7. **Regimes (`regimes.*`)**
   - `auto_k.k_min/k_max`: Cluster bounds (2-6); smaller range avoids over-segmentation on short baselines
   - `auto_k.max_models`, `auto_k.max_eval_samples`: Auto-k evaluation limits (10, 5000)
   - `quality.silhouette_min`: Minimum acceptable clustering quality (0.3)
   - `smoothing.*`: Label smoothing (passes=3, window=7, min_dwell_samples=10, min_dwell_seconds=900)
   - `transient_detection.*`: Change detection (roc_window=10, roc_threshold_high=0.15, roc_threshold_trip=0.3)
   - `health.*`: Health-based regime boundaries (fused_warn_z=2.5, fused_alert_z=4.0)

8. **Drift Detection (`drift.*`)**
   - `cusum.*`: CUSUM drift detector (threshold=2.0, smoothing_alpha=0.3, drift=0.1)
   - `p95_threshold`: Drift vs fault classification (2.0)
   - `multi_feature.*`: Multi-feature drift (enabled, trend_window=20, hysteresis_on=3.0, hysteresis_off=1.5)

9. **Forecasting (`forecasting.*`)**
   - `enhanced_enabled`, `enable_continuous`: Unified/continuous forecasting (both True)
   - `failure_threshold`: Health threshold for failure prediction (70.0)
   - `max_forecast_hours`: Maximum horizon (168 = 7 days)
   - `confidence_k`: CI multiplier (1.96 for 95% confidence)
   - `blend_tau_hours`: Exponential blending time constant (12 hours)
   - `hazard_smoothing_alpha`: EWMA alpha for hazard rate smoothing (0.3)

10. **Runtime (`runtime.*`)**
    - `storage_backend`: Storage mode (`sql`; file mode deprecated)
    - `reuse_model_fit`: Legacy joblib cache (False in SQL mode; use ModelRegistry)
    - `tick_minutes`: Batch cadence (30 for FD_FAN, 1440 for GAS_TURBINE)
    - `version`: Current ACM version (v10.1.0)
    - `phases.*`: Pipeline phase toggles (features, regimes, drift, models, fuse, report)

**Equipment-Specific Overrides:**
- Global defaults: `EquipID=0`
- FD_FAN (EquipID=1): `mahl.regularization=1.0`, `episodes.cpd.k_sigma=4.0`, `min_train_samples=200`
- GAS_TURBINE (EquipID=2621): `timestamp_col=Ts`, `tick_minutes=1440`, `min_train_samples=200`

**Config sync:** When `configs/config_table.csv` changes, run `python scripts/sql/populate_acm_config.py` to update `ACM_Config` so SQL mode stays authoritative (238 params synced as of Dec 2025 cleanup).  
**Config history:** `ACM_ConfigHistory` tracks all adaptive tuning changes via `core.config_history_writer.ConfigHistoryWriter` with timestamp, parameter path, old/new values, reason, and UpdatedBy tag.

**Removed/Deprecated (v10.1.0 cleanup):**
- `river.*`: Streaming features removed (never implemented)
- `fusion.weights.river_hst_z`, `fusion.weights.pca_t2_z`: Zero-weight detectors removed
- Duplicate `detectors.*` namespace: Merged into `models.*` to eliminate redundancy

### SQL connection
- `configs/sql_connection.ini` (or `.example.ini`) supplies DSN/user/pass; `SQLClient.from_ini` loads it.

### CLI overrides
- `--config` loads a YAML overrides file merged atop the config table row.
- `--train-csv` / `--score-csv` override ingestion paths per run.
- `--clear-cache` forces detector refit, ignoring cached joblib.
- Logging overrides: `--log-level`, `--log-format`, `--log-module-level`, `--log-file` (SQL sink is always on in SQL mode).

---

## 4) Module-by-Module Map

### Orchestrator: `core/acm_main.py`
- **Helpers:** `_get_equipment_id`, `_load_config`, `_compute_config_signature`, `_ensure_local_index`, `_sql_start_run`/`_sql_finalize_run`, `_calculate_adaptive_thresholds`, `_compute_drift_trend`, `_compute_regime_volatility`.
- **Data load:** file via `OutputManager.load_data`; SQL via `SmartColdstart.load_with_retry`. Dedup timestamps, enforce monotonicity, guard empty SCORE (NOOP).
- **Baseline buffer:** seeds TRAIN from SQL `ACM_BaselineBuffer`, local `baseline_buffer.csv`, or SCORE head when baseline thin.
- **Guardrails:** overlap checks, low-variance sensors, data quality export (`tables/data_quality.csv` or SQL `ACM_DataQuality`), cadence checks.
- **Features:** `fast_features.compute_basic_features` with Polars fast-path; uses TRAIN medians to impute SCORE (prevents leakage).
- **Heads fit/score:** PCA (PCASubspaceDetector), IsolationForest, GMM, AR1 (forecasting.AR1Detector), OMR (core.omr), optional River streaming. Reuses cached detectors when signature matches. Note: Mahalanobis deprecated v10.2.0.
- **Regimes:** builds feature basis (PCA scores + optional raw tags), clusters with MiniBatchKMeans, smooths labels, transient detection, per-regime stats; supports load/save of regime model.
- **Calibration:** z-score calibration per head (ScoreCalibrator), optional per-regime thresholds (DET-07), adaptive thresholds (adaptive_thresholds.py).
- **Fusion:** `fuse.Fuser.fuse` combines z streams under configured weights; auto-tunes weights via episode separability (tune_detector_weights).
- **Episodes:** `fuse.Fuser.detect_episodes` finds sustained excursions with hysteresis; writes culprits via `episode_culprits_writer`.
- **Drift:** CUSUM on fused (`drift.compute`), plots via `drift.run`.
- **Forecast/RUL:** optional health/failure forecasts and RUL surfaces via `forecasting.run_and_persist_enhanced_forecasting` and `rul_estimator`/`enhanced_rul_estimator`.
- **Analytics/output:** centralized `OutputManager` for CSV/PNG/SQL (scores, drift, events, regimes, PCA artifacts, health timelines, fusion quality, OMR contributions, etc.). Writes run stats, run metadata (`run_metadata_writer`), config history, culprits, caches models.
- **Finalize:** always calls `_sql_finalize_run`; writes `meta.json` in file mode only.

### Feature Builder: `core/fast_features.py`
- Rolling statistics: `rolling_median`, `rolling_mad`, `rolling_mean_std`, `rolling_skew_kurt`, `rolling_ols_slope`.
- Spectral/lag: `rolling_spectral_energy`, `rolling_xcorr`, `rolling_pairwise_lag`, `compute_basic_features(_pl)`.
- Fills missing values with medians/ffill/bfill; Polars acceleration when available, pandas fallback otherwise. Uses TRAIN-derived fill values for SCORE to prevent leakage.

### Detectors (6 Active Heads - v10.2.0)

Each detector answers a specific "what's wrong?" question:

| Detector | Z-Score | What's Wrong? | Fault Types |
|----------|---------|---------------|-------------|
| **AR1** | `ar1_z` | "A sensor is drifting/spiking" | Sensor degradation, control loop issues, actuator wear |
| **PCA-SPE** | `pca_spe_z` | "Sensors are decoupled" | Mechanical coupling loss, thermal expansion, structural fatigue |
| **PCA-T²** | `pca_t2_z` | "Operating point is abnormal" | Process upset, load imbalance, off-design operation |
| **IForest** | `iforest_z` | "This is a rare state" | Novel failure mode, rare transient, unknown condition |
| **GMM** | `gmm_z` | "Doesn't match known clusters" | Regime transition, mode confusion, startup/shutdown anomaly |
| **OMR** | `omr_z` | "Sensors don't predict each other" | Fouling, wear, misalignment, calibration drift |

- **Correlation (`core/correlation.py`):**
  - `MahalanobisDetector`: DEPRECATED v10.2.0 - redundant with PCA-T² (both compute Mahalanobis distance). PCA-T² is numerically stable because covariance is diagonal in PCA space.
  - `PCASubspaceDetector`: cleans non-finite, drops constants, scales, fits PCA; returns SPE (Q) and T²; handles low-sample fallback.
- **Outliers (`core/outliers.py`):**
  - `IsolationForestDetector`: fits/uses scikit IF, stores columns, optional quantile threshold when contamination numeric.
  - `GMMDetector`: BIC-driven component selection, variance guards, scaling; returns neg log-likelihood style scores.
- **OMR (`core/omr.py`):**
  - Multivariate reconstruction error via PLS/linear ensemble/PCA. Features auto model selection, diagnostics, per-sensor contribution extraction, z clipping, min sample guards.
- **AR1 (`core/ar1_detector.py`):** Per-sensor AR(1) residual detector for temporal pattern anomalies.
- **Forecasting/RUL (`core/forecasting.py`, `core/rul_estimator.py`, `core/enhanced_rul_estimator.py`):**
  - **Health & Failure Forecasting:** Exponential smoothing with bootstrap confidence intervals, adaptive parameters (alpha/beta), quality gates (blocks SPARSE/FLAT/NOISY but allows GAPPY data for historical replay)
  - **Physical Sensor Forecasting:** Predicts future values for top 10 changing sensors using LinearTrend or VAR (Vector AutoRegression) methods; includes confidence intervals and regime awareness
  - **RUL Estimation:** Monte Carlo simulations with multiple degradation paths (trajectory, hazard, energy-based); P10/P50/P90 bounds with confidence scoring
  - **State Management:** Persistent forecast state with version tracking (ACM_ForecastingState); retrain decisions based on RMSE degradation, data quality, and time thresholds
  - AR1 per-sensor residual detector, data hash for retrain decisions, SQL/file dual persistence of forecast state
  - Enhanced RUL: multiple degradation models (AR1, exponential, Weibull-inspired, linear), learning state, attribution, maintenance recommendations, hazard smoothing
  - **NEW v10.0.0 - Continuous Forecasting Architecture:**
    - `merge_forecast_horizons()` (lines 888-988): Exponential temporal blending with tau=12h; dual weighting (recency × horizon); NaN-aware merging; weight capping at 0.9
    - `ForecastState` class: Version-tracked state persistence (v807→v813) in `ACM_ForecastState` table; audit trail with RunID + BatchNum + timestamp
    - `write_continuous_health_forecast()` (lines 1014-1110): Bulk insert merged health forecasts into `ACM_HealthForecast_Continuous` with transaction handling
    - `write_continuous_hazard_forecast()` (lines 1111-1209): EWMA-smoothed hazard curves into `ACM_FailureHazard_TS` with survival probability calculation
    - Hazard calculation: `lambda(t) = -ln(1-p(t))/dt`, survival: `S(t) = exp(-∫ lambda_smooth)`
    - Integration points: Lines 2938-2988 (horizon merging), 3029-3043 (continuous writes), ~2559 (Method='GaussianTail' for hazard_df)
    - Quality assurance: RMSE validation gates, MAPE tracking (33.8%), TheilU coefficient (1.098), P10/P50/P90 confidence bounds
    - Multi-signal evolution: Drift (CUSUM P95), regimes (MiniBatchKMeans auto-k), detector correlation (28 pairwise), adaptive thresholds with PR-AUC throttling
    - Benefit: Single continuous forecast line per equipment; smooth 12h transitions; no Grafana per-run duplicates; production-validated (see `docs/CONTINUOUS_LEARNING_ROBUSTNESS.md`)
- **Drift (`core/drift.py`):** CUSUMDetector with z calibration; report plot generator.
- **Regimes (`core/regimes.py`):** feature basis builder, auto-k (silhouette/Calinski-Harabasz), smoothing, transient detection via ROC energy, health labeling, persistence to joblib/json, loading with version guard.
- **Fusion (`core/fuse.py`):** `Fuser.fuse` (weighted sum with z clipping), `detect_episodes` (hysteresis thresholds), `ScoreCalibrator`, `tune_detector_weights` (PR-AUC against episode windows).

### Thresholding & Adaptation
- **adaptive_thresholds.py:** quantile/MAD/hybrid fused threshold calculator; supports per-regime thresholds; validation to avoid degenerate values.
- **config_history_writer.py:** append-only JSON log of auto-tune changes to track self-modifying behavior.

### Persistence & Metadata
- **model_persistence.py:** versioned model storage under `artifacts/{equip}/models/vN`, manifest generation, SQL dual-write hooks. Forecast state save/load helpers.
- **run_metadata_writer.py:** writes ACM_Runs / ACM_RunMetadata; extracts health metrics, data quality, refit flags; guards SQL errors.
- **model_evaluation.py:** quality monitor to decide retrain based on anomaly rates, regime quality, episode quality.

### I/O & SQL
- **output_manager.py:** single point for CSV/JSON/PNG + SQL writes. Batched writes with buffering, schema discovery, health index generation, analytics tables (scores_wide/long, drift, episodes, timelines, fusion quality, OMR contributions, calibration summaries, hotspots, forecast quality metrics, OMR diagnostics, RUL summaries, etc.). Generates charts and descriptors; coordinates dual mode.
- **simplified_output_manager.py:** trimmed SQL writer for legacy paths.
- **sql_client.py / sql_protocol.py:** thin wrappers over pyodbc for safe cursor operations and protocol typing.
- **sql_logger.py:** SQL sink that mirrors Console logs into ACM_Log.
- **sql_performance.py:** timing/batching helpers; `SQLBatchWriter` for efficient inserts.
- **historian.py:** historian client abstraction (cyclic/full tag retrieval).

### Cold start & utilities
- **smart_coldstart.py:** determines if baseline ready; backfills from historian/score; tracks progress.
- **utils.config_dict / validators / timestamp_utils / logger / timer:** config merging with type safety, timestamp normalization, structured logging, scoped timers.

### Episode culprits
- **episode_culprits_writer.py:** parses culprits strings, computes detector contributions, writes enhanced culprits (per-episode sensor attributions).

### Scripts (operations)
- `scripts/sql_batch_runner.py`: continuous SQL processing with ticked windows, resume, cold-start retries, historian coverage checks.
- `python scripts/sql_batch_runner.py ...`: single entry point for SQL batch/continuous runs.
- `scripts/sql/populate_acm_config.py`: pushes `config_table.csv` rows into SQL `ACM_Config`.
- Validation/check scripts: `check_*`, `validate_*`, `monitor_*`, `analyze_*` for dashboards, data gaps, drift, forecast status, table population.
- `scripts/sql/*`: SQL schema helpers and tests.

---

## 5) Data & Artifact Layout

- **Input data:** `data/` for CSV baselines/batches; SQL mode pulls from historian tables configured per equipment.
- **Artifacts (diagnostic/file mode):** `artifacts/{EQUIP}/run_<timestamp>/` with `scores.csv`, `drift.csv`, `episodes.csv`, `tables/*.csv`, `charts/*.png`, `meta.json`.
- **Models/cache:** `artifacts/{EQUIP}/models/` containing `detectors.joblib`, regime model joblib/json, forecast state, baseline buffer.
- **Grafana dashboards:** Active work is on `grafana_dashboards/ACM Claude Generated To Be Fixed.json`. All other dashboards are archived under `grafana_dashboards/archive/` (see `grafana_dashboards/archive/ARCHIVE_LOG.md` for the move list) to keep the working set lean.

---

## 6) How to Run ACM (End-to-End)

1) **Environment**
- Python >= 3.11. `python -m venv .venv && .\\.venv\\Scripts\\activate` (Windows).
- Install deps: `pip install -r requirements.txt` (or `pip install .`).

2) **Config & data**
- Populate `configs/config_table.csv` (or SQL `ACM_Config`). Ensure `data.train_csv` and `data.score_csv` exist for file mode.
- For SQL mode, fill `configs/sql_connection.ini` with DSN/credentials and historian table names.

3) **Run (file mode example)**
```
python -m core.acm_main --equip FD_FAN ^
  --train-csv data/FD_FAN_BASELINE_DATA.csv ^
  --score-csv data/FD_FAN_BATCH_DATA.csv ^
  --log-level INFO
```

4) **Run (SQL mode example)**
```
python -m core.acm_main --equip FD_FAN --log-level INFO
# Uses SQL config row, historian, and SQL sinks (SQL logging always enabled in SQL mode).
```

5) **Batch runner (SQL, continuous)**
```
python scripts/sql_batch_runner.py --equip FD_FAN GAS_TURBINE --max-workers 2 --tick-minutes 240 --resume
```

6) **Outputs**
- File mode (diagnostics): check `artifacts/FD_FAN/run_<ts>/tables/` and `charts/`.
- SQL mode (primary): check ACM_* tables (Scores_Wide/Long, Episodes, Drift_TS, Run_Stats, PCA artifacts, OMR contributions, FusionQualityReport, OMR diagnostics, forecast quality metrics, RUL summaries, etc.) and ACM_Runs metadata.

---

## 7) Analytical Reasoning (Why each step exists)

- **Dedup + local timestamps:** prevents double-counting and timezone drift; many detectors assume monotonic time.
- **Baseline seeding:** keeps the system from stalling when no explicit TRAIN exists; uses stable SQL buffer or SCORE head to bootstrap with minimum sample guard.
- **Feature windowing & spectral energy:** captures local trends/oscillations; window tuned per equipment cadence (`features.window`, `features.fs_hz`).
- **Median-based fill:** robust to spikes; TRAIN-derived stats avoid leakage into SCORE.
- **Detector diversity:** PCA catches correlated variance loss; Mahalanobis catches covariance shifts; IF/GMM catch density anomalies; OMR captures multivariate reconstruction error; AR1 residuals catch temporal shifts. Fusion combines complementary signals.
- **Calibration to z-scores:** normalizes heterogeneous detectors so fusion weights are meaningful; per-regime thresholds prevent over-alerting in regimes with naturally higher variance.
- **Episode detection with hysteresis:** avoids flapping; only sustained fused excursions become episodes.
- **Drift (CUSUM):** monitors slow changes even when fused stays below alert; informs refit policies.
- **Regimes:** clusters operating states to contextualize anomalies and enable per-regime thresholds/health; transient detection distinguishes startups/trips.
- **Adaptive thresholds & auto-tuned fusion:** adjusts to drift and class imbalance without manual retuning; logged via config_history_writer for auditability.
- **Caching/persistence:** speeds re-runs and supports continuous learning; config signature guards stale models.
- **Run metadata & quality checks:** ACM_Runs + data quality score make operational health observable; refit flags trigger retraining pipelines.
- **SQL-first posture:** production runs should use SQL sinks; file mode is only for diagnostics and should not accumulate bespoke branches.

---

## 8) Quick Reference: Functions by File

- **core/acm_main.py:** `main` (CLI), `_sql_start_run`, `_sql_finalize_run`, `_calculate_adaptive_thresholds`, `_compute_drift_trend`, `_compute_regime_volatility`, `_load_config`, `_compute_config_signature`, `_nearest_indexer`, `_ensure_local_index`.
- **core/fast_features.py:** rolling_* funcs, `compute_basic_features`, Polars fast-path helpers.
- **core/correlation.py:** `MahalanobisDetector.fit/score`, `PCASubspaceDetector.fit/score`.
- **core/outliers.py:** `IsolationForestDetector.fit/score/predict`, `GMMDetector.fit/score`.
- **core/omr.py:** `OMRDetector.fit/score/get_top_contributors/get_diagnostics`, `OMRModel.to_dict`.
- **core/fuse.py:** `Fuser.fuse/detect_episodes`, `ScoreCalibrator.fit/transform`, `tune_detector_weights`.
- **core/regimes.py:** `build_feature_basis`, `build_regime_model`, `apply_regime_labels`, `detect_transient_states`, `save_regime_model`, `load_regime_model`.
- **core/drift.py:** `CUSUMDetector.fit/score`, `compute`, `run`.
- **core/forecasting.py:** `AR1Detector`, `estimate_rul`, `run_and_persist_enhanced_forecasting`, `run_enhanced_forecasting_sql`, `should_retrain`, `compute_data_hash`.
- **core/rul_estimator.py / enhanced_rul_estimator.py:** `_simple_ar1_forecast`, `estimate_rul_and_failure`, `compute_rul_multipath`, degradation model classes, attribution and maintenance recommendations.
- **core/adaptive_thresholds.py:** `AdaptiveThresholdCalculator.calculate_fused_threshold/_calculate_per_regime`, `calculate_warn_threshold`.
- **core/output_manager.py:** `create_output_manager`, `write_scores_ts`, `write_drift_ts`, `write_anomaly_events`, `write_regime_episodes`, `write_pca_model/loadings/metrics`, analytics generators (`_generate_*`), `flush/close`.
- **core/model_persistence.py:** `ModelVersionManager.get_*`, `save_models/load_models`, `save_forecast_state/load_forecast_state`, manifest helpers.
- **core/run_metadata_writer.py:** `write_run_metadata`, `extract_run_metadata_from_scores`, `extract_data_quality_score`, `write_retrain_metadata`.
- **core/config_history_writer.py:** `write_config_change(s)`, `log_auto_tune_changes`.
- **core/episode_culprits_writer.py:** `compute_detector_contributions`, `write_episode_culprits_enhanced`.
- **core/smart_coldstart.py:** `SmartColdstart.load_with_retry`, cadence detection, progress tracking.
- **core/sql_*:** `SQLClient`, `SqlLogSink`, `SQLPerformanceMonitor`, `SQLBatchWriter`, `sql_protocol` mocks.
- **core/historian.py:** historian fetch helpers.
- **scripts/sql_batch_runner.py:** `SQLBatchRunner` orchestrates continuous historian processing.

---

## 9) Onboarding Checklist

- Install deps and run a file-mode smoke test with sample CSVs (diagnostic).
- Configure SQL credentials and run a single SQL-mode job with `--equip` pointed at a known equipment row; verify ACM_Runs, Scores_Wide, Episodes populate.
- When `config_table.csv` is edited, run `python scripts/sql/populate_acm_config.py` to sync `ACM_Config`; confirm via a quick select.
- Inspect `artifacts/{EQUIP}/run_<ts>/meta.json` (file mode) or ACM_Runs (SQL) to confirm health indices and thresholds.
- Review `config_history.log` to ensure auto-tuning events are recorded.
- Validate dashboards using `grafana_dashboards/` JSONs (Ops Command Center, Asset Health Deep Dive, Failure & Maintenance Planner, Sensor & Regime Forensics, Ops & Model Observability) and `docs/` quick refs (e.g., `BATCH_MODE_SQL_QUICK_REF.md`, `SQL_MODE_CONFIGURATION.md`).

---

## 10) Troubleshooting Signals

- Empty `scores.csv` or zero rows written: check data guardrails, duplicate timestamp removal, historian coverage (`check_historian_data.py`, `_log_historian_overview` in sql_batch_runner).
- Flat fused z near zero: baseline too short or detectors cached with mismatched signature; rerun with `--clear-cache`.
- Frequent false alerts: adjust `thresholds.clip_z`, enable per-regime thresholds (DET-07), or lower fusion weights for noisy heads via config or auto-tuning.
- Missing SQL writes: confirm `enable_sql_sink` not disabled, SQL connectivity via `OutputManager` diagnostic, table existence (`check_tables_existence.py`).
- Slow runs: enable Polars, reduce feature window, trim tag list (`features.top_k_tags`), or limit max models in regimes auto-k.

---

## 11) Extend/Modify Safely

- Add detectors by producing z-scores and registering them in fusion weight config; ensure calibration and episode detection include them.
- Update configs via `configs/config_table.csv` or SQL `ACM_Config`; keep `config_signature` changes in mind for cache invalidation.
- When changing schema/outputs, extend `OutputManager.ALLOWED_TABLES` and add write helpers + analytics generators to keep file/SQL parity.
- For new regimes logic, bump `REGIME_MODEL_VERSION` to invalidate stale caches.
- Add logging via `Console` with module-level overrides (`--log-module-level core.fast_features=DEBUG`).

---

## 12) Minimal Config Examples

**File mode row (config_table.csv)**
```
EquipID,Category,ParamPath,ParamValue,ValueType
0,data,train_csv,data/FD_FAN_BASELINE_DATA.csv,string
0,data,score_csv,data/FD_FAN_BATCH_DATA.csv,string
0,data,timestamp_col,Timestamp,string
0,data,sampling_secs,60,int
0,features,window,16,int
0,models,pca.n_components,5,int
0,models,iforest.contamination,0.001,float
0,fusion,weights,"{""pca_spe_z"":0.25,""pca_t2_z"":0.25,""iforest_z"":0.2,""gmm_z"":0.15,""omr_z"":0.15}",json
```

**Equipment-specific overrides (config_table.csv)**
```
EquipID,Category,ParamPath,ParamValue,ValueType
1,data,timestamp_col,EntryDateTime,string
1,data,sampling_secs,60,int
1,models,pca.incremental,true,bool
1,runtime,reuse_model_fit,false,bool
```

**SQL connection (configs/sql_connection.ini)**
```
[sqlserver]
driver=ODBC Driver 17 for SQL Server
server=YOUR_SQL_SERVER_HOST
database=ACM
uid=acm_user
pwd=***REDACTED***
trust_certificate=yes
timeout=30
```

---

## 13) Data & Timestamp Expectations

- **Index:** DatetimeIndex, timezone-naive, monotonic increasing; duplicates are dropped in `acm_main`.
- **Timestamp column name:** `timestamp_col` config (defaults to `Timestamp`/`EntryDateTime`); historian uses `EntryDateTime`.
- **Cadence tolerance:** configured via `data.sampling_secs`; cadence check warns when drifted; `SmartColdstart` enforces window discovery.
- **Columns:** numeric sensor columns; non-numeric are dropped before detectors; low-variance columns are warned and optionally excluded.
- **Missing data:** TRAIN medians used to fill SCORE; remaining NaN columns trigger validation errors upstream.
- **Windowing:** feature window (`features.window`) should reflect process dynamics (too small = noisy, too large = lag).

---

## 14) Detector & Fusion Defaults (quick reference)

- **Weights (example):** pca_spe_z 0.25, pca_t2_z 0.25, iforest_z 0.20, gmm_z 0.15, omr_z 0.15 (tuned per asset).
- **Thresholds:** fused_z alert ~3.0 unless adaptive/per-regime enabled; detector z clipping via `thresholds.clip_z` (e.g., 8–10) to avoid extreme leverage.
- **PCA:** n_components=5, randomized SVD, incremental optional; drops constant cols.
- **Mahalanobis:** regularization 1e-3 with auto-escalation on high condition number.
- **Isolation Forest:** contamination auto or numeric; max_samples auto; random_state=17.
- **GMM:** BIC search enabled, reg_covar=1e-3, covariance_type=diag, cap k by samples.
- **OMR:** model_type=auto, n_components=5, alpha=1.0, min_samples=100, z clip=10.
- **AR1:** min_samples=3; falls back to mean residual when under-sampled.
- **Fusion tuning:** episode-separability method, temperature=2.0, min_weight=0.05, learning_rate=0.3.

---

## 15) Validation & Testing Recipes

- **Smoke (file mode):**
  - `python -m core.acm_main --equip FD_FAN --train-csv data/FD_FAN_BASELINE_DATA.csv --score-csv data/FD_FAN_BATCH_DATA.csv --log-level INFO`
  - Verify `artifacts/FD_FAN/run_<ts>/scores.csv`, `episodes.csv`, `meta.json` exist; inspect fused z distribution and episode count.
- **SQL dry-run sanity:**
  - Ensure historian coverage via `_log_historian_overview` (runs inside `sql_batch_runner`); or run `scripts/check_historian_data.py`.
  - `python -m core.acm_main --equip FD_FAN --log-level INFO` (SQL mode default). Check ACM_Runs row inserted and Scores_Wide populated.
- **Cache/threshold checks:**
  - Run once, rerun with `--clear-cache` and compare PCA/IFOREST model hashes in `detectors.joblib` to ensure cache invalidation.
- **Regression spot-checks:**
  - Compare P95 fused z and episode counts across two runs; large deltas suggest config drift or data shift.

---

## 16) Operational Signals & Runbook

- **Primary health:** ACM_Runs.health_status, avg/min health index, max_fused_z.
- **Data quality:** ACM_DataQuality score; low values imply ingestion or sensor issues.
- **Drift:** Drift_TS p95 and slope from `_compute_drift_trend`; spikes may warrant refit.
- **Regimes:** silhouette score, regime_count stability; high volatility can mean process change or bad clustering.
- **OMR diagnostics:** ACM_OMR_Diagnostics residual std and calibration status; saturation indicates recalibration needed.
- **Forecast quality:** ACM_Forecast_QualityMetrics RMSE/MAE/MAPE trends; retrain when sustained degradation.
- **RUL multipath:** ACM_RUL_Summary consensus vs paths; large divergence signals uncertainty.
- **Actions:**
  - Frequent false alerts: lower fusion weights for noisy heads, enable per-regime thresholds, or raise fused warn/alert thresholds.
  - Flat fused z: baseline too small or stale cache; rerun with `--clear-cache` or expand baseline window.
  - SQL write gaps: check SQL sink enabled, table existence (`check_tables_existence.py`), and connectivity.
  - Drift spikes: trigger retrain by clearing cache or setting refit flag in `artifacts/{equip}/models/refit_requested.flag`.

---

## 17) Versioning & Change Control

- **Config signature:** `_compute_config_signature` hashes model/feature/fusion/regime/threshold sections; cache reuse only when hash matches.
- **Regime model version:** `REGIME_MODEL_VERSION` (regimes.py) gates loading; bump when changing clustering logic or feature basis.
- **Forecast state version:** stored in manifest under `artifacts/{equip}/models`; retrain when data hash changes.
- **Output schema changes:** update `OutputManager.ALLOWED_TABLES`, add write helpers, and align SQL schemas before deployment.
- **Documentation alignment:** update README and this handbook when changing defaults or run modes to keep ops expectations accurate.

---

## 18) Security & Secrets

- **Credentials:** keep `configs/sql_connection.ini` out of VCS; use `.example.ini` as template. Do not commit real passwords/DSNs.
- **Access:** restrict artifacts and configs directories to least privilege; SQL user should have only required read/write on ACM_* tables and historian views.
- **Logging:** avoid logging credentials or PII; Console outputs may be mirrored to SQL via `SqlLogSink`.
- **Environment overrides:** prefer environment variables or secure secret stores for credentials in CI/CD; validate that `SqlLogSink` is enabled only where allowed.

---

## 19) Adding New Equipment

To import a new equipment's sensor data into ACM, see [EQUIPMENT_IMPORT_PROCEDURE.md](EQUIPMENT_IMPORT_PROCEDURE.md).

Quick summary:
1. Prepare CSV with datetime + sensor columns
2. Run `python scripts/sql/import_csv_to_acm.py --csv file.csv --equip-code CODE --equip-name "Name"`
3. Run batch processing: `python scripts/sql_batch_runner.py --equip CODE --max-batches 10`
4. View in dashboard (select equipment, adjust time range)

---

## 20) Configuration Reference

ACM uses a cascading configuration system where **global defaults** (EquipID=0) can be overridden by **equipment-specific values**. Configuration is stored in `configs/config_table.csv` and synced to SQL `ACM_Config` table via `scripts/sql/populate_acm_config.py`.

### Configuration Architecture

```
configs/config_table.csv  -->  Python ConfigDict  -->  SQL ACM_Config (optional sync)
         |                           |
         v                           v
   EquipID=0 (global)        Dot-path access: cfg['models.pca.n_components']
   EquipID=N (override)      Equipment-specific lookup with fallback
```

### Parameter Categories

| Category | Purpose | Typical Tuning Frequency |
|----------|---------|--------------------------|
| `data` | Data loading, cadence, timestamps | Per-Equipment |
| `features` | Window sizes, spectral analysis | Rarely |
| `models` | Detector hyperparameters | Per-Equipment (some) |
| `fusion` | Detector weight blending | Auto-tuned |
| `episodes` | Anomaly event detection | Per-Equipment |
| `thresholds` | Alert/warning levels | Per-Equipment |
| `regimes` | Operating mode clustering | Per-Equipment |
| `drift` | Slow change detection | Rarely |
| `forecasting` | Health/RUL prediction | Rarely |
| `runtime` | Execution control | Per-Equipment |
| `health` | Health index calculation | Rarely |

---

### DATA CATEGORY - Most Likely to Need Per-Equipment Tuning

| Parameter | Default | Type | Description | Equipment-Specific? |
|-----------|---------|------|-------------|---------------------|
| **`sampling_secs`** | 1800 | int | **CRITICAL**: Data sampling interval in seconds. Must match equipment's native data cadence. Examples: FD_FAN/GAS_TURBINE=1800 (30 min), ELECTRIC_MOTOR=60 (1 min). Wrong value causes massive data loss during resampling. | **YES - Always override for new equipment** |
| `timestamp_col` | EntryDateTime | string | Name of timestamp column in historian data. Some assets use "Ts" or other names. | YES if asset uses different column |
| `min_train_samples` | 200 | int | Minimum training rows for coldstart. Affects how long coldstart takes to accumulate enough data. | YES if equipment has sparse data |
| `max_rows` | 100000 | int | Maximum rows to process per batch. Prevents memory issues. | Rarely |
| `train_csv` / `score_csv` | - | string | File paths for file-mode operation (SQL mode ignores these). | Only for file mode |

**Critical Insight**: If you see "Insufficient data: N rows (required: 200)" but the SP returns many more rows, check that `sampling_secs` matches the actual data cadence.

---

### MODEL CATEGORY - Detector Hyperparameters

#### PCA (Principal Component Analysis)
| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `pca.n_components` | 5 | Latent dimensions retained. More = captures more variance, less robust to noise. | Rarely |
| `pca.svd_solver` | randomized | SVD algorithm. "randomized" is faster for large datasets. | No |

#### AR1 (Autoregressive Time-Series)
| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `ar1.window` | 256 | Lookback window for AR model. Should be > data cadence. | YES if very different cadence |
| `ar1.alpha` | 0.05 | Significance level for residual threshold. | Rarely |
| `ar1.z_cap` | 8.0 | Maximum z-score cap to prevent outlier dominance. | Rarely |

#### IsolationForest
| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `iforest.n_estimators` | 100 | Number of trees. More = more accurate but slower. | Rarely |
| `iforest.contamination` | 0.01 | Expected anomaly fraction. Lower = fewer false positives. | YES for very noisy equipment |
| `iforest.max_samples` | 2048 | Samples per tree. Affects training speed. | No |

#### GMM (Gaussian Mixture Model)
| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `gmm.k_min` / `gmm.k_max` | 2/3 | Component range for BIC search. | Rarely |
| `gmm.covariance_type` | diag | Covariance structure. "diag" is faster. | No |

#### Mahalanobis Distance
| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| **`mahl.regularization`** | 0.1 | Covariance matrix regularization. **Increase if you see "high condition number" warnings**. | **YES - auto-tuned for ill-conditioned data** |

#### OMR (Overall Model Residual)
| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `omr.model_type` | auto | Model type: auto/pca/ae. Auto selects based on data. | Rarely |
| `omr.n_components` | 5 | Latent components. | Rarely |
| `omr.min_samples` | 100 | Minimum samples to fit OMR. | Rarely |

#### Continuous Learning
| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `auto_retrain.enabled` | True | Enable automatic retraining. | No |
| `auto_retrain.max_anomaly_rate` | 0.25 | Retrain if anomaly rate exceeds 25%. | YES if expected high anomaly rate |
| `auto_retrain.max_drift_score` | 2.0 | Retrain if drift exceeds threshold. | Rarely |
| `auto_retrain.max_model_age_hours` | 720 | Retrain if model older than 30 days. | Rarely |

---

### FUSION CATEGORY - Detector Weight Blending

| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `weights.ar1_z` | 0.2 | Weight for AR1 detector (0-1). | Auto-tuned |
| `weights.iforest_z` | 0.2 | Weight for IsolationForest. | Auto-tuned |
| `weights.gmm_z` | 0.1 | Weight for GMM density. | Auto-tuned |
| `weights.pca_spe_z` | 0.2 | Weight for PCA squared prediction error. | Auto-tuned |
| `weights.mhal_z` | 0.2 | Weight for Mahalanobis distance. | Auto-tuned |
| `weights.omr_z` | 0.1 | Weight for OMR residual. | Auto-tuned |
| `auto_tune.enabled` | True | Enable automatic weight tuning based on episode separability. | No |
| `per_regime` | True | Compute separate fusion weights per regime. | No |
| `cooldown` | 10 | Samples after episode before new episode can start. | Rarely |

---

### EPISODES CATEGORY - Anomaly Event Detection

| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| **`cpd.k_sigma`** | 2.0 | Episode start threshold (z-score). **Increase if too many episodes detected**. | **YES - often auto-tuned based on anomaly rate** |
| `cpd.h_sigma` | 12.0 | Episode severity threshold (high severity). | Rarely |
| `min_len` | 3 | Minimum episode length in samples. | Rarely |
| `gap_merge` | 5 | Merge episodes separated by fewer than N samples. | Rarely |
| `cpd.auto_tune.enabled` | True | Auto-adjust thresholds based on anomaly rate. | No |

---

### THRESHOLDS CATEGORY - Alert Levels

| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `q` | 0.98 | Calibration quantile for z-score thresholds. | Rarely |
| **`self_tune.clip_z`** | 100.0 | Maximum z-score cap. **Reduce if high saturation warnings**. | **YES - auto-tuned for saturating detectors** |
| `self_tune.target_fp_rate` | 0.001 | Target false positive rate for adaptive thresholds. | Rarely |
| `alert` | 0.85 | Health fraction triggering ALERT status. | Rarely |
| `warn` | 0.7 | Health fraction triggering CAUTION status. | Rarely |
| `adaptive.enabled` | True | Enable per-regime adaptive thresholds. | No |
| `adaptive.confidence` | 0.997 | Confidence level (99.7% = 3-sigma). | Rarely |
| `adaptive.fallback_threshold` | 3.0 | Default z-threshold if calculation fails. | Rarely |

---

### REGIMES CATEGORY - Operating Mode Clustering

| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `auto_k.k_min` / `auto_k.k_max` | 2/6 | Range for regime count search. | YES if known modes |
| `quality.silhouette_min` | 0.3 | Minimum clustering quality. | Rarely |
| `smoothing.passes` | 3 | Regime smoothing passes. | Rarely |
| `smoothing.window` | 7 | Smoothing window size. | Rarely |
| `smoothing.min_dwell_samples` | 10 | Minimum samples to stay in regime. | YES for fast-cycling equipment |
| `smoothing.min_dwell_seconds` | 900 | Minimum regime duration (15 min). | YES for fast-cycling equipment |
| `health.fused_warn_z` | 2.5 | Z-score for health warning zone. | Rarely |
| `health.fused_alert_z` | 4.0 | Z-score for health alert zone. | Rarely |

---

### DRIFT CATEGORY - Slow Change Detection

| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `cusum.threshold` | 2.0 | CUSUM drift detection threshold. | Rarely |
| `cusum.smoothing_alpha` | 0.3 | EWMA smoothing for drift curves. | Rarely |
| `cusum.drift` | 0.1 | Drift allowance before detection. | Rarely |
| `p95_threshold` | 2.0 | P95 threshold for drift vs fault distinction. | Rarely |
| `multi_feature.enabled` | True | Multi-feature drift detection. | No |

---

### FORECASTING CATEGORY - Health & RUL Prediction

| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `enhanced_enabled` | True | Enable enhanced forecasting module. | No |
| `enable_continuous` | True | Enable continuous stateful forecasting. | No |
| `failure_threshold` | 70.0 | Health % defining failure. | Rarely |
| `max_forecast_hours` | 168.0 | Maximum forecast horizon (7 days). | Rarely |
| `forecast_horizons` | [24, 72, 168] | Horizons to compute (hours). | Rarely |
| `training_window_hours` | 72 | Lookback window for training. | YES if sparse data |
| `hazard_smoothing_alpha` | 0.3 | Hazard rate smoothing. | Rarely |
| `hazard_failure_prob` | 0.6 | Probability level for failure time. | Rarely |
| `multivariate.enabled` | True | VAR-based sensor forecasting. | No |

---

### RUNTIME CATEGORY - Execution Control

| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `storage_backend` | sql | Storage mode: sql or file. | YES if equipment uses file mode |
| `tick_minutes` | 30 | Batch interval in minutes. | YES if different cadence |
| `future_grace_minutes` | 1051200 | Allow historian backfill (2 years). | Rarely |
| `log_level` | INFO | Logging verbosity. | No |
| `max_fill_ratio` | 0.2 | Maximum missing data ratio to accept. | Rarely |
| `phases.*` | True | Enable/disable pipeline phases. | Rarely |

---

### HEALTH CATEGORY - Health Index Calculation

| Parameter | Default | Description | Equipment-Specific? |
|-----------|---------|-------------|---------------------|
| `smoothing_alpha` | 0.3 | Exponential smoothing (0=smooth, 1=raw). | Rarely |
| `max_change_per_period` | 20.0 | Max health change % for volatility flag. | Rarely |
| `extreme_z_threshold` | 10.0 | Z-score for extreme anomaly flag. | Rarely |
| `z_threshold` | 5.0 | Z-score at ~15% health (sigmoid). | Rarely |
| `steepness` | 1.5 | Sigmoid steepness. | Rarely |

---

### Equipment-Specific Override Examples

Current overrides in production:

| EquipID | Equipment | Parameter | Value | Reason |
|---------|-----------|-----------|-------|--------|
| 1 | FD_FAN | `mahl.regularization` | 1.0 | High condition number fix |
| 1 | FD_FAN | `self_tune.clip_z` | 100.0 | High saturation |
| 1 | FD_FAN | `episodes.cpd.k_sigma` | 4.0 | High anomaly rate |
| 2621 | GAS_TURBINE | `data.timestamp_col` | Ts | Different column name |
| 2621 | GAS_TURBINE | `self_tune.clip_z` | 43.2 | High saturation |
| 2621 | GAS_TURBINE | `episodes.cpd.k_sigma` | 4.0 | High anomaly rate |
| 2621 | GAS_TURBINE | `runtime.tick_minutes` | 1440 | Hourly cadence |
| 8634 | ELECTRIC_MOTOR | **`data.sampling_secs`** | **60** | **1-minute data cadence** |

---

### Adding Equipment-Specific Overrides

1. **Find EquipID**: `SELECT EquipID, EquipCode FROM Equipment WHERE EquipCode = 'YOUR_CODE'`

2. **Add row to config_table.csv**:
   ```csv
   8634,data,sampling_secs,60,int,2025-12-13 00:00:00,COPILOT,ELECTRIC_MOTOR has 1-minute data cadence,
   ```

3. **Sync to SQL**:
   ```powershell
   python scripts/sql/populate_acm_config.py
   ```

4. **Reset coldstart** (if needed):
   ```sql
   SET QUOTED_IDENTIFIER ON;
   DELETE FROM ACM_ColdstartState WHERE EquipID = 8634;
   ```

---

### Auto-Tuning Parameters

ACM automatically tunes these parameters based on observed behavior:

| Parameter | Trigger | What Happens |
|-----------|---------|--------------|
| `mahl.regularization` | High condition number warning | Increased to stabilize covariance matrix |
| `self_tune.clip_z` | High z-score saturation (>15%) | Reduced to prevent detector clipping |
| `episodes.cpd.k_sigma` | High anomaly rate (>10%) | Increased to reduce false episodes |
| `fusion.weights.*` | Episode separability analysis | Weights adjusted based on detector contribution |

Auto-tuned values are logged in ACM_RunLogs with `[ADAPTIVE]` prefix.

---

### Configuration Validation Checklist for New Equipment

1. **Data Cadence**: Verify `sampling_secs` matches actual data interval
   - Check: `SELECT TOP 10 EntryDateTime, DATEDIFF(SECOND, LAG(EntryDateTime) OVER (ORDER BY EntryDateTime), EntryDateTime) AS GapSecs FROM {Equipment}_Data ORDER BY EntryDateTime`
   
2. **Timestamp Column**: Verify column name exists
   - Check: `SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{Equipment}_Data' AND COLUMN_NAME LIKE '%time%'`

3. **Tag Mapping**: Verify all sensors are mapped
   - Check: `SELECT * FROM ACM_TagEquipmentMap WHERE EquipID = N AND IsActive = 1`

4. **Coldstart Requirements**: Verify sufficient data exists
   - Need: `min_train_samples` (default 200) / (60 / `sampling_secs`) hours of data minimum
   - Example: 200 samples at 60-second cadence = 200 minutes = 3.3 hours
