# ACM V8 - System Handbook

This handbook is a complete, implementation-level walkthrough of ACM V8 for new maintainers. It covers the end-to-end data flow, the role of every module, configuration surfaces, and the reasoning behind each major decision so that a new engineer can operate, extend, and hand off the system confidently.

---

## 1) Mental Model (Top-Level Flow)

```
        +--------------+    +-----------------+    +----------------+    +--------------------+
        | Ingestion    |    | Feature Builder |    | Detector Heads |    | Fusion & Episodes  |
        | (CSV / SQL)  | -> | (fast_features) | -> | (PCA/MHAL/IF/GMM|
        | acm_main     |    |                 |    |  AR1/OMR/etc.)  |    |  (fuse)            |
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
  - `utils/logger.py`, `utils/timer.py`: Console logging, SQL sink integration, heartbeat, timing helpers.

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

---

## 3) Configuration Surfaces

### Primary table (`configs/config_table.csv` or SQL `ACM_Config`)
Key paths (ParamPath) and reasoning:
- `data.train_csv`, `data.score_csv`, `data_dir`, `timestamp_col`, `tag_columns`, `sampling_secs`, `max_rows`: defines ingestion sources and schema expectations.
- `features.window`, `features.fft_bands`, `features.top_k_tags`, `features.fs_hz`: controls rolling window size and spectral bins; window must match process dynamics to capture oscillations without oversmoothing.
- `models.*` (pca, ar1, iforest, gmm, omr): detector hyperparameters; tighter contamination lowers false positives, higher `pca.n_components` trades explainability vs. residual sensitivity.
- `thresholds.*`: fused/detector z clipping, quantiles for calibration.
- `fusion.*`: detector weights and auto-tuning knobs (episode separability-based).
- `regimes.*`: auto-k bounds, smoothing, transient detection, health thresholds; smaller `k_min/k_max` avoids over-segmentation on short baselines.
- `drift.*`: CUSUM thresholds and drift aggregation.
- `runtime.*`: version tag, heartbeat, reuse_model_fit, baseline buffer, phases toggles.
- `output.*`: dual_mode (file + SQL), enable_forecast, enable_enhanced_forecast, destinations. (Strategic direction: SQL-first; retire file-only branches over time.)

**Config sync:** when `configs/config_table.csv` changes, run `python scripts/sql/populate_acm_config.py` to update `ACM_Config` so SQL mode stays authoritative.  
**Config history:** `ACM_ConfigHistory` tracks changes when enabled.

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
- **Heads fit/score:** PCA (PCASubspaceDetector), Mahalanobis, IsolationForest, GMM, AR1 (forecasting.AR1Detector), OMR (core.omr), optional River streaming. Reuses cached detectors when signature matches.
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

### Detectors
- **Correlation (`core/correlation.py`):**
  - `MahalanobisDetector`: ridge-regularized covariance with NaN audits; guards ill-conditioning and under-sampled cases.
  - `PCASubspaceDetector`: cleans non-finite, drops constants, scales, fits PCA; returns SPE (Q) and T2; handles low-sample fallback.
- **Outliers (`core/outliers.py`):**
  - `IsolationForestDetector`: fits/uses scikit IF, stores columns, optional quantile threshold when contamination numeric.
  - `GMMDetector`: BIC-driven component selection, variance guards, scaling; returns neg log-likelihood style scores.
- **OMR (`core/omr.py`):**
  - Multivariate reconstruction error via PLS/linear ensemble/PCA. Features auto model selection, diagnostics, per-sensor contribution extraction, z clipping, min sample guards.
- **Forecasting/RUL (`core/forecasting.py`, `core/rul_estimator.py`, `core/enhanced_rul_estimator.py`):**
  - AR1 per-sensor residual detector, data hash for retrain decisions, SQL/file dual persistence of forecast state.
  - Enhanced RUL: multiple degradation models (AR1, exponential, Weibull-inspired, linear), learning state, attribution, maintenance recommendations, hazard smoothing.
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
- **Grafana dashboards:** `grafana_dashboards/` holds JSON panels and docs; operator suite includes `acm_ops_command_center`, `acm_asset_health_deep_dive`, `acm_failure_maintenance_planner`, `acm_sensor_regime_forensics`, `acm_ops_model_observability` (all JSON import-ready).

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
0,output,dual_mode,false,bool
```

**SQL mode row (config_table.csv)**
```
EquipID,Category,ParamPath,ParamValue,ValueType
1,data,storage_backend,sql,string
1,data,timestamp_col,EntryDateTime,string
1,data,sampling_secs,60,int
1,models,pca.incremental,true,bool
1,runtime,reuse_model_fit,false,bool
1,output,dual_mode,true,bool
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
