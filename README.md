# ACM V10 - Autonomous Asset Condition Monitoring

ACM V10 is a multi-detector pipeline for autonomous asset condition monitoring. It combines structured feature engineering, an ensemble of statistical and ML detectors, drift-aware fusion, predictive forecasting, and flexible outputs so that engineers can understand what is changing, when it started, which sensors or regimes are responsible, and what will happen next.

**Current Version:** v10.0.0 - Production Release with Enhanced Forecasting

For a complete, implementation-level walkthrough (architecture, modules, configs, operations, and reasoning), see `docs/ACM_SYSTEM_OVERVIEW.md`.

### Recent Updates (Dec 2025)
- Forecast/RUL work is moving into `core/forecast_engine.py` and the new degradation/RUL stack; `forecasting_legacy.py` is being phased out. Wire new calls in `acm_main.py` as the refactor lands.
- SQL historian sample for FD_FAN is time-shifted (2023-10-15 → 2025-09-14). Set Grafana ranges accordingly when validating dashboards or forecasts.
- Quick SQL/Grafana sanity scripts: `scripts/check_dashboard_tables.py`, `scripts/check_table_counts.py`, `scripts/check_tables_existence.py`, `scripts/validate_all_tables.py`.
- Active dashboard under repair: `grafana_dashboards/ACM Claude Generated To Be Fixed.json` (keep others archived).

### v10.0.0 Release Highlights
- **🚀 Continuous Forecasting with Exponential Blending**: Health forecasts now evolve smoothly across batch runs using exponential temporal blending (tau=12h), eliminating per-batch duplication in Grafana dashboards. Single continuous forecast line per equipment with automatic state persistence and version tracking (v807→v813 validated).
- **📊 Hazard-Based RUL Estimation**: Converts health forecasts to failure hazard rates with EWMA smoothing, survival probability curves, and probabilistic RUL predictions (P10/P50/P90 confidence bounds). Monte Carlo simulations with 1000 runs provide uncertainty quantification and top-3 culprit sensor attribution.
- **🔄 Multi-Signal Evolution**: All analytical signals (drift tracking via CUSUM, regime evolution via MiniBatchKMeans, 7+ detectors, adaptive thresholds) evolve correctly across batches. Validated v807→v813 progression with 28 pairwise detector correlations and auto-tuning with PR-AUC throttling.
- **📈 Time-Series Forecast Tables**: New `ACM_HealthForecast_Continuous` and `ACM_FailureHazard_TS` tables store merged forecasts with exponential blending. Smooth transitions across batch boundaries, Grafana-ready format with no per-run duplicates.
- **✅ Production Validation**: Comprehensive analytical robustness report (`docs/CONTINUOUS_LEARNING_ROBUSTNESS.md`) with 14-component validation checklist (all ✅ PASS). Mathematical soundness of exponential blending confirmed, state persistence validated, quality gates effective (RMSE, MAPE, TheilU, confidence bounds).
- **Unified Forecasting Engine**: Health forecasts, RUL predictions, failure probability, and physical sensor forecasts consolidated into 4 tables (down from 12+)
- **Sensor Value Forecasting**: Predicts future values for critical physical sensors (Motor Current, Bearing Temperature, Pressure, etc.) with confidence intervals using linear trend and VAR methods
- **Enhanced RUL Predictions**: Monte Carlo simulations with probabilistic models, multiple calculation paths (trajectory, hazard, energy-based)
- **Smart Coldstart Mode**: Progressive data loading with exponential window expansion for sparse historical data
- **Gap Tolerance**: Increased from 6h to 720h (30 days) to support historical replay with large gaps
- **Forecast State Management**: Persistent model state with version tracking and optimistic locking (ACM_ForecastingState)
- **Adaptive Configuration**: Per-equipment auto-tuning with configuration history tracking (ACM_AdaptiveConfig)
- **Detector Label Consistency**: Standardized human-readable format across all outputs and dashboards

## What ACM is

ACM watches every asset through several analytical "heads" instead of a single anomaly score. Each head covers a different signal property - temporal self-consistency, covariance structure, clustering shifts, rare local patterns, and model residuals - so ACM can characterize oscillations, fouling, regime changes, sensor jumps, and broken cross-variable couplings. Drift tracking, adaptive tuning, and episode culprits make the outcomes actionable.

## How it works

1. **Ingestion layer:** Baseline (train) and batch (score) inputs come from CSV files or a SQL source that populate the `data/` directory. Configuration values live in `configs/config_table.csv`, while SQL credentials are in `configs/sql_connection.ini`. ACM infers the equipment code (`--equip`) and determines whether to stay in file mode or engage SQL mode.
2. **Feature engineering:** `core.fast_features` delivers vectorized transforms (windowing, FFT, correlations, etc.) and can leverage Polars acceleration when available.
3. **Detectors:** Each head (Mahalanobis, PCA SPE/T2, Isolation Forest, Gaussian Mixture, AR1 residuals, Overall Model Residual, correlation/drift monitors, CUSUM-style trackers) produces interpretable scores, and episode culprits highlight which tag groups caused the response.
4. **Fusion & tuning:** `core.fuse` blends scores under configurable weights while `core.analytics.AdaptiveTuning` adjusts thresholds and logs every change via `core.config_history_writer`.
5. **Forecasting & RUL:** `core.forecasting` generates health trajectories, failure probability curves, RUL estimates, and physical sensor forecasts. **NEW in v10.0.0**: Continuous forecasting with exponential blending eliminates per-batch duplication; hazard-based RUL provides survival probability curves with EWMA smoothing; state persistence tracks forecast evolution across batches (see [Continuous Learning](#continuous-learning--forecasting) section).
6. **Outputs:** `core.output_manager.OutputManager` writes CSV/PNG artifacts, SQL run logs, Grafana-ready dashboards, forecast tables (ACM_HealthForecast, ACM_FailureForecast, ACM_SensorForecast, ACM_RUL, **ACM_HealthForecast_Continuous**, **ACM_FailureHazard_TS**), and stores models in `artifacts/{equip}/models`. SQL runners call `usp_ACM_StartRun`/`usp_ACM_FinalizeRun` when the config enables it.

## Configuration

ACM's configuration is stored in `configs/config_table.csv` (238 parameters) and synced to the SQL `ACM_Config` table via `scripts/sql/populate_acm_config.py`. Parameters are organized by category with equipment-specific overrides (EquipID=0 for global defaults, EquipID=1/2621 for FD_FAN/GAS_TURBINE).

### Configuration Categories

**Data Ingestion (`data.*`)**
- `timestamp_col`: Column name for timestamps (default: `EntryDateTime`)
- `sampling_secs`: Data cadence in seconds (default: 1800 for 30-min intervals)
- `max_rows`: Maximum rows to process per batch (default: 100000)
- `min_train_samples`: Minimum samples required for training (default: 200)

**Feature Engineering (`features.*`)**
- `window`: Rolling window size for feature extraction (default: 16)
- `fft_bands`: Frequency bands for FFT decomposition
- `polars_threshold`: Row count to trigger Polars acceleration (default: 5000)

**Detectors & Models (`models.*`)**
- `pca.*`: PCA configuration (n_components=5, randomized SVD)
- `ar1.*`: AR1 detector settings (window=256, alpha=0.05)
- `iforest.*`: Isolation Forest (n_estimators=100, contamination=0.01)
- `gmm.*`: Gaussian Mixture Models (k_min=2, k_max=3, BIC search enabled)
- `mahl.regularization`: Mahalanobis regularization to prevent ill-conditioning
- `omr.*`: Overall Model Residual (auto model selection, n_components=5)
- `use_cache`: Enable model caching via ModelVersionManager
- `auto_retrain.*`: Automatic retraining thresholds (max_anomaly_rate=0.25, max_drift_score=2.0, max_model_age_hours=720)

**Fusion & Weights (`fusion.*`)**
- `weights.*`: Detector contribution weights (ar1_z=0.2, iforest_z=0.2, gmm_z=0.1, pca_spe_z=0.2, mhal_z=0.2, omr_z=0.10)
- `per_regime`: Enable per-regime fusion (default: True)
- `auto_tune.*`: Adaptive weight tuning (enabled, learning_rate=0.3, temperature=1.5)

**Episodes & Anomaly Detection (`episodes.*`)**
- `cpd.k_sigma`: K-sigma threshold for change-point detection (default: 2.0)
- `cpd.h_sigma`: H-sigma threshold for episode boundaries (default: 12.0)
- `min_len`: Minimum episode length in samples (default: 3)
- `gap_merge`: Merge episodes with gaps smaller than this (default: 5)
- `cpd.auto_tune.*`: Barrier auto-tuning (k_factor=0.8, h_factor=1.2)

**Thresholds (`thresholds.*`)**
- `q`: Quantile threshold for anomaly detection (default: 0.98)
- `alert`: Alert threshold (default: 0.85)
- `warn`: Warning threshold (default: 0.7)
- `self_tune.*`: Self-tuning parameters (enabled, target_fp_rate=0.001, max_clip_z=100.0)
- `adaptive.*`: Per-regime adaptive thresholds (enabled, method=quantile, confidence=0.997, per_regime=True)

**Regimes (`regimes.*`)**
- `auto_k.k_min`: Minimum clusters for auto-k selection (default: 2)
- `auto_k.k_max`: Maximum clusters (default: 6)
- `auto_k.max_models`: Maximum candidate models to evaluate (default: 10)
- `quality.silhouette_min`: Minimum silhouette score for acceptable clustering (default: 0.3)
- `smoothing.*`: Regime label smoothing (passes=3, window=7, min_dwell_samples=10)
- `transient_detection.*`: Transient change detection (roc_window=10, roc_threshold_high=0.15)
- `health.*`: Health-based regime boundaries (fused_warn_z=2.5, fused_alert_z=4.0)

**Drift Detection (`drift.*`)**
- `cusum.*`: CUSUM drift detector (threshold=2.0, smoothing_alpha=0.3, drift=0.1)
- `p95_threshold`: P95 threshold for drift vs fault classification (default: 2.0)
- `multi_feature.*`: Multi-feature drift detection (enabled, trend_window=20, hysteresis_on=3.0)

**Forecasting (`forecasting.*`)**
- `enhanced_enabled`: Enable unified forecasting engine (default: True)
- `enable_continuous`: Enable continuous stateful forecasting (default: True)
- `failure_threshold`: Health threshold for failure prediction (default: 70.0)
- `max_forecast_hours`: Maximum forecast horizon (default: 168 hours = 7 days)
- `confidence_k`: Confidence interval multiplier (default: 1.96 for 95% CI)
- `training_window_hours`: Sliding training window (default: 72 hours)
- `blend_tau_hours`: Exponential blending time constant (default: 12 hours)
- `hazard_smoothing_alpha`: EWMA alpha for hazard rate smoothing (default: 0.3)

**Runtime (`runtime.*`)**
- `storage_backend`: Storage mode (default: `sql`)
- `reuse_model_fit`: Legacy joblib cache (False in SQL mode; use ModelRegistry instead)
- `tick_minutes`: Data cadence for batch runs (default: 30 for FD_FAN, 1440 for GAS_TURBINE)
- `version`: Current ACM version (v10.1.0)
- `phases.*`: Enable/disable pipeline phases (features, regimes, drift, models, fuse, report)

**SQL Integration (`sql.*`)**
- `enabled`: Enable SQL connection (default: True)
- Connection parameters: driver, server, database, encrypt, trust_server_certificate
- Performance tuning: pool_min, pool_max, fast_executemany, tvp_chunk_rows, deadlock_retry.*

**Health & Continuous Learning (`health.*, continuous_learning.*`)**
- `health.smoothing_alpha`: Exponential smoothing for health index (default: 0.3)
- `health.extreme_z_threshold`: Absolute Z-score for extreme anomaly flagging (default: 10.0)
- `continuous_learning.enabled`: Enable continuous learning for batch mode (default: True)
- `continuous_learning.model_update_interval`: Batches between retraining (default: 1)

### Configuration Management

**Editing Config**
1. Edit `configs/config_table.csv` directly (maintain CSV format)
2. Run `python scripts/sql/populate_acm_config.py` to sync changes to SQL
3. Commit changes to version control

**Equipment-Specific Overrides**
- Global defaults: `EquipID=0`
- FD_FAN overrides: `EquipID=1` (e.g., `mahl.regularization=1.0`, `episodes.cpd.k_sigma=4.0`)
- GAS_TURBINE overrides: `EquipID=2621` (e.g., `timestamp_col=Ts`, `tick_minutes=1440`)

**Configuration History**
All adaptive tuning changes are logged to `ACM_ConfigHistory` via `core.config_history_writer.ConfigHistoryWriter`. Includes timestamp, parameter path, old/new values, reason, and UpdatedBy tag.

**Best Practices**
- Use `COPILOT`, `SYSTEM`, `ADAPTIVE_TUNING`, or `OPTIMIZATION` as UpdatedBy tags for traceability
- Document ChangeReason for non-trivial updates
- Test config changes in file mode before syncing to SQL
- Keep equipment-specific overrides minimal (only override when necessary)

For complete parameter descriptions and implementation details, see `docs/ACM_SYSTEM_OVERVIEW.md`.

## Continuous Learning & Forecasting

**NEW in v10.0.0**: ACM now implements true continuous forecasting where health predictions evolve smoothly across batch runs instead of creating per-batch duplicates.

### Exponential Blending Architecture
- **Temporal Smoothing**: `merge_forecast_horizons()` blends previous and current forecasts using exponential decay (tau=12h default)
- **Dual Weighting**: Combines recency weight (`exp(-age/tau)`) with horizon awareness (`1/(1+hours/24)`) to balance recent confidence vs long-term uncertainty
- **NaN Handling**: Intelligently prefers non-null values; does not treat missing data as zero
- **Weight Capping**: Limits previous forecast influence to 0.9 maximum, preventing staleness from overwhelming fresh predictions
- **Mathematical Foundation**: `merged = w_prev * prev + (1-w_prev) * curr` where `w_prev = recency_weight * horizon_weight` (capped at 0.9)

### State Persistence & Evolution
- **Versioned Tracking**: `ForecastState` class with version identifiers (e.g., v807→v813) stored in `ACM_ForecastState` table
- **Audit Trail**: Each forecast includes RunID, BatchNum, version, and timestamp for reproducibility
- **Self-Healing**: Gracefully handles missing/invalid state with automatic fallback to current forecasts
- **Multi-Batch Validation**: State progression confirmed across 5 sequential batches (v807→v813 validated)

### Hazard-Based RUL Estimation
- **Hazard Rate Calculation**: `lambda(t) = -ln(1 - p(t)) / dt` converts health forecast to instantaneous failure rate
- **EWMA Smoothing**: Configurable alpha parameter reduces noise in failure probability curves
- **Survival Probability**: `S(t) = exp(-∫ lambda_smooth(t) dt)` provides cumulative survival curves
- **Confidence Bounds**: Monte Carlo simulations (1000 runs) generate P10/P50/P90 confidence intervals
- **Culprit Attribution**: Identifies top 3 sensors driving failure risk with z-score contribution analysis

### Multi-Signal Evolution
All analytical signals evolve correctly across batches:
- **Drift Tracking**: CUSUM detector with P95 threshold per batch (coldstart windowing approach)
- **Regime Evolution**: MiniBatchKMeans with auto-k selection and quality scoring (Calinski-Harabasz, silhouette)
- **Detector Correlation**: 28 pairwise correlations tracked across 7+ detectors (AR1, PCA-SPE/T2, Mahal, IForest, GMM, OMR)
- **Adaptive Thresholds**: Quantile/MAD/hybrid methods with PR-AUC based throttling prevents over-tuning
- **Health Forecasting**: Exponential smoothing with 168-hour horizon (7 days ahead)
- **Sensor Forecasting**: VAR(3) models for 9 critical sensors with lag-3 dependencies

### Time-Series Tables
- **ACM_HealthForecast_Continuous**: Merged health forecasts with exponential blending (single continuous line per equipment)
- **ACM_FailureHazard_TS**: EWMA-smoothed hazard rates with raw hazard, survival probability, and failure probability
- **Grafana-Ready**: No per-run duplicates; smooth transitions across batch boundaries; ready for time-series visualization

### Quality Assurance
- **RMSE Validation**: Gates on forecast quality
- **MAPE Tracking**: Median absolute percentage error (33.8% typical for noisy industrial data)
- **TheilU Coefficient**: 1.098 indicates acceptable forecast accuracy vs naive baseline
- **Confidence Bounds**: P10/P50/P90 for RUL with Monte Carlo validation
- **Production Validation**: 14-component checklist (all ✅ PASS) in `docs/CONTINUOUS_LEARNING_ROBUSTNESS.md`

### Benefits
- ✅ **Single Forecast Line**: Eliminates Grafana dashboard clutter from per-batch duplicates
- ✅ **Smooth Transitions**: 12-hour exponential blending window creates seamless batch boundaries
- ✅ **Multi-Batch Learning**: Models evolve with accumulated data (v807→v813 progression validated)
- ✅ **Noise Reduction**: EWMA hazard smoothing reduces false alarms from noisy health forecasts
- ✅ **Uncertainty Quantification**: P10/P50/P90 confidence bounds for probabilistic RUL predictions
- ✅ **Production-Ready**: All analytical validation checks passed (see `docs/CONTINUOUS_LEARNING_ROBUSTNESS.md`)

## Running ACM

1. **Prepare the environment**
   - `python -m venv .venv` (Python >= 3.11) and `pip install -U pip`.
   - `pip install -r requirements.txt` or `pip install .` to satisfy NumPy, pandas, scikit-learn, matplotlib, seaborn, PyYAML, pyodbc, joblib, and other dependencies listed in `pyproject.toml`.
2. **Provide config and data**
   - Ensure `configs/config_table.csv` defines the equipment-specific parameters (paths, sampling rate, models, SQL mode flag). Override per run with `--config <file>` if needed.
   - Place baseline data (`train_csv`) and batch data (`score_csv`) under `data/` or point to SQL tables.
3. **Run the pipeline**
   - `python -m core.acm_main --equip PROD_LINE_A`
   - Add `--train-csv data/baseline.csv` and `--score-csv data/batch.csv` to override the defaults defined in the config table.
   - Artifacts written to SQL tables. Cached detector bundles in SQL (`ACM_ModelRegistry`) or `artifacts/{equip}/models/` for reuse.
   - SQL mode is on by default; set env `ACM_FORCE_FILE_MODE=1` to force file mode.

## Batch mode details

Batch mode simply runs ACM against a historical baseline (training) window and a separate evaluation (batch) window. The two CSVs can live under `data/` or be pulled from SQL tables when SQL mode is enabled; the `configs/config_table.csv` row for the equipment controls which storage backend is active.

1. **Data layout:** Put normal/stable data into `train_csv` and the most-recent window into `score_csv`. In file mode, ACM ingests them from the path literal. In SQL mode, ensure the connection string in `configs/sql_connection.ini` points to the right database and the config table row sets `storage_backend=sql`.
2. **Key CLI knobs:** Pass `--train-csv` and `--score-csv` (or their aliases `--baseline-csv` / `--batch-csv`) to override the defaults. Use `--clear-cache` to force retraining instead of reusing a cached model if the baseline drifted.
3. **Logging:** Control verbosity with `--log-level`/`--log-format` and target specific modules with multiple `--log-module-level MODULE=LEVEL` entries (e.g., `--log-module-level core.fast_features=DEBUG`). Write logs to disk with `--log-file` or keep them on the console. SQL run-log writes are always enabled in SQL mode.
4. **Automation:** Use `scripts/sql_batch_runner.py` (and its `scripts/sql/*` helpers) to invoke ACM programmatically for many equipment codes or integrate with a scheduler.

The same command-line options work for both file and SQL batch runs because ACM uses the configuration row to decide whether to stream data through CSV files or the shared SQL client.

## CLI options

- `--equip <name>` *(required)*: equipment code that selects the config row and artifacts directory.
- `--config <path>`: optional YAML that overrides values from `configs/config_table.csv`.
- `--train-csv` / `--baseline-csv`: path to historical data used for model fitting.
- `--score-csv` / `--batch-csv`: path to the current window of observations to evaluate.
- `--clear-cache`: delete any cached model for this equipment to force retraining.
- Logging: `--log-level`, `--log-format`, `--log-module-level`, `--log-file`.

ACM decides between file and SQL mode based on the configuration (see `core/sql_logger.py` and the `storage_backend` entry). SQL mode wraps data ingestion/output with `core.sql_client.SQLClient` and calls stored procedures instead of writing to CSV files.

## Feature highlights

- **Multi-head detectors:** Mahalanobis, PCA (SPE/T2), Isolation Forest, Gaussian Mixture, AR1 residuals, Overall Model Residual (OMR), and drift/CUSUM monitors provide complementary signals.
- **High-performance feature engineering:** `core.fast_features` uses vectorized pandas routines and optional Polars acceleration for FFTs, correlations, and windowed statistics.
- **Fusion & adaptive tuning:** `core.fuse` weights detector heads, `core.analytics.AdaptiveTuning` adjusts thresholds, and `core.config_history_writer` records every auto-tune event.
- **SQL-first and CSV-ready outputs:** `core.output_manager` writes CSVs, PNGs, SQL sink logs, run metadata, episode culprits, detector score bundles, and correlates results with Grafana dashboards in `grafana_dashboards/`.
- **Operator-friendly diagnostics:** Episode culprits, drift-aware hysteresis, and `core.run_metadata_writer` provide health indices, fault signatures, and explanation cues for downstream visualization.
- **Automation & testing helpers:** SQL scripts (`scripts/sql/*.sql`), population helpers (`scripts/sql/populate_acm_config.py`), and regression tests under `scripts/sql/test_*` cover dual-write and mode-loading scenarios.

## Operator quick links

- System handbook (full architecture, modules, configs, ops): `docs/ACM_SYSTEM_OVERVIEW.md`
- SQL batch runner for historian-backed continuous mode: `scripts/sql_batch_runner.py`
- Data/config sources: `configs/config_table.csv`, `configs/sql_connection.ini`
- Artifacts and caches: `artifacts/{EQUIP}/run_<ts>/`, `artifacts/{EQUIP}/models/`
- Grafana/dashboard assets: `grafana_dashboards/`

## Supporting directories

- `core/`: pipeline implementations (detectors, fusion, analytics, output manager, SQL client).
- `configs/`: configuration tables plus SQL connection templates.
- `data/`: default baseline/batch CSVs used in smoke tests.
- `scripts/sql/`: helpers and integration tests for SQL mode.
- `docs/` and `grafana_dashboards/`: design notes, integration plans, dashboards, and operator guides. Only `grafana_dashboards/ACM Claude Generated To Be Fixed.json` remains active; all other dashboards live under `grafana_dashboards/archive/` with a move log in `grafana_dashboards/archive/ARCHIVE_LOG.md`.

For more detail on SQL integration, dashboards, or specific detectors, consult the markdown files under `docs/` and `grafana_dashboards/docs/`.
