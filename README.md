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
