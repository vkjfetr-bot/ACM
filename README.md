# ACM V8 - Autonomous Asset Condition Monitoring

ACM V8 is a multi-detector pipeline for autonomous asset condition monitoring. It combines structured feature engineering, an ensemble of statistical and ML detectors, drift-aware fusion, and flexible outputs so that engineers can understand what is changing, when it started, and which sensors or regimes are responsible.

For a complete, implementation-level walkthrough (architecture, modules, configs, operations, and reasoning), see `docs/ACM_SYSTEM_OVERVIEW.md`.

## What ACM is

ACM watches every asset through several analytical "heads" instead of a single anomaly score. Each head covers a different signal property - temporal self-consistency, covariance structure, clustering shifts, rare local patterns, and model residuals - so ACM can characterize oscillations, fouling, regime changes, sensor jumps, and broken cross-variable couplings. Drift tracking, adaptive tuning, and episode culprits make the outcomes actionable.

## How it works

1. **Ingestion layer:** Baseline (train) and batch (score) inputs come from CSV files or a SQL source that populate the `data/` directory. Configuration values live in `configs/config_table.csv`, while SQL credentials are in `configs/sql_connection.ini`. ACM infers the equipment code (`--equip`) and determines whether to stay in file mode or engage SQL mode.
2. **Feature engineering:** `core.fast_features` delivers vectorized transforms (windowing, FFT, correlations, etc.) and can leverage Polars acceleration when available.
3. **Detectors:** Each head (Mahalanobis, PCA SPE/T2, Isolation Forest, Gaussian Mixture, AR1 residuals, Overall Model Residual, correlation/drift monitors, CUSUM-style trackers) produces interpretable scores, and episode culprits highlight which tag groups caused the response.
4. **Fusion & tuning:** `core.fuse` blends scores under configurable weights while `core.analytics.AdaptiveTuning` adjusts thresholds and logs every change via `core.config_history_writer`.
5. **Outputs:** `core.output_manager.OutputManager` writes CSV/PNG artifacts, SQL run logs, Grafana-ready dashboards, and stores models in `artifacts/{equip}/models`. SQL runners can call `usp_ACM_StartRun`/`usp_ACM_FinalizeRun` when the config enables it.

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
- `docs/` and `grafana_dashboards/`: design notes, integration plans, dashboards, and operator guides.

For more detail on SQL integration, dashboards, or specific detectors, consult the markdown files under `docs/` and `grafana_dashboards/docs/`.
