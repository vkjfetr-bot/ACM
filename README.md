# ACM V8 - Autonomous Asset Condition Monitoring

ACM V8 is an autonomous condition monitoring pipeline for industrial equipment. 

## System Overview

- `python -m core.acm_main` orchestrates ingestion, feature engineering, detector fitting, scoring,

  and reporting.------

- `core/fast_features.py` delivers vectorized feature engineering with optional Polars acceleration.

- Detector implementations live under `core/` (Mahalanobis, PCA SPE/T2, Isolation Forest,  Gaussian Mixture, AR1 residual, Overall Model Residual).

- Fusion (`core/fuse.py`) combines detector scores; `core/episode_culprits_writer.py` highlights## System Overview## System Overview

  culprit sensors with drift-aware hysteresis from `core/drift.py`.

- Adaptive tuning (`core/analytics.AdaptiveTuning`) adjusts thresholds and fusion weights, logging

  changes with `core/config_history_writer.log_auto_tune_changes`.

- `core/output_manager.OutputManager` governs all CSV/PNG/SQL writes and enforces the local time- **Entry point**: `python -m core.acm_main` orchestrates ingestion, feature engineering, detector fitting, scoring, and reporting.- **Entry point**: `python -m core.acm_main` orchestrates ingestion, feature engineering, detector fitting, scoring, and reporting.

- `core/model_persistence.py` caches detector bundles under `artifacts/{equip}/models` by- **Feature engineering**: `core/fast_features.py` delivers vectorized pandas transforms with optional Polars acceleration.- **Feature engineering**: `core/fast_features.py` delivers vectorized pandas transforms with optional Polars acceleration.

  configuration signature.

- **Detectors**: Mahalanobis, PCA (SPE/T2), Isolation Forest, GMM, AR1 residuals, and the Overall Model Residual live under `core/`.- **Detectors**: Mahalanobis, PCA (SPE/T2), Isolation Forest, GMM, AR1 residuals, and the Overall Model Residual live under `core/`.

## How ACM Thinks: The Multi-Head Framework

**Version:** 2025-11-11  
**Purpose:** Explain, in human terms, what ACM does, why it uses many detectors, what "fusion" means, and how to interpret the outcomes.

ACM (Autonomous Condition Monitoring) is not a single algorithm — it is a **multi-view analytical system** that watches every asset from several perspectives simultaneously.

Each *detector head* captures a unique kind of abnormality. The goal is **not just to flag "something's wrong"**, but to **understand what kind of deviation** is happening, when it began, and which sensors are responsible.

### Detector Heads – Core Idea

| Head                             | Method                                          | Focus                                 | What It Detects                                                 |
| -------------------------------- | ----------------------------------------------- | ------------------------------------- | --------------------------------------------------------------- |
| **AR1**                          | 1-step autoregression                           | Temporal self-consistency             | Detects control oscillations, instability, erratic noise bursts |
| **PCA (SPE/T²)**                 | Principal Component projection & reconstruction | Statistical variance structure        | Detects broad data shape change, abnormal spread or new cluster |
| **Mahalanobis Distance**         | Multivariate covariance distance                | Geometric distance from normal cloud  | Detects uniform multi-sensor drift or scaling                   |
| **GMM (Gaussian Mixture Model)** | Probabilistic cluster membership                | Regime membership probability         | Detects entry into new operating condition or unseen regime     |
| **Isolation Forest**             | Tree-based isolation                            | Local density outliers                | Detects rare, abrupt or localized deviations                    |
| **OMR (Overall Model Residual)** | PLS / PCA / Ridge reconstruction                | Cross-sensor functional relationships | Detects broken coupling among process variables                 |
| **CUSUM / Drift Monitor**        | Sequential residual tracking                    | Slow trend evolution                  | Detects gradual degradation, fouling, efficiency loss           |

### What Each Detector Identifies in Physical Terms

Below is a *fault-mapping layer* — how each head translates to physical conditions commonly observed in rotating and process equipment.

#### AR1 – Dynamic Instability & Control Oscillation

| Fault Type                 | Example Physical Behavior                             |
| -------------------------- | ----------------------------------------------------- |
| PID tuning error           | Pressure, temperature, or flow oscillating cyclically |
| Sensor noise or chattering | Sudden erratic spikes uncorrelated to process load    |
| Mechanical looseness       | Vibration amplitude oscillations, alternating sign    |
| Electrical instability     | Current or speed jitter, fluctuating torque feedback  |

**Interpretation:** High AR1 z-score means **temporal predictability broke** — the signal no longer follows its own historical inertia.

#### PCA (SPE/T²) – Shape and Variance Anomalies

| Fault Type                   | Example Physical Behavior                   |
| ---------------------------- | ------------------------------------------- |
| Process fluctuation increase | Flow, pressure, or temp spread widening     |
| Nonlinearity introduction    | Process deviating from established manifold |
| Regime overlap               | System operating between two steady states  |
| Data saturation              | Sensor range limiting → compressed variance |

**Interpretation:** High PCA-SPE or T² indicates the **overall "cloud" of normal operation deformed** — e.g., more scattered, tilted, or squashed. Usually an *early symptom* of control loss, valve sticking, or feed variability.

#### Mahalanobis Distance – Global Shift or Scaling

| Fault Type               | Example Physical Behavior                      |
| ------------------------ | ---------------------------------------------- |
| Uniform temperature rise | All temperatures increase proportionally       |
| Common-mode bias         | Pressure and flow both rise together by offset |
| Calibration shift        | Sensor zero drift or scaling error             |
| Step change in baseline  | Operation under a new global condition         |

**Interpretation:** High Mahalanobis z means the **entire operating point shifted**, but internal relationships remain mostly consistent. Useful for catching *load changes, bias drifts, or setpoint moves*.

#### GMM – Regime Recognition / Novel Condition

| Fault Type                    | Example Physical Behavior                               |
| ----------------------------- | ------------------------------------------------------- |
| Startup vs steady operation   | Transitions between regimes not seen before             |
| Ambient or seasonal variation | Conditions outside historical distribution              |
| Process reconfiguration       | Valve sequencing changed, new product grade             |
| Major load change             | Flow, current, and temperature cluster into new pattern |

**Interpretation:** Low GMM probability indicates **the system entered a new statistical regime**. Not always a fault — often an *operating-mode change*. Helps isolate "new but healthy" vs "new and abnormal" contexts.

#### Isolation Forest – Localized, Sparse Outliers

| Fault Type           | Example Physical Behavior          |
| -------------------- | ---------------------------------- |
| Momentary spikes     | Sensor dropout or EMI noise        |
| One-time pulse       | Transient process upset or blowoff |
| Sudden discontinuity | Step fault (valve jam, trip event) |
| Sensor saturation    | Out-of-range single-sensor spike   |

**Interpretation:** High IForest score means **"this sample stands alone"** — rare behavior not repeated nearby in time. Acts as a *spike detector* or *transient marker* complementing drift detectors.

#### OMR – Cross-Sensor Coupling Breaks

| Fault Type         | Example Physical Behavior                                          |
| ------------------ | ------------------------------------------------------------------ |
| Process decoupling | Flow and pressure no longer move in sync                           |
| Efficiency loss    | Temperature and load diverge (e.g., fouling, heat loss)            |
| Sensor drift       | One variable stops tracking others though both change slowly       |
| Partial failure    | Vibration increases while power constant, or torque–speed mismatch |
| Valve sticking     | Pressure–flow response slope changes abnormally                    |

**Interpretation:** High OMR score means **the physical relationships between variables are no longer consistent with healthy physics**. It's the detector that senses **"behavioral inconsistency," not just appearance change.**

#### CUSUM / Drift Detector – Slow Degradation

| Fault Type                 | Example Physical Behavior                        |
| -------------------------- | ------------------------------------------------ |
| Bearing wear               | Slow rising vibration baseline                   |
| Heat exchanger fouling     | Gradual temp differential increase               |
| Filter choking             | Pressure drop creep                              |
| Sensor bias drift          | Output offset growing over time                  |
| Process loss of efficiency | Gradual divergence of expected vs achieved value |

**Interpretation:** Positive drift slope = **slow degradation**, often before any alarm is triggered. CUSUM provides early-warning of persistent bias trends.

### Fusion – The Consensus Layer

#### Why We Fuse

Fusion builds a **stable consensus** of these detectors instead of replacing them. It smooths noise, balances strengths, and adapts automatically.

* If *all* rise → clear fault.
* If *only one or two* rise → partial degradation.
* If they conflict → investigate whether a sensor or regime shift is occurring.

Fusion z-score = *"how confident we are something truly changed."*

#### What Head Contributions Mean

| Dominant Head         | Interpretation         | Typical Fault Signature                 |
| --------------------- | ---------------------- | --------------------------------------- |
| **AR1**               | Dynamic instability    | Fast oscillations, control loop hunting |
| **PCA / Mahalanobis** | Statistical distortion | Process spread or baseline shift        |
| **GMM / IForest**     | Novel regime           | Startup, untrained operating zone       |
| **OMR**               | Broken coupling        | Process decoupling, loss of efficiency  |
| **CUSUM / Drift**     | Slow degradation       | Wear, fouling, or drift buildup         |

### Reading ACM Outputs – The Human Hierarchy

```
Fused Health (Is it abnormal?)
   ↓
Dominant Head(s) (What kind of deviation?)
   ↓
Top Sensors (Where is it happening?)
```

| Level                    | Role           | Interpretation                                                     |
| ------------------------ | -------------- | ------------------------------------------------------------------ |
| **Fusion Score / Zone**  | Overall health | Red = consensus fault, Yellow = watch, Green = healthy             |
| **Head Mix / Type**      | Fault nature   | OMR↑ = physical decoupling; PCA↑ = variability; AR1↑ = instability |
| **Sensor Contributions** | Fault source   | OMR residuals show responsible sensors                             |
| **Regime Context**       | Operating mode | Helps confirm process state                                        |
| **Drift Metrics**        | Persistence    | Determines trend vs transient                                      |

### Physical Example: Gas Turbine Case

| Observation               | Detector Reaction         | Interpretation                        |
| ------------------------- | ------------------------- | ------------------------------------- |
| Sudden speed oscillation  | AR1 ↑, PCA ↑              | Control loop oscillation              |
| Gradual exhaust temp rise | Drift ↑, OMR ↑            | Efficiency degradation, fouling       |
| Flow-pressure decoupling  | OMR ↑↑, PCA normal        | Broken thermodynamic coupling         |
| Global temperature shift  | Mahalanobis ↑, OMR stable | Load or ambient condition change      |
| Noise spike in vibration  | IForest ↑ only            | Transient event, likely not sustained |
| Operation in startup mode | GMM ↓ probability         | Entered new regime; re-learn required |

### Operator-Facing Summary

| Display                  | Shows                           | Takeaway                    |
| ------------------------ | ------------------------------- | --------------------------- |
| **Health Gauge**         | Fused health (Green/Yellow/Red) | Is asset normal?            |
| **Deviation Type Chart** | Head contributions              | What kind of deviation?     |
| **Sensor Hotspot Map**   | OMR top contributors            | Which sensors or subsystem? |
| **Regime Tracker**       | Operating mode context          | Was it steady or transient? |
| **Drift Trend Graph**    | Slow degradation path           | Is this trending worse?     |

### Why Multiple Heads Are Still Needed

Even if several detectors correlate on healthy data, they **diverge under fault conditions** — and that divergence *defines* the fault character.

| Healthy State | All heads ≈ correlated (0.9+) → stable ensemble |
| Fault State | Different heads spike differently → diagnostic fingerprint |

This fingerprint enables automatic **fault typing** and better RCA (Root Cause Analysis).

### Summary Takeaways

1. **ACM doesn't just detect change — it categorizes it.**
2. **Fusion = confidence; head contributions = explanation.**
3. **OMR = relational truth detector** — tells whether physics between signals still holds.
4. **Operators see one score**, but **engineers can unpack the anatomy of deviation.**
5. The combination of **distributional, temporal, relational, and drift-based views** allows ACM to detect and classify almost every kind of degradation seen in rotating and process equipment.

---

## Quick Start (File Mode)

1. Place training and scoring CSV files in `data/` (datetime index plus sensor columns).- **Fusion and episodes**: `core/fuse.py` combines detector scores; `core/episode_culprits_writer.py` highlights culprit sensors with drift-aware hysteresis from `core/drift.py`.- **Fusion and episodes**: `core/fuse.py` combines detector scores; `core/episode_culprits_writer.py` highlights culprit sensors with drift-aware hysteresis from `core/drift.py`.

2. (Optional) Add equipment-specific overrides to `configs/config_table.csv`.

4. Run the pipeline:

   ```powershell
   python -m core.acm_main --equip FD_FAN
   ```

   The run always writes to `artifacts/FD_FAN/run_{timestamp}/` and emits the full report bundle by default.

## Quick Start (File Mode)



5. Review outputs in `artifacts/FD_FAN/run_{timestamp}/`.



Common CLI flags:

- `--score-start` and `--score-end` constrain backfill windows when replaying historian slices.
- `--train-csv` / `--score-csv` override the config-defined CSV paths for file-mode experiments.
- `--clear-cache` removes persisted detector bundles for a cold start (alias: `--no-cache`).
- `--log-level`, `--log-format`, `--log-file`, and `--log-module-level` control console/file logging.

Artifacts now always live under `artifacts/{EQUIP}/run_{timestamp}` and SQL ingestion is the default runtime.
Use `scripts/chunk_replay.py` or file-mode config overrides when you need to run against CSV snapshots.



## SQL Mode Checklist

1. Provision the ACM schema using helpers in `scripts/sql/` (see `docs/SQL_SCHEMA_DESIGN.md`).   ```powershell   ```powershell

2. Populate `configs/sql_connection.ini` or equivalent environment variables (credentials stay local).

3. Smoke test connectivity:   python -m venv .venv   python -m venv .venv



   ```powershell   

   .venv\Scripts\activate   
   .venv\Scripts\activate

   python scripts/sql/verify_acm_connection.py

   ```   pip install -e ".[dev]"   pip install -e ".[dev]"



4. Run `python -m core.acm_main --equip FD_FAN`. SQL writes flow through `core/output_manager.ALLOWED_TABLES`
   and are automatically batched for transactional safety. No additional flags are required.



## Configuration Model

Configuration lives in `configs/config_table.csv` and is loaded with `utils/config_dict.ConfigDict`.4. Run the pipeline:4. Run the pipeline:

Rows cascade by priority: global defaults (`*`) are overridden by equipment hashes (for example the

hash for `FD_FAN`). Access values with dot paths such as `cfg["fusion"]["weights"]["omr_z"]`.

When introducing new parameters:

- Document the change in `docs/CHANGELOG.md` and the backlog (`Task Backlog.md`).   ```powershell   ```powershell

- Update dependent design notes (for example `docs/Analytics Backbone.md`).

- Expect cached models to invalidate automatically on the next run.

## Pipeline Stages

1. **Ingestion** – Historian CSVs are still available for development, but production runs read from SQL via `core/sql_client.SQLClient`.
2. **Cleaning** – `core/clean.py` standardizes timestamps, fixes gaps, and enforces the sensor schema.
3. **Feature Engineering** – `core/fast_features.py` builds rolling stats, regressors, and deltas.
4. **Detector Training** – Models consume engineered features and persist through `core/model_persistence.py` when file artifacts are enabled.
5. **Scoring and Fusion** – Detector outputs feed `core/fuse.py`, producing fused z-scores that drive episodes.
6. **Reporting** – `core/output_manager.OutputManager` writes tables, charts, SQL tables, and run metadata per `docs/CHART_TABLES_SPEC.md`.

## Output Contract

Each run produces timezone-naive local artifacts under the fixed structure below. SQL-only mode may skip filesystem artifacts, but the layout remains reserved for file runs:

```
artifacts/{EQUIP}/run_{timestamp}/
    tables/            # Operator and engineering CSV tables
    charts/            # PNG visuals (health timelines, detector comparison, heatmaps)
    plots/             # Legacy plots (being consolidated)
    scores.csv         # Detector and fused z-scores
    episodes.csv       # Episode timeline with culprit sensors
    meta.json          # Durations, cache state, and health metrics
    run.jsonl          # Execution log stream
    models/            # Cached detector bundles (managed automatically)
```

Always route new exports through the output manager to keep naming, batching, and SQL guardrails
consistent. Respect the local time policy by using helper functions `_to_naive*` in
`core/output_manager.py` and `core/acm_main.py`.



## Adaptive Tuning, Drift, and Persistence

- Adaptive tuning monitors detector saturation, silhouette, and anomaly rates to adjust thresholds

  and fusion weights.1. Provision the ACM schema using scripts in `scripts/sql/` (see `docs/SQL_SCHEMA_DESIGN.md`).1. Provision the ACM schema using scripts in `scripts/sql/` (see `docs/SQL_SCHEMA_DESIGN.md`).

- Drift and regime detection (`core/drift.py`, `core/regimes.py`) quantify operating state changes

  and feed episode logic.2. Populate `configs/sql_connection.ini` or equivalent environment variables (credentials are never committed).2. Populate `configs/sql_connection.ini` or equivalent environment variables (credentials are never committed).

- Cache hits typically reduce retrain times by 5-8x; use `--no-cache` or `--clear-cache` when

  validating new behavior.3. Smoke test the connection:3. Smoke test the connection:

- Preserve configuration history and tuning logs for auditability via

  `core/config_history_writer.log_auto_tune_changes`.



## Operational Tooling   ```powershell   ```powershell

- `python scripts/analyze_latest_run.py --equip FD_FAN` reviews the most recent artifact bundle.

- `python scripts/chunk_replay.py --equip FD_FAN GAS_TURBINE` replays historian batches stored in   python scripts/sql/verify_acm_connection.py   python scripts/sql/verify_acm_connection.py

  `data/chunked/`.

- `scripts/run_file_mode.ps1` wraps the default file-mode execution on Windows.   ```   ```

- `python scripts/sql/verify_acm_connection.py` validates SQL credentials and permissions.

- `python scripts/polars_benchmark.py` compares pandas vs. Polars performance for fast features.



## Testing and Quality Gates

Run `python -m core.acm_main --equip FD_FAN` for smoke validation; SQL writes are restricted to
`core/output_manager.ALLOWED_TABLES` and batched for transactional safety.

- Core feature coverage: `pytest tests/test_fast_features.py`

- Output dual-write guardrails: `pytest tests/test_dual_write.py`

- Pipeline progress tracking: `pytest tests/test_progress_tracking.py`

------

Optional linting and typing live under `pyproject.toml` extras (`ruff`, `mypy`). Run them when

modifying shared modules.



## Repository Layout (abridged)## Configuration Model## Configuration Model

```

.

├── core/                 # Pipeline modules (orchestration, detectors, fusion, outputs)

├── configs/              # Config table and SQL connection template (local only)Configuration lives in `configs/config_table.csv` and is loaded through `utils/config_dict.ConfigDict`.Configuration lives in `configs/config_table.csv` and is loaded through `utils/config_dict.ConfigDict`.

├── data/                 # Sample historian exports and chunked replay sets

├── docs/                 # Design notes, analytics specs, and workflow guides

├── scripts/              # Batch harnesses, SQL utilities, and helper tooling

├── tests/                # Targeted pytest suites| Priority | EquipID value | Purpose || Priority | EquipID value | Purpose |

├── utils/                # Shared helpers (config dict, logging, paths)

├── artifacts/            # Gitignored run outputs|---------|---------------|---------||---------|---------------|---------|

├── backups/              # Gitignored backup directory

├── pyproject.toml        # Project metadata and dependencies| 1 | `*` | Global defaults || 1 | `*` | Global defaults |

└── Task Backlog.md            # Backlog tracking

```

## Documentation Map

- `docs/Analytics Backbone.md` - Detector, fusion, and episode contracts.

- `docs/COLDSTART_MODE.md` - Cold-start strategy and guardrails.Access values via dot paths (`cfg["fusion"]["weights"]["omr_z"]`). When adding parameters:Access values via dot paths (`cfg["fusion"]["weights"]["omr_z"]`). When adding parameters:

- `docs/BATCH_PROCESSING.md` - Chunk replay and batch harness procedures.

- `docs/OMR_DETECTOR.md` - Overall Model Residual theory and troubleshooting.

- `docs/SQL_INTEGRATION_PLAN.md` and `docs/SQL_SCHEMA_DESIGN.md` - Database architecture details.

- `docs/PROJECT_STRUCTURE.md` - Deep dive into directory layout and integration points.- Document the change in `docs/CHANGELOG.md` and `Task Backlog.md`.- Document the change in `docs/CHANGELOG.md` and `Task Backlog.md`.

- `docs/CHANGELOG.md` - Authoritative change history.

- Update dependent docs such as `docs/Analytics Backbone.md`.- Update dependent docs such as `docs/Analytics Backbone.md`.

## Contribution and Support

- Keep file mode healthy before extending SQL paths; regression-test with chunk replay when in

  doubt.

- Document schema or configuration changes in this README and the relevant design notes; update theConfiguration feeds the cache signature, so edits invalidate cached models on the next run.Configuration feeds the cache signature, so edits invalidate cached models on the next run.

  backlog in `Task Backlog.md` when work lands.

- Follow the local time policy and route all outputs through `core/output_manager`.

- Never commit credentials; `configs/sql_connection.ini` stays local and should rely on environment

  variables where possible.------

- Open issues with run metadata (`meta.json`) and relevant logs, or coordinate via the backlog for

  feature requests. For ambiguous instructions, document assumptions in the same commit to keep the

  knowledge base current.

## Output Contract

Each run creates a timestamped directory with timezone-naive local data:Each run creates a timestamped directory with timezone-naive local data:

```

artifacts/{EQUIP}/run_{timestamp}/artifacts/{EQUIP}/run_{timestamp}/

   tables/            # Operator and engineering CSV tables   tables/            # Operator and engineering CSV tables

   charts/            # PNG visuals (health timeline, detector comparison, heatmaps, etc.)   charts/            # PNG visuals (health timeline, detector comparison, heatmaps, etc.)

   plots/             # Legacy detector plots (migration in progress)   plots/             # Legacy detector plots (migration in progress)

   scores.csv         # Detector and fused z-scores   scores.csv         # Detector and fused z-scores

   episodes.csv       # Episode timeline with culprit sensors   episodes.csv       # Episode timeline with culprit sensors

   meta.json          # Durations, cache state, health metrics   meta.json          # Durations, cache state, health metrics

   run.jsonl          # Execution log stream   run.jsonl          # Execution log stream

   models/            # Cached detector bundles managed by model_persistence   models/            # Cached detector bundles managed by model_persistence

```



Always route new exports through `core/output_manager.OutputManager` to keep naming, batching, and SQL guardrails consistent.Always route new exports through `core/output_manager.OutputManager` to keep naming, batching, and SQL guardrails consistent.



------



## Adaptive Tuning, Drift, and Persistence## Adaptive Tuning, Drift, and Persistence



- **Adaptive tuning** monitors detector saturation, silhouette, and anomaly rates to auto-adjust thresholds and weights.- **Adaptive tuning** monitors detector saturation, silhouette, and anomaly rates to auto-adjust thresholds and weights.

- **Drift and regimes** in `core/drift.py` and `core/regimes.py` quantify operating-state changes that drive episode logic.- **Drift and regimes** in `core/drift.py` and `core/regimes.py` quantify operating-state changes that drive episode logic.

- **Caching** in `core/model_persistence.py` stores detector bundles keyed by config, feature, and schema signatures; cache hits typically reduce retrain time by 5-8x. Use `--no-cache` or `--clear-cache` when validating new behavior.- **Caching** in `core/model_persistence.py` stores detector bundles keyed by config, feature, and schema signatures; cache hits typically reduce retrain time by 5-8x. Use `--no-cache` or `--clear-cache` when validating new behavior.

- **Time policy** keeps all timestamps timezone-naive local; use `_to_naive*` helpers in `core/output_manager.py` and `core/acm_main.py` for conversions.- **Time policy** keeps all timestamps timezone-naive local; use `_to_naive*` helpers in `core/output_manager.py` and `core/acm_main.py` for conversions.



------



## Operational Tooling



- `python scripts/analyze_latest_run.py --equip FD_FAN` reviews the most recent artifact bundle.- `python scripts/analyze_latest_run.py --equip FD_FAN` reviews the most recent artifact bundle.

- `python scripts/chunk_replay.py --equip FD_FAN GAS_TURBINE` replays historian batches from `data/chunked/`.- `python scripts/chunk_replay.py --equip FD_FAN GAS_TURBINE` replays historian batches from `data/chunked/`.

- `scripts/run_file_mode.ps1` wraps the default file-mode execution on Windows.- `scripts/run_file_mode.ps1` wraps the default file-mode execution on Windows.

- `python scripts/sql/verify_acm_connection.py` validates SQL credentials and permissions.- `python scripts/sql/verify_acm_connection.py` validates SQL credentials and permissions.



------



## Testing and Quality Gates



- Core feature coverage: `pytest tests/test_fast_features.py`- Core feature coverage: `pytest tests/test_fast_features.py`

- Output dual-write guardrails: `pytest tests/test_dual_write.py`- Output dual-write guardrails: `pytest tests/test_dual_write.py`

- Pipeline progress tracking: `pytest tests/test_progress_tracking.py`- Pipeline progress tracking: `pytest tests/test_progress_tracking.py`



Optional linting and typing live in `pyproject.toml` extras (`ruff`, `mypy`).Optional linting and typing live in `pyproject.toml` extras (`ruff`, `mypy`).



------



## Repository Layout (abridged)



```

core/core/

   acm_main.py            # Pipeline entry point   acm_main.py            # Pipeline entry point

   analytics.py           # Adaptive tuning logic   analytics.py           # Adaptive tuning logic

   fast_features.py       # Vectorized feature engineering   fast_features.py       # Vectorized feature engineering

   fuse.py                # Detector fusion   fuse.py                # Detector fusion

   output_manager.py      # Tables/charts/SQL orchestration   output_manager.py      # Tables/charts/SQL orchestration

   model_persistence.py   # Cache management   model_persistence.py   # Cache management

configs/configs/

   config_table.csv       # Cascading configuration table   config_table.csv       # Cascading configuration table

   sql_connection.ini     # SQL credentials template (local only)   sql_connection.ini     # SQL credentials template (local only)

data/data/

   *.csv                  # Sample historian exports   *.csv                  # Sample historian exports

docs/docs/

   Analytics Backbone.md  # Detector and fusion contracts   Analytics Backbone.md  # Detector and fusion contracts

   COLDSTART_MODE.md      # Cold-start workflow details   COLDSTART_MODE.md      # Cold-start workflow details

scripts/scripts/

   chunk_replay.py        # Historian replay harness   chunk_replay.py        # Historian replay harness

   analyze_latest_run.py  # Artifact inspection helper   analyze_latest_run.py  # Artifact inspection helper

tests/tests/

   test_fast_features.py   test_fast_features.py

   test_dual_write.py   test_dual_write.py

```



See the `docs/` directory for deeper design notes (`docs/PROJECT_STRUCTURE.md`, `docs/OUTPUT_CONSOLIDATION.md`, etc.).See the `docs/` directory for deeper design notes (`docs/PROJECT_STRUCTURE.md`, `docs/OUTPUT_CONSOLIDATION.md`, etc.).



------



## Change Management## Change Management



- Keep this README in sync with pipeline behavior, output schemas, and CLI changes.- Keep this README in sync with pipeline behavior, output schemas, and CLI changes.

- Record significant updates in `docs/CHANGELOG.md` and track follow-up work in `Task Backlog.md`.- Record significant updates in `docs/CHANGELOG.md` and track follow-up work in `Task Backlog.md`.

- When detector logic changes, update `docs/Analytics Backbone.md` and related detector guides (for example `docs/OMR_DETECTOR.md`).- When detector logic changes, update `docs/Analytics Backbone.md` and related detector guides (for example `docs/OMR_DETECTOR.md`).



------



## Support and Contact## Support and Contact



- Consult the `docs/` directory for design deep dives, validation reports, and operating procedures.- Consult the `docs/` directory for design deep dives, validation reports, and operating procedures.

- Coordinate enhancements through `Task Backlog.md`, linking to relevant design notes.- Coordinate enhancements through `Task Backlog.md`, linking to relevant design notes.

- Maintain file-mode health before expanding SQL paths; use the chunk replay harness for regression testing.- Maintain file-mode health before expanding SQL paths; use the chunk replay harness for regression testing.



ACM V8 is designed to run unattended once configured. Preserve configuration history, tuning logs, and artifacts to maintain a full operational audit trail.ACM V8 is designed to run unattended once configured. Preserve configuration history, tuning logs, and artifacts to maintain a full operational audit trail.

- PowerShell shortcut: `scripts/run_file_mode.ps1` executes the default file-mode pipeline.

## Repository Layout
```
.
├── core/                 # Pipeline modules (orchestration, detectors, fusion, outputs)
├── configs/              # Config table + SQL connection template
├── data/                 # Sample historian exports and chunked replay sets
├── docs/                 # Detailed design notes, analytics specs, and workflow guides
├── scripts/              # Batch harnesses, SQL utilities, and helper tooling
├── tests/                # Targeted pytest suites
├── utils/                # Shared helpers (config dict, logger, paths, timer)
├── artifacts/            # Gitignored run outputs
├── backups/              # Gitignored backup directory
├── pyproject.toml        # Project metadata and dependency declarations
└── Task Backlog.md            # Backlog tracking (keep in sync when work lands)
```

## Documentation Map
- `docs/Analytics Backbone.md` detector, fusion, and episode contract.
- `docs/COLDSTART_MODE.md` cold-start strategy and guardrails.
- `docs/BATCH_PROCESSING.md` chunk replay, batch harness, and replay validation.
- `docs/OMR_DETECTOR.md` overall model residual theory, charts, and troubleshooting.
- `docs/SQL_INTEGRATION_PLAN.md` & `docs/SQL_SCHEMA_DESIGN.md` database architecture.
- `docs/PROJECT_STRUCTURE.md` deep dive into directory layout and SQL integration.
- `docs/CHANGELOG.md` authoritative change history.

## Contribution Notes
- Keep file-mode stability before modifying SQL paths.
- Document schema or config changes in the README and relevant docs; update backlog entries in `Task Backlog.md`.
- Follow the local time policy (no UTC conversions) and route all outputs through `core/output_manager`.
- Never commit credentials; `configs/sql_connection.ini` must remain local.

## Support
Open an issue in this repository with run metadata (`meta.json`) and relevant logs, or reference the backlog (`Task Backlog.md`) when requesting new features. For ambiguous instructions, document assumptions in the same commit to keep the knowledge base current.
