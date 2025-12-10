# Refactor Plan 10 December 2025

## 1. Overall refactor strategy

**Principle:**
First **amputate** everything that’s half-broken / redundant around forecasting & RUL.
Then **promote** the v10 modules (health_tracker, degradation_model, rul_estimator, failure_probability, forecast_engine, state_manager, sensor_attribution, output_manager) into the **only** forecasting/RUL path.

Long-term behaviour:

* Use **rolling health window** (not entire history) for models.
* Keep **compressed history** via SQL tables (episodes, fault families, baselines).
* Use **episodes + regimes + sensor attribution** to identify “kind of fault” and “not supposed to be” behaviour.

---

## 2. Module-by-module actionable refactor plan

### 2.1 Kill / Quarantine legacy forecasting + RUL

**Files involved**

* `forecasting.py` (big monolith)
* `rul_engine.py` (older “unified” RUL)
* Any lingering references in `acm_main.py`, `model_persistence.py`, others.

**Tasks**

1. **Search for all imports/usages of `forecasting` and `rul_engine`.**

   * In `acm_main.py` and the whole repo, locate:

     * `from core import forecasting`
     * `from . import forecasting`
     * `from core import rul_engine`
     * Any direct call like `forecasting.run_*`, `forecasting.estimate_rul`, `rul_engine.*`
   * **Goal:** there should be **zero live calls** to these modules.

2. **Comment/remove legacy calls in `acm_main.py`.**

   * In the “FORECAST GENERATION / enhanced forecasting” sections:

     * Delete or comment the blocks that mention:

       * “enhanced forecasting (forecasting.py has the working implementation)”
       * Any `forecasting.run_and_persist_enhanced_forecasting` or similar.
   * Keep only the **ForecastEngine-based** block for forecasting & RUL in file mode.
   * In SQL mode, leave the legacy enhanced forecasting commented; you’ll add ForecastEngine there later.

3. **Downgrade `forecasting.py` to “legacy helper only”.**

   * At top of `forecasting.py`, add a clear internal note that it is deprecated and not called.
   * Identify **small pure functions** that are still genuinely useful and not yet migrated:

     * e.g. data hashing, small math helpers, etc.
   * For each such helper:

     * Decide which v10 module it belongs to (`failure_probability`, `metrics`, `forecast_utils`).
     * Move the implementation there (by hand, no copy/paste via Copilot).
   * After moving, mark the corresponding section in `forecasting.py` as unreachable/dead (or remove it entirely, leaving only a stub header).

4. **Retire `rul_engine.py`.**

   * Confirm there are **no imports** of `rul_engine` anywhere.
   * Add an internal comment at the top that the file is deprecated and replaced by:

     * `health_tracker.py`
     * `degradation_model.py`
     * `rul_estimator.py`
     * `forecast_engine.py`
   * Do not delete the file yet (in case you need to mine ideas), but treat it as **read-only reference**.

**What to be careful about**

* Don’t leave any half-working path where ACM calls both ForecastEngine and old modules.
* Make sure no config flags still mention legacy forecast modes.
* Avoid accidental circular imports when moving helpers out of `forecasting.py`.

---

### 2.2 Make ForecastEngine the single forecasting/RUL orchestrator

**File:** `forecast_engine.py`

This already has a good structure, but it probably still contains legacy bits.

**Tasks**

1. **Enforce a strict workflow inside `run_forecast`.**

   * Ensure `run_forecast` does **exactly** and only:

     1. Load health timeline (via `HealthTimeline`).
     2. Load forecasting state (via `StateManager` or model persistence).
     3. Load adaptive config (via `AdaptiveConfigManager`).
     4. Decide whether to retrain or reuse (via retrain policy).
     5. Fit/update degradation model (via `LinearTrendModel`).
     6. Generate health forecast + uncertainty.
     7. Estimate RUL via `RULEstimator`.
     8. Compute failure probability statistics via `failure_probability`.
     9. Get sensor attributions via `SensorAttributor`.
     10. Write results to SQL via `OutputManager`.
     11. Update forecasting state.

2. **Delete / strip anything that doesn’t belong in the above.**

   * Remove references to:

     * Per-sensor ARIMA/VAR forecasts.
     * Direct CSV writing.
     * Code paths that try to forecast arbitrary sensor columns.
   * If some logic is important (e.g. logging, metrics) but sitting in the wrong place:

     * Move it to `metrics.py` (for metrics) or `sql_logger.py` (for logs).

3. **Unify its dependencies.**

   * Ensure imports come only from:

     * `health_tracker`
     * `degradation_model`
     * `rul_estimator`
     * `failure_probability`
     * `metrics` (optional for diagnostics)
     * `sensor_attribution`
     * `state_manager` or `model_persistence`
     * `output_manager` / `simplified_output_manager` (choose one, see below)
   * Remove any direct imports from `forecasting`, `rul_engine`, or file-mode-only helpers.

4. **Clarify its return contract.**

   * `run_forecast` should always return a **small dict** with:

     * `success` (bool)
     * RUL summary (`rul_p50`, `rul_p10`, `rul_p90`, `rul_mean`, `rul_std`)
     * `top_sensors` as a string or list
     * `data_quality` flag (from HealthTimeline)
     * `error` if `success` is False
   * Ensure `acm_main` uses these keys consistently in logs.

**What to be careful about**

* Keep error handling inside `ForecastEngine`: on failure, log and return `success=False` without raising up to ACM.
* Don’t let ForecastEngine talk directly to the filesystem; funnel outputs via `OutputManager` only.

---

### 2.3 Health history & quality: HealthTimeline as single source

**File:** `health_tracker.py`

This is your **entry point** into historical health and the rolling window concept.

**Tasks**

1. **Make SQL + rolling window the default.**

   * Ensure `HealthTimeline`:

     * Uses SQL (ACM_HealthTimeline) as the primary source.
     * Limits the loaded window by:

       * A configurable `history_window_hours` or
       * A max number of rows with resampling (`max_timeline_rows`, `downsample_freq`).
   * Confirm that it always returns:

     * A resampled, monotonic, timezone-naive `DatetimeIndex`.
     * A single main health column (e.g. `HealthIndex` or `fused_z`).

2. **Tighten quality flags.**

   * Confirm the 5 states are used and documented internally:

     * `OK`, `SPARSE`, `GAPPY`, `FLAT`, `NOISY`.
   * Define in code comments:

     * Minimum sample count.
     * Max allowed gap.
     * Min/max standard deviation.
   * Decide for forecasting:

     * Which levels allow full forecasting (e.g. `OK`, `GAPPY`).
     * Which should cause ForecastEngine to skip with a clear reason.

3. **Provide a compact “data summary” object.**

   * Ensure HealthTimeline exposes:

     * `dt_hours`
     * `n_samples`
     * `start_time`, `end_time`
     * `quality_flag`
   * ForecastEngine should consume this summary, not recompute.

**What to be careful about**

* Don’t accidentally load entire multi-year history by default.
* Ensure any timezone handling is consistent with the ACM policy (naive local timestamps).

---

### 2.4 Degradation & RUL core

**Files:** `degradation_model.py`, `rul_estimator.py`, `failure_probability.py`, `metrics.py`

These form your mathematical backbone.

**Tasks**

1. **Degradation model (trend).**

   * Decide that `LinearTrendModel` is the **primary model** for now.
   * Ensure it:

     * Accepts the rolling window series from HealthTimeline.
     * Stores `level`, `trend`, `dt_hours`, `residual_std`.
     * Has:

       * A clear fit method on recent history.
       * A forecast method that returns horizon series + CI.

2. **RUL via Monte Carlo.**

   * In `rul_estimator.py`, check:

     * It can take a `degradation_model` + failure threshold + dt_hours + max horizon.
     * It returns an object with quantiles and mean/std.
   * Make ForecastEngine call this as the **only RUL driver**.

3. **Failure probability & hazard.**

   * In `failure_probability.py`:

     * Ensure there’s a clear mapping:

       * health → failure prob vs horizon,
       * failure prob → survival,
       * survival → hazard rate,
       * optional mean time to failure.
   * ForecastEngine should:

     * Compute these once per forecast run.
     * Persist them into `ACM_FailureForecast` and summarize into `ACM_RUL`.

4. **Metrics (optional diagnostics).**

   * In `metrics.py`, keep it for offline diagnostics:

     * Model bias, MAE, coverage etc.
   * Optionally hook it into ForecastEngine to log forecast quality to `ACM_ForecastingState`.

**What to be careful about**

* Use **one consistent failure threshold** for RUL across modules; store it in config.
* Ensure all functions assume same dt_hours and horizon units (hours).

---

### 2.5 State & adaptive config

**Files:** `state_manager.py`, `model_persistence.py`

You currently have:

* `ForecastingState` in `state_manager`.
* `ForecastState` inside `model_persistence`.

You need a clean story.

**Tasks**

1. **Decide on a single “forecast state” abstraction.**

   * Choose **one** of these classes as canonical for forecasting:

     * Either re-use `ForecastState` from `model_persistence`, or
     * Prefer `ForecastingState` from `state_manager`.
   * The main criteria:

     * Fields relevant: model coefficients, last forecast, last retrain time, data hash, volume analyzed, quality metrics.
     * It must map cleanly to SQL (ACM_ForecastingState table).

2. **Align state schema with SQL table.**

   * In `state_manager.py`:

     * Make sure `load_state` and `save_state` read/write the same columns, matching the dataclass.
     * Use optimistic locking (ROWVERSION) as designed.

3. **Adaptive configuration.**

   * Use `AdaptiveConfigManager` to:

     * Load base config from global defaults.
     * Apply per-equip overrides from `ACM_AdaptiveConfig` (if present).
   * Validation:

     * Make sure it returns only **simple Python types** (floats, ints, strings, dicts) suitable for JSON.

4. **ForecastEngine integration.**

   * ForecastEngine should:

     * Load state config once.
     * Decide retrain vs reuse with a single policy (see below).
     * Save state only once per run.

**What to be careful about**

* Avoid having **two different state tables** for forecasting.
* Ensure you’re not serializing huge arrays into state JSON; store only model parameters, not the full history.

---

### 2.6 Output Manager and SQL tables

**Files:** `output_manager.py`, `simplified_output_manager.py`, SQL schemas

**Tasks**

1. **Choose one OutputManager implementation.**

   * If `acm_main.py` uses the full `OutputManager`, keep that as canonical.
   * If `simplified_output_manager` is unused:

     * Mark as candidate for future removal or keep only as experimental.

2. **Add explicit forecast/RUL writers.**

   * In `output_manager.py`, ensure there are clear helper methods for:

     * `ACM_HealthForecast`
     * `ACM_FailureForecast`
     * `ACM_RUL`
   * Each method should:

     * Delete any existing rows for `(RunID, EquipID)` in that table.
     * Perform bulk insert using the configured batch size.

3. **Align schemas.**

   * Ensure SQL tables have columns matching:

     * Health forecast: ForecastTimestamp, Health, Lower, Upper, Std, plus RunID/EquipID.
     * Failure forecast: ForecastTimestamp, FailureProbability, SurvivalProbability, HazardRate, etc.
     * RUL summary: RUL_P50, RUL_P10, RUL_P90, RUL_Mean, RUL_Std, DataQuality, TopSensors, etc.

4. **Hook ForecastEngine to use OutputManager.**

   * Inside ForecastEngine’s write phase, ensure all writing is routed through these helper methods, not raw SQL.

**What to be careful about**

* Make sure the transaction semantics in `OutputManager` do not conflict with the main SQL batch in `acm_main`.
* Keep error logging around forecast writes isolated so a bad forecast doesn’t break the whole ACM run.

---

### 2.7 ACM main wiring (file + SQL modes)

**File:** `acm_main.py`

**Tasks**

1. **File mode path (already mostly wired).**

   * Confirm that after analytics outputs:

     * `ForecastEngine` is called with:

       * A valid `sql_client` (if available) or `None`.
       * The **same** `OutputManager` instance.
     * Log statements print RUL summary and top sensors from the result dict.

2. **SQL mode path (currently has forecasting disabled).**

   * In the SQL mode section where forecasting is commented as disabled:

     * Insert a section similar to file mode, but using SQL `OutputManager`.
     * The flow should be:

       * Write all analytics tables via OutputManager.
       * Then call ForecastEngine (SQL mode).
       * Then finalize the run in `ACM_Runs`.

3. **Coldstart integration.**

   * In the main run flow:

     * Ensure smart coldstart module (`smart_coldstart.py`) is used to decide if ACM should:

       * Run full analytics + forecasting, or
       * Short-circuit early with a “coldstart incomplete” flag.
   * ForecastEngine should check HealthTimeline `quality_flag` and skip if data is obviously insufficient, returning a clean failure reason.

4. **Remove any hidden forecasting branches.**

   * Scan `acm_main.py` for:

     * `forecast_*` and `rul_*` references that are not using ForecastEngine.
   * Remove/comment them so there is a single forecasting/RUL path.

**What to be careful about**

* Keep the run lifecycle unchanged: start, analytics, forecasting, finalize.
* Make sure forecasting failures do not alter the RunStatus to failed; treat them as warnings.

---

### 2.8 Fault “kind” & “not supposed to be” (foundation only in this pass)

You already have:

* `episode_culprits_writer.py` – writes culprits per episode.
* `sensor_attribution.py` – reads `ACM_SensorHotspots`.
* `regimes.py`, `drift.py`, `correlation.py`, `omr.py`, `fast_features.py` – all feed anomaly structure and health.

**Tasks (foundation, no big modeling yet)**

1. **Standardise fault episodes.**

   * Confirm that ACM already writes:

     * `ACM_Episodes`
     * `ACM_EpisodeCulprits` via `episode_culprits_writer.py`.
   * If not, ensure the writer is invoked after episodes detection.

2. **Sensor-based fault explanation.**

   * Ensure:

     * `sensor_attribution.py` can read `ACM_SensorHotspots` for `EquipID, RunID`.
   * ForecastEngine should:

     * Use `SensorAttributor` (or equivalent) to get **top N sensors** and include them in:

       * The result dict.
       * `ACM_RUL.TopSensors` field.

3. **Behaviour deviation hooks.**

   * Mark a design point around:

     * Using per-regime stats (in `regimes.py` + your fused score `fuse.py`) for baselines.
   * In this pass, just:

     * Ensure drift and regime outputs are written properly.
     * Leave more advanced “not supposed to be like this” checks for a separate milestone.

**What to be careful about**

* Don’t overload ForecastEngine with clustering logic yet.
* Keep the “kind of fault” initially as **“which sensors and detectors are most involved”** rather than a full family ID.

---

## 3. Detailed task planner with milestones & source control

### Milestone M0 – Branch & safety net

**Actions**

* Create a feature branch, e.g. `feature/forecast-rul-v10`.
* Tag current main as something like `acm-v10-preforecast`.
* Turn on more verbose SQL logging via `SqlLogSink` around forecasting sections for traceability.

---

### Milestone M1 – Legacy forecasting/RUL removal

| Task ID | Area         | Description                                                                                 | Files                         | Done when                                                                  |
| ------: | ------------ | ------------------------------------------------------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------- |
|    M1.1 | Imports      | Remove all active imports/usages of `forecasting` and `rul_engine`.                         | `acm_main.py`, whole repo     | Global search shows no live references; only deprecated comments remain.   |
|    M1.2 | Legacy calls | Delete/comment enhanced forecasting calls that use `forecasting` in SQL path.               | `acm_main.py`                 | SQL-mode path has no call to `forecasting`; only ForecastEngine planned.   |
|    M1.3 | Monolith     | Strip `forecasting.py` down to non-used reference/notes; move any useful helpers elsewhere. | `forecasting.py`, v10 modules | `forecasting.py` no longer imported anywhere; only small helpers migrated. |
|    M1.4 | Old RUL      | Mark `rul_engine.py` as deprecated and unused.                                              | `rul_engine.py`               | No imports anywhere; file only kept as reference.                          |

**Source control**

* Commit as “M1: Remove legacy forecasting/RUL entrypoints”.

---

### Milestone M2 – ForecastEngine consolidation

| Task ID | Area             | Description                                                                                            | Files                | Done when                                                                           |
| ------: | ---------------- | ------------------------------------------------------------------------------------------------------ | -------------------- | ----------------------------------------------------------------------------------- |
|    M2.1 | Workflow         | Restructure `run_forecast` to follow the 11-step workflow (load → state → config → fit → RUL → write). | `forecast_engine.py` | `run_forecast` body is clean, linear, and only uses v10 modules.                    |
|    M2.2 | Strip legacy     | Remove per-sensor ARIMA/VAR, direct CSV writes, and direct calls to old modules from ForecastEngine.   | `forecast_engine.py` | No ARIMA/VAR or file I/O remains; only health/RUL/failure forecast via v10 modules. |
|    M2.3 | Return contract  | Standardise result dict keys (success, RUL numbers, top_sensors, data_quality, error).                 | `forecast_engine.py` | Logs in `acm_main` match these keys; no missing key errors.                         |
|    M2.4 | Dependency scope | Ensure ForecastEngine imports only v10 modules + utilities, not legacy forecasting/rul_engine.         | `forecast_engine.py` | Import section shows only v10 modules & utilities.                                  |

**Source control**

* Commit as “M2: Consolidate ForecastEngine to v10 path only”.

---

### Milestone M3 – HealthTimeline & rolling window

| Task ID | Area    | Description                                                                                 | Files               | Done when                                                    |
| ------: | ------- | ------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------ |
|    M3.1 | Window  | Configure HealthTimeline to load only a rolling history (e.g. last 30–90 days).             | `health_tracker.py` | SQL query and/or resampling enforces a bounded window.       |
|    M3.2 | Quality | Confirm quality flags are used (`OK`, `SPARSE`, `GAPPY`, `FLAT`, `NOISY`) with clear rules. | `health_tracker.py` | ForecastEngine can act based on the flag (skip or proceed).  |
|    M3.3 | Summary | Expose dt_hours, n_samples, start/end, quality from HealthTimeline for ForecastEngine.      | `health_tracker.py` | ForecastEngine uses these fields and doesn’t recompute them. |

**Source control**

* Commit as “M3: Rolling health window & quality for forecasting”.

---

### Milestone M4 – Degradation, RUL & failure probability

| Task ID | Area      | Description                                                                       | Files                                          | Done when                                                                          |
| ------: | --------- | --------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------- |
|    M4.1 | Trend     | Use `LinearTrendModel` as the sole degradation model for health index.            | `degradation_model.py`, `forecast_engine.py`   | ForecastEngine always calls `LinearTrendModel` to fit & forecast.                  |
|    M4.2 | RUL       | Wire `RULEstimator` as the only RUL mechanism.                                    | `rul_estimator.py`, `forecast_engine.py`       | RUL summary in result dict comes from RULEstimator output.                         |
|    M4.3 | Fail prob | Use `failure_probability` to compute failure, survival, and hazard vs horizon.    | `failure_probability.py`, `forecast_engine.py` | AC tables for failure forecast are populated using these functions.                |
|    M4.4 | Metrics   | Optionally log forecast quality metrics to forecasting state, using `metrics.py`. | `metrics.py`, `state_manager.py`               | Forecasting state includes simple metrics (e.g. MAE, bias) when data is available. |

**Source control**

* Commit as “M4: Degradation model, RUL and failure probability integrated”.

---

### Milestone M5 – State, adaptive config & retrain policy

| Task ID | Area       | Description                                                                            | Files                                      | Done when                                                                                  |
| ------: | ---------- | -------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------ |
|    M5.1 | State type | Choose and document a single forecasting state class and ensure it matches SQL schema. | `state_manager.py`, `model_persistence.py` | Only one state type in use; load/save functions align with table columns.                  |
|    M5.2 | State I/O  | Ensure ForecastEngine loads/saves state once per run with optimistic locking.          | `forecast_engine.py`, `state_manager.py`   | Forecast state is persisted each run; concurrent runs won’t overwrite each other silently. |
|    M5.3 | Config     | Use AdaptiveConfigManager to apply per-equip overrides for forecasting parameters.     | `state_manager.py`, `forecast_engine.py`   | Config values seen by ForecastEngine match overrides stored in SQL when present.           |
|    M5.4 | Retrain    | Implement a simple retrain policy (no complicated auto-tuning yet).                    | `forecast_engine.py`                       | A deterministic rule decides retrain vs reuse (e.g. time passed, data hash changed, etc.). |

**Source control**

* Commit as “M5: Forecasting state and adaptive config wired”.

---

### Milestone M6 – OutputManager & SQL table integration

| Task ID | Area           | Description                                                                             | Files                              | Done when                                                                 |
| ------: | -------------- | --------------------------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------- |
|    M6.1 | Helper methods | Add dedicated write helpers for health, failure forecast and RUL summary tables.        | `output_manager.py`                | ForecastEngine calls only these helpers to persist forecasts/RUL.         |
|    M6.2 | Schemas        | Confirm SQL tables match the planned column sets for forecasts & RUL.                   | DB + `output_manager.py`           | Inserts succeed without column name mismatches; no runtime insert errors. |
|    M6.3 | Cleanup        | Ensure SQL cleanup (DELETE per RunID/EquipID) happens before writing new forecast rows. | `acm_main.py`, `output_manager.py` | Re-running a run overwrites forecast/RUL rows instead of duplicating.     |

**Source control**

* Commit as “M6: Forecast/RUL SQL wiring via OutputManager”.

---

### Milestone M7 – ACM main integration (both modes)

| Task ID | Area      | Description                                                                                   | Files                               | Done when                                                                                    |
| ------: | --------- | --------------------------------------------------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------- |
|    M7.1 | File mode | Confirm file mode path calls ForecastEngine after analytics, logs RUL & top sensors.          | `acm_main.py`                       | A standard file-mode run completes with forecast logs and forecast tables populated.         |
|    M7.2 | SQL mode  | Enable ForecastEngine in SQL path after analytics; remove “Forecasting and RUL disabled” log. | `acm_main.py`                       | SQL-mode runs populate forecast tables and `ACM_RUL` and do not log forecasting as disabled. |
|    M7.3 | Coldstart | Respect smart coldstart decisions when deciding whether to run forecasting.                   | `acm_main.py`, `smart_coldstart.py` | When coldstart is incomplete, ACM skips forecasting with a clear reason, not an error.       |

**Source control**

* Commit as “M7: ForecastEngine integrated into SQL mode”.

---

### Milestone M8 – Fault explanation foundation

| Task ID | Area           | Description                                                                                | Files                                       | Done when                                                                                |
| ------: | -------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------- | ---------------------------------------------------------------------------------------- |
|    M8.1 | Episodes       | Make sure episodes and culprits are consistently written to `ACM_Episodes` and *_Culprits. | `episode_culprits_writer.py`, `acm_main.py` | Runs consistently populate these tables for assets with episodes.                        |
|    M8.2 | Sensor explain | Ensure SensorAttributor reads from `ACM_SensorHotspots` and returns top-N sensors.         | `sensor_attribution.py`                     | ForecastEngine can log and store sensor names responsible for degradation/failure risk.  |
|    M8.3 | RUL context    | Include top sensors and data quality in `ACM_RUL` for operator explanation.                | `forecast_engine.py`, `output_manager.py`   | RUL table includes clear explanation fields for operators (which sensors, data quality). |

**Source control**

* Commit as “M8: Fault explanation basics through sensor attribution”.

---

After M8, you have:

* All dead forecasting/RUL bloat removed.
* A single, well-defined forecasting/RUL system.
* Rolling health window + quality.
* Degradation + RUL + failure probability pipeline.
* SQL-only persistence through OutputManager.
* Basic “kind of fault” explanation via sensor attribution.

At that point, you can plan a **separate** refactor for:

* Fault families (episode clustering),
* Contextual “not supposed to be like this” baselines per regime,
* Early fault-family prediction models.

But the core foundation will already be clean and coherent for Copilot to work on.
