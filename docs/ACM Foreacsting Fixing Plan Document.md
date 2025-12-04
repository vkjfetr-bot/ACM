# ACM Foreacsting Fixing Plan Document
## 1. Diagnosis – what’s wrong today

From going through **forecasting.py, rul_engine.py, acm_main.py, model_persistence.py, output_manager.py, regimes.py, adaptive_thresholds.py, smart_coldstart.py, model_evaluation.py, fast_features.py, fuse.py, historian.py, sql_protocol.py, sql_logger.py, omr.py, outliers.py, correlation.py**, plus your **Implementation Guide**:

Refer to ACM Forecasting Fixing Guide.py  in addition to this guide to understand the entire plan of action.

### 1.1 Forecasting & RUL are split, duplicated, and leaky

* You have **two “unified” things**:

  * `core/forecasting.py` – “Unified Forecasting Module”
  * `core/rul_engine.py` – “Unified RUL Engine”
* Both:

  * Talk to **the same SQL tables** (`ACM_HealthForecast_TS`, `ACM_FailureForecast_TS`, `ACM_RUL_Summary`)
  * Contain **overlapping RUL logic** (Monte Carlo, hazard, etc.)
  * Implement their **own state / cleanup / retention** logic.
* Result: the **single source of truth is gone**:

  * Hard to know which path wrote a given forecast.
  * Any change must be mirrored in two places.
  * High risk of subtle divergence in behaviour.

### 1.2 No clean separation of responsibilities

Your Implementation Guide has a clean picture:

* `HealthTimeline` → `DegradationModel` → `FailureProbability` → `RULEstimator` → `StateManager` → `ForecastEngine`.

Current code instead:

* `forecasting.py` mixes:

  * SQL I/O + data loading
  * health timeline building
  * AR1 / detector-level forecasting
  * hazard calculation
  * Monte Carlo RUL
  * quality metrics
  * run retention
  * state management bits
* `rul_engine.py` again repeats:

  * RUL logic, hazard, SQL writes, run retention, sensor attribution.

You **do have** many of the pieces described in the guide, but they’re:

* **Functions buried in giant modules**, not small focused modules.
* Repeated / slightly different between `forecasting.py` and `rul_engine.py`.

### 1.3 State is ad-hoc

* Model state is handled in **`model_persistence.py`** (for anomaly models) and bits of **“ forecasting state”** are pushed via hashes / flags in `forecasting.py` and `rul_engine.py` – but:

  * There is **no single `ACM_ForecastingState` abstraction** yet.
  * Forecast blending / retrain decisions are scattered.
* This is exactly what your **`state_manager.py`** spec is meant to fix.

### 1.4 SQL coupling is too deep

* SQL details (table names, MERGE / pruning logic, retention policies) are sprinkled inside:

  * `forecasting.py`
  * `rul_engine.py`
  * `output_manager.py`
* The new spec wants **forecast engine** to produce **DataFrames**, and a thin layer to:

  * Write to `ACM_HealthForecast_TS`, `ACM_FailureForecast_TS`, `ACM_RUL_Summary`
  * Maintain `ACM_ForecastingState`.

Right now, **analytics, orchestration, and storage are tangled**.

### 1.5 “Health” vs detections vs RUL

* Health index is produced by ACM heads (AR1, PCA, MHAL, etc.) via **fuse / omr / fast_features / regimes** → eventually `ACM_HealthTimeline`.
* Forecasting & RUL code sometimes:

  * Operates on health index (as desired).
  * Sometimes mixes detector-level things or uses **score series** / hazard over anomaly scores.
* This makes behaviour less interpretable: sometimes RUL is really “time until health < threshold”, sometimes “time until detectors freak out”.

---

## 2. Global design decisions

Given your Implementation Guide is the **canonical architecture**, here’s the high-level decision set:

1. **Single source of truth**:

   * **One** Forecast + RUL pipeline = `core/forecast_engine.py` + its modules.
   * `rul_engine.py` and the forecasting pieces in `forecasting.py` become **legacy / compatibility shims** and then are retired.

2. **New module boundaries (as per your guide)**:

   * `core/health_tracker.py`
   * `core/degradation_model.py`
   * `core/failure_probability.py`
   * `core/rul_estimator.py`
   * `core/state_manager.py`
   * `core/forecast_engine.py`
   * `core/sensor_attribution.py`
   * `core/metrics.py`

3. **Keep existing analytics backbone intact**:

   * **Do NOT touch**:

     * `fast_features.py`
     * `fuse.py`
     * `omr.py`
     * `regimes.py`
     * `outliers.py`
     * `correlation.py`
     * `model_persistence.py`
     * `smart_coldstart.py`
     * `historian.py`
   * They continue to produce `ACM_HealthTimeline` & friends.
   * Forecasting only **consumes** `ACM_HealthTimeline` + sensor hotspot tables.

4. **State & SQL**:

   * Introduce `ACM_ForecastingState` and use it **only via `state_manager.py`**.
   * Keep current tables:

     * `ACM_HealthForecast_TS`
     * `ACM_FailureForecast_TS`
     * `ACM_RUL_Summary`
   * Existing retention logic in `forecasting.py` / `rul_engine.py` is moved into a **single place** or re-used via helper.

5. **Migration must be feature-flagged and reversible**:

   * Config flag per equipment: `UseNewForecastEngine = 0/1` (stored in config table or JSON config).
   * For some time, **write both old and new outputs** for comparison before fully switching.

---

## 3. Phased migration plan (high-level)

### Phase 0 – Safety net & groundwork (no behaviour change)

**Goal:** Prepare schemas and flags, but keep current ACM behaviour.

1. **Create / confirm SQL schema:**

   * Ensure the four tables from the guide exist:

     * `ACM_ForecastingState`
     * `ACM_HealthForecast_TS`
     * `ACM_FailureForecast_TS`
     * `ACM_RUL_Summary`
   * If they already exist (they appear in current code), **align columns** with the spec and:

     * Add missing columns with defaults (e.g. `Confidence`, `Method`).
     * Avoid breaking existing queries (add columns, don’t drop).

2. **Add config flag(s) – in `config_table` / config JSON:**

   * `UseNewForecastEngine` (bool/int).
   * `ForecastMode` (optional: `"legacy" | "new" | "dual"`).

3. **Add no-op wrappers:**

   * `core/health_tracker.py`, `core/degradation_model.py`, etc. created with **stubs** or simple wrappers around existing logic.
   * No code path calls them yet.

---

### Phase 1 – Extract and modularise without wiring

**Goal:** Build the new modules by **lifting** existing working logic out of `forecasting.py` / `rul_engine.py`, not re-inventing.

#### 1.1 `health_tracker.py`

* **Reuse from**:

  * Health loading logic in `forecasting.py` that queries `ACM_HealthTimeline`.
  * Any existing data quality checks (sample counts, gaps, std, etc.).
* **Tasks:**

  * Implement `HealthTimeline` as per guide:

    * `load_from_sql(sql_client, lookback_hours)`
    * `append_batch(new_data)`
    * `get_sliding_window(hours)`
    * `quality_check() -> HealthQuality`
    * `get_statistics()`
    * `detect_regime_shift()`
  * Replace ad-hoc “load health & check quality” functions in `forecasting.py` with calls to this class (but keep old function signatures as thin adapters for now).

#### 1.2 `degradation_model.py`

* **Current situation:**

  * `forecasting.py` already has trend estimation and Monte-Carlo helpers, but not a clean model class.
* **Tasks:**

  * Implement `BaseDegradationModel` and `LinearTrendModel` **from the spec**.
  * For now, keep it **health-only**:

    * Input: (timestamps, HealthIndex)
    * Output: `ForecastResult(timestamps, mean, std)`.
  * Optionally reuse any existing slope/linear regression helper from `forecasting.py`; if not cleanly reusable, just implement fresh here – it’s small and self-contained.

#### 1.3 `failure_probability.py`

* **Reuse from**:

  * The P0 hazard/failure-probability helpers already in `forecasting.py` (you have a P0-FIX for hazard and probability).
* **Tasks:**

  * Move / adapt the logic into:

    * `health_to_failure_probability(mean, std, failure_threshold)`
    * `compute_survival_curve(failure_probs)`
    * `compute_hazard_rate(failure_probs, dt_hours)`
    * `mean_time_to_failure(survival_probs, dt_hours)`
  * Replace internal calls in `forecasting.py`/`rul_engine.py` with imports from this module.

#### 1.4 `rul_estimator.py`

* **Reuse from**:

  * `_estimate_rul_monte_carlo(...)` in `forecasting.py` and RUL stats logic in `rul_engine.py`.
* **Tasks:**

  * Implement `RULResult` dataclass and `RULEstimator` class.
  * Move Monte Carlo logic into `RULEstimator.compute_rul_probabilistic(...)` (keeping your proven code).
  * Add deterministic crossing method for fallback.
  * Implement `_compute_confidence` as per spec, possibly reusing any existing “confidence” heuristics.

#### 1.5 `state_manager.py`

* **Current:** state scattered across hash checks / retention / flags.
* **Tasks:**

  * Implement `ForecastingState` dataclass exactly as spec.
  * Implement:

    * `save_state(state, sql_client)`
    * `load_state(equip_id, sql_client)`
    * `should_retrain(state, health_timeline, ...)`
    * `blend_forecasts(old_mean, old_std, new_mean, new_std, alpha)`
  * You can **reuse** run-retention / cleanup SQL snippets from both `forecasting.py` and `rul_engine.py` by:

    * Moving pruning logic into a small helper in `state_manager.py` (or a new `forecast_sql_utils.py`), then calling it from the new engine.

#### 1.6 `sensor_attribution.py` and `metrics.py`

* **Reuse from**:

  * `sensor hotspots` logic and model evaluation helpers already present (in `rul_engine.py`, `model_evaluation.py`, possibly `correlation.py`).
* **Tasks:**

  * Implement:

    * `rank_sensors_by_correlation(...)`
    * `compute_sensor_contributions(...)`
  * Implement:

    * `compute_forecast_error(predicted, actual, horizons=[1,24,168])`
  * Later wire these into `ForecastEngine` for metrics and into `output_manager` to persist.

> At the end of Phase 1: all **new modules exist and are testable in isolation**, but the main ACM flow still uses the current `forecasting.py` / `rul_engine.py` entrypoints.

---

### Phase 2 – Implement `forecast_engine.py` and adapter layer

**Goal:** One orchestrator that matches your spec, while keeping legacy code alive via adapters.

#### 2.1 Implement `ForecastEngine`

* Implement `ForecastEngine(sql_client, equip_id, config)` as per guide:

  * Step 1: `load_state`
  * Step 2: `HealthTimeline.load_from_sql` + `append_batch`
  * Step 3: `quality_check`
  * Step 4: `should_retrain`
  * Step 5: Fit or update `LinearTrendModel`
  * Step 6: Blend via `blend_forecasts` (depending on `needs_retrain` & quality)
  * Step 7: Failure prob + survival + hazard
  * Step 8: RUL via `RULEstimator`
  * Step 9: Build DataFrames (health forecast, failure forecast, RUL summary)
  * Step 10: Write via **existing `output_manager.write_dataframe`** (so we reuse your IO layer)
  * Step 11: Save `ForecastingState`
  * Step 12: Return dict with DataFrames + metrics.

#### 2.2 Integrate with existing output pipeline

* In `forecasting.py` and/or `rul_engine.py`:

  * Add a **thin wrapper** function, e.g.:

  ```python
  def run_new_forecast(sql_client, equip_id, run_id, config, new_batch_df=None):
      engine = ForecastEngine(sql_client, equip_id, config)
      return engine.run_forecast(run_id=run_id, new_batch_data=new_batch_df)
  ```

  * Respect `UseNewForecastEngine` flag:

    * If flag = 0 → keep existing behaviour.
    * If flag = 1 → call the new engine.
    * If flag = “dual” → call both and:

      * Write legacy tables as today.
      * Also write new tables (or tag outputs differently) for offline comparison.

> At this stage, **no SQL schema changes**, only new writes and adapters. Existing jobs keep running.

---

### Phase 3 – Wire into ACM main / batch runners & retire duplication

**Goal:** Make the new engine the default, and remove the double-write / double-logic mess.

#### 3.1 Identify all call sites

* Find everywhere forecasting / RUL is invoked:

  * Any calls inside `acm_main.py` (if present).
  * Any `scripts/sql_batch_runner.py` / other runner scripts (even if not uploaded now, you know where they are).
  * Any explicit calls from `output_manager` or other modules (unlikely but check).

#### 3.2 Switch orchestration

* Replace “legacy” entrypoint with a clear structure:

  ```python
  if use_new_engine:
      result = run_new_forecast(...)
  else:
      result = run_legacy_forecast(...)  # existing flow
  ```

* Keep this branching **only at orchestration level**, not in low-level analytics.

#### 3.3 Retire `rul_engine.py`

* Once:

  * `ForecastEngine` is stable,
  * RUL outputs from new pipeline are validated against legacy,
* Then:

  * Remove or freeze RUL-specific logic from `rul_engine.py`.
  * If some utilities are still useful (sensor attribution, hazard plots), move them into:

    * `sensor_attribution.py`
    * `failure_probability.py`
    * `metrics.py`
  * Leave `rul_engine.py` as a thin compatibility shim (or delete when you’re fully confident).

#### 3.4 Consolidate state & retention logic

* Remove duplicated run-retention code:

  * Keep a **single implementation** of “keep last N runs in `ACM_HealthForecast_TS` / `ACM_FailureForecast_TS` / `ACM_RUL_Summary`”.
  * Place it either:

    * In `state_manager.py`, or
    * In a new `forecast_sql_utils.py`.
  * Call this from `ForecastEngine` after each successful run.

---

### Phase 4 – Analytics robustness and extensions (once stable)

After new architecture is in place and stable:

1. **Add ExponentialDecayModel / asset-specific models** into `degradation_model.py` as needed.
2. **Use `metrics.py` to track forecast accuracy** and feed back into `ForecastingState.recent_mae`, `recent_rmse`, etc.
3. **Use `sensor_attribution` outputs** to enrich RUL summary / operator dashboards (top sensors driving the predicted failure).

---

## 4. Concrete backlog (for Copilot / you)

Here’s a concise backlog table focused on **forecasting & RUL**, not touching the anomaly backbone.

| ID   | Area / Module          | Pri | What’s wrong now / Gap                                                          | Action / How to fix (brief)                                                                                                                                                          | Reuse from                                                              |
| ---- | ---------------------- | --- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| F-01 | SQL Schema & Flags     | P0  | Forecast tables exist but state table not standardised, no engine flag          | Create/align `ACM_ForecastingState` as per spec; add `UseNewForecastEngine` / `ForecastMode` in config table                                                                         | Existing table defs in `forecasting.py`                                 |
| F-02 | `health_tracker.py`    | P0  | Health loading + quality logic scattered in `forecasting.py`, no reusable class | Implement `HealthTimeline` + `HealthQuality`; move existing SQL load + quality checks behind this class                                                                              | Health load/query logic in `forecasting.py`                             |
| F-03 | Quality flags          | P0  | No centralised “OK / SPARSE / GAPPY / FLAT / NOISY / MISSING” logic             | Implement `quality_check()` on `HealthTimeline` and adopt it wherever forecasting currently does ad-hoc checks                                                                       | Current warnings/guards in forecasting                                  |
| F-04 | `degradation_model.py` | P0  | Linear trend forecast logic not encapsulated; mixed with forecaster             | Implement `BaseDegradationModel` + `LinearTrendModel`; use np.polyfit / linregress as per spec                                                                                       | Trend estimation in `forecasting.py`                                    |
| F-05 | Forecast uncertainty   | P0  | Uncertainty growth vs horizon is ad-hoc / inconsistent between modules          | Standardise in `LinearTrendModel.predict()` – `std(t) = residual_std * sqrt(1 + t/T)`                                                                                                | Any existing growth logic in forecasting                                |
| F-06 | Failure probability    | P0  | Hazard / failure probability logic duplicated & partly embedded                 | Create `failure_probability.py` with `health_to_failure_probability`, `compute_survival_curve`, `compute_hazard_rate`, `mean_time_to_failure`; replace in-module copies with imports | P0 hazard/failure code in `forecasting.py`                              |
| F-07 | Monte Carlo RUL        | P0  | Monte Carlo RUL exists but not as reusable class; duplicated across modules     | Implement `RULEstimator` (`compute_rul_deterministic`, `compute_rul_probabilistic`, `_compute_confidence`) and wire to use existing `_estimate_rul_monte_carlo` logic                | `_estimate_rul_monte_carlo` in forecasting, RUL bits in `rul_engine.py` |
| F-08 | State Manager          | P0  | No single place for model coefficients / last forecast / retrain decision       | Implement `ForecastingState`, `save_state`, `load_state`, `should_retrain`, `blend_forecasts` as per guide                                                                           | Current per-run retention & state hints                                 |
| F-09 | ForecastEngine         | P0  | Orchestration currently spread across `forecasting.py`/`rul_engine.py`          | Implement `ForecastEngine.run_forecast()` 12-step pipeline; use new modules; return DataFrames & metrics                                                                             | Existing orchestration in forecasting                                   |
| F-10 | Output wiring          | P1  | SQL write logic embedded in multiple modules                                    | Make `ForecastEngine` build DataFrames & call `output_manager.write_dataframe`; centralise run retention                                                                             | `output_manager.py`, existing writes                                    |
| F-11 | Legacy adapters        | P1  | Callers currently expect old forecasting / RUL entrypoints                      | In `forecasting.py` / `rul_engine.py`, add thin wrappers that delegate to `ForecastEngine` based on config flag                                                                      | Existing public functions                                               |
| F-12 | Remove duplication     | P1  | Same tables (`ACM_HealthForecast_TS`, etc.) written from multiple modules       | After new engine is proven, remove/disable direct SQL writers in `rul_engine.py` and legacy forecasting branches                                                                     | Forecast table writes in both modules                                   |
| F-13 | Sensor attribution     | P2  | Attribution logic is entangled with RUL engine                                  | Implement `sensor_attribution.py` (`rank_sensors_by_correlation`, `compute_sensor_contributions`) and use from dashboards/RUL summary                                                | Sensor hotspot logic in `rul_engine.py`                                 |
| F-14 | Metrics & accuracy     | P2  | No standard place for forecast error metrics                                    | Implement `metrics.py.compute_forecast_error`; call from `ForecastEngine` for backtesting; update `ForecastingState.recent_mae`/`rmse`                                               | `model_evaluation.py`                                                   |
| F-15 | Batch continuity tests | P2  | No automated test for 100-batch continuity                                      | Add integration test harness (pytest) simulating batches with new engine; assert no >X% jump between runs                                                                            | Smart coldstart + forecasting pieces                                    |
| F-16 | Regime shift trigger   | P2  | Regime change detection not standardised into forecasting state                 | Implement `HealthTimeline.detect_regime_shift()` and call it from `should_retrain`                                                                                                   | Any existing slope / drift checks                                       |
| F-17 | Documentation          | P3  | Architecture exists as a single guide doc but not linked to code                | Add `FORECASTING.md` in repo summarising module structure + key entrypoints + SQL contracts                                                                                          | Your Implementation Guide                                               |

---

### Answering your core ask in one line

**The ACM isn’t “broken” in the analytic sense – the detectors & health generation are fine – but the forecasting/RUL layer is overgrown and duplicated.**

The plan above **keeps your existing analytics backbone**, lifts the good bits out of `forecasting.py`/`rul_engine.py` into clean modules exactly matching your Implementation Guide, adds a single `ForecastEngine` as source of truth, and then safely retires the duplicate paths once the new engine is validated.
