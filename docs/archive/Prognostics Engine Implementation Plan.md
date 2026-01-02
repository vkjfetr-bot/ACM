# Prognostics Engine


## 1. High-level architecture & development plan

### 1.1 Single engine, multiple “views” on future

We build **one unified engine** that outputs:

* **Health trajectory** forecasts (what you already do).
* **Failure probability curve** over time (already partly there).
* **Multi-horizon risk**: e.g. risk of trip/failure in next 3d / 7d / 30d.
* **RUL bands** derived from the above (not just 1 number).
* **Attribution**:

  * Sensor level (hotspots).
  * Episode/drift/regime level (which “histories” contribute to risk).
* **Maintenance recommendation** (you already have a version; we extend it).

All of this is driven by **one entrypoint**, say:

```text
run_prognostics(sql_client, equip_id, run_id, output_manager=None, config_row=None)
```

This replaces/extends the current `run_rul` but internally separates:

* **Layer 1 – Data layer** (SQL-only I/O, caches, config).
* **Layer 2 – Feature layer** (health trajectory features + episode/drift/regime features).
* **Layer 3 – Models**:

  * Degradation ensemble (AR1/Exp/Weibull/etc.) for health forecast.
  * Risk / hazard models for event-in-window probabilities.
* **Layer 4 – Fusion layer**:

  * Combine hazard + health forecasts into RUL & risk bands.
  * Compute confidence.
* **Layer 5 – Attribution & outputs**:

  * Build TS + summary + attribution + recommendations.
  * Write to SQL.

Everything lives in **ONE script / module** (or a strict package) called e.g. `prognostics_engine.py` which internally has sub-classes, but there is **no second engine**.

---

## 2. Detailed design & pseudocode

### 2.1 Schema / tables (conceptual)

Use / extend existing tables:

* `ACM_HealthTimeline (EquipID, RunID, Timestamp, HealthIndex, FusedZ, …)`
* `ACM_SensorHotspots (EquipID, RunID, SensorName, FailureContribution, ZScoreAtFailure, AlertCount, …)`
* `ACM_RUL_LearningState` (existing; may be extended to risk fields).
* `ACM_HealthForecast_TS`
* `ACM_FailureForecast_TS`
* `ACM_RUL_TS`
* `ACM_RUL_Summary`
* `ACM_RUL_Attribution`
* `ACM_MaintenanceRecommendation`

New / extended tables (suggested):

* `ACM_RiskFeatures_TS`
  Per asset/time snapshot of risk features (for training + debugging).
* `ACM_Risk_TS`
  Time series of risk probabilities per horizon.
* `ACM_Risk_Summary`
  One row per (EquipID, RunID) summarising risk per horizon.
* (Optional) `ACM_Risk_LearningState` or extra columns in `ACM_RUL_LearningState`.

---

### 2.2 Entry point pseudocode

```pseudo
function run_prognostics(sql_client, equip_id, run_id, output_manager=None, config_row=None):

    log("=== PROG: Starting prognostics (RUL + Risk) ===")

    # 1. Normalize IDs
    equip_id_int = ensure_equipid_int(equip_id)   # reuse / extend existing helpers
    run_id_str   = ensure_runid_str(run_id)

    # 2. Build config object
    cfg = build_prognostics_config(config_row)

    # 3. Cleanup old forecast / risk runs (to keep charts clean)
    cleanup_old_forecasts(sql_client, equip_id_int, cfg.forecast_runs_retain)
    cleanup_old_risk(sql_client, equip_id_int, cfg.risk_runs_retain)

    # 4. Load core time-series & contextual data
    health_df, health_quality = load_health_timeline(sql_client, equip_id_int, run_id_str, output_manager, cfg)
    episode_df               = load_episode_history(sql_client, equip_id_int, cfg)
    drift_df                 = load_drift_history(sql_client, equip_id_int, cfg)
    regime_df                = load_regime_history(sql_client, equip_id_int, cfg)
    maint_df, failure_df     = load_events_and_maintenance(sql_client, equip_id_int, cfg)
    sensor_hotspots_df       = load_sensor_hotspots(sql_client, equip_id_int, run_id_str)

    if health_df is None or health_df.empty:
        raise RuntimeError("Health timeline unavailable – cannot compute RUL / Risk")

    # 5. Load / initialize learning states
    rul_learning_state   = load_rul_learning_state(sql_client, equip_id_int)
    risk_learning_state  = load_risk_learning_state(sql_client, equip_id_int)

    # 6. Compute health-based RUL:
    rul_result = compute_rul_block(
        health_df = health_df,
        cfg = cfg.rul_cfg,
        learning_state = rul_learning_state,
        data_quality_flag = health_quality
    )

    # 7. Build risk features from ACM outputs (episodes, drift, regimes, health & hotspots)
    current_time = rul_result.current_time
    risk_features_row = build_risk_features_snapshot(
        equip_id = equip_id_int,
        run_id   = run_id_str,
        current_time = current_time,
        health_df = health_df,
        episode_df = episode_df,
        drift_df = drift_df,
        regime_df = regime_df,
        sensor_hotspots_df = sensor_hotspots_df,
        cfg = cfg.risk_cfg
    )

    # 8. Compute risk / hazard for multiple horizons
    risk_result = compute_risk_block(
        risk_features = risk_features_row,
        risk_cfg = cfg.risk_cfg,
        risk_learning_state = risk_learning_state,
        failure_events = failure_df,
        maintenance_events = maint_df
    )

    # 9. Fuse RUL and Risk into unified prognosis
    fused_prognosis = fuse_rul_and_risk(
        rul_result = rul_result,
        risk_result = risk_result,
        cfg = cfg
    )

    # 10. Compute final confidence score combining both blocks
    confidence = compute_overall_confidence(
        rul_result = rul_result,
        risk_result = risk_result,
        rul_learning_state = rul_learning_state,
        risk_learning_state = risk_learning_state,
        data_quality = health_quality
    )

    # 11. Build all output dataframes
    tables = build_all_outputs(
        equip_id = equip_id_int,
        run_id = run_id_str,
        rul_result = rul_result,
        risk_result = risk_result,
        fused_prognosis = fused_prognosis,
        sensor_hotspots_df = sensor_hotspots_df,
        confidence = confidence,
        data_quality = health_quality
    )

    # 12. Persist to SQL via OutputManager or direct bulk insert
    if output_manager is not None:
        write_outputs_via_output_manager(output_manager, tables, equip_id_int, run_id_str)
    else:
        write_outputs_direct(sql_client, tables)

    # 13. Update & save learning states (online updates)
    updated_rul_state  = update_rul_learning_state(rul_learning_state, fused_prognosis, failure_df, cfg)
    updated_risk_state = update_risk_learning_state(risk_learning_state, risk_result, failure_df, cfg)

    save_rul_learning_state(sql_client, updated_rul_state)
    save_risk_learning_state(sql_client, updated_risk_state)

    log("=== PROG: Prognostics complete – RUL, risk & recommendations updated ===")

    return tables
```

---

### 2.3 Config object

```pseudo
class PrognosticsConfig:
    rul_cfg: RULConfig            # essentially your existing RULConfig plus tweaks
    risk_cfg: RiskConfig          # new
    forecast_runs_retain: int
    risk_runs_retain: int

class RiskConfig:
    horizons_hours: list[float]       # e.g. [72, 168, 720] for 3d, 7d, 30d
    feature_windows_hours: list[float] # e.g. [24, 168, 720] for history windows
    label_min_severity: str        # what counts as failure (trip, BD, etc.)
    censoring_policy: dict         # e.g. {"min_future_window_coverage": 0.8}
    classifier_type: str           # e.g. "xgboost", "logistic", "random_forest"
    risk_learning_rate: float
    calibration_window: int        # number of samples for recalibration
    max_feature_age_days: int      # for training data
```

`build_prognostics_config(config_row)` pulls forecasting + risk-specific fields from your `ACM_Config` JSON blob, similar to how `rul_engine.py` reads `forecasting_cfg`. 

---

### 2.4 RUL Block: `compute_rul_block(...)`

This is essentially your current `compute_rul(...)` + multipath logic, cleaned & extended but conceptually similar.

```pseudo
function compute_rul_block(health_df, cfg: RULConfig, learning_state, data_quality_flag):

    # 1. Validate & sort
    assert "Timestamp" in health_df.columns
    df = health_df.sort_by("Timestamp").drop_duplicates("Timestamp").set_index("Timestamp")

    timestamps   = df.index
    health_values = df["HealthIndex"]

    if len(health_values) < cfg.min_points:
        return build_default_rul_result(cfg, data_quality_flag, current_time=last(timestamps))

    current_health = health_values[-1]
    current_time   = timestamps[-1]

    # 2. Healthy short-circuit
    if current_health >= cfg.healthy_threshold:
        return build_healthy_rul_result(df, cfg, data_quality_flag, current_time, current_health)

    # 3. Detect sampling interval and build future index
    sampling_interval_hours = median_diff_in_hours(timestamps)
    n_future_steps = max(ceil(cfg.max_forecast_hours / sampling_interval_hours), 10)
    future_index = date_range(start=current_time, periods=n_future_steps + 1, freq=f"{sampling_interval_hours}H")[1:]

    # 4. Fit degradation ensemble
    ensemble = RULDegradationEnsemble(cfg, learning_state)
    fit_status = ensemble.fit(timestamps, health_values)

    if no_true(fit_status.values):
        return build_default_rul_result(cfg, data_quality_flag, current_time)

    forecast = ensemble.forecast(future_index)
    # forecast.mean, forecast.std, forecast.per_model, forecast.weights, forecast.fit_status

    # 5. Build health forecast TS
    health_forecast_df = DataFrame({
        "Timestamp": future_index,
        "ForecastHealth": forecast.mean,
        "CI_Lower": forecast.mean - 1.96 * forecast.std,
        "CI_Upper": forecast.mean + 1.96 * forecast.std
    })

    # 6. Compute failure curve using health threshold
    failure_probs = compute_failure_probs_from_forecast(
        t_future = hours_since(current_time, future_index),
        mean = forecast.mean,
        std = forecast.std,
        threshold = cfg.health_threshold
    )

    failure_curve_df = DataFrame({
        "Timestamp": future_index,
        "FailureProb": failure_probs,
        "ThresholdUsed": cfg.health_threshold
    })

    # 7. Multipath RUL
    rul_multipath = compute_multipath_rul(
        health_forecast_df,
        failure_curve_df,
        current_time,
        cfg
    )

    # 8. Diagnostics
    model_diagnostics = {
        "weights": forecast.weights,
        "fit_status": forecast.fit_status,
        "per_model": forecast.per_model,
        "sampling_interval_hours": sampling_interval_hours
    }

    return RULResult(
        health_forecast = health_forecast_df,
        failure_curve = failure_curve_df,
        rul_multipath = rul_multipath,
        model_diagnostics = model_diagnostics,
        data_quality = data_quality_flag,
        current_time = current_time
    )
```

The internal ensemble class essentially wraps your existing AR1/Exponential/Weibull models but can later add new degradation models without changing the outer contract.

---

### 2.5 Risk feature builder: `build_risk_features_snapshot(...)`

This is where the **new theory** you asked for gets encoded.

```pseudo
function build_risk_features_snapshot(
    equip_id, run_id, current_time,
    health_df, episode_df, drift_df, regime_df, sensor_hotspots_df,
    cfg: RiskConfig
) -> dict:

    features = {}
    features["EquipID"] = equip_id
    features["RunID"]   = run_id
    features["SnapshotTime"] = current_time

    # --- A. Health-based features over windows ---
    for W_hours in cfg.feature_windows_hours:
        window_start = current_time - timedelta(hours=W_hours)
        sub = health_df[health_df["Timestamp"].between(window_start, current_time)]

        prefix = f"h{int(W_hours)}h"

        if sub not empty:
            H = sub["HealthIndex"]
            features[f"{prefix}_health_mean"] = mean(H)
            features[f"{prefix}_health_min"]  = min(H)
            features[f"{prefix}_health_max"]  = max(H)
            features[f"{prefix}_health_std"]  = std(H)
            features[f"{prefix}_health_below_thr_frac"] = frac(H < RULConfig.health_threshold)
            features[f"{prefix}_health_below_critical_frac"] = frac(H < cfg.critical_health_threshold)
            features[f"{prefix}_z_mean"]   = mean(sub["FusedZ"])
            features[f"{prefix}_z_max"]    = max(sub["FusedZ"])
            features[f"{prefix}_z_95pct"]  = percentile(sub["FusedZ"], 95)
        else:
            # fill with NaNs or encoded missing tokens
            set_health_window_features_to_missing(features, prefix)

    # --- B. Episode-based features ---
    # episode_df assumed to have: StartTime, EndTime, Severity, Type, Head, etc.
    for W_hours in cfg.feature_windows_hours:
        window_start = current_time - timedelta(hours=W_hours)
        ep_sub = episode_df[episode_df["StartTime"].between(window_start, current_time)]

        prefix = f"e{int(W_hours)}h"

        features[f"{prefix}_episode_count"] = len(ep_sub)
        features[f"{prefix}_max_severity"]  = max(ep_sub["Severity"], default=0)
        features[f"{prefix}_mean_severity"] = mean(ep_sub["Severity"], default=0)
        features[f"{prefix}_total_duration_h"] = sum((EndTime-StartTime).hours for each row)
        # counts by type/head:
        features[f"{prefix}_count_AR1"] = count(ep_sub where Head=="AR1")
        features[f"{prefix}_count_PCA"] = count(ep_sub where Head=="PCA")
        features[f"{prefix}_count_MHAL"] = ...
        # Recency
        last_ep = max(ep_sub["EndTime"], default=None)
        features[f"{prefix}_hours_since_last_episode"] =
            (current_time - last_ep).hours if last_ep else large_constant

    # --- C. Drift-based features ---
    # drift_df: (Timestamp, DriftMagnitude, Direction, etc.)
    for W_hours in cfg.feature_windows_hours:
        window_start = current_time - timedelta(hours=W_hours)
        d_sub = drift_df[drift_df["Timestamp"].between(window_start, current_time)]

        prefix = f"d{int(W_hours)}h"
        features[f"{prefix}_drift_count"] = len(d_sub)
        features[f"{prefix}_drift_max"]   = max(abs(d_sub["DriftMagnitude"]), default=0)
        features[f"{prefix}_drift_mean"]  = mean(abs(d_sub["DriftMagnitude"]), default=0)

        last_drift_time = max(d_sub["Timestamp"], default=None)
        features[f"{prefix}_hours_since_last_drift"] =
            (current_time - last_drift_time).hours if last_drift_time else large_constant

    # --- D. Regime-based features ---
    # regime_df: (Timestamp, RegimeID)
    for W_hours in cfg.feature_windows_hours:
        window_start = current_time - timedelta(hours=W_hours)
        r_sub = regime_df[regime_df["Timestamp"].between(window_start, current_time)]

        prefix = f"r{int(W_hours)}h"

        if r_sub not empty:
            # compute regime occupancy
            regime_counts = count_by(r_sub, "RegimeID")
            p = normalized_counts(regime_counts) # probabilities
            features[f"{prefix}_regime_entropy"] = -sum(p_i * log(p_i))
            features[f"{prefix}_regime_switches"] = count_switches(r_sub["RegimeID"])
            features[f"{prefix}_dominant_regime_id"] = argmax(regime_counts)
            features[f"{prefix}_dominant_regime_frac"] = max(p)
        else:
            set_regime_window_features_to_missing(features, prefix)

    # --- E. Sensor hotspot features ---
    # Using top-k sensors by FailureContribution or MaxAbsZ
    k = cfg.top_k_sensors
    hotspots = sensor_hotspots_df.sort_by("FailureContribution", desc=True).head(k)

    features["hotspot_count"] = len(hotspots)
    features["hotspot_total_contribution"] = sum(hotspots["FailureContribution"])
    features["hotspot_max_contribution"]   = max(hotspots["FailureContribution"], default=0)
    features["hotspot_mean_contribution"]  = mean(hotspots["FailureContribution"], default=0)

    # Optionally encode sensor-level features (e.g., top 3 sensor names via hashing)
    for i, row in enumerate(hotspots):
        idx = i + 1
        features[f"hotspot_{idx}_sensor_hash"] = hash_sensor_name(row["SensorName"])
        features[f"hotspot_{idx}_contribution"] = row["FailureContribution"]
        features[f"hotspot_{idx}_z_at_failure"] = row["ZScoreAtFailure"]
        features[f"hotspot_{idx}_alert_count"]  = row["AlertCount"]

    # --- F. Meta & context ---
    features["current_health"] = health_df[health_df["Timestamp"] == current_time]["HealthIndex"].iloc[0]
    features["current_fused_z"] = health_df[health_df["Timestamp"] == current_time]["FusedZ"].iloc[0]
    features["data_quality"] = encode_data_quality_flag(health_quality)
    features["asset_age_days"] = compute_asset_age_days(equip_id, maintenance_df, current_time)

    return features
```

This single `features` dict (or Series) is then used both for:

* **Online inference** at runtime.
* **Training dataset** creation offline.

---

### 2.6 Risk / hazard block: `compute_risk_block(...)`

We model **event-in-window probabilities** for multiple horizons (3d, 7d, 30d). The pseudocode assumes you already have **trained models** periodically retrained offline; online you just load & apply, plus do some lightweight calibration / learning.

```pseudo
function compute_risk_block(risk_features, risk_cfg, risk_learning_state, failure_events):

    # 1. Extract feature vector X from risk_features
    X = vectorize_features(risk_features)

    # 2. Load (or have injected) trained models per horizon
    models = load_risk_models_from_disk_or_sql(risk_cfg)  
      # e.g. models["72h"], models["168h"], models["720h"]

    # 3. For each horizon, compute probability:
    risk_per_horizon = {}
    for H in risk_cfg.horizons_hours:
        model = models[str(H)]
        raw_prob = model.predict_proba(X)[1]  # probability of event in (t, t+H]

        # Apply horizon-specific calibration (e.g. isotonic, Platt) stored in learning_state
        calibrated_prob = apply_calibration(raw_prob, horizon=H, state=risk_learning_state)

        risk_per_horizon[H] = {
            "raw_prob": raw_prob,
            "calibrated_prob": calibrated_prob
        }

    # 4. Compute derived quantities (e.g. monotonic hazard)
    # Ensure P(3d) <= P(7d) <= P(30d) by isotonic correction if needed
    ordered_H = sorted(risk_cfg.horizons_hours)
    probs = [risk_per_horizon[H]["calibrated_prob"] for H in ordered_H]
    monotone_probs = enforce_monotonic_increasing(probs)
    for H, p in zip(ordered_H, monotone_probs):
        risk_per_horizon[H]["monotone_prob"] = p

    # 5. Derive discrete hazard / approximate survival curve
    # Convert window probs into approximate hazard or "risk density"
    hazard_curve = derive_discrete_hazard_from_window_probs(ordered_H, monotone_probs)

    return RiskResult(
        risk_per_horizon = risk_per_horizon,
        hazard_curve = hazard_curve,
        features_snapshot = risk_features
    )
```

`derive_discrete_hazard_from_window_probs` is where we turn multi-horizon window probabilities into an approximate survival distribution (for RUL banding). Implementation detail:

* For example, assume constant hazard within each band and solve for hazard that yields those cumulative risks.

---

### 2.7 Fusion layer: `fuse_rul_and_risk(...)`

This layer combines **health-driven RUL** and **risk-driven horizons** into a single interpretable RUL band and final RUL.

```pseudo
function fuse_rul_and_risk(rul_result: RULResult, risk_result: RiskResult, cfg: PrognosticsConfig):

    # 1. Read RUL from health block
    health_rul = rul_result.rul_multipath["rul_final_hours"]
    health_lower = rul_result.rul_multipath["lower_bound_hours"]
    health_upper = rul_result.rul_multipath["upper_bound_hours"]

    # 2. Read risk horizon probabilities
    # Example: {72: p72, 168: p168, 720: p720}
    risk_probs = {H: risk_result.risk_per_horizon[H]["monotone_prob"] for H in cfg.risk_cfg.horizons_hours}

    # 3. Derive risk-based RUL band:
    # Example heuristic: choose the smallest horizon where P > threshold
    risk_threshold = cfg.risk_cfg.risk_threshold   # e.g. 0.4–0.6 depending on appetite
    sorted_H = sorted(cfg.risk_cfg.horizons_hours)
    risk_rul_band = "> max"  # default
    risk_rul_midpoint_hours = cfg.rul_cfg.max_forecast_hours

    for H in sorted_H:
        if risk_probs[H] >= risk_threshold:
            risk_rul_band = f"<= {H}h"
            risk_rul_midpoint_hours = H * 0.7  # e.g. assume event around 70% of horizon
            break

    # 4. Combine:
    # - Health RUL gives a continuous number (with CI).
    # - Risk RUL gives a coarse “should be within X hours/days”.
    #
    # We can blend them:
    #
    # health_weight, risk_weight determined by config and data quality.
    health_weight = cfg.fusion.health_weight
    risk_weight   = cfg.fusion.risk_weight

    fused_rul = health_weight * health_rul + risk_weight * risk_rul_midpoint_hours

    # 5. Adjust bounds by intersecting both:
    fused_lower = max(0, min(health_lower, risk_rul_midpoint_hours))
    fused_upper = min(cfg.rul_cfg.max_forecast_hours, max(health_upper, risk_rul_midpoint_hours))

    # 6. Also classify final RUL band (for operators)
    fused_band = classify_rul_band(fused_rul, cfg.rul_cfg)

    return FusedPrognosis(
        fused_rul_hours = fused_rul,
        fused_lower_hours = fused_lower,
        fused_upper_hours = fused_upper,
        fused_band = fused_band,
        health_rul_hours = health_rul,
        health_bounds = (health_lower, health_upper),
        risk_probs = risk_probs,
        risk_rul_band = risk_rul_band
    )
```

Later you can refine the fusion rule (e.g. if health block has very low confidence, lean more on risk block, etc.).

---

### 2.8 Overall confidence: `compute_overall_confidence(...)`

Extend your existing confidence logic:

```pseudo
function compute_overall_confidence(rul_result, risk_result, rul_learning_state, risk_learning_state, data_quality):

    # Reuse existing CI-based, model-agreement, calibration, data-quality factors from rul_result
    rul_conf = compute_confidence_from_rul(rul_result, rul_learning_state, data_quality)

    # Risk confidence factors:
    # - Model agreement across horizons (e.g. well-calibrated monotone profile).
    # - Calibration residuals (from risk_learning_state).
    # - Feature completeness (no missing major windows).
    risk_conf = compute_confidence_from_risk(risk_result, risk_learning_state, data_quality)

    # Combine with weights
    alpha = 0.6  # emphasise RUL block if you trust it more
    overall = alpha * rul_conf + (1 - alpha) * risk_conf

    return clamp(overall, 0.0, 1.0)
```

---

### 2.9 Output builders

Extend your existing `make_*` functions to also produce risk tables, but keep schema-compatible.

```pseudo
function build_all_outputs(equip_id, run_id, rul_result, risk_result, fused_prognosis, sensor_hotspots_df, confidence, data_quality):

    tables = {}

    # Existing RUL outputs
    tables["ACM_HealthForecast_TS"] = make_health_forecast_ts(rul_result.health_forecast, run_id, equip_id)
    tables["ACM_FailureForecast_TS"] = make_failure_forecast_ts(rul_result.failure_curve, run_id, equip_id)
    tables["ACM_RUL_TS"] = make_rul_ts(rul_result.health_forecast, fused_prognosis, rul_result.current_time, run_id, equip_id, confidence)
    tables["ACM_RUL_Summary"] = make_rul_summary(fused_prognosis, rul_result.model_diagnostics, data_quality, run_id, equip_id, confidence)
    tables["ACM_RUL_Attribution"] = build_sensor_attribution(sensor_hotspots_df, fused_prognosis, rul_result.current_time, run_id, equip_id)
    tables["ACM_MaintenanceRecommendation"] = build_maintenance_recommendation(fused_prognosis, data_quality, confidence, run_id, equip_id)

    # New risk outputs
    tables["ACM_Risk_TS"] = make_risk_ts(
        risk_result = risk_result,
        fused_prognosis = fused_prognosis,
        equip_id = equip_id,
        run_id = run_id
    )

    tables["ACM_Risk_Summary"] = make_risk_summary(
        risk_result = risk_result,
        fused_prognosis = fused_prognosis,
        equip_id = equip_id,
        run_id = run_id,
        confidence = confidence,
        data_quality = data_quality
    )

    tables["ACM_RiskFeatures_TS"] = make_risk_features_ts(
        risk_result.features_snapshot,
        equip_id,
        run_id
    )

    return tables
```

`make_risk_ts` would create rows like `(RunID, EquipID, SnapshotTime, HorizonHours, RiskProbRaw, RiskProbCalibrated, RiskProbMonotone, HazardValue, ...)`.

---

## 3. Separate explainer document (for humans)

You can treat this as a Markdown doc (`docs/Prognostics_Engine_Explainer.md`).

### 3.1 Purpose

* **What this engine does**:

  * Transforms ACM outputs (health, anomalies, drift, regimes, hotspots) into:

    * Projected health trajectory.
    * Probabilities of failure/trip within given horizons.
    * An interpretable RUL estimate + band.
    * Actionable maintenance recommendations.
* **Why unify**:

  * No separate legacy RUL engines.
  * One canonical “source of truth” for RUL & risk.

### 3.2 Inputs

1. **Health timeline**

   * `ACM_HealthTimeline` for a specific `EquipID` and `RunID`.
   * Fields: `Timestamp`, `HealthIndex (0–100)`, `FusedZ` etc.
2. **Episodes**

   * Detected anomaly episodes (AR1, PCA, MHAL, etc.).
   * Each episode has start/end, severity, head, root cause hints.
3. **Drift events**

   * Times when statistical drift is detected (data, regimes, environment).
4. **Regime timeline**

   * Operating states (e.g. load regimes) with timestamps.
5. **Sensor hotspots**

   * Sensor-level contributions to anomalies and potential failures.
6. **Events & maintenance logs**

   * CMMS / trip events used for risk model training and weak labels.
7. **Config** (`ACM_Config` row)

   * RUL thresholds, max horizon, risk horizons, banding rules, etc.
8. **Learning state / calibration**

   * RUL & risk learning state per equipment.

### 3.3 How it works – simplified storyline

1. **Health forecast**:

   * Fit several degradation models to past `HealthIndex`:

     * AR1 with drift.
     * Exponential decay.
     * Weibull-inspired power law.
   * Combine them into an ensemble health forecast with uncertainty.
   * Find when the forecast is likely to cross the health threshold (and CIs).
2. **Risk feature extraction**:

   * Summarise **recent history** as:

     * Time spent in bad health.
     * Number, severity & recency of episodes.
     * Amount and recency of drift.
     * Stability / volatility of regimes.
     * Sensor-level hotspot patterns.
   * This becomes the **“clinical history” vector** of the asset.
3. **Risk prediction**:

   * Use trained models to estimate the probability that:

     * A disruptive event (trip, breakdown, major maintenance) occurs in:

       * Next 3 days.
       * Next 7 days.
       * Next 30 days.
   * Calibrate these probabilities and enforce logical monotonicity.
4. **RUL fusion**:

   * Combine health-based RUL and risk-based horizon probabilities.
   * Output:

     * RUL value.
     * RUL band (e.g. `<12h`, `12–72h`, `3–7d`, `>7d`).
     * Confidence score.
5. **Attribution**:

   * At sensor level:

     * Which sensors contribute most to failure risk and anomalies.
   * At “history” level:

     * Which aspects of the recent past (e.g. frequent episodes, drift) drive risk.
6. **Outputs**:

   * **Time series**: forecasted health, failure probability, RUL trajectory, risk vs horizon.
   * **Summaries**: single row per run with RUL, risk band, main drivers.
   * **Operator-facing interpretation**:

     * Clear recommendations (continue, intensify monitoring, plan maintenance, act now).

### 3.4 Interpretation guidelines for operators

* **RUL (hours / days)**:

  * Not an exact countdown; think of it as a median scenario under current conditions.
* **Risk per horizon (e.g. “Risk(7d) = 0.45”)**:

  * Probability of a disruptive event if nothing changes in operation or maintenance.
* **Bands & recommendations**:

  * Normal operation, watch, plan, urgent.
  * Confidence and data quality are always shown:

    * Low confidence or poor data quality means “verify” and possibly escalate.

### 3.5 Limitations & assumptions

* No explicit physics-based degradation model (purely data-driven).
* Labels are weak; risk models are conditioned on observed historical practice.
* RUL / risk assume similar operating conditions to the past.
* Non-stationarity is mitigated but not eliminated; models require periodic retraining and calibration.

---

## 4. Detailed implementation task list

(High-level IDs; you can expand to your backlog structure later.)

| ID      | Area          | Task                                                                                                                                    | Notes / Dependencies                                                                                                      |
| ------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| CFG-01  | Config        | Define `PrognosticsConfig` and `RiskConfig` classes (Python dataclasses)                                                                | Extend existing `RULConfig`; include horizons, windows, thresholds, fusion weights, etc.                                  |
| CFG-02  | Config        | Extend ACM config table / JSON schema to include risk-specific settings                                                                 | Keep backwards compatibility if risk not configured.                                                                      |
| IO-01   | I/O           | Refactor existing `run_rul` into `compute_rul_block` with same behaviour but internal API returning `RULResult` object                  | No functional regression; just rewrap.                                                                                    |
| IO-02   | I/O           | Implement new `run_prognostics` entrypoint that calls `compute_rul_block` and new risk functions                                        | Ensure existing callers of `run_rul` can be migrated or preserved.                                                        |
| IO-03   | I/O           | Implement data loaders for episodes, drift, regimes, maintenance/failure events                                                         | SQL-only; include robust logging when tables missing or empty.                                                            |
| IO-04   | I/O           | Implement loader for pre-trained risk models (per horizon) from SQL or filesystem                                                       | Decide storage format (joblib in artifacts per equip vs global models).                                                   |
| IO-05   | I/O           | Implement new write functions: `write_outputs_direct` and `write_outputs_via_output_manager` that handle all tables incl. new risk ones | Respect autocommit, error handling, row counts.                                                                           |
| RUL-01  | RUL Block     | Encapsulate AR1/Exp/Weibull & ensemble into `RULDegradationEnsemble` class                                                              | Use existing model implementations but standardise interface `fit()`, `forecast()`.                                       |
| RUL-02  | RUL Block     | Cleanly separate health-threshold crossing and multipath RUL logic into `compute_multipath_rul`                                         | Reuse current multipath logic; parameterise thresholds & quantiles.                                                       |
| RUL-03  | RUL Block     | Confirm & document assumptions for failure probability computation from health forecast                                                 | Normal approximation, independence, clipping; add comments in code.                                                       |
| RUL-04  | RUL Block     | Extend existing `RUL_LearningState` to include any new metrics required for calibration or confidence                                   | Migration script for existing rows.                                                                                       |
| RISK-01 | Risk Features | Design and document `ACM_RiskFeatures_TS` schema                                                                                        | Columns for all snapshot features; keys: (EquipID, RunID, SnapshotTime).                                                  |
| RISK-02 | Risk Features | Implement `build_risk_features_snapshot` exactly as per pseudocode (health, episodes, drift, regimes, hotspots, meta)                   | Handle missing data gracefully; unify naming conventions; normalise durations.                                            |
| RISK-03 | Risk Features | Implement optional helper to write the snapshot to `ACM_RiskFeatures_TS`                                                                | Controlled via config flag.                                                                                               |
| RISK-04 | Risk Models   | Define `RiskResult` dataclass (risk_per_horizon, hazard_curve, features_snapshot)                                                       | Provide typed structure to downstream functions.                                                                          |
| RISK-05 | Risk Models   | Implement `compute_risk_block` for **inference** using trained horizon models                                                           | Including calibration and monotonicity enforcement.                                                                       |
| RISK-06 | Risk Models   | Design offline training pipeline (outside this script) to train models per horizon on historical `ACM_RiskFeatures_TS` + event labels   | Not in this script but must be documented and tracked.                                                                    |
| RISK-07 | Risk Models   | Define label creation logic for failures (trip/BD/long maintenance) and censoring                                                       | Implementation likely as SQL procedure or Python offline job.                                                             |
| RISK-08 | Risk Models   | Implement `derive_discrete_hazard_from_window_probs`                                                                                    | Use simple piecewise-constant hazard assumption; unit-tested with toy examples.                                           |
| FUSE-01 | Fusion        | Implement `fuse_rul_and_risk` function exactly per pseudocode                                                                           | Support configurable fusion weights; handle missing risk or RUL gracefully.                                               |
| FUSE-02 | Fusion        | Implement `classify_rul_band` function mapping RUL hours into band names (Normal/Watch/Plan/Urgent) based on config                     | Align with existing maintenance bands used in `ACM_MaintenanceRecommendation`.                                            |
| CONF-01 | Confidence    | Refactor existing `compute_confidence` into `compute_confidence_from_rul` for RUL-only                                                  | Preserve current behaviour initially.                                                                                     |
| CONF-02 | Confidence    | Implement `compute_confidence_from_risk` using: horizon consistency, calibration error, feature completeness, data quality              | Use 0–1 scale; documented formula.                                                                                        |
| CONF-03 | Confidence    | Implement `compute_overall_confidence` that blends RUL and risk confidence                                                              | Configurable weights; clamp to [0,1].                                                                                     |
| ATTR-01 | Attribution   | Extend attribution logic to include **risk drivers** in addition to sensor hotspots                                                     | e.g. log: “High risk driven by high episode frequency + large drift + stable hotspot set”.                                |
| ATTR-02 | Attribution   | Extend `ACM_RUL_Attribution` or create `ACM_Risk_Attribution` table to store risk-driver explanations                                   | Decide schema – maybe JSON blob of top risk features per run.                                                             |
| OUT-01  | Outputs       | Implement `make_risk_ts` to create `ACM_Risk_TS` with per-horizon probabilities and hazard                                              | Columns: RunID, EquipID, SnapshotTime, HorizonHours, RiskProbRaw, RiskProbCalibrated, RiskProbMonotone, HazardValue.      |
| OUT-02  | Outputs       | Implement `make_risk_summary` to create `ACM_Risk_Summary`                                                                              | Columns: RunID, EquipID, SnapshotTime, Risk_3d, Risk_7d, Risk_30d, HighestRiskHorizon, RiskBand, Confidence, DataQuality. |
| OUT-03  | Outputs       | Implement `make_risk_features_ts` to persist snapshot features                                                                          | Ensure only one row per run by default; or per SnapshotTime if multiple snapshots are supported.                          |
| OUT-04  | Outputs       | Update `build_maintenance_recommendation` to consider both fused RUL band and horizon risk                                              | e.g. urgent if short horizon risk above threshold even if health RUL is large.                                            |
| LRN-01  | Learning      | Implement `load_risk_learning_state` / `save_risk_learning_state` with SQL-backed upsert                                                | Similar to existing RUL learning state pattern.                                                                           |
| LRN-02  | Learning      | Implement `update_risk_learning_state` to track calibration residuals, Brier scores, etc.                                               | Online updates when you observe real failures vs predicted risk (future extension).                                       |
| LRN-03  | Learning      | Implement `update_rul_learning_state` to incorporate realised failure times where available                                             | Adjust model weights & calibration; ensure numerically stable.                                                            |
| VAL-01  | Validation    | Add unit tests for: health forecast, failure prob calculation, multipath RUL, risk feature building                                     | Synthetic data; check invariants and edge cases.                                                                          |
| VAL-02  | Validation    | Add unit tests for monotone horizon probabilities and hazard derivation                                                                 | Guarantee P(3d) ≤ P(7d) ≤ P(30d).                                                                                         |
| VAL-03  | Validation    | Add regression tests to ensure new engine reproduces existing RUL results (within tolerance) when risk module is disabled               | Protect against regressions during refactor.                                                                              |
| DOC-01  | Docs          | Create `Prognostics_Engine_Explainer.md` with sections: Purpose, Inputs, Workflow, Interpretation, Limitations                          | Base on section 3 above.                                                                                                  |
| DOC-02  | Docs          | Update ACM Knowledge Base / README to mark old RUL engines as deprecated and reference the unified prognostics engine                   | Include migration notes for callers.                                                                                      |
| GRAF-01 | Visualisation | Design Grafana panels to show: health forecast + RUL band, risk per horizon, fused recommendation                                       | Use existing ACM dashboard as base; define JSON layout.                                                                   |
| GRAF-02 | Visualisation | Add operator-facing legends & tooltips on dashboards to explain RUL & risk numbers and confidence                                       | Keep wording aligned with explainer document.                                                                             |

---

