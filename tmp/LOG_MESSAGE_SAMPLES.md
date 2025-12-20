# Detailed Log Message Analysis by Module


## acm_main.py - Main Pipeline Orchestrator

**Total log calls**: 291

### Log Level Distribution

- **info**: 158 calls
- **warn**: 117 calls
- **error**: 11 calls
- **debug**: 2 calls
- **warning**: 1 calls
- **section**: 1 calls
- **status**: 1 calls

### Component Distribution

- **MODEL**: 48 calls
- **THRESHOLD**: 18 calls
- **FEAT**: 18 calls
- **DATA**: 16 calls
- **RUN**: 16 calls
- **REGIME**: 14 calls
- **BASELINE**: 11 calls
- **ADAPTIVE**: 11 calls
- **CFG**: 10 calls
- **AUTO-TUNE**: 10 calls

### Sample Messages

#### INFO Messages (samples)

- L365: [META] `Written run metadata to {meta_path}`
- L513: [CFG] `Loaded config from SQL for {equip_label}`
- L521: [CFG] `Loading config for {equipment_name} (EquipID={equip_id}) from {csv_path}`
- L523: [CFG] `Loading global config from {csv_path}`
- L647: [THRESHOLD] `Adaptive thresholds disabled - using static config values`

#### WARN Messages (samples)

- L191: [LOG] `SQL sink disable flag in config is ignored; SQL logging is always enabled in SQL mode.`
- L198: [CONFIG] `File logging disabled in SQL-only mode (ignoring --log-file={log_file})`
- L237: [DATA] `[{label}] Falling back to manual nearest mapping: {err}`
- L264: [META] `Skipped meta.json: invalid run_dir value`
- L367: [META] `Failed to write meta.json: {meta_err}`

#### ERROR Messages (samples)

- L731: [THRESHOLD] `Adaptive threshold calculation failed: {threshold_e}`
- L785: [RUN] `Failed to start SQL run: {exc}`
- L1034: [RUN] `Failed to start SQL run: {e}`
- L1071: [OUTPUT] `Diagnostic: SQL client attached but health check failed: {_e}`
- L2254: [ADAPTIVE] `Failed to update config CSV: {e}`

---

## output_manager.py - Data Persistence Manager

**Total log calls**: 114

### Log Level Distribution

- **warn**: 66 calls
- **info**: 36 calls
- **error**: 11 calls
- **debug**: 1 calls

### Component Distribution

- **OUTPUT**: 49 calls
- **DATA**: 26 calls
- **ANALYTICS**: 18 calls
- **RUL**: 4 calls
- **THRESHOLD**: 3 calls
- **SCHEMA**: 2 calls
- **EPISODES**: 2 calls
- **HEALTH**: 2 calls
- **DEFECTS**: 2 calls

### Sample Messages

#### INFO Messages (samples)

- L635: [OUTPUT] ` + f`
- L663: [OUTPUT] ` + `
- L669: [OUTPUT] ` + `
- L673: [OUTPUT] ` + `
- L677: [OUTPUT] ` + f`

#### WARN Messages (samples)

- L250: [DATA] `Dropped {before_drop - len(df)} rows with invalid timestamps from {label}`
- L255: [DATA] `Dropping {future_rows} future timestamp row(s) from {label} (cutoff={now_cutoff:%Y-%m-%d %H:%M:%S})`
- L675: [OUTPUT] `Autocommit is ON - no explicit commit needed`
- L754: [DATA] `Invalid cold_start_split_ratio={cold_start_split_ratio}, using default 0.6`
- L777: [DATA] `Cold-start training data ({len(train_raw)} rows) is below recommended minimum ({min_train_samples} rows)`

#### ERROR Messages (samples)

- L679: [OUTPUT] `Batched transaction commit failed: {e}`
- L688: [OUTPUT] `Batched transaction rolled back: {e}`
- L945: [DATA] `Failed to load from SQL historian: {e}`
- L1137: [OUTPUT] `SQL health check failed: {e}`
- L1345: [OUTPUT] `Failed to process artifact {artifact_name}: {e}`

---

## model_persistence.py - Model Storage

**Total log calls**: 62

### Log Level Distribution

- **warn**: 28 calls
- **info**: 23 calls
- **error**: 10 calls
- **debug**: 1 calls

### Component Distribution

- **MODEL-SQL**: 21 calls
- **MODEL**: 18 calls
- **REGIME_STATE**: 11 calls
- **FORECAST_STATE**: 10 calls
- **META**: 2 calls

### Sample Messages

#### INFO Messages (samples)

- L194: [FORECAST_STATE] `Saved state v{state.state_version} to ACM_ForecastState (EquipID={state.equip_id})`
- L248: [FORECAST_STATE] `Loaded state v{state.state_version} from SQL (EquipID={equip_id})`
- L251: [FORECAST_STATE] `No prior forecast state found for EquipID={equip_id}`
- L409: [REGIME_STATE] `Saved state v{state.state_version} to ACM_RegimeState (EquipID={state.equip_id})`
- L464: [REGIME_STATE] `Loaded state v{state.state_version} from SQL (EquipID={equip_id})`

#### WARN Messages (samples)

- L117: [FORECAST_STATE] `Failed to deserialize forecast horizon: {e}`
- L133: [FORECAST_STATE] `Failed to serialize forecast horizon: {e}`
- L314: [REGIME_STATE] `Failed to deserialize cluster centers: {e}`
- L324: [REGIME_STATE] `Failed to deserialize scaler params: {e}`
- L336: [REGIME_STATE] `Failed to deserialize PCA params: {e}`

#### ERROR Messages (samples)

- L147: [FORECAST_STATE] `SQL client required for SQL-only mode`
- L196: [FORECAST_STATE] `Failed to save state to SQL: {e}`
- L212: [FORECAST_STATE] `SQL client required for SQL-only mode`
- L216: [FORECAST_STATE] `equip_id required for SQL-only mode`
- L254: [FORECAST_STATE] `Failed to load state from SQL: {e}`

---

## fuse.py - Anomaly Fusion Engine

**Total log calls**: 24

### Log Level Distribution

- **warn**: 14 calls
- **info**: 7 calls
- **debug**: 3 calls

### Component Distribution

- **TUNE**: 10 calls
- **CAL**: 4 calls
- **FUSE**: 3 calls

### Sample Messages

#### INFO Messages (samples)

- L402: [TUNE] `Detector weight auto-tuning ({tuning_method}):`
- L474: [CAL] `Self-tuning enabled. Target FP rate {target_fp_rate:.3%} -> q={auto_q:.4f}, threshold={self.q_thresh:.4f}`
- L504: [CAL] `Fitting per-regime thresholds for {len(unique_regimes)} regimes.`
- L609: [FUSE] `{len(missing)} detector(s) absent at fusion time: {missing}`
- L886: [NO COMPONENT] `[FUSE] Auto-tuned CUSUM parameters (source=%s):`

#### WARN Messages (samples)

- L72: [TUNE] `Unknown tuning method `
- L108: [TUNE] `Episodes provided but fused_index missing or misaligned; skipping PR-AUC labeling`
- L176: [TUNE] `{detector_name}: under-sampled ({n_valid}/{min_samples_required}) - using prior`
- L194: [TUNE] `{detector_name}: all zeros - limited separability`
- L210: [TUNE] `{detector_name}: all same sign - limited separability`

---

## regimes.py - Operating Regime Detection

**Total log calls**: 39

### Log Level Distribution

- **warn**: 24 calls
- **info**: 15 calls

### Component Distribution

- **REGIME**: 29 calls
- **TRANSIENT**: 6 calls
- **REGIME_ALIGN**: 4 calls

### Sample Messages

#### INFO Messages (samples)

- L350: [REGIME] `Using {len(available_operational)} raw operational sensors for regime clustering: {available_operational[:5]}{`
- L595: [REGIME] `Silhouette sweep: {formatted}`
- L598: [REGIME] `Score sweep: {formatted}`
- L1232: [REGIME] `Episodes after validation: {final_count}/{initial_count}`
- L1605: [REGIME_ALIGN] `Cluster count changed: prev_k={prev_k}, new_k={new_k}`

#### WARN Messages (samples)

- L357: [REGIME] `No operational columns found matching keywords {operational_keywords[:5]}. Falling back to PCA features.`
- L379: [REGIME] `PCA variance ratio out of bounds: {pca_variance_ratio}. Resetting to NaN.`
- L383: [REGIME] `PCA variance vector contains non-finite values. Check numerical stability.`
- L490: [REGIME] `Limiting auto-k sweep to {max_models} models (k_max {k_max}->{allowed_max}) for budget`
- L612: [REGIME] `Input validation: {issue}`

---

## forecast_engine.py - RUL Forecasting

**Total log calls**: 25

### Log Level Distribution

- **warn**: 13 calls
- **info**: 6 calls
- **error**: 6 calls

### Sample Messages

#### INFO Messages (samples)

- L244: [NO COMPONENT] `[ForecastEngine] Auto-tuning triggered at DataVolume={state.data_volume_analyzed}`
- L415: [NO COMPONENT] `[ForecastEngine] Warm-started model from previous state`
- L759: [NO COMPONENT] `[ForecastEngine] Wrote {len(tables_written)} forecast tables to SQL`
- L773: [NO COMPONENT] `[ForecastEngine] Wrote sensor forecasts for {len(sensor_attributions)} sensors`
- L1381: [NO COMPONENT] `[RegimeConditioned] Computed stats for {len(self._regime_stats)} regimes`

#### WARN Messages (samples)

- L199: [NO COMPONENT] `[ForecastEngine] No health data available; skipping forecast`
- L216: [NO COMPONENT] `[ForecastEngine] {friendly_msg}; skipping forecast (not fatal)`
- L224: [NO COMPONENT] `[ForecastEngine] GAPPY data detected - proceeding with available data (historical replay mode)`
- L290: [NO COMPONENT] `[ForecastEngine] Regime-conditioned forecasting skipped: {e}`
- L417: [NO COMPONENT] `[ForecastEngine] Failed to warm-start model: {e}`

#### ERROR Messages (samples)

- L318: [NO COMPONENT] `[ForecastEngine] Forecast failed: {e}`
- L822: [NO COMPONENT] `[ForecastEngine] Failed to write outputs: {e}`
- L915: [NO COMPONENT] `[ForecastEngine] Regime-conditioned forecasting failed: {e}`
- L1129: [NO COMPONENT] `[ForecastEngine] Sensor forecasting failed: {e}`
- L1385: [NO COMPONENT] `[RegimeConditioned] Failed to compute regime stats: {e}`

---

## smart_coldstart.py - Coldstart Handler

**Total log calls**: 29

### Log Level Distribution

- **info**: 15 calls
- **warn**: 8 calls
- **error**: 6 calls

### Component Distribution

- **COLDSTART**: 29 calls

### Sample Messages

#### INFO Messages (samples)

- L81: [COLDSTART] `Auto-detected tick_minutes from data cadence: {tick_minutes} minutes`
- L170: [COLDSTART] `Detected data cadence: {most_common_interval} seconds ({most_common_interval/60:.1f} minutes)`
- L231: [COLDSTART] `Loading from EARLIEST data: {start_time}`
- L232: [COLDSTART] `Calculated optimal window: {required_minutes} minutes ({required_minutes/60:.1f} hours)`
- L233: [COLDSTART] `Expected rows: ~{int(required_minutes / cadence_minutes)} (target: {required_rows})`

#### WARN Messages (samples)

- L84: [COLDSTART] `Could not detect cadence, using default tick_minutes: {tick_minutes}`
- L151: [COLDSTART] `Insufficient data for cadence detection: {len(rows)} rows`
- L207: [COLDSTART] `Could not detect cadence, assuming {data_cadence_seconds}s`
- L238: [COLDSTART] `No data found in {table_name}, using lookback from current batch`
- L299: [COLDSTART] `Insufficient data in {window_hours:.1f}h window - batch will NOOP (models exist but no new data)`

#### ERROR Messages (samples)

- L115: [COLDSTART] `Failed to check status: {e}`
- L175: [COLDSTART] `Failed to detect cadence: {e}`
- L244: [COLDSTART] `Error querying earliest timestamp: {e}`
- L393: [COLDSTART] `Unexpected error on attempt {attempt}: {e}`
- L414: [COLDSTART] `Failed to load data window: {e}`

---

## sql_batch_runner.py - Batch Processing Script

**Total log calls**: 94

### Log Level Distribution

- **info**: 42 calls
- **warn**: 24 calls
- **error**: 14 calls
- **ok**: 5 calls
- **status**: 5 calls
- **header**: 4 calls

### Component Distribution

- **SQL**: 1 calls
- **DRY**: 1 calls

### Sample Messages

#### INFO Messages (samples)

- L139: [SQL] `Connection test OK`
- L168: [NO COMPONENT] `[ID] Registered/Resolved EquipID={eid} for {equip_name}`
- L223: [NO COMPONENT] `[CFG] Set runtime.tick_minutes={minutes} for EquipID={equip_id}`
- L278: [NO COMPONENT] `[RESET] Truncating {total_tables} ACM output tables for EquipID={equip_id}...`
- L318: [NO COMPONENT] `[RESET] Deleted {total_deleted:,} rows from {table}`

#### WARN Messages (samples)

- L92: [NO COMPONENT] `[PRECHECK] {equip_name}: Historian table query returned no result`
- L96: [NO COMPONENT] `[PRECHECK] {equip_name}: Historian table has no min/max timestamps`
- L118: [NO COMPONENT] `[PRECHECK] {equip_name}: Historian overview failed: {e}`
- L182: [NO COMPONENT] `Could not resolve EquipID for {equip_name}: {e}`
- L201: [NO COMPONENT] `Could not read config {param_path} for EquipID={equip_id}: {e}`

#### ERROR Messages (samples)

- L142: [NO COMPONENT] `SQL connection test failed: {exc}`
- L496: [NO COMPONENT] `[QA] Output inspection failed for {equip_name}: {e}`
- L570: [NO COMPONENT] `Failed to get data range for {equip_name}: {e}`
- L741: [NO COMPONENT] `[RUN-DEBUG] {equip_name}: acm_main exited with code {process.returncode}`
- L774: [NO COMPONENT] `[COLDSTART] {equip_name}: No data available in historian`

---
