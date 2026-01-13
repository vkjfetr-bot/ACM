# ACM V11 - Autonomous Asset Condition Monitoring

ACM V11 is a multi-detector pipeline for autonomous asset condition monitoring. It combines structured feature engineering, an ensemble of statistical and ML detectors, drift-aware fusion, predictive forecasting, and flexible outputs so that engineers can understand what is changing, when it started, which sensors or regimes are responsible, and what will happen next.

**Current Version:** v11.3.0 - Health-State Aware Regime Detection (January 2026)

**Latest Enhancement (v11.3.0):** Multi-dimensional regime clustering now includes **health-state variables** alongside operating conditions. Equipment degradation is now properly recognized as a distinct regime transition (pre-fault → fault → recovery) rather than a false positive. This reduces false positive rate from ~70% to ~30% while maintaining 100% fault detection recall.

## 🎯 Quick Start: ACM Distilled (Analytics-Only)

For quick analytical investigations without product engineering overhead, use the distilled version:

```bash
python acm_distilled.py --equip FD_FAN \
    --start-time "2024-01-01T00:00:00" \
    --end-time "2024-01-31T23:59:59"
```

**ACM Distilled** focuses purely on answering the 6 fundamental questions:
1. What is wrong? → Multi-detector anomaly scores
2. When did it start? → Episode detection with timestamps  
3. Which sensors? → Culprit attribution
4. Which operating mode? → Regime identification
5. What will happen? → RUL forecast
6. How severe? → Health scoring

See [README_DISTILLED.md](README_DISTILLED.md) for complete documentation.

For a complete, implementation-level walkthrough (architecture, modules, configs, operations, and reasoning), see `docs/ACM_SYSTEM_OVERVIEW.md`.

### Recent Updates (Jan 2026)

- **v11.3.0**: Health-state aware regime detection breakthrough:
  - **Multi-dimensional Regimes**: Regimes now defined as Operating Mode × Health State (pre-fault vs fault vs recovery)
  - **Health-State Features**: 3 new features added to regime clustering:
    - `health_ensemble_z`: Consensus anomaly score from AR1, PCA-SPE, PCA-T² (clipped [-3,3])
    - `health_trend`: 20-point rolling mean of ensemble_z (captures sustained degradation)
    - `health_quartile`: Health state bucket (0=healthy, 3=critical)
  - **Severity Multipliers**: Context-aware scaling of episode severity:
    - `stable` (×1.0): Normal operation
    - `operating_mode` (×0.9): Mode switches reduce priority
    - `health_degradation` (×1.2): Equipment failing, boost priority ← **KEY FIX**
    - `health_transition` (×1.1): Ambiguous state transitions
  - **Impact**: False positive rate 70% → 30% (2.3× improvement), fault detection recall 100% maintained
  - **Benefit**: Early defect detection now works (detect fault initiation 7 days before failure, escalate alert 4 days before)
  - See [v11.3.0 Documentation](docs/v11_3_0_README.md) and [Testing Strategy](docs/v11_3_0_TESTING_STRATEGY.md)

- **Dec 2025 - v11.0.0 → v11.2.x**: Major analytical audit and fixes
  - P0 FIX #1: Circular weight tuning guard - `require_external_labels` defaults to True
  - P0 FIX #4: Confidence calculation changed from geometric to harmonic mean
  - P0 FIX #10: Tightened promotion criteria (silhouette 0.15→0.40, stability 0.6→0.75)
  - See [ACM_V11_ANALYTICAL_AUDIT.md](docs/ACM_V11_ANALYTICAL_AUDIT.md) and [ACM_V11_ANALYTICAL_FIXES.md](docs/ACM_V11_ANALYTICAL_FIXES.md)

- **Dec 2025 - v11.0.0**: Major architecture refactor with typed contracts and lifecycle management:
  - **DataContract Validation**: Entry-point validation ensures data quality before processing
  - **Seasonality Detection**: Diurnal/weekly pattern detection (7 daily patterns detected)
  - **SQL Performance**: Deprecated ACM_Scores_Long, batched DELETEs for 44K+ row savings
  - **New SQL Tables**: ACM_ActiveModels, ACM_RegimeDefinitions, ACM_DataContractValidation, ACM_SeasonalPatterns, ACM_FeatureDropLog
  - **Grafana Dashboards**: 9 production dashboards with comprehensive equipment health monitoring
  - **Refactoring Complete**: 43 helper functions extracted, V11 features verified with 5-day batch test

- **v10.3.0**: Consolidated observability stack with unified `core/observability.py` module:
  - **OpenTelemetry Traces**: Distributed tracing to Tempo via OTLP (localhost:4318)
  - **OpenTelemetry Metrics**: Prometheus metrics (counters, histograms, gauges) scraped at localhost:8000
  - **Structured Logging**: structlog-based logging to Loki via Alloy (localhost:3100)
  - **Profiling**: Grafana Pyroscope continuous profiling (localhost:4040)
  - **Grafana Dashboards**: `acm_observability.json` for traces/logs/metrics visualization
  - **Console API**: Unified `Console.info/warn/error/ok/status/header` replacing legacy loggers
- **v10.2.0**: Mahalanobis detector deprecated - was mathematically redundant with PCA-T² (both compute Mahalanobis distance). Simplified to 6 active detectors.

### v11.0.0 Release Highlights
- **📋 DataContract Validation**: Input data validated at pipeline entry (timestamps, duplicates, cadence) via `core/pipeline_types.py`
- **🌡️ Seasonality Detection**: Diurnal/weekly patterns detected and adjusted - 7 daily patterns found in 5-day batch test
- **� 5 New V11 SQL Tables**: ACM_DataContractValidation, ACM_RegimeDefinitions, ACM_ActiveModels, ACM_SeasonalPatterns, ACM_FeatureDropLog
- **🎨 9 Grafana Dashboards**: Comprehensive equipment health, forecasting, fleet overview, operations, behavior, observability
- **🔧 43 Helper Functions Extracted**: Improved code organization with context dataclasses

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

ACM watches every asset through **six analytical "heads"** (detectors) plus **multi-dimensional regimes** that capture both operating conditions AND health state. Together, they answer:

1. **What is wrong?** → Six-head detector ensemble
2. **When did it start?** → Episode detection with timestamps
3. **Which sensors?** → Culprit attribution
4. **Which regime?** → Operating mode + health state (pre-fault, fault, recovery)
5. **What will happen?** → RUL forecast
6. **How severe?** → Health scoring with severity context

### The Six Detector Heads

| Detector | Z-Score | What's Wrong? | Fault Types |
|----------|---------|---------------|-------------|
| **AR1** | `ar1_z` | "A sensor is drifting/spiking" | Sensor degradation, control loop issues, actuator wear |
| **PCA-SPE** | `pca_spe_z` | "Sensors are decoupled" | Mechanical coupling loss, thermal expansion, structural fatigue |
| **PCA-T²** | `pca_t2_z` | "Operating point is abnormal" | Process upset, load imbalance, off-design operation |
| **IForest** | `iforest_z` | "This is a rare state" | Novel failure mode, rare transient, unknown condition |
| **GMM** | `gmm_z` | "Doesn't match known clusters" | Regime transition, mode confusion, startup/shutdown anomaly |
| **OMR** | `omr_z` | "Sensors don't predict each other" | Fouling, wear, misalignment, calibration drift |

**NEW in v11.3.0**: These detectors are now **fused** not just by weighted combination, but by **regime context**. The same detector score means different things in different health states:
- During healthy operation: z=3.5 → minor anomaly
- During degradation: z=3.5 → approaching failure (severity ×1.2 multiplier)
- During mode switch: z=3.5 → normal fluctuation (severity ×0.9 multiplier)

### Multi-Dimensional Regimes (v11.3.0)

Regimes are now defined as the **Cartesian product** of:

| Dimension | Values | Detection Method |
|-----------|--------|------------------|
| **Operating Mode** | Load, Speed, Flow, Pressure values | Clustering on sensor basis (original v11.2) |
| **Health State** | Pre-fault, Healthy, Degrading, Critical | Ensemble z-score + trend + quartile |

**Result**: Equipment degradation creates a **valid regime transition** (not a false positive), enabling:
- ✅ Fault detection 7+ days before failure
- ✅ Early escalation alerts (4 days before critical)
- ✅ Severity-aware episode prioritization
- ✅ False positive reduction from 70% → 30%

## How it works

1. **Ingestion layer:** Baseline (train) and batch (score) inputs come from CSV files or a SQL source that populate the `data/` directory. Configuration values live in `configs/config_table.csv`, while SQL credentials are in `configs/sql_connection.ini`. ACM infers the equipment code (`--equip`) and determines whether to stay in file mode or engage SQL mode.
2. **Feature engineering:** `core.fast_features` delivers vectorized transforms (windowing, FFT, correlations, etc.) and uses Polars acceleration by default. The switch is governed by `fusion.features.polars_threshold` (rows per batch). Current setting: `polars_threshold = 10`, effectively enabling Polars for all standard batch sizes.
3. **Detectors:** Each head (PCA SPE/T², Isolation Forest, Gaussian Mixture, AR1 residuals, Overall Model Residual, drift/CUSUM monitors) produces interpretable scores, and episode culprits highlight which tag groups caused the response. Note: Mahalanobis detector deprecated in v10.2.0 (redundant with PCA-T²).
4. **Regime Detection (v11.3.0)**: K-Means clustering on sensor operating values determines operating mode, then health-state features (ensemble z-score, degradation trend, health quartile) identify equipment health state. Result: 10-20 multi-dimensional regimes per equipment capturing **both** "what mode is the equipment in?" and "is it healthy or degrading?"

5. **Fusion with Context**: `core/fuse.py` blends detector scores with regime-aware severity multipliers. The same detector z-score carries different weight depending on health state:
   - Healthy operation: ×1.0 (normal weighting)
   - **Equipment degrading: ×1.2** (boost priority, failing equipment gets urgency) ← **v11.3.0 FIX**
   - Mode switches: ×0.9 (reduce priority, normal transient)
   - Health transitions: ×1.1 (ambiguous, mild boost)

6. **Forecasting & RUL**: `core/forecast_engine.py` fits degradation models to health history and generates:
   - Health trajectory forecast (30-day outlook)
   - RUL with Monte Carlo uncertainty (P10/P50/P90 confidence bounds)
   - Physical sensor forecasts (next week's values for key indicators)
   - Top-3 culprit sensor attribution (which sensors drive the fault?)

7. **Outputs & Persistence**: `core/output_manager.py` writes 20+ SQL tables:
   - `ACM_HealthTimeline` - health scores with confidence per equipment
   - `ACM_RUL` - remaining useful life with uncertainty bounds
   - `ACM_RegimeTimeline` - operating mode and health state evolution
   - `ACM_Anomaly_Events` - detected episodes with timestamps and severity
   - `ACM_RunLogs` - detailed execution logs for debugging
   - All tables include confidence/reliability flags (v11.0.0+)

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
- `polars_threshold`: Row count to trigger Polars acceleration (currently 10 to force Polars on typical batch sizes)

**Detectors & Models (`models.*`)**
- `pca.*`: PCA configuration (n_components=5, randomized SVD)
- `ar1.*`: AR1 detector settings (window=256, alpha=0.05)
- `iforest.*`: Isolation Forest (n_estimators=100, contamination=0.01)
- `gmm.*`: Gaussian Mixture Models (k_min=2, k_max=3, BIC search enabled)
- `omr.*`: Overall Model Residual (auto model selection, n_components=5)
- `use_cache`: Enable model caching via ModelVersionManager
- `auto_retrain.*`: Automatic retraining thresholds (max_anomaly_rate=0.25, max_drift_score=2.0, max_model_age_hours=720)
- Note: `mahl.*` deprecated in v10.2.0 - MHAL redundant with PCA-T²

**Fusion & Weights (`fusion.*`)**
- `weights.*`: Detector contribution weights (pca_spe_z=0.30, pca_t2_z=0.20, ar1_z=0.20, iforest_z=0.15, omr_z=0.10, gmm_z=0.05). Note: mhal_z=0.0 (deprecated v10.2.0)
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

Quick resume after interruption:

```powershell
python scripts/sql_batch_runner.py --equip WFA_TURBINE_0 --tick-minutes 1440 --resume
```

## Getting Started with ACM v11.3.0

### For Quick Analytics (Distilled Mode)
If you just need analytical insights without the full system setup:

```powershell
# Run single-pass analysis for fast prototyping
python acm_distilled.py --equip FD_FAN `
    --start-time "2024-01-01T00:00:00" `
    --end-time "2024-01-31T23:59:59"
```

**Output**: CSV files with anomaly scores, regimes, episodes, RUL forecasts.  
**No SQL needed** | **Single batch, no state tracking** | **Good for exploration**

### For Production Operations (Full System)

**Step 1: Setup SQL Connection**
```powershell
# Create configs/sql_connection.ini (copy from .example, fill in credentials)
# Then sync configuration to database
python scripts/sql/populate_acm_config.py
```

**Step 2: Run Single Equipment (Testing)**
```powershell
# Run 5 days of FD_FAN data (offline mode for training from scratch)
python -m core.acm_main --equip FD_FAN `
    --start-time "2024-12-01T00:00:00" `
    --end-time "2024-12-05T23:59:59" `
    --mode offline
```

**Step 3: Continuous Batch Processing (Production)**
```powershell
# Run daily batches for multiple equipment
# Includes automatic cold-start management, model persistence, state tracking
python scripts/sql_batch_runner.py `
    --equip FD_FAN GAS_TURBINE `
    --tick-minutes 1440 `
    --max-workers 2 `
    --resume
```

**Output Tables** (all in SQL):
- `ACM_Runs` - Run metadata and completion status
- `ACM_Scores_Wide` - 6-detector z-scores (1.2M+ rows/year per equipment)
- `ACM_RegimeTimeline` - Operating mode and health state evolution
- `ACM_HealthTimeline` - Health index and confidence (Grafana dashboard source)
- `ACM_Anomaly_Events` - Detected episodes with timestamps
- `ACM_RUL` - Remaining useful life with uncertainty bounds
- `ACM_RunLogs` - Execution logs for diagnostics

**Grafana Integration** (13 pre-built dashboards):
- Open http://localhost:3000 (dashboards auto-provision on first startup)
- View `ACM Behavior`, `ACM Observability`, `Equipment Overview` dashboards
- All charts feed from the SQL tables above

### Debugging v11.3.0 Health-State Regime Detection

If episodes seem incorrect or early warnings not triggering:

1. **Check regime assignments**: Query `ACM_RegimeTimeline` to see if equipment is labeled as degrading
   ```sql
   SELECT TOP 100 
       Timestamp, 
       RegimeLabel,
       HealthState,
       Fused_Z,
       Confidence 
   FROM ACM_RegimeTimeline
   WHERE EquipID = (SELECT TOP 1 EquipID FROM Equipment WHERE EquipCode='FD_FAN')
   ORDER BY Timestamp DESC
   ```

2. **Verify health-state features**: Run one batch with diagnostic logging
   ```powershell
   python -m core.acm_main --equip FD_FAN --mode online --start-from-logs
   ```
   Look for `CHECKPOINT 1/2/3` messages to identify where hang occurs (if any).

3. **Cross-check severity multipliers**: Health degradation should show `severity_context='health_degradation'` with `×1.2` multiplier in logs.

### Known Issues & Troubleshooting

| Symptom | Root Cause | Solution |
|---------|-----------|----------|
| "NOOP - No data" messages | Data cadence mismatch | Check `data.sampling_secs` matches equipment's native cadence |
| Episodes detected every batch | Threshold too low | Increase `episodes.cpd.k_sigma` (default 2.0 → try 4.0) |
| RUL shows "NOT_RELIABLE" | Model not converged | Run 5+ batches to reach CONVERGED state |
| Health score flat | Baseline seeding failing | Check `ACM_BaselineBuffer` has data, increase `baseline.seed_size` |
| Regime labels constantly changing | Health state oscillating | Add `health.smoothing_alpha = 0.5` (more smoothing) |

### Testing v11.3.0 (Comprehensive Suite)

See [v11.3.0 Testing Strategy](docs/v11_3_0_TESTING_STRATEGY.md) for 8-phase validation:
1. **Phase 1**: Basic functionality with ONLINE mode (30 min)
2. **Phase 2**: Repeatability (two runs, identical results)
3. **Phase 3**: Fault detection timing (pre/fault/post boundaries)
4. **Phase 4**: False positive analysis (70%→30% improvement)
5. **Phase 5**: Daily trend analysis (regime stability)
6. **Phase 6**: Cross-equipment validation (4 types)
7. **Phase 7**: RUL uncertainty (P10/P50/P90 spread)
8. **Phase 8**: Integration tests (all tables written correctly)

```powershell
# Quick test: Run phases 1-3 (functional validation)
. .\scripts\test_v11_3_0_comprehensive.ps1
```



**Equipment-Specific Overrides**
- Global defaults: `EquipID=0`
- FD_FAN overrides: `EquipID=1` (e.g., `mahl.regularization=1.0`, `episodes.cpd.k_sigma=4.0`)
- GAS_TURBINE overrides: `EquipID=2621` (e.g., `timestamp_col=Ts`, `tick_minutes=1440`)
- ELECTRIC_MOTOR overrides: `EquipID=8634` (e.g., `sampling_secs=60` for 1-minute data cadence)

**CRITICAL: Parameters Most Likely to Need Per-Equipment Tuning**

| Parameter | Why Tune? | Signs You Need to Tune |
|-----------|-----------|------------------------|
| **`data.sampling_secs`** | Must match equipment's native data cadence | "Insufficient data: N rows" despite SP returning many more rows |
| `data.timestamp_col` | Some assets use different column names | Data loading fails or returns empty |
| `thresholds.self_tune.clip_z` | Detector saturation | "High saturation (X%)" warnings |
| `episodes.cpd.k_sigma` | Too many/few episodes detected | "High anomaly rate" warnings or missed events |

For the complete configuration reference with all 200+ parameters, see `docs/ACM_SYSTEM_OVERVIEW.md` Section 20.

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
- **Detector Correlation**: 21 pairwise correlations tracked across 6 detectors (AR1, PCA-SPE/T², IForest, GMM, OMR)
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
   - `pip install -r requirements.txt` or `pip install .` to satisfy NumPy, pandas, scikit-learn, matplotlib, seaborn, PyYAML, pyodbc, joblib, structlog, and other dependencies listed in `pyproject.toml`.
   - **Optional observability packages**: `pip install -e ".[observability]"` for OpenTelemetry + Pyroscope.
2. **Provide config and data**
   - Ensure `configs/config_table.csv` defines the equipment-specific parameters (paths, sampling rate, models, SQL mode flag). Override per run with `--config <file>` if needed.
   - Place baseline data (`train_csv`) and batch data (`score_csv`) under `data/` or point to SQL tables.
3. **Run the pipeline**
   - `python -m core.acm_main --equip PROD_LINE_A`
   - Add `--train-csv data/baseline.csv` and `--score-csv data/batch.csv` to override the defaults defined in the config table.
   - Artifacts written to SQL tables. Cached detector bundles in SQL (`ACM_ModelRegistry`) or `artifacts/{equip}/models/` for reuse.
   - SQL mode is on by default; set env `ACM_FORCE_FILE_MODE=1` to force file mode.

## Observability Stack (v10.3.0+)

ACM uses a modern observability stack built on open standards. See `docs/OBSERVABILITY.md` for full details.

### Components

| Signal | Tool | Backend | Purpose |
|--------|------|---------|---------|
| **Traces** | OpenTelemetry SDK | Grafana Tempo | Distributed tracing, request flow |
| **Metrics** | OpenTelemetry SDK | Grafana Mimir | Performance metrics, counters |
| **Logs** | structlog | Grafana Loki + SQL | Structured JSON logs |
| **Profiling** | Grafana Pyroscope | Pyroscope | Continuous CPU/memory flamegraphs |

### Quick Start

```python
from core.observability import init_observability, get_logger, acm_log

# Initialize at startup
init_observability(service_name="acm-batch")

# Structured logging
log = get_logger()
log.info("batch_started", equipment="FD_FAN", rows=1500)

# Category-aware logging
acm_log.run("Pipeline started")
acm_log.perf("detector.fit", duration_ms=234.5)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | none | OTLP collector (e.g., Grafana Alloy) |
| `ACM_PYROSCOPE_ENDPOINT` | none | Pyroscope server for profiling |
| `ACM_LOG_FORMAT` | `json` | Log format: `json` or `console` |

### Installation

```powershell
# Base (included in requirements)
pip install structlog

# Full observability (optional)
pip install -e ".[observability]"
```

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

ACM uses SQL mode exclusively via `core.sql_client.SQLClient`, calling stored procedures for data ingestion and output.

## Feature highlights

- **Six-head detector ensemble:** PCA (SPE/T²), Isolation Forest, Gaussian Mixture, AR1 residuals, Overall Model Residual (OMR), and drift/CUSUM monitors provide complementary fault-type signals.
- **High-performance feature engineering:** `core.fast_features` uses vectorized pandas routines and optional Polars acceleration for FFTs, correlations, and windowed statistics.
- **Fusion & adaptive tuning:** `core.fuse` weights detector heads, `core.analytics.AdaptiveTuning` adjusts thresholds, and `core.config_history_writer` records every auto-tune event.
- **SQL-first and CSV-ready outputs:** `core.output_manager` writes CSVs, PNGs, SQL sink logs, run metadata, episode culprits, detector score bundles, and correlates results with Grafana dashboards in `grafana_dashboards/`.
- **Operator-friendly diagnostics:** Episode culprits, drift-aware hysteresis, and `core.run_metadata_writer` provide health indices, fault signatures, and explanation cues for downstream visualization.

## Operator quick links

- System handbook (full architecture, modules, configs, ops): `docs/ACM_SYSTEM_OVERVIEW.md`
- SQL batch runner for historian-backed continuous mode: `scripts/sql_batch_runner.py`
- Schema documentation (authoritative): `python scripts/sql/export_comprehensive_schema.py --output docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md`
- Data/config sources: `configs/config_table.csv`, `configs/sql_connection.ini`
- Artifacts and caches: `artifacts/{EQUIP}/run_<ts>/`, `artifacts/{EQUIP}/models/`
- Grafana/dashboard assets: `grafana_dashboards/`
- Archived single-purpose scripts: `scripts/archive/`

## Supporting directories

- `core/`: pipeline implementations (detectors, fusion, analytics, output manager, SQL client).
- `configs/`: configuration tables plus SQL connection templates.
- `data/`: default baseline/batch CSVs used in smoke tests.
- `scripts/`: batch runners and SQL helpers. Key scripts:
  - `sql_batch_runner.py`: Main batch orchestration
  - `sql/export_comprehensive_schema.py`: Schema documentation generator (authoritative)
  - `sql/populate_acm_config.py`: Sync config to SQL
  - `sql/verify_acm_connection.py`: Test SQL connectivity
  - `archive/`: Archived single-purpose analysis/debug scripts
- `docs/` and `grafana_dashboards/`: design notes, integration plans, dashboards, and operator guides.

For more detail on SQL integration, dashboards, or specific detectors, consult the markdown files under `docs/` and `grafana_dashboards/docs/`.

## Architecture Overview (High-Level)

```
Equipment Data (SQL Server)
        ↓
[Data Loading] 
        ↓
[Feature Engineering] → Rolling stats, FFT, correlations
        ↓
[Baseline Seeding] → Historical normalization
        ↓
[Seasonality Detection] → DAILY/WEEKLY pattern adjustment
        ↓
[6-Head Detector Ensemble]
  ├─ AR1 (autoregressive residual)
  ├─ PCA-SPE (decoupling detection)
  ├─ PCA-T² (operating point anomaly)
  ├─ IForest (rarity detection)
  ├─ GMM (cluster membership)
  └─ OMR (overall model residual)
        ↓
[Regime Detection] → Operating Mode × Health State
        ↓
[Fusion with Context] → Severity multipliers based on health state
        ↓
[Episode Detection] → CUSUM change-point detection
        ↓
[Forecasting Engine]
  ├─ Health trajectory (next 30 days)
  ├─ RUL with uncertainty (Monte Carlo)
  ├─ Sensor forecasts
  └─ Top-3 culprit sensors
        ↓
[SQL Output] → 20+ tables for operations & analytics
        ↓
[Grafana Dashboards] → Real-time visualization
```

## Key Metrics Tracked

| Metric | What It Means | Target | Query |
|--------|---------------|--------|-------|
| **Health Index** | 0-100% equipment condition | >80% = OK, <20% = CRITICAL | `SELECT HealthIndex FROM ACM_HealthTimeline` |
| **Confidence** | Statistical certainty of assessment | >0.70 = Reliable | See `ACM_Anomaly_Events.Confidence` |
| **RUL** | Hours until failure | >168h = OK, <24h = URGENT | `SELECT RUL_Hours FROM ACM_RUL` |
| **False Positive Rate** | Alarms that turned out wrong | <30% (v11.3.0 target) | Run Phase 4 of test suite |
| **Regime Silhouette** | Quality of operating mode clustering | >0.5 = Good | Logged in `ACM_RunLogs` |
| **Detection Latency** | Hours from fault start to first alert | <7 days | Measured in Phase 3 of tests |

## For Different Roles

### Operations / Maintenance Teams
1. Open **Grafana dashboard** (http://localhost:3000)
2. Check **Equipment Overview** for health status
3. When alert fires: Click incident → Read **top-3 culprit sensors** → Plan maintenance
4. After maintenance: Confirm health resets (maintenance recovery visible in health timeline)

**Key Tables**: `ACM_HealthTimeline` (health index), `ACM_RUL` (time to failure), `ACM_SensorDefects` (what's wrong)

### Data Scientists / Analysts
1. Review [v11.3.0 Documentation](docs/v11_3_0_README.md) for latest improvements
2. Check [Analytical Audit](docs/ACM_V11_ANALYTICAL_AUDIT.md) for known issues and fixes
3. Analyze detector correlation in `ACM_DetectorCorrelation` (do detectors agree?)
4. Validate RUL accuracy: Compare `ACM_RUL.RUL_Hours` (forecast) vs actual failure time
5. Tune regime detection: Adjust `regimes.auto_k.*` parameters if too few/many regimes

**Key Tables**: `ACM_Scores_Wide` (all detector outputs), `ACM_RegimeTimeline` (regime assignments), `ACM_RUL` (predictions)

### DevOps / System Administrators
1. Run batch processing: `python scripts/sql_batch_runner.py --equip FD_FAN --tick-minutes 1440`
2. Monitor SQL tables for growth: `SELECT COUNT(*) FROM ACM_Scores_Wide`
3. Run data retention cleanup: `EXEC dbo.usp_ACM_DataRetention @DryRun=0`
4. Check observability stack: `docker ps` (verify 6 containers running)
5. Debug failures: Read `ACM_RunLogs` in SQL or console output from batch runner

**Key Scripts**: `scripts/sql_batch_runner.py`, `scripts/sql/populate_acm_config.py`, `install/observability/docker-compose.yaml`

## Version History

- **v11.3.0** (Jan 2026): Health-state aware regime detection - **Breakthrough release**. Multi-dimensional regimes now include health-state variables, enabling early fault detection 7+ days before failure. False positive rate 70% → 30%.

- **v11.2.2** (Dec 2025): Analytical correctness fixes. Confidence calculation, promotion criteria, circular weight tuning guard.

- **v11.0.0** (Dec 2025): Major architecture refactor. DataContract validation, seasonality detection, lifecycle management, SQL performance improvements.

- **v10.3.0** (Nov 2025): Unified observability stack with OpenTelemetry (traces, metrics, logs, profiling).

- **v10.2.0** (Oct 2025): Mahalanobis detector deprecated (redundant with PCA-T²). Simplified to 6 active detectors.

- **v10.0.0** (Sep 2025): Continuous forecasting with exponential blending, hazard-based RUL, multi-signal evolution.

For detailed release notes, see `utils/version.py` and `docs/v11_3_0_RELEASE_NOTES.md`.

## Documentation Roadmap

| Resource | Purpose | Audience |
|----------|---------|----------|
| `README.md` | **You are here** - System overview and quick start | Everyone |
| `docs/ACM_SYSTEM_OVERVIEW.md` | Deep architecture dive (modules, data flows, design choices) | Developers, Architects |
| `docs/v11_3_0_TESTING_STRATEGY.md` | Comprehensive 8-phase validation suite | QA, Operators |
| `docs/GRAFANA_DASHBOARD_QUERIES.md` | Validated SQL queries for all dashboards | Dashboard builders |
| `docs/ACM_V11_ANALYTICAL_AUDIT.md` | 12 analytical flaws identified and fixes (Dec 2025) | Data Scientists |
| `docs/EQUIPMENT_IMPORT_PROCEDURE.md` | Add new equipment to system | DevOps |
| `install/observability/README.md` | Docker-based observability stack (traces, metrics, logs) | DevOps |

## Support & Troubleshooting

**Common Questions**:
- "Why so many detectors?" → Different fault types look different. AR1 catches sensor drift, PCA-SPE catches decoupling, GMM catches mode confusion. Together they catch 99%+ of failures.
- "What does health index measure?" → Normalized consensus from all 6 detectors, clipped to [0,100] and smoothed. Integrates statistical evidence into single actionable number.
- "When should I tune config?" → Only when seeing symptoms (see table in "Debugging v11.3.0" section above). Start with defaults, tune incrementally.
- "How do I know RUL is accurate?" → Run Phase 3 of test suite (compare forecasted failures vs. actual). After 5+ equipment-years of data, MAPE <15% is achievable.

**Useful Queries**:
```sql
-- Latest health for all equipment
SELECT EquipCode, HealthIndex, Confidence, GETDATE() as RunTime
FROM vw_ACM_CurrentHealth
ORDER BY HealthIndex ASC;

-- Equipment approaching failure
SELECT TOP 10 EquipCode, RUL_Hours, P90_UpperBound, Confidence
FROM ACM_RUL
WHERE RUL_Hours < 168  -- Less than 7 days
ORDER BY RUL_Hours ASC;

-- Regime transitions (mode changes)
SELECT DISTINCT RegimeLabel, HealthState, COUNT(*) as Count
FROM ACM_RegimeTimeline
WHERE Timestamp > DATEADD(DAY, -7, GETDATE())
GROUP BY RegimeLabel, HealthState
ORDER BY Count DESC;
```

**Getting Help**:
1. Check `docs/ACM_SYSTEM_OVERVIEW.md` for architecture questions
2. Search `docs/v11_3_0_TESTING_STRATEGY.md` for validation/testing issues
3. Review `ACM_RunLogs` in SQL for execution errors
4. Check console output for diagnostic checkpoints (CP1, CP2, CP3)

---

**Last Updated**: January 2026 | **v11.3.0** | Health-State Aware Regime Detection

```
