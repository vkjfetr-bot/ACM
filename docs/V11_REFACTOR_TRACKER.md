# ACM v11.0.0 Refactor Task Tracker

**Created**: 2025-12-22  
**Target Version**: 11.0.0  
**Branch**: `feature/v11-refactor`  
**Status**: Planning

---

## Milestone Overview

| Phase | Name | Items | Status | Progress |
|-------|------|-------|--------|----------|
| 0 | Setup & Versioning | 3 | ⏳ Not Started | 0/3 |
| 1 | Core Architecture | 9 | ⏳ Not Started | 0/9 |
| 2 | Regime System | 12 | ⏳ Not Started | 0/12 |
| 3 | Detector/Fusion | 6 | ⏳ Not Started | 0/6 |
| 4 | Health/Episode/RUL | 6 | ⏳ Not Started | 0/6 |
| 5 | Operational Infrastructure | 14 | ⏳ Not Started | 0/14 |
| **Total** | | **50** | | **0/50** |

---

## Phase 0: Setup & Versioning

### P0.1 — Branch Setup
- [ ] Merge `feature/profiling-and-observability` → `main`
- [ ] Tag release as `v10.3.0`
- [ ] Create `feature/v11-refactor` branch from `main`

### P0.2 — Version Bump
- [ ] Update `utils/version.py` to `11.0.0`
- [ ] Update CHANGELOG with v11.0.0 section
- [ ] Update `.github/copilot-instructions.md` with v11 contracts

### P0.3 — Documentation Foundation
- [ ] Create `docs/V11_ARCHITECTURE.md` with new system design
- [ ] Create `docs/V11_MIGRATION_GUIDE.md` for breaking changes
- [ ] Update `docs/ACM_SYSTEM_OVERVIEW.md` with v11 concepts

---

## Phase 1: Core Architecture

**Goal**: Split pipeline into ONLINE/OFFLINE modes, establish data contracts, standardize feature matrix

### P1.1 — Pipeline Mode Split (Item 1)
| Task | File | Status |
|------|------|--------|
| [ ] Create `PipelineMode` enum (ONLINE, OFFLINE) | `core/pipeline_modes.py` | ⏳ |
| [ ] Create `PipelineContext` dataclass | `core/pipeline_modes.py` | ⏳ |
| [ ] Define stage enum (LOAD, PREPROCESS, DETECT, FUSE, HEALTH, FORECAST, PERSIST) | `core/pipeline_modes.py` | ⏳ |
| [ ] Refactor `run_pipeline()` to accept mode parameter | `core/acm_main.py` | ⏳ |
| [ ] Create `OnlinePipeline` class (assignment-only) | `core/acm_main.py` | ⏳ |
| [ ] Create `OfflinePipeline` class (discovery-only) | `core/acm_main.py` | ⏳ |
| [ ] Ensure no regime discovery in ONLINE mode | `core/acm_main.py` | ⏳ |

### P1.2 — Data Contract Gate (Item 9)
| Task | File | Status |
|------|------|--------|
| [ ] Create `DataContract` class | `core/data_contract.py` | ⏳ |
| [ ] Implement timestamp order validation | `core/data_contract.py` | ⏳ |
| [ ] Implement duplicate detection | `core/data_contract.py` | ⏳ |
| [ ] Implement cadence validation | `core/data_contract.py` | ⏳ |
| [ ] Implement future row rejection | `core/data_contract.py` | ⏳ |
| [ ] Add `ContractViolation` exception | `core/data_contract.py` | ⏳ |
| [ ] Integrate gate at pipeline entry | `core/acm_main.py` | ⏳ |

### P1.3 — Sensor Validity Checks (Item 28)
| Task | File | Status |
|------|------|--------|
| [ ] Create `SensorValidator` class | `core/data_contract.py` | ⏳ |
| [ ] Implement range plausibility checks | `core/data_contract.py` | ⏳ |
| [ ] Implement stuck-value detection | `core/data_contract.py` | ⏳ |
| [ ] Create `ACM_SensorValidity` table schema | `scripts/sql/migrations/` | ⏳ |
| [ ] Persist sensor validity mask per run | `core/output_manager.py` | ⏳ |

### P1.4 — Maintenance Event Handling (Item 29)
| Task | File | Status |
|------|------|--------|
| [ ] Create `MaintenanceEventHandler` | `core/data_contract.py` | ⏳ |
| [ ] Detect recalibration signatures | `core/data_contract.py` | ⏳ |
| [ ] Implement baseline segmentation on events | `core/data_contract.py` | ⏳ |
| [ ] Create `ACM_MaintenanceEvents` table schema | `scripts/sql/migrations/` | ⏳ |

### P1.5 — Pipeline Stage Instrumentation (Item 18)
| Task | File | Status |
|------|------|--------|
| [ ] Add `StageTimer` context manager | `core/pipeline_modes.py` | ⏳ |
| [ ] Emit per-stage timing via `Metrics.time()` | `core/acm_main.py` | ⏳ |
| [ ] Emit per-stage row counts | `core/acm_main.py` | ⏳ |
| [ ] Emit per-stage feature counts | `core/acm_main.py` | ⏳ |
| [ ] Create `ACM_PipelineMetrics` table | `scripts/sql/migrations/` | ⏳ |

### P1.6 — Standardized Feature Matrix (Item 19)
| Task | File | Status |
|------|------|--------|
| [ ] Create `FeatureMatrix` class | `core/feature_matrix.py` | ⏳ |
| [ ] Define canonical column schema | `core/feature_matrix.py` | ⏳ |
| [ ] Refactor `fast_features.py` to produce `FeatureMatrix` | `core/fast_features.py` | ⏳ |
| [ ] Create `ACM_FeatureMatrix` table schema | `scripts/sql/migrations/` | ⏳ |
| [ ] Update all detectors to consume `FeatureMatrix` | `core/*.py` | ⏳ |

### P1.7 — SQL-Only Persistence (Item 34)
| Task | File | Status |
|------|------|--------|
| [ ] Audit all file-based artifact paths | `core/*.py` | ⏳ |
| [ ] Remove CSV/PNG artifact writes | `core/output_manager.py` | ⏳ |
| [ ] Migrate model persistence to SQL-only | `core/model_persistence.py` | ⏳ |
| [ ] Update `ALLOWED_TABLES` with new tables | `core/output_manager.py` | ⏳ |

### P1.8 — Hardened OutputManager (Item 35)
| Task | File | Status |
|------|------|--------|
| [ ] Add strict schema guards | `core/output_manager.py` | ⏳ |
| [ ] Add mandatory version keys | `core/output_manager.py` | ⏳ |
| [ ] Add column type validation | `core/output_manager.py` | ⏳ |
| [ ] Add NOT NULL enforcement | `core/output_manager.py` | ⏳ |

### P1.9 — Idempotent SQL Writes (Item 47)
| Task | File | Status |
|------|------|--------|
| [ ] Convert INSERT to MERGE statements | `core/output_manager.py` | ⏳ |
| [ ] Add run-completeness checks | `core/output_manager.py` | ⏳ |
| [ ] Implement transaction batching | `core/output_manager.py` | ⏳ |
| [ ] Add duplicate prevention logic | `core/output_manager.py` | ⏳ |

---

## Phase 2: Regime System Overhaul

**Goal**: Implement versioned regime model management with maturity states and offline discovery

### P2.1 — ACM_ActiveModels Pointer (Item 2)
| Task | File | Status |
|------|------|--------|
| [ ] Create `ACM_ActiveModels` table schema | `scripts/sql/migrations/` | ⏳ |
| [ ] Create `ActiveModelsManager` class | `core/regime_manager.py` | ⏳ |
| [ ] Force all regime reads through pointer | `core/regime_manager.py` | ⏳ |
| [ ] Force all threshold reads through pointer | `core/regime_manager.py` | ⏳ |
| [ ] Force all forecasting reads through pointer | `core/regime_manager.py` | ⏳ |

### P2.2 — Cold Start Handling (Item 3)
| Task | File | Status |
|------|------|--------|
| [ ] Define `ActiveRegimeVersion = NULL` as cold-start | `core/regime_manager.py` | ⏳ |
| [ ] Disable all regime-aware logic when NULL | `core/regime_manager.py` | ⏳ |
| [ ] Add `is_cold_start()` method | `core/regime_manager.py` | ⏳ |
| [ ] Update pipeline to check cold-start state | `core/acm_main.py` | ⏳ |

### P2.3 — UNKNOWN/EMERGING Regime (Item 4)
| Task | File | Status |
|------|------|--------|
| [ ] Allow `RegimeLabel = -1` for UNKNOWN | `core/regimes.py` | ⏳ |
| [ ] Allow `RegimeLabel = -2` for EMERGING | `core/regimes.py` | ⏳ |
| [ ] Remove forced nearest-regime assignment | `core/regimes.py` | ⏳ |
| [ ] Update downstream logic for unknown regimes | `core/*.py` | ⏳ |

### P2.4 — Clean Regime Discovery Inputs (Item 5)
| Task | File | Status |
|------|------|--------|
| [ ] Remove anomaly scores from regime inputs | `core/regimes.py` | ⏳ |
| [ ] Remove health indices from regime inputs | `core/regimes.py` | ⏳ |
| [ ] Remove residuals from regime inputs | `core/regimes.py` | ⏳ |
| [ ] Remove detector outputs from regime inputs | `core/regimes.py` | ⏳ |
| [ ] Document clean input requirements | `docs/V11_ARCHITECTURE.md` | ⏳ |

### P2.5 — ACM_RegimeDefinitions Table (Item 11)
| Task | File | Status |
|------|------|--------|
| [ ] Create `ACM_RegimeDefinitions` table schema | `scripts/sql/migrations/` | ⏳ |
| [ ] Implement write-once semantics (immutable) | `core/regime_definitions.py` | ⏳ |
| [ ] Store centroids, boundaries, labels | `core/regime_definitions.py` | ⏳ |
| [ ] Store transition matrix | `core/regime_definitions.py` | ⏳ |
| [ ] Add version column with auto-increment | `scripts/sql/migrations/` | ⏳ |

### P2.6 — RegimeVersion on All Writes (Item 12)
| Task | File | Status |
|------|------|--------|
| [ ] Add `RegimeVersion` column to `ACM_RegimeTimeline` | `scripts/sql/migrations/` | ⏳ |
| [ ] Add `AssignmentConfidence` column | `scripts/sql/migrations/` | ⏳ |
| [ ] Update all regime timeline writes | `core/regimes.py` | ⏳ |
| [ ] Update all regime timeline queries | `core/*.py` | ⏳ |

### P2.7 — MaturityState Gating (Item 13)
| Task | File | Status |
|------|------|--------|
| [ ] Create `MaturityState` enum (INITIALIZING, LEARNING, CONVERGED, DEPRECATED) | `core/regime_manager.py` | ⏳ |
| [ ] Gate regime-conditioned thresholds on CONVERGED | `core/adaptive_thresholds.py` | ⏳ |
| [ ] Gate regime-conditioned forecasting on CONVERGED | `core/forecast_engine.py` | ⏳ |
| [ ] Add maturity state to `ACM_ActiveModels` | `scripts/sql/migrations/` | ⏳ |

### P2.8 — Offline Historical Replay (Item 14)
| Task | File | Status |
|------|------|--------|
| [ ] Create `scripts/offline_replay.py` runner | `scripts/offline_replay.py` | ⏳ |
| [ ] Load accumulated history from SQL | `scripts/offline_replay.py` | ⏳ |
| [ ] Run regime discovery on full history | `scripts/offline_replay.py` | ⏳ |
| [ ] Write new `RegimeVersion` without affecting production | `scripts/offline_replay.py` | ⏳ |
| [ ] Add CLI arguments for date range, equipment | `scripts/offline_replay.py` | ⏳ |

### P2.9 — Regime Evaluation Metrics (Item 15)
| Task | File | Status |
|------|------|--------|
| [ ] Create `RegimeEvaluator` class | `core/regime_evaluation.py` | ⏳ |
| [ ] Implement stability metric | `core/regime_evaluation.py` | ⏳ |
| [ ] Implement novelty rate metric | `core/regime_evaluation.py` | ⏳ |
| [ ] Implement overlap entropy metric | `core/regime_evaluation.py` | ⏳ |
| [ ] Implement transition entropy metric | `core/regime_evaluation.py` | ⏳ |
| [ ] Implement consistency metric | `core/regime_evaluation.py` | ⏳ |
| [ ] Create `ACM_RegimeMetrics` table | `scripts/sql/migrations/` | ⏳ |

### P2.10 — Promotion Procedure (Item 16)
| Task | File | Status |
|------|------|--------|
| [ ] Create `RegimePromoter` class | `core/regime_promotion.py` | ⏳ |
| [ ] Define acceptance criteria | `core/regime_promotion.py` | ⏳ |
| [ ] Implement promotion from LEARNING → CONVERGED | `core/regime_promotion.py` | ⏳ |
| [ ] Update `ACM_ActiveModels` on promotion | `core/regime_promotion.py` | ⏳ |
| [ ] Create `ACM_RegimePromotionLog` audit table | `scripts/sql/migrations/` | ⏳ |

### P2.11 — Confidence-Gated Normalization (Item 17)
| Task | File | Status |
|------|------|--------|
| [ ] Add `AssignmentConfidence` threshold check | `core/fast_features.py` | ⏳ |
| [ ] Condition anomaly normalization on confidence | `core/fast_features.py` | ⏳ |
| [ ] Condition thresholds on regime confidence | `core/adaptive_thresholds.py` | ⏳ |
| [ ] Fall back to global normalization when low confidence | `core/fast_features.py` | ⏳ |

### P2.12 — Replay Reproducibility (Item 26)
| Task | File | Status |
|------|------|--------|
| [ ] Add hash-based input validation | `core/regime_manager.py` | ⏳ |
| [ ] Verify identical inputs + params → identical assignments | `core/regime_manager.py` | ⏳ |
| [ ] Create reproducibility test suite | `tests/test_reproducibility.py` | ⏳ |

---

## Phase 3: Detector/Fusion Refactor

**Goal**: Standardize detector API, enforce train-score separation, calibrate fusion

### P3.1 — Train-Score Separation (Item 8)
| Task | File | Status |
|------|------|--------|
| [ ] Define separation contract (batch cannot influence own score) | `docs/V11_ARCHITECTURE.md` | ⏳ |
| [ ] Audit AR1 detector for separation | `core/ar1_detector.py` | ⏳ |
| [ ] Audit PCA detector for separation | `core/outliers.py` | ⏳ |
| [ ] Audit IForest detector for separation | `core/outliers.py` | ⏳ |
| [ ] Audit GMM detector for separation | `core/outliers.py` | ⏳ |
| [ ] Audit OMR detector for separation | `core/omr.py` | ⏳ |
| [ ] Add separation validation in tests | `tests/test_detectors.py` | ⏳ |

### P3.2 — Unified Baseline Normalization (Item 20)
| Task | File | Status |
|------|------|--------|
| [ ] Create `BaselineNormalizer` class | `core/baseline_normalizer.py` | ⏳ |
| [ ] Remove detector-specific normalization | `core/ar1_detector.py` | ⏳ |
| [ ] Remove detector-specific normalization | `core/outliers.py` | ⏳ |
| [ ] Integrate normalizer into pipeline | `core/acm_main.py` | ⏳ |

### P3.3 — Strict Detector Protocol (Item 21)
| Task | File | Status |
|------|------|--------|
| [ ] Create `DetectorProtocol` ABC | `core/detector_protocol.py` | ⏳ |
| [ ] Define `fit_baseline(X_train)` method | `core/detector_protocol.py` | ⏳ |
| [ ] Define `score(X_score) -> DataFrame` method | `core/detector_protocol.py` | ⏳ |
| [ ] Define output schema (z_score, raw_score, etc.) | `core/detector_protocol.py` | ⏳ |
| [ ] Refactor AR1 to implement protocol | `core/ar1_detector.py` | ⏳ |
| [ ] Refactor PCA to implement protocol | `core/outliers.py` | ⏳ |
| [ ] Refactor IForest to implement protocol | `core/outliers.py` | ⏳ |
| [ ] Refactor GMM to implement protocol | `core/outliers.py` | ⏳ |
| [ ] Refactor OMR to implement protocol | `core/omr.py` | ⏳ |

### P3.4 — Calibrated Fusion (Item 22)
| Task | File | Status |
|------|------|--------|
| [ ] Redesign fusion as calibrated evidence combiner | `core/fuse.py` | ⏳ |
| [ ] Add explicit missingness handling (NaN → confidence dampening) | `core/fuse.py` | ⏳ |
| [ ] Add detector weight calibration | `core/fuse.py` | ⏳ |
| [ ] Add disagreement penalty | `core/fuse.py` | ⏳ |

### P3.5 — Per-Run Fusion Quality (Item 23)
| Task | File | Status |
|------|------|--------|
| [ ] Create `FusionQualityMetrics` class | `core/fuse.py` | ⏳ |
| [ ] Track which detectors contributed | `core/fuse.py` | ⏳ |
| [ ] Track detector agreement level | `core/fuse.py` | ⏳ |
| [ ] Track confidence impact | `core/fuse.py` | ⏳ |
| [ ] Create `ACM_FusionQuality` table | `scripts/sql/migrations/` | ⏳ |
| [ ] Persist fusion quality per run | `core/output_manager.py` | ⏳ |

### P3.6 — Detector Correlation Tracking (Item 33)
| Task | File | Status |
|------|------|--------|
| [ ] Create `DetectorCorrelation` class | `core/detector_correlation.py` | ⏳ |
| [ ] Track pairwise correlations per run | `core/detector_correlation.py` | ⏳ |
| [ ] Flag redundant detectors (correlation > 0.95) | `core/detector_correlation.py` | ⏳ |
| [ ] Flag unstable detectors (high variance) | `core/detector_correlation.py` | ⏳ |
| [ ] Create `ACM_DetectorCorrelation` table | `scripts/sql/migrations/` | ⏳ |

---

## Phase 4: Health/Episode/RUL Redesign

**Goal**: Make episodes the only alerting primitive, redefine health as time-evolving state

### P4.1 — Episode-Only Alerting (Item 6)
| Task | File | Status |
|------|------|--------|
| [ ] Create `EpisodeManager` class | `core/episode_manager.py` | ⏳ |
| [ ] Make episode construction the only alerting primitive | `core/episode_manager.py` | ⏳ |
| [ ] Remove point-anomaly-driven alerts | `core/acm_main.py` | ⏳ |
| [ ] Refactor episode culprits writer | `core/episode_culprits_writer.py` | ⏳ |

### P4.2 — Time-Evolving Health State (Item 24)
| Task | File | Status |
|------|------|--------|
| [ ] Redefine health as time-evolving state | `core/health_tracker.py` | ⏳ |
| [ ] Add `HealthConfidence` field | `core/health_tracker.py` | ⏳ |
| [ ] Add state persistence across runs | `core/health_tracker.py` | ⏳ |
| [ ] Add `HealthState` enum (HEALTHY, DEGRADED, CRITICAL, UNKNOWN) | `core/health_tracker.py` | ⏳ |

### P4.3 — Recovery Logic (Item 25)
| Task | File | Status |
|------|------|--------|
| [ ] Implement hysteresis for state transitions | `core/health_tracker.py` | ⏳ |
| [ ] Implement cooldown after critical state | `core/health_tracker.py` | ⏳ |
| [ ] Implement exponential decay for recovery | `core/health_tracker.py` | ⏳ |
| [ ] Add configurable thresholds | `configs/config_table.csv` | ⏳ |

### P4.4 — RUL Reliability Gate (Item 7)
| Task | File | Status |
|------|------|--------|
| [ ] Add `RUL_NOT_RELIABLE` outcome | `core/rul_estimator.py` | ⏳ |
| [ ] Define prerequisite checks | `core/rul_estimator.py` | ⏳ |
| [ ] Prevent numeric RUL when prerequisites fail | `core/rul_estimator.py` | ⏳ |
| [ ] Add `RULStatus` enum (RELIABLE, NOT_RELIABLE, INSUFFICIENT_DATA) | `core/rul_estimator.py` | ⏳ |
| [ ] Update SQL writes to include status | `core/output_manager.py` | ⏳ |

### P4.5 — Forecasting Diagnostics (Item 36)
| Task | File | Status |
|------|------|--------|
| [ ] Create `ForecastDiagnostics` class | `core/forecast_diagnostics.py` | ⏳ |
| [ ] Implement coverage metric | `core/forecast_diagnostics.py` | ⏳ |
| [ ] Implement sharpness metric | `core/forecast_diagnostics.py` | ⏳ |
| [ ] Implement calibration metric | `core/forecast_diagnostics.py` | ⏳ |
| [ ] Create `ACM_ForecastDiagnostics` table | `scripts/sql/migrations/` | ⏳ |
| [ ] Persist diagnostics on every run | `core/output_manager.py` | ⏳ |

### P4.6 — Unified Confidence Model (Item 49)
| Task | File | Status |
|------|------|--------|
| [ ] Create `ConfidenceModel` class | `core/confidence_model.py` | ⏳ |
| [ ] Combine regime confidence | `core/confidence_model.py` | ⏳ |
| [ ] Combine detector agreement | `core/confidence_model.py` | ⏳ |
| [ ] Combine data quality signal | `core/confidence_model.py` | ⏳ |
| [ ] Apply to health outputs | `core/health_tracker.py` | ⏳ |
| [ ] Apply to episode outputs | `core/episode_manager.py` | ⏳ |
| [ ] Apply to RUL outputs | `core/rul_estimator.py` | ⏳ |

---

## Phase 5: Operational Infrastructure

**Goal**: Drift/novelty control plane, feedback loops, operational contracts

### P5.1 — Drift/Novelty Control Plane (Item 10)
| Task | File | Status |
|------|------|--------|
| [ ] Create `DriftController` class | `core/drift_controller.py` | ⏳ |
| [ ] Promote drift signals to control-plane triggers | `core/drift_controller.py` | ⏳ |
| [ ] Promote novelty signals to control-plane triggers | `core/drift_controller.py` | ⏳ |
| [ ] Trigger offline replay when thresholds exceeded | `core/drift_controller.py` | ⏳ |

### P5.2 — Novelty Pressure Tracking (Item 30)
| Task | File | Status |
|------|------|--------|
| [ ] Track novelty pressure independent of regimes | `core/drift_controller.py` | ⏳ |
| [ ] Create `ACM_NoveltyPressure` table | `scripts/sql/migrations/` | ⏳ |
| [ ] Add novelty pressure to run metadata | `core/run_metadata_writer.py` | ⏳ |

### P5.3 — Drift Events as Objects (Item 31)
| Task | File | Status |
|------|------|--------|
| [ ] Create `DriftEvent` class | `core/drift_controller.py` | ⏳ |
| [ ] Persist drift events to SQL | `core/output_manager.py` | ⏳ |
| [ ] Down-weight confidence when drift detected | `core/confidence_model.py` | ⏳ |
| [ ] Create `ACM_DriftEvents` table | `scripts/sql/migrations/` | ⏳ |

### P5.4 — Unified Sensor Attribution (Item 32)
| Task | File | Status |
|------|------|--------|
| [ ] Refactor sensor attribution to use frozen normalized artifacts | `core/sensor_attribution.py` | ⏳ |
| [ ] Unify attribution across all modules | `core/sensor_attribution.py` | ⏳ |
| [ ] Add attribution to episode explanation | `core/episode_manager.py` | ⏳ |

### P5.5 — Baseline Window Policy (Item 27)
| Task | File | Status |
|------|------|--------|
| [ ] Create `BaselinePolicy` class | `core/baseline_policy.py` | ⏳ |
| [ ] Define per-equipment baseline window requirements | `core/baseline_policy.py` | ⏳ |
| [ ] Persist policy per run | `core/output_manager.py` | ⏳ |
| [ ] Create `ACM_BaselinePolicy` table | `scripts/sql/migrations/` | ⏳ |

### P5.6 — Regression Harness (Item 37)
| Task | File | Status |
|------|------|--------|
| [ ] Create regression harness | `tests/regression_harness.py` | ⏳ |
| [ ] Define golden datasets | `tests/golden_data/` | ⏳ |
| [ ] Compare before/after behavior | `tests/regression_harness.py` | ⏳ |
| [ ] Detect unintended behavioral changes | `tests/regression_harness.py` | ⏳ |

### P5.7 — Truth Dashboard (Item 38)
| Task | File | Status |
|------|------|--------|
| [ ] Create `acm_truth.json` dashboard | `grafana_dashboards/acm_truth.json` | ⏳ |
| [ ] Expose data quality invariants | `grafana_dashboards/acm_truth.json` | ⏳ |
| [ ] Expose drift status | `grafana_dashboards/acm_truth.json` | ⏳ |
| [ ] Expose novelty pressure | `grafana_dashboards/acm_truth.json` | ⏳ |
| [ ] Expose fusion health | `grafana_dashboards/acm_truth.json` | ⏳ |

### P5.8 — Operational Decision Contract (Item 39)
| Task | File | Status |
|------|------|--------|
| [ ] Create `DecisionContract` dataclass | `core/decision_policy.py` | ⏳ |
| [ ] Define State, Confidence, Action, RULStatus fields | `core/decision_policy.py` | ⏳ |
| [ ] Create compact output format | `core/decision_policy.py` | ⏳ |
| [ ] Create `ACM_DecisionOutput` table | `scripts/sql/migrations/` | ⏳ |

### P5.9 — Seasonality Handling (Item 40)
| Task | File | Status |
|------|------|--------|
| [ ] Create `SeasonalityHandler` class | `core/seasonality.py` | ⏳ |
| [ ] Detect diurnal patterns | `core/seasonality.py` | ⏳ |
| [ ] Detect day-of-week patterns | `core/seasonality.py` | ⏳ |
| [ ] Adjust baselines for seasonality | `core/seasonality.py` | ⏳ |

### P5.10 — Asset Similarity Priors (Item 41)
| Task | File | Status |
|------|------|--------|
| [ ] Create `AssetSimilarity` class | `core/asset_similarity.py` | ⏳ |
| [ ] Define similarity metrics | `core/asset_similarity.py` | ⏳ |
| [ ] Implement transfer learning for cold start | `core/asset_similarity.py` | ⏳ |
| [ ] Add full auditability | `core/asset_similarity.py` | ⏳ |

### P5.11 — Operator Feedback (Item 42)
| Task | File | Status |
|------|------|--------|
| [ ] Create `OperatorFeedback` class | `core/operator_feedback.py` | ⏳ |
| [ ] Capture false alarm feedback | `core/operator_feedback.py` | ⏳ |
| [ ] Capture valid alarm feedback | `core/operator_feedback.py` | ⏳ |
| [ ] Capture maintenance feedback | `core/operator_feedback.py` | ⏳ |
| [ ] Create `ACM_OperatorFeedback` table | `scripts/sql/migrations/` | ⏳ |

### P5.12 — Alert Fatigue Controls (Item 43)
| Task | File | Status |
|------|------|--------|
| [ ] Create `AlertFatigueController` class | `core/alert_fatigue.py` | ⏳ |
| [ ] Implement rate limits | `core/alert_fatigue.py` | ⏳ |
| [ ] Implement escalation ladders | `core/alert_fatigue.py` | ⏳ |
| [ ] Implement suppression logging | `core/alert_fatigue.py` | ⏳ |

### P5.13 — Episode Clustering (Item 44)
| Task | File | Status |
|------|------|--------|
| [ ] Create `EpisodeClusterer` class | `core/episode_clustering.py` | ⏳ |
| [ ] Cluster episodes into recurring families | `core/episode_clustering.py` | ⏳ |
| [ ] Enable pattern mining | `core/episode_clustering.py` | ⏳ |
| [ ] Enable proto-RCA | `core/episode_clustering.py` | ⏳ |
| [ ] Create `ACM_EpisodeFamilies` table | `scripts/sql/migrations/` | ⏳ |

### P5.14 — Failure Mode Unknown Semantics (Item 45)
| Task | File | Status |
|------|------|--------|
| [ ] Add explicit "failure mode unknown" outcome | `core/episode_manager.py` | ⏳ |
| [ ] Prevent implied fault labels | `core/episode_manager.py` | ⏳ |
| [ ] Add `FailureMode` enum (KNOWN, UNKNOWN, EMERGING) | `core/episode_manager.py` | ⏳ |

### P5.15 — Configuration/Version Management (Item 46)
| Task | File | Status |
|------|------|--------|
| [ ] Create experiment tracking system | `core/experiment_manager.py` | ⏳ |
| [ ] Version all analytics configurations | `core/experiment_manager.py` | ⏳ |
| [ ] Create `ACM_ExperimentLog` table | `scripts/sql/migrations/` | ⏳ |

### P5.16 — Model Deprecation Workflow (Item 48)
| Task | File | Status |
|------|------|--------|
| [ ] Create `ModelDeprecator` class | `core/model_persistence.py` | ⏳ |
| [ ] Implement formal deprecation workflow | `core/model_persistence.py` | ⏳ |
| [ ] Enable forensic comparison | `core/model_persistence.py` | ⏳ |
| [ ] Create `ACM_ModelDeprecationLog` table | `scripts/sql/migrations/` | ⏳ |

### P5.17 — Separate Analytics from Decision Policy (Item 50)
| Task | File | Status |
|------|------|--------|
| [ ] Create `DecisionPolicy` class | `core/decision_policy.py` | ⏳ |
| [ ] Separate analytics outputs from operational behavior | `core/decision_policy.py` | ⏳ |
| [ ] Allow policy changes without model re-training | `core/decision_policy.py` | ⏳ |
| [ ] Create policy versioning | `core/decision_policy.py` | ⏳ |

---

## New SQL Tables Summary

| Table | Phase | Purpose |
|-------|-------|---------|
| `ACM_SensorValidity` | 1 | Sensor validity mask per run |
| `ACM_MaintenanceEvents` | 1 | Detected maintenance/recalibration events |
| `ACM_PipelineMetrics` | 1 | Per-stage timing and row counts |
| `ACM_FeatureMatrix` | 1 | Canonical standardized features |
| `ACM_ActiveModels` | 2 | Version pointer for active models |
| `ACM_RegimeDefinitions` | 2 | Versioned, immutable regime models |
| `ACM_RegimeMetrics` | 2 | Regime evaluation metrics |
| `ACM_RegimePromotionLog` | 2 | Promotion audit trail |
| `ACM_FusionQuality` | 3 | Per-run fusion diagnostics |
| `ACM_DetectorCorrelation` | 3 | Detector redundancy tracking |
| `ACM_ForecastDiagnostics` | 4 | Forecasting quality metrics |
| `ACM_NoveltyPressure` | 5 | Novelty pressure tracking |
| `ACM_DriftEvents` | 5 | Drift events as objects |
| `ACM_BaselinePolicy` | 5 | Per-equipment baseline window policy |
| `ACM_DecisionOutput` | 5 | Compact operational output |
| `ACM_OperatorFeedback` | 5 | Operator feedback capture |
| `ACM_EpisodeFamilies` | 5 | Clustered episode patterns |
| `ACM_ExperimentLog` | 5 | Experiment/configuration tracking |
| `ACM_ModelDeprecationLog` | 5 | Model deprecation audit |

---

## Progress Log

| Date | Phase | Task | Status | Notes |
|------|-------|------|--------|-------|
| 2025-12-22 | 0 | Initial planning | ✅ Done | Tracker created |
| | | | | |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking model serialization | High | Add version-gated loading in ModelRegistry |
| SQL schema migrations fail | Medium | Test migrations on copy of production DB first |
| Regression in anomaly detection | High | Golden dataset regression tests |
| Performance degradation | Medium | Benchmark each phase before/after |

---

## Definition of Done

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Regression harness shows no unintended changes
- [ ] SQL schema migrations tested
- [ ] Documentation updated
- [ ] Grafana dashboards updated
- [ ] CHANGELOG updated
- [ ] Version bumped appropriately
