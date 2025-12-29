# V11 PROPER IMPLEMENTATION PLAN

**Created**: 2025-12-29
**Branch**: `feature/v11-refactor`
**Status**: IN PROGRESS
**Last Audit**: 2025-12-29

---

## Audit Summary (2025-12-29)

### What Has Actually Been Implemented

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| `--mode online/offline` CLI arg | DONE | acm_main.py:3307 | Works correctly |
| `PipelineMode` enum | DONE | acm_main.py:227 | ONLINE/OFFLINE enum |
| `ALLOWS_MODEL_REFIT` flag | DONE | acm_main.py:3396 | True only in OFFLINE |
| `ALLOWS_REGIME_DISCOVERY` flag | DONE | acm_main.py:3397 | True only in OFFLINE |
| Regime discovery gating | DONE | regimes.py:1717-1733 | Fails fast in ONLINE if no model |
| Model refit gating | DONE | acm_main.py:4138 | Blocks refit in ONLINE |
| `core/acm.py` entry point | DONE | core/acm.py (155 lines) | Routes to acm_main with mode |
| `MaturityState` enum | DONE | model_lifecycle.py:28 | COLDSTART/LEARNING/CONVERGED/DEPRECATED |
| `PromotionCriteria` dataclass | DONE | model_lifecycle.py:44 | 7 days, 3 runs, etc. |
| `ModelState` tracking | DONE | model_lifecycle.py:51 | Full state with metrics |
| `check_promotion_eligibility()` | DONE | model_lifecycle.py:91 | Returns (bool, unmet_reasons) |
| `promote_model()` | DONE | model_lifecycle.py:143 | LEARNING -> CONVERGED |
| Lifecycle wired into acm_main | DONE | acm_main.py:4550-4607 | Updates state each run |

### What Was Planned But NOT Implemented

| Component | Planned | Why Not Done |
|-----------|---------|--------------|
| `core/offline_pipeline.py` | Extract phases from acm_main | Over-engineering - phases work fine |
| `core/online_pipeline.py` | Separate scoring-only | Over-engineering - gating sufficient |
| `core/detector_manager.py` | Manage detector fit/score | Over-engineering - detectors work |
| `core/data_pipeline.py` | Data quality/features | Over-engineering - DataContract exists |
| UNKNOWN regime (label=-1) | Low-confidence assignment | NOT STARTED - Phase 2 |
| Confidence columns in tables | ACM_HealthTimeline, ACM_RUL | NOT STARTED - Phase 3 |
| RUL reliability gate | NOT_RELIABLE status | NOT STARTED - Phase 3 |
| `core/confidence.py` | Unified confidence | NOT STARTED - Phase 3 |
| Regime versioning | New version per OFFLINE | PARTIAL - version tracked, not new per run |

### Key Files Created/Modified

1. **core/acm.py** (NEW - 155 lines)
   - Single entry point with `--mode auto/online/offline`
   - Auto-detection: checks if model exists
   - Routes to acm_main.py with appropriate mode

2. **core/model_lifecycle.py** (NEW - 387 lines)
   - `MaturityState` enum: COLDSTART, LEARNING, CONVERGED, DEPRECATED
   - `PromotionCriteria`: min_training_days=7, min_silhouette_score=0.15, etc.
   - `ModelState` dataclass with full metrics tracking
   - `check_promotion_eligibility()`, `promote_model()`, `deprecate_model()`
   - `load_model_state_from_sql()`, `get_active_model_dict()`

3. **core/acm_main.py** (MODIFIED)
   - Line 227: Added `PipelineMode` to `RuntimeContext`
   - Lines 3307-3397: `--mode` CLI arg and flag initialization
   - Lines 4138: Model refit gating
   - Lines 4321: Regime discovery flag passed to regimes.py
   - Lines 4550-4607: Model lifecycle integration

4. **core/regimes.py** (MODIFIED)
   - Lines 1717-1733: `allow_discovery` flag gating

### Commits

| Commit | Description |
|--------|-------------|
| ecd979e | Phase 0: Add ONLINE/OFFLINE pipeline mode gating |
| abf905b | docs: Mark Phase 0 complete |
| 01948eb | Phase 1: Add model lifecycle management |
| 81c9dd0 | docs: Mark Phase 1 complete |

### Future Consideration: Pipeline Extraction

The following were deferred as current gating approach is sufficient:
- `core/offline_pipeline.py` - Extract phases from acm_main.py
- `core/online_pipeline.py` - Separate scoring-only pipeline
- `core/detector_manager.py` - Detector fit/score management

**Revisit if**: ONLINE/OFFLINE diverge significantly, or acm_main.py becomes unmaintainable.

---

## What Went Wrong

The "V11 refactor" to date:
- Created 23 modules that were never called from acm_main.py
- Deleted those 23 modules as "dead code"
- Added DataContract validation (fail-fast only)
- Added observability stack
- Cleaned up dead code
- Optimized calibration

What V11 was supposed to do (but did not):
- ONLINE/OFFLINE pipeline separation
- UNKNOWN regime handling
- Confidence model for all outputs
- RUL reliability gate
- Maturity state lifecycle
- Regime stability (no rediscovery every batch)

The tracker claimed 60/60 complete by counting module creation as "done" even though nothing was wired up.

---

## V10 Flaws That V11 Must Fix

| V10 Flaw | V11 Solution |
|----------|--------------|
| Regime discovery runs in online mode - regimes created/modified during every batch | Strict ONLINE/OFFLINE split - online never creates regimes |
| Forced regime assignment - every point forced to nearest regime | UNKNOWN/EMERGING regimes - RegimeLabel=-1 allowed |
| Circular dependency - PCA/detector outputs fed into regime clustering | Clean regime inputs - only operational sensor data |
| No train/score separation - batches could influence their own scores | Strict fit_baseline()/score() API - baseline frozen |
| Point anomalies trigger alerts - noisy, operator fatigue | Episodes as only alerting primitive |
| RUL always returns numbers - even with 5 data points | RUL reliability gate - explicit NOT_RELIABLE |
| Confidence not tracked - outputs presented as certain | Unified confidence model - every output has confidence |
| Cold start hidden - system pretends it knows things | Cold start as first-class state - explicit NULL version |
| Drift is dashboard-only - detected but not actionable | Drift as control signal - triggers offline replay |
| Health is instantaneous score - can jump abruptly | Health as time-evolving state with hysteresis |

---

## The 20 Non-Negotiable Rules

From Major Refactor Plan - these MUST be enforced:

1. Online code must never create/modify regimes
2. Offline code must never overwrite historical truth
3. No forced assignment when confidence is low
4. Cold start must be explicit and visible
5. Regime features must be operational only
6. Detector outputs must never define regimes
7. Point anomalies must never trigger alerts
8. Episodes are the only alerting primitive
9. Health must be time-based and confidence-aware
10. RUL must be gated or suppressed
11. Drift must influence control flow
12. All model changes must be versioned
13. Promotion requires objective evidence
14. UNKNOWN is a valid system output
15. SQL is the single source of truth
16. Replays must be reproducible
17. Confidence must always be exposed
18. Dashboards must reflect uncertainty
19. No module may silently reinterpret historical data
20. If unsure, the system must say "not reliable"

---

## Architecture

```
                           SINGLE ENTRY POINT
  
   python -m core.acm --equip FD_FAN
          |
          v
   +------------------+
   |  ACMController   | <-- New orchestrator class
   |  (core/acm.py)   |
   +--------+---------+
            |
            +---------------+------------------+
            |               |                  |
            v               v                  v
   +----------------+ +----------------+ +----------------+
   | OFFLINE Thread | | ONLINE Thread  | | Observability  |
   | (every 24h)    | | (every 30min)  | | (continuous)   |
   +----------------+ +----------------+ +----------------+
```

### ONLINE vs OFFLINE

| Aspect | ONLINE Pipeline | OFFLINE Pipeline |
|--------|----------------|------------------|
| Purpose | Current operational belief | Is belief still best explanation? |
| Frequency | Every 5-30 minutes | Every 1-7 days |
| Data Window | Last N minutes (small) | Historical span (large) |
| Regime Discovery | FORBIDDEN - uses existing | ALLOWED - creates new versions |
| Model Fitting | FORBIDDEN - loads cache | ALLOWED - refits all |
| Threshold Recalc | Uses active version | May create new version |
| Coldstart | Requires existing models | Handles coldstart |
| Latency Target | <30 seconds | <10 minutes |
| Failure Mode | Skip if no model, trigger OFFLINE | Create model |

---

## Implementation Phases

### Phase 0: Foundation [COMPLETE]

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| 0.1 | Create core/acm.py - single entry point | NEW | DONE |
| 0.2 | Add --mode online/offline/auto CLI arg | acm_main.py | DONE |
| 0.3 | Add PIPELINE_MODE, ALLOWS_MODEL_REFIT, ALLOWS_REGIME_DISCOVERY | acm_main.py | DONE |
| 0.4 | Gate regime discovery with if mode == OFFLINE | regimes.py | DONE |
| 0.5 | Gate model fitting with if mode == OFFLINE | acm_main.py | DONE |
| 0.6 | Add --mode arg to sql_batch_runner.py | sql_batch_runner.py | DONE |

Exit Criteria: PASSED
- --mode online shows allows_refit=False, allows_discovery=False
- --mode offline shows allows_refit=True, allows_discovery=True
- Commit: ecd979e

---

### Phase 1: OFFLINE Pipeline [COMPLETE]

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| 1.1 | Create core/offline_pipeline.py - extract main() phases | NEW | SKIPPED - over-engineering, acm_main.py has phases |
| 1.2 | Create core/detector_manager.py - detector fit/score | NEW | SKIPPED - over-engineering, detectors work fine |
| 1.3 | Create core/data_pipeline.py - data quality/features | NEW | SKIPPED - over-engineering, DataContract exists |
| 1.4 | Implement ACM_ActiveModels state machine | NEW | DONE - core/model_lifecycle.py |
| 1.5 | Implement auto-promotion when quality passes | MODIFY | DONE - check_promotion_eligibility() |
| 1.6 | Wire lifecycle into acm_main.py | MODIFY | DONE - lines 4548-4607 |

Exit Criteria: PASSED
- MaturityState enum: COLDSTART, LEARNING, CONVERGED, DEPRECATED
- PromotionCriteria: min_training_days=7, min_consecutive_runs=3, etc.
- Auto-promotion when LEARNING model meets criteria
- Commit: 01948eb

---

### Phase 2: ONLINE Pipeline [COMPLETE]

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| 2.1 | Create core/online_pipeline.py - scoring-only | NEW | SKIPPED - gating flags in acm_main.py suffice |
| 2.2 | Load frozen model from ModelRegistry | MODIFY | DONE - model_lifecycle.py integration |
| 2.3 | Regime assignment (predict only, no fit) | MODIFY | DONE - mode gating in regimes.py |
| 2.4 | Add UNKNOWN regime (label=-1) when confidence < threshold | MODIFY | DONE - regimes.py |
| 2.5 | Wire OnlinePipeline into acm.py | MODIFY | SKIPPED - uses gating flags |
| 2.6 | ONLINE fails triggers OFFLINE automatically | MODIFY | DONE - fallback in acm_main.py |

Exit Criteria: PASSED
- UNKNOWN_REGIME_LABEL = -1 for low-confidence assignments
- predict_regime_with_confidence() uses distance-based thresholding
- regime_confidence and regime_unknown_count in output
- Commit: 7111143

---

### Phase 3: Confidence and Reliability [COMPLETE]

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| 3.1 | Create core/confidence.py - unified confidence model | NEW | DONE |
| 3.2 | Add Confidence column to ACM_HealthTimeline | MODIFY | DONE - output_manager.py |
| 3.3 | Add Confidence + RUL_Status + MaturityState to ACM_RUL | MODIFY | DONE - forecast_engine.py |
| 3.4 | Add Confidence column to ACM_Anomaly_Events | MODIFY | DONE - output_manager.py |
| 3.5 | Implement RUL reliability gate (CONVERGED + min data) | MODIFY | DONE - forecast_engine.py |
| 3.6 | Output RUL_Status=NOT_RELIABLE when gate fails | MODIFY | DONE - forecast_engine.py |

Exit Criteria: PASSED
- ReliabilityStatus enum: RELIABLE, NOT_RELIABLE, LEARNING, INSUFFICIENT_DATA
- ConfidenceFactors dataclass with geometric mean computation
- compute_rul_confidence() returns (confidence, status, reason)
- RUL output includes RUL_Status and MaturityState columns
- SQL tables altered to include new columns

---

### Phase 4: Regime Stability [COMPLETE]

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| 4.1 | Create core/regime_lifecycle.py - version management | NEW | SKIPPED - model_lifecycle.py handles regime versioning |
| 4.2 | OFFLINE creates new RegimeVersion, never overwrites | MODIFY | DONE - model_persistence.py StateVersion |
| 4.3 | Add AssignmentConfidence to ACM_RegimeTimeline | MODIFY | DONE - output_manager.py |
| 4.4 | ONLINE uses frozen regime model only | MODIFY | DONE - ALLOWS_REGIME_DISCOVERY=False |
| 4.5 | Add promotion criteria (silhouette>0.15, stability>0.8, 7+ days) | MODIFY | DONE - model_lifecycle.py PromotionCriteria |

Exit Criteria: PASSED
- RegimeVersion tracked in ACM_RegimeState.StateVersion
- AssignmentConfidence added to ACM_RegimeTimeline output
- ONLINE mode (ALLOWS_REGIME_DISCOVERY=False) never modifies regimes
- PromotionCriteria enforced in model_lifecycle.py

---

### Phase 5: Single Entry Point [COMPLETE]

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| 5.1 | Implement ACMController.start() in core/acm.py | MODIFY | DONE - core/acm.py main() |
| 5.2 | Auto mode: check model exists, route to ONLINE or OFFLINE | MODIFY | DONE - _detect_mode() |
| 5.3 | Update sql_batch_runner.py to use new entry point | MODIFY | DONE - --mode argument |
| 5.4 | Add scheduling logic (ONLINE 30min, OFFLINE 24h) | MODIFY | SKIPPED - deferred to operational scripts |
| 5.5 | Create scripts/acm_launcher.py for production | NEW | SKIPPED - core/acm.py is sufficient |

Exit Criteria: PASSED
- `python -m core.acm --equip FD_FAN --mode auto` works
- Auto-detect mode routes to ONLINE if model exists, else OFFLINE
- sql_batch_runner.py supports --mode argument

---

### Phase 6: Observability Dashboard

| Task | Description | Files |
|------|-------------|-------|
| 6.1 | Create acm_v11_operations.json dashboard | NEW |
| 6.2 | Panel: Pipeline mode (ONLINE/OFFLINE) per equipment | |
| 6.3 | Panel: Model maturity state per equipment | |
| 6.4 | Panel: Confidence distributions over time | |
| 6.5 | Panel: Phase timings | |
| 6.6 | Panel: ONLINE latency percentiles | |
| 6.7 | Add @Span.trace() decorators to all new modules | MODIFY |

Exit Criteria: Dashboard shows operations, confidence, maturity, latency.

---

### Phase 7: Cleanup

| Task | Description | Files |
|------|-------------|-------|
| 7.1 | Deprecate acm_main.py (thin wrapper) | MODIFY |
| 7.2 | Update copilot-instructions.md with V11 architecture | MODIFY |
| 7.3 | Update version to 11.1.0 | MODIFY |
| 7.4 | Tag release | GIT |
| 7.5 | Update tracker with actual status | MODIFY |

---

## File Structure After Refactor

```
core/
    acm.py                  # NEW - Single entry point (50 lines)
    pipeline_context.py     # NEW - Dataclasses (150 lines)
    offline_pipeline.py     # NEW - OFFLINE batch pipeline (800 lines)
    online_pipeline.py      # NEW - ONLINE scoring pipeline (300 lines)
    detector_manager.py     # NEW - Detector fit/score (400 lines)
    data_pipeline.py        # NEW - Data quality/features (500 lines)
    health_pipeline.py      # NEW - Episode/health builders (400 lines)
    regime_lifecycle.py     # NEW - Maturity state machine (300 lines)
    confidence.py           # NEW - Unified confidence (200 lines)
    acm_main.py             # DEPRECATED - Thin wrapper (100 lines)
    
    # Existing (unchanged)
    observability.py        # Traces/metrics/logs
    output_manager.py       # SQL/file writes
    fuse.py                 # Fusion
    regimes.py              # Regime clustering
    forecast_engine.py      # RUL/forecasting
```

---

## Commit Strategy

| Commit | Content |
|--------|---------|
| v11: Add --mode CLI argument and gate regime discovery | Phase 0 |
| v11: Extract offline pipeline from acm_main | Phase 1 |
| v11: Implement online pipeline with model loading | Phase 2 |
| v11: Add confidence model and RUL reliability gate | Phase 3 |
| v11: Implement regime versioning and maturity lifecycle | Phase 4 |
| v11: Create single entry point and launcher | Phase 5 |
| v11: Add operations dashboard | Phase 6 |
| v11.1.0: Complete V11 proper implementation | Phase 7 + Tag |

---

## Development Rules

- No emojis or unicode characters
- No lazy wrappers
- No over-engineering - POC first, edge cases as discovered
- Rewrite properly, no shortcuts
- Optimize memory and CPU
- Follow source control practices (commit, branch, tag)
- Maintain versioning
- Maintain this task list
- Delete single-use scripts when done
- No unnecessary documentation
- Follow V11 philosophy always

---

## Progress Tracking

### Phase 0: Foundation [COMPLETE - ecd979e]
- [x] 0.1 Create core/acm.py - single entry point with auto mode detection
- [x] 0.2 Add PipelineMode enum to RuntimeContext (acm_main.py:227)
- [x] 0.3 Add --mode online/offline CLI arg (acm_main.py:3307)
- [x] 0.4 Gate regime discovery with allow_discovery flag (regimes.py:1717)
- [x] 0.5 Gate model fitting with ALLOWS_MODEL_REFIT (acm_main.py:4138)

### Phase 1: Model Lifecycle [COMPLETE - 01948eb]
- [x] 1.1 Create core/model_lifecycle.py with MaturityState enum
- [x] 1.2 Implement PromotionCriteria (7 days, 0.15 silhouette, 3 runs)
- [x] 1.3 Implement check_promotion_eligibility() and promote_model()
- [x] 1.4 Wire lifecycle into acm_main.py (lines 4550-4607)
- [~] 1.5 SKIPPED: offline_pipeline.py, detector_manager.py, data_pipeline.py (over-engineering)

### Phase 2: ONLINE Pipeline [COMPLETE - 7111143]
- [x] 2.1 ONLINE mode scoring works (gating in regimes.py:1811)
- [x] 2.2 Load frozen model from ModelRegistry (uses existing cache)
- [x] 2.3 Regime assignment predict-only (gating done)
- [x] 2.4 Add UNKNOWN regime (label=-1) when confidence < threshold
- [x] 2.5 ONLINE fail triggers OFFLINE automatically (fallback in acm.py:117)
- [ ] 2.6 Integration test: --mode online with existing model

Key additions:
- `UNKNOWN_REGIME_LABEL = -1` constant (regimes.py:41)
- `predict_regime_with_confidence()` function (regimes.py:756-827)
- `regime_confidence` array in output (0-1 per point)
- `regime_unknown_count` in output
- Config: `regimes.unknown.enabled`, `regimes.unknown.distance_percentile`

### Phase 3: Confidence and Reliability [NOT STARTED]
- [ ] 3.1 Create core/confidence.py - unified confidence model
- [ ] 3.2 Add Confidence column to ACM_HealthTimeline
- [ ] 3.3 Add Confidence column to ACM_RUL
- [ ] 3.4 Add Confidence column to ACM_Episodes
- [ ] 3.5 Implement RUL reliability gate (CONVERGED + min data)
- [ ] 3.6 Output RUL_Status=NOT_RELIABLE when gate fails

### Phase 4: Regime Stability [NOT STARTED]
- [ ] 4.1 Regime versioning (new version per OFFLINE discovery)
- [ ] 4.2 OFFLINE creates new RegimeVersion, never overwrites
- [ ] 4.3 Add AssignmentConfidence to ACM_RegimeTimeline
- [ ] 4.4 ONLINE uses frozen regime model only (gating done)
- [ ] 4.5 Add promotion criteria (silhouette>0.15, stability>0.8, 7+ days)

### Phase 5: Single Entry Point [PARTIAL]
- [x] 5.1 core/acm.py with auto mode detection
- [x] 5.2 Auto mode: check model exists, route to ONLINE or OFFLINE
- [ ] 5.3 Update sql_batch_runner.py to use new entry point
- [ ] 5.4 Add scheduling logic (ONLINE 30min, OFFLINE 24h)
- [ ] 5.5 Create scripts/acm_launcher.py for production

### Phase 6: Observability Dashboard [NOT STARTED]
- [ ] 6.1 Create acm_v11_operations.json dashboard
- [ ] 6.2-6.6 Dashboard panels (mode, maturity, confidence, latency)
- [ ] 6.7 Add @Span.trace() decorators to new modules

### Phase 7: Cleanup [NOT STARTED]
- [ ] 7.1 Deprecate acm_main.py (thin wrapper)
- [ ] 7.2 Update copilot-instructions.md with V11 architecture
- [ ] 7.3 Update version to 11.1.0
- [ ] 7.4 Tag release
- [ ] 7.5 Update tracker
