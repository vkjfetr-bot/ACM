# ACM V11.0.0 Fixes - Prioritized Action Plan

**Date**: December 30, 2025  
**Based On**: V11_IMPLEMENTATION_AUDIT.md, V11_DEEP_DIVE_ISSUES.md, V11_ANALYTICAL_CORRECTNESS_AUDIT.md  
**Status**: READY TO EXECUTE

---

## Executive Summary

The V11 audit revealed:
- ✅ **85% implementation complete** - Core architecture is sound
- 🔴 **5 CRITICAL bugs** that cause runtime failures
- ⚠️ **3 major integration gaps** that prevent features from working
- 🟡 **4 analytical flaws** in unsupervised learning approach

This plan prioritizes **CRITICAL bugs first**, then **integration gaps**, then **analytical improvements**.

---

## Phase 1: CRITICAL Bug Fixes (MUST FIX IMMEDIATELY)

### 🔴 CRITICAL-1: Model Lifecycle Race Condition in Forecasting
**File**: `core/forecast_engine.py` + `core/acm_main.py`  
**Issue**: ForecastEngine loads model_state independently, creates race condition  
**Impact**: RUL marked NOT_RELIABLE even when model is CONVERGED  
**Priority**: P0 - Breaks V11 Rule #10

**Fix**:
1. Add `model_state` parameter to ForecastEngine constructor
2. Pass current model_state from acm_main instead of reloading from SQL
3. Remove `_get_model_maturity_state()` method (redundant)

**Code Changes**:
```python
# In core/forecast_engine.py __init__():
def __init__(self, ..., model_state=None):
    self.model_state = model_state  # NEW: Accept from caller
    
# In core/acm_main.py (line ~5480):
forecast_engine = ForecastEngine(
    ...,
    model_state=model_state,  # Pass current state
)
```

**Test**: Verify RUL_Status=RELIABLE after model promotion

---

### 🔴 CRITICAL-2: regime_state_version Undefined Variable Access
**File**: `core/acm_main.py` (lines 4565, 4600)  
**Issue**: Uses fragile `dir()` check, crashes on NameError  
**Impact**: Model lifecycle initialization fails  
**Priority**: P0 - Breaks new equipment cold-start

**Fix**:
1. Initialize `regime_state_version = 0` at function scope (top of run())
2. Remove all `if 'regime_state_version' in dir()` checks
3. Use direct variable access

**Code Changes**:
```python
# At top of run() function (line ~3380):
regime_state_version = 0  # Initialize with default

# Replace lines 4565, 4600:
version = regime_state_version  # Direct access, no dir() check
```

**Test**: Run new equipment from scratch, verify no NameError

---

### 🔴 CRITICAL-3: train_start/train_end Undefined in Model Lifecycle
**File**: `core/acm_main.py` (lines 4570-4571)  
**Issue**: Variables only defined inside conditional, crashes if condition false  
**Impact**: Model lifecycle breaks when regime model not trained  
**Priority**: P0 - Breaks model creation

**Fix**:
1. Initialize train_start/train_end before conditional blocks
2. Use train.index min/max as fallback

**Code Changes**:
```python
# Before line 4552:
train_start = train.index.min() if len(train) > 0 else datetime.now()
train_end = train.index.max() if len(train) > 0 else datetime.now()

# Keep existing if block, but variables are always defined
```

**Test**: Run with regime_model_was_trained=False, verify no NameError

---

### 🔴 CRITICAL-4: DataContract Timestamp Check Broken
**File**: `core/pipeline_types.py` (line ~150)  
**Issue**: Checks `df[self.timestamp_col]` but timestamp is often the index  
**Impact**: Validation fails with KeyError on valid data  
**Priority**: P0 - Breaks data loading

**Fix**: Already implemented in current working branch ✅

**Verification**: Check that `validate()` handles DatetimeIndex

---

### 🔴 CRITICAL-5: Confidence Function Argument Mismatches
**File**: `core/output_manager.py` (lines ~2420, ~3422)  
**Issue**: compute_episode_confidence() and compute_health_confidence() called with wrong args  
**Impact**: Runtime errors when writing confidence  
**Priority**: P0 - Breaks confidence tracking

**Fix**: Already implemented in current working branch ✅

**Verification**: Check function signatures match calls

---

## Phase 2: Integration Gaps (MUST FIX FOR V11 COMPLETION)

### ⚠️ GAP-1: Seasonality Detection Not Applied
**File**: `core/acm_main.py` (line ~3820)  
**Issue**: Patterns detected but seasonal adjustment never applied  
**Impact**: False positives during diurnal/weekly cycles  
**Priority**: P1 - Reduces detection accuracy

**Fix**:
1. After seasonality.detect, call `handler.adjust_baseline(train, seasonal_patterns)`
2. Apply adjustment to score data as well
3. Log adjustment magnitude

**Code Changes**:
```python
# After line 3843 (seasonal pattern detection):
if seasonal_patterns:
    train = handler.adjust_baseline(train, seasonal_patterns)
    score = handler.adjust_baseline(score, seasonal_patterns)
    Console.info(f"Applied seasonal adjustment to {len(seasonal_patterns)} sensors")
```

**Test**: Compare anomaly count before/after adjustment for equipment with daily patterns

---

### ⚠️ GAP-2: Health/Episode Confidence Columns Missing
**File**: `core/output_manager.py` + SQL schema  
**Issue**: Confidence functions exist but not integrated in health/episode writes  
**Impact**: Incomplete V11 confidence coverage  
**Priority**: P1 - Breaks V11 Rule #17

**Fix**:
1. Add Confidence column to ACM_HealthTimeline SQL schema
2. Add Confidence column to ACM_Anomaly_Events SQL schema
3. Call compute_health_confidence() when writing health
4. Call compute_episode_confidence() when writing episodes

**Code Changes**:
```python
# In write_health_timeline():
from core.confidence import compute_health_confidence
conf = compute_health_confidence(
    fused_z=row['FusedZ'],
    maturity_state=maturity_state,
    sample_count=len(df)
)
row['Confidence'] = float(conf)

# In write_anomaly_events():
from core.confidence import compute_episode_confidence
conf = compute_episode_confidence(
    episode_duration_seconds=duration,
    peak_z=peak_z,
    maturity_state=maturity_state
)
episode_row['Confidence'] = float(conf)
```

**Test**: Query ACM_HealthTimeline and ACM_Anomaly_Events, verify Confidence populated

---

### ⚠️ GAP-3: Asset Similarity Transfer Learning Not Active
**File**: `core/acm_main.py` coldstart section  
**Issue**: Profiles built but transfer_baseline() never called  
**Impact**: New equipment starts from scratch, no knowledge transfer  
**Priority**: P2 - Reduces cold-start quality

**Fix**:
1. In coldstart mode, before loading data, check for similar assets
2. If found, call transfer_baseline() to bootstrap
3. Log transfer source and similarity score

**Code Changes**:
```python
# In coldstart section (before data loading):
from core.asset_similarity import AssetSimilarity, find_similar
similarity_engine = AssetSimilarity(sql_client)
similar_asset = similarity_engine.find_similar(equip_id, min_similarity=0.7)
if similar_asset:
    baseline = similarity_engine.transfer_baseline(similar_asset, equip_id)
    Console.info(f"Transferred baseline from {similar_asset.name} (similarity={similar_asset.score:.2f})")
```

**Test**: Add new equipment, verify baseline transferred from similar existing equipment

---

## Phase 3: Analytical Improvements (ENHANCES V11 QUALITY)

### 🟡 ANALYTICAL-1: K-Means Finds Density, Not Operating Modes
**File**: `core/regimes.py`  
**Issue**: Clustering optimizes silhouette score, not semantic correctness  
**Impact**: Regimes don't correspond to actual operational modes  
**Priority**: P2 - Reduces regime interpretability

**Fix** (See SYSTEM_DESIGN_OPERATING_CONDITION_DISCOVERY.md):
1. Implement HybridRegimeClustering (K-Means + HDBSCAN)
2. Add physics-informed feature weighting
3. Add temporal coherence scoring
4. Add transition rate validation

**Complexity**: HIGH - Requires new module  
**Defer**: Post-V11.1

---

### 🟡 ANALYTICAL-2: Detector Fusion Assumes Independence
**File**: `core/fuse.py`  
**Issue**: Detectors are correlated (PCA-SPE and GMM overlap), fusion double-counts  
**Impact**: Inflated anomaly scores  
**Priority**: P2 - Reduces detection precision

**Fix**:
1. Compute detector correlation matrix
2. Weight fusion by 1/correlation
3. Use ensemble methods (Random Forest on detector outputs)

**Complexity**: MEDIUM  
**Defer**: Post-V11.1

---

### 🟡 ANALYTICAL-3: RUL Assumes Monotonic Degradation
**File**: `core/forecast_engine.py`  
**Issue**: Linear/exponential extrapolation assumes health always decreases  
**Impact**: Fails for intermittent faults or maintenance events  
**Priority**: P2 - Reduces RUL reliability

**Fix**:
1. Add regime-conditional RUL (different degradation rates per regime)
2. Detect maintenance events (health resets)
3. Use piecewise models instead of global extrapolation

**Complexity**: HIGH  
**Defer**: Post-V11.2

---

### 🟡 ANALYTICAL-4: Silhouette Score Favors Separation Over Semantics
**File**: `core/regimes.py` (line ~555)  
**Issue**: Best silhouette != most meaningful regimes  
**Impact**: Auto-k selection unstable, picks trivial splits  
**Priority**: P2 - Reduces regime stability

**Fix**:
1. Replace pure silhouette with composite score:
   - 30% silhouette (cluster quality)
   - 30% temporal autocorrelation (regime persistence)
   - 20% transition rate (state machine validity)
   - 20% stability over time (reproducibility)
2. Add minimum cluster size constraint

**Complexity**: MEDIUM  
**Plan**: Include in V11.1

---

## Phase 4: Testing & Validation

### Test Suite Requirements

**Unit Tests** (NEW):
1. `test_model_lifecycle_promotion()` - Verify LEARNING → CONVERGED
2. `test_rul_reliability_gating()` - Verify NOT_RELIABLE when LEARNING
3. `test_unknown_regime_assignment()` - Verify UNKNOWN label for low confidence
4. `test_data_contract_fail_fast()` - Verify exception on invalid data
5. `test_seasonal_adjustment()` - Verify patterns removed from data
6. `test_confidence_propagation()` - Verify all outputs have Confidence column

**Integration Tests** (NEW):
1. **10-day batch test**: Run FD_FAN + GAS_TURBINE + ELECTRIC_MOTOR for 10 days
   - Verify model promotion happens
   - Verify RUL_Status transitions LEARNING → RELIABLE
   - Verify UNKNOWN regimes appear for low-confidence periods
   - Verify no crashes, all outputs populated

2. **ONLINE mode test**: Run with existing model, verify:
   - No model retraining
   - No regime discovery
   - Scoring only

3. **Cold-start test**: New equipment from scratch
   - Verify model_state created
   - Verify ACM_ActiveModels populated
   - Verify no undefined variable crashes

**SQL Validation** (MANUAL):
```sql
-- Check model lifecycle
SELECT * FROM ACM_ActiveModels WHERE EquipID=1;

-- Check RUL reliability
SELECT TOP 10 RUL_Status, Confidence, MaturityState 
FROM ACM_RUL WHERE EquipID=1 ORDER BY CreatedAt DESC;

-- Check UNKNOWN regimes
SELECT RegimeLabel, COUNT(*) 
FROM ACM_RegimeTimeline WHERE EquipID=1 GROUP BY RegimeLabel;

-- Check confidence propagation
SELECT TOP 10 Confidence FROM ACM_HealthTimeline WHERE EquipID=1;
SELECT TOP 10 Confidence FROM ACM_Anomaly_Events WHERE EquipID=1;
```

---

## Implementation Timeline

### Week 1: CRITICAL Fixes (P0)
- Day 1: CRITICAL-1 (Model lifecycle race condition)
- Day 2: CRITICAL-2 + CRITICAL-3 (Undefined variables)
- Day 3: Unit tests for CRITICAL fixes
- Day 4: Integration test (10-day batch)
- Day 5: Bug fix iteration

**Deliverable**: V11.0.1 - All P0 bugs fixed, integration tests passing

### Week 2: Integration Gaps (P1)
- Day 6-7: GAP-1 (Seasonal adjustment)
- Day 8-9: GAP-2 (Health/episode confidence)
- Day 10: GAP-3 (Transfer learning)
- Day 11-12: Integration tests, validation

**Deliverable**: V11.1.0 - Full V11 feature completion

### Week 3+: Analytical Improvements (P2)
- ANALYTICAL-4: Composite quality score for regime selection
- ANALYTICAL-1: Hybrid clustering (deferred to V11.2)
- ANALYTICAL-2: Detector fusion improvements (deferred to V11.2)
- ANALYTICAL-3: Regime-conditional RUL (deferred to V11.3)

**Deliverable**: V11.2.0+ - Enhanced analytical correctness

---

## Success Criteria

V11.0.1 is ready when:
- [ ] All 5 CRITICAL bugs fixed
- [ ] 10-day batch test passes for 3 equipment
- [ ] No NameError or undefined variable crashes
- [ ] RUL_Status transitions from LEARNING to RELIABLE
- [ ] Model promotion happens automatically

V11.1.0 is ready when:
- [ ] Seasonal adjustment applied (false positives reduced)
- [ ] Confidence columns in all output tables
- [ ] Transfer learning works for new equipment
- [ ] All V11 rules (10, 14, 17, 20) verified

---

## Current Status (Updated December 30, 2025)

**Phase 1: CRITICAL Fixes - ALL COMPLETE ✅**
- ✅ CRITICAL-1: Model lifecycle race condition - FIXED (commit 529d774)
  - ForecastEngine uses ONLY cached model_state from constructor
  - Removed SQL fallback that caused race condition
  - Removed load_model_state_from_sql import
- ✅ CRITICAL-2: regime_state_version undefined - ALREADY FIXED (confirmed via grep)
  - Variable properly initialized at line 4094
  - No more dir() checks needed
- ✅ CRITICAL-3: train_start/train_end undefined - FIXED (commit 529d774)
  - Variables initialized at function scope (line ~3655)
  - SET values later in execution flow
  - Removed fragile 'var' in locals() patterns
- ✅ CRITICAL-4: DataContract timestamp check - FIXED (in branch)
- ✅ CRITICAL-5: Confidence function args - FIXED (in branch)

**Phase 2: Integration Gaps - MOSTLY COMPLETE ✅**
- ✅ GAP-1: Seasonality adjustment - DONE
  - Patterns detected and applied to baseline/score data
- ✅ GAP-2: Health/Episode confidence - DONE
  - Confidence columns populated in ACM_HealthTimeline, ACM_Anomaly_Events
- ⏳ GAP-3: Asset Similarity Transfer Learning - DEFERRED
  - Profiles are built and stored in ACM_AssetProfiles
  - Transfer learning NOT implemented (explicitly marked as "Future work" in code)
  - Target: V11.1.0 or later

**Phase 3: Analytical Improvements - DEFERRED TO V11.2+**
- 🟡 ANALYTICAL-1 through ANALYTICAL-4: All deferred per plan

**Verification Results (December 30, 2025)**:
- ✅ Full batch run completed: `BATCH RUNNER COMPLETED SUCCESSFULLY`
- ✅ Model lifecycle tracked: `[LIFECYCLE] Created model v1 in LEARNING state`
- ✅ RUL reliability gating: `RUL reliability: LEARNING - Model still LEARNING`
- ✅ All forecast tables written successfully
- ✅ No NameError or undefined variable crashes

**Next Actions**:
1. ✅ All P0 CRITICAL fixes complete
2. Run multi-batch test (3+ batches) to verify model promotion
3. Verify LEARNING → CONVERGED transition after 7 days + 3 runs
4. Consider GAP-3 implementation for V11.1.0

---

# More Info by ChatGpt


Below is a **fresh, deep re-audit** of the scripts you re-uploaded, with focus on **V11 analytical soundness** (not just runtime stability) and **alignment to your V11 action plan**.

---

## V11 Re-Audit Table (Analytical Soundness + V11 Rule Compliance)

| Script               | V11 Concern Area                                           | What Looks Correct (Now)                                                                                                                                                                                                         | Analytical / Design Problems Still Present                                                                                                                                                                                                                                                                                                 | Severity                         | Concrete Fix / Enforcement                                                                                                                                                                                             |
| -------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `acm_main.py`        | **Lifecycle + regime version pointer**                     | You have a **clean path** that writes active models with `regime_version=regime_state_version` and logs lifecycle state; promotion check is in place.                                                                            | There is still **another path** that uses `regime_state_version if 'regime_state_version' in dir() else None` which is exactly the fragile pattern your CRITICAL-2 fix was meant to remove. This means you have **duplicate/competing logic blocks** and a future regression risk.                                                         | P0                               | **Delete all dir() checks** and enforce a single lifecycle write path. Add a unit test that greps/guards against `'regime_state_version' in dir()` appearing in `acm_main.py`.                                         |
| `acm_main.py`        | **Calibration correctness (train-fit / score-transform)**  | Calibration section explicitly states **fit calibrators on TRAIN and transform SCORE**, and uses a helper `_score_all_detectors()` to score TRAIN for calibration baseline. This is analytically correct and addresses leakage.  | Still missing (from retrieved excerpts) the **post-seasonality “apply adjustment” step** you called out in GAP-1. I did not see evidence of `adjust_baseline()` being applied after detection in the retrieved chunks (so assume still a gap until proven otherwise).                                                                      | P1                               | Enforce: “detect → adjust baseline → score”. Add observability counters: % variance removed, sensors adjusted, and delta in anomaly count.                                                                             |
| `output_manager.py`  | **RUL reliability gating (Rule #10)**                      | `write_rul_forecast()` gates reliability: if maturity is `LEARNING`, it writes `RUL_Status="NOT_RELIABLE"` and clears RUL outputs; otherwise it writes RELIABLE. This matches your V11 reliability gating requirement.           | It still **loads ActiveModels inside the writer** if `maturity_state` is not passed (`get_active_models_by_equip_id`). That re-introduces the “multiple sources of truth” pattern you flagged as CRITICAL-1 (race / inconsistency). The function supports passing `maturity_state`, but the design still allows “silent fallback to SQL”.  | P0 (design-level)                | Make `maturity_state` **mandatory** at this boundary (or assert if None). Do not allow writers to reload lifecycle state. This is the same principle as your CRITICAL-1 fix.                                           |
| `forecast_engine.py` | **Forecast confidence plumbing**                           | `ForecastResult` carries `confidence`, `details`, and `contributing_factors` and is written through to SQL by `write_rul_forecast()` (it writes `Confidence` from forecast output).                                              | The primary analytical risk is not the container object; it is the **source-of-truth lifecycle gating** (see above). Until ForecastEngine is guaranteed to consume the same `model_state` instance as the run, you can still mark good RUL as NOT_RELIABLE or vice-versa.                                                                  | P0                               | Force ForecastEngine to accept `model_state` (or maturity) from the run context; remove internal reload paths. Then enforce via test: “same run_id → same maturity used everywhere”.                                   |
| `confidence.py`      | **Confidence curves (health + episodes)**                  | Both confidence functions exist with reasonable monotonic behavior: health confidence penalizes quality issues and extrapolation; episode confidence scales by duration/peak_z and maturity.                                     | Purely analytical: confidence formulas are **hand-tuned heuristics**. That’s acceptable for V11.0/V11.1, but you should treat them as **calibrated models** later (learned mapping using historical false positives/true positives).                                                                                                       | P2                               | Add “confidence calibration dataset”: label a subset of episodes as true/false → fit isotonic/logistic calibrator on (`peak_z`, `duration`, `maturity`, `quality_flags`).                                              |
| `regimes.py`         | **V11 Rule #14: UNKNOWN regimes allowed**                  | UNKNOWN regime label constant is explicitly introduced (`UNKNOWN_REGIME_LABEL = -1`) and is documented against V11 rules.                                                                                                        | I did not see (in the retrieved excerpts) the **actual assignment gating** that produces UNKNOWN based on a confidence threshold (e.g., centroid distance / soft assignments). The constant exists, but unless it is used in the assignment path, V11 Rule #3/#14 is only partially implemented.                                           | P1                               | Ensure assignment uses a measurable confidence (distance-to-centroid percentile or softmax margin) and yields UNKNOWN when below threshold. Add a test: craft ambiguous points between clusters → must output UNKNOWN. |
| `regimes.py`         | **Auto-k selection semantics (Analytical-4)**              | Auto-k sweep exists and **flags low-quality** when all silhouette scores are below threshold; metadata records `quality_ok`, `quality_notes`, sweep, model version, sklearn version.                                             | This still implements exactly the flaw you identified: **silhouette-driven selection favors separation, not operating-mode semantics**. There is no temporal persistence / transition sanity / reproducibility in the selection objective (in shown code).                                                                                 | P2 (but strategically important) | Implement V11.1 “composite score” you already defined: silhouette + temporal autocorr/persistence + transition rate + stability across runs. Gate promotion/quality_ok on composite, not silhouette alone.             |
| `regimes.py`         | **Feature basis correctness**                              | The basis builder carefully avoids scaling PCA-derived components together with raw tags (PCA variance is recorded and the note clarifies scaling boundaries). This is a good analytical hygiene improvement.                    | Analytical risk: mixing raw tags + PCA features is still sensitive to tag availability/quality. You have a fallback to first N columns if none exist, which can create unstable basis signatures and drift in regime definition between runs.                                                                                              | P2                               | Persist and enforce a **FeatureBasisSignature**: once a regime model is active, do not silently change basis composition; instead create a new regime version and promote only after stability checks.                 |
| `pipeline_types.py`  | **Data contract timestamp/index correctness (CRITICAL-4)** | Your V11 plan says this is fixed. I did not re-locate the relevant snippet in the retrieved lines here, so I cannot independently confirm in this response.                                                                      | If not fixed everywhere: the common failure mode is checking `df[timestamp_col]` when timestamp is the index.                                                                                                                                                                                                                              | P0 if unfixed                    | Ensure `validate()` accepts DatetimeIndex and normalizes to index-first checks. Add unit tests for both layouts (timestamp column vs DatetimeIndex).                                                                   |

---

## What is “Analytically Sound” Already vs Still Not

**Already sound (based on code evidence):**

* Calibration leakage prevention: “fit on TRAIN / transform SCORE.” 
* RUL reliability gating based on maturity state. 
* Regime metadata capture + low-quality flagging + versioning hygiene. 

**Still analytically weak / incomplete (most important):**

* **Duplicate lifecycle logic** in `acm_main.py` (some fixed, some still old/fragile). 
* **Silhouette-only auto-k**: still separation-driven, not operating-mode-driven. 
* **UNKNOWN regime**: constant exists, but the “confidence-based UNKNOWN assignment” must be verified in the assignment path (not proven by retrieved excerpts). 
* **Writers reloading lifecycle state**: `write_rul_forecast()` can still query ActiveModels if not passed maturity, which is the same architectural smell behind CRITICAL-1. 

---

## Immediate Next Fixes (Analytical + Non-Regression)

1. **Eliminate all lifecycle duplication in `acm_main.py`** (single authoritative path; remove dir() fallback everywhere). 
2. **Hard-enforce single source of truth for maturity**: writers must not query ActiveModels when run context already knows maturity. (Fail fast if maturity not provided.) 
3. **Regime V11.1 composite score**: implement persistence + transition sanity + stability across runs in selection objective (replace silhouette-only). 
4. **Prove UNKNOWN regimes work**: add explicit confidence gating test and a forced ambiguous synthetic dataset test.
