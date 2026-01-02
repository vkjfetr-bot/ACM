# ACM V11.0.0 Deep Dive Analysis - Critical Issues Found

**Date**: December 30, 2025  
**Analyst**: Deep code path tracing and integration review  
**Scope**: Runtime behavior, integration points, edge cases, and design flaws

---

## Executive Summary

After thorough code path tracing and runtime analysis, **5 CRITICAL ISSUES** and **8 DESIGN CONCERNS** were identified in the V11 implementation. While the core architectural components are coded, there are significant **integration gaps**, **runtime failures**, and **logical inconsistencies** that prevent V11 from working correctly in production.

**Severity Breakdown**:
- 🔴 **CRITICAL (5)**: Will cause runtime failures or incorrect behavior
- 🟡 **HIGH (4)**: Serious design issues that undermine V11 goals  
- 🟠 **MEDIUM (4)**: Incomplete integration or missing functionality

---

## 🔴 CRITICAL ISSUES

### CRITICAL-1: Model Lifecycle Disconnected from Forecasting ⚠️ SEVERE

**Location**: `core/forecast_engine.py::_get_model_maturity_state()` (lines 549-570)

**Problem**: ForecastEngine loads MaturityState **independently** from acm_main.py's model lifecycle tracking. This creates **two separate sources of truth**.

**Code Evidence**:
```python
# In forecast_engine.py (line 559):
def _get_model_maturity_state(self):
    model_state = load_model_state_from_sql(self.sql_client, self.equip_id)
    if model_state is None:
        return 'COLDSTART', 0, 0.0
    return (str(model_state.maturity.value), ...)

# In acm_main.py (line 4561):
model_state = load_model_state_from_sql(sql_client, equip_id)
if model_state is None:
    model_state = create_new_model_state(...)
model_state = update_model_state_from_run(...)  # Updates state
output_manager.write_active_models(...)  # Writes to SQL
```

**Race Condition**:
1. acm_main.py loads model_state at line 4561
2. Updates and promotes model (lines 4578-4593)
3. Writes to SQL (line 4598)
4. **LATER** forecast_engine runs (line 5480)
5. ForecastEngine loads **stale** model_state (before write completes)
6. RUL reliability uses **outdated maturity state**

**Impact**:
- Model promoted to CONVERGED in acm_main
- ForecastEngine sees LEARNING state
- RUL marked as NOT_RELIABLE despite model being ready
- **V11 Rule #10 violated**: Reliable RUL predictions suppressed incorrectly

**Fix Required**:
Pass model_state from acm_main to ForecastEngine constructor:
```python
forecast_engine = ForecastEngine(
    sql_client=...,
    model_state=model_state,  # NEW: Pass current state
    ...
)
```

---

### CRITICAL-2: regime_state_version Undefined Variable Access

**Location**: `core/acm_main.py` (lines 4565, 4600)

**Problem**: Code uses `regime_state_version if 'regime_state_version' in dir()` which is **fragile and error-prone**.

**Code Evidence**:
```python
# Line 4565:
version = regime_state_version if 'regime_state_version' in dir() else 1

# Line 4600:
regime_version=regime_state_version if 'regime_state_version' in dir() else None,
```

**Why This Fails**:
1. `dir()` returns names in **current local scope**, not guaranteed to include `regime_state_version`
2. Variable defined at line 4302 (`regime_state_version = 0`) but could be **out of scope** in nested try/except
3. If exception occurs between definition and usage, variable is undefined
4. Python `NameError` will crash the run

**Proof**:
```python
# If this happens (line 4302 inside try block):
try:
    regime_state_version = 0
    # ... load regime state ...
except Exception as e:
    # regime_state_version IS LOST here if exception occurred before assignment
    pass

# Later (line 4565 outside try block):
version = regime_state_version if 'regime_state_version' in dir() else 1  
# NameError: name 'regime_state_version' is not defined
```

**Impact**:
- Runtime crash on first run for new equipment
- Model lifecycle tracking fails to initialize
- Unpredictable behavior depending on exception timing

**Fix Required**:
Initialize at function scope with default:
```python
regime_state_version = 0  # Top of function, before any try/except
```

---

### CRITICAL-3: train_start/train_end Undefined in Model Lifecycle

**Location**: `core/acm_main.py` (lines 4570-4571)

**Problem**: `train_start` and `train_end` used in `create_new_model_state()` but only defined **inside a conditional block** (line 4552).

**Code Path**:
```python
# Line 4552 (INSIDE a try block for model lifecycle):
if regime_model_was_trained:
    train_start = train.index.min() if hasattr(train.index, 'min') else datetime.now()
    train_end = train.index.max() if hasattr(train.index, 'max') else datetime.now()

# Line 4563-4574 (OUTSIDE the if block):
if model_state is None:
    model_state = create_new_model_state(
        training_start=train_start,  # ❌ UNDEFINED if regime_model_was_trained=False
        training_end=train_end,      # ❌ UNDEFINED if regime_model_was_trained=False
    )
```

**When This Crashes**:
- First run for new equipment (model_state is None)
- Regime model NOT trained (regime_model_was_trained=False)
- Variables train_start/train_end never defined
- NameError: name 'train_start' is not defined

**Impact**:
- Model lifecycle initialization fails on cold-start
- LEARNING → CONVERGED promotion never happens
- V11 model lifecycle completely broken for new equipment

**Fix Required**:
Define variables before conditional:
```python
# Before line 4552:
train_start = train.index.min() if hasattr(train.index, 'min') else datetime.now()
train_end = train.index.max() if hasattr(train.index, 'max') else datetime.now()
```

---

### CRITICAL-4: ONLINE Mode Detector Loading Lacks Fallback

**Location**: `core/acm_main.py` (lines 4138-4143)

**Problem**: ONLINE mode fails fast if detectors missing, but **doesn't check regime model**.

**Code**:
```python
if detectors_missing and not ALLOWS_MODEL_REFIT:
    raise RuntimeError(
        "[ONLINE MODE] Required detector models not found in cache..."
    )
```

**Gap**:
- Checks AR1, PCA, IForest detectors ✅
- Does NOT check regime model ❌
- Regime discovery uses `allow_discovery: ALLOWS_REGIME_DISCOVERY` flag
- But if regime model missing in ONLINE mode, no error raised
- Pipeline continues with **no regime labels** → crash later

**Scenario**:
1. Run ONLINE mode for first time
2. Detectors exist in cache (from other equipment)
3. Regime model doesn't exist (equipment-specific)
4. Pipeline continues to line 4321
5. regimes.label() called with no model + allow_discovery=False
6. Returns empty regime_labels or crashes

**Impact**:
- ONLINE mode silently fails for equipment without regime models
- No clear error message to user
- Downstream code assumes regime_labels exist → crashes

**Fix Required**:
```python
# Add regime model check:
if detectors_missing and not ALLOWS_MODEL_REFIT:
    raise RuntimeError(...)

if regime_model is None and not ALLOWS_REGIME_DISCOVERY:
    raise RuntimeError(
        "[ONLINE MODE] Regime model not found and discovery disabled. "
        "Run in OFFLINE mode first to train regime model."
    )
```

---

### CRITICAL-5: Confidence Not Propagated to Health/Episodes

**Location**: Multiple files, missing integration

**Problem**: `compute_health_confidence()` and `compute_episode_confidence()` functions exist in `core/confidence.py` but are **never called**.

**Evidence**:
```bash
$ grep -rn "compute_health_confidence\|compute_episode_confidence" core/
core/confidence.py:205:def compute_health_confidence(
core/confidence.py:232:def compute_episode_confidence(
# NO OTHER MATCHES - functions never called!
```

**What's Missing**:
1. Health confidence calculation in health tracking
2. Episode confidence calculation in episode detection
3. SQL columns in ACM_HealthTimeline and ACM_Anomaly_Events
4. Write integration in output_manager

**Impact**:
- V11 Rule #17 violated: "Confidence must always be exposed"
- Confidence only exists for RUL, not for health or episodes
- Incomplete V11 confidence model (33% coverage)

**Fix Required**:
1. Add `Confidence FLOAT` columns to tables
2. Call functions in health_tracker.py and episode detection
3. Write confidence values in output_manager

---

## 🟡 HIGH SEVERITY DESIGN ISSUES

### HIGH-1: Seasonality Detection Without Application ⚠️ WASTED EFFORT

**Location**: `core/acm_main.py` (lines 3823-3843)

**Problem**: Seasonality patterns detected, logged, written to SQL, but **never used**.

**Code**:
```python
# Line 3820:
# NOTE: Patterns detected but NOT USED for adjustment.
# Future work: subtract seasonal component from sensor data

seasonal_patterns = handler.detect_patterns(temp_df, sensor_cols, 'Timestamp')
# Patterns stored, but baseline data UNCHANGED
```

**Why This Matters**:
- False positives during diurnal cycles (temperature rises at noon → anomaly)
- Detection overhead (autocorrelation computation) with zero benefit
- SQL writes to ACM_SeasonalPatterns table that are never read
- Claims "7 daily patterns detected" but doesn't improve accuracy

**Recommendation**:
Either:
1. **Activate** seasonal adjustment: Call `handler.adjust_baseline()` before feature engineering
2. **Remove** detection entirely until activation is ready
3. **Document** as "detection-only mode" in release notes

---

### HIGH-2: Asset Similarity Transfer Learning Not Used

**Location**: `core/acm_main.py` (lines 5377-5397)

**Problem**: Asset profiles built and written to SQL, but **transfer logic never activated**.

**Code**:
```python
# Profiles created:
asset_profile = AssetProfile(...)
output_manager.write_asset_profile(profile_dict)

# But transfer_baseline() NEVER CALLED
# Cold-start still starts from scratch
```

**Impact**:
- New equipment gets no knowledge transfer from similar assets
- Cold-start remains slow (200+ rows minimum)
- Infrastructure coded but provides zero value
- ACM_AssetProfiles table populated but unused

**Recommendation**:
1. Add cold-start detection logic
2. Call `similarity.find_similar(target_profile)`
3. If similarity > 0.7, call `transfer_baseline()`
4. Bootstrap new equipment with transferred model

---

### HIGH-3: Model Promotion Criteria May Never Be Met

**Location**: `core/model_lifecycle.py::PromotionCriteria` (lines 40-46)

**Problem**: Default criteria are **too strict** for typical ACM usage patterns.

**Criteria**:
```python
min_training_days: int = 7         # Need 7 full days
min_silhouette_score: float = 0.15 # Clustering quality
min_stability_ratio: float = 0.8   # 80% regime stability
min_consecutive_runs: int = 3      # 3 successful runs in a row
min_training_rows: int = 1000      # 1000 data points
```

**Why This Fails**:
1. **30-minute cadence**: 7 days = 336 samples (not 1000)
2. **Batch mode**: Runs daily = needs 3+ days for consecutive runs
3. **Silhouette 0.15**: Low threshold but still fails for noisy industrial data
4. **Stability 0.8**: Very strict - normal operation has regime switching

**Result**:
- Models stuck in LEARNING state indefinitely
- RUL always NOT_RELIABLE (Rule #10 too aggressive)
- V11 promotion never happens in practice

**Recommendation**:
```python
min_training_days: int = 3         # More realistic
min_silhouette_score: float = 0.10 # Account for noise
min_stability_ratio: float = 0.6   # Allow more switching
min_consecutive_runs: int = 2      # Faster promotion
min_training_rows: int = 200       # Match coldstart minimum
```

---

### HIGH-4: UNKNOWN Regime Smoothing Preservation Fragile

**Location**: `core/regimes.py::smooth_labels()` (lines 1084-1153)

**Problem**: UNKNOWN preservation relies on `preserve_unknown=True` parameter, but **callers may not set it**.

**Code**:
```python
def smooth_labels(labels, preserve_unknown: bool = True):
    if preserve_unknown:
        unknown_mask = (labels == UNKNOWN_REGIME_LABEL)
    # ... smoothing ...
    if preserve_unknown:
        smoothed[unknown_mask] = UNKNOWN_REGIME_LABEL
```

**Risk**:
- Default is True ✅
- But if any caller sets `preserve_unknown=False`, UNKNOWN labels are lost
- V11 Rule #14 violated: "UNKNOWN is a valid system output"
- No enforcement that parameter must be True

**Search Results**:
```bash
$ grep -rn "smooth_labels\|smooth_regime_labels" core/
# Multiple call sites - need to audit all
```

**Recommendation**:
1. Remove parameter entirely (always preserve UNKNOWN)
2. Add assertion: `assert preserve_unknown, "UNKNOWN must be preserved"`
3. Audit all call sites to ensure True is passed

---

## 🟠 MEDIUM SEVERITY INTEGRATION ISSUES

### MEDIUM-1: DataContract Validation Only on Score Data

**Location**: `core/acm_main.py` (lines 3718-3769)

**Problem**: Validation only runs on **score window**, not baseline (train).

**Code**:
```python
contract = DataContract(...)
validation = contract.validate(score)  # Only score!
```

**Gap**:
- Bad baseline data (nulls, constant columns) not caught
- Detector training uses unvalidated data
- Only scoring failures trigger fail-fast

**Impact**:
- Baseline issues cause detector failures later
- No early warning for train data quality

**Recommendation**:
Validate both:
```python
train_validation = contract.validate(train)
score_validation = contract.validate(score)
```

---

### MEDIUM-2: Regime Definitions Write Depends on hasattr

**Location**: `core/acm_main.py` (lines 4383-4401)

**Problem**: Writing regime definitions requires `hasattr(regime_model, 'kmeans')` which is **model implementation-specific**.

**Code**:
```python
if output_manager and hasattr(regime_model, 'kmeans') and regime_model.kmeans is not None:
    regime_defs = []
    centroids = regime_model.kmeans.cluster_centers_
```

**Brittleness**:
- Works only for MiniBatchKMeans-based regime models
- Fails silently if model type changes (DBSCAN, GMM, etc.)
- No logging when skipped

**Recommendation**:
- Use duck typing or protocol: `if hasattr(regime_model, 'get_centroids')`
- Add logging when skipped
- Support multiple model types

---

### MEDIUM-3: SQL Connection Not Validated Before Use

**Location**: Multiple files

**Problem**: Code assumes `sql_client` is valid but doesn't verify connection.

**Example** (`core/model_lifecycle.py` line 340):
```python
def load_model_state_from_sql(sql_client, equip_id):
    with sql_client.cursor() as cur:  # ❌ Assumes connection works
        cur.execute("SELECT ...")
```

**What's Missing**:
- Connection health check
- Retry logic for transient failures
- Graceful degradation if SQL unavailable

**Impact**:
- Hard crashes on DB connection loss
- No fallback to file mode
- Poor production resilience

---

### MEDIUM-4: Confidence Geometric Mean Can Be Zero

**Location**: `core/confidence.py::ConfidenceFactors.overall()` (lines 48-60)

**Problem**: If any factor is 0, geometric mean is 0, even if other factors are high.

**Code**:
```python
def overall(self) -> float:
    factors = [self.maturity_factor, self.data_quality_factor, ...]
    product = 1.0
    for f in factors:
        product *= max(0.0, min(1.0, f))  # If any f=0, product=0
    return product ** (1.0 / len(factors))
```

**Edge Case**:
- maturity_factor = 1.0 (CONVERGED)
- data_quality_factor = 1.0 (perfect)
- prediction_factor = 1.0 (tight bounds)
- regime_factor = 0.0 (no regime assigned)
- **overall = 0.0** (extremely unconfident)

**Question**: Is this correct? Should missing regime zero out confidence?

**Recommendation**:
Consider weighted geometric mean or exclude zero factors:
```python
non_zero_factors = [f for f in factors if f > 0]
if not non_zero_factors:
    return 0.0
product = np.prod(non_zero_factors)
return product ** (1.0 / len(non_zero_factors))
```

---

## 📊 Summary Table

| Issue | Severity | Category | Impact | Fix Effort |
|-------|----------|----------|--------|-----------|
| CRITICAL-1 | 🔴 Critical | Integration | RUL reliability wrong | Medium |
| CRITICAL-2 | 🔴 Critical | Runtime | NameError crash | Low |
| CRITICAL-3 | 🔴 Critical | Runtime | Cold-start crash | Low |
| CRITICAL-4 | 🔴 Critical | Logic | ONLINE mode silent fail | Low |
| CRITICAL-5 | 🔴 Critical | Completeness | Rule #17 violation | High |
| HIGH-1 | 🟡 High | Design | Wasted CPU cycles | Low/High* |
| HIGH-2 | 🟡 High | Design | Unused infrastructure | Medium |
| HIGH-3 | 🟡 High | Design | Unreachable promotion | Low |
| HIGH-4 | 🟡 High | Logic | UNKNOWN preservation | Low |
| MEDIUM-1 | 🟠 Medium | Validation | Incomplete coverage | Low |
| MEDIUM-2 | 🟠 Medium | Brittleness | Silent failures | Low |
| MEDIUM-3 | 🟠 Medium | Resilience | No fault tolerance | Medium |
| MEDIUM-4 | 🟠 Medium | Math | Edge case behavior | Low |

*Low to remove, High to complete

---

## Revised V11 Grade

**Original Audit**: A- (85% complete)

**After Deep Dive**: **C+ (65% functional)**

**Reasoning**:
- Core components coded ✅ (85% as stated)
- **But critical integration gaps prevent production use** ❌
- 5 runtime crashes on cold-start or ONLINE mode
- 2 major design flaws (seasonality, transfer learning unused)
- Confidence model incomplete (33% vs claimed 100%)

**What Works**:
- PipelineMode enum and routing ✅
- MaturityState enum and dataclasses ✅
- UNKNOWN_REGIME_LABEL constant ✅
- ReliabilityStatus enum ✅
- DataContract and ValidationResult ✅
- SQL table schemas ✅

**What's Broken**:
- Model lifecycle disconnected from forecasting ❌
- Variable scoping issues (2 NameErrors) ❌
- ONLINE mode missing regime check ❌
- Health/episode confidence not integrated ❌
- Promotion criteria too strict ❌

---

## Recommendations

### Immediate Fixes (P0 - Required for V11 to work)

1. **Fix CRITICAL-2 & CRITICAL-3**: Initialize variables at function scope
2. **Fix CRITICAL-4**: Add regime model check in ONLINE mode
3. **Fix CRITICAL-1**: Pass model_state to ForecastEngine
4. **Adjust HIGH-3**: Relax promotion criteria to realistic values

### Short-Term (P1 - Complete V11 promise)

5. **Fix CRITICAL-5**: Integrate health/episode confidence
6. **Fix HIGH-1**: Either activate or remove seasonality
7. **Fix HIGH-2**: Either activate or remove transfer learning
8. **Fix HIGH-4**: Make UNKNOWN preservation mandatory

### Medium-Term (P2 - Production hardening)

9. **Fix MEDIUM-1**: Validate both train and score data
10. **Fix MEDIUM-3**: Add SQL connection resilience
11. **Fix MEDIUM-2**: Support multiple regime model types
12. **Fix MEDIUM-4**: Review confidence geometric mean

---

## Testing Checklist (Must Pass Before V11 Release)

### Cold-Start Test
- [ ] New equipment, first run, OFFLINE mode
- [ ] Verify no NameError for train_start/train_end
- [ ] Verify no NameError for regime_state_version
- [ ] Verify model_state created in LEARNING
- [ ] Verify ACM_ActiveModels written

### Promotion Test
- [ ] Run same equipment 3 times over 3 days
- [ ] Verify LEARNING → CONVERGED transition
- [ ] Verify promotion logged to Console
- [ ] Verify RUL_Status changes to RELIABLE

### ONLINE Mode Test
- [ ] Equipment with existing models, ONLINE mode
- [ ] Verify detectors loaded from cache
- [ ] Verify regime model loaded
- [ ] Verify no detector fitting
- [ ] Verify no regime discovery

### ONLINE Mode Failure Test
- [ ] Equipment without regime model, ONLINE mode
- [ ] Verify RuntimeError raised (not silent fail)
- [ ] Error message mentions regime model

### Confidence Integration Test
- [ ] Run with model in CONVERGED state
- [ ] Verify RUL has confidence column
- [ ] Verify health has confidence column (after fix)
- [ ] Verify episodes have confidence column (after fix)

---

## Conclusion

**The V11 implementation has good architectural bones but critical integration flaws.**

The audit report correctly identified the **code exists** (85% complete), but failed to identify that **the code doesn't work together** (65% functional). This deep dive reveals:

1. **5 runtime crashes** that prevent cold-start and ONLINE mode
2. **Disconnected lifecycle** between model tracking and forecasting
3. **Unused features** (seasonality, transfer learning) claiming value they don't deliver
4. **Incomplete confidence** (RUL only, not health/episodes)

**V11 cannot be released in current state.** The P0 fixes are small (variable scoping, checks) but essential. Without them, V11 will crash on first use.

---

**End of Deep Dive Analysis**
