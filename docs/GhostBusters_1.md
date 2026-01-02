# GhostBusters Phase 1: Silent Analytical Corruption (Risk #1)

## Executive Summary

The pipeline can complete with `outcome="OK"` while producing **analytically invalid results**. 

---

## ✅ FIXED in v11.2.0 (Commit 72dc189)

### Critical Phases Now Fail-Fast

| Phase | Before | After | Status |
|-------|--------|-------|--------|
| `features.build` | `safe_step()` wrapper | **No try/except** - crashes on failure | ✅ FIXED |
| `features.impute` | `safe_step()` wrapper | **No try/except** - crashes on failure | ✅ FIXED |
| `persist.write_scores` | `safe_step()` wrapper | **No try/except** - crashes on failure | ✅ FIXED |
| `persist.write_episodes` | `safe_step()` wrapper | **No try/except** - crashes on failure | ✅ FIXED |
| `load_data` | Already critical | Already critical | ✅ Already OK |
| `train.detector_fit` | Already critical | Already critical | ✅ Already OK |
| `score.detector_score` | Already critical | Already critical | ✅ Already OK |
| `fusion` | Already critical | Already critical | ✅ Already OK |

### String Sentinel Replaced

| Pattern | Before | After | Status |
|---------|--------|-------|--------|
| Regime state marker | `regime_model = "STATE_LOADED"` | `regime_loaded_from_state = True` | ✅ FIXED |
| Regime check | `if regime_model == "STATE_LOADED"` | `if regime_loaded_from_state` | ✅ FIXED |

### Regime Basis Build

| Change | Status |
|--------|--------|
| Inline try/except with explicit `degradations.append()` | ✅ FIXED |
| Clear error message with truncated exception | ✅ FIXED |
| Proper variable initialization before try block | ✅ FIXED |

---

## ❌ NOT NEEDED (Over-Engineering)

These items from the original plan are **NOT needed**:

| Item | Why Not Needed |
|------|----------------|
| `PhaseCriticality` enum | Just don't catch exceptions in critical phases |
| `PhaseResult` dataclass | Phases either succeed or crash - no complex state |
| `run_phase()` wrapper | Remove try/except instead of wrapping it |
| `PipelineState` dataclass | Existing variables work fine |
| `RegimeModelContext` dataclass | Boolean flag is simpler and type-safe |
| `FRAME_CONTRACTS` dict | Detectors already validate their inputs |
| Frame mutation tracking | Over-engineering for minimal benefit |

---

## 🔧 REMAINING SIMPLE FIXES
## 🔧 REMAINING SIMPLE FIXES

### 1. Remove `safe_step()` Function Entirely

The function is no longer called anywhere. Delete it.

**File**: `core/acm_main.py` lines ~547-600
**Action**: Delete the `safe_step()` function definition

---

### 2. Clean Up Bare `except Exception:` Without Binding

These swallow errors without even logging them:

| Line | Context | Simple Fix |
|------|---------|------------|
| ~44 | `from core import fast_features` | OK - import fallback |
| ~150 | `from core.sql_client import SQLClient` | OK - import fallback |
| ~163 | `from utils.timer import Timer` | OK - import fallback |
| ~764 | Run count query | OK - defaults to 0 |
| ~1296 | Regime label lookup | OK - cosmetic fallback |

**Verdict**: These are all acceptable import/query fallbacks. No action needed.

---

### 3. Audit `Console.warn()` After Failures

Some warnings allow invalid state to propagate. Review each:

| Location | Warning | Should Crash Instead? |
|----------|---------|----------------------|
| Regime state load failure | "Failed to load regime state" | No - new equipment has no state |
| Regime reconstruction failure | "Failed to reconstruct model from state" | No - will retrain |
| Calibration failure | "Calibration failed" | **MAYBE** - investigate |
| Threshold update failure | Various | **MAYBE** - investigate |

**Action**: Review calibration and threshold phases for criticality.

---

### 4. Delete Unused `safe_step()` Function

Since all calls are removed, delete the function definition:

```python
# DELETE THIS (~lines 547-600):
def safe_step(
    fn: Callable[[], T],
    step_name: str,
    ...
) -> Tuple[Optional[T], bool]:
    ...
```

---

## Next Simple Fixes to Consider

### A. Make `calibrate` Phase Critical (Simple)

Calibration affects threshold accuracy. If it fails, z-scores may be wrong.

**Current**: Has try/except with warning
**Fix**: Remove try/except, let it crash

### B. Make `data.contract` Phase Critical (Simple)

If data contract validation fails, the run should abort.

**Current**: Returns early on failure (already correct!)
**Status**: ✅ Already OK

### C. Review Forecasting Error Handling

Forecasting is DEGRADED (not CRITICAL), but failures should be tracked.

**Current**: Has complex try/except
**Fix**: Ensure `degradations.append("forecasting")` on any failure

---

## Summary

| Category | Original Plan Items | Kept | Discarded |
|----------|---------------------|------|-----------|
| Enums/Dataclasses | 4 | 0 | 4 |
| Wrapper Functions | 2 | 0 | 2 |
| Critical Phase Fixes | 8 | 8 | 0 |
| Sentinel Cleanup | 1 | 1 | 0 |
| Simple Audits | 3 | 3 | 0 |

**Approach**: Remove try/except from critical phases. Don't add infrastructure.
