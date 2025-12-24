# 🔧 ACM Main Refactoring Analysis - Actionable Guide

## Executive Summary
**Current State**: `main()` function is **2,500+ lines** - a monolithic pipeline  
**Target State**: Extract **8-10 logical phase functions** + cleanup dead code  
**Impact**: 70% code reduction in main(), improved testability, better error isolation

---

## 📊 REFACTORING PROGRESS

### Wave 1: Dead Code Removal ✅ COMPLETE (Dec 24, 2025)
| Task | Status | Commit |
|------|--------|--------|
| Delete `_apply_module_overrides()` | ✅ Done | 4b3f3db |
| Delete `_ensure_dir()` | ✅ Done | 4b3f3db |
| Inline `_sql_mode()` | ✅ Done | 4b3f3db |
| Inline `_batch_mode()` | ✅ Done | 4b3f3db |

### Wave 2: File-Mode Branch Removal ✅ COMPLETE (Dec 24, 2025)
| Task | Status | Lines Removed |
|------|--------|---------------|
| Remove `if not SQL_MODE:` branches | ✅ Done | 18 branches |
| Remove file-mode CSV writes | ✅ Done | ~150 lines |
| Remove `refit_flag_path` | ✅ Done | ~20 lines |
| Remove `file_mode_enabled` | ✅ Done | ~10 lines |
| Simplify persist section | ✅ Done | ~50 lines |

**Wave 1+2 Results**: 4,663 → 4,328 lines (**335 lines removed, 7.2% reduction**)

### Wave 3: Helper Function Extraction 🔄 IN PROGRESS
| Helper Function | Purpose | Status |
|-----------------|---------|--------|
| Context dataclasses | RuntimeContext, DataContext, FeatureContext, ModelContext, ScoreContext, FusionContext | ✅ Done (b6fa58e) |
| `_score_all_detectors()` | Score data with all fitted detectors | ✅ Done (d9401f7, 1004d8e) |
| `_calibrate_all_detectors()` | Calibrate detector outputs to z-scores | ✅ Done (7d8f410) |
| `_fit_all_detectors()` | Fit all enabled detectors on train data | ✅ Done (d16d2a8) |
| `_get_detector_enable_flags()` | Get detector enable flags from fusion weights | ✅ Done (ab8f1d9) |
| `_deduplicate_index()` | Remove duplicate timestamps from DataFrame index | ✅ Done (86580f7) |
| `_rebuild_detectors_from_cache()` | Reconstruct detector objects from cached model data | ✅ Done (42ea199) |
| `_update_baseline_buffer()` | Update ACM_BaselineBuffer with latest score data | ✅ Done (3ba69df) |
| `_compute_stable_feature_hash()` | Compute stable hash for training features | ✅ Done (e829f35) |
| `_check_refit_request()` | Check for pending refit requests in SQL | ✅ Done (1d591c6) |
| `_phase_initialize_runtime()` | 840-1183 (~343 lines) | 🔲 Not Started |
| `_phase_load_data()` | 1183-1915 (~732 lines) | 🔲 Not Started |
| `_phase_fit_models()` | 1915-2311 (~396 lines) | 🔲 Not Started |
| `_phase_label_regimes()` | 2333-2595 (~262 lines) | 🔲 Not Started |
| `_phase_calibrate()` | 2595-2769 (~174 lines) | 🔲 Not Started |
| `_phase_fuse_and_episodes()` | 2769-3419 (~650 lines) | 🔲 Not Started |
| `_phase_persist_results()` | 3771-end (~490 lines) | 🔲 Not Started |

**Wave 3 Progress**: Added context dataclasses + extracted **10 helper functions**:
- `_score_all_detectors()` - Score data through all detectors
- `_calibrate_all_detectors()` - Calibrate detector outputs
- `_fit_all_detectors()` - Fit all detectors on training data
- `_get_detector_enable_flags()` - Get enable flags from fusion weights
- `_deduplicate_index()` - Remove duplicate timestamps
- `_rebuild_detectors_from_cache()` - Reconstruct detectors from cached models
- `_update_baseline_buffer()` - Update baseline buffer with vectorized SQL writes
- `_compute_stable_feature_hash()` - Stable cross-platform hash for training data
- `_check_refit_request()` - Check and acknowledge SQL refit requests

### Wave 4: Pattern Improvements 🔲 PLANNED
| Pattern | Status |
|---------|--------|
| `@safe_section` decorator | 🔲 Not Started |
| `ConfigAccessor` class | 🔲 Not Started |

**Current Line Count**: 4,557 lines (original: 4,663; helpers add testable structure)

---

## 🎯 PRIORITY 1: Extract Core Pipeline Phases (70% Impact)

### Current Problem
The `main()` function contains the entire ACM pipeline in one massive function with 13 interleaved stages.

### Solution: Extract 8 Phase Functions

```python
# TARGET STRUCTURE (show this to Copilot)

def main() -> None:
    """Orchestrator only - delegates to phase functions"""
    
    # 1. Startup
    runtime_ctx = initialize_runtime(args, cfg)  # Lines 500-700
    
    # 2. Data
    data_ctx = load_and_validate_data(runtime_ctx, cfg)  # Lines 700-1100
    
    # 3. Models
    model_ctx = train_or_load_models(data_ctx, cfg)  # Lines 1100-1500
    
    # 4. Scoring
    score_ctx = score_detectors(data_ctx, model_ctx, cfg)  # Lines 1500-1700
    
    # 5. Regimes
    regime_ctx = label_regimes(score_ctx, model_ctx, cfg)  # Lines 1700-1900
    
    # 6. Fusion
    fusion_ctx = fuse_and_detect_episodes(score_ctx, regime_ctx, cfg)  # Lines 1900-2400
    
    # 7. Analytics
    persist_results(fusion_ctx, runtime_ctx, cfg)  # Lines 2700-3100
    
    # 8. Finalize
    finalize_run(runtime_ctx, fusion_ctx, cfg)  # Lines 3100-end
```

**Each phase function should:**
- Take clear input context (NamedTuple/dataclass)
- Return output context for next phase
- Have one `try-except` at phase boundary
- Be independently testable

---

## 🗑️ PRIORITY 2: Remove Dead Code (20% Impact) ✅ COMPLETE

### ~~Still Present - REMOVE THESE~~ ALL REMOVED:

| Line | Function/Code | Why Dead | Status |
|------|---------------|----------|--------|
| ~~256-258~~ | `_apply_module_overrides()` | Just `pass` | ✅ DELETED |
| ~~638-639~~ | `_ensure_dir(p)` | Wraps `mkdir()` | ✅ DELETED |
| ~~661-667~~ | `_sql_mode(cfg)` | Checks one env var | ✅ INLINED |
| ~~668-670~~ | `_batch_mode()` | One-liner | ✅ INLINED |
| ~~264-279~~ | `_maybe_write_run_meta_json` | File-mode only | ✅ DELETED |
| N/A | All `if not SQL_MODE:` branches | 18 branches | ✅ REMOVED |
| N/A | `refit_flag_path` variable | File-mode only | ✅ REMOVED |
| N/A | `file_mode_enabled` variable | Dead code | ✅ REMOVED |

---

## 🔧 PRIORITY 3: Extract SQL Operations (10% Impact)

### Problem: SQL writes are inline throughout main()

Lines with inline SQL that should be extracted:

| Lines | Operation | Extract To |
|-------|-----------|------------|
| 2850-2900 | Write health timeline | `_write_health_timeline(output_mgr, frame, cfg)` |
| 2900-2950 | Write regime timeline | `_write_regime_timeline(output_mgr, frame, cfg)` |
| 2400-2500 | Update baseline buffer | `_update_baseline_buffer(sql_client, score_numeric, cfg)` |
| 3150-3200 | Write episode culprits | Already extracted ✅ |

---

## 📦 RECOMMENDED CONTEXT CLASSES

### Define These Once, Use Throughout:

```python
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class RuntimeContext:
    """Startup phase output"""
    equip: str
    equip_id: int
    run_id: str
    sql_client: Optional[Any]
    output_manager: OutputManager
    cfg: Dict[str, Any]
    SQL_MODE: bool
    BATCH_MODE: bool

@dataclass
class DataContext:
    """Data loading phase output"""
    train: pd.DataFrame
    score: pd.DataFrame
    train_numeric: pd.DataFrame
    score_numeric: pd.DataFrame
    meta: Any

@dataclass
class ModelContext:
    """Model training/loading phase output"""
    ar1_detector: Optional[Any]
    pca_detector: Optional[Any]
    iforest_detector: Optional[Any]
    gmm_detector: Optional[Any]
    omr_detector: Optional[Any]
    regime_model: Optional[Any]
    models_fitted: bool

@dataclass
class ScoreContext:
    """Scoring phase output"""
    frame: pd.DataFrame  # Contains all z-scores
    train_frame: pd.DataFrame
    calibrators: Dict[str, Any]

@dataclass
class FusionContext:
    """Fusion phase output"""
    frame: pd.DataFrame  # With fused scores
    episodes: pd.DataFrame
    fusion_weights: Dict[str, float]
```

---

## 🎨 REFACTORING PATTERNS

### Pattern 1: Extract Phase Function

```python
# BEFORE: Lines 700-1100 in main()
with T.section("load_data"):
    if SQL_MODE:
        # 200 lines of SQL loading logic
    else:
        # 100 lines of file loading logic
    # 50 lines of validation
    # 100 lines of preprocessing

# AFTER:
def load_and_validate_data(runtime_ctx: RuntimeContext, cfg: Dict) -> DataContext:
    """Phase 2: Load and validate input data
    
    Returns:
        DataContext with train/score DataFrames and metadata
    
    Raises:
        RuntimeError: If data quality checks fail
    """
    with runtime_ctx.timer.section("load_data"):
        if runtime_ctx.SQL_MODE:
            train, score, meta = _load_from_sql(runtime_ctx, cfg)
        else:
            train, score, meta = _load_from_csv(runtime_ctx, cfg)
        
        _validate_data_quality(train, score, meta, cfg)
        train, score = _preprocess_data(train, score, cfg)
        
        return DataContext(
            train=train,
            score=score,
            train_numeric=train.copy(),
            score_numeric=score.copy(),
            meta=meta
        )
```

### Pattern 2: Extract Repeated Try-Except

```python
# BEFORE: Repeated 30+ times in main()
with T.section("some_operation"):
    try:
        # operation code
        Console.info(f"Success: {result}", component="OP")
    except Exception as e:
        Console.warn(f"Failed: {e}", component="OP",
                     equip=equip, error=str(e)[:200])

# AFTER: Create decorator
def safe_section(section_name: str, component: str = "PIPELINE"):
    """Decorator for consistent error handling in pipeline sections"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ctx = args[0] if args else None
            equip = getattr(ctx, 'equip', 'UNKNOWN')
            
            with Timer().section(section_name):
                try:
                    result = func(*args, **kwargs)
                    Console.info(f"{section_name} completed", component=component)
                    return result
                except Exception as e:
                    Console.warn(
                        f"{section_name} failed: {e}",
                        component=component,
                        equip=equip,
                        error_type=type(e).__name__,
                        error=str(e)[:200]
                    )
                    raise
        return wrapper
    return decorator

# USAGE:
@safe_section("load_data", component="DATA")
def load_and_validate_data(runtime_ctx, cfg):
    # No try-except needed!
    return DataContext(...)
```

### Pattern 3: Extract Config Access

```python
# BEFORE: Scattered 100+ times
drift_cfg = (cfg.get("drift", {}) or {})
multi_feat_cfg = drift_cfg.get("multi_feature", {})
enabled = bool(multi_feat_cfg.get("enabled", False))

# AFTER: Create config accessor
class ConfigAccessor:
    """Type-safe config access with defaults"""
    def __init__(self, cfg: Dict):
        self._cfg = cfg
    
    def drift_multi_feature_enabled(self) -> bool:
        return bool(
            self._cfg.get("drift", {})
            .get("multi_feature", {})
            .get("enabled", False)
        )
    
    def fusion_weights(self) -> Dict[str, float]:
        return self._cfg.get("fusion", {}).get("weights", {
            "pca_spe_z": 0.30,
            "pca_t2_z": 0.20,
            # ... defaults
        })

# USAGE:
cfg_accessor = ConfigAccessor(cfg)
if cfg_accessor.drift_multi_feature_enabled():
    # ...
```

---

## 📋 STEP-BY-STEP IMPLEMENTATION PLAN

### Phase 1: Cleanup Dead Code ✅ COMPLETE (1 hour)
1. ✅ Delete `_apply_module_overrides()` (line 256-258)
2. ✅ Inline `_ensure_dir()` calls (search for `_ensure_dir(` - ~15 occurrences)
3. ✅ Inline `_sql_mode()` → `SQL_MODE = os.getenv("ACM_FORCE_FILE_MODE") != "1"`
4. ✅ Inline `_batch_mode()` → `BATCH_MODE = bool(os.getenv("ACM_BATCH_MODE") == "1")`
5. ✅ Delete all `if not SQL_MODE:` branches (18 removed)

### Phase 2: Extract Data Loading 🔄 NEXT (2 hours)
1. Create `DataContext` dataclass
2. Extract lines 700-1100 to `_phase_load_data()`
3. Extract SQL loading to `_load_from_sql()`
4. Test data loading in isolation

### Phase 3: Extract Model Training (3 hours)
1. Create `ModelContext` dataclass
2. Extract lines 1100-1500 to `_phase_fit_models()`
3. Extract cache validation to `_validate_model_cache()`
4. Extract detector fitting to `_fit_detectors()`
5. Test model training independently

### Phase 4: Extract Remaining Phases (4 hours)
1. Extract scoring (lines 1500-1700)
2. Extract regime labeling (lines 1700-1900)
3. Extract fusion (lines 1900-2400)
4. Extract persistence (lines 2700-3100)

### Phase 5: Create Safe Section Decorator (1 hour)
1. Implement `@safe_section` decorator
2. Apply to all phase functions
3. Remove redundant try-except blocks

---

## 🎯 SUCCESS METRICS

After refactoring:

| Metric | Before | Target | How to Measure |
|--------|--------|--------|----------------|
| `main()` length | 2,500 lines | <300 lines | `wc -l acm_main.py` |
| Cyclomatic complexity | >50 | <10 | `radon cc acm_main.py` |
| Test coverage | ~30% | >80% | Can mock phase functions |
| Git blame lines | >500 touches | Isolated to phases | `git blame` |

---

## 🚀 COPILOT PROMPT TEMPLATE

Use this to start each phase:

```
I need to refactor acm_main.py by extracting the [DATA LOADING / MODEL TRAINING / etc] phase.

CURRENT STATE:
- Lines [START]-[END] in main() contain [DESCRIPTION]
- This code is [MIXED WITH / DEPENDS ON] [OTHER CONCERNS]

TARGET STATE:
- Extract to new function `[FUNCTION_NAME](ctx: [CONTEXT_TYPE], cfg: Dict) -> [RETURN_TYPE]`
- Function should [SINGLE RESPONSIBILITY]
- Use context class [CONTEXT_NAME] for inputs/outputs

CONTEXT DEFINITION:
[Paste relevant @dataclass from above]

Please:
1. Extract the function with proper type hints
2. Update main() to call the new function
3. Add docstring with Args/Returns/Raises
4. Keep existing error handling patterns

Current code section:
[Paste lines START-END]
```

---

## 🔍 VALIDATION CHECKLIST

Before marking refactoring complete:

- [ ] `main()` is under 300 lines
- [ ] Each phase function is under 200 lines
- [ ] All phase functions have type hints
- [ ] Dead code deleted (grep for `_ensure_dir`, `_apply_module_overrides`)
- [ ] SQL operations extracted from inline
- [ ] Try-except blocks use `@safe_section` decorator
- [ ] Config access uses `ConfigAccessor` class
- [ ] All phases independently testable
- [ ] Git diff shows clean separation of concerns

---

## 💡 EXAMPLE: Before/After Main()

### BEFORE (Current - 2,500 lines):
```python
def main() -> None:
    # Parse args (50 lines)
    # Init observability (30 lines)
    # Load config (40 lines)
    # Connect SQL (50 lines)
    # Load data (400 lines)
    # Build features (200 lines)
    # Train models (400 lines)
    # Score detectors (200 lines)
    # ... (1,000 more lines)
```

### AFTER (Target - 250 lines):
```python
def main() -> None:
    """ACM pipeline orchestrator - delegates to phase functions"""
    try:
        # Startup (30 lines)
        runtime_ctx = initialize_runtime(parse_args(), load_config())
        
        # Pipeline phases (100 lines total)
        data_ctx = load_and_validate_data(runtime_ctx)
        model_ctx = train_or_load_models(data_ctx, runtime_ctx)
        score_ctx = score_detectors(data_ctx, model_ctx, runtime_ctx)
        regime_ctx = label_regimes(score_ctx, model_ctx, runtime_ctx)
        fusion_ctx = fuse_and_detect_episodes(score_ctx, regime_ctx, runtime_ctx)
        
        # Outputs (50 lines)
        persist_results(fusion_ctx, runtime_ctx)
        
    except Exception as e:
        handle_pipeline_failure(e, runtime_ctx)
    finally:
        finalize_run(runtime_ctx)
```

---

**Start with Phase 1 (cleanup) - it's quickest win and unblocks the rest!**