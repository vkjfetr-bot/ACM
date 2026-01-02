# Remediation Plan: Silent Analytical Corruption (Risk #1)

## Executive Summary

The pipeline can complete with `outcome="OK"` or `outcome="DEGRADED"` while producing **analytically invalid results**. This happens because:

1. **36 pipeline phases** have no formal criticality classification
2. **15 bare `except: pass` blocks** silently discard errors
3. **20+ `Console.warn()` handlers** allow execution to continue after failures
4. **Only 5 calls to `safe_step()`** with proper degradation tracking
5. **No phase contracts** - downstream phases don't validate upstream preconditions

---

## Phase 1: Define Phase Criticality Model

### 1.1 Create `PhaseCriticality` Enum

```python
class PhaseCriticality(Enum):
    """Defines how phase failures should be handled.
    
    CRITICAL: Pipeline MUST abort. Downstream results would be invalid.
    REQUIRED: Phase must produce output. Fallback to safe defaults allowed.
    DEGRADED: Failure reduces quality but doesn't invalidate results.
    OPTIONAL: Nice-to-have. Silent failure acceptable.
    """
    CRITICAL = "CRITICAL"   # Abort run
    REQUIRED = "REQUIRED"   # Must succeed or provide safe fallback
    DEGRADED = "DEGRADED"   # Track in degradations list
    OPTIONAL = "OPTIONAL"   # Can silently fail
```

### 1.2 Phase Criticality Classification

| Phase | Section Name | Criticality | Rationale |
|-------|--------------|-------------|-----------|
| **Data Loading** | `load_data` | CRITICAL | No data = no run |
| **Data Contract** | `data.contract` | CRITICAL | Invalid schema = invalid results |
| **Feature Build** | `features.build` | CRITICAL | Features are detector inputs |
| **Feature Impute** | `features.impute` | REQUIRED | Missing values break detectors |
| **Detector Fit** | `train.detector_fit` | CRITICAL | No models = no scoring |
| **Detector Score** | `score.detector_score` | CRITICAL | This IS the core output |
| **Fusion** | `fusion` | CRITICAL | fused_z is the primary health signal |
| **Calibration** | `calibrate` | REQUIRED | Affects threshold accuracy |
| **Regime Label** | `regimes.label` | REQUIRED | Used by thresholds, RUL |
| **Baseline Seed** | `baseline.seed` | DEGRADED | Can fall back to no baseline |
| **Seasonality** | `seasonality.detect` | DEGRADED | Enhances, not required |
| **Guardrails** | `data.guardrails` | DEGRADED | Warnings, not blocks |
| **Model Persistence** | `models.persistence.save` | DEGRADED | Won't affect this run's results |
| **Regime Occupancy** | `regimes.occupancy` | OPTIONAL | Dashboard analytics only |
| **Sensor Correlation** | `persist.sensor_correlation` | OPTIONAL | Analytical output |
| **Detector Correlation** | `persist.detector_correlation` | OPTIONAL | Analytical output |
| **Forecasting** | `outputs.forecasting` | DEGRADED | Separate from health scoring |
| **Transient Detection** | `regimes.transient_detection` | OPTIONAL | Enhancement only |
| **Drift** | `drift` | DEGRADED | Trend indicator |

---

## Phase 2: Implement Phase Gate Infrastructure

### 2.1 Create `PhaseResult` Contract

```python
@dataclass
class PhaseResult:
    """Standard result from any pipeline phase."""
    success: bool
    phase_name: str
    criticality: PhaseCriticality
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    @property
    def should_abort(self) -> bool:
        return not self.success and self.criticality == PhaseCriticality.CRITICAL
    
    @property
    def should_track_degradation(self) -> bool:
        return not self.success and self.criticality in (
            PhaseCriticality.REQUIRED, PhaseCriticality.DEGRADED
        )
```

### 2.2 Create `run_phase()` Wrapper

Replace ad-hoc `safe_step()` with uniform phase execution:

```python
def run_phase(
    phase_name: str,
    criticality: PhaseCriticality,
    operation: Callable[[], Dict[str, Any]],
    preconditions: Optional[List[Tuple[str, Any]]] = None,
    degradations: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> PhaseResult:
    """Execute a pipeline phase with proper error handling and contracts.
    
    Args:
        phase_name: Section name for timing/tracing
        criticality: How to handle failures
        operation: Callable that returns dict of outputs
        preconditions: List of (name, value) tuples that must be truthy
        degradations: List to append to if phase degrades
        context: Logging context (equip, run_id, etc.)
    """
    ctx = context or {}
    
    # Validate preconditions
    if preconditions:
        for name, value in preconditions:
            if value is None or (isinstance(value, pd.DataFrame) and value.empty):
                msg = f"Precondition failed: {name} is None/empty"
                Console.error(msg, phase=phase_name, **ctx)
                if criticality == PhaseCriticality.CRITICAL:
                    raise PreconditionError(phase_name, name)
                return PhaseResult(
                    success=False,
                    phase_name=phase_name,
                    criticality=criticality,
                    error=msg,
                )
    
    try:
        with T.section(phase_name):
            outputs = operation()
            return PhaseResult(
                success=True,
                phase_name=phase_name,
                criticality=criticality,
                outputs=outputs,
            )
    except Exception as e:
        err_msg = f"{phase_name} failed: {e}"
        
        if criticality == PhaseCriticality.CRITICAL:
            Console.error(err_msg, component=phase_name.upper(), **ctx)
            raise  # Abort pipeline
        
        if criticality == PhaseCriticality.OPTIONAL:
            # Silent failure - but still log for debugging
            Console.status(f"[optional] {err_msg}")
            return PhaseResult(success=False, phase_name=phase_name, 
                             criticality=criticality, error=str(e))
        
        # REQUIRED or DEGRADED - log warning, track degradation
        Console.warn(err_msg, component=phase_name.upper(), **ctx)
        if degradations is not None:
            degradations.append(phase_name)
        
        return PhaseResult(
            success=False,
            phase_name=phase_name,
            criticality=criticality,
            error=str(e),
        )
```

---

## Phase 3: Add Precondition Assertions

### 3.1 Define Phase Preconditions

Each CRITICAL/REQUIRED phase must validate its inputs:

```python
# Feature Build preconditions
FEATURE_BUILD_PRECONDITIONS = [
    ("train", train),
    ("score", score),
    ("train has rows", len(train) > 0),
]

# Detector Score preconditions  
DETECTOR_SCORE_PRECONDITIONS = [
    ("score data", score),
    ("ar1_detector fitted", ar1_detector is not None),
    ("pca_detector fitted", pca_detector is not None),
    ("feature columns match", set(score.columns) >= set(expected_feature_cols)),
]

# Fusion preconditions
FUSION_PRECONDITIONS = [
    ("frame with detector scores", frame is not None),
    ("ar1_z column", "ar1_z" in frame.columns),
    ("fused_z computable", any(c in frame.columns for c in DETECTOR_Z_COLS)),
]
```

### 3.2 Create `PreconditionError`

```python
class PreconditionError(Exception):
    """Raised when a phase's preconditions are not met."""
    def __init__(self, phase: str, failed_condition: str):
        self.phase = phase
        self.failed_condition = failed_condition
        super().__init__(f"Phase '{phase}' precondition failed: {failed_condition}")
```

---

## Phase 4: Eliminate Sentinel String Patterns

### 4.1 Replace `"STATE_LOADED"` Sentinel

Current dangerous pattern:
```python
regime_model = "STATE_LOADED"  # String sentinel
# ... later ...
if regime_model == "STATE_LOADED" and regime_state is not None:
    regime_model = regimes.regime_state_to_model(...)
```

Replace with:
```python
class RegimeModelSource(Enum):
    NOT_LOADED = "NOT_LOADED"
    FROM_STATE = "FROM_STATE"
    NEWLY_TRAINED = "NEWLY_TRAINED"
    CACHED = "CACHED"

@dataclass
class RegimeModelContext:
    model: Optional[Any] = None
    source: RegimeModelSource = RegimeModelSource.NOT_LOADED
    state: Optional[Any] = None
    state_version: int = 0
```

### 4.2 File Changes Required

| File | Change |
|------|--------|
| acm_main.py | Replace `regime_model = "STATE_LOADED"` with `RegimeModelContext` |
| regimes.py | Update `label()` to accept/return `RegimeModelContext` |
| model_persistence.py | Update `load_regime_state()` return type |

---

## Phase 5: Initialize All Variables at Function Scope

### 5.1 Current Problem Locations

The `run_acm()` function has ~20 variables that are conditionally initialized:

```python
# Line ~890 - partial initialization
model_state = None  # V11 CRITICAL-1
train: Optional[pd.DataFrame] = None
col_meds: Optional[pd.Series] = None
# NOTE: regime_model already declared at line 2602, don't redeclare  ← THIS IS THE PROBLEM
```

### 5.2 Create `PipelineState` Dataclass

Move ALL mutable state into a single container:

```python
@dataclass
class PipelineState:
    """All mutable state for a pipeline run. Initialized once at run start."""
    # Data state
    train: Optional[pd.DataFrame] = None
    score: Optional[pd.DataFrame] = None
    raw_train: Optional[pd.DataFrame] = None
    raw_score: Optional[pd.DataFrame] = None
    frame: Optional[pd.DataFrame] = None
    
    # Feature state
    col_meds: Optional[pd.Series] = None
    train_feature_hash: Optional[str] = None
    current_train_columns: Optional[List[str]] = None
    
    # Model state
    ar1_detector: Optional[Any] = None
    pca_detector: Optional[Any] = None
    iforest_detector: Optional[Any] = None
    gmm_detector: Optional[Any] = None
    omr_detector: Optional[Any] = None
    regime_context: Optional[RegimeModelContext] = None
    model_state: Optional[Any] = None
    
    # Timing
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    score_start: Optional[datetime] = None
    score_end: Optional[datetime] = None
    
    # Regime state
    regime_state_version: int = 0
    score_regime_labels: Optional[np.ndarray] = None
    train_regime_labels: Optional[np.ndarray] = None
    
    # Tracking
    degradations: List[str] = field(default_factory=list)
    rows_read: int = 0
    rows_written: int = 0
```

---

## Phase 6: Eliminate `except: pass` Blocks

### 6.1 Audit Each Block

| Line | Current Code | Action |
|------|--------------|--------|
| 676 | `except ImportError: pass` | OK - optional integration |
| 1393 | `except Exception: pass` | CHANGE to degradation tracking |
| 1471 | `except Exception: pass` | CHANGE to degradation tracking |
| 1617 | `except Exception: pass` | CHANGE - promotion log should track |
| 2101 | `except Exception: pass` | OK - marked optional |
| 2118 | `except Exception: pass` | OK - marked optional |
| 2172 | `except NameError: pass` | ELIMINATE - use PipelineState |
| 2182 | `except Exception: pass` | CHANGE to degradation tracking |
| 2226 | `except Exception: pass` | OK - OTEL truly optional |
| 2310 | `except Exception: pass` | CHANGE - finalization should track |
| 2317 | `except Exception: pass` | CHANGE - finalization should track |
| 2391 | `except Exception: pass` | CHANGE - cleanup should log |
| 2395 | `except Exception: pass` | CHANGE - cleanup should log |
| 2406 | `except Exception: pass` | CHANGE - cleanup should log |
| 2414 | `except Exception: pass` | CHANGE - cleanup should log |

### 6.2 New Pattern for "Optional with Tracking"

```python
# Instead of:
try:
    write_detector_correlation(...)
except Exception:
    pass  # Detector correlation is optional

# Use:
result = run_phase(
    "persist.detector_correlation",
    PhaseCriticality.OPTIONAL,
    lambda: {"rows": write_detector_correlation(...)},
    context={"equip": equip, "run_id": run_id},
)
# Result is logged, but no degradation tracked for OPTIONAL
```

---

## Phase 7: Add Frame Mutation Tracking

### 7.1 Define Column Contracts per Phase

```python
FRAME_CONTRACTS = {
    "score.detector_score": {
        "required_in": ["Timestamp"],  # Input requirements
        "adds": ["ar1_z", "pca_spe_z", "pca_t2_z", "iforest_z", "gmm_z", "omr_z"],
    },
    "regimes.label": {
        "required_in": ["ar1_z"],  # At least one detector score
        "adds": ["regime"],
    },
    "fusion": {
        "required_in": ["ar1_z"],  # Detector z-scores
        "adds": ["fused_z", "fused_alert", "episode_id"],
    },
    "calibrate": {
        "required_in": ["ar1_z", "pca_spe_z"],
        "modifies": ["ar1_z", "pca_spe_z", "pca_t2_z", "iforest_z", "gmm_z", "omr_z"],
    },
}

def validate_frame_contract(frame: pd.DataFrame, phase: str, direction: str) -> None:
    """Validate frame has required columns before/after a phase."""
    contract = FRAME_CONTRACTS.get(phase)
    if not contract:
        return
    
    required = contract.get(f"required_{direction}", [])
    missing = [c for c in required if c not in frame.columns]
    
    if missing:
        raise FrameContractError(
            f"Phase '{phase}' {direction}put missing columns: {missing}"
        )
```

---

## Phase 8: Implementation Order

### Week 1: Infrastructure
1. Add `PhaseCriticality` enum and `PhaseResult` dataclass
2. Add `PreconditionError` and `FrameContractError`
3. Create `run_phase()` wrapper function
4. Add `PipelineState` dataclass

### Week 2: Phase Classification
5. Classify all 36 phases with criticality levels
6. Define preconditions for CRITICAL phases
7. Define frame contracts for data-passing phases

### Week 3: Migration
8. Migrate CRITICAL phases to `run_phase()` first (load_data, features, detectors, fusion)
9. Migrate REQUIRED phases (calibrate, regimes)
10. Migrate remaining phases

### Week 4: Cleanup
11. Eliminate all `except: pass` blocks
12. Replace `"STATE_LOADED"` sentinel with `RegimeModelContext`
13. Replace scattered variables with `PipelineState`
14. Add integration tests for failure scenarios

---

## Success Criteria

1. **No silent analytical corruption**: CRITICAL phase failures abort the run immediately
2. **Complete degradation tracking**: All non-OPTIONAL failures appear in `degradations` list
3. **Type-safe state**: No string sentinels, all state in typed containers
4. **Auditable**: Every phase has declared criticality and preconditions
5. **Testable**: Unit tests can inject failures at any phase and verify correct handling

---

## Files Modified

| File | Changes |
|------|---------|
| acm_main.py | Major refactor - phases, state container, error handling |
| pipeline_types.py | Add `PhaseCriticality`, `PhaseResult`, `PipelineState` |
| regimes.py | Update to use `RegimeModelContext` |
| model_persistence.py | Update return types |
| `tests/test_phase_failures.py` | New test file for failure injection |

---
