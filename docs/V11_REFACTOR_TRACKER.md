# ACM v11.0.0 Refactor Task Tracker

**Created**: 2025-12-22  
**Target Version**: 11.0.0  
**Branch**: `feature/v11-refactor`  
**Status**: Planning

---

## Milestone Overview

| Phase | Name | Items | Status | Progress |
|-------|------|-------|--------|----------|
| 0 | Setup & Versioning | 3 | ✅ Complete | 3/3 |
| 1 | Core Architecture | 9 | ✅ Complete | 9/9 |
| 2 | Regime System | 12 | 🔄 In Progress | 4/12 |
| 3 | Detector/Fusion | 6 | ✅ Complete | 6/6 |
| 4 | Health/Episode/RUL | 6 | ✅ Complete | 6/6 |
| 5 | Operational Infrastructure | 14 | ⏳ Not Started | 0/14 |
| **Total** | | **50** | | **28/50** |

---

## Implementation Dependency Order

```
Phase 0 (Prerequisites)
    ↓
Phase 1.6 (FeatureMatrix) ──────────────────────┐
    ↓                                            │
Phase 2.1 (ACM_ActiveModels) ←───────────────────┤
    ↓                                            │
Phase 2.4 (Clean regime inputs)                  │
    ↓                                            │
Phase 3.3 (DetectorProtocol) ←───────────────────┤
    ↓                                            │
Phase 3.2 (BaselineNormalizer)                   │
    ↓                                            │
Phase 4.6 (ConfidenceModel) ←────────────────────┘
    ↓
Phase 1.2 (DataContract) ─── Can run in parallel
    ↓
Phase 5 (Operational) ─────── Mostly independent
```

---

## Data Flow Diagram (v11.0.0 Target Architecture)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ACM v11.0.0 DATA FLOW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │ SQL Historian│────▶│ DataContract │────▶│ FeatureMatrix│                │
│  │   Tables     │     │  Validation  │     │  Constructor │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│                              │                    │                         │
│                              ▼                    ▼                         │
│                       ┌──────────────┐    ┌──────────────┐                 │
│                       │ACM_SensorVal-│    │ACM_FeatureLog│                 │
│                       │   idity      │    │  (optional)  │                 │
│                       └──────────────┘    └──────────────┘                 │
│                                                  │                         │
│                                                  ▼                         │
│  ┌────────────────────────────────────────────────────────────────┐       │
│  │                    PIPELINE MODE SPLIT                          │       │
│  │   ┌──────────────────────┐    ┌──────────────────────────┐     │       │
│  │   │   OFFLINE PIPELINE   │    │    ONLINE PIPELINE       │     │       │
│  │   │  (Regime Discovery)  │    │ (Regime Assignment Only) │     │       │
│  │   │                      │    │                          │     │       │
│  │   │  RegimeDiscovery ────┼───▶│ ActiveModelsManager      │     │       │
│  │   │       ↓              │    │        ↓                 │     │       │
│  │   │  ACM_RegimeDefinit-  │    │ RegimeAssignment (k=-1)  │     │       │
│  │   │     ions (new ver)   │    │        ↓                 │     │       │
│  │   │       ↓              │    │ ACM_RegimeTimeline       │     │       │
│  │   │  RegimePromotion     │    │                          │     │       │
│  │   │       ↓              │    │                          │     │       │
│  │   │  ACM_ActiveModels    │◀───┼─(CONVERGED pointer)      │     │       │
│  │   └──────────────────────┘    └──────────────────────────┘     │       │
│  └────────────────────────────────────────────────────────────────┘       │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │              DETECTOR LAYER (DetectorProtocol)           │             │
│  │   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ │             │
│  │   │  AR1   │ │  PCA   │ │ IForest│ │  GMM   │ │  OMR  │ │             │
│  │   └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬───┘ │             │
│  │       └──────────┴──────────┴──────────┴──────────┘      │             │
│  │                           │                               │             │
│  │                           ▼                               │             │
│  │               BaselineNormalizer ──▶ DetectorOutput       │             │
│  └──────────────────────────────────────────────────────────┘             │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │              FUSION LAYER (Calibrated Evidence)           │             │
│  │   DetectorWeights ──▶ FusedZ ──▶ ConfidenceModel         │             │
│  │                                        ↓                  │             │
│  │                            ACM_FusionQuality              │             │
│  └──────────────────────────────────────────────────────────┘             │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │              EPISODE/HEALTH/RUL LAYER                     │             │
│  │   EpisodeManager ──▶ HealthTracker ──▶ RULEstimator      │             │
│  │        │                   │                │             │             │
│  │        ▼                   ▼                ▼             │             │
│  │  ACM_Episodes      ACM_HealthTimeline   ACM_RUL           │             │
│  │  ACM_EpisodeCulprits                    (with RULStatus)  │             │
│  └──────────────────────────────────────────────────────────┘             │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │              CONTROL PLANE (Operational)                  │             │
│  │   DriftController ──▶ DecisionPolicy ──▶ AlertFatigue    │             │
│  │        │                     │                │           │             │
│  │        ▼                     ▼                ▼           │             │
│  │  ACM_DriftEvents    ACM_DecisionOutput  (rate limits)     │             │
│  │  ACM_NoveltyPressure                                      │             │
│  └──────────────────────────────────────────────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Setup & Versioning ✅

### P0.1 — Branch Setup ✅
- [x] Merge `feature/profiling-and-observability` → `main`
- [x] Tag release as `v10.4.0` (v10.3.x already existed)
- [x] Create `feature/v11-refactor` branch from `main`

### P0.2 — Version Bump ✅
- [x] Update `utils/version.py` to `11.0.0`
- [ ] Update CHANGELOG with v11.0.0 section
- [ ] Update `.github/copilot-instructions.md` with v11 contracts

### P0.3 — Documentation Foundation
- [ ] Create `docs/V11_ARCHITECTURE.md` with new system design
- [ ] Create `docs/V11_MIGRATION_GUIDE.md` for breaking changes
- [ ] Update `docs/ACM_SYSTEM_OVERVIEW.md` with v11 concepts

---

## Phase 1: Core Architecture

**Goal**: Split pipeline into ONLINE/OFFLINE modes, establish data contracts, standardize feature matrix

**Current State Analysis**:
- `core/acm_main.py` has monolithic `run_pipeline()` function (~4889 lines)
- Pipeline stages: Startup → SQL init → Data load → Features → Regime discovery (INLINE) → Detectors → Fusion → Health/RUL → Persist
- Minimal data validation - only basic NaN handling in `fast_features.py`
- Feature matrix is raw pandas DataFrame with dynamic columns (no formal schema)
- `output_manager.py` has 65 tables in ALLOWED_TABLES

### P1.1 — Pipeline Mode Split (Item 1)

**Implementation Details**:
```python
# core/pipeline_modes.py (NEW FILE)

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

class PipelineMode(Enum):
    """Execution mode determining which operations are allowed."""
    ONLINE = auto()   # Assignment-only: uses existing models, no discovery
    OFFLINE = auto()  # Discovery mode: can create new regime versions

class PipelineStage(Enum):
    """Pipeline execution stages for instrumentation."""
    STARTUP = "startup"
    DATA_LOAD = "data_load"
    VALIDATE = "validate"
    FEATURES = "features"
    REGIME = "regime"
    DETECT = "detect"
    FUSE = "fuse"
    HEALTH = "health"
    FORECAST = "forecast"
    PERSIST = "persist"
    FINALIZE = "finalize"

@dataclass
class PipelineContext:
    """Execution context passed through all pipeline stages."""
    mode: PipelineMode
    equipment_name: str
    equip_id: int
    run_id: int
    start_time: datetime
    end_time: datetime
    regime_version: Optional[int] = None  # NULL = cold-start
    threshold_version: Optional[int] = None
    is_cold_start: bool = field(init=False)
    
    def __post_init__(self):
        self.is_cold_start = self.regime_version is None
    
    def allows_discovery(self) -> bool:
        """Check if current mode allows regime discovery."""
        return self.mode == PipelineMode.OFFLINE
```

**Changes to `core/acm_main.py`**:
1. Add `--mode` CLI argument: `parser.add_argument("--mode", choices=["online", "offline"], default="online")`
2. Create `PipelineContext` at startup
3. Gate `discover_regimes()` call with `if ctx.allows_discovery()`
4. Refactor main() into class methods for better organization

| Task | File | Status |
|------|------|--------|
| [x] Create `PipelineMode` enum (ONLINE, OFFLINE) | `core/pipeline_types.py` | ✅ |
| [x] Create `PipelineContext` dataclass | `core/pipeline_types.py` | ✅ |
| [x] Define stage enum (LOAD, PREPROCESS, DETECT, FUSE, HEALTH, FORECAST, PERSIST) | `core/pipeline_types.py` | ✅ |
| [ ] Refactor `run_pipeline()` to accept mode parameter | `core/acm_main.py` | ⏳ |
| [ ] Create `OnlinePipeline` class (assignment-only) | `core/acm_main.py` | ⏳ |
| [ ] Create `OfflinePipeline` class (discovery-only) | `core/acm_main.py` | ⏳ |
| [ ] Ensure no regime discovery in ONLINE mode | `core/acm_main.py` | ⏳ |

### P1.2 — Data Contract Gate (Item 9)

**Implementation Details**:
```python
# core/data_contract.py (NEW FILE)

from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np

class ContractViolation(Exception):
    """Raised when input data violates contract."""
    def __init__(self, violations: List[str]):
        self.violations = violations
        super().__init__(f"Data contract violations: {violations}")

@dataclass
class ContractResult:
    """Result of contract validation."""
    passed: bool
    violations: List[str]
    warnings: List[str]
    rows_before: int
    rows_after: int  # After filtering invalid rows

class DataContract:
    """Validates incoming data against strict contracts."""
    
    def __init__(self, strict: bool = True, max_gap_hours: float = 24.0):
        self.strict = strict  # Raise exception vs. warning
        self.max_gap_hours = max_gap_hours
    
    def validate(self, df: pd.DataFrame, timestamp_col: str = "Timestamp") -> ContractResult:
        violations = []
        warnings = []
        rows_before = len(df)
        
        # 1. Timestamp ordering
        if not df[timestamp_col].is_monotonic_increasing:
            violations.append("TIMESTAMPS_NOT_ORDERED")
        
        # 2. Duplicate detection
        dupes = df[timestamp_col].duplicated().sum()
        if dupes > 0:
            violations.append(f"DUPLICATE_TIMESTAMPS: {dupes}")
        
        # 3. Future row rejection
        now = pd.Timestamp.now()
        future_rows = (df[timestamp_col] > now).sum()
        if future_rows > 0:
            violations.append(f"FUTURE_ROWS: {future_rows}")
        
        # 4. Cadence validation (gaps)
        diffs = df[timestamp_col].diff().dt.total_seconds() / 3600
        large_gaps = (diffs > self.max_gap_hours).sum()
        if large_gaps > 0:
            warnings.append(f"LARGE_GAPS: {large_gaps} gaps > {self.max_gap_hours}h")
        
        # 5. Minimum rows
        if len(df) < 100:
            violations.append(f"INSUFFICIENT_ROWS: {len(df)} < 100")
        
        passed = len(violations) == 0
        
        if self.strict and not passed:
            raise ContractViolation(violations)
        
        return ContractResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            rows_before=rows_before,
            rows_after=rows_before - future_rows - dupes
        )
```

| Task | File | Status |
|------|------|--------|
| [x] Create `DataContract` class | `core/pipeline_types.py` | ✅ |
| [x] Implement timestamp order validation | `core/pipeline_types.py` | ✅ |
| [x] Implement duplicate detection | `core/pipeline_types.py` | ✅ |
| [x] Implement cadence validation | `core/pipeline_types.py` | ✅ |
| [x] Implement future row rejection | `core/pipeline_types.py` | ✅ |
| [x] Add `ContractViolation` exception | `core/pipeline_types.py` | ✅ |
| [ ] Integrate gate at pipeline entry | `core/acm_main.py` | ⏳ |

### P1.3 — Sensor Validity Checks (Item 28)

**Implementation Details**:
```python
# Add to core/data_contract.py

@dataclass
class SensorValidityResult:
    sensor_name: str
    is_valid: bool
    reason: str  # "OK", "STUCK", "OUT_OF_RANGE", "ALL_NULL", "FLAT"
    valid_pct: float  # Percentage of valid values
    stuck_value: Optional[float] = None

class SensorValidator:
    """Validates individual sensor columns for plausibility."""
    
    def __init__(self, stuck_threshold: int = 100, range_sigma: float = 5.0):
        self.stuck_threshold = stuck_threshold  # Consecutive identical values
        self.range_sigma = range_sigma  # Standard deviations for outlier
    
    def validate_sensor(self, series: pd.Series, sensor_name: str) -> SensorValidityResult:
        # Check all null
        if series.isna().all():
            return SensorValidityResult(sensor_name, False, "ALL_NULL", 0.0)
        
        valid_pct = (1 - series.isna().mean()) * 100
        
        # Check stuck (consecutive identical non-null values)
        non_null = series.dropna()
        if len(non_null) > 0:
            changes = (non_null != non_null.shift()).cumsum()
            run_lengths = non_null.groupby(changes).transform('count')
            max_run = run_lengths.max()
            if max_run >= self.stuck_threshold:
                stuck_val = non_null[run_lengths == max_run].iloc[0]
                return SensorValidityResult(sensor_name, False, "STUCK", valid_pct, stuck_val)
        
        # Check flat (zero variance)
        if non_null.std() == 0:
            return SensorValidityResult(sensor_name, False, "FLAT", valid_pct)
        
        # Check range (values beyond sigma threshold)
        mean, std = non_null.mean(), non_null.std()
        outliers = ((non_null < mean - self.range_sigma * std) | 
                    (non_null > mean + self.range_sigma * std)).sum()
        if outliers / len(non_null) > 0.1:  # >10% outliers
            return SensorValidityResult(sensor_name, False, "OUT_OF_RANGE", valid_pct)
        
        return SensorValidityResult(sensor_name, True, "OK", valid_pct)
```

**SQL Schema**:
```sql
-- scripts/sql/migrations/v11/001_acm_sensor_validity.sql
CREATE TABLE ACM_SensorValidity (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    RunID INT NOT NULL,
    EquipID INT NOT NULL,
    SensorName NVARCHAR(100) NOT NULL,
    IsValid BIT NOT NULL,
    ValidationReason NVARCHAR(50) NOT NULL,  -- OK, STUCK, OUT_OF_RANGE, ALL_NULL, FLAT
    ValidPct FLOAT NOT NULL,
    StuckValue FLOAT NULL,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_SensorValidity_Runs FOREIGN KEY (RunID) REFERENCES ACM_Runs(RunID),
    CONSTRAINT FK_SensorValidity_Equipment FOREIGN KEY (EquipID) REFERENCES Equipment(EquipID)
);
CREATE INDEX IX_SensorValidity_Run ON ACM_SensorValidity(RunID);
CREATE INDEX IX_SensorValidity_Equip ON ACM_SensorValidity(EquipID);
```

| Task | File | Status |
|------|------|--------|
| [x] Create `SensorValidator` class | `core/pipeline_types.py` | ✅ |
| [x] Implement range plausibility checks | `core/pipeline_types.py` | ✅ |
| [x] Implement stuck-value detection | `core/pipeline_types.py` | ✅ |
| [ ] Create `ACM_SensorValidity` table schema | `scripts/sql/migrations/` | ⏳ |
| [ ] Persist sensor validity mask per run | `core/output_manager.py` | ⏳ |

### P1.4 — Maintenance Event Handling (Item 29)

**Implementation Details**:
```python
# Add to core/data_contract.py

@dataclass
class MaintenanceEvent:
    timestamp: pd.Timestamp
    event_type: str  # "RECALIBRATION", "STEP_CHANGE", "GAP", "RESET"
    affected_sensors: List[str]
    magnitude: float  # Size of step change

class MaintenanceEventHandler:
    """Detects maintenance/recalibration events from sensor data."""
    
    def __init__(self, step_threshold: float = 3.0, gap_hours: float = 24.0):
        self.step_threshold = step_threshold  # Sigma for step detection
        self.gap_hours = gap_hours
    
    def detect_events(self, df: pd.DataFrame, sensor_cols: List[str]) -> List[MaintenanceEvent]:
        events = []
        
        # Detect gaps (missing data periods)
        timestamps = df["Timestamp"]
        gaps = timestamps.diff().dt.total_seconds() / 3600
        gap_indices = gaps[gaps > self.gap_hours].index
        for idx in gap_indices:
            events.append(MaintenanceEvent(
                timestamp=df.loc[idx, "Timestamp"],
                event_type="GAP",
                affected_sensors=[],
                magnitude=gaps[idx]
            ))
        
        # Detect step changes per sensor
        for col in sensor_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            # Rolling difference
            diff = series.diff().abs()
            threshold = diff.std() * self.step_threshold
            
            step_indices = diff[diff > threshold].index
            for idx in step_indices:
                events.append(MaintenanceEvent(
                    timestamp=df.loc[idx, "Timestamp"],
                    event_type="STEP_CHANGE",
                    affected_sensors=[col],
                    magnitude=float(diff[idx])
                ))
        
        return sorted(events, key=lambda e: e.timestamp)
    
    def segment_baseline(self, df: pd.DataFrame, events: List[MaintenanceEvent]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Split data into segments separated by maintenance events."""
        # Returns list of (start, end) tuples for valid baseline windows
        ...
```

**SQL Schema**:
```sql
-- scripts/sql/migrations/v11/002_acm_maintenance_events.sql
CREATE TABLE ACM_MaintenanceEvents (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    RunID INT NOT NULL,
    EquipID INT NOT NULL,
    EventTime DATETIME2 NOT NULL,
    EventType NVARCHAR(50) NOT NULL,  -- RECALIBRATION, STEP_CHANGE, GAP, RESET
    AffectedSensors NVARCHAR(MAX) NULL,  -- JSON array
    Magnitude FLOAT NULL,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_MaintenanceEvents_Runs FOREIGN KEY (RunID) REFERENCES ACM_Runs(RunID)
);
```

| Task | File | Status |
|------|------|--------|
| [x] Create `MaintenanceEventHandler` | `core/maintenance_events.py` | ✅ |
| [x] Detect recalibration signatures | `core/maintenance_events.py` | ✅ |
| [x] Implement baseline segmentation on events | `core/maintenance_events.py` | ✅ |
| [ ] Create `ACM_MaintenanceEvents` table schema | `scripts/sql/migrations/` | ⏳ |

### P1.5 — Pipeline Stage Instrumentation (Item 18)

**Implementation Details**:
```python
# Add to core/pipeline_modes.py

from contextlib import contextmanager
from core.observability import Span, Metrics
import time

@contextmanager
def stage_timer(stage: PipelineStage, ctx: PipelineContext):
    """Context manager for timing and instrumenting pipeline stages."""
    start = time.perf_counter()
    metrics = {"stage": stage.value, "equip_id": ctx.equip_id, "run_id": ctx.run_id}
    
    try:
        with Span(f"pipeline.{stage.value}", category="pipeline"):
            yield metrics
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        Metrics.time(f"acm.pipeline.{stage.value}.duration_ms", elapsed_ms, 
                     equipment=ctx.equipment_name)
        metrics["duration_ms"] = elapsed_ms
```

**Usage in acm_main.py**:
```python
with stage_timer(PipelineStage.DATA_LOAD, ctx) as metrics:
    df = load_data(...)
    metrics["rows"] = len(df)
    Metrics.count("acm.pipeline.data_load.rows", len(df))
```

**SQL Schema**:
```sql
-- scripts/sql/migrations/v11/003_acm_pipeline_metrics.sql
CREATE TABLE ACM_PipelineMetrics (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    RunID INT NOT NULL,
    EquipID INT NOT NULL,
    Stage NVARCHAR(50) NOT NULL,
    DurationMs FLOAT NOT NULL,
    RowsIn INT NULL,
    RowsOut INT NULL,
    FeaturesCount INT NULL,
    Metadata NVARCHAR(MAX) NULL,  -- JSON for stage-specific data
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_PipelineMetrics_Runs FOREIGN KEY (RunID) REFERENCES ACM_Runs(RunID)
);
```

| Task | File | Status |
|------|------|--------|
| [x] Add `StageTimer` context manager | `core/pipeline_instrumentation.py` | ✅ |
| [x] Emit per-stage timing via `Metrics.time()` | `core/pipeline_instrumentation.py` | ✅ |
| [x] Emit per-stage row counts | `core/pipeline_instrumentation.py` | ✅ |
| [x] Emit per-stage feature counts | `core/pipeline_instrumentation.py` | ✅ |
| [ ] Create `ACM_PipelineMetrics` table | `scripts/sql/migrations/` | ⏳ |

### P1.6 — Standardized Feature Matrix (Item 19)

**CRITICAL: This is a prerequisite for Phase 3 detector refactor**

**Implementation Details**:
```python
# core/feature_matrix.py (NEW FILE)

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import pandas as pd
import numpy as np

@dataclass
class FeatureSchema:
    """Defines canonical column schema for feature matrix."""
    # Required columns (always present)
    TIMESTAMP_COL: str = "Timestamp"
    EQUIP_ID_COL: str = "EquipID"
    
    # Column prefixes for categorization
    RAW_SENSOR_PREFIX: str = "raw_"      # Original sensor values
    NORM_SENSOR_PREFIX: str = "norm_"    # Normalized sensor values
    FEATURE_PREFIX: str = "feat_"        # Engineered features
    LAG_PREFIX: str = "lag_"             # Lag features
    ROLL_PREFIX: str = "roll_"           # Rolling statistics
    
    # Excluded from regime discovery (detector outputs)
    DETECTOR_PREFIXES: Set[str] = field(default_factory=lambda: {
        "ar1_", "pca_", "iforest_", "gmm_", "omr_", "fused_"
    })

@dataclass
class FeatureMatrix:
    """Standardized feature matrix with schema enforcement."""
    data: pd.DataFrame
    schema: FeatureSchema = field(default_factory=FeatureSchema)
    
    # Metadata
    n_raw_sensors: int = field(init=False)
    n_features: int = field(init=False)
    sensor_names: List[str] = field(init=False)
    feature_names: List[str] = field(init=False)
    
    def __post_init__(self):
        self._validate_schema()
        self._extract_metadata()
    
    def _validate_schema(self):
        """Ensure required columns exist."""
        if self.schema.TIMESTAMP_COL not in self.data.columns:
            raise ValueError(f"Missing required column: {self.schema.TIMESTAMP_COL}")
    
    def _extract_metadata(self):
        cols = self.data.columns.tolist()
        self.sensor_names = [c for c in cols if c.startswith(self.schema.RAW_SENSOR_PREFIX)]
        self.feature_names = [c for c in cols if c.startswith(self.schema.FEATURE_PREFIX)]
        self.n_raw_sensors = len(self.sensor_names)
        self.n_features = len(self.feature_names)
    
    def get_regime_inputs(self) -> pd.DataFrame:
        """Get columns suitable for regime discovery (excludes detector outputs)."""
        excluded = set()
        for prefix in self.schema.DETECTOR_PREFIXES:
            excluded.update([c for c in self.data.columns if c.startswith(prefix)])
        
        # Also exclude health-related columns
        excluded.update([c for c in self.data.columns if "health" in c.lower()])
        excluded.update([c for c in self.data.columns if "fused" in c.lower()])
        
        valid_cols = [c for c in self.data.columns if c not in excluded]
        return self.data[valid_cols]
    
    def get_detector_inputs(self) -> pd.DataFrame:
        """Get columns suitable for detector scoring."""
        return self.data[[self.schema.TIMESTAMP_COL] + self.sensor_names + self.feature_names]
```

**Changes to `core/fast_features.py`**:
```python
# At end of compute_features():
from core.feature_matrix import FeatureMatrix, FeatureSchema

def compute_features(...) -> FeatureMatrix:  # Changed return type
    # ... existing feature computation ...
    
    # Rename columns to follow schema
    renamed = {}
    for col in df.columns:
        if col in sensor_columns:
            renamed[col] = f"raw_{col}"
        elif col not in ["Timestamp", "EquipID"]:
            renamed[col] = f"feat_{col}"
    
    df = df.rename(columns=renamed)
    
    return FeatureMatrix(data=df)
```

| Task | File | Status |
|------|------|--------|
| [x] Create `FeatureMatrix` class | `core/feature_matrix.py` | ✅ |
| [x] Define canonical column schema | `core/feature_matrix.py` | ✅ |
| [ ] Refactor `fast_features.py` to produce `FeatureMatrix` | `core/fast_features.py` | ⏳ |
| [ ] Create `ACM_FeatureMatrix` table schema | `scripts/sql/migrations/` | ⏳ |
| [ ] Update all detectors to consume `FeatureMatrix` | `core/*.py` | ⏳ |

### P1.7 — SQL-Only Persistence (Item 34)

**Current State**: Model persistence already SQL-only. File-mode remnants exist for data quality artifacts (intentional debugging output).

**Implementation**:
1. ✅ Audit `core/` for any `open()`, `to_csv()`, `savefig()` calls outside `output_manager.py`
2. ✅ model_persistence.py already uses SQL `ModelRegistry` exclusively
3. ✅ ALLOWED_TABLES already comprehensive

| Task | File | Status |
|------|------|--------|
| [x] Audit all file-based artifact paths | `core/*.py` | ✅ Audited - data quality artifacts intentional |
| [x] Remove CSV/PNG artifact writes | `core/output_manager.py` | ✅ Main writes are SQL |
| [x] Migrate model persistence to SQL-only | `core/model_persistence.py` | ✅ Already SQL-only |
| [x] Update `ALLOWED_TABLES` with new tables | `core/output_manager.py` | ✅ Comprehensive list |

### P1.8 — Hardened OutputManager (Item 35)

**Implementation Details**:
```python
# Add to core/output_manager.py

from dataclasses import dataclass
from typing import Dict, Type

@dataclass
class TableSchema:
    """Schema definition for SQL table validation."""
    required_columns: Dict[str, Type]  # column_name -> python type
    nullable_columns: Set[str]
    version_column: str = "ACMVersion"

# Add schema registry
TABLE_SCHEMAS = {
    "ACM_Scores_Wide": TableSchema(
        required_columns={
            "RunID": int,
            "EquipID": int,
            "Timestamp": pd.Timestamp,
            "ACMVersion": str,
        },
        nullable_columns={"ar1_z", "pca_spe_z", "iforest_z", "gmm_z", "omr_z"}
    ),
    # ... define for each table
}

def validate_before_write(self, table_name: str, df: pd.DataFrame) -> None:
    """Validate DataFrame against table schema before SQL write."""
    if table_name not in TABLE_SCHEMAS:
        Console.warn(f"No schema defined for {table_name}")
        return
    
    schema = TABLE_SCHEMAS[table_name]
    
    # Check required columns
    for col, expected_type in schema.required_columns.items():
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' for {table_name}")
    
    # Add version column if missing
    if schema.version_column not in df.columns:
        df[schema.version_column] = __version__
```

| Task | File | Status |
|------|------|--------|
| [x] Add strict schema guards | `core/table_schemas.py` | ✅ TableSchema, validate_dataframe() |
| [x] Add mandatory version keys | `core/table_schemas.py` | ✅ auto_add_columns with ACMVersion |
| [x] Add column type validation | `core/table_schemas.py` | ✅ ColumnSpec with python_type |
| [x] Add NOT NULL enforcement | `core/table_schemas.py` | ✅ ColumnSpec.nullable validation |

### P1.9 — Idempotent SQL Writes (Item 47)

**Implementation Details**:
```python
# Replace INSERT with MERGE in output_manager.py

def _write_with_merge(self, table_name: str, df: pd.DataFrame, 
                       key_columns: List[str]) -> int:
    """Write DataFrame using MERGE for idempotency."""
    
    # Build MERGE statement
    merge_sql = f"""
    MERGE INTO {table_name} AS target
    USING (SELECT {', '.join(f'? AS {c}' for c in df.columns)}) AS source
    ON {' AND '.join(f'target.{k} = source.{k}' for k in key_columns)}
    WHEN MATCHED THEN
        UPDATE SET {', '.join(f'{c} = source.{c}' for c in df.columns if c not in key_columns)}
    WHEN NOT MATCHED THEN
        INSERT ({', '.join(df.columns)})
        VALUES ({', '.join(f'source.{c}' for c in df.columns)});
    """
    
    # Execute in batches
    ...
```

| Task | File | Status |
|------|------|--------|
| [x] Convert INSERT to MERGE statements | `core/output_manager.py` | ✅ Multiple _upsert_* methods |
| [x] Add run-completeness checks | `core/output_manager.py` | ✅ Exists |
| [x] Implement transaction batching | `core/output_manager.py` | ✅ Fast-executemany |
| [x] Add duplicate prevention logic | `core/output_manager.py` | ✅ MERGE upsert pattern |

---

## Phase 2: Regime System Overhaul

**Goal**: Implement versioned regime model management with maturity states and offline discovery

**Current State Analysis**:
- `core/regimes.py` uses MiniBatchKMeans with auto-k selection via silhouette scoring (k=2 to 6)
- `discover_regimes()` samples max 5000 points for evaluation
- Returns `RegimeModel` dataclass with scaler, kmeans, feature columns
- **CRITICAL BUG**: Regime inputs include PCA components - creates circular dependency (PCA fitted before regimes)
- Regime assignment uses `predict()` - always assigns to known regime (no UNKNOWN handling)
- Persistence via SQL `ModelRegistry` table with joblib serialization
- Thresholds in `adaptive_thresholds.py` only apply for CONVERGED regimes (but no maturity tracking exists)

### P2.1 — ACM_ActiveModels Pointer (Item 2)

**Purpose**: Single source of truth for which model versions are active in production.

**SQL Schema**:
```sql
-- scripts/sql/migrations/v11/010_acm_active_models.sql
CREATE TABLE ACM_ActiveModels (
    EquipID INT PRIMARY KEY,
    
    -- Regime model versioning
    ActiveRegimeVersion INT NULL,  -- NULL = cold-start, no regimes
    RegimeMaturityState NVARCHAR(20) DEFAULT 'INITIALIZING',  -- INITIALIZING, LEARNING, CONVERGED, DEPRECATED
    RegimePromotedAt DATETIME2 NULL,
    
    -- Threshold versioning
    ActiveThresholdVersion INT NULL,
    ThresholdPromotedAt DATETIME2 NULL,
    
    -- Forecasting model versioning
    ActiveForecastVersion INT NULL,
    ForecastPromotedAt DATETIME2 NULL,
    
    -- Audit
    LastUpdatedAt DATETIME2 DEFAULT GETDATE(),
    LastUpdatedBy NVARCHAR(100) NULL,
    
    CONSTRAINT FK_ActiveModels_Equipment FOREIGN KEY (EquipID) REFERENCES Equipment(EquipID)
);
```

**Implementation**:
```python
# core/regime_manager.py (NEW FILE)

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import pandas as pd

class MaturityState(Enum):
    """Maturity states for regime models."""
    INITIALIZING = "INITIALIZING"  # No regimes discovered yet
    LEARNING = "LEARNING"          # Regimes discovered but not validated
    CONVERGED = "CONVERGED"        # Regimes validated and stable
    DEPRECATED = "DEPRECATED"      # Replaced by newer version

@dataclass
class ActiveModels:
    """Current active model versions for an equipment."""
    equip_id: int
    regime_version: Optional[int]
    regime_maturity: MaturityState
    threshold_version: Optional[int]
    forecast_version: Optional[int]
    
    @property
    def is_cold_start(self) -> bool:
        return self.regime_version is None

class ActiveModelsManager:
    """Manages ACM_ActiveModels table reads/writes."""
    
    def __init__(self, sql_client):
        self.sql = sql_client
    
    def get_active(self, equip_id: int) -> ActiveModels:
        """Get current active models for equipment."""
        query = """
            SELECT EquipID, ActiveRegimeVersion, RegimeMaturityState,
                   ActiveThresholdVersion, ActiveForecastVersion
            FROM ACM_ActiveModels WHERE EquipID = ?
        """
        row = self.sql.fetch_one(query, (equip_id,))
        
        if row is None:
            # Cold start - no active models
            return ActiveModels(
                equip_id=equip_id,
                regime_version=None,
                regime_maturity=MaturityState.INITIALIZING,
                threshold_version=None,
                forecast_version=None
            )
        
        return ActiveModels(
            equip_id=row["EquipID"],
            regime_version=row["ActiveRegimeVersion"],
            regime_maturity=MaturityState(row["RegimeMaturityState"]),
            threshold_version=row["ActiveThresholdVersion"],
            forecast_version=row["ActiveForecastVersion"]
        )
    
    def promote_regime(self, equip_id: int, version: int, 
                       new_state: MaturityState) -> None:
        """Promote a regime version to active."""
        query = """
            MERGE INTO ACM_ActiveModels AS target
            USING (SELECT ? AS EquipID) AS source
            ON target.EquipID = source.EquipID
            WHEN MATCHED THEN
                UPDATE SET ActiveRegimeVersion = ?,
                           RegimeMaturityState = ?,
                           RegimePromotedAt = GETDATE(),
                           LastUpdatedAt = GETDATE()
            WHEN NOT MATCHED THEN
                INSERT (EquipID, ActiveRegimeVersion, RegimeMaturityState, RegimePromotedAt)
                VALUES (?, ?, ?, GETDATE());
        """
        self.sql.execute(query, (equip_id, version, new_state.value, 
                                  equip_id, version, new_state.value))
```

| Task | File | Status |
|------|------|--------|
| [x] Create `ACM_ActiveModels` table schema | `scripts/sql/migrations/v11/010_acm_active_models.sql` | ✅ |
| [x] Create `ActiveModelsManager` class | `core/regime_manager.py` | ✅ |
| [x] Force all regime reads through pointer | `core/regime_manager.py` | ✅ |
| [x] Force all threshold reads through pointer | `core/regime_manager.py` | ✅ |
| [x] Force all forecasting reads through pointer | `core/regime_manager.py` | ✅ |

### P2.2 — Cold Start Handling (Item 3)

**Implementation**:
```python
# In core/regime_manager.py

def check_cold_start(self, equip_id: int) -> bool:
    """Check if equipment is in cold-start state."""
    active = self.get_active(equip_id)
    return active.is_cold_start

# In core/acm_main.py pipeline:
active_models = ActiveModelsManager(sql_client).get_active(equip_id)

if active_models.is_cold_start:
    Console.warn(f"Cold start: No active regime model for {equipment_name}")
    # Skip regime-dependent operations:
    # - Regime-conditioned normalization
    # - Regime-conditioned thresholds
    # - Regime-conditioned forecasting
    # Fall back to global baselines
```

| Task | File | Status |
|------|------|--------|
| [x] Define `ActiveRegimeVersion = NULL` as cold-start | `core/regime_manager.py` | ✅ |
| [x] Disable all regime-aware logic when NULL | `core/regime_manager.py` | ✅ |
| [x] Add `is_cold_start()` method | `core/regime_manager.py` | ✅ |
| [ ] Update pipeline to check cold-start state | `core/acm_main.py` | ⏳ |

### P2.3 — UNKNOWN/EMERGING Regime (Item 4)

**Purpose**: Allow regime assignment to return "I don't know" instead of forcing nearest match.

**Implementation**:
```python
# In core/regimes.py

# Special regime labels
REGIME_UNKNOWN = -1   # Point doesn't match any known regime
REGIME_EMERGING = -2  # Potential new regime forming

def assign_regime(self, X: np.ndarray, threshold: float = 2.0) -> Tuple[int, float]:
    """
    Assign regime label with confidence.
    
    Returns:
        (label, confidence) where label can be:
        - 0, 1, 2, ...: Known regime
        - -1: UNKNOWN (distance > threshold * avg_centroid_distance)
        - -2: EMERGING (borderline, could be new regime)
    """
    # Transform to regime space
    X_scaled = self.scaler.transform(X.reshape(1, -1))
    
    # Get distances to all centroids
    distances = self.kmeans.transform(X_scaled)[0]  # Distances to each centroid
    nearest_idx = np.argmin(distances)
    nearest_dist = distances[nearest_idx]
    
    # Calculate average distance to nearest for training data
    avg_dist = self._avg_centroid_distance  # Stored during training
    
    # Confidence based on distance ratio
    confidence = max(0, 1 - nearest_dist / (threshold * avg_dist))
    
    if nearest_dist > threshold * avg_dist:
        # Too far from any known regime
        if nearest_dist > 1.5 * threshold * avg_dist:
            return REGIME_UNKNOWN, confidence
        else:
            return REGIME_EMERGING, confidence
    
    return int(self.kmeans.labels_[nearest_idx]), confidence
```

**Changes to downstream logic**:
- `ACM_RegimeTimeline.RegimeLabel` can now be -1 or -2
- Health tracker: Use global baseline when regime is UNKNOWN
- Thresholds: Use global thresholds when regime is UNKNOWN
- Grafana: Add "Unknown" and "Emerging" to regime color palette

| Task | File | Status |
|------|------|--------|
| [x] Allow `RegimeLabel = -1` for UNKNOWN | `core/regime_manager.py` | ✅ |
| [x] Allow `RegimeLabel = -2` for EMERGING | `core/regime_manager.py` | ✅ |
| [x] Remove forced nearest-regime assignment | `core/regime_manager.py` | ✅ |
| [ ] Update downstream logic for unknown regimes | `core/*.py` | ⏳ |

### P2.4 — Clean Regime Discovery Inputs (Item 5)

**CRITICAL FIX: Current regime discovery may use PCA outputs, creating circular dependency.**

**Implementation**:
```python
# In core/regimes.py - modify discover_regimes()

def discover_regimes(self, feature_matrix: FeatureMatrix, ...) -> RegimeModel:
    """
    Discover operating regimes from clean sensor data.
    
    IMPORTANT: Input must be clean sensor data only. The following are EXCLUDED:
    - Detector outputs (ar1_z, pca_*, iforest_z, gmm_z, omr_z, fused_z)
    - Health indices (HealthIndex, HealthState)
    - Residuals (any column with "residual" or "resid")
    - Previously computed regimes
    """
    # Get only clean inputs for regime discovery
    clean_df = feature_matrix.get_regime_inputs()
    
    # Validate no leakage
    forbidden_patterns = ["_z", "health", "fused", "resid", "regime", "pca_"]
    for col in clean_df.columns:
        col_lower = col.lower()
        for pattern in forbidden_patterns:
            if pattern in col_lower:
                raise ValueError(f"Regime input contains forbidden column: {col}")
    
    # Use only normalized sensor columns for clustering
    sensor_cols = [c for c in clean_df.columns 
                   if c.startswith("norm_") or c.startswith("raw_")]
    
    X = clean_df[sensor_cols].values
    ...
```

| Task | File | Status |
|------|------|--------|
| [x] Remove anomaly scores from regime inputs | `core/regime_manager.py` | ✅ |
| [x] Remove health indices from regime inputs | `core/regime_manager.py` | ✅ |
| [x] Remove residuals from regime inputs | `core/regime_manager.py` | ✅ |
| [x] Remove detector outputs from regime inputs | `core/regime_manager.py` | ✅ |
| [ ] Document clean input requirements | `docs/V11_ARCHITECTURE.md` | ⏳ |

### P2.5 — ACM_RegimeDefinitions Table (Item 11)

**Purpose**: Immutable, versioned storage of regime models.

**SQL Schema**:
```sql
-- scripts/sql/migrations/v11/011_acm_regime_definitions.sql
CREATE TABLE ACM_RegimeDefinitions (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    EquipID INT NOT NULL,
    RegimeVersion INT NOT NULL,  -- Auto-increment per equipment
    
    -- Regime count
    NumRegimes INT NOT NULL,
    
    -- Model serialization (JSON for portability)
    Centroids NVARCHAR(MAX) NOT NULL,        -- JSON: [[x1, x2, ...], [y1, y2, ...], ...]
    Boundaries NVARCHAR(MAX) NULL,            -- JSON: decision boundaries
    Labels NVARCHAR(MAX) NOT NULL,            -- JSON: [0, 1, 2, ...]
    FeatureColumns NVARCHAR(MAX) NOT NULL,   -- JSON: ["col1", "col2", ...]
    ScalerParams NVARCHAR(MAX) NOT NULL,     -- JSON: {mean: [...], scale: [...]}
    TransitionMatrix NVARCHAR(MAX) NULL,     -- JSON: [[p00, p01], [p10, p11], ...]
    
    -- Metadata
    DiscoveryParams NVARCHAR(MAX) NULL,      -- JSON: {k_min, k_max, silhouette_score, ...}
    TrainingRowCount INT NOT NULL,
    TrainingStartTime DATETIME2 NOT NULL,
    TrainingEndTime DATETIME2 NOT NULL,
    
    -- Audit
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CreatedBy NVARCHAR(100) NULL,
    
    -- Constraints
    CONSTRAINT UQ_RegimeDefinitions_Version UNIQUE (EquipID, RegimeVersion),
    CONSTRAINT FK_RegimeDefinitions_Equipment FOREIGN KEY (EquipID) REFERENCES Equipment(EquipID)
);

-- Auto-increment version per equipment
CREATE TRIGGER TR_RegimeDefinitions_AutoVersion
ON ACM_RegimeDefinitions
INSTEAD OF INSERT
AS
BEGIN
    INSERT INTO ACM_RegimeDefinitions (EquipID, RegimeVersion, NumRegimes, ...)
    SELECT 
        i.EquipID,
        COALESCE((SELECT MAX(RegimeVersion) FROM ACM_RegimeDefinitions WHERE EquipID = i.EquipID), 0) + 1,
        i.NumRegimes,
        ...
    FROM inserted i;
END;
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `ACM_RegimeDefinitions` table schema | `scripts/sql/migrations/` | ⏳ |
| [ ] Implement write-once semantics (immutable) | `core/regime_definitions.py` | ⏳ |
| [ ] Store centroids, boundaries, labels | `core/regime_definitions.py` | ⏳ |
| [ ] Store transition matrix | `core/regime_definitions.py` | ⏳ |
| [ ] Add version column with auto-increment | `scripts/sql/migrations/` | ⏳ |

### P2.6 — RegimeVersion on All Writes (Item 12)

**Changes to ACM_RegimeTimeline**:
```sql
-- scripts/sql/migrations/v11/012_alter_regime_timeline.sql
ALTER TABLE ACM_RegimeTimeline 
ADD RegimeVersion INT NULL,
    AssignmentConfidence FLOAT NULL;

-- Update existing rows with default
UPDATE ACM_RegimeTimeline SET RegimeVersion = 0 WHERE RegimeVersion IS NULL;
```

| Task | File | Status |
|------|------|--------|
| [ ] Add `RegimeVersion` column to `ACM_RegimeTimeline` | `scripts/sql/migrations/` | ⏳ |
| [ ] Add `AssignmentConfidence` column | `scripts/sql/migrations/` | ⏳ |
| [ ] Update all regime timeline writes | `core/regimes.py` | ⏳ |
| [ ] Update all regime timeline queries | `core/*.py` | ⏳ |

### P2.7 — MaturityState Gating (Item 13)

**Implementation**:
```python
# In core/adaptive_thresholds.py

def get_thresholds(self, equip_id: int, regime_label: int) -> Optional[Thresholds]:
    """Get thresholds only if regime model is CONVERGED."""
    active = self.active_models_manager.get_active(equip_id)
    
    if active.regime_maturity != MaturityState.CONVERGED:
        Console.info(f"Using global thresholds: regime not CONVERGED ({active.regime_maturity})")
        return self._get_global_thresholds(equip_id)
    
    return self._get_regime_thresholds(equip_id, regime_label, active.regime_version)

# In core/forecast_engine.py

def run_forecast(self, ...):
    active = self.active_models_manager.get_active(equip_id)
    
    if active.regime_maturity != MaturityState.CONVERGED:
        Console.info(f"Skipping regime-conditioned forecast: not CONVERGED")
        return self._run_global_forecast(...)
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `MaturityState` enum (INITIALIZING, LEARNING, CONVERGED, DEPRECATED) | `core/regime_manager.py` | ⏳ |
| [ ] Gate regime-conditioned thresholds on CONVERGED | `core/adaptive_thresholds.py` | ⏳ |
| [ ] Gate regime-conditioned forecasting on CONVERGED | `core/forecast_engine.py` | ⏳ |
| [ ] Add maturity state to `ACM_ActiveModels` | `scripts/sql/migrations/` | ⏳ |

### P2.8 — Offline Historical Replay (Item 14)

**Implementation**:
```python
# scripts/offline_replay.py (NEW FILE)

"""
Offline Historical Replay for Regime Discovery

This script runs regime discovery on accumulated history without affecting
production models. New regime versions are written but not promoted to active.

Usage:
    python scripts/offline_replay.py --equip FD_FAN --start 2024-01-01 --end 2024-12-31
    python scripts/offline_replay.py --equip FD_FAN --all-history
"""

import argparse
from datetime import datetime
from core.sql_client import SQLClient
from core.regimes import discover_regimes
from core.regime_manager import ActiveModelsManager, MaturityState
from core.observability import Console

def main():
    parser = argparse.ArgumentParser(description="Offline regime discovery")
    parser.add_argument("--equip", required=True, help="Equipment name")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--all-history", action="store_true", help="Use all available history")
    parser.add_argument("--promote", action="store_true", help="Promote to LEARNING if discovery succeeds")
    args = parser.parse_args()
    
    sql = SQLClient()
    
    # Load historical data
    if args.all_history:
        query = """
            SELECT * FROM ACM_HealthTimeline 
            WHERE EquipID = (SELECT EquipID FROM Equipment WHERE EquipCode = ?)
            ORDER BY Timestamp
        """
        df = sql.query(query, (args.equip,))
    else:
        start = datetime.fromisoformat(args.start)
        end = datetime.fromisoformat(args.end)
        query = """
            SELECT * FROM ACM_HealthTimeline 
            WHERE EquipID = (SELECT EquipID FROM Equipment WHERE EquipCode = ?)
              AND Timestamp BETWEEN ? AND ?
            ORDER BY Timestamp
        """
        df = sql.query(query, (args.equip, start, end))
    
    Console.info(f"Loaded {len(df)} rows for regime discovery")
    
    # Run regime discovery
    regime_model = discover_regimes(df, ...)
    
    # Write new version (but don't promote)
    new_version = write_regime_definitions(sql, equip_id, regime_model)
    Console.ok(f"Created RegimeVersion {new_version}")
    
    if args.promote:
        manager = ActiveModelsManager(sql)
        manager.promote_regime(equip_id, new_version, MaturityState.LEARNING)
        Console.ok(f"Promoted to LEARNING state")
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `scripts/offline_replay.py` runner | `scripts/offline_replay.py` | ⏳ |
| [ ] Load accumulated history from SQL | `scripts/offline_replay.py` | ⏳ |
| [ ] Run regime discovery on full history | `scripts/offline_replay.py` | ⏳ |
| [ ] Write new `RegimeVersion` without affecting production | `scripts/offline_replay.py` | ⏳ |
| [ ] Add CLI arguments for date range, equipment | `scripts/offline_replay.py` | ⏳ |

### P2.9 — Regime Evaluation Metrics (Item 15)

**Implementation**:
```python
# core/regime_evaluation.py (NEW FILE)

from dataclasses import dataclass
import numpy as np
from scipy.stats import entropy

@dataclass
class RegimeMetrics:
    """Evaluation metrics for a regime model."""
    stability: float        # % of points that don't change regime within 1 hour
    novelty_rate: float     # % of points assigned to UNKNOWN/EMERGING
    overlap_entropy: float  # Entropy of regime overlap (lower = better separation)
    transition_entropy: float  # Entropy of regime transitions (lower = more predictable)
    consistency: float      # Reproducibility score (0-1)
    silhouette_score: float # Cluster quality

class RegimeEvaluator:
    """Evaluates regime model quality."""
    
    def evaluate(self, assignments: np.ndarray, timestamps: np.ndarray,
                 centroids: np.ndarray) -> RegimeMetrics:
        
        # Stability: How often does regime stay same over time?
        changes = np.sum(assignments[1:] != assignments[:-1])
        stability = 1 - changes / len(assignments)
        
        # Novelty rate: How many UNKNOWN/EMERGING?
        novelty_rate = np.mean((assignments == -1) | (assignments == -2))
        
        # Transition entropy
        transitions = self._compute_transition_matrix(assignments)
        transition_entropy = np.mean([entropy(row) for row in transitions if row.sum() > 0])
        
        # Overlap entropy (from centroid distances)
        overlap_entropy = self._compute_overlap_entropy(centroids)
        
        return RegimeMetrics(
            stability=stability,
            novelty_rate=novelty_rate,
            overlap_entropy=overlap_entropy,
            transition_entropy=transition_entropy,
            consistency=0.0,  # Computed via replay test
            silhouette_score=0.0  # Computed from clustering
        )
```

**SQL Schema**:
```sql
CREATE TABLE ACM_RegimeMetrics (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    EquipID INT NOT NULL,
    RegimeVersion INT NOT NULL,
    Stability FLOAT NOT NULL,
    NoveltyRate FLOAT NOT NULL,
    OverlapEntropy FLOAT NOT NULL,
    TransitionEntropy FLOAT NOT NULL,
    Consistency FLOAT NOT NULL,
    SilhouetteScore FLOAT NOT NULL,
    EvaluatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_RegimeMetrics_Definitions FOREIGN KEY (EquipID, RegimeVersion) 
        REFERENCES ACM_RegimeDefinitions(EquipID, RegimeVersion)
);
```

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

**Implementation**:
```python
# core/regime_promotion.py (NEW FILE)

@dataclass
class PromotionCriteria:
    """Criteria for promoting regime model from LEARNING to CONVERGED."""
    min_stability: float = 0.85      # 85% stable
    max_novelty_rate: float = 0.10   # <10% unknown
    max_overlap_entropy: float = 1.0 # Low overlap
    min_sample_count: int = 1000     # Minimum evaluations
    min_days_in_learning: int = 7    # At least 7 days

class RegimePromoter:
    """Handles promotion workflow for regime models."""
    
    def evaluate_for_promotion(self, equip_id: int, 
                                version: int) -> Tuple[bool, List[str]]:
        """Check if regime version meets promotion criteria."""
        metrics = self._get_metrics(equip_id, version)
        criteria = self._get_criteria(equip_id)
        
        failures = []
        
        if metrics.stability < criteria.min_stability:
            failures.append(f"Stability {metrics.stability:.2f} < {criteria.min_stability}")
        
        if metrics.novelty_rate > criteria.max_novelty_rate:
            failures.append(f"Novelty rate {metrics.novelty_rate:.2f} > {criteria.max_novelty_rate}")
        
        # ... more checks
        
        return len(failures) == 0, failures
    
    def promote(self, equip_id: int, version: int) -> None:
        """Promote regime version to CONVERGED."""
        can_promote, failures = self.evaluate_for_promotion(equip_id, version)
        
        if not can_promote:
            raise ValueError(f"Cannot promote: {failures}")
        
        # Log promotion
        self._log_promotion(equip_id, version, MaturityState.CONVERGED)
        
        # Update active models
        self.active_models_manager.promote_regime(equip_id, version, MaturityState.CONVERGED)
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `RegimePromoter` class | `core/regime_promotion.py` | ⏳ |
| [ ] Define acceptance criteria | `core/regime_promotion.py` | ⏳ |
| [ ] Implement promotion from LEARNING → CONVERGED | `core/regime_promotion.py` | ⏳ |
| [ ] Update `ACM_ActiveModels` on promotion | `core/regime_promotion.py` | ⏳ |
| [ ] Create `ACM_RegimePromotionLog` audit table | `scripts/sql/migrations/` | ⏳ |

### P2.11 — Confidence-Gated Normalization (Item 17)

**Implementation**:
```python
# In core/fast_features.py

def normalize_features(self, df: pd.DataFrame, regime_label: int, 
                       confidence: float, regime_version: int) -> pd.DataFrame:
    """
    Normalize features with confidence-gated regime conditioning.
    
    If confidence < threshold, fall back to global normalization.
    """
    CONFIDENCE_THRESHOLD = 0.7
    
    if confidence < CONFIDENCE_THRESHOLD or regime_label < 0:
        Console.info(f"Using global normalization: confidence={confidence:.2f}")
        return self._normalize_global(df)
    
    return self._normalize_regime(df, regime_label, regime_version)
```

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

**Current State Analysis**:
- **AR1Detector** (`ar1_detector.py`): Has `fit()` and `score()` methods properly separated. Stores train-time residual std. ✅ Good separation.
- **IsolationForestDetector** (`outliers.py`): Has `fit()` and `score()`. Uses `baseline_score_` calibrated from training. ✅ Good separation.
- **GMMDetector** (`outliers.py`): Has `fit()` and `score()`. Uses `baseline_mean_` and `baseline_std_` from training. ✅ Good separation.
- **OMRDetector** (`omr.py`): Has `fit()` and `score()`. Stores `residual_std_`. ✅ Good separation.
- **PCADetector** (`outliers.py`): Has `fit()` and `score()`. Uses training PCA for SPE/T2 thresholds. ✅ Good separation.

**Issues Found**:
- No formal `DetectorProtocol` ABC enforcing API consistency
- Each detector has its own normalization approach (some return z-scores, some raw scores)
- Detector outputs not standardized - inconsistent column naming
- `fuse.py` uses episode separability metrics but no explicit disagreement penalty
- Missing detector fallback when detectors are unavailable

### P3.1 — Train-Score Separation (Item 8)

**Contract Definition**:
```python
"""
TRAIN-SCORE SEPARATION CONTRACT

1. The fit_baseline() method learns ONLY from training data (baseline period)
2. The score() method uses ONLY parameters learned during fit_baseline()
3. A batch being scored CANNOT influence its own anomaly scores
4. All normalization parameters (mean, std, thresholds) come from training data

VIOLATIONS:
- Using score batch mean/std for normalization
- Updating model parameters during scoring
- Adaptive thresholds based on current batch
"""
```

**Audit Checklist**:
```python
# For each detector, verify:
# 1. fit_baseline() stores all needed statistics
# 2. score() uses ONLY stored statistics
# 3. No calls to .fit() or .partial_fit() in score()
# 4. No batch statistics used in normalization

# Example audit for AR1:
class AR1Detector:
    def fit_baseline(self, X_train):
        self.coefficients_ = ...  # Learned from X_train
        self.residual_std_ = ...  # Learned from X_train
        self.residual_mean_ = ... # Learned from X_train
        
    def score(self, X_score):
        residuals = self._compute_residuals(X_score, self.coefficients_)
        z_scores = (residuals - self.residual_mean_) / self.residual_std_
        # ✅ Uses ONLY stored statistics, not batch statistics
        return z_scores
```

**Audit Summary (2025-01-08)**: All existing detectors PASS train-score separation:
- **AR1Detector** (`core/ar1_detector.py`): `phimap`, `sdmap` stored at fit time, used in score()
- **PCASubspaceDetector** (`core/correlation.py`): `pca`, `scaler`, `col_medians` from training
- **IsolationForestDetector** (`core/outliers.py`): sklearn model trained once, score_samples() used
- **GMMDetector** (`core/outliers.py`): model, scaler, _score_mu_, _score_sd_ from training
- **OMRDetector** (`core/omr.py`): OMRModel contains train_residual_std for z-scoring
- **ScoreCalibrator** (`core/fuse.py`): med, mad, scale, regime_params_ stored at fit time

| Task | File | Status |
|------|------|--------|
| [x] Define separation contract (batch cannot influence own score) | `docs/V11_REFACTOR_TRACKER.md` | ✅ |
| [x] Audit AR1 detector for separation | `core/ar1_detector.py` | ✅ |
| [x] Audit PCA detector for separation | `core/correlation.py` | ✅ |
| [x] Audit IForest detector for separation | `core/outliers.py` | ✅ |
| [x] Audit GMM detector for separation | `core/outliers.py` | ✅ |
| [x] Audit OMR detector for separation | `core/omr.py` | ✅ |
| [x] Add separation validation in tests | `tests/test_detector_protocol.py` | ✅ |

### P3.2 — Unified Baseline Normalization (Item 20)

**Implementation**:
```python
# core/baseline_normalizer.py (NEW FILE)

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import pandas as pd

@dataclass
class SensorBaseline:
    """Baseline statistics for a single sensor."""
    mean: float
    std: float
    median: float
    mad: float  # Median Absolute Deviation
    min: float
    max: float
    p05: float  # 5th percentile
    p95: float  # 95th percentile

@dataclass
class BaselineStatistics:
    """Baseline statistics for all sensors."""
    sensor_stats: Dict[str, SensorBaseline] = field(default_factory=dict)
    computed_at: pd.Timestamp = None
    n_samples: int = 0
    
class BaselineNormalizer:
    """
    Centralized baseline normalization.
    
    Computes statistics once from training data and applies consistently
    across all detectors and modules.
    """
    
    def __init__(self):
        self.baseline: BaselineStatistics = None
        self._fitted = False
    
    def fit(self, X_train: pd.DataFrame, sensor_cols: List[str]) -> "BaselineNormalizer":
        """Compute baseline statistics from training data."""
        stats = {}
        
        for col in sensor_cols:
            series = X_train[col].dropna()
            if len(series) == 0:
                continue
            
            stats[col] = SensorBaseline(
                mean=series.mean(),
                std=series.std(),
                median=series.median(),
                mad=(series - series.median()).abs().median(),
                min=series.min(),
                max=series.max(),
                p05=series.quantile(0.05),
                p95=series.quantile(0.95)
            )
        
        self.baseline = BaselineStatistics(
            sensor_stats=stats,
            computed_at=pd.Timestamp.now(),
            n_samples=len(X_train)
        )
        self._fitted = True
        return self
    
    def normalize(self, X: pd.DataFrame, method: str = "z-score") -> pd.DataFrame:
        """
        Normalize data using baseline statistics.
        
        Methods:
        - "z-score": (x - mean) / std
        - "robust": (x - median) / mad
        - "minmax": (x - min) / (max - min)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before normalize()")
        
        result = X.copy()
        
        for col, stats in self.baseline.sensor_stats.items():
            if col not in result.columns:
                continue
            
            if method == "z-score":
                result[col] = (result[col] - stats.mean) / (stats.std + 1e-10)
            elif method == "robust":
                result[col] = (result[col] - stats.median) / (stats.mad + 1e-10)
            elif method == "minmax":
                result[col] = (result[col] - stats.min) / (stats.max - stats.min + 1e-10)
        
        return result
    
    def to_dict(self) -> dict:
        """Serialize for SQL storage."""
        return {
            "sensor_stats": {k: asdict(v) for k, v in self.baseline.sensor_stats.items()},
            "computed_at": self.baseline.computed_at.isoformat(),
            "n_samples": self.baseline.n_samples
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BaselineNormalizer":
        """Deserialize from SQL storage."""
        normalizer = cls()
        normalizer.baseline = BaselineStatistics(
            sensor_stats={k: SensorBaseline(**v) for k, v in data["sensor_stats"].items()},
            computed_at=pd.Timestamp(data["computed_at"]),
            n_samples=data["n_samples"]
        )
        normalizer._fitted = True
        return normalizer
```

**Integration in pipeline**:
```python
# In core/acm_main.py

# Compute baseline normalization ONCE at pipeline start
baseline_normalizer = BaselineNormalizer().fit(baseline_df, sensor_cols)

# Pass to all detectors
ar1_detector.fit_baseline(baseline_normalizer.normalize(baseline_df))
pca_detector.fit_baseline(baseline_normalizer.normalize(baseline_df))
# etc.
```

| Task | File | Status |
|------|------|--------|
| [x] Create `BaselineNormalizer` class | `core/baseline_normalizer.py` | ✅ |
| [ ] Remove detector-specific normalization | `core/ar1_detector.py` | ⏳ |
| [ ] Remove detector-specific normalization | `core/outliers.py` | ⏳ |
| [ ] Integrate normalizer into pipeline | `core/acm_main.py` | ⏳ |

### P3.3 — Strict Detector Protocol (Item 21)

**Implementation**:
```python
# core/detector_protocol.py (NEW FILE)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np

@dataclass
class DetectorOutput:
    """Standardized output from all detectors."""
    timestamp: pd.Series           # Timestamp column
    z_score: pd.Series             # Standardized anomaly score
    raw_score: pd.Series           # Raw detector-specific score
    is_anomaly: pd.Series          # Boolean anomaly flag
    confidence: pd.Series          # Confidence in the score (0-1)
    detector_name: str             # Name of detector
    feature_contributions: Optional[pd.DataFrame] = None  # Per-feature contributions

class DetectorProtocol(ABC):
    """
    Abstract base class for all ACM detectors.
    
    All detectors MUST implement this interface to ensure:
    1. Consistent API across all detectors
    2. Train-score separation
    3. Standardized output format
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Detector name for logging and output columns."""
        pass
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether detector has been fitted."""
        pass
    
    @abstractmethod
    def fit_baseline(self, X_train: pd.DataFrame) -> "DetectorProtocol":
        """
        Fit detector on baseline (training) data.
        
        MUST learn all parameters from X_train only.
        MUST NOT be called during scoring.
        
        Args:
            X_train: Baseline data with normalized sensor columns
            
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def score(self, X_score: pd.DataFrame) -> DetectorOutput:
        """
        Score new data using parameters from fit_baseline().
        
        MUST use only parameters learned during fit_baseline().
        MUST NOT update any model parameters.
        MUST NOT use batch statistics for normalization.
        
        Args:
            X_score: Data to score (same columns as X_train)
            
        Returns:
            DetectorOutput with standardized scores
        """
        pass
    
    def get_params(self) -> dict:
        """Get fitted parameters for serialization."""
        return {}
    
    def set_params(self, params: dict) -> None:
        """Set parameters from deserialization."""
        pass
    
    def validate_input(self, X: pd.DataFrame) -> None:
        """Validate input data structure."""
        if X.empty:
            raise ValueError("Input DataFrame is empty")
```

**Refactored AR1 Example**:
```python
# core/ar1_detector.py (refactored)

from core.detector_protocol import DetectorProtocol, DetectorOutput

class AR1Detector(DetectorProtocol):
    """AR(1) Autoregressive anomaly detector."""
    
    @property
    def name(self) -> str:
        return "ar1"
    
    @property
    def is_fitted(self) -> bool:
        return hasattr(self, 'coefficients_') and self.coefficients_ is not None
    
    def fit_baseline(self, X_train: pd.DataFrame) -> "AR1Detector":
        # Learn AR(1) coefficients from training data
        self.coefficients_ = self._fit_ar1(X_train)
        
        # Compute baseline residual statistics
        residuals = self._compute_residuals(X_train)
        self.residual_mean_ = residuals.mean()
        self.residual_std_ = residuals.std()
        
        return self
    
    def score(self, X_score: pd.DataFrame) -> DetectorOutput:
        if not self.is_fitted:
            raise RuntimeError("Must call fit_baseline() before score()")
        
        # Compute residuals using learned coefficients
        residuals = self._compute_residuals(X_score)
        
        # Normalize using BASELINE statistics (not batch statistics)
        z_scores = (residuals - self.residual_mean_) / (self.residual_std_ + 1e-10)
        
        return DetectorOutput(
            timestamp=X_score["Timestamp"],
            z_score=z_scores,
            raw_score=residuals,
            is_anomaly=z_scores.abs() > 3.0,
            confidence=1 - z_scores.abs().clip(0, 10) / 10,
            detector_name=self.name
        )
```

| Task | File | Status |
|------|------|--------|
| [x] Create `DetectorProtocol` ABC | `core/detector_protocol.py` | ✅ |
| [x] Define `fit_baseline(X_train)` method | `core/detector_protocol.py` | ✅ |
| [x] Define `score(X_score) -> DataFrame` method | `core/detector_protocol.py` | ✅ |
| [x] Define output schema (z_score, raw_score, etc.) | `core/detector_protocol.py` | ✅ |
| [ ] Refactor AR1 to implement protocol | `core/ar1_detector.py` | ⏳ |
| [ ] Refactor PCA to implement protocol | `core/outliers.py` | ⏳ |
| [ ] Refactor IForest to implement protocol | `core/outliers.py` | ⏳ |
| [ ] Refactor GMM to implement protocol | `core/outliers.py` | ⏳ |
| [ ] Refactor OMR to implement protocol | `core/omr.py` | ⏳ |

### P3.4 — Calibrated Fusion (Item 22)

**Implementation**:
```python
# core/fuse.py (enhanced)

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

@dataclass
class FusionWeights:
    """Calibrated weights for detector fusion."""
    detector_weights: Dict[str, float]  # detector_name -> weight
    calibration_method: str  # "equal", "performance", "inverse_correlation"
    calibrated_at: pd.Timestamp

@dataclass
class FusionResult:
    """Result of multi-detector fusion."""
    fused_z: pd.Series           # Combined anomaly score
    confidence: pd.Series        # Fusion confidence (0-1)
    detector_agreement: pd.Series  # How much detectors agree (0-1)
    contributing_detectors: List[str]  # Which detectors contributed

class CalibratedFusion:
    """
    Calibrated evidence combiner for multi-detector fusion.
    
    Improvements over v10:
    1. Explicit NaN handling with confidence dampening
    2. Detector disagreement penalty
    3. Calibrated weights based on historical performance
    """
    
    def __init__(self, weights: Optional[FusionWeights] = None):
        self.weights = weights or self._default_weights()
        self.nan_penalty = 0.2  # Reduce confidence by 20% per missing detector
        self.disagreement_penalty = 0.1  # Reduce confidence when detectors disagree
    
    def fuse(self, detector_outputs: Dict[str, DetectorOutput]) -> FusionResult:
        """
        Fuse multiple detector outputs into single anomaly score.
        """
        if not detector_outputs:
            raise ValueError("No detector outputs to fuse")
        
        # Collect z-scores
        z_scores = {}
        for name, output in detector_outputs.items():
            z_scores[name] = output.z_score
        
        z_df = pd.DataFrame(z_scores)
        
        # Count missing values per row
        missing_count = z_df.isna().sum(axis=1)
        total_detectors = len(detector_outputs)
        
        # Weighted average (ignoring NaN)
        weights = np.array([self.weights.detector_weights.get(n, 1.0) for n in z_df.columns])
        
        # Weighted mean ignoring NaN
        weighted_sum = (z_df * weights).sum(axis=1, skipna=True)
        weight_sum = (~z_df.isna() * weights).sum(axis=1)
        fused_z = weighted_sum / (weight_sum + 1e-10)
        
        # Compute detector agreement (inverse of variance)
        z_std = z_df.std(axis=1, skipna=True)
        agreement = 1 / (1 + z_std)  # High std = low agreement
        
        # Compute confidence with penalties
        base_confidence = agreement.copy()
        
        # Penalty for missing detectors
        missing_penalty = missing_count * self.nan_penalty
        base_confidence -= missing_penalty
        
        # Penalty for disagreement
        disagreement = z_std > 2.0  # Significant disagreement
        base_confidence -= disagreement * self.disagreement_penalty
        
        confidence = base_confidence.clip(0, 1)
        
        return FusionResult(
            fused_z=fused_z,
            confidence=confidence,
            detector_agreement=agreement,
            contributing_detectors=list(detector_outputs.keys())
        )
    
    def calibrate_weights(self, historical_performance: pd.DataFrame) -> FusionWeights:
        """
        Calibrate weights based on historical detector performance.
        
        Uses inverse of false positive rate and detection accuracy.
        """
        weights = {}
        for detector in historical_performance["detector"].unique():
            detector_data = historical_performance[historical_performance["detector"] == detector]
            accuracy = detector_data["true_positive_rate"].mean()
            fpr = detector_data["false_positive_rate"].mean()
            
            # Weight based on accuracy and inverse FPR
            weight = accuracy * (1 - fpr)
            weights[detector] = max(0.1, weight)  # Minimum weight
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        return FusionWeights(
            detector_weights=weights,
            calibration_method="performance",
            calibrated_at=pd.Timestamp.now()
        )
```

| Task | File | Status |
|------|------|--------|
| [x] Redesign fusion as calibrated evidence combiner | `core/calibrated_fusion.py` | ✅ |
| [x] Add explicit missingness handling (NaN → confidence dampening) | `core/calibrated_fusion.py` | ✅ |
| [x] Add detector weight calibration | `core/calibrated_fusion.py` | ✅ |
| [x] Add disagreement penalty | `core/calibrated_fusion.py` | ✅ |

### P3.5 — Per-Run Fusion Quality (Item 23)

**SQL Schema**:
```sql
-- scripts/sql/migrations/v11/030_acm_fusion_quality.sql
CREATE TABLE ACM_FusionQuality (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    RunID INT NOT NULL,
    EquipID INT NOT NULL,
    
    -- Detector participation
    DetectorsUsed NVARCHAR(500) NOT NULL,  -- Comma-separated list
    DetectorCount INT NOT NULL,
    MissingDetectors NVARCHAR(500) NULL,
    
    -- Quality metrics
    AgreementMean FLOAT NOT NULL,          -- Mean detector agreement
    AgreementStd FLOAT NOT NULL,           -- Std of agreement
    ConfidenceMean FLOAT NOT NULL,         -- Mean fusion confidence
    DisagreementRate FLOAT NOT NULL,       -- % of rows with significant disagreement
    
    -- Weight information
    WeightsUsed NVARCHAR(MAX) NULL,        -- JSON: {detector: weight}
    CalibrationMethod NVARCHAR(50) NULL,
    
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_FusionQuality_Runs FOREIGN KEY (RunID) REFERENCES ACM_Runs(RunID)
);
```

| Task | File | Status |
|------|------|--------|
| [x] Create `FusionQualityMetrics` class | `core/calibrated_fusion.py` | ✅ |
| [x] Track which detectors contributed | `core/calibrated_fusion.py` | ✅ |
| [x] Track detector agreement level | `core/calibrated_fusion.py` | ✅ |
| [x] Track confidence impact | `core/calibrated_fusion.py` | ✅ |
| [ ] Create `ACM_FusionQuality` table | `scripts/sql/migrations/` | ⏳ |
| [ ] Persist fusion quality per run | `core/output_manager.py` | ⏳ |

### P3.6 — Detector Correlation Tracking (Item 33)

**Implementation**:
```python
# core/detector_correlation.py (NEW FILE)

from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

@dataclass
class CorrelationResult:
    """Pairwise correlation between detectors."""
    detector1: str
    detector2: str
    correlation: float
    is_redundant: bool  # correlation > 0.95

class DetectorCorrelation:
    """Track and flag redundant detectors."""
    
    REDUNDANCY_THRESHOLD = 0.95
    INSTABILITY_THRESHOLD = 0.5  # Correlation variance over time
    
    def compute_correlations(self, detector_outputs: Dict[str, DetectorOutput]) -> List[CorrelationResult]:
        """Compute pairwise correlations between all detectors."""
        z_df = pd.DataFrame({name: output.z_score for name, output in detector_outputs.items()})
        
        results = []
        detectors = list(z_df.columns)
        
        for i, d1 in enumerate(detectors):
            for d2 in detectors[i+1:]:
                corr = z_df[d1].corr(z_df[d2])
                results.append(CorrelationResult(
                    detector1=d1,
                    detector2=d2,
                    correlation=corr,
                    is_redundant=abs(corr) > self.REDUNDANCY_THRESHOLD
                ))
        
        return results
    
    def identify_redundant_detectors(self, results: List[CorrelationResult]) -> List[str]:
        """Identify detectors that should be considered for removal."""
        redundant = set()
        for r in results:
            if r.is_redundant:
                # Flag the "simpler" detector (heuristic: alphabetical)
                redundant.add(max(r.detector1, r.detector2))
        return list(redundant)
```

**SQL Schema**:
```sql
CREATE TABLE ACM_DetectorCorrelation (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    RunID INT NOT NULL,
    EquipID INT NOT NULL,
    Detector1 NVARCHAR(50) NOT NULL,
    Detector2 NVARCHAR(50) NOT NULL,
    Correlation FLOAT NOT NULL,
    IsRedundant BIT NOT NULL,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_DetectorCorrelation_Runs FOREIGN KEY (RunID) REFERENCES ACM_Runs(RunID)
);
```

| Task | File | Status |
|------|------|--------|
| [x] Create `DetectorCorrelation` class | `core/calibrated_fusion.py` | ✅ |
| [x] Track pairwise correlations per run | `core/calibrated_fusion.py` | ✅ |
| [x] Flag redundant detectors (correlation > 0.95) | `core/calibrated_fusion.py` | ✅ |
| [x] Flag unstable detectors (high variance) | `core/calibrated_fusion.py` | ✅ |
| [ ] Create `ACM_DetectorCorrelation` table | `scripts/sql/migrations/` | ⏳ |

---

## Phase 4: Health/Episode/RUL Redesign

**Goal**: Make episodes the only alerting primitive, redefine health as time-evolving state

**Current State Analysis**:
- **Episode Detection** (`episode_culprits_writer.py`): Episodes built from consecutive anomaly points, but point-level alerts still exist elsewhere
- **Health Tracker** (`health_tracker.py`): Computes health as simple weighted average of detector z-scores
- **RUL Estimator** (`rul_estimator.py`): Monte Carlo forecasting, but no explicit reliability gate - can return RUL even with insufficient data
- **Confidence**: No unified confidence model - each component has its own approach

**Issues Found**:
- Point-level anomalies in `ACM_Scores_Wide` can trigger alerts independently of episodes
- Health has no state machine (HEALTHY→DEGRADED→CRITICAL transitions not formal)
- RUL can be computed with 5 data points - no minimum data requirements
- No recovery hysteresis - health can oscillate rapidly
- No explicit "I don't know" state for RUL

### P4.1 — Episode-Only Alerting (Item 6)

**Implementation**:
```python
# core/episode_manager.py (NEW FILE)

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import pandas as pd
import numpy as np

class EpisodeSeverity(Enum):
    """Episode severity levels."""
    LOW = 1       # Anomaly detected, likely noise
    MEDIUM = 2    # Persistent anomaly, monitor closely
    HIGH = 3      # Significant deviation, investigate
    CRITICAL = 4  # Imminent failure risk

class EpisodeStatus(Enum):
    """Episode lifecycle status."""
    ACTIVE = "ACTIVE"         # Currently ongoing
    RESOLVED = "RESOLVED"     # Returned to normal
    SUPPRESSED = "SUPPRESSED" # Operator dismissed
    ESCALATED = "ESCALATED"   # Escalated to higher tier

@dataclass
class Episode:
    """The SOLE alerting primitive in ACM v11."""
    id: str
    equip_id: int
    start_time: pd.Timestamp
    end_time: Optional[pd.Timestamp]
    severity: EpisodeSeverity
    status: EpisodeStatus
    
    # Evidence
    peak_z_score: float
    mean_z_score: float
    duration_hours: float
    affected_sensors: List[str]
    
    # Attribution
    top_contributors: List[str]  # Ranked sensor contributions
    regime_at_onset: int
    detector_agreement: float  # How much detectors agreed
    
    # Confidence
    confidence: float  # 0-1, how confident we are this is real
    false_positive_probability: float

class EpisodeManager:
    """
    Central episode lifecycle manager.
    
    CRITICAL: This is the ONLY way alerts are generated in ACM v11.
    Point-level anomalies are NEVER surfaced to operators.
    """
    
    def __init__(self, config: Dict):
        self.min_duration_minutes = config.get("episode.min_duration_minutes", 30)
        self.severity_thresholds = {
            EpisodeSeverity.LOW: config.get("episode.threshold.low", 3.0),
            EpisodeSeverity.MEDIUM: config.get("episode.threshold.medium", 4.0),
            EpisodeSeverity.HIGH: config.get("episode.threshold.high", 5.5),
            EpisodeSeverity.CRITICAL: config.get("episode.threshold.critical", 7.0),
        }
        self.cooldown_hours = config.get("episode.cooldown_hours", 6)
        self.active_episodes: Dict[int, Episode] = {}  # equip_id -> episode
    
    def detect_episodes(self, 
                        fused_z: pd.DataFrame,
                        detector_outputs: Dict[str, pd.DataFrame],
                        regime_labels: pd.Series) -> List[Episode]:
        """
        Detect episodes from fused anomaly scores.
        
        Steps:
        1. Find contiguous regions where fused_z > threshold
        2. Filter by minimum duration
        3. Compute severity based on peak score
        4. Extract sensor attribution
        5. Compute confidence
        """
        episodes = []
        threshold = self.severity_thresholds[EpisodeSeverity.LOW]
        
        # Find anomaly regions
        is_anomaly = fused_z["fused_z"] > threshold
        
        # Label contiguous regions
        region_id = (is_anomaly != is_anomaly.shift()).cumsum()
        region_id[~is_anomaly] = 0
        
        for rid in region_id.unique():
            if rid == 0:
                continue
            
            mask = region_id == rid
            region_data = fused_z[mask]
            
            # Check minimum duration
            duration = (region_data["Timestamp"].max() - 
                       region_data["Timestamp"].min()).total_seconds() / 3600
            if duration * 60 < self.min_duration_minutes:
                continue
            
            # Compute severity
            peak_z = region_data["fused_z"].max()
            severity = self._compute_severity(peak_z)
            
            # Compute confidence from detector agreement
            agreement = region_data.get("detector_agreement", pd.Series([0.7])).mean()
            
            episode = Episode(
                id=f"EP_{rid}_{region_data['Timestamp'].iloc[0].strftime('%Y%m%d%H%M')}",
                equip_id=fused_z.get("EquipID", pd.Series([0])).iloc[0],
                start_time=region_data["Timestamp"].min(),
                end_time=region_data["Timestamp"].max() if not is_anomaly.iloc[-1] else None,
                severity=severity,
                status=EpisodeStatus.ACTIVE if is_anomaly.iloc[-1] else EpisodeStatus.RESOLVED,
                peak_z_score=peak_z,
                mean_z_score=region_data["fused_z"].mean(),
                duration_hours=duration,
                affected_sensors=self._get_affected_sensors(detector_outputs, mask),
                top_contributors=[],  # Filled by attribution
                regime_at_onset=int(regime_labels[mask].iloc[0]) if not regime_labels[mask].empty else -1,
                detector_agreement=agreement,
                confidence=agreement * (1 - 0.1 * max(0, 5 - peak_z)),
                false_positive_probability=1 - agreement
            )
            episodes.append(episode)
        
        return episodes
    
    def _compute_severity(self, peak_z: float) -> EpisodeSeverity:
        """Determine severity from peak z-score."""
        if peak_z >= self.severity_thresholds[EpisodeSeverity.CRITICAL]:
            return EpisodeSeverity.CRITICAL
        elif peak_z >= self.severity_thresholds[EpisodeSeverity.HIGH]:
            return EpisodeSeverity.HIGH
        elif peak_z >= self.severity_thresholds[EpisodeSeverity.MEDIUM]:
            return EpisodeSeverity.MEDIUM
        return EpisodeSeverity.LOW
```

**Migration Path**:
1. Create `EpisodeManager` class
2. Redirect all alerting through episodes
3. Remove point-level anomaly surfacing from dashboards
4. Update `ACM_Anomaly_Events` to use new Episode schema

| Task | File | Status |
|------|------|--------|
| [x] Create `EpisodeManager` class | `core/episode_manager.py` | ✅ |
| [x] Make episode construction the only alerting primitive | `core/episode_manager.py` | ✅ |
| [ ] Remove point-anomaly-driven alerts | `core/acm_main.py` | ⏳ |
| [ ] Refactor episode culprits writer | `core/episode_culprits_writer.py` | ⏳ |

### P4.2 — Time-Evolving Health State (Item 24)

**Implementation**:
```python
# core/health_state.py (NEW FILE)

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd

class HealthState(Enum):
    """Discrete health states with clear semantics."""
    HEALTHY = "HEALTHY"       # Normal operation, no action needed
    DEGRADED = "DEGRADED"     # Deviation detected, monitor
    CRITICAL = "CRITICAL"     # Significant issue, action needed
    UNKNOWN = "UNKNOWN"       # Insufficient data to determine
    RECOVERING = "RECOVERING" # Returning from degraded/critical

@dataclass
class HealthSnapshot:
    """Point-in-time health assessment."""
    timestamp: pd.Timestamp
    state: HealthState
    health_pct: float           # 0-100 for backwards compat
    confidence: float           # 0-1, how confident in this state
    state_duration_hours: float # How long in current state
    
    # Transition info
    previous_state: Optional[HealthState]
    transition_reason: Optional[str]
    
    # Underlying signals
    fused_z_mean: float
    active_episode_count: int
    worst_detector: str
    worst_detector_z: float

class HealthTracker:
    """
    Time-evolving health state machine.
    
    Key improvements:
    1. Explicit state machine with hysteresis
    2. Confidence attached to every state
    3. Recovery logic with cooldown
    4. State persistence across runs
    """
    
    # Transition thresholds (with hysteresis)
    THRESHOLDS = {
        (HealthState.HEALTHY, HealthState.DEGRADED): 3.0,
        (HealthState.DEGRADED, HealthState.HEALTHY): 2.0,  # Lower to go back
        (HealthState.DEGRADED, HealthState.CRITICAL): 5.5,
        (HealthState.CRITICAL, HealthState.DEGRADED): 4.0,  # Lower to go back
    }
    
    # Minimum time in state before transition (hours)
    MIN_STATE_DURATION = {
        HealthState.HEALTHY: 0,      # Can leave immediately
        HealthState.DEGRADED: 0.5,   # 30 min minimum
        HealthState.CRITICAL: 2.0,   # 2 hour minimum
        HealthState.RECOVERING: 6.0, # 6 hour cooldown
    }
    
    def __init__(self, initial_state: HealthState = HealthState.UNKNOWN):
        self.current_state = initial_state
        self.state_entered_at: Optional[pd.Timestamp] = None
        self.history: List[HealthSnapshot] = []
    
    def update(self, 
               timestamp: pd.Timestamp,
               fused_z: float,
               confidence: float,
               active_episodes: List) -> HealthSnapshot:
        """
        Update health state based on new evidence.
        
        Implements hysteresis: thresholds differ based on direction.
        """
        # Determine target state from z-score
        target_state = self._z_to_state(fused_z)
        
        # Check if transition allowed (duration, hysteresis)
        new_state = self._check_transition(target_state, timestamp)
        
        # Handle recovery state
        if (self.current_state == HealthState.CRITICAL and 
            new_state == HealthState.DEGRADED):
            new_state = HealthState.RECOVERING
        
        # Create snapshot
        snapshot = HealthSnapshot(
            timestamp=timestamp,
            state=new_state,
            health_pct=self._state_to_pct(new_state, fused_z),
            confidence=confidence,
            state_duration_hours=self._state_duration(timestamp),
            previous_state=self.current_state if new_state != self.current_state else None,
            transition_reason=f"fused_z={fused_z:.2f}" if new_state != self.current_state else None,
            fused_z_mean=fused_z,
            active_episode_count=len(active_episodes),
            worst_detector="",
            worst_detector_z=0.0
        )
        
        # Update state
        if new_state != self.current_state:
            self.current_state = new_state
            self.state_entered_at = timestamp
        
        self.history.append(snapshot)
        return snapshot
    
    def _z_to_state(self, z: float) -> HealthState:
        """Map z-score to target health state."""
        if z >= 5.5:
            return HealthState.CRITICAL
        elif z >= 3.0:
            return HealthState.DEGRADED
        else:
            return HealthState.HEALTHY
    
    def _state_to_pct(self, state: HealthState, z: float) -> float:
        """Convert state to backwards-compatible percentage."""
        if state == HealthState.HEALTHY:
            return 100 - min(z * 5, 20)  # 80-100%
        elif state == HealthState.DEGRADED:
            return 60 - min((z - 3) * 10, 30)  # 30-60%
        elif state == HealthState.CRITICAL:
            return max(0, 30 - (z - 5.5) * 10)  # 0-30%
        elif state == HealthState.RECOVERING:
            return 50  # Fixed during recovery
        return 50  # UNKNOWN
```

| Task | File | Status |
|------|------|--------|
| [x] Redefine health as time-evolving state | `core/health_state.py` | ✅ |
| [x] Add `HealthConfidence` field | `core/health_state.py` | ✅ |
| [x] Add state persistence across runs | `core/health_state.py` | ✅ |
| [x] Add `HealthState` enum (HEALTHY, DEGRADED, CRITICAL, UNKNOWN) | `core/health_state.py` | ✅ |

### P4.3 — Recovery Logic (Item 25)

**Implementation** (continuation of HealthTracker):
```python
# In core/health_state.py

def _check_transition(self, target: HealthState, timestamp: pd.Timestamp) -> HealthState:
    """
    Check if state transition is allowed.
    
    Implements:
    1. Minimum duration in current state
    2. Hysteresis (different thresholds for up/down)
    3. Cooldown after critical state
    """
    # If in UNKNOWN, allow any transition
    if self.current_state == HealthState.UNKNOWN:
        return target
    
    # Check minimum duration
    if self.state_entered_at:
        duration_hours = (timestamp - self.state_entered_at).total_seconds() / 3600
        min_duration = self.MIN_STATE_DURATION.get(self.current_state, 0)
        if duration_hours < min_duration:
            return self.current_state  # Stay in current state
    
    # RECOVERING must complete cooldown
    if self.current_state == HealthState.RECOVERING:
        cooldown_hours = self.MIN_STATE_DURATION[HealthState.RECOVERING]
        if duration_hours < cooldown_hours:
            return HealthState.RECOVERING
        # After cooldown, can transition
        return target
    
    return target

def exponential_decay_recovery(self, 
                               timestamps: pd.Series, 
                               fused_z: pd.Series,
                               decay_rate: float = 0.1) -> pd.Series:
    """
    Apply exponential decay for health recovery.
    
    After leaving CRITICAL, health recovers gradually:
    health(t) = health_target + (health_critical - health_target) * exp(-decay_rate * t)
    """
    if self.current_state != HealthState.RECOVERING:
        return self._state_to_pct(self.current_state, fused_z)
    
    hours_since_recovery = (timestamps - self.state_entered_at).dt.total_seconds() / 3600
    
    # Exponential approach to target health
    target_health = 70  # Target when fully recovered
    critical_health = 20  # Health when left critical
    
    health = target_health + (critical_health - target_health) * np.exp(-decay_rate * hours_since_recovery)
    return health
```

| Task | File | Status |
|------|------|--------|
| [x] Implement hysteresis for state transitions | `core/health_state.py` | ✅ |
| [x] Implement cooldown after critical state | `core/health_state.py` | ✅ |
| [x] Implement exponential decay for recovery | `core/health_state.py` | ✅ |
| [ ] Add configurable thresholds | `configs/config_table.csv` | ⏳ |

### P4.4 — RUL Reliability Gate (Item 7)

**Implementation**:
```python
# core/rul_reliability.py (NEW FILE)

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import pandas as pd

class RULStatus(Enum):
    """RUL prediction reliability status."""
    RELIABLE = "RELIABLE"               # High confidence prediction
    NOT_RELIABLE = "NOT_RELIABLE"       # Insufficient evidence
    INSUFFICIENT_DATA = "INSUFFICIENT"  # Not enough historical data
    NO_DEGRADATION = "NO_DEGRADATION"   # Equipment is healthy, no RUL needed
    REGIME_UNSTABLE = "REGIME_UNSTABLE" # Regime changes invalidate prediction

@dataclass
class RULPrerequisites:
    """Prerequisites for reliable RUL prediction."""
    min_data_points: int = 500
    min_degradation_episodes: int = 2
    min_health_trend_points: int = 50
    min_regime_stability_hours: int = 24
    max_data_gap_hours: int = 48
    min_detector_agreement: float = 0.6

@dataclass
class RULResult:
    """RUL prediction with reliability status."""
    status: RULStatus
    rul_hours: Optional[float]
    p10_lower: Optional[float]
    p50_median: Optional[float]
    p90_upper: Optional[float]
    confidence: float
    method: str
    
    # Diagnostic info
    prerequisite_failures: List[str]
    data_quality_score: float
    regime_stability_score: float

class RULReliabilityGate:
    """
    Gate RUL predictions on prerequisites.
    
    CRITICAL: If prerequisites fail, return RUL_NOT_RELIABLE instead of
    a numeric prediction that might mislead operators.
    """
    
    def __init__(self, prereqs: RULPrerequisites = None):
        self.prereqs = prereqs or RULPrerequisites()
    
    def check_prerequisites(self, 
                            data: pd.DataFrame,
                            episodes: List,
                            health_trend: pd.DataFrame,
                            current_regime: int,
                            regime_history: pd.Series) -> RULResult:
        """
        Check all prerequisites for reliable RUL.
        
        Returns RUL_NOT_RELIABLE with explanations if any fail.
        """
        failures = []
        
        # 1. Minimum data points
        if len(data) < self.prereqs.min_data_points:
            failures.append(f"Only {len(data)} data points (need {self.prereqs.min_data_points})")
        
        # 2. Minimum degradation episodes
        if len(episodes) < self.prereqs.min_degradation_episodes:
            failures.append(f"Only {len(episodes)} episodes (need {self.prereqs.min_degradation_episodes})")
        
        # 3. Minimum health trend points
        if len(health_trend) < self.prereqs.min_health_trend_points:
            failures.append(f"Only {len(health_trend)} health points (need {self.prereqs.min_health_trend_points})")
        
        # 4. Regime stability
        if current_regime == -1:  # UNKNOWN regime
            failures.append("Current regime is UNKNOWN")
        else:
            regime_duration = self._regime_duration_hours(regime_history, current_regime)
            if regime_duration < self.prereqs.min_regime_stability_hours:
                failures.append(f"Regime only stable for {regime_duration:.1f}h (need {self.prereqs.min_regime_stability_hours}h)")
        
        # 5. Data gaps
        max_gap = self._max_data_gap_hours(data)
        if max_gap > self.prereqs.max_data_gap_hours:
            failures.append(f"Data gap of {max_gap:.1f}h exceeds {self.prereqs.max_data_gap_hours}h")
        
        # 6. No degradation detected
        if len(episodes) == 0 and health_trend.get("health_pct", pd.Series([100])).mean() > 90:
            return RULResult(
                status=RULStatus.NO_DEGRADATION,
                rul_hours=None,
                p10_lower=None,
                p50_median=None,
                p90_upper=None,
                confidence=0.9,
                method="none",
                prerequisite_failures=[],
                data_quality_score=1.0,
                regime_stability_score=1.0
            )
        
        # Return failure result if any prerequisites failed
        if failures:
            return RULResult(
                status=RULStatus.NOT_RELIABLE,
                rul_hours=None,
                p10_lower=None,
                p50_median=None,
                p90_upper=None,
                confidence=0.0,
                method="none",
                prerequisite_failures=failures,
                data_quality_score=len(data) / self.prereqs.min_data_points,
                regime_stability_score=0.0
            )
        
        # All prerequisites passed
        return None  # Signal that RUL can be computed
    
    def _regime_duration_hours(self, regime_history: pd.Series, regime: int) -> float:
        """Calculate how long we've been in current regime."""
        if regime_history.empty:
            return 0
        
        # Find last regime change
        changes = regime_history != regime_history.shift()
        if not changes.any():
            return len(regime_history)  # Entire history is same regime
        
        last_change_idx = changes[changes].index[-1]
        return (regime_history.index[-1] - last_change_idx).total_seconds() / 3600
```

**SQL Schema Update**:
```sql
-- Add RULStatus column to ACM_RUL
ALTER TABLE ACM_RUL ADD 
    RULStatus NVARCHAR(20) NOT NULL DEFAULT 'RELIABLE',
    PrerequisiteFailures NVARCHAR(MAX) NULL,
    DataQualityScore FLOAT NULL,
    RegimeStabilityScore FLOAT NULL;
```

| Task | File | Status |
|------|------|--------|
| [x] Add `RUL_NOT_RELIABLE` outcome | `core/rul_reliability.py` | ✅ |
| [x] Define prerequisite checks | `core/rul_reliability.py` | ✅ |
| [x] Prevent numeric RUL when prerequisites fail | `core/rul_reliability.py` | ✅ |
| [x] Add `RULStatus` enum (RELIABLE, NOT_RELIABLE, INSUFFICIENT_DATA) | `core/rul_reliability.py` | ✅ |
| [ ] Update SQL writes to include status | `core/output_manager.py` | ⏳ |

### P4.5 — Forecasting Diagnostics (Item 36)

**Implementation**:
```python
# core/forecast_diagnostics.py (NEW FILE)

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class ForecastDiagnostics:
    """Diagnostic metrics for forecast quality assessment."""
    
    # Coverage: % of actual values within prediction interval
    coverage_p80: float  # % within 10-90 percentile bounds
    coverage_p50: float  # % within 25-75 percentile bounds
    
    # Sharpness: width of prediction intervals (narrower = better)
    sharpness_p80: float  # Average width of 80% interval
    sharpness_p50: float  # Average width of 50% interval
    
    # Calibration: how well probabilities match actual frequencies
    calibration_score: float  # 0-1, 1 = perfect calibration
    
    # Bias: systematic over/under prediction
    mean_bias: float  # Positive = overpredicting, negative = underpredicting
    
    # Skill score: improvement over naive baseline
    skill_score: float  # 0 = same as baseline, 1 = perfect
    
    n_forecasts: int
    n_validated: int  # Forecasts with ground truth available

class ForecastValidator:
    """Validate forecast quality with diagnostic metrics."""
    
    def compute_diagnostics(self,
                            forecasts: pd.DataFrame,
                            actuals: pd.DataFrame) -> ForecastDiagnostics:
        """
        Compute forecast diagnostics by comparing predictions to actuals.
        
        forecasts columns: timestamp, p10, p25, p50, p75, p90
        actuals columns: timestamp, actual_value
        """
        # Merge on timestamp
        merged = pd.merge(forecasts, actuals, on="timestamp", how="inner")
        
        if merged.empty:
            return ForecastDiagnostics(
                coverage_p80=np.nan, coverage_p50=np.nan,
                sharpness_p80=np.nan, sharpness_p50=np.nan,
                calibration_score=np.nan, mean_bias=np.nan,
                skill_score=np.nan, n_forecasts=len(forecasts), n_validated=0
            )
        
        # Coverage: % within bounds
        in_p80 = (merged["actual_value"] >= merged["p10"]) & (merged["actual_value"] <= merged["p90"])
        in_p50 = (merged["actual_value"] >= merged["p25"]) & (merged["actual_value"] <= merged["p75"])
        coverage_p80 = in_p80.mean()
        coverage_p50 = in_p50.mean()
        
        # Sharpness: interval width
        sharpness_p80 = (merged["p90"] - merged["p10"]).mean()
        sharpness_p50 = (merged["p75"] - merged["p25"]).mean()
        
        # Calibration: 80% interval should contain ~80% of points
        # Score = 1 - |actual_coverage - expected_coverage|
        calibration_score = 1 - abs(coverage_p80 - 0.8)
        
        # Bias: median prediction vs actual
        mean_bias = (merged["p50"] - merged["actual_value"]).mean()
        
        # Skill score vs naive (persistence) baseline
        naive_error = (merged["actual_value"].shift(1) - merged["actual_value"]).dropna().abs().mean()
        forecast_error = (merged["p50"] - merged["actual_value"]).abs().mean()
        skill_score = 1 - (forecast_error / (naive_error + 1e-10)) if naive_error > 0 else 0
        
        return ForecastDiagnostics(
            coverage_p80=coverage_p80,
            coverage_p50=coverage_p50,
            sharpness_p80=sharpness_p80,
            sharpness_p50=sharpness_p50,
            calibration_score=calibration_score,
            mean_bias=mean_bias,
            skill_score=skill_score,
            n_forecasts=len(forecasts),
            n_validated=len(merged)
        )
```

**SQL Schema**:
```sql
CREATE TABLE ACM_ForecastDiagnostics (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    RunID INT NOT NULL,
    EquipID INT NOT NULL,
    ForecastType NVARCHAR(50) NOT NULL,  -- 'health', 'sensor', 'rul'
    
    -- Coverage metrics
    CoverageP80 FLOAT NULL,
    CoverageP50 FLOAT NULL,
    
    -- Sharpness metrics  
    SharpnessP80 FLOAT NULL,
    SharpnessP50 FLOAT NULL,
    
    -- Calibration
    CalibrationScore FLOAT NULL,
    MeanBias FLOAT NULL,
    SkillScore FLOAT NULL,
    
    -- Counts
    NForecastsMade INT NOT NULL,
    NForecastsValidated INT NOT NULL,
    
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_ForecastDiagnostics_Runs FOREIGN KEY (RunID) REFERENCES ACM_Runs(RunID)
);
```

| Task | File | Status |
|------|------|--------|
| [x] Create `ForecastDiagnostics` class | `core/forecast_diagnostics.py` | ✅ |
| [x] Implement coverage metric | `core/forecast_diagnostics.py` | ✅ |
| [x] Implement sharpness metric | `core/forecast_diagnostics.py` | ✅ |
| [x] Implement calibration metric | `core/forecast_diagnostics.py` | ✅ |
| [ ] Create `ACM_ForecastDiagnostics` table | `scripts/sql/migrations/` | ⏳ |
| [ ] Persist diagnostics on every run | `core/output_manager.py` | ⏳ |

### P4.6 — Unified Confidence Model (Item 49)

**Implementation**:
```python
# core/confidence_model.py (NEW FILE)

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class ConfidenceSignals:
    """Individual confidence signals."""
    regime_confidence: float      # 0-1, how stable is current regime
    detector_agreement: float     # 0-1, how much detectors agree
    data_quality: float           # 0-1, sensor validity and completeness
    model_maturity: float         # 0-1, how trained are the models
    drift_indicator: float        # 0-1, inverse of drift severity
    data_recency: float           # 0-1, how recent is the data

@dataclass
class CombinedConfidence:
    """Combined confidence with breakdown."""
    confidence: float             # 0-1, overall confidence
    signals: ConfidenceSignals    # Individual components
    limiting_factor: str          # Which signal is pulling down confidence
    is_trustworthy: bool          # confidence > threshold

class ConfidenceModel:
    """
    Unified confidence model for all ACM outputs.
    
    Combines multiple signals into a single confidence score:
    1. Regime confidence (regime stability, not UNKNOWN)
    2. Detector agreement (fusion quality)
    3. Data quality (sensor validity, no gaps)
    4. Model maturity (training data quantity)
    5. Drift indicator (no recent drift)
    6. Data recency (fresh data)
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "regime_confidence": 0.2,
            "detector_agreement": 0.25,
            "data_quality": 0.2,
            "model_maturity": 0.15,
            "drift_indicator": 0.1,
            "data_recency": 0.1
        }
        self.trustworthy_threshold = 0.6
    
    def compute(self, signals: ConfidenceSignals) -> CombinedConfidence:
        """Compute combined confidence from signals."""
        
        # Weighted average
        weighted_sum = (
            signals.regime_confidence * self.weights["regime_confidence"] +
            signals.detector_agreement * self.weights["detector_agreement"] +
            signals.data_quality * self.weights["data_quality"] +
            signals.model_maturity * self.weights["model_maturity"] +
            signals.drift_indicator * self.weights["drift_indicator"] +
            signals.data_recency * self.weights["data_recency"]
        )
        
        # Find limiting factor (lowest signal)
        signal_dict = {
            "regime_confidence": signals.regime_confidence,
            "detector_agreement": signals.detector_agreement,
            "data_quality": signals.data_quality,
            "model_maturity": signals.model_maturity,
            "drift_indicator": signals.drift_indicator,
            "data_recency": signals.data_recency
        }
        limiting_factor = min(signal_dict, key=signal_dict.get)
        
        # Apply minimum gate: if any signal is very low, cap confidence
        min_signal = min(signal_dict.values())
        if min_signal < 0.3:
            weighted_sum = min(weighted_sum, 0.5)  # Cap at 50% if any signal < 30%
        
        confidence = np.clip(weighted_sum, 0, 1)
        
        return CombinedConfidence(
            confidence=confidence,
            signals=signals,
            limiting_factor=limiting_factor,
            is_trustworthy=confidence >= self.trustworthy_threshold
        )
    
    def from_run_context(self,
                         regime_label: int,
                         regime_stability: float,
                         detector_outputs: Dict,
                         sensor_validity: Dict[str, bool],
                         model_age_hours: float,
                         drift_detected: bool,
                         data_age_minutes: float) -> CombinedConfidence:
        """Build confidence from pipeline context."""
        
        signals = ConfidenceSignals(
            regime_confidence=0.0 if regime_label == -1 else regime_stability,
            detector_agreement=self._compute_detector_agreement(detector_outputs),
            data_quality=sum(sensor_validity.values()) / len(sensor_validity) if sensor_validity else 0,
            model_maturity=min(1.0, model_age_hours / 720),  # 30 days = 1.0
            drift_indicator=0.3 if drift_detected else 1.0,
            data_recency=max(0, 1 - data_age_minutes / 60)  # Decay over 1 hour
        )
        
        return self.compute(signals)
```

| Task | File | Status |
|------|------|--------|
| [x] Create `ConfidenceModel` class | `core/confidence_model.py` | ✅ |
| [x] Combine regime confidence | `core/confidence_model.py` | ✅ |
| [x] Combine detector agreement | `core/confidence_model.py` | ✅ |
| [x] Combine data quality signal | `core/confidence_model.py` | ✅ |
| [ ] Apply to health outputs | `core/health_tracker.py` | ⏳ |
| [ ] Apply to episode outputs | `core/episode_manager.py` | ⏳ |
| [ ] Apply to RUL outputs | `core/rul_estimator.py` | ⏳ |

---

## Phase 5: Operational Infrastructure

**Goal**: Drift/novelty control plane, feedback loops, operational contracts

**Current State Analysis**:
- **Drift Detection** (`drift.py`): Uses KL divergence and PSI, but drift signals are logged only - not actionable triggers
- **Novelty**: No explicit novelty pressure tracking separate from regimes
- **Operator Feedback**: No mechanism to capture operator confirmation/dismissal of alerts
- **Decision Policy**: Analytics and operational behavior are coupled in same code paths

### P5.1 — Drift/Novelty Control Plane (Item 10)

**Implementation**:
```python
# core/drift_controller.py (NEW FILE)

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, List
import pandas as pd

class DriftAction(Enum):
    """Actions triggered by drift detection."""
    NONE = "NONE"                    # No action needed
    LOG_WARNING = "LOG_WARNING"      # Log and continue
    REDUCE_CONFIDENCE = "REDUCE_CONFIDENCE"  # Dampen output confidence
    TRIGGER_REPLAY = "TRIGGER_REPLAY"  # Queue offline replay
    HALT_PREDICTIONS = "HALT_PREDICTIONS"  # Stop making predictions

@dataclass
class DriftThresholds:
    """Thresholds for drift severity levels."""
    psi_warning: float = 0.1      # PSI > 0.1 = warning
    psi_critical: float = 0.25    # PSI > 0.25 = critical
    kl_warning: float = 0.1       # KL divergence warning
    kl_critical: float = 0.5      # KL divergence critical
    novelty_warning: float = 0.3  # % unknown regime > 30%
    novelty_critical: float = 0.5 # % unknown regime > 50%

@dataclass
class DriftSignal:
    """A detected drift or novelty signal."""
    timestamp: pd.Timestamp
    signal_type: str  # "PSI", "KL", "NOVELTY", "COVARIATE"
    severity: str     # "WARNING", "CRITICAL"
    value: float
    affected_sensors: List[str]
    recommended_action: DriftAction

class DriftController:
    """
    Central controller for drift and novelty signals.
    
    Promotes drift from passive logging to active control plane triggers.
    """
    
    def __init__(self, thresholds: DriftThresholds = None,
                 on_replay_trigger: Optional[Callable] = None):
        self.thresholds = thresholds or DriftThresholds()
        self.on_replay_trigger = on_replay_trigger
        self.signal_history: List[DriftSignal] = []
        self.replay_queued = False
    
    def evaluate(self, 
                 psi_values: pd.DataFrame,
                 kl_values: pd.DataFrame,
                 unknown_regime_pct: float,
                 timestamp: pd.Timestamp) -> List[DriftSignal]:
        """
        Evaluate all drift/novelty signals and determine actions.
        """
        signals = []
        
        # Evaluate PSI per sensor
        for col in psi_values.columns:
            psi = psi_values[col].iloc[-1] if not psi_values.empty else 0
            if psi > self.thresholds.psi_critical:
                signals.append(DriftSignal(
                    timestamp=timestamp,
                    signal_type="PSI",
                    severity="CRITICAL",
                    value=psi,
                    affected_sensors=[col],
                    recommended_action=DriftAction.TRIGGER_REPLAY
                ))
            elif psi > self.thresholds.psi_warning:
                signals.append(DriftSignal(
                    timestamp=timestamp,
                    signal_type="PSI",
                    severity="WARNING",
                    value=psi,
                    affected_sensors=[col],
                    recommended_action=DriftAction.LOG_WARNING
                ))
        
        # Evaluate novelty pressure
        if unknown_regime_pct > self.thresholds.novelty_critical:
            signals.append(DriftSignal(
                timestamp=timestamp,
                signal_type="NOVELTY",
                severity="CRITICAL",
                value=unknown_regime_pct,
                affected_sensors=[],
                recommended_action=DriftAction.TRIGGER_REPLAY
            ))
        elif unknown_regime_pct > self.thresholds.novelty_warning:
            signals.append(DriftSignal(
                timestamp=timestamp,
                signal_type="NOVELTY",
                severity="WARNING",
                value=unknown_regime_pct,
                affected_sensors=[],
                recommended_action=DriftAction.REDUCE_CONFIDENCE
            ))
        
        # Store and execute actions
        self.signal_history.extend(signals)
        self._execute_actions(signals)
        
        return signals
    
    def _execute_actions(self, signals: List[DriftSignal]) -> None:
        """Execute recommended actions from signals."""
        for signal in signals:
            if signal.recommended_action == DriftAction.TRIGGER_REPLAY:
                if not self.replay_queued and self.on_replay_trigger:
                    self.on_replay_trigger(signal)
                    self.replay_queued = True
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `DriftController` class | `core/drift_controller.py` | ⏳ |
| [ ] Promote drift signals to control-plane triggers | `core/drift_controller.py` | ⏳ |
| [ ] Promote novelty signals to control-plane triggers | `core/drift_controller.py` | ⏳ |
| [ ] Trigger offline replay when thresholds exceeded | `core/drift_controller.py` | ⏳ |

### P5.2 — Novelty Pressure Tracking (Item 30)

**Implementation**:
```python
# Add to core/drift_controller.py

@dataclass
class NoveltyPressure:
    """Track novelty pressure independent of regime labels."""
    timestamp: pd.Timestamp
    unknown_pct: float           # % of recent rows in UNKNOWN regime
    emerging_pct: float          # % in EMERGING (not yet promoted)
    out_of_distribution: float   # Average distance from known regime centroids
    trend: str                   # "INCREASING", "STABLE", "DECREASING"

class NoveltyTracker:
    """
    Track novelty pressure as a first-class metric.
    
    Unlike regime labels (which are discrete), novelty pressure
    is a continuous measure of how "unusual" recent data is.
    """
    
    def __init__(self, window_hours: float = 24.0):
        self.window_hours = window_hours
        self.history: List[NoveltyPressure] = []
    
    def compute(self,
                regime_labels: pd.Series,
                regime_distances: pd.DataFrame,
                timestamps: pd.Series) -> NoveltyPressure:
        """
        Compute current novelty pressure.
        """
        # Filter to window
        window_start = timestamps.max() - pd.Timedelta(hours=self.window_hours)
        mask = timestamps >= window_start
        
        labels = regime_labels[mask]
        
        # % in UNKNOWN (-1) or EMERGING (-2)
        unknown_pct = (labels == -1).mean()
        emerging_pct = (labels == -2).mean()
        
        # Average distance from nearest known regime
        if regime_distances is not None and not regime_distances.empty:
            min_distances = regime_distances[mask].min(axis=1)
            out_of_distribution = min_distances.mean()
        else:
            out_of_distribution = 0.0
        
        # Compute trend from history
        trend = self._compute_trend()
        
        pressure = NoveltyPressure(
            timestamp=timestamps.max(),
            unknown_pct=unknown_pct,
            emerging_pct=emerging_pct,
            out_of_distribution=out_of_distribution,
            trend=trend
        )
        
        self.history.append(pressure)
        return pressure
    
    def _compute_trend(self) -> str:
        """Determine if novelty pressure is increasing/decreasing."""
        if len(self.history) < 3:
            return "STABLE"
        
        recent = [h.unknown_pct + h.emerging_pct for h in self.history[-5:]]
        if len(recent) < 2:
            return "STABLE"
        
        slope = (recent[-1] - recent[0]) / len(recent)
        if slope > 0.05:
            return "INCREASING"
        elif slope < -0.05:
            return "DECREASING"
        return "STABLE"
```

**SQL Schema**:
```sql
CREATE TABLE ACM_NoveltyPressure (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    RunID INT NOT NULL,
    EquipID INT NOT NULL,
    Timestamp DATETIME2 NOT NULL,
    UnknownPct FLOAT NOT NULL,
    EmergingPct FLOAT NOT NULL,
    OutOfDistribution FLOAT NOT NULL,
    Trend NVARCHAR(20) NOT NULL,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_NoveltyPressure_Runs FOREIGN KEY (RunID) REFERENCES ACM_Runs(RunID)
);
CREATE INDEX IX_NoveltyPressure_Equip ON ACM_NoveltyPressure(EquipID, Timestamp);
```

| Task | File | Status |
|------|------|--------|
| [ ] Track novelty pressure independent of regimes | `core/drift_controller.py` | ⏳ |
| [ ] Create `ACM_NoveltyPressure` table | `scripts/sql/migrations/` | ⏳ |
| [ ] Add novelty pressure to run metadata | `core/run_metadata_writer.py` | ⏳ |

### P5.3 — Drift Events as Objects (Item 31)

**Implementation**:
```python
# Add to core/drift_controller.py

from dataclasses import asdict
import json

@dataclass
class DriftEvent:
    """Persisted drift event with full context."""
    id: str
    equip_id: int
    detected_at: pd.Timestamp
    event_type: str  # "COVARIATE", "CONCEPT", "NOVELTY"
    severity: str    # "WARNING", "CRITICAL"
    
    # Evidence
    psi_value: Optional[float]
    kl_value: Optional[float]
    affected_sensors: List[str]
    
    # Impact
    confidence_reduction: float  # How much to reduce output confidence
    models_invalidated: List[str]  # Which models need retraining
    
    # Resolution
    resolved_at: Optional[pd.Timestamp] = None
    resolution: Optional[str] = None  # "RETRAINED", "FALSE_ALARM", "ACKNOWLEDGED"
    
    def to_sql_row(self) -> dict:
        """Convert to dict for SQL insert."""
        return {
            "DriftEventID": self.id,
            "EquipID": self.equip_id,
            "DetectedAt": self.detected_at,
            "EventType": self.event_type,
            "Severity": self.severity,
            "PSIValue": self.psi_value,
            "KLValue": self.kl_value,
            "AffectedSensors": ",".join(self.affected_sensors),
            "ConfidenceReduction": self.confidence_reduction,
            "ModelsInvalidated": ",".join(self.models_invalidated),
            "ResolvedAt": self.resolved_at,
            "Resolution": self.resolution
        }

class DriftEventManager:
    """Manage drift events as first-class objects."""
    
    def __init__(self):
        self.active_events: Dict[str, DriftEvent] = {}
    
    def create_event(self, signal: DriftSignal, equip_id: int) -> DriftEvent:
        """Create drift event from signal."""
        event = DriftEvent(
            id=f"DRIFT_{equip_id}_{signal.timestamp.strftime('%Y%m%d%H%M%S')}",
            equip_id=equip_id,
            detected_at=signal.timestamp,
            event_type=signal.signal_type,
            severity=signal.severity,
            psi_value=signal.value if signal.signal_type == "PSI" else None,
            kl_value=signal.value if signal.signal_type == "KL" else None,
            affected_sensors=signal.affected_sensors,
            confidence_reduction=0.3 if signal.severity == "CRITICAL" else 0.1,
            models_invalidated=[]
        )
        self.active_events[event.id] = event
        return event
    
    def resolve_event(self, event_id: str, resolution: str) -> None:
        """Mark drift event as resolved."""
        if event_id in self.active_events:
            event = self.active_events[event_id]
            event.resolved_at = pd.Timestamp.now()
            event.resolution = resolution
    
    def get_confidence_penalty(self, equip_id: int) -> float:
        """Get total confidence penalty from active drift events."""
        penalty = 0.0
        for event in self.active_events.values():
            if event.equip_id == equip_id and event.resolved_at is None:
                penalty += event.confidence_reduction
        return min(0.5, penalty)  # Cap at 50% reduction
```

**SQL Schema**:
```sql
CREATE TABLE ACM_DriftEvents (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    DriftEventID NVARCHAR(100) NOT NULL UNIQUE,
    EquipID INT NOT NULL,
    DetectedAt DATETIME2 NOT NULL,
    EventType NVARCHAR(20) NOT NULL,
    Severity NVARCHAR(20) NOT NULL,
    PSIValue FLOAT NULL,
    KLValue FLOAT NULL,
    AffectedSensors NVARCHAR(MAX) NULL,
    ConfidenceReduction FLOAT NOT NULL,
    ModelsInvalidated NVARCHAR(MAX) NULL,
    ResolvedAt DATETIME2 NULL,
    Resolution NVARCHAR(50) NULL,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_DriftEvents_Equipment FOREIGN KEY (EquipID) REFERENCES Equipment(EquipID)
);
CREATE INDEX IX_DriftEvents_Active ON ACM_DriftEvents(EquipID, ResolvedAt) WHERE ResolvedAt IS NULL;
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `DriftEvent` class | `core/drift_controller.py` | ⏳ |
| [ ] Persist drift events to SQL | `core/output_manager.py` | ⏳ |
| [ ] Down-weight confidence when drift detected | `core/confidence_model.py` | ⏳ |
| [ ] Create `ACM_DriftEvents` table | `scripts/sql/migrations/` | ⏳ |

### P5.4 — Unified Sensor Attribution (Item 32)

**Implementation**:
```python
# core/sensor_attribution.py (REFACTORED)

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

@dataclass
class SensorContribution:
    """Attribution of a single sensor to an anomaly."""
    sensor_name: str
    contribution_pct: float      # 0-100, % contribution to anomaly
    z_score: float               # Individual sensor z-score
    direction: str               # "HIGH", "LOW", "VOLATILE"
    baseline_deviation: float    # How far from baseline

@dataclass
class AttributionResult:
    """Complete attribution for an anomaly or episode."""
    timestamp: pd.Timestamp
    total_z_score: float
    contributions: List[SensorContribution]
    top_3_sensors: List[str]
    explanation: str  # Human-readable explanation

class UnifiedAttribution:
    """
    Unified sensor attribution using frozen baseline artifacts.
    
    Key principle: All attribution uses the SAME normalization as detection.
    No separate statistics computed during attribution.
    """
    
    def __init__(self, baseline_normalizer: "BaselineNormalizer"):
        self.normalizer = baseline_normalizer
    
    def attribute(self,
                  raw_data: pd.DataFrame,
                  fused_z: float,
                  detector_outputs: Dict[str, pd.DataFrame],
                  sensor_cols: List[str]) -> AttributionResult:
        """
        Compute sensor contributions to an anomaly.
        
        Uses frozen baseline statistics from normalizer.
        """
        contributions = []
        
        for col in sensor_cols:
            if col not in self.normalizer.baseline.sensor_stats:
                continue
            
            stats = self.normalizer.baseline.sensor_stats[col]
            value = raw_data[col].iloc[-1] if col in raw_data.columns else np.nan
            
            if pd.isna(value):
                continue
            
            # Compute z-score using BASELINE statistics
            z = (value - stats.mean) / (stats.std + 1e-10)
            
            # Determine direction
            if z > 2:
                direction = "HIGH"
            elif z < -2:
                direction = "LOW"
            else:
                direction = "NORMAL"
            
            contributions.append(SensorContribution(
                sensor_name=col,
                contribution_pct=0,  # Filled below
                z_score=z,
                direction=direction,
                baseline_deviation=abs(z)
            ))
        
        # Compute contribution percentages
        total_deviation = sum(c.baseline_deviation for c in contributions)
        if total_deviation > 0:
            for c in contributions:
                c.contribution_pct = (c.baseline_deviation / total_deviation) * 100
        
        # Sort by contribution
        contributions.sort(key=lambda c: c.contribution_pct, reverse=True)
        top_3 = [c.sensor_name for c in contributions[:3]]
        
        # Generate explanation
        explanation = self._generate_explanation(contributions[:3], fused_z)
        
        return AttributionResult(
            timestamp=raw_data["Timestamp"].iloc[-1],
            total_z_score=fused_z,
            contributions=contributions,
            top_3_sensors=top_3,
            explanation=explanation
        )
    
    def _generate_explanation(self, top_contributors: List[SensorContribution], 
                               fused_z: float) -> str:
        """Generate human-readable explanation."""
        if not top_contributors:
            return "No significant sensor deviations detected."
        
        parts = []
        for c in top_contributors:
            parts.append(f"{c.sensor_name} is {c.direction} ({c.z_score:.1f}σ)")
        
        severity = "Critical" if fused_z > 5 else "Elevated" if fused_z > 3 else "Minor"
        return f"{severity} anomaly driven by: {', '.join(parts)}"
```

| Task | File | Status |
|------|------|--------|
| [ ] Refactor sensor attribution to use frozen normalized artifacts | `core/sensor_attribution.py` | ⏳ |
| [ ] Unify attribution across all modules | `core/sensor_attribution.py` | ⏳ |
| [ ] Add attribution to episode explanation | `core/episode_manager.py` | ⏳ |

### P5.5 — Baseline Window Policy (Item 27)

**Implementation**:
```python
# core/baseline_policy.py (NEW FILE)

from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum
import pandas as pd

class BaselineQuality(Enum):
    """Quality assessment of baseline data."""
    EXCELLENT = "EXCELLENT"  # Meets all requirements
    ADEQUATE = "ADEQUATE"    # Meets minimum requirements
    MARGINAL = "MARGINAL"    # Below minimum, proceed with caution
    INSUFFICIENT = "INSUFFICIENT"  # Cannot proceed

@dataclass
class BaselineRequirements:
    """Per-equipment baseline requirements."""
    min_rows: int = 500
    min_hours: float = 168.0        # 7 days
    max_gap_hours: float = 24.0
    min_sensor_coverage: float = 0.9  # 90% non-null
    require_regime_diversity: bool = True
    min_regimes: int = 1

@dataclass
class BaselineAssessment:
    """Assessment of baseline data quality."""
    quality: BaselineQuality
    actual_rows: int
    actual_hours: float
    max_gap_hours: float
    sensor_coverage: float
    regime_count: int
    violations: list
    can_proceed: bool

class BaselinePolicy:
    """
    Define and enforce per-equipment baseline window requirements.
    """
    
    # Default requirements by equipment type
    DEFAULT_REQUIREMENTS = {
        "*": BaselineRequirements(),  # Global default
        "GAS_TURBINE": BaselineRequirements(min_rows=1000, min_hours=336),  # 14 days
        "FD_FAN": BaselineRequirements(min_rows=500, min_hours=168),
    }
    
    def __init__(self, equipment_requirements: Optional[Dict[str, BaselineRequirements]] = None):
        self.requirements = equipment_requirements or self.DEFAULT_REQUIREMENTS
    
    def get_requirements(self, equipment_type: str) -> BaselineRequirements:
        """Get requirements for equipment type."""
        return self.requirements.get(equipment_type, self.requirements.get("*", BaselineRequirements()))
    
    def assess(self, 
               data: pd.DataFrame,
               equipment_type: str,
               regime_labels: Optional[pd.Series] = None) -> BaselineAssessment:
        """Assess baseline data against requirements."""
        reqs = self.get_requirements(equipment_type)
        violations = []
        
        # Check row count
        actual_rows = len(data)
        if actual_rows < reqs.min_rows:
            violations.append(f"Only {actual_rows} rows (need {reqs.min_rows})")
        
        # Check time span
        if "Timestamp" in data.columns:
            time_span = (data["Timestamp"].max() - data["Timestamp"].min())
            actual_hours = time_span.total_seconds() / 3600
        else:
            actual_hours = 0
        if actual_hours < reqs.min_hours:
            violations.append(f"Only {actual_hours:.1f}h of data (need {reqs.min_hours}h)")
        
        # Check gaps
        if "Timestamp" in data.columns:
            gaps = data["Timestamp"].diff().dt.total_seconds() / 3600
            max_gap = gaps.max() if not gaps.empty else 0
        else:
            max_gap = 0
        if max_gap > reqs.max_gap_hours:
            violations.append(f"Gap of {max_gap:.1f}h exceeds {reqs.max_gap_hours}h")
        
        # Check sensor coverage
        sensor_cols = [c for c in data.columns if c not in ["Timestamp", "EquipID", "EntryDateTime"]]
        if sensor_cols:
            coverage = data[sensor_cols].notna().mean().mean()
        else:
            coverage = 0
        if coverage < reqs.min_sensor_coverage:
            violations.append(f"Sensor coverage {coverage:.1%} below {reqs.min_sensor_coverage:.1%}")
        
        # Check regime diversity
        regime_count = len(regime_labels.unique()) if regime_labels is not None else 0
        if reqs.require_regime_diversity and regime_count < reqs.min_regimes:
            violations.append(f"Only {regime_count} regimes (need {reqs.min_regimes})")
        
        # Determine quality
        if not violations:
            quality = BaselineQuality.EXCELLENT
        elif len(violations) == 1 and actual_rows >= reqs.min_rows * 0.7:
            quality = BaselineQuality.ADEQUATE
        elif actual_rows >= reqs.min_rows * 0.5:
            quality = BaselineQuality.MARGINAL
        else:
            quality = BaselineQuality.INSUFFICIENT
        
        return BaselineAssessment(
            quality=quality,
            actual_rows=actual_rows,
            actual_hours=actual_hours,
            max_gap_hours=max_gap,
            sensor_coverage=coverage,
            regime_count=regime_count,
            violations=violations,
            can_proceed=quality != BaselineQuality.INSUFFICIENT
        )
```

**SQL Schema**:
```sql
CREATE TABLE ACM_BaselinePolicy (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    RunID INT NOT NULL,
    EquipID INT NOT NULL,
    EquipmentType NVARCHAR(50) NOT NULL,
    
    -- Assessment results
    Quality NVARCHAR(20) NOT NULL,
    ActualRows INT NOT NULL,
    ActualHours FLOAT NOT NULL,
    MaxGapHours FLOAT NOT NULL,
    SensorCoverage FLOAT NOT NULL,
    RegimeCount INT NOT NULL,
    Violations NVARCHAR(MAX) NULL,
    CanProceed BIT NOT NULL,
    
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_BaselinePolicy_Runs FOREIGN KEY (RunID) REFERENCES ACM_Runs(RunID)
);
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `BaselinePolicy` class | `core/baseline_policy.py` | ⏳ |
| [ ] Define per-equipment baseline window requirements | `core/baseline_policy.py` | ⏳ |
| [ ] Persist policy per run | `core/output_manager.py` | ⏳ |
| [ ] Create `ACM_BaselinePolicy` table | `scripts/sql/migrations/` | ⏳ |

### P5.6 — Regression Harness (Item 37)

**Implementation**:
```python
# tests/regression_harness.py (NEW FILE)

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import json

@dataclass
class RegressionResult:
    """Result of comparing current vs golden output."""
    test_name: str
    passed: bool
    metric_name: str
    golden_value: float
    current_value: float
    tolerance: float
    deviation_pct: float

@dataclass
class GoldenDataset:
    """Reference dataset for regression testing."""
    name: str
    equipment: str
    input_file: Path
    expected_outputs: Dict[str, float]
    tolerance: Dict[str, float]

class RegressionHarness:
    """
    Compare ACM behavior against golden reference datasets.
    
    Detects unintended behavioral changes during refactoring.
    """
    
    def __init__(self, golden_dir: Path):
        self.golden_dir = golden_dir
        self.datasets: Dict[str, GoldenDataset] = {}
        self._load_golden_datasets()
    
    def _load_golden_datasets(self) -> None:
        """Load all golden datasets from directory."""
        manifest_path = self.golden_dir / "manifest.json"
        if not manifest_path.exists():
            return
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        for ds in manifest.get("datasets", []):
            self.datasets[ds["name"]] = GoldenDataset(
                name=ds["name"],
                equipment=ds["equipment"],
                input_file=self.golden_dir / ds["input_file"],
                expected_outputs=ds["expected_outputs"],
                tolerance=ds.get("tolerance", {})
            )
    
    def run_regression(self, 
                       pipeline_fn,
                       dataset_name: Optional[str] = None) -> List[RegressionResult]:
        """
        Run regression tests against golden datasets.
        
        Args:
            pipeline_fn: Function that takes input DataFrame and returns output dict
            dataset_name: Specific dataset to test, or None for all
        """
        results = []
        
        datasets_to_test = (
            [self.datasets[dataset_name]] if dataset_name 
            else list(self.datasets.values())
        )
        
        for ds in datasets_to_test:
            # Load input
            input_df = pd.read_csv(ds.input_file)
            
            # Run pipeline
            outputs = pipeline_fn(input_df, ds.equipment)
            
            # Compare each expected output
            for metric_name, expected_value in ds.expected_outputs.items():
                actual_value = outputs.get(metric_name, np.nan)
                tolerance = ds.tolerance.get(metric_name, 0.05)  # Default 5%
                
                if expected_value == 0:
                    deviation = abs(actual_value)
                else:
                    deviation = abs(actual_value - expected_value) / abs(expected_value)
                
                passed = deviation <= tolerance
                
                results.append(RegressionResult(
                    test_name=f"{ds.name}/{metric_name}",
                    passed=passed,
                    metric_name=metric_name,
                    golden_value=expected_value,
                    current_value=actual_value,
                    tolerance=tolerance,
                    deviation_pct=deviation * 100
                ))
        
        return results
    
    def create_golden_dataset(self,
                              name: str,
                              equipment: str,
                              input_df: pd.DataFrame,
                              outputs: Dict[str, float],
                              tolerance: Optional[Dict[str, float]] = None) -> None:
        """Create a new golden dataset from current run."""
        # Save input
        input_path = self.golden_dir / f"{name}_input.csv"
        input_df.to_csv(input_path, index=False)
        
        # Update manifest
        manifest_path = self.golden_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {"datasets": []}
        
        manifest["datasets"].append({
            "name": name,
            "equipment": equipment,
            "input_file": f"{name}_input.csv",
            "expected_outputs": outputs,
            "tolerance": tolerance or {}
        })
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
```

**Golden Dataset Structure**:
```
tests/golden_data/
├── manifest.json
├── fd_fan_baseline_input.csv
├── fd_fan_degradation_input.csv
├── gas_turbine_normal_input.csv
└── gas_turbine_failure_input.csv
```

**Example manifest.json**:
```json
{
  "datasets": [
    {
      "name": "fd_fan_baseline",
      "equipment": "FD_FAN",
      "input_file": "fd_fan_baseline_input.csv",
      "expected_outputs": {
        "mean_health_pct": 95.2,
        "episode_count": 0,
        "fused_z_mean": 1.2
      },
      "tolerance": {
        "mean_health_pct": 0.05,
        "episode_count": 0,
        "fused_z_mean": 0.1
      }
    }
  ]
}
```

| Task | File | Status |
|------|------|--------|
| [ ] Create regression harness | `tests/regression_harness.py` | ⏳ |
| [ ] Define golden datasets | `tests/golden_data/` | ⏳ |
| [ ] Compare before/after behavior | `tests/regression_harness.py` | ⏳ |
| [ ] Detect unintended behavioral changes | `tests/regression_harness.py` | ⏳ |

### P5.7 — Truth Dashboard (Item 38)

**Implementation**:
The Truth Dashboard exposes internal ACM invariants and health metrics for operators to understand system reliability.

**Dashboard Panels**:
```json
{
  "title": "ACM System Truth",
  "panels": [
    {
      "title": "Data Quality Score",
      "type": "gauge",
      "query": "SELECT TOP 1 SensorCoverage * 100 as value FROM ACM_BaselinePolicy WHERE EquipID = $equipment ORDER BY CreatedAt DESC"
    },
    {
      "title": "Active Drift Events",
      "type": "stat",
      "query": "SELECT COUNT(*) as value FROM ACM_DriftEvents WHERE EquipID = $equipment AND ResolvedAt IS NULL"
    },
    {
      "title": "Novelty Pressure Trend",
      "type": "timeseries",
      "query": "SELECT Timestamp as time, UnknownPct * 100 as 'Unknown %', EmergingPct * 100 as 'Emerging %' FROM ACM_NoveltyPressure WHERE EquipID = $equipment AND Timestamp BETWEEN $__timeFrom() AND $__timeTo() ORDER BY time ASC"
    },
    {
      "title": "Detector Agreement",
      "type": "timeseries", 
      "query": "SELECT CreatedAt as time, AgreementMean as value FROM ACM_FusionQuality WHERE EquipID = $equipment AND CreatedAt BETWEEN $__timeFrom() AND $__timeTo() ORDER BY time ASC"
    },
    {
      "title": "Regime Stability",
      "type": "gauge",
      "query": "SELECT TOP 1 CASE WHEN MaturityState = 'CONVERGED' THEN 100 WHEN MaturityState = 'LEARNING' THEN 60 ELSE 30 END as value FROM ACM_ActiveModels WHERE EquipID = $equipment ORDER BY ActivatedAt DESC"
    },
    {
      "title": "RUL Reliability Status",
      "type": "stat",
      "query": "SELECT TOP 1 RULStatus as value FROM ACM_RUL WHERE EquipID = $equipment ORDER BY CreatedAt DESC"
    },
    {
      "title": "Confidence Breakdown",
      "type": "barchart",
      "description": "Shows limiting factors for output confidence"
    }
  ]
}
```

**Key Invariants to Expose**:
1. **Data Quality**: Sensor coverage, gap detection, stuck sensors
2. **Drift Status**: Active drift events, PSI trends, confidence penalties
3. **Novelty Pressure**: % unknown regime, trend direction
4. **Fusion Health**: Detector agreement, contributing detectors
5. **Model State**: Maturity level, last training time, version

| Task | File | Status |
|------|------|--------|
| [ ] Create `acm_truth.json` dashboard | `grafana_dashboards/acm_truth.json` | ⏳ |
| [ ] Expose data quality invariants | `grafana_dashboards/acm_truth.json` | ⏳ |
| [ ] Expose drift status | `grafana_dashboards/acm_truth.json` | ⏳ |
| [ ] Expose novelty pressure | `grafana_dashboards/acm_truth.json` | ⏳ |
| [ ] Expose fusion health | `grafana_dashboards/acm_truth.json` | ⏳ |

### P5.8 — Operational Decision Contract (Item 39)

**Implementation**:
```python
# core/decision_policy.py (NEW FILE)

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd

class RecommendedAction(Enum):
    """Discrete actions for operators."""
    NO_ACTION = "NO_ACTION"
    MONITOR = "MONITOR"
    INVESTIGATE = "INVESTIGATE"
    SCHEDULE_MAINTENANCE = "SCHEDULE_MAINTENANCE"
    IMMEDIATE_ACTION = "IMMEDIATE_ACTION"

@dataclass
class DecisionContract:
    """
    Compact operational output contract.
    
    This is the ONLY output operators should act on.
    Decouples analytics from operational decisions.
    """
    timestamp: pd.Timestamp
    equip_id: int
    
    # State summary (from HealthState)
    state: str           # HEALTHY, DEGRADED, CRITICAL, UNKNOWN
    state_confidence: float  # 0-1
    
    # RUL summary (from RULStatus)
    rul_status: str      # RELIABLE, NOT_RELIABLE, INSUFFICIENT_DATA
    rul_hours: Optional[float]
    rul_confidence: float
    
    # Episode summary
    active_episodes: int
    worst_episode_severity: Optional[str]
    
    # Action recommendation
    recommended_action: RecommendedAction
    action_reason: str
    action_urgency_hours: Optional[float]
    
    # System health
    system_confidence: float  # Overall confidence in this output
    limiting_factor: str      # What's reducing confidence

@dataclass  
class DecisionPolicy:
    """
    Maps analytics outputs to operational decisions.
    
    Can be modified WITHOUT retraining models.
    """
    # State → Action mapping
    state_actions = {
        "HEALTHY": RecommendedAction.NO_ACTION,
        "DEGRADED": RecommendedAction.MONITOR,
        "CRITICAL": RecommendedAction.INVESTIGATE,
        "UNKNOWN": RecommendedAction.MONITOR,
    }
    
    # RUL thresholds for escalation
    rul_immediate_hours: float = 24.0
    rul_schedule_hours: float = 168.0  # 7 days
    
    # Confidence thresholds
    min_confidence_for_action: float = 0.5
    
    def evaluate(self,
                 health_state: str,
                 health_confidence: float,
                 rul_status: str,
                 rul_hours: Optional[float],
                 rul_confidence: float,
                 active_episodes: int,
                 worst_severity: Optional[str],
                 system_confidence: float,
                 limiting_factor: str) -> DecisionContract:
        """
        Apply policy to determine recommended action.
        """
        # Start with state-based action
        action = self.state_actions.get(health_state, RecommendedAction.MONITOR)
        reason = f"State is {health_state}"
        urgency = None
        
        # Escalate based on RUL if reliable
        if rul_status == "RELIABLE" and rul_hours is not None:
            if rul_hours < self.rul_immediate_hours:
                action = RecommendedAction.IMMEDIATE_ACTION
                reason = f"RUL {rul_hours:.0f}h < 24h threshold"
                urgency = rul_hours
            elif rul_hours < self.rul_schedule_hours:
                action = max(action, RecommendedAction.SCHEDULE_MAINTENANCE, key=lambda x: x.value)
                reason = f"RUL {rul_hours:.0f}h < 7d threshold"
                urgency = rul_hours
        
        # Escalate based on episodes
        if worst_severity == "CRITICAL":
            action = max(action, RecommendedAction.INVESTIGATE, key=lambda x: x.value)
            reason = f"Critical episode active"
        
        # Dampen if low confidence
        if system_confidence < self.min_confidence_for_action:
            if action in [RecommendedAction.IMMEDIATE_ACTION, RecommendedAction.SCHEDULE_MAINTENANCE]:
                action = RecommendedAction.INVESTIGATE
                reason += f" (confidence {system_confidence:.0%} too low for action)"
        
        return DecisionContract(
            timestamp=pd.Timestamp.now(),
            equip_id=0,  # Set by caller
            state=health_state,
            state_confidence=health_confidence,
            rul_status=rul_status,
            rul_hours=rul_hours,
            rul_confidence=rul_confidence,
            active_episodes=active_episodes,
            worst_episode_severity=worst_severity,
            recommended_action=action,
            action_reason=reason,
            action_urgency_hours=urgency,
            system_confidence=system_confidence,
            limiting_factor=limiting_factor
        )
```

**SQL Schema**:
```sql
CREATE TABLE ACM_DecisionOutput (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    RunID INT NOT NULL,
    EquipID INT NOT NULL,
    Timestamp DATETIME2 NOT NULL,
    
    -- State
    HealthState NVARCHAR(20) NOT NULL,
    StateConfidence FLOAT NOT NULL,
    
    -- RUL
    RULStatus NVARCHAR(20) NOT NULL,
    RULHours FLOAT NULL,
    RULConfidence FLOAT NOT NULL,
    
    -- Episodes
    ActiveEpisodes INT NOT NULL,
    WorstEpisodeSeverity NVARCHAR(20) NULL,
    
    -- Decision
    RecommendedAction NVARCHAR(30) NOT NULL,
    ActionReason NVARCHAR(500) NOT NULL,
    ActionUrgencyHours FLOAT NULL,
    
    -- Confidence
    SystemConfidence FLOAT NOT NULL,
    LimitingFactor NVARCHAR(50) NOT NULL,
    
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_DecisionOutput_Runs FOREIGN KEY (RunID) REFERENCES ACM_Runs(RunID)
);
CREATE INDEX IX_DecisionOutput_Equip ON ACM_DecisionOutput(EquipID, Timestamp DESC);
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `DecisionContract` dataclass | `core/decision_policy.py` | ⏳ |
| [ ] Define State, Confidence, Action, RULStatus fields | `core/decision_policy.py` | ⏳ |
| [ ] Create compact output format | `core/decision_policy.py` | ⏳ |
| [ ] Create `ACM_DecisionOutput` table | `scripts/sql/migrations/` | ⏳ |

### P5.9 — Seasonality Handling (Item 40)

**Implementation**:
```python
# core/seasonality.py (NEW FILE)

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class SeasonalPattern:
    """Detected seasonal pattern."""
    period_type: str  # "HOURLY", "DAILY", "WEEKLY"
    period_hours: float
    amplitude: float  # Size of seasonal variation
    phase_shift: float  # Hours offset from midnight/Monday
    confidence: float  # How confident in this pattern

@dataclass
class SeasonalAdjustment:
    """Adjustment to apply for seasonality."""
    timestamp: pd.Timestamp
    sensor: str
    expected_offset: float  # Expected deviation from baseline
    adjusted_value: float   # Value after removing seasonality

class SeasonalityHandler:
    """
    Detect and adjust for seasonal patterns in sensor data.
    
    Patterns detected:
    - Diurnal (24-hour cycle) - temperature, load
    - Weekly (168-hour cycle) - operational patterns
    """
    
    def __init__(self, min_periods: int = 3):
        self.min_periods = min_periods  # Minimum cycles to detect pattern
        self.patterns: Dict[str, List[SeasonalPattern]] = {}
    
    def detect_patterns(self, 
                        data: pd.DataFrame,
                        sensor_cols: List[str],
                        timestamp_col: str = "Timestamp") -> Dict[str, List[SeasonalPattern]]:
        """
        Detect seasonal patterns in sensor data.
        """
        patterns = {}
        
        for col in sensor_cols:
            col_patterns = []
            series = data[[timestamp_col, col]].dropna()
            
            if len(series) < 168:  # Need at least 1 week
                continue
            
            # Check for diurnal pattern (24h)
            diurnal = self._detect_periodic_pattern(series, col, 24)
            if diurnal:
                col_patterns.append(diurnal)
            
            # Check for weekly pattern (168h)
            if len(series) >= 168 * 3:  # Need 3 weeks
                weekly = self._detect_periodic_pattern(series, col, 168)
                if weekly:
                    col_patterns.append(weekly)
            
            if col_patterns:
                patterns[col] = col_patterns
        
        self.patterns = patterns
        return patterns
    
    def _detect_periodic_pattern(self,
                                  data: pd.DataFrame,
                                  col: str,
                                  period_hours: float) -> Optional[SeasonalPattern]:
        """Detect pattern with specific period using FFT."""
        series = data[col].values
        timestamps = pd.to_datetime(data["Timestamp"])
        
        # Resample to hourly for consistent analysis
        hourly = data.set_index("Timestamp").resample("1H").mean()
        if len(hourly) < period_hours * self.min_periods:
            return None
        
        values = hourly[col].interpolate().values
        
        # FFT to find dominant frequencies
        fft = np.fft.fft(values)
        freqs = np.fft.fftfreq(len(values))
        
        # Find peak near expected frequency
        expected_freq = 1 / period_hours
        freq_mask = np.abs(np.abs(freqs) - expected_freq) < 0.01
        
        if not freq_mask.any():
            return None
        
        peak_power = np.abs(fft[freq_mask]).max()
        total_power = np.abs(fft).sum()
        
        # Confidence based on how much variance is explained
        confidence = peak_power / (total_power + 1e-10)
        
        if confidence < 0.1:  # Less than 10% of variance
            return None
        
        # Compute amplitude
        amplitude = np.std(values)
        
        return SeasonalPattern(
            period_type="DAILY" if period_hours == 24 else "WEEKLY",
            period_hours=period_hours,
            amplitude=amplitude,
            phase_shift=0,  # Simplified - could compute from FFT phase
            confidence=confidence
        )
    
    def adjust_baseline(self,
                        data: pd.DataFrame,
                        sensor_cols: List[str]) -> pd.DataFrame:
        """
        Adjust baseline statistics for seasonality.
        
        Returns data with seasonal component removed.
        """
        result = data.copy()
        
        for col in sensor_cols:
            if col not in self.patterns:
                continue
            
            for pattern in self.patterns[col]:
                # Compute expected seasonal component
                hours = (data["Timestamp"] - data["Timestamp"].min()).dt.total_seconds() / 3600
                phase = (hours % pattern.period_hours) / pattern.period_hours * 2 * np.pi
                
                # Simple sinusoidal model
                seasonal = pattern.amplitude * np.sin(phase + pattern.phase_shift)
                
                # Remove seasonal component
                result[col] = result[col] - seasonal
        
        return result
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `SeasonalityHandler` class | `core/seasonality.py` | ⏳ |
| [ ] Detect diurnal patterns | `core/seasonality.py` | ⏳ |
| [ ] Detect day-of-week patterns | `core/seasonality.py` | ⏳ |
| [ ] Adjust baselines for seasonality | `core/seasonality.py` | ⏳ |

### P5.10 — Asset Similarity Priors (Item 41)

**Implementation**:
```python
# core/asset_similarity.py (NEW FILE)

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class AssetProfile:
    """Profile of an asset for similarity computation."""
    equip_id: int
    equip_type: str
    sensor_names: List[str]
    sensor_means: Dict[str, float]
    sensor_stds: Dict[str, float]
    regime_count: int
    typical_health: float
    data_hours: float

@dataclass
class SimilarityScore:
    """Similarity between two assets."""
    source_equip_id: int
    target_equip_id: int
    overall_similarity: float  # 0-1
    sensor_similarity: float   # Based on sensor overlap and statistics
    behavior_similarity: float # Based on regime patterns
    transferable: bool         # Can we transfer models?
    transfer_confidence: float

class AssetSimilarity:
    """
    Compute asset similarity for cold-start transfer learning.
    
    Enables bootstrapping new assets from similar existing assets
    with full auditability.
    """
    
    def __init__(self, min_similarity: float = 0.7):
        self.min_similarity = min_similarity
        self.profiles: Dict[int, AssetProfile] = {}
    
    def build_profile(self, 
                      equip_id: int,
                      equip_type: str,
                      data: pd.DataFrame,
                      regime_labels: pd.Series) -> AssetProfile:
        """Build profile from historical data."""
        sensor_cols = [c for c in data.columns 
                       if c not in ["Timestamp", "EquipID", "EntryDateTime"]]
        
        profile = AssetProfile(
            equip_id=equip_id,
            equip_type=equip_type,
            sensor_names=sensor_cols,
            sensor_means={c: data[c].mean() for c in sensor_cols},
            sensor_stds={c: data[c].std() for c in sensor_cols},
            regime_count=len(regime_labels.unique()) if regime_labels is not None else 0,
            typical_health=85.0,  # Would come from ACM_HealthTimeline
            data_hours=(data["Timestamp"].max() - data["Timestamp"].min()).total_seconds() / 3600
        )
        
        self.profiles[equip_id] = profile
        return profile
    
    def find_similar(self, 
                     target_profile: AssetProfile,
                     candidates: Optional[List[int]] = None) -> List[SimilarityScore]:
        """
        Find assets similar to target.
        
        Matching criteria:
        1. Same equipment type
        2. Overlapping sensors
        3. Similar sensor statistics
        4. Similar regime structure
        """
        results = []
        
        candidate_ids = candidates or list(self.profiles.keys())
        
        for source_id in candidate_ids:
            if source_id == target_profile.equip_id:
                continue
            
            source = self.profiles.get(source_id)
            if source is None:
                continue
            
            # Must be same type
            if source.equip_type != target_profile.equip_type:
                continue
            
            # Compute sensor similarity
            sensor_sim = self._sensor_similarity(source, target_profile)
            
            # Compute behavior similarity
            behavior_sim = self._behavior_similarity(source, target_profile)
            
            # Overall weighted
            overall = 0.6 * sensor_sim + 0.4 * behavior_sim
            
            results.append(SimilarityScore(
                source_equip_id=source_id,
                target_equip_id=target_profile.equip_id,
                overall_similarity=overall,
                sensor_similarity=sensor_sim,
                behavior_similarity=behavior_sim,
                transferable=overall >= self.min_similarity,
                transfer_confidence=overall if overall >= self.min_similarity else 0
            ))
        
        return sorted(results, key=lambda x: x.overall_similarity, reverse=True)
    
    def _sensor_similarity(self, 
                           source: AssetProfile, 
                           target: AssetProfile) -> float:
        """Compute sensor overlap and statistical similarity."""
        # Sensor name overlap
        common = set(source.sensor_names) & set(target.sensor_names)
        if not common:
            return 0.0
        
        overlap = len(common) / max(len(source.sensor_names), len(target.sensor_names))
        
        # Statistical similarity for common sensors
        stat_diffs = []
        for sensor in common:
            mean_diff = abs(source.sensor_means.get(sensor, 0) - target.sensor_means.get(sensor, 0))
            std_source = source.sensor_stds.get(sensor, 1)
            std_target = target.sensor_stds.get(sensor, 1)
            
            # Normalized difference
            normalized_diff = mean_diff / (max(std_source, std_target) + 1e-10)
            stat_diffs.append(normalized_diff)
        
        # Convert to similarity (exponential decay)
        mean_diff = np.mean(stat_diffs) if stat_diffs else 0
        stat_similarity = np.exp(-mean_diff)
        
        return 0.5 * overlap + 0.5 * stat_similarity
    
    def _behavior_similarity(self, 
                             source: AssetProfile, 
                             target: AssetProfile) -> float:
        """Compute behavioral similarity based on regimes and health."""
        # Regime count similarity
        regime_diff = abs(source.regime_count - target.regime_count)
        regime_sim = 1 / (1 + regime_diff)
        
        # Health similarity
        health_diff = abs(source.typical_health - target.typical_health)
        health_sim = 1 - (health_diff / 100)
        
        return 0.5 * regime_sim + 0.5 * health_sim
    
    def transfer_baseline(self,
                          source_id: int,
                          target_id: int,
                          source_baseline: "BaselineNormalizer") -> Tuple["BaselineNormalizer", float]:
        """
        Transfer baseline from source to target asset.
        
        Returns transferred baseline and confidence.
        """
        source_profile = self.profiles.get(source_id)
        target_profile = self.profiles.get(target_id)
        
        if not source_profile or not target_profile:
            raise ValueError("Profiles not found")
        
        # Find similarity score
        scores = self.find_similar(target_profile, [source_id])
        if not scores or not scores[0].transferable:
            raise ValueError(f"Assets not similar enough for transfer: {scores[0].overall_similarity if scores else 0}")
        
        confidence = scores[0].transfer_confidence
        
        # Clone baseline with adjusted statistics
        # (In practice, would scale statistics based on target's observed ranges)
        return source_baseline, confidence
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `AssetSimilarity` class | `core/asset_similarity.py` | ⏳ |
| [ ] Define similarity metrics | `core/asset_similarity.py` | ⏳ |
| [ ] Implement transfer learning for cold start | `core/asset_similarity.py` | ⏳ |
| [ ] Add full auditability | `core/asset_similarity.py` | ⏳ |

### P5.11 — Operator Feedback (Item 42)

**Implementation**:
```python
# core/operator_feedback.py (NEW FILE)

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import pandas as pd

class FeedbackType(Enum):
    """Types of operator feedback."""
    FALSE_ALARM = "FALSE_ALARM"       # Alert was wrong
    TRUE_ALARM = "TRUE_ALARM"         # Alert was correct
    MISSED_ALARM = "MISSED_ALARM"     # Should have alerted but didn't
    MAINTENANCE = "MAINTENANCE"       # Maintenance was performed
    CALIBRATION = "CALIBRATION"       # Sensor was recalibrated
    OPERATIONAL = "OPERATIONAL"       # Operational change (not fault)

class FeedbackResolution(Enum):
    """How feedback should affect the system."""
    ADJUST_THRESHOLD = "ADJUST_THRESHOLD"  # Tune thresholds
    RETRAIN_MODEL = "RETRAIN_MODEL"        # Trigger retraining
    IGNORE = "IGNORE"                       # No action needed
    BASELINE_RESET = "BASELINE_RESET"      # Reset baseline window

@dataclass
class OperatorFeedbackRecord:
    """Record of operator feedback."""
    id: str
    equip_id: int
    timestamp: pd.Timestamp
    feedback_type: FeedbackType
    
    # Context
    related_episode_id: Optional[str]
    related_alert_id: Optional[str]
    
    # Details
    operator_notes: str
    affected_sensors: List[str]
    
    # Resolution
    resolution: Optional[FeedbackResolution]
    resolution_applied_at: Optional[pd.Timestamp]

class OperatorFeedback:
    """
    Capture and process operator feedback for continuous improvement.
    """
    
    def __init__(self):
        self.feedback_history: List[OperatorFeedbackRecord] = []
        self.false_alarm_rate: float = 0.0
        self.detection_rate: float = 1.0
    
    def record_feedback(self,
                        equip_id: int,
                        feedback_type: FeedbackType,
                        episode_id: Optional[str] = None,
                        notes: str = "",
                        sensors: Optional[List[str]] = None) -> OperatorFeedbackRecord:
        """Record operator feedback."""
        record = OperatorFeedbackRecord(
            id=f"FB_{equip_id}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
            equip_id=equip_id,
            timestamp=pd.Timestamp.now(),
            feedback_type=feedback_type,
            related_episode_id=episode_id,
            related_alert_id=None,
            operator_notes=notes,
            affected_sensors=sensors or [],
            resolution=None,
            resolution_applied_at=None
        )
        
        self.feedback_history.append(record)
        self._update_rates()
        
        return record
    
    def _update_rates(self) -> None:
        """Update false alarm and detection rates from history."""
        recent = [f for f in self.feedback_history 
                  if (pd.Timestamp.now() - f.timestamp).days < 30]
        
        if not recent:
            return
        
        false_alarms = sum(1 for f in recent if f.feedback_type == FeedbackType.FALSE_ALARM)
        true_alarms = sum(1 for f in recent if f.feedback_type == FeedbackType.TRUE_ALARM)
        missed = sum(1 for f in recent if f.feedback_type == FeedbackType.MISSED_ALARM)
        
        total_alarms = false_alarms + true_alarms
        if total_alarms > 0:
            self.false_alarm_rate = false_alarms / total_alarms
        
        total_events = true_alarms + missed
        if total_events > 0:
            self.detection_rate = true_alarms / total_events
    
    def suggest_resolution(self, record: OperatorFeedbackRecord) -> FeedbackResolution:
        """Suggest resolution based on feedback type and history."""
        if record.feedback_type == FeedbackType.FALSE_ALARM:
            # Check if repeated false alarms on same sensors
            similar = [f for f in self.feedback_history
                       if f.feedback_type == FeedbackType.FALSE_ALARM
                       and set(f.affected_sensors) & set(record.affected_sensors)]
            
            if len(similar) >= 3:
                return FeedbackResolution.ADJUST_THRESHOLD
            return FeedbackResolution.IGNORE
        
        elif record.feedback_type == FeedbackType.MISSED_ALARM:
            return FeedbackResolution.RETRAIN_MODEL
        
        elif record.feedback_type in [FeedbackType.MAINTENANCE, FeedbackType.CALIBRATION]:
            return FeedbackResolution.BASELINE_RESET
        
        return FeedbackResolution.IGNORE
    
    def get_threshold_adjustment(self, equip_id: int) -> float:
        """Compute threshold adjustment based on feedback."""
        # If too many false alarms, raise threshold
        if self.false_alarm_rate > 0.3:
            return 0.5  # Raise by 0.5 sigma
        
        # If too many missed alarms, lower threshold  
        if self.detection_rate < 0.8:
            return -0.5  # Lower by 0.5 sigma
        
        return 0.0
```

**SQL Schema**:
```sql
CREATE TABLE ACM_OperatorFeedback (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    FeedbackID NVARCHAR(100) NOT NULL UNIQUE,
    EquipID INT NOT NULL,
    Timestamp DATETIME2 NOT NULL,
    FeedbackType NVARCHAR(30) NOT NULL,
    
    -- Context
    RelatedEpisodeID NVARCHAR(100) NULL,
    RelatedAlertID NVARCHAR(100) NULL,
    
    -- Details
    OperatorNotes NVARCHAR(MAX) NULL,
    AffectedSensors NVARCHAR(MAX) NULL,
    
    -- Resolution
    Resolution NVARCHAR(30) NULL,
    ResolutionAppliedAt DATETIME2 NULL,
    
    -- Metadata
    OperatorID NVARCHAR(50) NULL,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_OperatorFeedback_Equipment FOREIGN KEY (EquipID) REFERENCES Equipment(EquipID)
);
CREATE INDEX IX_OperatorFeedback_Equip ON ACM_OperatorFeedback(EquipID, Timestamp DESC);
CREATE INDEX IX_OperatorFeedback_Type ON ACM_OperatorFeedback(FeedbackType, Timestamp DESC);
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `OperatorFeedback` class | `core/operator_feedback.py` | ⏳ |
| [ ] Capture false alarm feedback | `core/operator_feedback.py` | ⏳ |
| [ ] Capture valid alarm feedback | `core/operator_feedback.py` | ⏳ |
| [ ] Capture maintenance feedback | `core/operator_feedback.py` | ⏳ |
| [ ] Create `ACM_OperatorFeedback` table | `scripts/sql/migrations/` | ⏳ |

### P5.12 — Alert Fatigue Controls (Item 43)

**Implementation**:
```python
# core/alert_fatigue.py (NEW FILE)

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import pandas as pd
from collections import defaultdict

class AlertTier(Enum):
    """Alert escalation tiers."""
    TIER_1 = 1  # Operator console only
    TIER_2 = 2  # Email notification
    TIER_3 = 3  # SMS/Phone alert
    TIER_4 = 4  # Management escalation

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_alerts_per_hour: int = 5
    max_alerts_per_day: int = 20
    cooldown_minutes: int = 30
    escalation_delay_hours: float = 2.0

@dataclass
class SuppressedAlert:
    """Record of a suppressed alert."""
    alert_id: str
    timestamp: pd.Timestamp
    reason: str
    original_tier: AlertTier
    would_have_been: str  # Description of suppressed alert

class AlertFatigueController:
    """
    Control alert volume to prevent operator fatigue.
    
    Features:
    - Rate limiting (per hour/day)
    - Escalation ladders (gradual escalation)
    - Suppression logging (audit trail)
    - Similar alert deduplication
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.alert_history: Dict[int, List[pd.Timestamp]] = defaultdict(list)
        self.suppressed: List[SuppressedAlert] = []
        self.current_tiers: Dict[int, AlertTier] = {}  # equip_id -> current tier
        self.last_escalation: Dict[int, pd.Timestamp] = {}
    
    def should_alert(self, 
                     equip_id: int, 
                     severity: str,
                     description: str) -> tuple[bool, Optional[AlertTier], Optional[str]]:
        """
        Determine if alert should be sent.
        
        Returns: (should_send, tier, suppression_reason)
        """
        now = pd.Timestamp.now()
        
        # Clean old history
        self._clean_history(equip_id, now)
        
        # Check hourly rate
        hour_ago = now - pd.Timedelta(hours=1)
        hourly_count = sum(1 for t in self.alert_history[equip_id] if t > hour_ago)
        
        if hourly_count >= self.config.max_alerts_per_hour:
            self._record_suppression(equip_id, "HOURLY_LIMIT", description)
            return False, None, f"Hourly limit ({self.config.max_alerts_per_hour}) exceeded"
        
        # Check daily rate
        day_ago = now - pd.Timedelta(days=1)
        daily_count = sum(1 for t in self.alert_history[equip_id] if t > day_ago)
        
        if daily_count >= self.config.max_alerts_per_day:
            self._record_suppression(equip_id, "DAILY_LIMIT", description)
            return False, None, f"Daily limit ({self.config.max_alerts_per_day}) exceeded"
        
        # Check cooldown from last alert
        if self.alert_history[equip_id]:
            last = max(self.alert_history[equip_id])
            minutes_since = (now - last).total_seconds() / 60
            if minutes_since < self.config.cooldown_minutes:
                self._record_suppression(equip_id, "COOLDOWN", description)
                return False, None, f"Cooldown ({self.config.cooldown_minutes}m) not elapsed"
        
        # Determine tier based on escalation ladder
        tier = self._get_tier(equip_id, severity, now)
        
        # Record alert
        self.alert_history[equip_id].append(now)
        
        return True, tier, None
    
    def _get_tier(self, equip_id: int, severity: str, now: pd.Timestamp) -> AlertTier:
        """Determine alert tier based on escalation ladder."""
        current = self.current_tiers.get(equip_id, AlertTier.TIER_1)
        last_esc = self.last_escalation.get(equip_id)
        
        # CRITICAL always gets highest tier
        if severity == "CRITICAL":
            self.current_tiers[equip_id] = AlertTier.TIER_3
            return AlertTier.TIER_3
        
        # Check if should escalate
        if last_esc:
            hours_since = (now - last_esc).total_seconds() / 3600
            if hours_since > self.config.escalation_delay_hours:
                # Escalate to next tier
                if current.value < AlertTier.TIER_4.value:
                    new_tier = AlertTier(current.value + 1)
                    self.current_tiers[equip_id] = new_tier
                    self.last_escalation[equip_id] = now
                    return new_tier
        else:
            self.last_escalation[equip_id] = now
        
        return current
    
    def reset_escalation(self, equip_id: int) -> None:
        """Reset escalation ladder (e.g., when issue resolved)."""
        self.current_tiers[equip_id] = AlertTier.TIER_1
        self.last_escalation.pop(equip_id, None)
    
    def _record_suppression(self, equip_id: int, reason: str, description: str) -> None:
        """Record suppressed alert for audit."""
        self.suppressed.append(SuppressedAlert(
            alert_id=f"SUP_{equip_id}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=pd.Timestamp.now(),
            reason=reason,
            original_tier=self.current_tiers.get(equip_id, AlertTier.TIER_1),
            would_have_been=description
        ))
    
    def _clean_history(self, equip_id: int, now: pd.Timestamp) -> None:
        """Remove alerts older than 1 day from history."""
        cutoff = now - pd.Timedelta(days=1)
        self.alert_history[equip_id] = [
            t for t in self.alert_history[equip_id] if t > cutoff
        ]
    
    def get_suppression_report(self, equip_id: int, hours: int = 24) -> Dict:
        """Get suppression statistics for reporting."""
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        recent = [s for s in self.suppressed 
                  if s.timestamp > cutoff]
        
        by_reason = defaultdict(int)
        for s in recent:
            by_reason[s.reason] += 1
        
        return {
            "total_suppressed": len(recent),
            "by_reason": dict(by_reason),
            "current_tier": self.current_tiers.get(equip_id, AlertTier.TIER_1).name
        }
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `AlertFatigueController` class | `core/alert_fatigue.py` | ⏳ |
| [ ] Implement rate limits | `core/alert_fatigue.py` | ⏳ |
| [ ] Implement escalation ladders | `core/alert_fatigue.py` | ⏳ |
| [ ] Implement suppression logging | `core/alert_fatigue.py` | ⏳ |

### P5.13 — Episode Clustering (Item 44)

**Implementation**:
```python
# core/episode_clustering.py (NEW FILE)

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

@dataclass
class EpisodeSignature:
    """Feature vector describing an episode."""
    episode_id: str
    equip_id: int
    
    # Timing features
    duration_hours: float
    time_of_day: float  # 0-24
    day_of_week: int    # 0-6
    
    # Severity features
    peak_z: float
    mean_z: float
    
    # Sensor features
    top_sensors: List[str]
    sensor_pattern_hash: str  # Hash of sensor involvement
    
    # Regime context
    regime_at_onset: int

@dataclass
class EpisodeFamily:
    """A cluster of similar episodes."""
    family_id: str
    name: str  # Auto-generated or operator-assigned
    member_count: int
    member_ids: List[str]
    
    # Centroid characteristics
    typical_duration_hours: float
    typical_peak_z: float
    typical_sensors: List[str]
    typical_regime: int
    
    # Pattern
    recurrence_rate: float  # Episodes per month
    last_occurrence: pd.Timestamp

class EpisodeClusterer:
    """
    Cluster episodes into recurring families for pattern mining.
    
    Enables:
    - Pattern recognition across historical episodes
    - Proto-RCA (what sensors/conditions cause this family)
    - Prediction (this family tends to escalate)
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 3):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.families: Dict[str, EpisodeFamily] = {}
    
    def extract_signature(self, episode: "Episode") -> EpisodeSignature:
        """Extract feature signature from episode."""
        return EpisodeSignature(
            episode_id=episode.id,
            equip_id=episode.equip_id,
            duration_hours=episode.duration_hours,
            time_of_day=episode.start_time.hour + episode.start_time.minute / 60,
            day_of_week=episode.start_time.dayofweek,
            peak_z=episode.peak_z_score,
            mean_z=episode.mean_z_score,
            top_sensors=episode.affected_sensors[:3],
            sensor_pattern_hash=self._hash_sensors(episode.affected_sensors),
            regime_at_onset=episode.regime_at_onset
        )
    
    def cluster_episodes(self, 
                         episodes: List["Episode"],
                         equip_id: int) -> List[EpisodeFamily]:
        """
        Cluster episodes into families using DBSCAN.
        """
        if len(episodes) < self.min_samples:
            return []
        
        # Extract signatures
        signatures = [self.extract_signature(e) for e in episodes]
        
        # Build feature matrix
        X = self._build_feature_matrix(signatures)
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X)
        
        # Cluster
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        # Build families
        families = []
        for label in set(labels):
            if label == -1:  # Noise
                continue
            
            member_indices = [i for i, l in enumerate(labels) if l == label]
            member_episodes = [episodes[i] for i in member_indices]
            member_sigs = [signatures[i] for i in member_indices]
            
            family = self._build_family(
                label, member_episodes, member_sigs, equip_id
            )
            families.append(family)
            self.families[family.family_id] = family
        
        return families
    
    def _build_feature_matrix(self, signatures: List[EpisodeSignature]) -> np.ndarray:
        """Build numeric feature matrix from signatures."""
        features = []
        for sig in signatures:
            features.append([
                sig.duration_hours,
                sig.time_of_day,
                sig.day_of_week,
                sig.peak_z,
                sig.mean_z,
                hash(sig.sensor_pattern_hash) % 1000 / 1000,  # Normalize hash
                sig.regime_at_onset if sig.regime_at_onset >= 0 else 0
            ])
        return np.array(features)
    
    def _build_family(self,
                      label: int,
                      episodes: List["Episode"],
                      signatures: List[EpisodeSignature],
                      equip_id: int) -> EpisodeFamily:
        """Build family from cluster members."""
        # Compute centroid characteristics
        durations = [s.duration_hours for s in signatures]
        peaks = [s.peak_z for s in signatures]
        
        # Find most common sensors
        sensor_counts: Dict[str, int] = {}
        for sig in signatures:
            for sensor in sig.top_sensors:
                sensor_counts[sensor] = sensor_counts.get(sensor, 0) + 1
        top_sensors = sorted(sensor_counts, key=sensor_counts.get, reverse=True)[:3]
        
        # Most common regime
        regimes = [s.regime_at_onset for s in signatures if s.regime_at_onset >= 0]
        typical_regime = max(set(regimes), key=regimes.count) if regimes else -1
        
        # Recurrence rate
        timestamps = sorted([e.start_time for e in episodes])
        if len(timestamps) > 1:
            span_days = (timestamps[-1] - timestamps[0]).days
            recurrence_rate = len(episodes) / max(span_days / 30, 1)
        else:
            recurrence_rate = 0
        
        return EpisodeFamily(
            family_id=f"FAM_{equip_id}_{label}",
            name=f"Pattern {label} ({top_sensors[0] if top_sensors else 'unknown'})",
            member_count=len(episodes),
            member_ids=[e.id for e in episodes],
            typical_duration_hours=np.mean(durations),
            typical_peak_z=np.mean(peaks),
            typical_sensors=top_sensors,
            typical_regime=typical_regime,
            recurrence_rate=recurrence_rate,
            last_occurrence=max(e.start_time for e in episodes)
        )
    
    def _hash_sensors(self, sensors: List[str]) -> str:
        """Create deterministic hash of sensor list."""
        return ",".join(sorted(sensors))
    
    def classify_new_episode(self, episode: "Episode") -> Optional[str]:
        """Classify new episode into existing family."""
        sig = self.extract_signature(episode)
        X = self._build_feature_matrix([sig])
        X_scaled = self.scaler.transform(X)
        
        # Find nearest family centroid
        best_family = None
        best_distance = float('inf')
        
        for family_id, family in self.families.items():
            # Simple distance to typical values
            distance = abs(sig.peak_z - family.typical_peak_z)
            distance += abs(sig.duration_hours - family.typical_duration_hours)
            
            if distance < best_distance and distance < self.eps * 2:
                best_distance = distance
                best_family = family_id
        
        return best_family
```

**SQL Schema**:
```sql
CREATE TABLE ACM_EpisodeFamilies (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    FamilyID NVARCHAR(100) NOT NULL,
    EquipID INT NOT NULL,
    Name NVARCHAR(200) NOT NULL,
    MemberCount INT NOT NULL,
    
    -- Centroid characteristics
    TypicalDurationHours FLOAT NOT NULL,
    TypicalPeakZ FLOAT NOT NULL,
    TypicalSensors NVARCHAR(500) NULL,
    TypicalRegime INT NULL,
    
    -- Pattern
    RecurrenceRate FLOAT NOT NULL,
    LastOccurrence DATETIME2 NOT NULL,
    
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    UpdatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_EpisodeFamilies_Equipment FOREIGN KEY (EquipID) REFERENCES Equipment(EquipID)
);

CREATE TABLE ACM_EpisodeFamilyMembers (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    FamilyID NVARCHAR(100) NOT NULL,
    EpisodeID NVARCHAR(100) NOT NULL,
    AddedAt DATETIME2 DEFAULT GETDATE()
);
CREATE INDEX IX_EpisodeFamilyMembers_Family ON ACM_EpisodeFamilyMembers(FamilyID);
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `EpisodeClusterer` class | `core/episode_clustering.py` | ⏳ |
| [ ] Cluster episodes into recurring families | `core/episode_clustering.py` | ⏳ |
| [ ] Enable pattern mining | `core/episode_clustering.py` | ⏳ |
| [ ] Enable proto-RCA | `core/episode_clustering.py` | ⏳ |
| [ ] Create `ACM_EpisodeFamilies` table | `scripts/sql/migrations/` | ⏳ |

### P5.14 — Failure Mode Unknown Semantics (Item 45)

**Implementation**:
```python
# Add to core/episode_manager.py

from enum import Enum

class FailureMode(Enum):
    """Failure mode classification."""
    KNOWN = "KNOWN"         # Matches a known failure pattern
    UNKNOWN = "UNKNOWN"     # Does not match any known pattern
    EMERGING = "EMERGING"   # New pattern being learned

@dataclass
class FailureModeAssessment:
    """Assessment of failure mode for an episode."""
    mode: FailureMode
    confidence: float
    matched_family_id: Optional[str]  # If KNOWN, which family
    explanation: str
    
    # For EMERGING
    similarity_to_known: float  # 0-1, how similar to known patterns
    novelty_score: float        # 0-1, how novel this pattern is

class FailureModeClassifier:
    """
    Classify episodes into known/unknown/emerging failure modes.
    
    CRITICAL: Prevents implying specific fault labels without evidence.
    """
    
    def __init__(self, 
                 clusterer: "EpisodeClusterer",
                 known_threshold: float = 0.8,
                 emerging_threshold: float = 0.5):
        self.clusterer = clusterer
        self.known_threshold = known_threshold
        self.emerging_threshold = emerging_threshold
    
    def classify(self, episode: "Episode") -> FailureModeAssessment:
        """
        Classify episode's failure mode.
        
        Returns UNKNOWN if episode doesn't match known patterns,
        rather than guessing a fault type.
        """
        # Try to match to existing family
        family_id = self.clusterer.classify_new_episode(episode)
        
        if family_id:
            family = self.clusterer.families.get(family_id)
            if family:
                # Compute match confidence
                confidence = self._compute_match_confidence(episode, family)
                
                if confidence >= self.known_threshold:
                    return FailureModeAssessment(
                        mode=FailureMode.KNOWN,
                        confidence=confidence,
                        matched_family_id=family_id,
                        explanation=f"Matches pattern '{family.name}' with {confidence:.0%} confidence",
                        similarity_to_known=confidence,
                        novelty_score=1 - confidence
                    )
                elif confidence >= self.emerging_threshold:
                    return FailureModeAssessment(
                        mode=FailureMode.EMERGING,
                        confidence=confidence,
                        matched_family_id=family_id,
                        explanation=f"Partially matches '{family.name}' - emerging pattern",
                        similarity_to_known=confidence,
                        novelty_score=1 - confidence
                    )
        
        # No match - failure mode unknown
        return FailureModeAssessment(
            mode=FailureMode.UNKNOWN,
            confidence=0.9,  # High confidence it's unknown
            matched_family_id=None,
            explanation="Does not match any known failure patterns",
            similarity_to_known=0.0,
            novelty_score=1.0
        )
    
    def _compute_match_confidence(self, 
                                   episode: "Episode", 
                                   family: "EpisodeFamily") -> float:
        """Compute confidence that episode matches family."""
        scores = []
        
        # Duration similarity
        duration_diff = abs(episode.duration_hours - family.typical_duration_hours)
        duration_sim = 1 / (1 + duration_diff)
        scores.append(duration_sim)
        
        # Peak z similarity
        peak_diff = abs(episode.peak_z_score - family.typical_peak_z)
        peak_sim = 1 / (1 + peak_diff)
        scores.append(peak_sim)
        
        # Sensor overlap
        episode_sensors = set(episode.affected_sensors)
        family_sensors = set(family.typical_sensors)
        if episode_sensors and family_sensors:
            overlap = len(episode_sensors & family_sensors) / len(episode_sensors | family_sensors)
            scores.append(overlap)
        
        # Regime match
        if episode.regime_at_onset == family.typical_regime:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
```

| Task | File | Status |
|------|------|--------|
| [ ] Add explicit "failure mode unknown" outcome | `core/episode_manager.py` | ⏳ |
| [ ] Prevent implied fault labels | `core/episode_manager.py` | ⏳ |
| [ ] Add `FailureMode` enum (KNOWN, UNKNOWN, EMERGING) | `core/episode_manager.py` | ⏳ |

### P5.15 — Configuration/Version Management (Item 46)

**Implementation**:
```python
# core/experiment_manager.py (NEW FILE)

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import pandas as pd
import json
import hashlib

@dataclass
class ExperimentConfig:
    """Versioned experiment configuration."""
    config_version: str           # Semantic version
    config_hash: str              # SHA256 of config contents
    created_at: pd.Timestamp
    
    # Detector configs
    ar1_enabled: bool = True
    pca_enabled: bool = True
    iforest_enabled: bool = True
    gmm_enabled: bool = True
    omr_enabled: bool = True
    
    # Threshold configs
    anomaly_threshold: float = 3.0
    critical_threshold: float = 5.5
    
    # Regime configs
    max_regimes: int = 6
    regime_min_samples: int = 100
    
    # Fusion configs
    fusion_method: str = "weighted_average"
    detector_weights: Dict[str, float] = field(default_factory=dict)
    
    # RUL configs
    rul_method: str = "monte_carlo"
    rul_simulations: int = 1000

@dataclass
class ExperimentRun:
    """Record of an experiment run."""
    run_id: str
    config_version: str
    config_hash: str
    equip_id: int
    
    started_at: pd.Timestamp
    completed_at: Optional[pd.Timestamp]
    
    # Outcomes
    health_mean: Optional[float]
    episode_count: Optional[int]
    rul_hours: Optional[float]
    
    # Comparison to baseline
    baseline_config_hash: Optional[str]
    health_delta: Optional[float]

class ExperimentManager:
    """
    Version and track analytics configurations.
    
    Enables:
    - A/B testing of configurations
    - Rollback to previous configs
    - Audit trail of config changes
    """
    
    def __init__(self):
        self.configs: Dict[str, ExperimentConfig] = {}
        self.runs: List[ExperimentRun] = []
        self.active_config: Optional[ExperimentConfig] = None
    
    def create_config(self, 
                      version: str,
                      **kwargs) -> ExperimentConfig:
        """Create new experiment configuration."""
        config = ExperimentConfig(
            config_version=version,
            config_hash="",  # Computed below
            created_at=pd.Timestamp.now(),
            **kwargs
        )
        
        # Compute hash of config contents
        config_dict = asdict(config)
        config_dict.pop("config_hash")
        config_dict.pop("created_at")
        config_json = json.dumps(config_dict, sort_keys=True)
        config.config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]
        
        self.configs[config.config_hash] = config
        return config
    
    def activate_config(self, config_hash: str) -> None:
        """Activate a configuration for use."""
        if config_hash not in self.configs:
            raise ValueError(f"Config {config_hash} not found")
        self.active_config = self.configs[config_hash]
    
    def start_run(self, equip_id: int) -> ExperimentRun:
        """Start an experiment run."""
        if not self.active_config:
            raise ValueError("No active config")
        
        run = ExperimentRun(
            run_id=f"EXP_{equip_id}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
            config_version=self.active_config.config_version,
            config_hash=self.active_config.config_hash,
            equip_id=equip_id,
            started_at=pd.Timestamp.now(),
            completed_at=None,
            health_mean=None,
            episode_count=None,
            rul_hours=None,
            baseline_config_hash=None,
            health_delta=None
        )
        self.runs.append(run)
        return run
    
    def complete_run(self,
                     run_id: str,
                     health_mean: float,
                     episode_count: int,
                     rul_hours: Optional[float]) -> None:
        """Complete an experiment run with results."""
        for run in self.runs:
            if run.run_id == run_id:
                run.completed_at = pd.Timestamp.now()
                run.health_mean = health_mean
                run.episode_count = episode_count
                run.rul_hours = rul_hours
                break
    
    def compare_configs(self,
                        config_hash_a: str,
                        config_hash_b: str,
                        equip_id: int) -> Dict[str, Any]:
        """Compare results between two configurations."""
        runs_a = [r for r in self.runs 
                  if r.config_hash == config_hash_a and r.equip_id == equip_id]
        runs_b = [r for r in self.runs 
                  if r.config_hash == config_hash_b and r.equip_id == equip_id]
        
        if not runs_a or not runs_b:
            return {"error": "Insufficient runs for comparison"}
        
        health_a = np.mean([r.health_mean for r in runs_a if r.health_mean])
        health_b = np.mean([r.health_mean for r in runs_b if r.health_mean])
        
        episodes_a = np.mean([r.episode_count for r in runs_a if r.episode_count is not None])
        episodes_b = np.mean([r.episode_count for r in runs_b if r.episode_count is not None])
        
        return {
            "config_a": config_hash_a,
            "config_b": config_hash_b,
            "health_delta": health_b - health_a,
            "episode_delta": episodes_b - episodes_a,
            "runs_a": len(runs_a),
            "runs_b": len(runs_b)
        }
```

**SQL Schema**:
```sql
CREATE TABLE ACM_ExperimentConfig (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    ConfigHash NVARCHAR(20) NOT NULL UNIQUE,
    ConfigVersion NVARCHAR(20) NOT NULL,
    ConfigJSON NVARCHAR(MAX) NOT NULL,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    IsActive BIT DEFAULT 0
);

CREATE TABLE ACM_ExperimentLog (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    RunID NVARCHAR(100) NOT NULL,
    ConfigHash NVARCHAR(20) NOT NULL,
    EquipID INT NOT NULL,
    
    StartedAt DATETIME2 NOT NULL,
    CompletedAt DATETIME2 NULL,
    
    -- Outcomes
    HealthMean FLOAT NULL,
    EpisodeCount INT NULL,
    RULHours FLOAT NULL,
    
    -- Comparison
    BaselineConfigHash NVARCHAR(20) NULL,
    HealthDelta FLOAT NULL,
    
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_ExperimentLog_Equipment FOREIGN KEY (EquipID) REFERENCES Equipment(EquipID)
);
CREATE INDEX IX_ExperimentLog_Config ON ACM_ExperimentLog(ConfigHash, EquipID);
```

| Task | File | Status |
|------|------|--------|
| [ ] Create experiment tracking system | `core/experiment_manager.py` | ⏳ |
| [ ] Version all analytics configurations | `core/experiment_manager.py` | ⏳ |
| [ ] Create `ACM_ExperimentLog` table | `scripts/sql/migrations/` | ⏳ |

### P5.16 — Model Deprecation Workflow (Item 48)

**Implementation**:
```python
# Add to core/model_persistence.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import pandas as pd

class DeprecationReason(Enum):
    """Reasons for model deprecation."""
    SUPERSEDED = "SUPERSEDED"       # New model replaced it
    DRIFT = "DRIFT"                 # Data drift invalidated it
    POOR_PERFORMANCE = "POOR_PERFORMANCE"  # Performance degraded
    MANUAL = "MANUAL"               # Operator decision
    EXPERIMENT = "EXPERIMENT"       # Experiment concluded

class ModelState(Enum):
    """Model lifecycle states."""
    ACTIVE = "ACTIVE"           # Currently in use
    DEPRECATED = "DEPRECATED"   # No longer used, kept for audit
    ARCHIVED = "ARCHIVED"       # Moved to cold storage
    DELETED = "DELETED"         # Marked for deletion

@dataclass
class DeprecationRecord:
    """Record of model deprecation."""
    model_id: str
    model_type: str
    equip_id: int
    
    deprecated_at: pd.Timestamp
    reason: DeprecationReason
    reason_details: str
    
    # Replacement info
    replaced_by: Optional[str]  # New model ID
    
    # Performance at deprecation
    final_accuracy: Optional[float]
    final_fpr: Optional[float]
    
    # Audit
    deprecated_by: str  # "SYSTEM" or operator ID

class ModelDeprecator:
    """
    Formal workflow for model deprecation.
    
    Ensures:
    - Models are never silently replaced
    - Audit trail exists for all deprecations
    - Old models available for forensic comparison
    """
    
    def __init__(self, registry: "ModelRegistry"):
        self.registry = registry
        self.deprecation_log: List[DeprecationRecord] = []
    
    def deprecate(self,
                  model_id: str,
                  reason: DeprecationReason,
                  reason_details: str,
                  replaced_by: Optional[str] = None,
                  deprecated_by: str = "SYSTEM") -> DeprecationRecord:
        """
        Deprecate a model with full audit trail.
        """
        # Get model metadata
        model = self.registry.get_model_metadata(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Create deprecation record
        record = DeprecationRecord(
            model_id=model_id,
            model_type=model.get("model_type", "unknown"),
            equip_id=model.get("equip_id", 0),
            deprecated_at=pd.Timestamp.now(),
            reason=reason,
            reason_details=reason_details,
            replaced_by=replaced_by,
            final_accuracy=model.get("accuracy"),
            final_fpr=model.get("false_positive_rate"),
            deprecated_by=deprecated_by
        )
        
        # Update model state in registry
        self.registry.update_model_state(model_id, ModelState.DEPRECATED)
        
        # Log deprecation
        self.deprecation_log.append(record)
        
        return record
    
    def compare_models(self,
                       old_model_id: str,
                       new_model_id: str,
                       test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Forensic comparison between old and new models.
        
        Useful for debugging why a new model behaves differently.
        """
        old_model = self.registry.load_model(old_model_id)
        new_model = self.registry.load_model(new_model_id)
        
        if not old_model or not new_model:
            return {"error": "Models not found"}
        
        # Score test data with both models
        old_scores = old_model.score(test_data)
        new_scores = new_model.score(test_data)
        
        # Compare outputs
        return {
            "old_model": old_model_id,
            "new_model": new_model_id,
            "score_correlation": old_scores["z_score"].corr(new_scores["z_score"]),
            "old_anomaly_rate": old_scores["is_anomaly"].mean(),
            "new_anomaly_rate": new_scores["is_anomaly"].mean(),
            "anomaly_agreement": (old_scores["is_anomaly"] == new_scores["is_anomaly"]).mean(),
            "max_score_diff": (old_scores["z_score"] - new_scores["z_score"]).abs().max()
        }
    
    def get_deprecation_history(self, 
                                 equip_id: Optional[int] = None,
                                 days: int = 90) -> List[DeprecationRecord]:
        """Get recent deprecation history."""
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        
        records = [r for r in self.deprecation_log if r.deprecated_at > cutoff]
        
        if equip_id:
            records = [r for r in records if r.equip_id == equip_id]
        
        return sorted(records, key=lambda r: r.deprecated_at, reverse=True)
    
    def can_restore(self, model_id: str) -> bool:
        """Check if deprecated model can be restored."""
        model = self.registry.get_model_metadata(model_id)
        if not model:
            return False
        
        state = model.get("state")
        return state == ModelState.DEPRECATED.value  # Not yet archived/deleted
    
    def restore(self, model_id: str) -> bool:
        """Restore a deprecated model to active status."""
        if not self.can_restore(model_id):
            return False
        
        self.registry.update_model_state(model_id, ModelState.ACTIVE)
        return True
```

**SQL Schema**:
```sql
CREATE TABLE ACM_ModelDeprecationLog (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    ModelID NVARCHAR(100) NOT NULL,
    ModelType NVARCHAR(50) NOT NULL,
    EquipID INT NOT NULL,
    
    DeprecatedAt DATETIME2 NOT NULL,
    Reason NVARCHAR(30) NOT NULL,
    ReasonDetails NVARCHAR(500) NULL,
    
    ReplacedBy NVARCHAR(100) NULL,
    FinalAccuracy FLOAT NULL,
    FinalFPR FLOAT NULL,
    
    DeprecatedBy NVARCHAR(50) NOT NULL,
    
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_ModelDeprecation_Equipment FOREIGN KEY (EquipID) REFERENCES Equipment(EquipID)
);
CREATE INDEX IX_ModelDeprecation_Model ON ACM_ModelDeprecationLog(ModelID);
CREATE INDEX IX_ModelDeprecation_Equip ON ACM_ModelDeprecationLog(EquipID, DeprecatedAt DESC);

-- Add state column to ModelRegistry if not exists
ALTER TABLE ModelRegistry ADD 
    ModelState NVARCHAR(20) DEFAULT 'ACTIVE',
    DeprecatedAt DATETIME2 NULL;
```

| Task | File | Status |
|------|------|--------|
| [ ] Create `ModelDeprecator` class | `core/model_persistence.py` | ⏳ |
| [ ] Implement formal deprecation workflow | `core/model_persistence.py` | ⏳ |
| [ ] Enable forensic comparison | `core/model_persistence.py` | ⏳ |
| [ ] Create `ACM_ModelDeprecationLog` table | `scripts/sql/migrations/` | ⏳ |

### P5.17 — Separate Analytics from Decision Policy (Item 50)

**Implementation**:
```python
# core/decision_policy.py (enhancement to P5.8)

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import json

@dataclass
class PolicyVersion:
    """Versioned decision policy."""
    version: str
    created_at: pd.Timestamp
    created_by: str
    
    # Policy parameters (can change WITHOUT retraining models)
    state_action_map: Dict[str, str]
    rul_immediate_hours: float
    rul_schedule_hours: float
    min_confidence_for_action: float
    escalation_thresholds: Dict[str, float]
    suppression_rules: Dict[str, Any]

class DecisionPolicyManager:
    """
    Manage decision policies separately from analytics models.
    
    KEY INSIGHT: Analytics outputs (health, RUL, episodes) are PURE.
    Decision policies (what to do about them) are CONFIGURABLE.
    
    This allows:
    - Policy changes without model retraining
    - Different policies per equipment type
    - A/B testing of policies
    - Policy rollback
    """
    
    def __init__(self):
        self.policies: Dict[str, PolicyVersion] = {}
        self.active_policy: Optional[str] = None
        self.policy_by_equipment: Dict[int, str] = {}  # equip_id -> policy version
    
    def create_policy(self,
                      version: str,
                      created_by: str = "SYSTEM",
                      **kwargs) -> PolicyVersion:
        """Create new policy version."""
        policy = PolicyVersion(
            version=version,
            created_at=pd.Timestamp.now(),
            created_by=created_by,
            state_action_map=kwargs.get("state_action_map", {
                "HEALTHY": "NO_ACTION",
                "DEGRADED": "MONITOR",
                "CRITICAL": "INVESTIGATE",
                "UNKNOWN": "MONITOR"
            }),
            rul_immediate_hours=kwargs.get("rul_immediate_hours", 24.0),
            rul_schedule_hours=kwargs.get("rul_schedule_hours", 168.0),
            min_confidence_for_action=kwargs.get("min_confidence_for_action", 0.5),
            escalation_thresholds=kwargs.get("escalation_thresholds", {
                "TIER_2": 2.0,  # hours before escalating
                "TIER_3": 6.0,
                "TIER_4": 24.0
            }),
            suppression_rules=kwargs.get("suppression_rules", {
                "max_alerts_per_hour": 5,
                "cooldown_minutes": 30
            })
        )
        
        self.policies[version] = policy
        return policy
    
    def activate_policy(self, 
                        version: str, 
                        equip_id: Optional[int] = None) -> None:
        """
        Activate a policy version.
        
        If equip_id provided, activates for that equipment only.
        Otherwise, sets as global default.
        """
        if version not in self.policies:
            raise ValueError(f"Policy {version} not found")
        
        if equip_id:
            self.policy_by_equipment[equip_id] = version
        else:
            self.active_policy = version
    
    def get_policy(self, equip_id: int) -> PolicyVersion:
        """Get active policy for equipment."""
        # Check equipment-specific override
        if equip_id in self.policy_by_equipment:
            return self.policies[self.policy_by_equipment[equip_id]]
        
        # Fall back to global
        if self.active_policy:
            return self.policies[self.active_policy]
        
        # Create default
        return self.create_policy("default")
    
    def apply_policy(self,
                     analytics_output: Dict[str, Any],
                     equip_id: int) -> "DecisionContract":
        """
        Apply policy to analytics output to get decision.
        
        Analytics output contains: health_state, health_confidence,
        rul_status, rul_hours, rul_confidence, active_episodes,
        worst_severity, system_confidence, limiting_factor
        """
        policy = self.get_policy(equip_id)
        
        # Map health state to action
        health_state = analytics_output.get("health_state", "UNKNOWN")
        base_action = policy.state_action_map.get(health_state, "MONITOR")
        
        # Override based on RUL
        rul_hours = analytics_output.get("rul_hours")
        rul_status = analytics_output.get("rul_status", "NOT_RELIABLE")
        
        if rul_status == "RELIABLE" and rul_hours is not None:
            if rul_hours < policy.rul_immediate_hours:
                base_action = "IMMEDIATE_ACTION"
            elif rul_hours < policy.rul_schedule_hours:
                if base_action in ["NO_ACTION", "MONITOR"]:
                    base_action = "SCHEDULE_MAINTENANCE"
        
        # Override based on episode severity
        if analytics_output.get("worst_severity") == "CRITICAL":
            if base_action in ["NO_ACTION", "MONITOR"]:
                base_action = "INVESTIGATE"
        
        # Apply confidence gate
        confidence = analytics_output.get("system_confidence", 0.5)
        if confidence < policy.min_confidence_for_action:
            if base_action in ["IMMEDIATE_ACTION", "SCHEDULE_MAINTENANCE"]:
                base_action = "INVESTIGATE"
        
        # Build decision contract
        from core.decision_policy import DecisionContract, RecommendedAction
        
        return DecisionContract(
            timestamp=pd.Timestamp.now(),
            equip_id=equip_id,
            state=health_state,
            state_confidence=analytics_output.get("health_confidence", 0.5),
            rul_status=rul_status,
            rul_hours=rul_hours,
            rul_confidence=analytics_output.get("rul_confidence", 0.5),
            active_episodes=analytics_output.get("active_episodes", 0),
            worst_episode_severity=analytics_output.get("worst_severity"),
            recommended_action=RecommendedAction[base_action],
            action_reason=f"Policy {policy.version}: {health_state} state",
            action_urgency_hours=rul_hours if base_action == "IMMEDIATE_ACTION" else None,
            system_confidence=confidence,
            limiting_factor=analytics_output.get("limiting_factor", "unknown")
        )
    
    def export_policy(self, version: str) -> str:
        """Export policy as JSON for versioning."""
        if version not in self.policies:
            raise ValueError(f"Policy {version} not found")
        
        policy = self.policies[version]
        return json.dumps({
            "version": policy.version,
            "created_at": policy.created_at.isoformat(),
            "created_by": policy.created_by,
            "state_action_map": policy.state_action_map,
            "rul_immediate_hours": policy.rul_immediate_hours,
            "rul_schedule_hours": policy.rul_schedule_hours,
            "min_confidence_for_action": policy.min_confidence_for_action,
            "escalation_thresholds": policy.escalation_thresholds,
            "suppression_rules": policy.suppression_rules
        }, indent=2)
    
    def import_policy(self, policy_json: str) -> PolicyVersion:
        """Import policy from JSON."""
        data = json.loads(policy_json)
        return self.create_policy(
            version=data["version"],
            created_by=data.get("created_by", "IMPORT"),
            state_action_map=data.get("state_action_map"),
            rul_immediate_hours=data.get("rul_immediate_hours"),
            rul_schedule_hours=data.get("rul_schedule_hours"),
            min_confidence_for_action=data.get("min_confidence_for_action"),
            escalation_thresholds=data.get("escalation_thresholds"),
            suppression_rules=data.get("suppression_rules")
        )
```

**SQL Schema**:
```sql
CREATE TABLE ACM_DecisionPolicy (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    PolicyVersion NVARCHAR(50) NOT NULL UNIQUE,
    PolicyJSON NVARCHAR(MAX) NOT NULL,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CreatedBy NVARCHAR(50) NOT NULL,
    IsActive BIT DEFAULT 0
);

CREATE TABLE ACM_PolicyAssignments (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    EquipID INT NOT NULL,
    PolicyVersion NVARCHAR(50) NOT NULL,
    AssignedAt DATETIME2 DEFAULT GETDATE(),
    AssignedBy NVARCHAR(50) NOT NULL,
    CONSTRAINT FK_PolicyAssignments_Equipment FOREIGN KEY (EquipID) REFERENCES Equipment(EquipID)
);
```

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
