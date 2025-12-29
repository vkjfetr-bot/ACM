# ACM v11.0.0 Architecture

**Version**: 11.0.0  
**Created**: 2025-01-18  
**Status**: Release

---

## Overview

ACM v11.0.0 introduces a refactored architecture focused on:

1. **Typed Data Contracts** - Validation at pipeline entry
2. **Maturity-Based Regime Lifecycle** - Clear state transitions
3. **Standardized Feature Matrix** - Schema-enforced feature engineering
4. **Detector Protocol ABC** - Unified detector interface
5. **Seasonality Detection** - Diurnal/weekly pattern handling
6. **Asset Similarity** - Cold-start transfer learning
7. **SQL Performance** - Deprecated redundant writes, batched operations

---

## Core Components

### 1. DataContract (core/pipeline_types.py)

Entry-point validation for all data entering the pipeline.

```python
from core.pipeline_types import DataContract, ValidationResult

contract = DataContract(equipment_name="GAS_TURBINE")
result: ValidationResult = contract.validate(dataframe)

if not result.is_valid:
    for issue in result.issues:
        Console.warn(f"Validation: {issue}")
```

**Validation Rules:**
- Required columns: `Timestamp`, `EquipID`
- Timestamp monotonic increasing
- No completely empty sensor columns
- Minimum row count (configurable)

### 2. FeatureMatrix (core/feature_matrix.py)

Standardized feature representation with schema enforcement.

```python
from core.feature_matrix import FeatureMatrix

matrix = FeatureMatrix(
    data=features_df,
    sensor_columns=["Temp1", "Pressure", "Vibration"],
    regime_id=1,
    timestamp_col="Timestamp"
)

# Access typed features
features_array = matrix.to_numpy()
```

**Schema Enforcement:**
- Column naming conventions
- Type checking (float64 for sensors)
- NaN handling policies
- Regime association

### 3. MaturityState (core/regime_manager.py)

Regime lifecycle management with clear state transitions.

```
INITIALIZING → LEARNING → CONVERGED → DEPRECATED
     ↑              ↓
     └──────────────┘ (restart on drift)
```

**States:**
- `INITIALIZING`: No regimes discovered yet (cold-start)
- `LEARNING`: Regimes discovered but not validated
- `CONVERGED`: Stable regimes, thresholds active
- `DEPRECATED`: Replaced by newer version

```python
from core.regime_manager import ActiveModelsManager, MaturityState

manager = ActiveModelsManager(sql_client)
active = manager.get_active(equip_id)

if active.regime_maturity == MaturityState.CONVERGED:
    # Use regime-specific thresholds
    ...
```

### 4. DetectorProtocol (core/detector_protocol.py)

Abstract base class for all anomaly detectors.

```python
from core.detector_protocol import DetectorProtocol, DetectorOutput

class MyDetector(DetectorProtocol):
    name = "my_detector"
    z_prefix = "my_z"
    
    def fit(self, X: np.ndarray, regime_id: int) -> None:
        ...
    
    def score(self, X: np.ndarray) -> DetectorOutput:
        return DetectorOutput(
            z_scores=z_array,
            raw_scores=raw_array,
            confidence=0.85
        )
```

**Protocol Requirements:**
- `name`: Unique detector identifier
- `z_prefix`: Column prefix for z-scores
- `fit()`: Train on baseline data
- `score()`: Produce anomaly scores
- `is_fitted`: Property indicating training status

### 5. SeasonalityHandler (core/seasonality.py)

Detection and adjustment for temporal patterns.

```python
from core.seasonality import SeasonalityHandler, SeasonalPattern

handler = SeasonalityHandler()
patterns = handler.detect(df, sensor_cols)

for pattern in patterns:
    if pattern.pattern_type == "diurnal":
        df = handler.adjust(df, pattern)
```

**Pattern Types:**
- `diurnal`: 24-hour cycles
- `weekly`: 7-day cycles
- `custom`: User-defined periods

### 6. AssetSimilarity (core/asset_similarity.py)

Cold-start transfer learning using similar equipment.

```python
from core.asset_similarity import AssetSimilarity

similarity = AssetSimilarity(sql_client)
donors = similarity.find_donors(
    target_equip_id=123,
    min_similarity=0.7,
    max_donors=3
)

for donor in donors:
    # Transfer baseline/thresholds from donor
    ...
```

**Similarity Metrics:**
- Sensor overlap (Jaccard)
- Value distribution (KS test)
- Regime structure similarity

---

## Data Flow

```
SQL Historian → DataContract → FeatureMatrix → Detectors → Fusion → Health/RUL
                    ↓                ↓              ↓
             ACM_DataContract  ACM_FeatureLog  ACM_Scores_Wide
                 Validation
```

### Pipeline Entry
1. Data loaded from SQL historian tables
2. `DataContract.validate()` checks data quality
3. Validation results logged to `ACM_DataContractValidation`

### Feature Engineering
1. Raw data → `FeatureMatrix` with schema enforcement
2. Seasonality detection/adjustment applied
3. Regime assignment with confidence scores

### Detection
1. Each detector implements `DetectorProtocol`
2. Detectors produce `DetectorOutput` with z-scores
3. Baseline normalization applied consistently

### Fusion
1. Detector outputs fused with calibrated weights
2. Fusion quality metrics persisted
3. Episodes triggered on fused anomaly scores

### Health/RUL
1. Health scores computed from episode severity
2. RUL estimated with Monte Carlo simulations
3. Confidence bounds (P10/P50/P90) provided

---

## SQL Tables (v11.0.0 New)

| Table | Purpose |
|-------|---------|
| `ACM_ActiveModels` | Current active model versions per equipment |
| `ACM_RegimeDefinitions` | Immutable regime centroid definitions |
| `ACM_DataContractValidation` | Validation history per run |
| `ACM_SeasonalPatterns` | Detected seasonality patterns |
| `ACM_AssetProfiles` | Equipment similarity profiles |

---

## Configuration

### New Config Keys (v11.0.0)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `datacontract.min_rows` | int | 100 | Minimum rows for valid batch |
| `datacontract.max_null_pct` | float | 0.5 | Max null percentage per sensor |
| `seasonality.enabled` | bool | true | Enable seasonality detection |
| `seasonality.min_cycles` | int | 3 | Minimum cycles to detect pattern |
| `similarity.min_overlap` | float | 0.7 | Minimum sensor overlap for transfer |
| `regime.maturity_window_days` | int | 30 | Days before LEARNING→CONVERGED |

---

## Migration from v10.x

See [V11_MIGRATION_GUIDE.md](V11_MIGRATION_GUIDE.md) for upgrade instructions.

---

## Observability

v11.0.0 includes full observability stack:

- **Traces**: OpenTelemetry → Tempo
- **Metrics**: Prometheus
- **Logs**: Loki (via Console class)
- **Profiling**: Pyroscope

See [OBSERVABILITY.md](OBSERVABILITY.md) for setup instructions.
