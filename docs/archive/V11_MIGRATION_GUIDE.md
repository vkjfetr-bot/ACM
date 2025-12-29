# ACM v11.0.0 Migration Guide

**Version**: 11.0.0  
**Created**: 2025-01-18  
**Status**: Release

---

## Overview

This guide covers upgrading from ACM v10.x to v11.0.0. The upgrade introduces breaking changes in:

1. SQL table structure (new tables, deprecated tables)
2. Logging API (Console class replaces print/legacy loggers)
3. Regime lifecycle management (MaturityState)
4. Detector interface (DetectorProtocol ABC)

---

## Prerequisites

- ACM v10.3.0 or later running
- SQL Server access with DDL permissions
- Python 3.11 environment
- Docker (for observability stack)

---

## Step 1: Database Schema Updates

### 1.1 Create New Tables

Run the SQL migration script to create v11 tables:

```powershell
# From ACM root directory
sqlcmd -S "YourServer\Instance" -d ACM -E -i "install/sql/10_tables.sql"
```

**New Tables Created:**
- `ACM_ActiveModels` - Model versioning per equipment
- `ACM_RegimeDefinitions` - Immutable regime definitions
- `ACM_DataContractValidation` - Validation history
- `ACM_SeasonalPatterns` - Detected patterns
- `ACM_AssetProfiles` - Similarity profiles

### 1.2 Deprecated Tables

**ACM_Scores_Long** is deprecated in v11.0.0:
- This table duplicated ACM_Scores_Wide in long format
- Was writing ~44K rows per batch unnecessarily
- `write_scores_ts()` now returns 0 (no-op)
- Table can be dropped after confirming no downstream dependencies

```sql
-- Optional: Drop deprecated table after migration
-- DROP TABLE ACM_Scores_Long;
```

---

## Step 2: Code Updates

### 2.1 Logging Migration

**Before (v10.x):**
```python
print(f"Processing {equipment}")
logger.info("Starting pipeline")
```

**After (v11.0.0):**
```python
from core.observability import Console

Console.info(f"Processing {equipment}")
Console.ok("Pipeline complete")
Console.status("Progress: 50%")  # Console-only, not logged
```

**Console Methods:**
| Method | Logged to Loki | Purpose |
|--------|----------------|---------|
| `Console.info()` | ✅ Yes | General info |
| `Console.warn()` | ✅ Yes | Warnings |
| `Console.error()` | ✅ Yes | Errors |
| `Console.ok()` | ✅ Yes | Success messages |
| `Console.status()` | ❌ No | Progress/temp output |
| `Console.header()` | ❌ No | Section headers |
| `Console.section()` | ❌ No | Light separators |

### 2.2 DataContract Integration

Add validation at pipeline entry:

```python
from core.pipeline_types import DataContract, ValidationResult

def run_pipeline(df, equipment_name):
    contract = DataContract(equipment_name=equipment_name)
    result = contract.validate(df)
    
    if not result.is_valid:
        Console.warn(f"Validation issues: {len(result.issues)}")
        for issue in result.issues:
            Console.warn(f"  - {issue}")
    
    # Continue with pipeline...
```

### 2.3 Regime Maturity Checks

Gate regime-dependent features on maturity:

```python
from core.regime_manager import ActiveModelsManager, MaturityState

manager = ActiveModelsManager(sql_client)
active = manager.get_active(equip_id)

if active.regime_maturity == MaturityState.CONVERGED:
    # Use regime-specific thresholds
    thresholds = get_regime_thresholds(active.regime_version)
else:
    # Fall back to global thresholds
    thresholds = get_global_thresholds()
```

---

## Step 3: Configuration Updates

### 3.1 New Config Keys

Add to `configs/config_table.csv`:

```csv
equipment,key,value,description
*,datacontract.min_rows,100,Minimum rows for valid batch
*,datacontract.max_null_pct,0.5,Max null percentage per sensor
*,seasonality.enabled,true,Enable seasonality detection
*,seasonality.min_cycles,3,Minimum cycles to detect pattern
*,similarity.min_overlap,0.7,Minimum sensor overlap for transfer
*,regime.maturity_window_days,30,Days before LEARNING to CONVERGED
```

### 3.2 Sync to SQL

```powershell
python scripts/sql/populate_acm_config.py
```

---

## Step 4: Observability Stack

### 4.1 Start Docker Stack

```powershell
cd install/observability
docker compose up -d
```

### 4.2 Verify Containers

```powershell
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Expected containers:
- `acm-grafana` (port 3000) - Dashboard UI
- `acm-alloy` (ports 4317, 4318) - OTLP collector
- `acm-tempo` (port 3200) - Traces
- `acm-loki` (port 3100) - Logs
- `acm-prometheus` (port 9090) - Metrics
- `acm-pyroscope` (port 4040) - Profiling

### 4.3 Access Grafana

Open http://localhost:3000 (admin/admin)

Dashboards are auto-provisioned in the ACM folder.

---

## Step 5: Testing

### 5.1 Verify Imports

```powershell
python -c "from core.pipeline_types import DataContract; print('OK')"
python -c "from core.feature_matrix import FeatureMatrix; print('OK')"
python -c "from core.detector_protocol import DetectorProtocol; print('OK')"
python -c "from core.regime_manager import ActiveModelsManager; print('OK')"
python -c "from core.seasonality import SeasonalityHandler; print('OK')"
python -c "from core.asset_similarity import AssetSimilarity; print('OK')"
```

### 5.2 Run Test Batch

```powershell
python scripts/sql_batch_runner.py --equip GAS_TURBINE --tick-minutes 1440 --max-workers 1
```

### 5.3 Check Validation Logs

```sql
SELECT TOP 10 * 
FROM ACM_DataContractValidation 
ORDER BY CreatedAt DESC
```

---

## Breaking Changes Summary

| Change | Impact | Migration |
|--------|--------|-----------|
| `print()` deprecated | Build warnings | Use `Console.*` |
| `ACM_Scores_Long` deprecated | No long-format scores | Use `ACM_Scores_Wide` |
| Regime maturity required | Thresholds gated | Initialize `ACM_ActiveModels` |
| DataContract validation | Pipeline entry check | Add validation call |
| DetectorProtocol ABC | Detector interface | Implement protocol methods |

---

## Rollback Procedure

If issues occur:

1. **Stop ACM services**
2. **Restore v10.x code**: `git checkout v10.3.0`
3. **Keep v11 tables**: They're additive, won't break v10
4. **Re-enable ACM_Scores_Long**: Remove deprecation in output_manager.py

---

## Support

- Documentation: `docs/` folder
- SQL Schema: `docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md`
- Observability: `install/observability/README.md`
