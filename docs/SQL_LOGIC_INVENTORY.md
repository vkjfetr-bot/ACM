# SQL Logic in acm_main.py - Complete Inventory

**Purpose**: Document all SQL operations currently in acm_main.py that should be moved to output_manager.py  
**Date**: January 2, 2026  
**Status**: BLOAT - SQL logic violates separation of concerns

---

## Summary

**Total SQL Operations Found**: 10 cursor contexts + multiple inline queries  
**Impact**: ~500-700 lines of SQL code mixed with business logic  
**Target**: Move ALL to output_manager.py

---

## Inventory by Function

### 1. `_get_equipment_id()` - Lines 480-510
**SQL Operation**: Equipment ID lookup  
**Current Code**:
```python
cur.execute("SELECT EquipID FROM Equipment WHERE EquipCode = ?", (equipment_name,))
row = cur.fetchone()
```

**Issue**: 
- Direct pyodbc connection created inline (bypasses sql_client)
- Should be centralized in output_manager

**Relocation Plan**:
- Move to: `output_manager.get_equipment_id(equipment_code: str) -> int`
- Reuse existing sql_client instead of creating new connection

---

### 2. `_sql_start_run()` - Lines 790-845
**SQL Operations**: Run creation and cleanup  
**Current Code**:
```sql
DELETE FROM dbo.ACM_HealthForecast WHERE RunID = ?
DELETE FROM dbo.ACM_FailureForecast WHERE RunID = ?
DELETE FROM dbo.ACM_RUL WHERE RunID = ?
DELETE FROM dbo.ACM_Runs WHERE RunID = ?
INSERT INTO dbo.ACM_Runs (RunID, EquipID, StartedAt, ConfigSignature) VALUES (?, ?, ?, ?)
```

**Issue**:
- Run management is persistence logic, not business logic
- Should be in output_manager

**Relocation Plan**:
- Move to: `output_manager.create_run(equip_code, config) -> Tuple[run_id, window_start, window_end, equip_id]`
- Include deadlock retry wrapper

---

### 3. `_update_baseline_buffer()` - Lines 1350-1435
**SQL Operations**: Baseline buffer writes and cleanup  
**Current Code**:
```sql
SELECT COUNT(*) FROM ACM_Runs WHERE EquipID = ? AND CreatedAt > DATEADD(DAY, -7, GETDATE())
INSERT INTO dbo.ACM_BaselineBuffer (EquipID, Timestamp, SensorName, SensorValue, DataQuality) VALUES (?, ?, ?, ?, ?)
EXEC dbo.usp_CleanupBaselineBuffer @EquipID=?, @RetentionHours=?, @MaxRowsPerEquip=?
```

**Issue**:
- 85 lines of SQL write logic in business logic layer
- Manual melt/pivot operations
- Direct cursor management

**Relocation Plan**:
- Move to: `output_manager.write_baseline_buffer(score_numeric, equip_id, coldstart_complete, cfg) -> bool`
- Return boolean success/failure

---

### 4. `_check_refit_request()` - Lines 1485-1520
**SQL Operations**: Refit request check and acknowledge  
**Current Code**:
```sql
SELECT TOP 1 RequestID, RequestedAt, Reason FROM [dbo].[ACM_RefitRequests] WHERE EquipID = ? AND Acknowledged = 0
UPDATE [dbo].[ACM_RefitRequests] SET Acknowledged = 1, AcknowledgedAt = SYSUTCDATETIME() WHERE RequestID = ?
```

**Issue**:
- Model lifecycle logic mixed with persistence
- Should be centralized

**Relocation Plan**:
- Move to: `output_manager.check_and_acknowledge_refit_request(equip_id) -> bool`

---

### 5. `_write_fusion_metrics()` - Lines 1745-1780
**SQL Operations**: Fusion quality metrics write  
**Current Code**:
```sql
INSERT INTO dbo.ACM_RunMetrics (RunID, EquipID, MetricName, MetricValue, Timestamp) VALUES (?, ?, ?, ?, ?)
```

**Issue**:
- Already has output_manager.write_run_metrics() - duplicate logic
- Should consolidate

**Relocation Plan**:
- **DELETE** this function entirely
- Call: `output_manager.write_run_metrics()` directly

---

### 6. `_log_dropped_features()` - Lines 1790-1850
**SQL Operations**: Feature drop log write  
**Current Code**:
```sql
INSERT INTO dbo.ACM_FeatureDropLog (RunID, EquipID, FeatureName, DropReason, DropValue, Threshold) VALUES (?, ?, ?, ?, ?, ?)
```

**Issue**:
- Diagnostic logging is persistence concern
- Already exists in output_manager

**Relocation Plan**:
- **DELETE** this function
- Use: `output_manager.write_feature_drop_log()` (verify exists or create)

---

### 7. `_write_data_quality()` - Lines 1860-1925
**SQL Operations**: Data quality metrics write  
**Current Code**:
```sql
INSERT INTO dbo.ACM_DataQuality (...24 columns...) VALUES (?, ?, ?, ...)
```

**Issue**:
- 65 lines of SQL write logic
- Already exists in output_manager

**Relocation Plan**:
- **DELETE** this function
- Use: `output_manager.write_data_quality()` (verify exists or create)

---

### 8. `_seed_baseline()` - Lines 2665-2710
**SQL Operations**: Baseline buffer read  
**Current Code**:
```sql
SELECT Timestamp, SensorName, SensorValue 
FROM dbo.ACM_BaselineBuffer
WHERE EquipID = ? AND Timestamp >= DATEADD(HOUR, -?, GETDATE())
```

**Issue**:
- Data loading logic in business layer
- Manual pivot operations

**Relocation Plan**:
- Move to: `output_manager.load_baseline_buffer(equip_id, window_hours) -> pd.DataFrame`
- Return pivoted wide-format DataFrame

---

### 9. `_auto_tune_parameters()` - Lines 3060-3110
**SQL Operations**: Refit request creation  
**Current Code**:
```sql
CREATE TABLE [dbo].[ACM_RefitRequests] IF NOT EXISTS (...)
INSERT INTO [dbo].[ACM_RefitRequests] (EquipID, Reason, AnomalyRate, DriftScore, RegimeQuality) VALUES (?, ?, ?, ?, ?)
```

**Issue**:
- DDL and DML mixed with business logic
- Table creation should be in schema management

**Relocation Plan**:
- Move to: `output_manager.create_refit_request(equip_id, reasons, metrics) -> bool`
- Table creation should be in migration scripts, not runtime code

---

## Relocation Strategy

### Phase 1: Create Output Manager Methods (Week 1)
1. `output_manager.get_equipment_id(equipment_code: str) -> int`
2. `output_manager.create_run(equip_code, config) -> Tuple[...]`
3. `output_manager.write_baseline_buffer(df, equip_id, cfg) -> bool`
4. `output_manager.load_baseline_buffer(equip_id, window_hours) -> pd.DataFrame`
5. `output_manager.check_and_acknowledge_refit_request(equip_id) -> bool`
6. `output_manager.create_refit_request(equip_id, reasons, metrics) -> bool`

### Phase 2: Update acm_main.py Call Sites (Week 2)
1. Replace `_get_equipment_id()` → `output_manager.get_equipment_id()`
2. Replace `_sql_start_run()` → `output_manager.create_run()`
3. Replace `_update_baseline_buffer()` → `output_manager.write_baseline_buffer()`
4. Replace `_seed_baseline()` SQL section → `output_manager.load_baseline_buffer()`
5. Replace `_check_refit_request()` → `output_manager.check_and_acknowledge_refit_request()`
6. Delete `_write_fusion_metrics()`, use existing output_manager method
7. Delete `_log_dropped_features()`, use existing output_manager method
8. Delete `_write_data_quality()`, use existing output_manager method

### Phase 3: Verify and Test (Week 3)
1. Run full batch mode tests
2. Verify no SQL logic remains in acm_main.py
3. Grep for `sql_client.cursor()` - should be ZERO matches
4. Grep for `INSERT INTO|SELECT.*FROM|UPDATE.*SET|DELETE FROM` - should be ZERO matches

---

## Expected Line Reduction

| Function | Current Lines | After Removal | Savings |
|----------|--------------|---------------|---------|
| `_get_equipment_id()` | 30 | 1 call | -29 |
| `_sql_start_run()` | 60 | 1 call | -59 |
| `_update_baseline_buffer()` | 85 | 1 call | -84 |
| `_check_refit_request()` | 35 | 1 call | -34 |
| `_write_fusion_metrics()` | 40 | 1 call | -39 |
| `_log_dropped_features()` | 60 | 1 call | -59 |
| `_write_data_quality()` | 65 | 1 call | -64 |
| `_seed_baseline()` SQL section | 50 | 1 call | -49 |
| `_auto_tune_parameters()` SQL | 50 | 1 call | -49 |
| **TOTAL** | **475** | **9 calls** | **-466 lines** |

---

## Success Criteria

✅ **Zero SQL operations in acm_main.py**  
✅ **All persistence goes through output_manager.py**  
✅ **~470 lines removed from acm_main.py**  
✅ **No functional regressions (batch mode works)**

---

## Next Steps

1. **TODAY**: Create inventory (DONE)
2. **Next**: Implement output_manager methods one by one
3. **Then**: Replace call sites in acm_main.py
4. **Finally**: Test and verify complete separation
