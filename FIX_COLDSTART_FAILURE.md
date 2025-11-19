# FD_FAN Coldstart Failure Fix

## Issues Identified

### 1. **Missing dbo.Equipments Table**
- Your environment expects `dbo.Equipments` (plural) but table is named `dbo.Equipment` (singular)
- Error: `Invalid object name 'dbo.Equipments'`
- Stored procedures (`usp_ACM_RegisterEquipment`) reference plural name

### 2. **Missing Equipment Seed Data**
- `ACM_ColdstartState` has FK constraint requiring `dbo.Equipment(EquipID)`
- EquipID 5396 (FD_FAN) not present in Equipment table
- Error: `FK_ColdstartState_Equipment constraint violation`

### 3. **Stored Procedure Signature Mismatch**
- Your `usp_ACM_StartRun` has different signature than expected
- Error 1: "Procedure has too many arguments specified" (expects fewer than 7 params)
- Error 2: "@WindowStartEntryDateTime was not declared as an OUTPUT parameter"
- Code was assuming a specific 7-parameter OUTPUT signature

### 4. **Inactive Historian Tags for FD_FAN**
- Error: `No active tags found for EquipID 5396`
- `usp_ACM_GetHistorianData_TEMP` cannot find tag mappings for FD_FAN

## Changes Made

### A. Updated `scripts/sql/99_env_compat_shims.sql`

**Change 1: Use table instead of synonym** (if synonyms disallowed in your environment)
- Creates `dbo.Equipments` as a separate table (not synonym)
- Copies rows from `dbo.Equipment` to `dbo.Equipments`
- Safe for environments that disable synonyms

**Change 2: Seeds both tables**
- Inserts FD_FAN (EquipID=5396) into both `Equipment` and `Equipments`
- Inserts GAS_TURBINE (EquipID=1) into both tables
- Uses `IF NOT EXISTS` guards for idempotence

### B. Updated `core/acm_main.py::_sql_start_run()`

**Changed SP call strategy from "try one signature with OUTPUT params" to "progressive fallback":**

1. **Try minimal 3-param signature first** (most compatible):
   ```sql
   EXEC dbo.usp_ACM_StartRun @EquipID = ?, @Stage = ?, @TickMinutes = ?;
   ```

2. **Try full OUTPUT signature** (if minimal fails):
   ```sql
   DECLARE @RunID UNIQUEIDENTIFIER, @WS DATETIME2(3), @WE DATETIME2(3), @EID INT;
   EXEC dbo.usp_ACM_StartRun
       @EquipID = ?, @Stage = ?, @TickMinutes = ?,
       @DefaultStartUtc = ?, @Version = ?, @ConfigHash = ?, @TriggerReason = ?,
       @RunID = @RunID OUTPUT, @WindowStartEntryDateTime = @WS OUTPUT,
       @WindowEndEntryDateTime = @WE OUTPUT, @EquipIDOut = @EID OUTPUT;
   SELECT CONVERT(varchar(36), @RunID) AS RunID, @WS, @WE, @EID AS EquipID;
   ```

3. **Try 7-param without OUTPUTs** (if OUTPUT signature fails):
   ```sql
   EXEC dbo.usp_ACM_StartRun
       @EquipID = ?, @Stage = ?, @TickMinutes = ?,
       @DefaultStartUtc = ?, @Version = ?, @ConfigHash = ?, @TriggerReason = ?;
   ```

4. **Synthetic fallback** (if all SP calls fail):
   - Generates UUID for RunID
   - Calculates window based on tick_minutes
   - Allows processing to continue

**Added detailed debug logging** to show which signature succeeded/failed.

## How to Apply Fix

### Step 1: Apply Environment Compatibility Shims

**Option A: Using SSMS**
1. Open SSMS and connect to your SQL Server instance
2. Open file: `scripts\sql\99_env_compat_shims.sql`
3. Select database: `ACM`
4. Execute (F5)

**Option B: Using sqlcmd**
```powershell
sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -i "scripts\sql\99_env_compat_shims.sql"
```

**Option C: Using PowerShell with SQL auth** (if Windows auth fails)
```powershell
$Server = "localhost\B19CL3PCQLSERVER"
$Database = "ACM"
$User = $env:ACM_SQL_USER
$Pass = $env:ACM_SQL_PASSWORD
sqlcmd -S $Server -d $Database -U $User -P $Pass -b -i "scripts\sql\99_env_compat_shims.sql"
```

### Step 2: Verify Tables Created

```sql
-- Check Equipment tables exist
SELECT * FROM dbo.Equipment WHERE EquipID IN (1, 5396);
SELECT * FROM dbo.Equipments WHERE EquipID IN (1, 5396);

-- Verify ACM_ColdstartState table exists
SELECT * FROM dbo.ACM_ColdstartState WHERE EquipID = 5396;

-- Verify RunLog exists (optional but eliminates warnings)
SELECT TOP 5 * FROM dbo.RunLog ORDER BY StartTime DESC;
```

### Step 3: Fix Historian Tag Mappings (CRITICAL)

The coldstart will still fail if historian has no active tags for FD_FAN (EquipID=5396).

**Check current tag mappings:**
```sql
-- Find your tag mapping table (name varies by installation)
-- Common names: TagMapping, EquipmentTags, HistorianTags, ACM_TagMap
SELECT * FROM <YourTagMappingTable> WHERE EquipID = 5396;
```

**If no rows found, you need to create tag mappings for FD_FAN.**

Example (adjust table/column names to match your schema):
```sql
-- Example: activate tags for FD_FAN
INSERT INTO TagMapping (EquipID, TagName, IsActive, CreatedAt)
VALUES 
    (5396, 'FD_FAN_SPEED', 1, GETDATE()),
    (5396, 'FD_FAN_CURRENT', 1, GETDATE()),
    (5396, 'FD_FAN_VIBRATION', 1, GETDATE()),
    (5396, 'FD_FAN_TEMP', 1, GETDATE());
```

**OR** modify your historian SP to use equipment name instead of ID:
```sql
-- Example: update usp_ACM_GetHistorianData_TEMP to accept @EquipmentName
-- and lookup tags by name if EquipID has no mappings
ALTER PROCEDURE dbo.usp_ACM_GetHistorianData_TEMP
    @StartTime DATETIME,
    @EndTime DATETIME,
    @EquipID INT = NULL,
    @EquipmentName NVARCHAR(128) = NULL
AS
BEGIN
    -- Add logic to lookup by name if ID lookup fails
    IF @EquipID IS NOT NULL AND NOT EXISTS (SELECT 1 FROM TagMapping WHERE EquipID = @EquipID AND IsActive = 1)
    BEGIN
        -- Fallback: lookup by equipment name
        SELECT @EquipID = EquipID FROM dbo.Equipment WHERE EquipCode = @EquipmentName;
    END
    -- ... rest of your historian query
END
```

### Step 4: Test FD_FAN Coldstart

```powershell
# Test single run
python -m core.acm_main --equip FD_FAN

# Test coldstart-specific run (if batch runner has coldstart mode)
python scripts/sql_batch_runner.py --equip FD_FAN --start-from-beginning --max-batches 1
```

**Expected output (success):**
```
[DEBUG] Trying minimal 3-param signature...
[DEBUG] Minimal signature succeeded
[DEBUG] Parsed row: RunID=<UUID>, WS=<timestamp>, WE=<timestamp>, EquipID=5396
[RUN] Started RunID=<UUID> window=[...] equip='FD_FAN' EquipID=5396
[DATA] Retrieved 1234 rows from SQL historian for window [...]
[COLDSTART] Accumulated 1234 rows (target: 500)
[COLDSTART] Status: COMPLETE
```

**If coldstart still fails**, check logs for:
- "No active tags found for EquipID 5396" → Fix historian mappings (Step 3)
- "Invalid object name" → Re-run Step 1
- "FK constraint violation" → Verify Step 2 succeeded

## Verification Queries

After applying fix, run these to confirm environment is ready:

```sql
-- 1. Check Equipment seeded
SELECT EquipID, EquipCode, EquipName, IsActive FROM dbo.Equipment WHERE EquipID IN (1, 5396);
SELECT EquipID, EquipCode, EquipName, IsActive FROM dbo.Equipments WHERE EquipID IN (1, 5396);

-- 2. Check coldstart table exists and has FK
SELECT 
    OBJECT_NAME(parent_object_id) AS TableName,
    name AS ConstraintName,
    type_desc AS ConstraintType
FROM sys.foreign_keys
WHERE OBJECT_NAME(parent_object_id) = 'ACM_ColdstartState';

-- 3. Check historian data available for FD_FAN
EXEC dbo.usp_ACM_GetHistorianData_TEMP 
    @StartTime = '2023-11-15', 
    @EndTime = '2023-11-16', 
    @EquipID = 5396;
-- Should return >0 rows. If 0 rows, fix tag mappings (Step 3)

-- 4. Test equipment registration
EXEC dbo.usp_ACM_RegisterEquipment @EquipCode = 'FD_FAN', @EquipID = NULL;
-- Should return EquipID = 5396 without error
```

## Rollback Instructions

If you need to undo changes:

```sql
-- Remove tables created by shim script
DROP TABLE IF EXISTS dbo.Equipments;
DROP TABLE IF EXISTS dbo.RunLog;
DROP TABLE IF EXISTS dbo.ACM_ColdstartState;

-- Remove seeded Equipment rows
DELETE FROM dbo.Equipment WHERE EquipID IN (1, 5396);
```

Python code changes are backward compatible - no rollback needed.

## Summary

**Root Cause**: Environment missing base tables/rows expected by ACM codebase; stored procedure signatures don't match code assumptions.

**Solution**: 
1. Create compatibility layer (Equipment/Equipments tables with seed data)
2. Make SP call logic adaptive to different signatures
3. Ensure historian tags configured for FD_FAN

**Next Steps After Fix**:
1. Apply `99_env_compat_shims.sql` 
2. Verify Equipment tables seeded
3. Configure historian tags for EquipID=5396
4. Test FD_FAN coldstart run
5. Once successful, run full batch: `python scripts/sql_batch_runner.py --equip FD_FAN GAS_TURBINE --start-from-beginning --max-batches 8`
