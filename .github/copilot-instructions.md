# ACM Copilot Guardrails (v9.0.0)

## PRIMARY DIRECTIVES
- **Mission**: Keep the ACM pipeline healthy and production-ready
- **Knowledge Base**: ALWAYS consult and reference these canonical documents:
  - `README.md` - Primary product overview, features, running ACM
  - `docs/ACM_SYSTEM_OVERVIEW.md` - Complete architectural walkthrough, module map, workflows
  - `docs/SOURCE_CONTROL_PRACTICES.md` - Git workflow, branching, versioning, release management
  - `docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md` - Authoritative ACM table/column definitions
  - `utils/version.py` - Current version info and release notes
  - Other relevant docs under `docs/` for specific topics (OMR_DETECTOR.md, COLDSTART_MODE.md, etc.)
  
- **Documentation Policy**: CRITICAL RULE
  - DO NOT create new explainer documents, guides, reports, or markdown files UNLESS EXPLICITLY REQUESTED
  - DO NOT auto-generate changelogs, task lists, or process documents
  - DO NOT create summary documents for routine changes
  - Reference existing knowledge base instead - update existing docs if needed
  - Only update README when explicitly asked to document new features
  - Update ONLY when user specifically requests documentation updates

- **Modes**: file-mode reads `data/*.csv`; SQL-mode uses `configs/sql_connection.ini` via `core/sql_client.SQLClient`. File-mode must stay working before SQL-path changes ship. DO NOT USE FILE MODE EVER!!!!!!!!
- **Batch mode**: run `python scripts/sql_batch_runner.py --equip FD_FAN GAS_TURBINE --tick-minutes 1440 --max-workers 2 --start-from-beginning` (adjust args as needed). This command sets `ACM_BATCH_MODE`/`ACM_BATCH_NUM`; do not set them in code.
- **No emojis ever** in code, comments, tests, or generated content. Never use Unicode emojis or special symbols in any output.
- **PowerShell Syntax**: ALWAYS use PowerShell v5.1 syntax for terminal commands. Use `;` for command chaining, avoid Unix-style pipes like `tail`. Use `Select-Object`, `Where-Object`, `Format-Table` instead. Example: `cd path; command1; command2` or `output | Select-Object -Last 10`
- **SQL Server Queries**: Create proper T-SQL queries using Microsoft SQL Server syntax. Always use `sqlcmd` with proper parameters: `sqlcmd -S "server\instance" -d database -E -Q "SELECT ..."`. Use T-SQL functions (DATEADD, CAST, ROUND, TOP, etc.). Never use generic SQL syntax; always target SQL Server dialect.
- **Version Management**: ACM is at v9.0.0 with centralized versioning in `utils/version.py`. When implementing changes, increment version number only when explicitly requested; follow semantic versioning (MAJOR.MINOR.PATCH).

## Core contracts
- **Config**: `utils/config_dict.ConfigDict` loads cascading `configs/config_table.csv` (global `*` rows overridden by equipment rows). Keep dot-path access intact (e.g., `cfg['fusion']['weights']['omr_z']`).
- **Output manager**: all CSV/PNG/SQL writes go through `core/output_manager.OutputManager`; respect `ALLOWED_TABLES` and batched transactions.
- **Time policy**: timestamps are local-naive; do not reintroduce UTC conversions. Use `_to_naive*` helpers in `core/output_manager.py` and `core/acm_main.py`.
- **Detectors/analytics**: detectors live in `core/` (`omr.py`, `correlation.py`, `outliers.py`, plus PCA/IForest/GMM/AR1). Fusion in `core/fuse.py`; regimes in `core/regimes.py`; drift in `core/drift.py`; episodes via `core/episode_culprits_writer`.
- **Performance**: `core/fast_features.py` supports pandas + optional Polars; keep API backward compatible and tested (`tests/test_fast_features.py`).
- **Rust bridge**: optional accelerator in `rust_bridge/`; Python path remains primary.

## Workflows
- Rapid run: `python -m core.acm_main --equip GAS_TURBINE`.
- SQL smoke: `python scripts/sql/verify_acm_connection.py` (needs SQL Server creds).
- **Schema inspection (SOLE AUTHORITATIVE TOOL)**: `python scripts/sql/export_comprehensive_schema.py --output docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md` regenerates the latest table definitions so SQL changes stay aligned with production. **CRITICAL**: This is the ONLY script to use for SQL table inspection, schema queries, dashboard panel validation, or any task requiring SQL table information. DO NOT create new scripts for table inspection, row counting, schema checks, or dashboard analysis—use this comprehensive tool exclusively.
- File-mode helper: `scripts/run_file_mode.ps1` wraps a baseline local run.

## Testing
- Targeted suites: `pytest tests/test_fast_features.py`, `pytest tests/test_dual_write.py`, `pytest tests/test_progress_tracking.py`. Respect existing skips/markers (Polars, SQL Server).

## Data & artifacts
- Sample CSVs under `data/` have datetime index + sensor columns; keep names aligned with `configs/config_table.csv`.
- `artifacts/` and `backups/` stay gitignored; do not add tracked artifact folders.

## Documentation policy
- README is authoritative. Do **not** auto-create new docs/changelogs for routine changes; only update docs when explicitly requested.
- **NO SUMMARY DOCUMENTS**: Do NOT create final summary documents, completion reports, or status summaries unless explicitly requested. Examples of prohibited documents:
  - Final completion/summary reports for code changes
  - Work completion documents
  - Status reports (unless user asks for one)
  - Analysis summaries for routine work
  - Migration or refactor summaries
  - Changelog auto-generation
- Report progress and results inline in responses only. Keep documentation to essential knowledge base only.

## Style & safety
- Python 3.11, ~100-char lines, vectorized pandas/NumPy. Use existing lint/type tooling (`ruff`, `mypy`) when touching shared modules.
- Never commit credentials; `configs/sql_connection.ini` stays local. Prefer env-var fallbacks when describing new settings.

## Source control hygiene
- Use branches for all non-trivial work: `feature/<topic>` or `fix/<topic>`; avoid pushing directly to `main`.
- Keep branches focused and short-lived; prefer small, frequent merges over large drops.
- Rebase onto `main` before opening/merging PRs to keep history clean and reduce conflicts.
- Write clear, imperative commits (e.g., “Add batch-mode env guards”); squash noisy fixups before merge.
- Run relevant tests before merging; do not merge with failing checks.
- Never commit artifacts/logs/secrets; respect `.gitignore`.
- Prefer review for all changes; merge only when checks are green and approvals are in. If self-merging is allowed, still require passing checks.

## Discoverability & Knowledge Base Quick-Links

### CANONICAL DOCUMENTS (Always consult first)
- **README.md**: Product overview, setup, features, running ACM
- **docs/ACM_SYSTEM_OVERVIEW.md**: Architecture, module map, data flow, detector heads, configuration
- **docs/SOURCE_CONTROL_PRACTICES.md**: Branching strategy, versioning, commit conventions, release process
- **utils/version.py**: Current version (v9.0.0), release notes, version helpers

### TECHNICAL DOCUMENTATION
- **Detector System**: `docs/OMR_DETECTOR.md` (Overall Model Residual), detector label consistency
- **Batch Operations**: `docs/BATCH_MODE_SQL_AUDIT.md`, `docs/BATCH_MODE_WAVE_PATTERN_FIX.md`
- **Coldstart**: `docs/COLDSTART_MODE.md` (cold-start strategy for sparse data)
- **SQL Integration**: `docs/SQL_BATCH_RUNNER.md`, `docs/SQL_BATCH_QUICK_REF.md`
- **Equipment Import**: `docs/EQUIPMENT_IMPORT_PROCEDURE.md` (how to add new equipment to ACM)
- **Schema Reference**: `docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md` (table definitions)
- **Grafana**: `grafana_dashboards/README.md` + dashboards under `grafana_dashboards/*.json`
- **Analytics**: `docs/Analytics Backbone.md` (detector fusion, episodes, regimes)

### OPERATIONAL COMMANDS & SCRIPTS
- **Entry point**: `python -m core.acm_main --equip <EQUIP>`
- **Batch runner**: `python scripts/sql_batch_runner.py --equip FD_FAN GAS_TURBINE --tick-minutes 1440`
- **SQL verification**: `python scripts/sql/verify_acm_connection.py`
- **Schema export (SOLE SQL INSPECTION TOOL)**: `python scripts/sql/export_comprehensive_schema.py --output docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md` - **Use this exclusively for:**
  - SQL table schema inspection
  - Table row counts and statistics
  - Sample data queries (top/bottom records)
  - Dashboard panel table validation
  - Schema documentation generation
  - **DO NOT create new scripts** for these purposes; this tool handles all SQL inspection needs
- **Config sync**: `python scripts/sql/populate_acm_config.py`
- **File mode helper**: `scripts/run_file_mode.ps1`

### CORE MODULES (Always check existing implementation before adding new code)
- **Orchestrator**: `core/acm_main.py`
- **I/O Hub**: `core/output_manager.py` (all CSV/PNG/SQL writes)
- **Detectors**: `core/omr.py`, `core/correlation.py`, `core/outliers.py`, `core/fast_features.py`
- **Analytics**: `core/fuse.py` (fusion), `core/regimes.py`, `core/drift.py`, `core/episode_culprits_writer.py`
- **RUL/Forecast**: `core/rul_engine.py`, `core/forecasting.py`
- **Utilities**: `utils/config_dict.py` (config loader), `utils/version.py` (versioning)
- **SQL**: `core/sql_client.py`, `core/smart_coldstart.py`

### CODE SEARCH TIPS
- `rg "ALLOWED_TABLES" core` - Find output table allowlist
- `rg "run_pipeline" core` - Locate pipeline orchestration
- `rg "ACM_" scripts docs core` - Find ACM-prefixed constants
- `rg --files -g "*.sql" scripts/sql` - Find SQL scripts

## Documentation Discipline (STRICT)

### DO NOT CREATE NEW DOCUMENTS UNLESS EXPLICITLY REQUESTED
This is a critical rule to maintain a clean, maintainable knowledge base:

**PROHIBITED WITHOUT EXPLICIT REQUEST**:
- [WRONG] New markdown files for analysis, reports, or summaries
- [WRONG] Auto-generated changelogs or task lists
- [WRONG] Process documents or workflow guides (unless asked)
- [WRONG] "Status report" or "analysis" documents for routine work
- [WRONG] Feature summaries or refactoring documentation
- [WRONG] Temporary guides or quick-reference cards

**ALLOWED/ENCOURAGED**:
- [OK] Update README.md when user explicitly requests documentation
- [OK] Update ACM_SYSTEM_OVERVIEW.md with architectural changes (when explicitly requested)
- [OK] Update SOURCE_CONTROL_PRACTICES.md with new processes (when explicitly requested)
- [OK] Add inline code comments and docstrings (encouraged for clarity)
- [OK] Update existing docs/ files when explicitly requested
- [OK] Create ONLY when user says "create a document", "write a guide", "document this", etc.

**REFERENCE INSTEAD OF CREATING**:
- User asks for analysis? → Reference relevant section of README or SYSTEM_OVERVIEW
- User needs guidance? → Reference docs/SOURCE_CONTROL_PRACTICES.md
- User wants status? → Report based on code/git/test results, not new docs
- User needs summary? → Provide inline in response, don't create document

**EXAMPLE ANTI-PATTERNS**:
- [WRONG] User: "Fix the detector labels" → Agent creates "DETECTOR_LABEL_FIX_REPORT.md"
- [WRONG] User: "What's the architecture?" → Agent creates "ARCHITECTURE_SUMMARY.md"
- [WRONG] User: "Update to v9" → Agent creates "V9_MIGRATION_GUIDE.md"
- [WRONG] Agent auto-creates changelogs, task lists, or work summaries

**PROPER BEHAVIOR**:
- [OK] User: "Fix detector labels" → Fix code, reference README, don't create doc
- [OK] User: "What's the architecture?" → Reference docs/ACM_SYSTEM_OVERVIEW.md
- [OK] User: "Update to v9" → Update code/version, reference existing docs
- [OK] Report status/analysis inline in response

## Config sync discipline
- When `configs/config_table.csv` changes, run `python scripts/sql/populate_acm_config.py` to sync `ACM_Config` in SQL.
- Keep `ConfigDict` dotted-path semantics intact so the populate script remains compatible.

## Troubleshooting & Common Fixes

### Data Loading Issues
**Problem**: Batch runs return NOOP despite data existing in SQL tables.
**Root Cause**: `core/output_manager.py::_load_data_from_sql()` passing wrong parameter to stored procedure.
**Solution**: Ensure stored procedure call uses `@EquipmentName` parameter (not `@EquipID`):
```python
cur.execute(
    "EXEC dbo.usp_ACM_GetHistorianData_TEMP @StartTime=?, @EndTime=?, @EquipmentName=?",
    (start_utc, end_utc, equipment_name)
)
```
**Verify**: Test SP directly: `EXEC usp_ACM_GetHistorianData_TEMP @StartTime='2024-03-01', @EndTime='2024-03-02', @EquipmentName='FD_FAN'`

### SQL Column Mismatch Errors
**Problem**: SQL writes fail with "Cannot insert NULL into column" errors for forecast tables.
**Root Cause**: DataFrame missing required NOT NULL columns.
**Solution**: Always include all required columns when building forecast DataFrames:
- `ACM_HealthForecast_TS`: requires `Method` (NVARCHAR, NOT NULL)
- `ACM_FailureForecast_TS`: requires `Method`, `ThresholdUsed` (FLOAT, NOT NULL)
- `ACM_SensorForecast_TS`: requires `Method` (NVARCHAR, NOT NULL)
- `ACM_RUL_Summary`: requires `Method`, `LastUpdate` (DATETIME2, NOT NULL)

**Check Schema**: `SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='ACM_HealthForecast_TS'`

### Coldstart Minimum Rows
**Problem**: Batch runs succeed but show NOOP with "Insufficient data" warnings.
**Root Cause**: Coldstart mode requires minimum 200 rows by default (configurable via `min_train_samples`).
**Solution**: 
- For 30-minute data cadence: need 100+ hours of data (200 samples / 2 samples/hour = 100 hours)
- For 1-hour cadence: need 200+ hours of data
- Increase batch window size: use 5-10 day windows for sparse data
**Example**: `.\scripts\run_data_range_batches.ps1 -Equipment "FD_FAN" -NumBatches 50 -StartDate "2024-03-01" -EndDate "2024-12-31" -BatchSizeMinutes 7200` (5 days = 120 hours)

### PowerShell Command Syntax
**Problem**: PowerShell multiline commands fail or get syntax errors.
**Correct Patterns**:
```powershell
# Batch runner (use -Parameter syntax, NOT quotes)
.\scripts\run_data_range_batches.ps1 -Equipment "FD_FAN" -NumBatches 50 -StartDate "2024-03-01" -EndDate "2024-12-31" -BatchSizeMinutes 7200

# Manual ACM run (use --arg syntax for Python)
python -m core.acm_main --equip FD_FAN --start-time "2024-03-01T00:00:00" --end-time "2024-03-02T00:00:00"

# SQL query with SELECT-Object piping
sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q "SELECT * FROM ACM_Runs" -W | Select-Object -First 10
```

### Batch Runner Best Practices
- **Date ranges**: Use actual data ranges (check with `SELECT MIN(EntryDateTime), MAX(EntryDateTime) FROM {Equipment}_Data`)
- **Batch sizing**: 5-10 days (7200-14400 minutes) works well for 30-min cadence data
- **Equipment names**: Must match `Equipment.EquipCode` in SQL (e.g., 'FD_FAN', 'GAS_TURBINE')
- **Monitor progress**: Look for "SUCCESS" vs "NOOP" - NOOP means no data in that window
- **Background mode**: Use `-isBackground true` in PowerShell for long runs

### SQL Server Connection Strings
**File**: `configs/sql_connection.ini`
**Format**:
```ini
[acm]
server = localhost\INSTANCENAME
database = ACM
trusted_connection = yes
driver = ODBC Driver 18 for SQL Server
TrustServerCertificate = yes
```
**Test**: `python scripts/sql/verify_acm_connection.py`

### Stored Procedure Debugging
**List all SPs**: `SELECT name FROM sys.procedures WHERE name LIKE '%ACM%'`
**View SP definition**: `SELECT OBJECT_DEFINITION(OBJECT_ID('dbo.usp_ACM_GetHistorianData_TEMP'))`
**Test SP**: `EXEC usp_ACM_GetHistorianData_TEMP @StartTime='2024-03-01', @EndTime='2024-03-02', @EquipmentName='FD_FAN'`
**Recreate SP**: `sqlcmd -S "localhost\INSTANCE" -d ACM -E -i "scripts\sql\51_create_historian_sp_temp.sql"`

### Common SQL Query Patterns
```sql
-- Check table row counts
SELECT COUNT(*) FROM ACM_Scores_Wide WHERE EquipID=1

-- Get recent run logs
SELECT TOP 20 LoggedAt, Level, Message FROM ACM_RunLogs WHERE Message LIKE '%FORECAST%' ORDER BY LoggedAt DESC

-- Find data date ranges
SELECT MIN(EntryDateTime), MAX(EntryDateTime), COUNT(*) FROM FD_FAN_Data

-- Check equipment IDs
SELECT EquipID, EquipCode, EquipName FROM Equipment
```

## CRITICAL ACM Table Schema Knowledge (MUST REMEMBER)

### ACM_RUL Table Columns (v10.0.0+)
**CORRECT Column Names** (NEVER use old names):
- `P10_LowerBound` (NOT `LowerBound`)
- `P50_Median` (median RUL)
- `P90_UpperBound` (NOT `UpperBound`)
- `RUL_Hours` (primary RUL estimate)
- `Confidence` (0-1 confidence score)
- `Method` (forecasting method used)
- `FailureTime` (predicted failure timestamp)
- `NumSimulations` (Monte Carlo runs)
- `TopSensor1`, `TopSensor2`, `TopSensor3` (culprit sensors)

**Example Query**:
```sql
SELECT Method, ROUND(RUL_Hours, 1) AS 'RUL Hours', 
       ROUND(P10_LowerBound, 1) AS 'P10', 
       ROUND(P90_UpperBound, 1) AS 'P90',
       Confidence 
FROM ACM_RUL WHERE EquipID = 1
```

### ACM_Anomaly_Events Time Series Query
**CRITICAL**: For Grafana time_series format, MUST use DATETIME not VARCHAR:
```sql
-- CORRECT: Returns datetime for time column
SELECT DATEADD(HOUR, DATEDIFF(HOUR, 0, StartTime), 0) AS time,
       COUNT(*) AS value, 'Events' AS metric
FROM ACM_Anomaly_Events 
WHERE EquipID = $equipment 
  AND StartTime BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY DATEADD(HOUR, DATEDIFF(HOUR, 0, StartTime), 0)
ORDER BY time ASC
```

**WRONG** (causes "unable to process dataframe" error):
```sql
-- DON'T USE FORMAT() for time series - returns VARCHAR
SELECT FORMAT(StartTime, 'yyyy-MM-dd HH:00') AS time, ...  -- [WRONG] WRONG
```

### ACM_EpisodeDiagnostics and ACM_EpisodeMetrics
**Column Names**:
- `ACM_EpisodeDiagnostics.duration_h` (per-episode duration in hours)
- `ACM_EpisodeMetrics.TotalDurationHours` (aggregated across run)
- Episode DataFrames use `duration_hours` column before SQL write

## Python Best Practices (Prevent f-string/syntax errors)

### String Formatting in Python
**ALWAYS use single quotes inside f-strings when generating SQL/JSON**:
```python
# [OK] CORRECT: Single quotes inside f-string
sql = f"SELECT * FROM Table WHERE Name = '{value}'"
json_str = f'{{"key": "value", "num": {num}}}'

# [WRONG] WRONG: Mixing quotes causes syntax errors
sql = f'SELECT * FROM Table WHERE Name = "{value}"'  # Breaks if value has quotes
```

### Python JSON Operations
```python
import json

# Read JSON safely
with open('file.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Write JSON with proper indentation
with open('file.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

# Escaping in f-strings for SQL queries
query = f"SELECT * FROM Table WHERE Status = 'Active'"  # Use single quotes for SQL strings
```

## PowerShell 5.1 Command Patterns

### Multi-line Commands
```powershell
# Use backtick for line continuation
sqlcmd -S "localhost\INSTANCE" `
       -d ACM `
       -E `
       -Q "SELECT * FROM ACM_Runs"

# Use semicolon for command chaining (NOT &&)
cd C:\path\to\dir; python script.py; echo "Done"

# Pipeline operations
Get-Content log.txt | Select-Object -Last 20
sqlcmd -S "server" -d db -E -Q "SELECT ..." -W | Select-Object -First 10
```

### Parameter Passing
```powershell
# Script parameters use -Name syntax
.\script.ps1 -Equipment "FD_FAN" -StartDate "2024-01-01" -BatchSize 7200

# Python args use --name syntax
python -m core.acm_main --equip FD_FAN --start-time "2024-01-01T00:00:00"
```

## T-SQL Best Practices (Microsoft SQL Server)

### Datetime Handling
```sql
-- Use DATEADD/DATEDIFF for time rounding (not DATE_TRUNC)
SELECT DATEADD(HOUR, DATEDIFF(HOUR, 0, Timestamp), 0) AS HourStart
FROM ACM_HealthTimeline

-- Convert to/from Unix timestamp
SELECT DATEDIFF(SECOND, '1970-01-01', Timestamp) AS UnixTime
SELECT DATEADD(SECOND, @UnixTime, '1970-01-01') AS Timestamp

-- Time range filters for Grafana
WHERE Timestamp BETWEEN $__timeFrom() AND $__timeTo()
```

### Aggregation and Grouping
```sql
-- TOP N (not LIMIT)
SELECT TOP 10 * FROM ACM_RUL ORDER BY RUL_Hours ASC

-- Window functions for running calculations
SELECT Timestamp, HealthIndex,
       AVG(HealthIndex) OVER (ORDER BY Timestamp ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS MA_7
FROM ACM_HealthTimeline

-- COALESCE for null handling
SELECT COALESCE(SUM(TotalEpisodes), 0) AS Total
FROM ACM_EpisodeMetrics WHERE EquipID = 1
```

### Column Casting
```sql
-- Explicit casting for parameters
SELECT CAST(EquipID AS VARCHAR) AS __value,
       COALESCE(EquipName, CONCAT('Equipment-', CAST(EquipID AS VARCHAR))) AS __text
FROM Equipment

-- Rounding with ROUND()
SELECT ROUND(HealthIndex, 1) AS Health,
       ROUND(P10_LowerBound, 2) AS P10
FROM ACM_RUL
```

## Grafana Dashboard Best Practices

### Time Series Panel Configuration
**ALWAYS set spanNulls to disconnect on gaps** (not true/false):
```json
{
  "custom": {
    "spanNulls": 3600000,  // Disconnect if gap > 1 hour (in ms)
    "lineInterpolation": "smooth",
    "showPoints": "auto"
  }
}
```

**Common thresholds**:
- Health/forecast panels: 1 hour (3600000 ms)
- Sensor readings: 30 min (1800000 ms)
- Aggregated metrics: 2 hours (7200000 ms)

### Per-field Min/Max Overrides
```json
{
  "fieldConfig": {
    "overrides": [
      {
        "matcher": {"id": "byName", "options": "value"},
        "properties": [
          {"id": "min", "value": 0},
          {"id": "max", "value": 100},
          {"id": "unit", "value": "percent"}
        ]
      }
    ]
  }
}
```

**When to use**:
- Health scores: min=0, max=100
- Z-scores (OMR, detectors): min=-10, max=10
- Probabilities: min=0, max=100, unit=percent
- Regime labels: min=0, max=(num_regimes-1)

### Default Time Range
**ACM dashboards should default to 5 years** to show full historical trends:
```json
{
  "time": {
    "from": "now-5y",
    "to": "now"
  }
}
```

### SQL Query Guidelines for Grafana
1. **Always add time range filters** for time series:
   ```sql
   WHERE Timestamp BETWEEN $__timeFrom() AND $__timeTo()
   ```

2. **Use proper datetime columns** (not FORMAT strings):
   ```sql
   SELECT Timestamp AS time, Value AS value  -- [OK] CORRECT
   SELECT FORMAT(Timestamp, 'yyyy-MM-dd') AS time  -- [WRONG] WRONG for time_series
   ```

3. **Order by time ASC** for time series (DESC causes rendering issues):
   ```sql
   ORDER BY Timestamp ASC  -- [OK] CORRECT for time series
   ```

4. **Use metric column** for multiple series:
   ```sql
   SELECT Timestamp AS time, HealthIndex AS value, 'Health' AS metric
   ```

## Common Mistakes to AVOID

### 1. SQL Column Name Mistakes
- [WRONG] `ACM_RUL.LowerBound` → [OK] `ACM_RUL.P10_LowerBound`
- [WRONG] `ACM_RUL.UpperBound` → [OK] `ACM_RUL.P90_UpperBound`
- [WRONG] `ACM_RUL.CreatedAt` in ORDER BY → [OK] Use `RUL_Hours` or `P50_Median`

### 2. Time Series Query Mistakes
- [WRONG] `FORMAT(time, 'yyyy-MM-dd')` → [OK] Return raw DATETIME
- [WRONG] `ORDER BY time DESC` → [OK] `ORDER BY time ASC`
- [WRONG] Missing time range filter → [OK] `WHERE ... BETWEEN $__timeFrom() AND $__timeTo()`

### 3. Python String Formatting
- [WRONG] `f'.."{variable}"...'` (quote mismatch) → [OK] `f"...'{variable}'..."`
- [WRONG] Unescaped quotes in JSON → [OK] Use `''` for single quotes in f-strings

### 4. PowerShell Command Mistakes
- [WRONG] `command1 && command2` (bash syntax) → [OK] `command1; command2`
- [WRONG] `tail -n 20` → [OK] `Select-Object -Last 20`
- [WRONG] Using --parameters for .ps1 scripts → [OK] Use -Parameters

### 5. Grafana spanNulls Mistakes
- [WRONG] `"spanNulls": true` (connects all gaps) → [OK] `"spanNulls": 3600000` (threshold in ms)
- [WRONG] `"spanNulls": false` (breaks all lines) → [OK] Use threshold value

### 6. RUL Prediction Mistakes (CRITICAL)
- [WRONG] `ORDER BY RUL_Hours ASC` → [OK] `ORDER BY CreatedAt DESC` (use most recent, not worst-case)
- [WRONG] Showing RUL with NULL confidence bounds → [OK] Filter: `WHERE (P10_LowerBound IS NOT NULL OR P50_Median IS NOT NULL)`
- [WRONG] Imminent failure (<24h) when Health > 80% → [OK] RUL predictions must correlate with health state
- [WRONG] Single moderate defect (Z < 5) justifies failure → [OK] Require multiple confirming signals

## RUL Prediction Validation Rules (MUST FOLLOW)

### CRITICAL: RUL Prediction Reliability Requirements
**RUL predictions showing imminent failure (<24 hours) MUST be validated against multiple signals:**

1. **Confidence Bounds Check**:
   - P10_LowerBound, P50_Median, P90_UpperBound must NOT be NULL
   - NULL bounds indicate unreliable prediction from insufficient data
   - ALWAYS filter: `WHERE (P10_LowerBound IS NOT NULL OR P50_Median IS NOT NULL)`

2. **Health State Correlation**:
   - Imminent failure (<24h) requires Health < 50% OR HealthZone = 'CRITICAL'
   - If Health > 80% (GOOD zone), RUL < 24h is likely FALSE POSITIVE
   - Cross-validate: `JOIN ACM_HealthTimeline` to check current health

3. **Active Defect Validation**:
   - Single moderate defect (CurrentZ < 5) does NOT justify imminent failure
   - Require MULTIPLE ActiveDefect=1 OR at least one detector with CurrentZ > 8
   - Check: `SELECT COUNT(*) FROM ACM_SensorDefects WHERE ActiveDefect=1`

4. **Fused Score Validation**:
   - Imminent failure requires FusedZ > 2.0 (indicating anomaly)
   - Negative FusedZ indicates NORMAL operation, contradicts failure prediction
   - Validate: Latest FusedZ from `ACM_HealthTimeline`

5. **Query Ordering - Use Most Recent, Not Worst-Case**:
   - ALWAYS: `ORDER BY CreatedAt DESC` (most recent prediction)
   - NEVER: `ORDER BY RUL_Hours ASC` (selects minimum/worst-case from all history)
   - Reasoning: Batch runs create multiple predictions; want latest assessment, not historical minimum

### Example Valid RUL Query (Grafana)
```sql
SELECT TOP 1 
    Method, 
    ROUND(RUL_Hours, 1) AS 'RUL (h)', 
    ROUND(P10_LowerBound, 1) AS 'P10',
    ROUND(P50_Median, 1) AS 'P50',
    ROUND(P90_UpperBound, 1) AS 'P90',
    ROUND(Confidence, 3) AS 'Confidence', 
    CASE 
        WHEN RUL_Hours > 168 THEN 'Healthy' 
        WHEN RUL_Hours > 72 THEN 'Caution' 
        WHEN RUL_Hours > 24 THEN 'Warning' 
        ELSE 'Critical' 
    END AS 'Status'
FROM ACM_RUL 
WHERE EquipID = $equipment 
    AND (P10_LowerBound IS NOT NULL OR P50_Median IS NOT NULL)  -- Filter invalid predictions
ORDER BY CreatedAt DESC  -- Most recent, not worst-case
```

### RUL Data Quality Indicators
- **GOOD**: P10/P50/P90 populated, Confidence > 0.7, TopSensor1-3 identified, NumSimulations > 0
- **QUESTIONABLE**: P10/P50/P90 NULL, Confidence < 0.5, TopSensors NULL
- **INVALID**: All confidence bounds NULL, Confidence = 0, Method shows error

## Learning from Errors

When encountering errors:
1. **Check table schema**: `SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='TableName'`
2. **Verify column exists** before using in queries
3. **Test queries in sqlcmd** before adding to dashboard
4. **Check data types**: `SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME='ColName'`
5. **Validate RUL predictions** against health state before trusting imminent failure alerts
6. **Update copilot-instructions.md** with new schema learnings immediately
