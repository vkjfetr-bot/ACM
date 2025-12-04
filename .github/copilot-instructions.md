# ACM Copilot Guardrails (v9.0.0)

## PRIMARY DIRECTIVES
- **Mission**: Keep the ACM pipeline healthy and production-ready
- **Knowledge Base**: ALWAYS consult and reference these canonical documents:
  - `README.md` - Primary product overview, features, running ACM
  - `docs/ACM_SYSTEM_OVERVIEW.md` - Complete architectural walkthrough, module map, workflows
  - `docs/SOURCE_CONTROL_PRACTICES.md` - Git workflow, branching, versioning, release management
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
- **No emojis ever** in code, comments, tests, or generated content.
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
- Schema snapshot: `python scripts/sql/export_schema_doc.py --output docs/sql/SQL_SCHEMA_REFERENCE.md` regenerates the latest table definitions so SQL changes stay aligned with production.
- File-mode helper: `scripts/run_file_mode.ps1` wraps a baseline local run.

## Testing
- Targeted suites: `pytest tests/test_fast_features.py`, `pytest tests/test_dual_write.py`, `pytest tests/test_progress_tracking.py`. Respect existing skips/markers (Polars, SQL Server).

## Data & artifacts
- Sample CSVs under `data/` have datetime index + sensor columns; keep names aligned with `configs/config_table.csv`.
- `artifacts/` and `backups/` stay gitignored; do not add tracked artifact folders.

## Documentation policy
- README is authoritative. Do **not** auto-create new docs/changelogs for routine changes; only update docs when explicitly requested.

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
- **Schema Reference**: `docs/sql/SQL_SCHEMA_REFERENCE.md` (table definitions)
- **Grafana**: `grafana_dashboards/README.md` + dashboards under `grafana_dashboards/*.json`
- **Analytics**: `docs/Analytics Backbone.md` (detector fusion, episodes, regimes)

### OPERATIONAL COMMANDS & SCRIPTS
- **Entry point**: `python -m core.acm_main --equip <EQUIP>`
- **Batch runner**: `python scripts/sql_batch_runner.py --equip FD_FAN GAS_TURBINE --tick-minutes 1440`
- **SQL verification**: `python scripts/sql/verify_acm_connection.py`
- **Schema export**: `python scripts/sql/export_schema_doc.py --output docs/sql/SQL_SCHEMA_REFERENCE.md`
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
- ❌ New markdown files for analysis, reports, or summaries
- ❌ Auto-generated changelogs or task lists
- ❌ Process documents or workflow guides (unless asked)
- ❌ "Status report" or "analysis" documents for routine work
- ❌ Feature summaries or refactoring documentation
- ❌ Temporary guides or quick-reference cards

**ALLOWED/ENCOURAGED**:
- ✅ Update README.md when user explicitly requests documentation
- ✅ Update ACM_SYSTEM_OVERVIEW.md with architectural changes (when explicitly requested)
- ✅ Update SOURCE_CONTROL_PRACTICES.md with new processes (when explicitly requested)
- ✅ Add inline code comments and docstrings (encouraged for clarity)
- ✅ Update existing docs/ files when explicitly requested
- ✅ Create ONLY when user says "create a document", "write a guide", "document this", etc.

**REFERENCE INSTEAD OF CREATING**:
- User asks for analysis? → Reference relevant section of README or SYSTEM_OVERVIEW
- User needs guidance? → Reference docs/SOURCE_CONTROL_PRACTICES.md
- User wants status? → Report based on code/git/test results, not new docs
- User needs summary? → Provide inline in response, don't create document

**EXAMPLE ANTI-PATTERNS**:
- ❌ User: "Fix the detector labels" → Agent creates "DETECTOR_LABEL_FIX_REPORT.md"
- ❌ User: "What's the architecture?" → Agent creates "ARCHITECTURE_SUMMARY.md"
- ❌ User: "Update to v9" → Agent creates "V9_MIGRATION_GUIDE.md"
- ❌ Agent auto-creates changelogs, task lists, or work summaries

**PROPER BEHAVIOR**:
- ✅ User: "Fix detector labels" → Fix code, reference README, don't create doc
- ✅ User: "What's the architecture?" → Reference docs/ACM_SYSTEM_OVERVIEW.md
- ✅ User: "Update to v9" → Update code/version, reference existing docs
- ✅ Report status/analysis inline in response

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
