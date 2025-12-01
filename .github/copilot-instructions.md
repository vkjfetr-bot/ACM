# ACM Copilot Guardrails (current)

- **Mission**: keep the ACM pipeline healthy (CSV/SQL ingest, detector fusion, analytics outputs under `artifacts/{EQUIP}/run_*`). Primary entrypoint: `python -m core.acm_main --equip FD_FAN` (`run_pipeline()` in `core/acm_main.py`).
- **Modes**: file-mode reads `data/*.csv`; SQL-mode uses `configs/sql_connection.ini` via `core/sql_client.SQLClient`. File-mode must stay working before SQL-path changes ship.
- **Batch mode**: run `python scripts/sql_batch_runner.py --equip FD_FAN GAS_TURBINE --tick-minutes 1440 --max-workers 2 --start-from-beginning` (adjust args as needed). This command sets `ACM_BATCH_MODE`/`ACM_BATCH_NUM`; do not set them in code.
- **No emojis ever** in code, comments, tests, or generated content.

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

## Discoverability quick-links
- **System overview**: `docs/ACM_SYSTEM_OVERVIEW.md`; analytics flow: `docs/Analytics Backbone.md`.
- **Coldstart**: `docs/COLDSTART_MODE.md`; **OMR**: `docs/OMR_DETECTOR.md`; **Batch SQL audit**: `docs/BATCH_MODE_SQL_AUDIT.md`.
- **Schema reference**: `docs/sql/SQL_SCHEMA_REFERENCE.md`; **Grafana**: `grafana_dashboards/README.md` + dashboards under `grafana_dashboards/*.json`.
- **Run helpers**: `python scripts/sql_batch_runner.py ...`, `scripts/run_file_mode.ps1`, `scripts/sql/verify_acm_connection.py`.
- **Core modules**: entry `core/acm_main.py`; writes `core/output_manager.py`; detectors/fusion/regimes/drift under `core/` (omr.py, correlation.py, outliers.py, fuse.py, regimes.py, drift.py); episodes `core/episode_culprits_writer.py`; run metadata `core/run_metadata_writer.py`; config loader `utils/config_dict.py`.
- **Search tips**: `rg "ALLOWED_TABLES" core`, `rg "run_pipeline" core`, `rg "ACM_" scripts docs core`, `rg --files -g "*.sql" scripts/sql`.

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
