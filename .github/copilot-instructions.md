# ACM - Automated Condition Monitoring (v10.3.0)

ACM is a predictive maintenance and equipment health monitoring system. It ingests sensor data from industrial equipment (FD_FAN, GAS_TURBINE, etc.) via SQL Server, runs anomaly detection algorithms, calculates health scores, and forecasts Remaining Useful Life (RUL). Results are visualized through Grafana dashboards for operations teams.

**Key Features**: Multi-detector fusion (AR1, PCA, IForest, GMM, OMR), regime detection, episode diagnostics, RUL forecasting with Monte Carlo simulations, SQL-only persistence, full observability stack (OpenTelemetry traces/metrics, Loki logs, Pyroscope profiling).

---

## Tech Stack

### Backend
- **Python 3.11** - Core runtime
- **pandas/NumPy** - Vectorized data processing
- **scikit-learn** - ML detectors (PCA, IForest, GMM)
- **Optional**: Polars (fast_features.py), Rust bridge (rust_bridge/)

### Database
- **Microsoft SQL Server** - Primary data store (historian data, ACM results)
- **pyodbc** - SQL connectivity via `core/sql_client.py`
- **T-SQL** - Stored procedures, queries (NEVER use generic SQL syntax)

### Visualization
- **Grafana** - Dashboards in `grafana_dashboards/*.json`
- **MS SQL datasource** - Direct SQL queries with `$__timeFrom()`, `$__timeTo()` macros

### Observability Stack (v10.3.0)
- **OpenTelemetry** - Distributed tracing (Tempo) and metrics (Prometheus)
- **Loki** - Structured log aggregation via Grafana Alloy
- **Pyroscope** - Continuous profiling for performance analysis
- **Unified API** - `core/observability.py` provides Console, Span, Metrics classes
- **Grafana Dashboards** - `acm_observability.json`, `acm_performance_monitor.json`
- **Install scripts** - `install/observability/` for Docker Compose stack

### Configuration
- **configs/config_table.csv** - Equipment-specific settings (cascading: `*` global, then equipment rows)
- **configs/sql_connection.ini** - SQL Server credentials (gitignored, local-only)

---

## Project Structure

```
ACM/
+-- .github/              # Copilot instructions (this file)
+-- configs/              # config_table.csv, sql_connection.ini
+-- core/                 # Main codebase
|   +-- acm_main.py       # Pipeline orchestrator (entry point)
|   +-- output_manager.py # All CSV/PNG/SQL writes (ALLOWED_TABLES)
|   +-- sql_client.py     # SQL Server connectivity
|   +-- observability.py  # Unified observability (Console, Span, Metrics, log_timer)
|   +-- omr.py            # Overall Model Residual detector
|   +-- correlation.py    # Correlation-based detector
|   +-- outliers.py       # Statistical outlier detection
|   +-- fast_features.py  # Feature engineering (pandas/Polars)
|   +-- fuse.py           # Multi-detector fusion
|   +-- regimes.py        # Operating regime detection
|   +-- drift.py          # Concept drift monitoring
|   +-- forecast_engine.py # RUL and health forecasting
|   +-- model_persistence.py  # SQL-only model storage
|   +-- episode_culprits_writer.py  # Episode diagnostics
+-- scripts/              # Operational scripts
|   +-- sql_batch_runner.py   # Batch processing
|   +-- sql/              # SQL utilities, schema export
+-- docs/                 # All documentation (NEVER create docs in root)
|   +-- archive/          # Historical/archived docs
|   +-- sql/              # COMPREHENSIVE_SCHEMA_REFERENCE.md
+-- grafana_dashboards/   # Grafana JSON dashboards
+-- tests/                # pytest test suites
+-- utils/                # config_dict.py, version.py
+-- artifacts/            # gitignored - runtime outputs
+-- data/                 # gitignored - sample CSVs for file-mode
+-- logs/                 # gitignored - runtime logs
```

---

## Coding Guidelines

### Python Style
- Python 3.11, ~100-char lines, type hints always
- Vectorized pandas/NumPy preferred over loops
- Use existing lint/type tooling (`ruff`, `mypy`)
- No emojis in code, comments, tests, or output

### SQL/T-SQL Rules
- ALWAYS use Microsoft SQL Server T-SQL syntax
- Use `sqlcmd -S "server\instance" -d database -E -Q "..."` for queries
- Use T-SQL functions: `DATEADD`, `DATEDIFF`, `TOP`, `CAST`, `ROUND`, `COALESCE`
- NEVER use: `LIMIT` (use `TOP`), `DATE_TRUNC` (use `DATEADD/DATEDIFF`), `FORMAT()` for time series
- **NEVER use reserved words as aliases**: `End`, `RowCount`, `Count`, `Date`, `Time`, `Order`, `Group`
- **Safe aliases**: `EndTimeStr`, `TotalRows`, `TotalCount`, `DateValue`, `TimeValue`, `OrderNum`, `GroupName`

### PowerShell 5.1 Rules
- Use `;` for command chaining (NOT `&&`)
- Use `Select-Object -Last 20` (NOT `tail -n 20`)
- Script params: `-Parameter value`; Python args: `--arg value`

### CRITICAL: Python Command-Line Execution in PowerShell (Windows)
**NEVER use inline Python with complex strings in PowerShell `python -c "..."` commands:**

❌ **FORBIDDEN PATTERNS:**
```powershell
# NEVER: f-strings with nested quotes break PowerShell parsing
python -c "import x; conn = x.connect(f'DRIVER={{{var}}}')"

# NEVER: Multi-line strings in -c flag
python -c "
import module
code here
"

# NEVER: Dictionary/config access in f-strings
python -c "s = config['acm']; print(f'x={s[\"key\"]}')"
```

✅ **ALWAYS USE THESE PATTERNS:**
```powershell
# ✓ Create a temporary .py script file instead
$code = @'
import module
# Multi-line code here
'@
$code | Out-File -Encoding UTF8 temp_script.py
python temp_script.py
Remove-Item temp_script.py

# ✓ Use scripts/ folder for any non-trivial Python
# Create scripts/verify_something.py, then:
python scripts/verify_something.py

# ✓ For simple one-liners, avoid f-strings entirely
python -c "import sys; print(sys.version)"
```

**WHY THIS MATTERS:**
- PowerShell interprets `{`, `}`, `$`, `"`, `'`, `\` as special characters
- F-strings with nested quotes create unescapable parsing conflicts
- Multi-line strings in `-c` fail due to PowerShell's line continuation rules
- **Solution: Always create a .py file for anything beyond trivial one-liners**

### Logging & Observability Rules (v10.3.0)
- **ALWAYS use `Console` class** from `core.observability` for all logging
- **NEVER use `print()` or legacy loggers** (`utils/logger.py`, `utils/acm_logger.py` are deleted)
- Console methods:
  - `Console.info(msg, **kwargs)` - General info (goes to Loki)
  - `Console.warn(msg, **kwargs)` - Warnings (goes to Loki)
  - `Console.error(msg, **kwargs)` - Errors (goes to Loki)
  - `Console.ok(msg, **kwargs)` - Success messages (goes to Loki)
  - `Console.status(msg)` - Console-only output (NO Loki)
  - `Console.header(title, char="=")` - Section headers (NO Loki)
  - `Console.section(title)` - Lighter separators (NO Loki)
- **Separator lines MUST use `Console.status/header/section`** to avoid leaking to Loki
- **Timer output** uses `Console.section/status` (console-only, not logged)

### Grafana Dashboard Rules
- Time series: Return raw `DATETIME` columns (not `FORMAT()` strings)
- Order: `ORDER BY time ASC` for time series
- Time filter: `WHERE Timestamp BETWEEN $__timeFrom() AND $__timeTo()`
- spanNulls: Use threshold in ms (e.g., `3600000`) not `true/false`

---

## Resources and Commands

### Primary Entry Points
```powershell
# Single equipment run
python -m core.acm_main --equip GAS_TURBINE

# Batch processing
python scripts/sql_batch_runner.py --equip FD_FAN GAS_TURBINE --tick-minutes 1440 --max-workers 2 --start-from-beginning
```

### SQL Tools
```powershell
# Verify SQL connection
python scripts/sql/verify_acm_connection.py

# Export schema (SOLE AUTHORITATIVE TOOL for SQL inspection)
python scripts/sql/export_comprehensive_schema.py --output docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md

# Sync config to SQL
python scripts/sql/populate_acm_config.py
```

### Observability Stack
```powershell
# Start observability stack (Prometheus, Loki, Tempo, Pyroscope, Alloy)
cd install/observability; docker compose up -d

# Verify stack is running
Invoke-RestMethod -Uri "http://localhost:3200/api/search?limit=1"  # Tempo
Invoke-RestMethod -Uri "http://localhost:3100/ready"                # Loki
Invoke-RestMethod -Uri "http://localhost:9090/-/healthy"            # Prometheus

# View traces in Grafana
# Open http://localhost:3000, go to Explore > Tempo > Search for service.name=acm-pipeline
```

### Testing
```powershell
pytest tests/test_fast_features.py
pytest tests/test_dual_write.py
pytest tests/test_progress_tracking.py
pytest tests/test_observability.py
```

---

## Critical Rules (MUST FOLLOW)

### NON-VIOLATABLE TESTING RULES
**CRITICAL: These rules are ABSOLUTE and may NEVER be violated:**

1. **ONLY WAY TO TEST ACM IS BATCH MODE**
   - The ONLY way to test ACM functionality is to run it in batch mode with `python scripts/sql_batch_runner.py`
   - NEVER create single-use diagnostic scripts to test ACM functionality
   - Run full batches to test the entire data flow end-to-end

2. **DIAGNOSIS IS THROUGH LOGGING ONLY**
   - Problems are diagnosed by reading ACM_RunLogs in SQL or console output
   - Add proper logging at the point where issues occur
   - NEVER create standalone scripts to "check" or "validate" ACM behavior

3. **ALLOWED TESTING SCRIPTS**
   - `python -c "import module; print('OK')"` - Test imports only
   - `sqlcmd -S ... -Q "SELECT ..."` - Read data from SQL tables
   - That's it. No other diagnostic scripts allowed.

4. **FORECASTING MUST ALWAYS WORK**
   - Forecasting must work in ALL modes: batch, single-run, streaming
   - NEVER skip forecasting because "data is missing" - populate the data
   - All required tables (ACM_SensorNormalized_TS, ACM_RegimeTimeline, etc.) MUST be populated

### Documentation Policy
- **NEVER** create new markdown/HTML files unless explicitly requested
- **ALL** documentation goes in `docs/` folder, archived docs in `docs/archive/`
- **NEVER** create files in root directory (keep root clean)
- Reference existing docs instead of creating new ones
- Report status/analysis inline in responses, not as new documents

### Root Directory Cleanliness
- Root contains ONLY: `README.md`, `pyproject.toml`, `.gitignore`, and folders
- NEVER create: markdown files, HTML files, log files, temp files, equipment folders in root
- Artifacts go to `artifacts/` (gitignored), logs to `logs/` (gitignored)

### SQL-Only Mode
- ACM uses SQL-mode exclusively via `core/sql_client.SQLClient`
- DO NOT USE FILE MODE EVER
- All writes go through `core/output_manager.OutputManager` (respects `ALLOWED_TABLES`)
- Model persistence uses SQL `ModelRegistry` table (not filesystem)

### Time Handling
- Timestamps are local-naive (no UTC conversions)
- Use `_to_naive*` helpers in output_manager.py and acm_main.py

### Version Management
- Version stored in `utils/version.py` (currently v10.3.0)
- Increment only when explicitly requested
- Follow semantic versioning (MAJOR.MINOR.PATCH)

### Config Sync Discipline
- When `configs/config_table.csv` changes, run `python scripts/sql/populate_acm_config.py` to sync `ACM_Config` in SQL
- Keep `ConfigDict` dotted-path semantics intact

---

## Key Knowledge Base Documents

| Document | Purpose |
|----------|---------||
| `README.md` | Product overview, setup, running ACM |
| `docs/ACM_SYSTEM_OVERVIEW.md` | Architecture, module map, data flow |
| `docs/OBSERVABILITY.md` | Observability stack (traces, metrics, logs, profiling) |
| `docs/SOURCE_CONTROL_PRACTICES.md` | Git workflow, branching, releases |
| `docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md` | Authoritative SQL table definitions |
| `utils/version.py` | Current version and release notes |
| `docs/OMR_DETECTOR.md` | Overall Model Residual detector |
| `docs/COLDSTART_MODE.md` | Cold-start strategy for sparse data |
| `docs/EQUIPMENT_IMPORT_PROCEDURE.md` | How to add new equipment |
| `install/observability/README.md` | Observability stack installation |

---

## Source Control

- Branch for all work: `feature/<topic>` or `fix/<topic>`
- Avoid pushing directly to `main`
- Rebase onto `main` before merging PRs
- Clear, imperative commits (e.g., "Add batch-mode env guards")
- Run tests before merging; never merge with failing checks
- Never commit: artifacts, logs, credentials (respect `.gitignore`)

---

## Active Detectors (v10.2.0)

| Detector | Column Prefix | Purpose |
|----------|---------------|---------|
| AR1 | `ar1_z` | Autoregressive residual |
| PCA-SPE | `pca_spe_z` | Squared prediction error |
| PCA-T2 | `pca_t2_z` | Hotelling T-squared |
| IForest | `iforest_z` | Isolation forest anomaly |
| GMM | `gmm_z` | Gaussian mixture model |
| OMR | `omr_z` | Overall model residual |

**Removed Detectors**:
- `mhal_z` (Mahalanobis): Removed - redundant with PCA-T2
- `river_hst_z` (River HST): Removed - not implemented, streaming detector not needed

---

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

### SQL Column Mismatch Errors
**Problem**: SQL writes fail with "Cannot insert NULL into column" errors.
**Solution**: Always include all required NOT NULL columns:
- `ACM_HealthForecast_TS`: requires `Method` (NVARCHAR)
- `ACM_FailureForecast_TS`: requires `Method`, `ThresholdUsed` (FLOAT)
- `ACM_SensorForecast_TS`: requires `Method` (NVARCHAR)

### Coldstart Minimum Rows
**Problem**: NOOP with "Insufficient data" warnings.
**Root Cause**: Coldstart requires 200+ rows by default.
**Solution**: Use 5-10 day batch windows for sparse data.

### SQL Server Connection
**File**: `configs/sql_connection.ini`
```ini
[acm]
server = localhost\INSTANCENAME
database = ACM
trusted_connection = yes
driver = ODBC Driver 18 for SQL Server
TrustServerCertificate = yes
```

### Common SQL Query Patterns
```sql
-- Check table row counts
SELECT COUNT(*) FROM ACM_Scores_Wide WHERE EquipID=1

-- Get recent run logs
SELECT TOP 20 LoggedAt, Level, Message FROM ACM_RunLogs ORDER BY LoggedAt DESC

-- Find data date ranges
SELECT MIN(EntryDateTime), MAX(EntryDateTime), COUNT(*) FROM FD_FAN_Data

-- Check equipment IDs
SELECT EquipID, EquipCode, EquipName FROM Equipment
```

---

## CRITICAL ACM Table Schema Knowledge (MUST REMEMBER)

### ACM_RUL Table Columns (v10.0.0+)
**CORRECT Column Names** (NEVER use old names):
- `P10_LowerBound` (NOT `LowerBound`)
- `P50_Median` (median RUL)
- `P90_UpperBound` (NOT `UpperBound`)
- `RUL_Hours`, `Confidence`, `Method`, `FailureTime`
- `TopSensor1`, `TopSensor2`, `TopSensor3` (culprit sensors)

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

### ACM_EpisodeDiagnostics and ACM_EpisodeMetrics
- `ACM_EpisodeDiagnostics.duration_h` (per-episode duration in hours)
- `ACM_EpisodeMetrics.TotalDurationHours` (aggregated across run)

---

## Python Best Practices

### String Formatting
```python
# CORRECT: Single quotes inside f-string for SQL
sql = f"SELECT * FROM Table WHERE Name = '{value}'"

# WRONG: Mixing quotes causes errors
sql = f'SELECT * FROM Table WHERE Name = "{value}"'
```

---

## PowerShell 5.1 Command Patterns

```powershell
# Use backtick for line continuation
sqlcmd -S "localhost\INSTANCE" `
       -d ACM `
       -E `
       -Q "SELECT * FROM ACM_Runs"

# Use semicolon for command chaining (NOT &&)
cd C:\path\to\dir; python script.py; echo "Done"

# Script parameters use -Name syntax
.\script.ps1 -Equipment "FD_FAN" -StartDate "2024-01-01"

# Python args use --name syntax
python -m core.acm_main --equip FD_FAN --start-time "2024-01-01T00:00:00"
```

---

## T-SQL Best Practices (Microsoft SQL Server)

### Datetime Handling
```sql
-- Use DATEADD/DATEDIFF for time rounding (not DATE_TRUNC)
SELECT DATEADD(HOUR, DATEDIFF(HOUR, 0, Timestamp), 0) AS HourStart
FROM ACM_HealthTimeline

-- Time range filters for Grafana
WHERE Timestamp BETWEEN $__timeFrom() AND $__timeTo()
```

### Aggregation
```sql
-- TOP N (not LIMIT)
SELECT TOP 10 * FROM ACM_RUL ORDER BY RUL_Hours ASC

-- COALESCE for null handling
SELECT COALESCE(SUM(TotalEpisodes), 0) AS Total
FROM ACM_EpisodeMetrics WHERE EquipID = 1
```

---

## Grafana Dashboard Best Practices

### Time Series Panel Configuration
**ALWAYS set spanNulls to disconnect on gaps** (threshold in ms, not true/false):
```json
{
  "custom": {
    "spanNulls": 3600000,  // Disconnect if gap > 1 hour
    "lineInterpolation": "smooth"
  }
}
```

### SQL Query Guidelines for Grafana
1. **Always add time range filters**: `WHERE Timestamp BETWEEN $__timeFrom() AND $__timeTo()`
2. **Use proper datetime columns** (not FORMAT strings)
3. **Order by time ASC** for time series (DESC causes rendering issues)
4. **Use metric column** for multiple series

### Default Time Range
ACM dashboards should default to 5 years: `"from": "now-5y"`

---

## Common Mistakes to AVOID

| Category | Wrong | Correct |
|----------|-------|---------|
| SQL columns | `ACM_RUL.LowerBound` | `ACM_RUL.P10_LowerBound` |
| SQL columns | `ACM_RUL.UpperBound` | `ACM_RUL.P90_UpperBound` |
| SQL reserved words | `AS End` | `AS EndTimeStr` (avoid reserved words) |
| SQL reserved words | `AS RowCount` | `AS TotalRows` (RowCount conflicts) |
| SQL column names | `StartTime` on ACM_Runs | `StartedAt` (check schema first!) |
| SQL aliases | `COUNT(*) as RowCount` | `COUNT(*) as TotalRows` |
| Time series | `FORMAT(time, 'yyyy-MM-dd')` | Return raw `DATETIME` |
| Time series | `ORDER BY time DESC` | `ORDER BY time ASC` |
| PowerShell | `command1 && command2` | `command1; command2` |
| PowerShell | `tail -n 20` | `Select-Object -Last 20` |
| PowerShell | `python -c "f'{var}'"` | Create `.py` file instead |
| Python inline | Multi-line in `-c` flag | **ALWAYS create script file** |
| Python inline | F-strings with `{}` or quotes | **ALWAYS create script file** |
| Grafana | `"spanNulls": true` | `"spanNulls": 3600000` |
| RUL queries | `ORDER BY RUL_Hours ASC` | `ORDER BY CreatedAt DESC` |

---

## RUL Prediction Validation Rules (MUST FOLLOW)

### CRITICAL: RUL Prediction Reliability Requirements
**RUL predictions showing imminent failure (<24 hours) MUST be validated:**

1. **Confidence Bounds Check**:
   - P10_LowerBound, P50_Median, P90_UpperBound must NOT be NULL
   - ALWAYS filter: `WHERE (P10_LowerBound IS NOT NULL OR P50_Median IS NOT NULL)`

2. **Health State Correlation**:
   - Imminent failure (<24h) requires Health < 50% OR HealthZone = 'CRITICAL'
   - If Health > 80%, RUL < 24h is likely FALSE POSITIVE

3. **Active Defect Validation**:
   - Single moderate defect (CurrentZ < 5) does NOT justify imminent failure
   - Require MULTIPLE ActiveDefect=1 OR at least one detector with CurrentZ > 8

4. **Query Ordering - Use Most Recent, Not Worst-Case**:
   - ALWAYS: `ORDER BY CreatedAt DESC` (most recent prediction)
   - NEVER: `ORDER BY RUL_Hours ASC` (selects worst-case from all history)

### Example Valid RUL Query (Grafana)
```sql
SELECT TOP 1
    Method,
    ROUND(RUL_Hours, 1) AS 'RUL (h)',
    ROUND(P10_LowerBound, 1) AS 'P10',
    ROUND(P50_Median, 1) AS 'P50',
    ROUND(P90_UpperBound, 1) AS 'P90',
    ROUND(Confidence, 3) AS 'Confidence'
FROM ACM_RUL
WHERE EquipID = $equipment
    AND (P10_LowerBound IS NOT NULL OR P50_Median IS NOT NULL)
ORDER BY CreatedAt DESC
```

---

## Learning from Errors

When encountering errors:
1. **Check table schema**: `SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='TableName'`
2. **Verify column exists** before using in queries
3. **Test queries in sqlcmd** before adding to dashboard
4. **Validate RUL predictions** against health state
5. **Update this file** with new schema learnings immediately

---

## Core Contracts (Internal Reference)

- **Config**: `utils/config_dict.ConfigDict` loads cascading `configs/config_table.csv`
- **Output manager**: all writes go through `core/output_manager.OutputManager`
- **Time policy**: timestamps are local-naive; use `_to_naive*` helpers
- **Performance**: `core/fast_features.py` supports pandas + optional Polars
- **Rust bridge**: optional accelerator in `rust_bridge/`; Python path remains primary
