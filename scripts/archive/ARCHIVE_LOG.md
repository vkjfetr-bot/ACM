# Scripts Archive Log

**Archived:** 2025-12-16 (v10.2.0)
**Reason:** Single-purpose analysis, check, debug, and validation scripts archived to reduce clutter. These were one-time development/debugging aids that are no longer needed for production operation.

## Active Scripts (Not Archived)

The following scripts remain in the main `scripts/` directory:

| Script | Purpose |
|--------|---------|
| `sql_batch_runner.py` | Main batch orchestration for continuous SQL-mode operation |
| `sql/export_comprehensive_schema.py` | **Authoritative** schema documentation generator |
| `sql/populate_acm_config.py` | Sync config_table.csv to SQL ACM_Config |
| `sql/verify_acm_connection.py` | Test SQL connectivity |
| `run_*.ps1` | PowerShell batch runner wrappers |
| `update_grafana_dashboards.ps1` | Bulk dashboard query updates |
| `wfa_bulk_load.ps1` | WFA equipment data loader |

## Archived Scripts (63 files)

### Analysis Scripts
- `analyze_batch_results.py`
- `analyze_charts.py`
- `analyze_dashboard_schema.py`
- `analyze_latest_run.py`
- `analyze_latest_run_tables.py`
- `analyze_run_failures.py`
- `analyze_run_logs.py`

### Check/Debug Scripts
- `check_available_equipment.py`
- `check_batch_status.py`
- `check_continuous_learning.py`
- `check_dashboard_data.py`
- `check_data.py`
- `check_data_gaps.py`
- `check_drift_values.py`
- `check_episode_metrics.py`
- `check_forecast_panels.py`
- `check_forecast_status.py`
- `check_forecast_table.py`
- `check_full_log.py`
- `check_gas_turbine_gaps.py`
- `check_historian_data.py`
- `check_omr_size.py`
- `check_runs.py`
- `check_schema.py`
- `check_sensor_names.py`
- `debug_check_health_timeline.py`
- `debug_env_schema.py`
- `debug_runs_simple.py`
- `debug_runs_table.py`
- `debug_sql_status.py`
- `debug_timer_table.py`

### Validation Scripts
- `validate_adaptive_thresholds.py`
- `validate_all_tables.py`
- `validate_batch_continuity.py`
- `validate_dashboard_queries.py`
- `validate_fixes.py`
- `validate_refactor.py`
- `validate_sql_only_mode.py`
- `verify_all_table_writes.py`
- `verify_forecast_v10.py`
- `verify_migrations.py`
- `verify_new_tables.py`

### One-Time Migration/Fix Scripts
- `apply_schema_fixes.py`
- `chunk_replay.py`
- `evaluate_rul_backtest.py`
- `generate_corrected_dashboard.py`
- `init_timer_table.py`
- `monitor_adaptive_thresholds.py`
- `monitor_progress.py`
- `phase2_remove_csv.py`
- `phase5_remove_schema_helper.py`
- `quick_check.py`
- `quick_check_runs.py`
- `quick_forecast_check.py`
- `remove_chart_code.py`
- `run_migration_60.py`
- `run_migration_61.py`
- `run_missing_tables_creation.py`
- `run_sql_repair.py`
- `seed_adaptive_config.py`

### Test Scripts
- `test_detector_correlation_query.py`
- `test_detector_labels.py`
- `test_detector_labels_fix.py`
- `test_model_registry.py`

## How to Use Archived Scripts

If you need to use an archived script:

```powershell
# Run from scripts/archive
cd scripts/archive
python check_dashboard_data.py

# Or specify full path
python scripts/archive/analyze_batch_results.py
```

## Notes

- These scripts may have stale imports or references
- They were functional at the time of archiving but are not maintained
- For SQL schema inspection, always use `scripts/sql/export_comprehensive_schema.py` instead of archived check scripts
