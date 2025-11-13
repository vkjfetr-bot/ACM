# SQL Batch Runner - Quick Reference

## Common Commands

### PowerShell (Recommended)

```powershell
# Basic: Process single equipment
.\scripts\run\run_sql_batch.ps1 -Equipment FD_FAN

# Resume: Continue from last batch
.\scripts\run\run_sql_batch.ps1 -Equipment FD_FAN -Resume

# Parallel: Process multiple equipment
.\scripts\run\run_sql_batch.ps1 -Equipment FD_FAN,GAS_TURBINE -MaxWorkers 2

# Preview: Dry run without execution
.\scripts\run\run_sql_batch.ps1 -Equipment FD_FAN -DryRun

# Custom: Adjust tick window
.\scripts\run\run_sql_batch.ps1 -Equipment FD_FAN -TickMinutes 15
```

### Python (Advanced)

```bash
# Basic
python scripts/sql_batch_runner.py --equip FD_FAN

# Multiple equipment
python scripts/sql_batch_runner.py --equip FD_FAN GAS_TURBINE --max-workers 2

# Resume
python scripts/sql_batch_runner.py --equip FD_FAN --resume

# Dry run
python scripts/sql_batch_runner.py --equip FD_FAN --dry-run
```

## Check Progress

```powershell
# View progress file
Get-Content "artifacts\.sql_batch_progress.json" | ConvertFrom-Json | Format-List

# Check coldstart status
sqlcmd -S "localhost\B19CL3PCQLSERVER" -E -d ACM -Q "SELECT * FROM ACM_ColdstartState"

# Check models
sqlcmd -S "localhost\B19CL3PCQLSERVER" -E -d ACM -Q "SELECT * FROM ModelRegistry"

# Check runs
sqlcmd -S "localhost\B19CL3PCQLSERVER" -E -d ACM -Q "SELECT TOP 10 * FROM ACM_Runs ORDER BY StartTime DESC"
```

## Troubleshooting

```powershell
# Lower minimum training samples if coldstart won't complete
sqlcmd -S "localhost\B19CL3PCQLSERVER" -E -d ACM -Q "UPDATE ACM_Config SET ParamValue='100' WHERE ParamPath='data.min_train_samples'"

# Reset coldstart state
sqlcmd -S "localhost\B19CL3PCQLSERVER" -E -d ACM -Q "DELETE FROM ACM_ColdstartState WHERE EquipID=1"

# Clear progress file to restart
Remove-Item "artifacts\.sql_batch_progress.json"

# Check available data
sqlcmd -S "localhost\B19CL3PCQLSERVER" -E -d ACM -Q "SELECT COUNT(*), MIN(EntryDateTime), MAX(EntryDateTime) FROM FD_FAN_Data"
```

## Key Features

✅ **Smart Coldstart**: Auto-detects cadence, loads optimal window  
✅ **Continuous Batch**: Processes all available data  
✅ **Resume Support**: Continue after interruption  
✅ **Parallel Processing**: Multiple equipment simultaneously  
✅ **Progress Tracking**: Database + JSON file tracking  
✅ **No File Fallback**: Pure SQL-only mode  

## Status: PRODUCTION READY 🚀
