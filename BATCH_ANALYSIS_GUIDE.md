# ACM Batch Analysis Script - Quick Reference

## Script Location
`scripts\run_batch_analysis.ps1`

## Usage Modes

### 1. Start Batch Analysis (Default)
Launches background jobs for ALL equipment with full historical data analysis:
```powershell
.\scripts\run_batch_analysis.ps1
```

### 2. Check Job Status
View current status of all batch jobs and their recent output:
```powershell
.\scripts\run_batch_analysis.ps1 -Status
```

### 3. Check SQL Table Counts
See how many rows have been written to each table:
```powershell
.\scripts\run_batch_analysis.ps1 -Tables
```

### 4. Monitor Continuously
Auto-refresh status and tables every 10 seconds (Ctrl+C to stop):
```powershell
.\scripts\run_batch_analysis.ps1 -Monitor
```

Customize refresh interval (in seconds):
```powershell
.\scripts\run_batch_analysis.ps1 -Monitor -MonitorInterval 30
```

### 5. Stop All Jobs
Stop and remove all ACM batch jobs:
```powershell
.\scripts\run_batch_analysis.ps1 -Stop
```

## Equipment Configuration

Currently configured equipment in the script:
- `FD_FAN` (EquipID: 1)
- `GAS_TURBINE` (EquipID: 2)

Each equipment processes 24-hour batches (1440 minutes) through all historical data.

## Expected Behavior

### When Starting Jobs:
- Creates background PowerShell jobs for each equipment
- Each job runs `sql_batch_runner.py --start-from-beginning`
- Jobs run independently and can be monitored via `-Status` or `-Monitor`

### Job Names:
- `ACM_FD_FAN`
- `ACM_GAS_TURBINE`

### Tables Populated:
1. **ACM_Runs** - One row per batch run
2. **ACM_Scores_Wide** - Anomaly scores for each timestamp
3. **ACM_HealthTimeline** - Health index tracking
4. **ACM_Episodes** - Detected anomaly episodes
5. **ModelRegistry** - Saved detector models
6. **ACM_BaselineBuffer** - Training data cache
7. **30+ Analytics Tables** - Comprehensive insights

## Typical Workflow

```powershell
# 1. Start batch analysis
.\scripts\run_batch_analysis.ps1

# 2. Check initial status
.\scripts\run_batch_analysis.ps1 -Status

# 3. Monitor progress (let it run for a while)
.\scripts\run_batch_analysis.ps1 -Monitor

# 4. Or check periodically
.\scripts\run_batch_analysis.ps1 -Tables

# 5. When done or to restart
.\scripts\run_batch_analysis.ps1 -Stop
```

## Processing Time Estimates

- **FD_FAN**: ~700 batches (historical data from 2023-2025)
  - ~5-10 seconds per batch
  - Total: ~1-2 hours

- **GAS_TURBINE**: ~680 batches
  - ~5-10 seconds per batch  
  - Total: ~1-2 hours

Both run in parallel, so total time is the longest of the two.

## Troubleshooting

### If jobs fail to start:
```powershell
# Check if jobs already exist
Get-Job -Name "ACM_*"

# Stop existing jobs
.\scripts\run_batch_analysis.ps1 -Stop

# Restart
.\scripts\run_batch_analysis.ps1
```

### If you see "No data returned from SQL historian":
- Normal for time windows outside the data range
- Jobs will skip empty windows and continue
- Check table counts to verify data is being written

### To clear state and restart:
```powershell
# Clear database state
sqlcmd -S localhost\B19CL3PCQLSERVER -d ACM -E -Q "DELETE FROM ACM_ColdstartState; DELETE FROM ACM_Runs"

# Restart jobs
.\scripts\run_batch_analysis.ps1 -Stop
.\scripts\run_batch_analysis.ps1
```

## Notes

- Jobs run in the background - you can close the terminal
- Use `Get-Job` to check jobs from any PowerShell window
- Logs are captured in job output (view with `-Status`)
- SQL tables are updated in real-time as batches complete
