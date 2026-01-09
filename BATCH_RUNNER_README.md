# ACM Distilled - Batch Runner Helper Scripts

## Overview

Helper scripts to run `acm_distilled.py` for multiple equipment automatically:
- **2 Wind Turbines**: WIND_TURBINE_01, WIND_TURBINE_02
- **FD Fan**: FD_FAN

## Scripts

### 1. `run_distilled_batch.py` (Python)
Full-featured Python script with all options.

**Features:**
- Customizable equipment list
- Flexible time ranges
- Optional file output
- Progress reporting
- Exit codes for automation

**Usage:**
```bash
# Run for all equipment (last 30 days, console output)
python run_distilled_batch.py

# Run for specific time range
python run_distilled_batch.py \
    --start-time "2024-01-01T00:00:00" \
    --end-time "2024-01-31T23:59:59"

# Run for specific equipment only
python run_distilled_batch.py --equip WIND_TURBINE_01 FD_FAN

# Save reports to files
python run_distilled_batch.py --output-dir reports/

# Combine options
python run_distilled_batch.py \
    --equip WIND_TURBINE_01 WIND_TURBINE_02 FD_FAN \
    --start-time "2024-11-01T00:00:00" \
    --end-time "2024-11-30T23:59:59" \
    --output-dir monthly_reports/
```

**Options:**
- `--equip EQUIP [EQUIP ...]` - Equipment codes (default: all 3)
- `--start-time TIME` - Start time ISO format (default: 30 days ago)
- `--end-time TIME` - End time ISO format (default: now)
- `--output-dir DIR` - Save reports to directory (default: console)

---

### 2. `run_distilled_quick.sh` (Bash)
Quick bash script for simple use cases.

**Features:**
- Simple command-line interface
- Defaults to all 3 equipment
- Auto-calculates time ranges

**Usage:**
```bash
# Last 30 days (default)
./run_distilled_quick.sh

# Last 7 days
./run_distilled_quick.sh 7

# Last 14 days, save to reports/
./run_distilled_quick.sh 14 reports/

# Last 1 day (yesterday + today)
./run_distilled_quick.sh 1
```

**Arguments:**
1. Number of days (default: 30)
2. Output directory (default: console)

---

## Output Formats

### Console Output
When no output directory specified, reports print to console:
```
================================================================================
ACM ANALYTICS REPORT - WIND_TURBINE_01
================================================================================
Analysis Period: 2024-01-01 00:00:00 to 2024-01-31 23:59:59

1. DATA SUMMARY
--------------------------------------------------------------------------------
  Train Rows: 1440
  Score Rows: 960
  ...
```

### File Output
When `--output-dir` specified, creates timestamped files:
```
reports/
├── WIND_TURBINE_01_20260109_120000.txt
├── WIND_TURBINE_02_20260109_120030.txt
└── FD_FAN_20260109_120100.txt
```

---

## Examples

### Quick Daily Check
```bash
# Check last 24 hours for all equipment
./run_distilled_quick.sh 1
```

### Weekly Report (Save to Files)
```bash
# Analyze last 7 days, save reports
python run_distilled_batch.py \
    --output-dir weekly_reports/ \
    --start-time "$(date -d '7 days ago' +%Y-%m-%dT00:00:00)" \
    --end-time "$(date +%Y-%m-%dT23:59:59)"
```

### Monthly Analysis (Specific Equipment)
```bash
# Analyze wind turbines only for December 2024
python run_distilled_batch.py \
    --equip WIND_TURBINE_01 WIND_TURBINE_02 \
    --start-time "2024-12-01T00:00:00" \
    --end-time "2024-12-31T23:59:59" \
    --output-dir december_2024/
```

### Investigate Specific Incident
```bash
# Analyze FD_FAN around incident time
python acm_distilled.py \
    --equip FD_FAN \
    --start-time "2024-11-15T10:00:00" \
    --end-time "2024-11-15T16:00:00" \
    --output incident_analysis.txt
```

---

## Automation Examples

### Cron Job (Daily Report)
```bash
# Run daily at 2 AM, analyze previous day
0 2 * * * cd /path/to/ACM && ./run_distilled_quick.sh 1 daily_reports/
```

### Scheduled Task (Windows)
```powershell
# Run weekly on Mondays at 6 AM
schtasks /create /tn "ACM Weekly" /tr "python C:\ACM\run_distilled_batch.py --output-dir C:\ACM\weekly" /sc weekly /d MON /st 06:00
```

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Run ACM Analysis
  run: |
    python run_distilled_batch.py --output-dir ${{ github.workspace }}/reports
    
- name: Upload Reports
  uses: actions/upload-artifact@v3
  with:
    name: acm-reports
    path: reports/
```

---

## Troubleshooting

### Script Fails Immediately
**Problem**: Equipment not found in database
```
ERROR: Equipment 'WIND_TURBINE_01' not found in database
```
**Solution**: Check equipment codes in your SQL Equipment table

### No Data for Time Range
**Problem**: Insufficient data
```
Insufficient data: train=0 rows, score=0 rows
```
**Solution**: Check historian tables have data for the time range

### Permission Denied (Bash Script)
**Problem**: Script not executable
```
bash: ./run_distilled_quick.sh: Permission denied
```
**Solution**: 
```bash
chmod +x run_distilled_quick.sh
```

---

## Requirements

Both scripts require:
- Python 3.11+
- ACM dependencies installed (`pip install -e .`)
- SQL Server connection configured (`configs/sql_connection.ini`)
- Equipment data in historian tables

---

## See Also

- `acm_distilled.py` - Main analytics script
- `README_DISTILLED.md` - ACM Distilled documentation
- `examples/acm_distilled_demo.py` - Usage examples
