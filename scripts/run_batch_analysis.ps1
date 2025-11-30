# ACM Batch Analysis Runner
# Runs batch mode for all configured equipment as background jobs
# and provides monitoring capabilities

param(
    [switch]$Monitor,
    [switch]$Stop,
    [switch]$Status,
    [switch]$Tables,
    [int]$MonitorInterval = 10
)

$ACM_ROOT = Split-Path -Parent $PSScriptRoot
$EQUIPMENT = @("FD_FAN", "GAS_TURBINE")
$TICK_MINUTES = 1440  # 24 hours per batch

function Start-BatchJobs {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "Starting ACM Batch Analysis for All Equipment" -ForegroundColor Cyan
    Write-Host "============================================================`n" -ForegroundColor Cyan
    
    foreach ($equip in $EQUIPMENT) {
        $jobName = "ACM_$equip"
        
        # Check if job already exists
        $existingJob = Get-Job -Name $jobName -ErrorAction SilentlyContinue
        if ($existingJob) {
            Write-Host "[WARN] Job '$jobName' already exists (State: $($existingJob.State))" -ForegroundColor Yellow
            Write-Host "       Stop it first with: .\scripts\run_batch_analysis.ps1 -Stop`n" -ForegroundColor Yellow
            continue
        }
        
        Write-Host "[START] Launching batch analysis for $equip..." -ForegroundColor Green
        
        $job = Start-Job -Name $jobName -ScriptBlock {
            param($root, $equip, $tick)
            Set-Location $root
            python scripts/sql_batch_runner.py --equip $equip --start-from-beginning --tick-minutes $tick
        } -ArgumentList $ACM_ROOT, $equip, $TICK_MINUTES
        
        Write-Host "        Job ID: $($job.Id) | Name: $($job.Name) | State: $($job.State)" -ForegroundColor Gray
    }
    
    Write-Host "`n[INFO] All jobs started. Use -Status or -Monitor to track progress`n" -ForegroundColor Cyan
}

function Stop-BatchJobs {
    Write-Host "`n[STOP] Stopping all ACM batch jobs..." -ForegroundColor Yellow
    
    foreach ($equip in $EQUIPMENT) {
        $jobName = "ACM_$equip"
        $job = Get-Job -Name $jobName -ErrorAction SilentlyContinue
        
        if ($job) {
            Write-Host "       Stopping $jobName..." -ForegroundColor Yellow
            Stop-Job -Name $jobName -ErrorAction SilentlyContinue
            Remove-Job -Name $jobName -Force -ErrorAction SilentlyContinue
            Write-Host "       $jobName stopped and removed" -ForegroundColor Gray
        }
    }
    
    Write-Host "[DONE] All jobs stopped`n" -ForegroundColor Green
}

function Show-JobStatus {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "ACM Batch Jobs Status" -ForegroundColor Cyan
    Write-Host "============================================================`n" -ForegroundColor Cyan
    
    $jobs = Get-Job -Name "ACM_*" -ErrorAction SilentlyContinue
    
    if (-not $jobs) {
        Write-Host "[INFO] No ACM batch jobs running`n" -ForegroundColor Yellow
        return
    }
    
    foreach ($job in $jobs) {
        $runtime = if ($job.PSBeginTime) {
            $elapsed = (Get-Date) - $job.PSBeginTime
            "{0:D2}h {1:D2}m {2:D2}s" -f $elapsed.Hours, $elapsed.Minutes, $elapsed.Seconds
        } else {
            "N/A"
        }
        
        $stateColor = switch ($job.State) {
            "Running" { "Green" }
            "Completed" { "Cyan" }
            "Failed" { "Red" }
            default { "Yellow" }
        }
        
        Write-Host "Job: " -NoNewline
        Write-Host $job.Name -ForegroundColor White -NoNewline
        Write-Host " | State: " -NoNewline
        Write-Host $job.State -ForegroundColor $stateColor -NoNewline
        Write-Host " | Runtime: $runtime"
        
        # Show last 5 lines of output
        $output = Receive-Job -Id $job.Id -Keep | Select-Object -Last 5
        if ($output) {
            Write-Host "  Last output:" -ForegroundColor Gray
            $output | ForEach-Object {
                $line = $_.ToString().Trim()
                if ($line) {
                    Write-Host "    $line" -ForegroundColor DarkGray
                }
            }
        }
        Write-Host ""
    }
}

function Show-TableCounts {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "ACM SQL Table Counts" -ForegroundColor Cyan
    Write-Host "============================================================`n" -ForegroundColor Cyan
    
    $tables = @(
        @{Name="ACM_Runs"; Desc="Run tracking"},
        @{Name="ACM_Scores_Wide"; Desc="Anomaly scores"},
        @{Name="ACM_HealthTimeline"; Desc="Health tracking"},
        @{Name="ACM_Episodes"; Desc="Anomaly episodes"},
        @{Name="ACM_BaselineBuffer"; Desc="Training data"},
        @{Name="ModelRegistry"; Desc="Saved models"},
        @{Name="ACM_SensorRanking"; Desc="Top sensors"},
        @{Name="ACM_DefectTimeline"; Desc="Defect events"},
        @{Name="ACM_RegimeTimeline"; Desc="Regime labels"}
    )
    
    foreach ($table in $tables) {
        $query = "SELECT COUNT(*) FROM dbo.$($table.Name) WHERE EquipID IN (1, 2)"
        $result = sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q $query -h -1 -W 2>&1 | Where-Object { $_ -match '^\s*\d+\s*$' }
        
        $count = 0
        if ($result) {
            $count = [int]($result.Trim())
        }
        
        $countColor = if ($count -eq 0) { "DarkGray" } elseif ($count -lt 100) { "Yellow" } else { "Green" }
        Write-Host ("{0,-25} " -f $table.Name) -NoNewline
        Write-Host ("{0,10} rows" -f $count) -ForegroundColor $countColor -NoNewline
        Write-Host ("   ({0})" -f $table.Desc) -ForegroundColor Gray
    }
    
    Write-Host "`n[INFO] Showing combined counts for both equipment (EquipID 1 & 2)`n" -ForegroundColor Gray
}

function Start-Monitoring {
    Write-Host "`n[MONITOR] Starting continuous monitoring (Ctrl+C to stop)..." -ForegroundColor Cyan
    Write-Host "[MONITOR] Refresh interval: $MonitorInterval seconds`n" -ForegroundColor Gray
    
    try {
        while ($true) {
            Clear-Host
            Write-Host "ACM Batch Analysis Monitor - " -NoNewline
            Write-Host (Get-Date -Format "yyyy-MM-dd HH:mm:ss") -ForegroundColor Yellow
            
            Show-JobStatus
            Show-TableCounts
            
            Write-Host "`n[MONITOR] Press Ctrl+C to stop monitoring..." -ForegroundColor DarkGray
            Start-Sleep -Seconds $MonitorInterval
        }
    }
    catch {
        Write-Host "`n[MONITOR] Monitoring stopped`n" -ForegroundColor Yellow
    }
}

# Main execution
if ($Stop) {
    Stop-BatchJobs
}
elseif ($Status) {
    Show-JobStatus
}
elseif ($Tables) {
    Show-TableCounts
}
elseif ($Monitor) {
    Start-Monitoring
}
else {
    # Default: Start jobs
    Start-BatchJobs
}
