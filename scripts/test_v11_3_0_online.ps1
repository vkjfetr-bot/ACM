# v11.3.0 Testing - ONLINE Mode (Avoids Hang)
# Purpose: Thoroughly test v11.3.0 repeatability and accuracy

param(
    [ValidateSet('Phase1','Phase2','Phase3','All')]
    [string]$Phase = 'All'
)

$ErrorActionPreference = 'Continue'

function Write-Section {
    param([string]$Title)
    Write-Host "`n$('='*80)"
    Write-Host $Title
    Write-Host "="*80
}

function Write-Result {
    param(
        [string]$Test,
        [string]$Status = 'OK',
        [string]$Message = ''
    )
    $symbol = @{
        'OK' = '[OK]'
        'FAIL' = '[FAIL]'
        'WARN' = '[WARN]'
    }[$Status]
    Write-Host "$symbol $Test" -ForegroundColor $(
        if($Status -eq 'OK') { 'Green' } 
        elseif($Status -eq 'FAIL') { 'Red' } 
        else { 'Yellow' }
    )
    if($Message) { Write-Host "     $Message" }
}

Write-Host "`nv11.3.0 COMPREHENSIVE TESTING - ONLINE MODE"
Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ==========================================
# PHASE 1: Setup and Verify ONLINE Mode
# ==========================================
if ($Phase -in @('Phase1', 'All')) {
    Write-Section "PHASE 1: SETUP ONLINE MODE (Avoid Hang)"
    
    Write-Host "`n[1.1] Verify SQL Connection..."
    $result = sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E `
        -Q "SELECT COUNT(*) as EquipmentCount FROM Equipment" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Result "SQL Connection" "OK"
    } else {
        Write-Result "SQL Connection" "FAIL" $result
        exit 1
    }
    
    Write-Host "`n[1.2] Check Regime Models Cache..."
    $modelPath = "artifacts/regime_models"
    if (Test-Path $modelPath) {
        $modelCount = @(Get-ChildItem -Path $modelPath -Recurse -File).Count
        Write-Result "Models Cached" "OK" "$modelCount files found"
    } else {
        Write-Result "Models Cache" "WARN" "Not cached yet (will train)"
    }
    
    Write-Host "`n[1.3] Clear Previous Test Outputs..."
    @('artifacts/run_test1', 'artifacts/run_test2', 'artifacts/run_test3') | 
        ForEach-Object {
            if (Test-Path $_) {
                Remove-Item -Path $_ -Recurse -Force -ErrorAction SilentlyContinue
                Write-Host "  Cleared $_"
            }
        }
    
    Write-Host "`n[1.4] Test ONLINE Mode on Small Batch..."
    Write-Host "  Running: WFA_TURBINE_10, Sep 9-10 (1 day)"
    
    $startTime = Get-Date
    python scripts/sql_batch_runner.py --equip WFA_TURBINE_10 `
        --start-time "2023-09-09T00:00:00" --end-time "2023-09-10T23:59:59" `
        --tick-minutes 1440 --mode online 2>&1 | Tee-Object -FilePath "artifacts/test_phase1.log"
    
    if ($LASTEXITCODE -eq 0) {
        $elapsed = ((Get-Date) - $startTime).TotalSeconds
        Write-Result "ONLINE Mode Test" "OK" "Completed in $([math]::Round($elapsed))s"
    } else {
        Write-Result "ONLINE Mode Test" "FAIL" "Batch runner failed"
    }
}

# ==========================================
# PHASE 2: Repeatability Test
# ==========================================
if ($Phase -in @('Phase2', 'All')) {
    Write-Section "PHASE 2: REPEATABILITY TEST (Same Data Twice)"
    
    $startTime = "2023-09-09T00:00:00"
    $endTime = "2023-09-16T23:59:59"
    
    Write-Host "`n[2.1] Run 1: Analyze fault period (first pass)..."
    python scripts/sql_batch_runner.py --equip WFA_TURBINE_10 `
        --start-time $startTime --end-time $endTime `
        --tick-minutes 1440 --mode online 2>&1 | Tee-Object -FilePath "artifacts/run1.log"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Result "Run 1" "OK"
    } else {
        Write-Result "Run 1" "FAIL"
        exit 1
    }
    
    # Extract metrics from Run 1
    Write-Host "`n[2.2] Extract metrics from Run 1..."
    sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E `
        -Q "SELECT COUNT(*) as EpisodeCount, ROUND(AVG(Severity),2) as AvgSeverity, MAX(Severity) as MaxSeverity, COUNT(DISTINCT regime_context) as ContextTypes FROM ACM_EpisodeDiagnostics WHERE EquipID=10 AND StartTime >= '$startTime' AND StartTime < '$endTime'" `
        -o "artifacts/metrics_run1.txt" 2>&1
    
    Write-Host "`n[2.3] Clear outputs and run again (same data)..."
    sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E `
        -Q "DELETE FROM ACM_EpisodeDiagnostics WHERE EquipID=10 AND StartTime >= DATEADD(HOUR, -3, GETDATE())" 2>&1
    
    Write-Host "`n[2.4] Run 2: Analyze same period (second pass)..."
    python scripts/sql_batch_runner.py --equip WFA_TURBINE_10 `
        --start-time $startTime --end-time $endTime `
        --tick-minutes 1440 --mode online 2>&1 | Tee-Object -FilePath "artifacts/run2.log"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Result "Run 2" "OK"
    } else {
        Write-Result "Run 2" "FAIL"
        exit 1
    }
    
    # Extract metrics from Run 2
    Write-Host "`n[2.5] Extract metrics from Run 2..."
    sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E `
        -Q "SELECT COUNT(*) as EpisodeCount, ROUND(AVG(Severity),2) as AvgSeverity, MAX(Severity) as MaxSeverity, COUNT(DISTINCT regime_context) as ContextTypes FROM ACM_EpisodeDiagnostics WHERE EquipID=10 AND StartTime >= '$startTime' AND StartTime < '$endTime'" `
        -o "artifacts/metrics_run2.txt" 2>&1
    
    # Compare results
    Write-Host "`n[2.6] Comparing Run 1 vs Run 2..."
    $m1 = Get-Content "artifacts/metrics_run1.txt" | Select-Object -Last 1
    $m2 = Get-Content "artifacts/metrics_run2.txt" | Select-Object -Last 1
    
    if ($m1 -eq $m2) {
        Write-Result "REPEATABILITY TEST" "OK" "Results identical"
    } else {
        Write-Result "REPEATABILITY TEST" "WARN" "Results differ"
        Write-Host "  Run 1: $m1"
        Write-Host "  Run 2: $m2"
    }
}

# ==========================================
# PHASE 3: Fault Accuracy & Early Detection
# ==========================================
if ($Phase -in @('Phase3', 'All')) {
    Write-Section "PHASE 3: FAULT ACCURACY & EARLY DETECTION"
    
    $periods = @(
        @{ Name='Pre-fault'; Start='2023-09-01'; End='2023-09-08' },
        @{ Name='Fault'; Start='2023-09-09'; End='2023-09-16' },
        @{ Name='Post-fault'; Start='2023-09-17'; End='2023-09-30' }
    )
    
    foreach ($period in $periods) {
        Write-Host "`n[3.$($periods.IndexOf($period)+1)] Testing $($period.Name)..."
        Write-Host "  Date range: $($period.Start) to $($period.End)"
        
        $cmd = "python scripts/sql_batch_runner.py --equip WFA_TURBINE_10 " +
               "--start-time '$($period.Start)T00:00:00' " +
               "--end-time '$($period.End)T23:59:59' " +
               "--tick-minutes 1440 --mode online 2>&1"
        
        Invoke-Expression $cmd | Tee-Object -FilePath "artifacts/period_$($period.Name).log"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Result "$($period.Name) Batch" "OK"
        } else {
            Write-Result "$($period.Name) Batch" "FAIL"
            continue
        }
        
        # Query episode statistics
        $query = @"
SELECT 
    COUNT(*) as Episodes,
    SUM(CASE WHEN Severity >= 3.0 THEN 1 ELSE 0 END) as HighSeverity,
    ROUND(AVG(Severity), 2) as AvgSeverity,
    COUNT(DISTINCT regime_context) as ContextTypes,
    CASE WHEN COUNT(*) > 0 THEN ROUND(SUM(CASE WHEN Severity >= 3.0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) ELSE 0 END as HighSevPct
FROM ACM_EpisodeDiagnostics
WHERE EquipID=10 AND StartTime >= '$($period.Start)T00:00:00' AND StartTime < '$($period.End)T23:59:59'
"@
        
        $result = sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q $query 2>&1
        Write-Host "  $result" -ForegroundColor Cyan
    }
}

Write-Section "TESTING COMPLETE"
Write-Host "End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "`nAll results saved to artifacts/"
