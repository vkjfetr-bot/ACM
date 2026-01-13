# v11.3.0 Comprehensive Testing
# Uses actual batch runner interface (processes historian data in batches)

param(
    [ValidateSet('All','Repeatability','FaultAccuracy','Trend','Analysis')]
    [string]$Phase = 'All'
)

$ErrorActionPreference = 'Continue'

function Write-Banner {
    param([string]$Text)
    Write-Host "`n$('='*80)"
    Write-Host $Text
    Write-Host "="*80
}

Write-Host "v11.3.0 COMPREHENSIVE TESTING SUITE"
Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Test Strategy: Run full batch processing on WFA_TURBINE_10, verify repeatability and accuracy"

# ============================================================================
# TEST 1: REPEATABILITY - Run same equipment twice, verify identical results
# ============================================================================
if ($Phase -in @('All', 'Repeatability')) {
    Write-Banner "TEST 1: REPEATABILITY (Same Data, Two Runs)"
    
    Write-Host "`n[1.1] PREPARATION: Clear previous test outputs..."
    sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E `
        -Q "DELETE FROM ACM_ColdstartState WHERE Equipment LIKE 'test_%'; DELETE FROM ACM_Runs WHERE Equipment LIKE 'test_%'" 2>&1 | Out-Null
    
    Write-Host "[1.2] RUN 1: First pass on WFA_TURBINE_10 (full batch processing)..."
    Write-Host "      Using --start-from-beginning to process all available historian data"
    Write-Host "      This will process data in ~30-minute windows (default tick-minutes)"
    
    $timer1 = Measure-Command {
        python scripts/sql_batch_runner.py --equip WFA_TURBINE_10 --max-batches 1 --start-from-beginning 2>&1 | `
            Tee-Object -FilePath "artifacts/repeatability_run1.log"
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Run 1 Completed in $([math]::Round($timer1.TotalMinutes, 1)) minutes" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Run 1 Failed" -ForegroundColor Red
        exit 1
    }
    
    # Extract metrics from Run 1
    Write-Host "`n[1.3] EXTRACT METRICS: Run 1 episode statistics..."
    $query1 = @"
SELECT 
    'Run1' as Run,
    COUNT(*) as TotalEpisodes,
    SUM(CASE WHEN Severity >= 3.0 THEN 1 ELSE 0 END) as HighSevEpisodes,
    ROUND(AVG(Severity), 3) as AvgSeverity,
    MAX(Severity) as MaxSeverity,
    COUNT(DISTINCT regime_context) as RegimeContextTypes
FROM ACM_EpisodeDiagnostics
WHERE EquipID = 10
"@
    
    $run1Metrics = sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q $query1 2>&1 | `
        Select-Object -Skip 2 | Select-Object -First 1
    
    Write-Host "Run 1 Metrics:"
    Write-Host $run1Metrics
    
    # Save to file for comparison
    $run1Metrics | Out-File "artifacts/metrics_run1.txt"
    
    Write-Host "`n[1.4] SETUP: Clear outputs for Run 2 (keep models)..."
    sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E `
        -Q "DELETE FROM ACM_EpisodeDiagnostics WHERE EquipID = 10; DELETE FROM ACM_Runs WHERE Equipment = 'WFA_TURBINE_10'" 2>&1 | Out-Null
    
    Write-Host "[1.5] RUN 2: Second pass on same data (resume mode)..."
    Write-Host "      Models are cached from Run 1, should reuse them"
    
    $timer2 = Measure-Command {
        python scripts/sql_batch_runner.py --equip WFA_TURBINE_10 --max-batches 1 --resume 2>&1 | `
            Tee-Object -FilePath "artifacts/repeatability_run2.log"
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Run 2 Completed in $([math]::Round($timer2.TotalMinutes, 1)) minutes" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Run 2 Failed" -ForegroundColor Red
        exit 1
    }
    
    # Extract metrics from Run 2
    Write-Host "`n[1.6] EXTRACT METRICS: Run 2 episode statistics..."
    $run2Metrics = sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q $query1.Replace('Run1', 'Run2').Replace('Run1', 'Run2') 2>&1 | `
        Select-Object -Skip 2 | Select-Object -First 1
    
    Write-Host "Run 2 Metrics:"
    Write-Host $run2Metrics
    $run2Metrics | Out-File "artifacts/metrics_run2.txt"
    
    # Compare
    Write-Host "`n[1.7] COMPARISON: Run 1 vs Run 2..."
    $m1 = Get-Content "artifacts/metrics_run1.txt"
    $m2 = Get-Content "artifacts/metrics_run2.txt"
    
    if ($m1 -eq $m2) {
        Write-Host "[OK] IDENTICAL: Results are perfectly repeatable!" -ForegroundColor Green
    } else {
        Write-Host "[WARN] DIFFERENT: Results vary between runs" -ForegroundColor Yellow
        Write-Host "Run 1: $m1"
        Write-Host "Run 2: $m2"
    }
    
    Write-Host "`nREPEATABILITY TEST COMPLETE"
    Write-Host "Full logs: artifacts/repeatability_run[1-2].log"
}

# ============================================================================
# TEST 2: FAULT ACCURACY - Measure detection quality
# ============================================================================
if ($Phase -in @('All', 'FaultAccuracy')) {
    Write-Banner "TEST 2: FAULT DETECTION ACCURACY"
    
    Write-Host "`nKnown fault period for WFA_TURBINE_10:"
    Write-Host "  Sep 1-8, 2023: PRE-FAULT (healthy baseline)"
    Write-Host "  Sep 9-16, 2023: FAULT PERIOD (hydraulic degradation)"
    Write-Host "  Sep 17-30, 2023: POST-FAULT (recovery)"
    
    Write-Host "`nExpected results:"
    Write-Host "  Pre-fault: 5-15 episodes total, <20% high severity"
    Write-Host "  Fault: 50+ episodes, 95%+ high severity"
    Write-Host "  Post-fault: 5-15 episodes, <20% high severity"
    
    Write-Host "`n[2.1] Query episode distribution across all periods..."
    
    $query = @"
SELECT 
    CASE 
        WHEN StartTime >= '2023-09-01' AND StartTime < '2023-09-09' THEN 'Pre-Fault (Sep 1-8)'
        WHEN StartTime >= '2023-09-09' AND StartTime < '2023-09-17' THEN 'Fault Period (Sep 9-16)'
        WHEN StartTime >= '2023-09-17' AND StartTime < '2023-10-01' THEN 'Post-Fault (Sep 17-30)'
        ELSE 'Other'
    END as Period,
    COUNT(*) as TotalEpisodes,
    SUM(CASE WHEN Severity >= 3.0 THEN 1 ELSE 0 END) as HighSeverity,
    SUM(CASE WHEN Severity < 3.0 THEN 1 ELSE 0 END) as LowSeverity,
    ROUND(AVG(Severity), 2) as AvgSeverity,
    ROUND(100.0 * SUM(CASE WHEN Severity >= 3.0 THEN 1 ELSE 0 END) / COUNT(*), 1) as HighSevPct
FROM ACM_EpisodeDiagnostics
WHERE EquipID = 10 AND StartTime >= '2023-09-01' AND StartTime < '2023-10-01'
GROUP BY CASE 
    WHEN StartTime >= '2023-09-01' AND StartTime < '2023-09-09' THEN 'Pre-Fault (Sep 1-8)'
    WHEN StartTime >= '2023-09-09' AND StartTime < '2023-09-17' THEN 'Fault Period (Sep 9-16)'
    WHEN StartTime >= '2023-09-17' AND StartTime < '2023-10-01' THEN 'Post-Fault (Sep 17-30)'
    ELSE 'Other'
END
ORDER BY Period
"@
    
    sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q $query 2>&1 | `
        Tee-Object -FilePath "artifacts/fault_accuracy_results.txt"
    
    Write-Host "`nSaved to: artifacts/fault_accuracy_results.txt"
}

# ============================================================================
# TEST 3: DAILY TREND ANALYSIS - Early detection capability
# ============================================================================
if ($Phase -in @('All', 'Trend')) {
    Write-Banner "TEST 3: DAILY DEGRADATION TREND (Early Detection)"
    
    Write-Host "`nAnalyzing daily episode patterns to demonstrate early detection:"
    Write-Host "  Day 9 (Sep 9): Should show initiation (3-5 episodes)"
    Write-Host "  Day 12 (Sep 12): Should show escalation (5-10 episodes)"
    Write-Host "  Day 16 (Sep 16): Should show failure (15+ episodes)"
    
    Write-Host "`n[3.1] Extracting daily trends for September 2023..."
    
    $query = @"
SELECT 
    DATEPART(DAY, StartTime) as Day,
    COUNT(*) as Episodes,
    SUM(CASE WHEN Severity >= 3.5 THEN 1 ELSE 0 END) as CriticalEpisodes,
    ROUND(AVG(Severity), 2) as AvgSeverity,
    MAX(Severity) as PeakSeverity,
    STUFF(
        (SELECT DISTINCT ', ' + regime_context 
         FROM ACM_EpisodeDiagnostics 
         WHERE EquipID=10 AND DATEPART(DAY, StartTime) = DATEPART(DAY, original.StartTime)
         FOR XML PATH ('')), 1, 2, ''
    ) as regime_contexts
FROM ACM_EpisodeDiagnostics original
WHERE EquipID = 10 AND YEAR(StartTime) = 2023 AND MONTH(StartTime) = 9
GROUP BY DATEPART(DAY, StartTime)
ORDER BY Day
"@
    
    sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q $query -W 200 2>&1 | `
        Tee-Object -FilePath "artifacts/daily_trend_september.txt"
    
    Write-Host "`n[3.2] Interpretation guide:"
    Write-Host "  Sep 1-8: Should be stable (1-3 episodes/day, avg severity 1-2)"
    Write-Host "  Sep 9: Fault initiation (episodes increase)"
    Write-Host "  Sep 10-15: Rapid degradation (10+ episodes/day, severity 3-4)"
    Write-Host "  Sep 16: Failure point (peak episodes and severity)"
    Write-Host "  Sep 17+: Recovery (return to baseline)"
    
    Write-Host "`nSaved to: artifacts/daily_trend_september.txt"
}

# ============================================================================
# TEST 4: REGIME CONTEXT DISTRIBUTION - Health-state classification
# ============================================================================
if ($Phase -in @('All', 'Analysis')) {
    Write-Banner "TEST 4: REGIME CONTEXT ANALYSIS (v11.3.0 Health-State Classification)"
    
    Write-Host "`nAnalyzing how v11.3.0 classifies episodes by regime context:"
    Write-Host "  health_degradation: Should dominate fault period (80%+)"
    Write-Host "  stable: Should be present during normal operation"
    Write-Host "  operating_mode: Should be minimal (mode switches are rare)"
    Write-Host "  health_transition: Ambiguous cases (10-20%)"
    
    Write-Host "`n[4.1] Regime context distribution for entire dataset..."
    
    $query = @"
SELECT 
    regime_context,
    COUNT(*) as EpisodeCount,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM ACM_EpisodeDiagnostics WHERE EquipID = 10), 1) as Percentage,
    ROUND(AVG(Severity), 2) as AvgSeverity,
    MAX(Severity) as MaxSeverity,
    COUNT(DISTINCT CAST(StartTime AS DATE)) as DaysActive
FROM ACM_EpisodeDiagnostics
WHERE EquipID = 10
GROUP BY regime_context
ORDER BY EpisodeCount DESC
"@
    
    sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q $query 2>&1 | `
        Tee-Object -FilePath "artifacts/regime_context_distribution.txt"
    
    Write-Host "`n[4.2] FAULT PERIOD ONLY - regime_context should show dominance of health_degradation..."
    
    $query2 = @"
SELECT 
    regime_context,
    COUNT(*) as EpisodeCount,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM ACM_EpisodeDiagnostics WHERE EquipID = 10 AND StartTime >= '2023-09-09' AND StartTime < '2023-09-17'), 1) as Percentage,
    ROUND(AVG(Severity), 2) as AvgSeverity
FROM ACM_EpisodeDiagnostics
WHERE EquipID = 10 AND StartTime >= '2023-09-09' AND StartTime < '2023-09-17'
GROUP BY regime_context
ORDER BY EpisodeCount DESC
"@
    
    sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q $query2 2>&1 | `
        Tee-Object -FilePath "artifacts/regime_context_fault_period.txt"
    
    Write-Host "`n[4.3] Check for UNKNOWN regimes (should be <5%)..."
    
    $unknownQuery = @"
SELECT 
    COUNT(CASE WHEN regime_context = 'UNKNOWN' OR regime_context IS NULL THEN 1 END) as UnknownCount,
    COUNT(*) as TotalEpisodes,
    ROUND(100.0 * COUNT(CASE WHEN regime_context = 'UNKNOWN' OR regime_context IS NULL THEN 1 END) / COUNT(*), 1) as UnknownPct
FROM ACM_EpisodeDiagnostics
WHERE EquipID = 10
"@
    
    $unknownResult = sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q $unknownQuery 2>&1 | Select-Object -Skip 2 | Select-Object -First 1
    Write-Host "Unknown regime count: $unknownResult"
    
    Write-Host "`nAnalysis files saved:"
    Write-Host "  artifacts/regime_context_distribution.txt"
    Write-Host "  artifacts/regime_context_fault_period.txt"
}

Write-Banner "ALL TESTING COMPLETE"
Write-Host "Summary of outputs in artifacts/:"
Write-Host "  - repeatability_run[1-2].log: Full batch runner logs"
Write-Host "  - metrics_run[1-2].txt: Extracted metrics for comparison"
Write-Host "  - fault_accuracy_results.txt: Pre/Fault/Post period analysis"
Write-Host "  - daily_trend_september.txt: Day-by-day progression"
Write-Host "  - regime_context_distribution.txt: Overall context classification"
Write-Host "  - regime_context_fault_period.txt: Fault-window context analysis"
Write-Host "`nEnd: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
