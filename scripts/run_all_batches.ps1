<#!
.SYNOPSIS
    One-touch ACM batch runner that reprocesses the full historian range for all equipment.

.DESCRIPTION
    Consolidates the scattered batch helpers (run_batch_mode.ps1, run_data_range_batches.ps1,
    run/run_sql_batch.ps1, sql_batch_runner.py) into a single ergonomic entrypoint.
    Defaults run ACM in SQL batch mode for both FD_FAN and GAS_TURBINE, walking the
    entire historian range from the beginning with large daily windows and parallel
    execution where possible.

.PARAMETER Equipment
    Equipment codes to process. Defaults to FD_FAN and GAS_TURBINE.

.PARAMETER TickMinutes
    Batch window size in minutes. Defaults to 1440 (1 day) to sweep large ranges quickly.

.PARAMETER SQLServer / SQLDatabase
    Target SQL Server instance/database.

.PARAMETER MaxWorkers
    Number of equipment pipelines to execute concurrently. Defaults to 2 for both assets.

.PARAMETER SkipFresh
    Skip the automatic `--start-from-beginning` reset (defaults to full reprocess).

.PARAMETER Resume
    Resume from the last successful batch checkpoint (disables StartFresh when used).

.PARAMETER MaxBatches
    Optional cap to stop after N batches per equipment (useful for smoke tests).

.PARAMETER DryRun
    Print the composed python command without executing it.

.EXAMPLE
    .\scripts\run_all_batches.ps1
        # Runs ACM for FD_FAN and GAS_TURBINE across the full historian, 1-day windows, 2 workers.

.EXAMPLE
    .\scripts\run_all_batches.ps1 -TickMinutes 360 -MaxWorkers 1 -Resume
        # Resume incremental processing with 6-hour windows.
#>

param(
    [string[]]$Equipment = @("FD_FAN", "GAS_TURBINE"),
    [int]$TickMinutes = 1440,
    [string]$SQLServer = "localhost\B19CL3PCQLSERVER",
    [string]$SQLDatabase = "ACM",
    [int]$MaxWorkers = 2,
    [switch]$SkipFresh,
    [switch]$Resume,
    [int]$MaxBatches,
    [int]$MaxColdstartAttempts = 10,
    [switch]$DryRun
)

# Default to full historian sweep unless explicitly told to resume/skip.
$runFromBeginning = -not $SkipFresh.IsPresent
if ($Resume.IsPresent) {
    $runFromBeginning = $false
}

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "ACM ALL-EQUIPMENT BATCH RUNNER" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ("Equipment       : {0}" -f ($Equipment -join ", ")) -ForegroundColor Yellow
Write-Host ("SQL             : {0}/{1}" -f $SQLServer, $SQLDatabase) -ForegroundColor Gray
Write-Host ("Tick Minutes    : {0}" -f $TickMinutes) -ForegroundColor Gray
Write-Host ("Max Workers     : {0}" -f $MaxWorkers) -ForegroundColor Gray
Write-Host ("Start Fresh     : {0}" -f $runFromBeginning) -ForegroundColor Gray
Write-Host ("Resume          : {0}" -f $Resume.IsPresent) -ForegroundColor Gray
if ($MaxBatches) {
    Write-Host ("Max Batches    : {0}" -f $MaxBatches) -ForegroundColor Gray
}
Write-Host "===============================================" -ForegroundColor Cyan

# Build python argument list explicitly to avoid quoting surprises.
$arguments = @("scripts/sql_batch_runner.py", "--equip")
$arguments += $Equipment
$arguments += @("--sql-server", $SQLServer)
$arguments += @("--sql-database", $SQLDatabase)
$arguments += @("--tick-minutes", $TickMinutes)
$arguments += @("--max-workers", $MaxWorkers)
$arguments += @("--max-coldstart-attempts", $MaxColdstartAttempts)

if ($runFromBeginning) {
    $arguments += "--start-from-beginning"
}
if ($Resume.IsPresent) {
    $arguments += "--resume"
}
if ($MaxBatches) {
    $arguments += @("--max-batches", $MaxBatches)
}
if ($DryRun.IsPresent) {
    $arguments += "--dry-run"
}

Write-Host "python $($arguments -join ' ')" -ForegroundColor DarkGray

if ($DryRun.IsPresent) {
    Write-Host "[DRY-RUN] Command printed only" -ForegroundColor Yellow
    return
}

& python @arguments
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "`n=== ACM batch run finished successfully ===" -ForegroundColor Green
} else {
    Write-Host "`n=== ACM batch run FAILED (exit code $exitCode) ===" -ForegroundColor Red
    exit $exitCode
}
