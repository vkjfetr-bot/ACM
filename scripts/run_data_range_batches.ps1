# ACM Data Range Batch Runner
# Processes batches over the actual data time range in ACM_HistorianData

param(
    [string]$Equipment = "FD_FAN",
    [int]$NumBatches = 1000,
    [string]$StartDate = "2023-10-15 00:00:00",
    [string]$EndDate = "2025-09-14 23:30:00",
    [int]$BatchSizeMinutes = 1440  # 1 day = 1440 minutes
)

$ErrorActionPreference = "Continue"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "ACM Data Range Batch Runner" -ForegroundColor Cyan
Write-Host "Equipment: $Equipment" -ForegroundColor Cyan
Write-Host "Date Range: $StartDate to $EndDate" -ForegroundColor Cyan
Write-Host "Batch Size: $BatchSizeMinutes minutes" -ForegroundColor Cyan
Write-Host "Number of Batches: $NumBatches" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

$start = [DateTime]::Parse($StartDate)
$end = [DateTime]::Parse($EndDate)
$batchSpan = New-TimeSpan -Minutes $BatchSizeMinutes

$successCount = 0
$failCount = 0
$currentTime = $start

for ($i = 1; $i -le $NumBatches; $i++) {
    $batchEnd = $currentTime.Add($batchSpan)
    
    if ($batchEnd -gt $end) {
        $batchEnd = $end
    }
    
    $startStr = $currentTime.ToString("yyyy-MM-ddTHH:mm:ss")
    $endStr = $batchEnd.ToString("yyyy-MM-ddTHH:mm:ss")
    
    Write-Host "`n[Batch $i/$NumBatches] $startStr to $endStr" -ForegroundColor Yellow
    
    $output = python -m core.acm_main --equip $Equipment --start-time $startStr --end-time $endStr 2>&1
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        # Check if it was a real success or NOOP
        $wasNoop = $output | Select-String -Pattern "outcome=NOOP" -Quiet
        if ($wasNoop) {
            Write-Host "[Batch $i] NOOP (no data in window)" -ForegroundColor DarkYellow
        } else {
            Write-Host "[Batch $i] SUCCESS" -ForegroundColor Green
            $successCount++
        }
    } else {
        Write-Host "[Batch $i] FAILED (exit code: $exitCode)" -ForegroundColor Red
        $failCount++
        
        # Show last 15 lines of output for failed runs
        $output | Select-Object -Last 15 | ForEach-Object {
            Write-Host $_ -ForegroundColor DarkRed
        }
    }
    
    $currentTime = $batchEnd
    
    if ($currentTime -ge $end) {
        Write-Host "`nReached end of data range at batch $i" -ForegroundColor Cyan
        break
    }
}

Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "Batch Run Complete" -ForegroundColor Cyan
Write-Host "Successful: $successCount" -ForegroundColor Green
Write-Host "Failed: $failCount" -ForegroundColor Red
Write-Host "======================================" -ForegroundColor Cyan
