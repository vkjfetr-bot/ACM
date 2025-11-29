# ACM Batch Mode Runner
# Runs ACM in batch mode for specified equipment with sequential batch numbers

param(
    [string]$Equipment = "FD_FAN",
    [int]$NumBatches = 5,
    [int]$StartBatch = 1
)

$ErrorActionPreference = "Continue"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "ACM Batch Mode Runner" -ForegroundColor Cyan
Write-Host "Equipment: $Equipment" -ForegroundColor Cyan
Write-Host "Batches: $StartBatch to $($StartBatch + $NumBatches - 1)" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

$successCount = 0
$failCount = 0

for ($i = $StartBatch; $i -lt ($StartBatch + $NumBatches); $i++) {
    Write-Host "`n[Batch $i] Starting..." -ForegroundColor Yellow
    
    $env:ACM_BATCH_MODE = "1"
    $env:ACM_BATCH_NUM = $i.ToString()
    
    $output = python -m core.acm_main --equip $Equipment 2>&1
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Host "[Batch $i] SUCCESS" -ForegroundColor Green
        $successCount++
    } else {
        Write-Host "[Batch $i] FAILED (exit code: $exitCode)" -ForegroundColor Red
        $failCount++
        
        # Show last 20 lines of output for failed runs
        $output | Select-Object -Last 20 | ForEach-Object {
            Write-Host $_ -ForegroundColor DarkRed
        }
    }
}

Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "Batch Run Complete" -ForegroundColor Cyan
Write-Host "Successful: $successCount" -ForegroundColor Green
Write-Host "Failed: $failCount" -ForegroundColor Red
Write-Host "======================================" -ForegroundColor Cyan
