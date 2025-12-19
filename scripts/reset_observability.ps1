# =============================================================================
# reset_observability.ps1 - Wipe all observability data and restart fresh
# =============================================================================
# Usage: .\scripts\reset_observability.ps1
#        .\scripts\reset_observability.ps1 -RunACM   # Also run ACM batch after reset
# =============================================================================

param(
    [switch]$RunACM,
    [string]$Equipment = "FD_FAN",
    [int]$TickMinutes = 1440
)

$ErrorActionPreference = "Stop"
$ObsDir = Join-Path $PSScriptRoot "..\install\observability"

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host " ACM Observability Reset Script" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------------------------
# Step 1: Stop all containers
# -----------------------------------------------------------------------------
Write-Host "[1/4] Stopping observability containers..." -ForegroundColor Yellow
Push-Location $ObsDir
try {
    docker compose down 2>$null
    Write-Host "      Containers stopped." -ForegroundColor Green
} catch {
    Write-Host "      Warning: Could not stop containers (may not be running)" -ForegroundColor DarkYellow
}

# -----------------------------------------------------------------------------
# Step 2: Remove Docker volumes (this wipes ALL stored data)
# -----------------------------------------------------------------------------
Write-Host "[2/4] Removing Docker volumes (wiping all data)..." -ForegroundColor Yellow

# Find and remove all observability-related volumes
$allVolumes = docker volume ls --format "{{.Name}}" 2>$null
$obsVolumes = $allVolumes | Where-Object { $_ -match "observability|acm.*data|prometheus|loki|tempo|pyroscope|grafana" }

if ($obsVolumes) {
    foreach ($vol in $obsVolumes) {
        $null = docker volume rm $vol 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "      Removed: $vol" -ForegroundColor Green
        } else {
            Write-Host "      Skipped: $vol (in use or error)" -ForegroundColor DarkGray
        }
    }
} else {
    Write-Host "      No observability volumes found." -ForegroundColor DarkGray
}

# -----------------------------------------------------------------------------
# Step 3: Restart containers fresh
# -----------------------------------------------------------------------------
Write-Host "[3/4] Starting fresh observability stack..." -ForegroundColor Yellow
docker compose up -d
Start-Sleep -Seconds 5

# Check container health
Write-Host "      Waiting for containers to be healthy..." -ForegroundColor DarkGray
$maxWait = 30
$waited = 0
while ($waited -lt $maxWait) {
    $unhealthy = docker compose ps --format json 2>$null | ConvertFrom-Json | Where-Object { $_.Health -eq "starting" }
    if (-not $unhealthy) {
        break
    }
    Start-Sleep -Seconds 2
    $waited += 2
}

# Show container status
Write-Host ""
docker compose ps --format "table {{.Name}}\t{{.Status}}"
Write-Host ""
Pop-Location

# -----------------------------------------------------------------------------
# Step 4: Optionally run ACM batch
# -----------------------------------------------------------------------------
if ($RunACM) {
    Write-Host "[4/4] Running ACM batch in background..." -ForegroundColor Yellow
    $acmRoot = Join-Path $PSScriptRoot ".."
    Push-Location $acmRoot
    
    Write-Host "      Equipment: $Equipment" -ForegroundColor Cyan
    Write-Host "      Tick: $TickMinutes minutes" -ForegroundColor Cyan
    Write-Host ""
    
    # Run in background
    $cmd = "python scripts/sql_batch_runner.py --equip $Equipment --tick-minutes $TickMinutes --max-workers 1"
    Write-Host "      Command: $cmd" -ForegroundColor DarkGray
    Write-Host ""
    
    Start-Process -FilePath "python" -ArgumentList "scripts/sql_batch_runner.py", "--equip", $Equipment, "--tick-minutes", $TickMinutes, "--max-workers", "1" -NoNewWindow
    
    Pop-Location
    Write-Host "      ACM batch started in background." -ForegroundColor Green
} else {
    Write-Host "[4/4] Skipped ACM run (use -RunACM to auto-run)" -ForegroundColor DarkGray
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host " Reset Complete!" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " Grafana:    http://localhost:3000  (admin/admin)" -ForegroundColor White
Write-Host " Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host " Loki:       http://localhost:3100" -ForegroundColor White
Write-Host " Tempo:      http://localhost:3200" -ForegroundColor White
Write-Host " Pyroscope:  http://localhost:4040" -ForegroundColor White
Write-Host ""

if (-not $RunACM) {
    Write-Host " To run ACM and generate fresh data:" -ForegroundColor Yellow
    Write-Host "   python scripts/sql_batch_runner.py --equip FD_FAN --tick-minutes 1440 --max-workers 1" -ForegroundColor White
    Write-Host ""
}
