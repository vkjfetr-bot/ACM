<#
.SYNOPSIS
    ACM Observability Stack Setup Script
    
.DESCRIPTION
    This script sets up the complete observability stack for ACM:
    1. Restarts Docker containers with proper permissions
    2. Installs Python profiling packages
    3. Sets up environment variables
    4. Validates all connections
    
.NOTES
    Run as Administrator for full functionality
    
.EXAMPLE
    .\scripts\setup_observability.ps1
#>

param(
    [switch]$UpgradeContainers,
    [switch]$SkipPythonPackages,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$ACM_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$OBSERVABILITY_DIR = Join-Path $ACM_ROOT "install\observability"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " ACM Observability Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# 1. Check Docker is running
# =============================================================================
Write-Host "[1/6] Checking Docker..." -ForegroundColor Yellow

try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is not running"
    }
    Write-Host "  Docker is running" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# =============================================================================
# 2. Fix Pyroscope permissions and restart containers
# =============================================================================
Write-Host ""
Write-Host "[2/6] Fixing container volumes and permissions..." -ForegroundColor Yellow

Push-Location $OBSERVABILITY_DIR

# Stop existing containers
Write-Host "  Stopping existing containers..."
docker-compose down 2>&1 | Out-Null

# Remove old volumes with permission issues
Write-Host "  Removing old volumes..."
docker volume rm observability_pyroscope-data 2>&1 | Out-Null
docker volume rm observability_tempo-data 2>&1 | Out-Null
docker volume rm observability_loki-data 2>&1 | Out-Null
docker volume rm observability_prometheus-data 2>&1 | Out-Null

# Create fresh volumes
Write-Host "  Creating fresh volumes..."
docker volume create observability_pyroscope-data 2>&1 | Out-Null
docker volume create observability_tempo-data 2>&1 | Out-Null
docker volume create observability_loki-data 2>&1 | Out-Null
docker volume create observability_prometheus-data 2>&1 | Out-Null

# Start containers
Write-Host "  Starting containers..."
docker-compose up -d 2>&1 | Out-Null

Pop-Location

# Wait for containers to be healthy
Write-Host "  Waiting for containers to become healthy..."
Start-Sleep -Seconds 10

$containers = @("acm-tempo", "acm-pyroscope", "acm-loki", "acm-prometheus")
foreach ($container in $containers) {
    $status = docker inspect --format='{{.State.Health.Status}}' $container 2>&1
    if ($status -eq "healthy") {
        Write-Host "    $container : HEALTHY" -ForegroundColor Green
    } else {
        Write-Host "    $container : $status" -ForegroundColor Yellow
    }
}

# =============================================================================
# 3. Install Python packages for profiling
# =============================================================================
Write-Host ""
Write-Host "[3/6] Installing Python profiling packages..." -ForegroundColor Yellow

if (-not $SkipPythonPackages) {
    # Note: pyroscope-io requires Rust toolchain on Windows
    # We'll try to install it, and if it fails, provide instructions
    
    Write-Host "  Checking for pyroscope-io..."
    $pyroscopeInstalled = python -c "import pyroscope; print('ok')" 2>&1
    
    if ($pyroscopeInstalled -ne "ok") {
        Write-Host "  pyroscope-io not installed. Attempting install..." -ForegroundColor Yellow
        
        # Try installing from wheel first
        $result = pip install pyroscope-io 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "  NOTE: pyroscope-io requires Rust toolchain to build on Windows." -ForegroundColor Yellow
            Write-Host "  To install Rust:" -ForegroundColor Yellow
            Write-Host "    1. Download from https://rustup.rs" -ForegroundColor Cyan
            Write-Host "    2. Run: rustup-init.exe" -ForegroundColor Cyan
            Write-Host "    3. Restart terminal and run: pip install pyroscope-io" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "  Alternative: Using py-spy for external profiling..." -ForegroundColor Yellow
            pip install py-spy 2>&1 | Out-Null
        } else {
            Write-Host "  pyroscope-io installed successfully" -ForegroundColor Green
        }
    } else {
        Write-Host "  pyroscope-io already installed" -ForegroundColor Green
    }
    
    # Install other observability packages
    Write-Host "  Installing other observability packages..."
    pip install --quiet opentelemetry-api opentelemetry-sdk 2>&1 | Out-Null
    pip install --quiet opentelemetry-exporter-otlp-proto-http 2>&1 | Out-Null
    pip install --quiet opentelemetry-instrumentation 2>&1 | Out-Null
    pip install --quiet structlog 2>&1 | Out-Null
    Write-Host "  OpenTelemetry packages installed" -ForegroundColor Green
} else {
    Write-Host "  Skipping Python packages (--SkipPythonPackages)" -ForegroundColor Yellow
}

# =============================================================================
# 4. Set up environment variables (user-level)
# =============================================================================
Write-Host ""
Write-Host "[4/6] Setting up environment variables..." -ForegroundColor Yellow

$envVars = @{
    "OTEL_EXPORTER_OTLP_ENDPOINT" = "http://localhost:4318"
    "OTEL_SERVICE_NAME" = "acm-pipeline"
    "ACM_PYROSCOPE_ENDPOINT" = "http://localhost:4040"
    "ACM_PROFILING_ENABLED" = "true"
    "ACM_LOG_FORMAT" = "json"
    "ACM_LOG_LEVEL" = "INFO"
}

foreach ($key in $envVars.Keys) {
    $value = $envVars[$key]
    [Environment]::SetEnvironmentVariable($key, $value, "User")
    # Also set for current session
    Set-Item -Path "env:$key" -Value $value
    Write-Host "  $key = $value" -ForegroundColor Gray
}

Write-Host "  Environment variables set (User level)" -ForegroundColor Green

# =============================================================================
# 5. Validate connections
# =============================================================================
Write-Host ""
Write-Host "[5/6] Validating connections..." -ForegroundColor Yellow

$endpoints = @{
    "Tempo" = "http://localhost:3200/ready"
    "Pyroscope" = "http://localhost:4040/ready"
    "Loki" = "http://localhost:3100/ready"
    "Prometheus" = "http://localhost:9090/-/ready"
    "Alloy OTLP" = "http://localhost:4318/v1/traces"
}

foreach ($name in $endpoints.Keys) {
    $url = $endpoints[$name]
    try {
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200 -or $response.StatusCode -eq 204 -or $response.StatusCode -eq 405) {
            Write-Host "  $name : OK" -ForegroundColor Green
        } else {
            Write-Host "  $name : HTTP $($response.StatusCode)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  $name : NOT REACHABLE" -ForegroundColor Red
    }
}

# =============================================================================
# 6. Create startup helper script
# =============================================================================
Write-Host ""
Write-Host "[6/6] Creating helper scripts..." -ForegroundColor Yellow

$helperScript = @'
# Quick ACM run with observability
# Usage: .\run_acm_with_observability.ps1 -Equipment WFA_TURBINE_10

param(
    [Parameter(Mandatory=$true)]
    [string]$Equipment,
    [int]$MaxBatches = 1
)

# Environment is already set up via setup_observability.ps1
# Just run the batch
cd "$PSScriptRoot\.."
python scripts/sql_batch_runner.py --equip $Equipment --max-batches $MaxBatches
'@

$helperPath = Join-Path $ACM_ROOT "scripts\run_acm_with_observability.ps1"
$helperScript | Out-File -FilePath $helperPath -Encoding UTF8
Write-Host "  Created: scripts\run_acm_with_observability.ps1" -ForegroundColor Green

# =============================================================================
# Summary
# =============================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Endpoints:" -ForegroundColor Yellow
Write-Host "  Grafana:    http://localhost:3000" -ForegroundColor Gray
Write-Host "  Tempo:      http://localhost:3200" -ForegroundColor Gray
Write-Host "  Pyroscope:  http://localhost:4040" -ForegroundColor Gray
Write-Host "  Loki:       http://localhost:3100" -ForegroundColor Gray
Write-Host "  Prometheus: http://localhost:9090" -ForegroundColor Gray
Write-Host "  Alloy OTLP: http://localhost:4318" -ForegroundColor Gray
Write-Host ""
Write-Host "To run ACM with full observability:" -ForegroundColor Yellow
Write-Host "  python scripts/sql_batch_runner.py --equip WFA_TURBINE_10 --max-batches 1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Environment variables are now set at User level." -ForegroundColor Green
Write-Host "You may need to restart your terminal for changes to take effect." -ForegroundColor Yellow
