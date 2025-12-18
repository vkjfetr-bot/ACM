<#
.SYNOPSIS
    Configure Grafana Alloy for ACM Observability Stack
    
.DESCRIPTION
    Copies the ACM Alloy config to the Grafana Alloy installation
    and restarts the Alloy service.
    
    REQUIRES: Run as Administrator
    
.EXAMPLE
    # Run PowerShell as Administrator, then:
    .\configure-alloy.ps1
#>

$ErrorActionPreference = "Stop"

$AlloyInstallPath = "C:\Program Files\GrafanaLabs\Alloy"
$AlloyConfigSource = Join-Path $PSScriptRoot "config.alloy"
$AlloyConfigDest = Join-Path $AlloyInstallPath "config.alloy"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Configure Alloy for ACM Observability" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[ERROR] This script must be run as Administrator!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Check Alloy installation
if (-not (Test-Path $AlloyInstallPath)) {
    Write-Host "[ERROR] Alloy not found at: $AlloyInstallPath" -ForegroundColor Red
    exit 1
}

Write-Host "[1/3] Backing up existing config..." -ForegroundColor Yellow
$backupPath = Join-Path $AlloyInstallPath "config.alloy.backup"
if (Test-Path $AlloyConfigDest) {
    Copy-Item $AlloyConfigDest $backupPath -Force
    Write-Host "      Backed up to: $backupPath" -ForegroundColor Green
}

Write-Host "[2/3] Copying ACM config..." -ForegroundColor Yellow
Copy-Item $AlloyConfigSource $AlloyConfigDest -Force
Write-Host "      Config copied to: $AlloyConfigDest" -ForegroundColor Green

Write-Host "[3/3] Restarting Alloy service..." -ForegroundColor Yellow
Restart-Service Alloy
Start-Sleep -Seconds 2
$status = (Get-Service Alloy).Status
Write-Host "      Alloy service status: $status" -ForegroundColor Green

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Alloy Configuration Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Alloy is now configured to:" -ForegroundColor Cyan
Write-Host "  - Receive OTLP on localhost:4317 (gRPC) and localhost:4318 (HTTP)"
Write-Host "  - Forward traces to Tempo on localhost:4320"
Write-Host "  - Forward logs to Loki on localhost:3100"
Write-Host ""
Write-Host "Python apps should set:" -ForegroundColor Yellow
Write-Host '  $env:OTEL_EXPORTER_OTLP_ENDPOINT = "http://localhost:4318"'
Write-Host ""
