<#
.SYNOPSIS
    Configure Grafana datasources for ACM Observability Stack

.DESCRIPTION
    Adds Tempo, Pyroscope, and Loki datasources to Grafana via API

.PARAMETER GrafanaUrl
    Grafana base URL (default: http://localhost:3000)

.PARAMETER Username  
    Grafana admin username (default: admin)

.PARAMETER Password
    Grafana admin password

.EXAMPLE
    .\configure-grafana-datasources.ps1 -Password "admin"
#>

param(
    [string]$GrafanaUrl = "http://localhost:3000",
    [string]$Username = "admin",
    [Parameter(Mandatory=$true)]
    [string]$Password
)

$ErrorActionPreference = "Stop"

# Build auth header
$pair = "${Username}:${Password}"
$bytes = [System.Text.Encoding]::ASCII.GetBytes($pair)
$base64 = [System.Convert]::ToBase64String($bytes)
$headers = @{
    "Authorization" = "Basic $base64"
    "Content-Type" = "application/json"
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ACM Observability - Grafana Datasources" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Grafana health
try {
    $health = Invoke-RestMethod -Uri "$GrafanaUrl/api/health" -Method Get
    Write-Host "[OK] Grafana is healthy (v$($health.version))" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Grafana not reachable at $GrafanaUrl" -ForegroundColor Red
    exit 1
}

# Function to create or update datasource
function Set-Datasource {
    param($Name, $Body)
    
    try {
        # Check if exists
        $existing = $null
        try {
            $existing = Invoke-RestMethod -Uri "$GrafanaUrl/api/datasources/name/$Name" -Headers $headers -Method Get
        } catch {}
        
        if ($existing) {
            # Update
            $result = Invoke-RestMethod -Uri "$GrafanaUrl/api/datasources/$($existing.id)" -Headers $headers -Method Put -Body ($Body | ConvertTo-Json -Depth 10)
            Write-Host "[UPDATED] $Name datasource" -ForegroundColor Yellow
        } else {
            # Create
            $result = Invoke-RestMethod -Uri "$GrafanaUrl/api/datasources" -Headers $headers -Method Post -Body ($Body | ConvertTo-Json -Depth 10)
            Write-Host "[CREATED] $Name datasource" -ForegroundColor Green
        }
        return $true
    } catch {
        Write-Host "[ERROR] Failed to configure $Name : $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# 1. Tempo datasource
Write-Host ""
Write-Host "Configuring Tempo (Distributed Tracing)..." -ForegroundColor White
$tempo = @{
    name = "Tempo"
    type = "tempo"
    url = "http://localhost:3200"
    access = "proxy"
    isDefault = $false
    uid = "tempo-ds"
    jsonData = @{
        httpMethod = "GET"
        nodeGraph = @{ enabled = $true }
        tracesToLogsV2 = @{
            datasourceUid = "loki-ds"
            spanStartTimeShift = "-1h"
            spanEndTimeShift = "1h"
            filterByTraceID = $true
        }
        tracesToProfiles = @{
            datasourceUid = "pyroscope-ds"
            profileTypeId = "process_cpu:cpu:nanoseconds:cpu:nanoseconds"
        }
    }
}
Set-Datasource -Name "Tempo" -Body $tempo

# 2. Pyroscope datasource
Write-Host ""
Write-Host "Configuring Pyroscope (Continuous Profiling)..." -ForegroundColor White
$pyroscope = @{
    name = "Pyroscope"
    type = "grafana-pyroscope-datasource"
    url = "http://localhost:4040"
    access = "proxy"
    isDefault = $false
    uid = "pyroscope-ds"
    jsonData = @{}
}
Set-Datasource -Name "Pyroscope" -Body $pyroscope

# 3. Loki datasource
Write-Host ""
Write-Host "Configuring Loki (Log Aggregation)..." -ForegroundColor White
$loki = @{
    name = "Loki"
    type = "loki"
    url = "http://localhost:3100"
    access = "proxy"
    isDefault = $false
    uid = "loki-ds"
    jsonData = @{
        maxLines = 1000
    }
}
Set-Datasource -Name "Loki" -Body $loki

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Datasource configuration complete!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Grafana Explore URLs:" -ForegroundColor White
Write-Host "  Traces: $GrafanaUrl/explore?orgId=1&left=[%22now-1h%22,%22now%22,%22Tempo%22]" -ForegroundColor Gray
Write-Host "  Logs:   $GrafanaUrl/explore?orgId=1&left=[%22now-1h%22,%22now%22,%22Loki%22]" -ForegroundColor Gray
Write-Host "  Profiles: $GrafanaUrl/explore?orgId=1&left=[%22now-1h%22,%22now%22,%22Pyroscope%22]" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan
