# Configure Grafana MSSQL Datasource for ACM
# =============================================
# This script configures the MSSQL datasource in Grafana running in Docker
# to connect to SQL Server on the Windows host.
#
# Prerequisites:
#   1. SQL Server must be listening on TCP/IP (port 1433)
#   2. Either:
#      a) Create a SQL Server login with appropriate permissions, OR
#      b) Use Windows Authentication via Kerberos (complex in Docker)
#
# Usage:
#   .\configure_grafana_mssql.ps1 -User "grafana_user" -Password "SecurePassword123"
#
# For Windows Authentication (not recommended from Docker):
#   .\configure_grafana_mssql.ps1 -WindowsAuth

param(
    [string]$GrafanaUrl = "http://localhost:3000",
    [string]$GrafanaUser = "admin",
    [string]$GrafanaPassword = "admin",
    [string]$SqlServer = "host.docker.internal",
    [int]$SqlPort = 1433,
    [string]$Database = "ACM",
    [string]$User = "",
    [string]$Password = "",
    [switch]$WindowsAuth
)

# Load SQL connection details from config if not provided
$configPath = Join-Path $PSScriptRoot "..\configs\sql_connection.ini"
if (Test-Path $configPath) {
    Write-Host "[INFO] Loading SQL connection config from $configPath" -ForegroundColor Cyan
    $configContent = Get-Content $configPath -Raw
    if ($configContent -match "database=(\w+)") {
        $Database = $Matches[1]
        Write-Host "[INFO] Using database: $Database" -ForegroundColor Green
    }
}

# Determine authentication type
if ($WindowsAuth) {
    $authType = "Windows Authentication"
    Write-Host "[WARN] Windows Authentication from Docker containers is complex and may not work." -ForegroundColor Yellow
    Write-Host "       Consider using SQL Server Authentication instead." -ForegroundColor Yellow
} else {
    $authType = "SQL Server Authentication"
    if (-not $User -or -not $Password) {
        Write-Host "[ERROR] For SQL Server Authentication, provide -User and -Password" -ForegroundColor Red
        Write-Host ""
        Write-Host "Example:" -ForegroundColor Yellow
        Write-Host "  .\configure_grafana_mssql.ps1 -User 'grafana_reader' -Password 'YourPassword123'" -ForegroundColor White
        Write-Host ""
        Write-Host "To create a SQL login for Grafana, run in SSMS:" -ForegroundColor Yellow
        Write-Host @"
  USE master;
  CREATE LOGIN grafana_reader WITH PASSWORD = 'YourPassword123';
  USE ACM;
  CREATE USER grafana_reader FOR LOGIN grafana_reader;
  ALTER ROLE db_datareader ADD MEMBER grafana_reader;
"@ -ForegroundColor White
        exit 1
    }
}

# Create Base64 auth header
$pair = "$($GrafanaUser):$($GrafanaPassword)"
$bytes = [System.Text.Encoding]::ASCII.GetBytes($pair)
$base64 = [System.Convert]::ToBase64String($bytes)

$headers = @{
    Authorization = "Basic $base64"
    "Content-Type" = "application/json"
}

# Check if MSSQL datasource already exists
Write-Host "[INFO] Checking for existing MSSQL datasource..." -ForegroundColor Cyan
try {
    $existing = Invoke-RestMethod -Uri "$GrafanaUrl/api/datasources/uid/mssql-ds" -Headers $headers -Method GET -ErrorAction SilentlyContinue
    $existingId = $existing.id
    Write-Host "[INFO] Found existing MSSQL datasource (ID: $existingId)" -ForegroundColor Green
} catch {
    $existingId = $null
    Write-Host "[INFO] No existing MSSQL datasource found" -ForegroundColor Yellow
}

# Build datasource configuration
$datasource = @{
    name = "MSSQL"
    type = "mssql"
    access = "proxy"
    url = "${SqlServer}:${SqlPort}"
    uid = "mssql-ds"
    jsonData = @{
        authenticationType = $authType
        database = $Database
        encrypt = "false"
        connMaxLifetime = 14400
        maxIdleConns = 100
        maxOpenConns = 100
    }
    isDefault = $false
}

if (-not $WindowsAuth) {
    $datasource["user"] = $User
    $datasource["secureJsonData"] = @{
        password = $Password
    }
}

$body = $datasource | ConvertTo-Json -Depth 5

# Create or update datasource
if ($existingId) {
    Write-Host "[INFO] Updating existing MSSQL datasource..." -ForegroundColor Cyan
    $datasource["id"] = $existingId
    $body = $datasource | ConvertTo-Json -Depth 5
    try {
        $result = Invoke-RestMethod -Uri "$GrafanaUrl/api/datasources/$existingId" -Headers $headers -Method PUT -Body $body
        Write-Host "[OK] MSSQL datasource updated successfully" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Failed to update datasource: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[INFO] Creating new MSSQL datasource..." -ForegroundColor Cyan
    try {
        $result = Invoke-RestMethod -Uri "$GrafanaUrl/api/datasources" -Headers $headers -Method POST -Body $body
        Write-Host "[OK] MSSQL datasource created successfully (ID: $($result.datasource.id))" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Failed to create datasource: $_" -ForegroundColor Red
        exit 1
    }
}

# Test datasource connection
Write-Host "[INFO] Testing datasource connection..." -ForegroundColor Cyan
try {
    $testResult = Invoke-RestMethod -Uri "$GrafanaUrl/api/datasources/uid/mssql-ds/health" -Headers $headers -Method GET
    if ($testResult.status -eq "OK") {
        Write-Host "[OK] Connection test successful!" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Connection test returned: $($testResult.message)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARN] Connection test failed: $_" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Troubleshooting tips:" -ForegroundColor Yellow
    Write-Host "  1. Ensure SQL Server is listening on TCP/IP port $SqlPort" -ForegroundColor White
    Write-Host "  2. Enable TCP/IP in SQL Server Configuration Manager" -ForegroundColor White
    Write-Host "  3. Check Windows Firewall allows port $SqlPort" -ForegroundColor White
    Write-Host "  4. Verify the SQL login credentials are correct" -ForegroundColor White
}

Write-Host ""
Write-Host "[DONE] MSSQL datasource configuration complete" -ForegroundColor Green
Write-Host "       Access Grafana at $GrafanaUrl" -ForegroundColor Cyan
