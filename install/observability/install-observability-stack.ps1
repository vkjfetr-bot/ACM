<# 
.SYNOPSIS
    Install Grafana Observability Stack for ACM on Windows
    
.DESCRIPTION
    Downloads and configures:
    - Grafana Tempo (distributed tracing)
    - Grafana Pyroscope (continuous profiling)
    - Grafana Alloy config (OTLP collector - assumes already installed)
    - Grafana Loki (log aggregation - optional, user installs separately)
    
.NOTES
    Run as Administrator if installing to Program Files
    Assumes Grafana and Alloy are already installed
    
.EXAMPLE
    .\install-observability-stack.ps1
    .\install-observability-stack.ps1 -InstallDir "C:\grafana-stack"
#>

param(
    [string]$InstallDir = "$env:USERPROFILE\grafana-stack",
    [switch]$SkipDownload,
    [switch]$StartServices
)

$ErrorActionPreference = "Stop"

# Versions - Tempo doesn't have Windows binaries, Pyroscope does as tar.gz
$TempoVersion = "2.9.0"
$PyroscopeVersion = "1.16.0"
$LokiVersion = "3.3.2"

# URLs - Note: Tempo has no Windows builds, must use Docker or WSL
# Pyroscope and Loki have Linux builds only
$TempoUrl = "https://github.com/grafana/tempo/releases/download/v$TempoVersion/tempo_${TempoVersion}_linux_amd64.tar.gz"
$PyroscopeUrl = "https://github.com/grafana/pyroscope/releases/download/v$PyroscopeVersion/pyroscope_${PyroscopeVersion}_linux_amd64.tar.gz"
$LokiUrl = "https://github.com/grafana/loki/releases/download/v$LokiVersion/loki-windows-amd64.exe.zip"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ACM Observability Stack Installer" -ForegroundColor Cyan
Write-Host "  Tempo v$TempoVersion + Pyroscope v$PyroscopeVersion" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Create install directory
Write-Host "[1/5] Creating installation directory: $InstallDir" -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\tempo" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\tempo\data" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\pyroscope" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\pyroscope\data" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\config" | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\logs" | Out-Null

if (-not $SkipDownload) {
    # Download Tempo
    Write-Host "[2/5] Downloading Grafana Tempo v$TempoVersion..." -ForegroundColor Yellow
    $tempoZip = "$InstallDir\tempo.zip"
    try {
        Invoke-WebRequest -Uri $TempoUrl -OutFile $tempoZip -UseBasicParsing
        Expand-Archive -Path $tempoZip -DestinationPath "$InstallDir\tempo" -Force
        Remove-Item $tempoZip
        Write-Host "      Tempo downloaded and extracted" -ForegroundColor Green
    } catch {
        Write-Host "      Failed to download Tempo: $_" -ForegroundColor Red
        Write-Host "      Manual download: $TempoUrl" -ForegroundColor Yellow
    }

    # Download Pyroscope
    Write-Host "[3/5] Downloading Grafana Pyroscope v$PyroscopeVersion..." -ForegroundColor Yellow
    $pyroscopeZip = "$InstallDir\pyroscope.zip"
    try {
        Invoke-WebRequest -Uri $PyroscopeUrl -OutFile $pyroscopeZip -UseBasicParsing
        Expand-Archive -Path $pyroscopeZip -DestinationPath "$InstallDir\pyroscope" -Force
        Remove-Item $pyroscopeZip
        Write-Host "      Pyroscope downloaded and extracted" -ForegroundColor Green
    } catch {
        Write-Host "      Failed to download Pyroscope: $_" -ForegroundColor Red
        Write-Host "      Manual download: $PyroscopeUrl" -ForegroundColor Yellow
    }
} else {
    Write-Host "[2/5] Skipping downloads (--SkipDownload)" -ForegroundColor Yellow
    Write-Host "[3/5] Skipping downloads (--SkipDownload)" -ForegroundColor Yellow
}

# Create Tempo config
Write-Host "[4/5] Creating configuration files..." -ForegroundColor Yellow

$tempoConfig = @"
# Tempo Configuration for ACM
# Minimal local config - stores traces in local filesystem

server:
  http_listen_port: 3200
  grpc_listen_port: 9095

distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317
        http:
          endpoint: 0.0.0.0:4318

ingester:
  max_block_duration: 5m

compactor:
  compaction:
    block_retention: 168h  # 7 days

storage:
  trace:
    backend: local
    local:
      path: $($InstallDir -replace '\\', '/')/tempo/data
    wal:
      path: $($InstallDir -replace '\\', '/')/tempo/data/wal

metrics_generator:
  registry:
    external_labels:
      source: tempo
      cluster: acm-local
  storage:
    path: $($InstallDir -replace '\\', '/')/tempo/data/generator/wal
    remote_write: []
"@

$tempoConfig | Out-File -FilePath "$InstallDir\config\tempo.yaml" -Encoding UTF8
Write-Host "      Created tempo.yaml" -ForegroundColor Green

# Create Pyroscope config
$pyroscopeConfig = @"
# Pyroscope Configuration for ACM
# Minimal local config - stores profiles in local filesystem

target: all

server:
  http_listen_port: 4040
  grpc_listen_port: 4041

storage:
  backend: filesystem
  filesystem:
    dir: $($InstallDir -replace '\\', '/')/pyroscope/data

self-profiling:
  disable-push: true

scrape-configs: []
"@

$pyroscopeConfig | Out-File -FilePath "$InstallDir\config\pyroscope.yaml" -Encoding UTF8
Write-Host "      Created pyroscope.yaml" -ForegroundColor Green

# Create Alloy config for ACM
$alloyConfig = @"
// Grafana Alloy Configuration for ACM Observability Stack
// Receives OTLP from Python apps and forwards to Tempo/Loki/Pyroscope

// ============================================================================
// OTLP Receiver - Accepts traces and metrics from Python OpenTelemetry SDK
// ============================================================================
otelcol.receiver.otlp "default" {
  grpc {
    endpoint = "0.0.0.0:4317"
  }
  http {
    endpoint = "0.0.0.0:4318"
  }

  output {
    traces  = [otelcol.processor.batch.default.input]
    metrics = [otelcol.processor.batch.default.input]
    logs    = [otelcol.processor.batch.default.input]
  }
}

// ============================================================================
// Batch Processor - Batches telemetry before export
// ============================================================================
otelcol.processor.batch "default" {
  timeout = "5s"
  send_batch_size = 1000

  output {
    traces  = [otelcol.exporter.otlp.tempo.input]
    metrics = [otelcol.exporter.prometheus.local.input]
    logs    = [otelcol.exporter.loki.local.input]
  }
}

// ============================================================================
// Tempo Exporter - Sends traces to Grafana Tempo
// ============================================================================
otelcol.exporter.otlp "tempo" {
  client {
    endpoint = "localhost:4317"
    tls {
      insecure = true
    }
  }
}

// ============================================================================
// Prometheus Exporter - Exposes metrics for Grafana scraping
// ============================================================================
otelcol.exporter.prometheus "local" {
  forward_to = [prometheus.remote_write.local.receiver]
}

prometheus.remote_write "local" {
  // If you have Mimir/Prometheus, configure endpoint here
  // For now, just expose metrics at /metrics
  endpoint {
    url = "http://localhost:9090/api/v1/write"
    // Disable if no Prometheus/Mimir available
    send_exemplars = false
  }
}

// ============================================================================
// Loki Exporter - Sends logs to Grafana Loki
// ============================================================================
otelcol.exporter.loki "local" {
  forward_to = [loki.write.local.receiver]
}

loki.write "local" {
  endpoint {
    url = "http://localhost:3100/loki/api/v1/push"
  }
}
"@

$alloyConfig | Out-File -FilePath "$InstallDir\config\alloy.config" -Encoding UTF8
Write-Host "      Created alloy.config" -ForegroundColor Green

# Create start scripts
Write-Host "[5/5] Creating start/stop scripts..." -ForegroundColor Yellow

$startScript = @"
@echo off
echo Starting ACM Observability Stack...
echo.

echo Starting Tempo (port 3200, OTLP 4317/4318)...
start "Tempo" /MIN cmd /c "$InstallDir\tempo\tempo-windows-amd64.exe -config.file=$InstallDir\config\tempo.yaml"

timeout /t 2 /nobreak > nul

echo Starting Pyroscope (port 4040)...
start "Pyroscope" /MIN cmd /c "$InstallDir\pyroscope\pyroscope.exe -config.file=$InstallDir\config\pyroscope.yaml"

echo.
echo Observability stack started!
echo.
echo   Tempo:     http://localhost:3200
echo   Pyroscope: http://localhost:4040
echo   OTLP gRPC: localhost:4317
echo   OTLP HTTP: localhost:4318
echo.
echo Add these as datasources in Grafana.
"@

$startScript | Out-File -FilePath "$InstallDir\start-stack.bat" -Encoding ASCII
Write-Host "      Created start-stack.bat" -ForegroundColor Green

$stopScript = @"
@echo off
echo Stopping ACM Observability Stack...
taskkill /IM tempo-windows-amd64.exe /F 2>nul
taskkill /IM pyroscope.exe /F 2>nul
echo Stack stopped.
"@

$stopScript | Out-File -FilePath "$InstallDir\stop-stack.bat" -Encoding ASCII
Write-Host "      Created stop-stack.bat" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Installation Directory: $InstallDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the stack:" -ForegroundColor Yellow
Write-Host "  $InstallDir\start-stack.bat" -ForegroundColor White
Write-Host ""
Write-Host "To stop the stack:" -ForegroundColor Yellow
Write-Host "  $InstallDir\stop-stack.bat" -ForegroundColor White
Write-Host ""
Write-Host "Endpoints:" -ForegroundColor Yellow
Write-Host "  Tempo API:     http://localhost:3200" -ForegroundColor White
Write-Host "  Pyroscope UI:  http://localhost:4040" -ForegroundColor White
Write-Host "  OTLP gRPC:     localhost:4317 (for Python SDK)" -ForegroundColor White
Write-Host "  OTLP HTTP:     localhost:4318 (for Python SDK)" -ForegroundColor White
Write-Host ""
Write-Host "Grafana Datasources to add:" -ForegroundColor Yellow
Write-Host "  1. Tempo:     http://localhost:3200" -ForegroundColor White
Write-Host "  2. Pyroscope: http://localhost:4040" -ForegroundColor White
Write-Host "  3. Loki:      http://localhost:3100 (install separately)" -ForegroundColor White
Write-Host ""
Write-Host "Python environment variables:" -ForegroundColor Yellow
Write-Host '  $env:OTEL_EXPORTER_OTLP_ENDPOINT = "http://localhost:4318"' -ForegroundColor White
Write-Host '  $env:ACM_PYROSCOPE_ENDPOINT = "http://localhost:4040"' -ForegroundColor White
Write-Host ""

if ($StartServices) {
    Write-Host "Starting services..." -ForegroundColor Yellow
    & "$InstallDir\start-stack.bat"
}
