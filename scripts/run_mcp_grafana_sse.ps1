Param(
    [string]$GrafanaUrl = $env:GRAFANA_URL,
    [string]$GrafanaToken = $env:GRAFANA_SERVICE_ACCOUNT_TOKEN
)

if (-not $GrafanaUrl) {
    Write-Error "GRAFANA_URL is not set. Set it in your environment or pass -GrafanaUrl."
    exit 1
}

if (-not $GrafanaToken) {
    Write-Error "GRAFANA_SERVICE_ACCOUNT_TOKEN is not set. Set it in your environment or pass -GrafanaToken."
    exit 1
}

Write-Host "Starting mcp-grafana SSE server on http://localhost:8000/sse"
Write-Host "Grafana URL: $GrafanaUrl"

docker run --rm -p 8000:8000 `
    -e GRAFANA_URL=$GrafanaUrl `
    -e GRAFANA_SERVICE_ACCOUNT_TOKEN=$GrafanaToken `
    mcp/grafana

