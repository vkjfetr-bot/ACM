# MCP Grafana integration (local setup)

This repo includes a local copy of `mcp-grafana` under `tools/mcp-grafana` and a basic VS Code + Docker setup so that MCP-aware assistants can talk to your local Grafana.

## 1. Prerequisites

- Docker is installed and running.
- Grafana is running and reachable from this machine (for local installs this is usually `http://localhost:3000`).
- A Grafana **service account token** with at least viewer/editor permissions for the dashboards you want to work with.

## 2. Start the MCP Grafana server (SSE mode)

From the repo root:

```powershell
$env:GRAFANA_URL = "http://localhost:3000"
$env:GRAFANA_SERVICE_ACCOUNT_TOKEN = "<your-service-account-token>"
.\scripts\run_mcp_grafana_sse.ps1
```

This runs the official `mcp/grafana` Docker image and exposes an MCP server on:

- `http://localhost:8000/sse` (SSE transport)

## 3. VS Code MCP configuration

The file `.vscode/settings.json` already contains:

```jsonc
{
  "mcp": {
    "servers": {
      "grafana": {
        "type": "sse",
        "url": "http://localhost:8000/sse"
      }
    }
  }
}
```

Any VS Code extension that understands the `"mcp.servers"` configuration (for example, MCP-enabled assistants) can now connect to the running `mcp-grafana` server and expose tools/resources like:

- Searching dashboards
- Getting dashboard summaries / JSON
- Updating or patching dashboards

## 4. Using this with Codex / other assistants

- In VS Code, use an MCP-aware AI extension and point it at the `grafana` server defined in `.vscode/settings.json`.
- Make sure the MCP server is running (step 2) before you start asking the assistant things like:
  - "List Grafana dashboards and show summaries."
  - "Load the `asset_health` dashboard and add a new panel with query X against the ACM SQL data."

### Important limitation for this Codex CLI

This Codex CLI environment cannot change its own MCP server configuration; MCP servers are wired in by the host. The setup above is for your local VS Code + AI assistant. Once your VS Code extension connects to the `grafana` server, Codex (or any MCP-aware model you use there) will be able to interact with your local Grafana over MCP.

