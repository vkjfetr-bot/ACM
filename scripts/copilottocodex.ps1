Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ===========================
# STATIC SOURCE (HARD-CODED)
# ===========================
$InstructionsFile = "C:\Users\bhadk\Documents\ACM V8 SQL\ACM\.github\copilot-instructions.md"

# Set $false if you want ONLY instructions and nothing else
$IncludeVSCodeMCP  = $true

# ===========================
# TARGET: CODEX CONFIG
# ===========================
$UserHome = [Environment]::GetFolderPath("UserProfile")
$CodexDir = Join-Path $UserHome ".codex"
$CodexCfg = Join-Path $CodexDir "config.toml"

# Optional: VS Code settings.json (for MCP-like keys)
$VSCodeUserSettings = Join-Path $UserHome "AppData\Roaming\Code\User\settings.json"

# ===========================
# VALIDATION
# ===========================
if (-not (Test-Path $InstructionsFile)) {
    throw "Static instructions file not found: $InstructionsFile"
}

if (-not (Test-Path $CodexDir)) {
    New-Item -ItemType Directory -Path $CodexDir | Out-Null
}

# Backup existing config
if (Test-Path $CodexCfg) {
    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    Copy-Item $CodexCfg "$CodexCfg.bak.$ts"
    Write-Host "Backup created: $CodexCfg.bak.$ts"
}

# ===========================
# READ STATIC INSTRUCTIONS
# ===========================
$InstructionText = Get-Content $InstructionsFile -Raw

# ===========================
# OPTIONAL: CAPTURE MCP-LIKE SETTINGS (LOSSLESS)
# ===========================
$MCPBlocks = @()
if ($IncludeVSCodeMCP -and (Test-Path $VSCodeUserSettings)) {
    try {
        $VSCodeSettings = Get-Content $VSCodeUserSettings -Raw | ConvertFrom-Json

        foreach ($prop in $VSCodeSettings.PSObject.Properties) {
            $k = $prop.Name
            if ($k -match "(?i)copilot|mcp|tool|agent") {
                $json = $prop.Value | ConvertTo-Json -Depth 10 -Compress
                $MCPBlocks += [PSCustomObject]@{ Key = $k; RawJson = $json }
            }
        }
    } catch {
        Write-Host "Warning: Failed to parse VS Code settings.json. MCP extraction skipped. Error: $($_.Exception.Message)"
    }
}

# ===========================
# BUILD CODEX TOML
# ===========================
$toml = New-Object System.Collections.Generic.List[string]
$toml.Add("# ==================================================")
$toml.Add("# AUTO-GENERATED: Static Instructions → Codex")
$toml.Add("# Generated: $(Get-Date -Format u)")
$toml.Add("# Source instructions: $InstructionsFile")
$toml.Add("# ==================================================")
$toml.Add("")

# Permissions (safe default; tighten later if needed)
$toml.Add("[permissions]")
$toml.Add("allow = [""*""]")
$toml.Add("deny  = []")
$toml.Add("")

# Environment
$toml.Add("[env]")
$toml.Add("PYTHONUNBUFFERED = ""1""")
$toml.Add("")

# Instructions
$toml.Add("[instructions]")
$toml.Add("system = '''")
$toml.Add($InstructionText.TrimEnd())
$toml.Add("'''")
$toml.Add("")

# MCP blocks (raw JSON capture)
foreach ($m in $MCPBlocks) {
    $safeKey = ($m.Key -replace "[^a-zA-Z0-9_.-]", "_")
    $toml.Add("[mcp.$safeKey]")
    $toml.Add("raw = '''")
    $toml.Add($m.RawJson)
    $toml.Add("'''")
    $toml.Add("")
}

# ===========================
# WRITE CODEX CONFIG
# ===========================
$tomlText = $toml -join "`n"
Set-Content -Path $CodexCfg -Value $tomlText -Encoding UTF8

Write-Host "Done."
Write-Host "Codex config written to: $CodexCfg"
Write-Host "Instructions imported from: $InstructionsFile"
Write-Host ("MCP-like VS Code settings captured: {0}" -f $MCPBlocks.Count)
