Param(
  [string]$Equip = "FD FAN",
  [string]$ArtDir = "$PSScriptRoot\acm_artifacts",
  [string]$Scores = "$PSScriptRoot\acm_artifacts\scores.csv",
  [string]$Events = "$PSScriptRoot\acm_artifacts\events.jsonl",
  [int]$TopTags = 9,
  [switch]$NoMatrix,
  [switch]$NoAttention,
  [switch]$NoLatent,
  [int]$Seed = 42
)

$py = "python"
$script = Join-Path $PSScriptRoot "next\build_report.py"

Write-Host "[run_acm_next] Build report for '$Equip'"

$args = @(
  "build-report",
  "--equip", $Equip,
  "--art-dir", $ArtDir,
  "--top-tags", $TopTags,
  "--scores", $Scores,
  "--events-json", $Events,
  "--seed", $Seed
)
if ($NoMatrix)   { $args += "--no-matrix" }
if ($NoAttention){ $args += "--no-attention" }
if ($NoLatent)   { $args += "--no-latent" }

& $py $script @args

