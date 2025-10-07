param(
  [Parameter(Mandatory=$true)][string]$TrainCsv,
  [Parameter(Mandatory=$true)][string]$TestCsv,
  [Parameter(Mandatory=$true)][string]$Equip,
  [string]$Out = ".\reports",
  [string]$Config = "report_config.yaml"
)
$ErrorActionPreference = "Stop"
function Step($m){ Write-Host "== $m ==" -ForegroundColor Yellow }
function Ok($m){ Write-Host $m -ForegroundColor Green }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Step "Build ACM report"
python -m acm.report.cli build --train "$TrainCsv" --test "$TestCsv" --equip "$Equip" --out "$Out" --config "$Config"
Ok "Report ready: $Out"

