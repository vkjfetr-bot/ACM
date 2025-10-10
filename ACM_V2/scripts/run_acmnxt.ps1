<#!
run_acmnxt.ps1 - Lightweight orchestrator for ACM_V2 pipeline.

This wrapper runs train -> score (and optionally report) for a single equipment dataset.
Enable the report step with -EnableReport once the HTML builder is implemented.
#>

param(
  [string]$Root        = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path,
  [string]$Artifacts   = (Resolve-Path (Join-Path $PSScriptRoot "..\artifacts" )).Path,
  [string]$TrainCsv,
  [string]$ScoreCsv,
  [string]$Equip       = "equipment",
  [switch]$SkipTrain,
  [switch]$SkipScore,
  [switch]$EnableReport
)

$ErrorActionPreference = "Stop"
function Die ($m){ Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
function Step($m){ Write-Host "== $m ==" -ForegroundColor Yellow }

if(-not $TrainCsv -and -not $SkipTrain){ Die "Provide -TrainCsv or pass -SkipTrain" }
if(-not $ScoreCsv -and -not $SkipScore){ Die "Provide -ScoreCsv or pass -SkipScore" }

$Core   = Join-Path $Root "acm_core_local_2.py"
$Report = Join-Path $Root "acm_report_basic.py"

foreach($p in @($Core)){
  if(!(Test-Path $p)){ Die "Missing $p" }
}
if($EnableReport -and -not (Test-Path $Report)){
  Die "Report script not found (acm_report_basic.py)"
}

$equipArtifacts = Join-Path $Artifacts $Equip
New-Item -ItemType Directory -Path $equipArtifacts -Force | Out-Null
$env:ACM_ART_DIR = $equipArtifacts
$env:ACM_EQUIP = $Equip

if(-not $SkipTrain){
  Step "Train"
  python $Core train --csv "$TrainCsv" --equip "$Equip"
  if($LASTEXITCODE){ Die "Train failed" }
}

if(-not $SkipScore){
  Step "Score"
  python $Core score --csv "$ScoreCsv" --equip "$Equip"
  if($LASTEXITCODE){ Die "Score failed" }
}

if($EnableReport){
  Step "Report"
  python $Report --artifacts "$equipArtifacts" --equip "$Equip"
  if($LASTEXITCODE){ Die "Report failed" }
}

Step "Done"
