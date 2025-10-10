<#!
run_acmnxt.ps1 - Lightweight orchestrator for ACM_V2 pipeline.

This wrapper runs train -> score (and optionally report) for a single equipment dataset.
Enable the report step with -EnableReport once the HTML builder is implemented.
#   -SplitTrainIntoTests N  -> auto-split train CSV into N score windows
#   -AdditionalScoreCsv ... -> array of extra score CSVs to evaluate
#>

param(
  [string]$Root        = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path,
  [string]$Artifacts   = (Resolve-Path (Join-Path $PSScriptRoot "..\artifacts" )).Path,
  [string]$TrainCsv,
  [string]$ScoreCsv,
  [string[]]$AdditionalScoreCsv = @(),
  [int]$SplitTrainIntoTests = 0,
  [string]$Equip       = "equipment",
  [switch]$SkipTrain,
  [switch]$SkipScore,
  [switch]$EnableReport
)

$ErrorActionPreference = "Stop"
function Die ($m){ Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
function Step($m){ Write-Host "== $m ==" -ForegroundColor Yellow }

if(-not $TrainCsv -and -not $SkipTrain){ Die "Provide -TrainCsv or pass -SkipTrain" }
if(-not $ScoreCsv -and -not $SkipScore -and $SplitTrainIntoTests -le 0 -and $AdditionalScoreCsv.Count -eq 0){
  Die "Provide -ScoreCsv, additional score CSVs, or enable -SplitTrainIntoTests"
}

$Core   = Join-Path $Root "acm_core_local_2.py"
$Report = Join-Path $Root "acm_report_basic.py"
$Splitter = Join-Path $Root "acm_split_train.py"

foreach($p in @($Core)){
  if(!(Test-Path $p)){ Die "Missing $p" }
}
if($EnableReport -and -not (Test-Path $Report)){
  Die "Report script not found (acm_report_basic.py)"
}
if($SplitTrainIntoTests -gt 0 -and -not (Test-Path $Splitter)){
  Die "Split utility not found (acm_split_train.py)"
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
  $scoreRuns = @()
  if($ScoreCsv){
    $scoreRuns += [PSCustomObject]@{ Path = (Resolve-Path $ScoreCsv).Path; Prefix = "acm"; Manifest = "acm"; Label = "Score" }
  }
  if($SplitTrainIntoTests -gt 0){
    Step "Split train into $SplitTrainIntoTests slices"
    $splitDir = Join-Path $equipArtifacts "splits"
    New-Item -ItemType Directory -Force -Path $splitDir | Out-Null
    python $Splitter --csv "$TrainCsv" --splits $SplitTrainIntoTests --out "$splitDir"
    if($LASTEXITCODE){ Die "Split failed" }
    $splitFiles = Get-ChildItem -Path $splitDir -Filter "*.csv" | Sort-Object Name
    $idx = 1
    foreach($f in $splitFiles){
      $scoreRuns += [PSCustomObject]@{
        Path = $f.FullName
        Prefix = "acm_split$idx"
        Manifest = "acm"
        Label = "Score split $idx"
      }
      $idx++
    }
  }
  if($AdditionalScoreCsv.Count -gt 0){
    $base = 1
    foreach($extra in $AdditionalScoreCsv){
      $scoreRuns += [PSCustomObject]@{
        Path = (Resolve-Path $extra).Path
        Prefix = "acm_extra$base"
        Manifest = "acm"
        Label = "Score extra $base"
      }
      $base++
    }
  }
  if($scoreRuns.Count -eq 0){
    Die "No score files resolved; pass -SkipScore if this is intentional."
  }

  foreach($run in $scoreRuns){
    Step $run.Label
    python $Core score --csv "$($run.Path)" --equip "$Equip" --prefix "$($run.Prefix)" --manifest-prefix "$($run.Manifest)"
    if($LASTEXITCODE){ Die "Score failed for $($run.Path)" }
  }
}

if($EnableReport){
  Step "Report"
  python $Report --artifacts "$equipArtifacts" --equip "$Equip"
  if($LASTEXITCODE){ Die "Report failed" }
}

Step "Done"
