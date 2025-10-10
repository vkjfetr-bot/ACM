# run_acm.ps1 — ACM pipeline wrapper
#
# Purpose
# - Orchestrates the ACM analysis pipeline: train → score → drift → aggregate → HTML report → LLM brief/prompt → archive outputs.
#
# Requirements
# - PowerShell 5.1+ (Windows) or PowerShell 7+
# - Python 3.9+ with project dependencies installed (numpy, pandas, scikit-learn, joblib, etc.)
# - Pipeline files under `-Root`: `acm_core_local_2.py`, `acm_score_local_2.py`, `report_main.py`, `acm_brief_local.py`
#
# Key Paths
# - `-Root` points to the ACM code directory (contains the Python scripts listed above)
# - `-Artifacts` is where outputs are written (models, scored window, events, drift, report, brief)
#
# Usage (single equipment)
#   powershell -ExecutionPolicy Bypass -File ACM/run_acm.ps1 `
#     -Root      "C:\\Users\\bhadk\\Documents\\CPCL\\ACM\\src" `
#     -Artifacts "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts" `
#     -TrainCsv  "...\Dummy Data\FD FAN TRAINING DATA.csv" `
#     -TestCsv   "...\Dummy Data\FD FAN TEST DATA.csv" `
#     -Equip     "FD FAN" `
#     -ForceTrain          # optional: retrain even if artifacts exist
#     #-NoBrief            # optional: skip LLM brief/prompt
#
# Usage (batch mode)
#   powershell -ExecutionPolicy Bypass -File ACM/run_acm.ps1 `
#     -Root      "C:\\Users\\bhadk\\Documents\\CPCL\\ACM\\src" `
#     -Artifacts "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts" `
#     -All
#   Scans `Dummy Data\* TRAINING DATA.csv` and pairs with `* TEST DATA.csv`.
#
# Notes
# - Pass `-ForceTrain` when switching equipment to avoid tag/manifest mismatches.
# - Gas Turbine demo CSVs include header rows and a non-time first column; use cleaned files in:
#   `ACM/Dummy Data/clean/` (TRAINING/TEST) or pre-clean similar inputs.
# - The script sets `ACM_ART_DIR` for the HTML report builder.
#
# Examples
# - FD Fan (train + score + drift)
#   powershell -ExecutionPolicy Bypass -File ACM/run_acm.ps1 `
#     -Root      "C:\\Users\\bhadk\\Documents\\CPCL\\ACM\\src" `
#     -Artifacts "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts" `
#     -TrainCsv  "C:\\Users\\bhadk\\Documents\\CPCL\\ACM\\Dummy Data\FD FAN TRAINING DATA.csv" `
#     -TestCsv   "C:\\Users\\bhadk\\Documents\\CPCL\\ACM\\Dummy Data\FD FAN TEST DATA.csv" `
#     -Equip     "FD FAN" `
#     -ForceTrain
#
# - Gas Turbine (use cleaned CSVs)
#   powershell -ExecutionPolicy Bypass -File ACM/run_acm.ps1 `
#     -Root      "C:\\Users\\bhadk\\Documents\\CPCL\\ACM\\src" `
#     -Artifacts "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts" `
#     -TrainCsv  "C:\\Users\\bhadk\\Documents\\CPCL\\ACM\\Dummy Data\clean\Gas Turbine TRAINING DATA.csv" `
#     -TestCsv   "C:\\Users\\bhadk\\Documents\\CPCL\\ACM\\Dummy Data\clean\Gas Turbine TEST DATA.csv" `
#     -Equip     "Gas Turbine" `
#     -ForceTrain
#
# - Batch mode (all equipment in Dummy Data)
#   powershell -ExecutionPolicy Bypass -File ACM/run_acm.ps1 `
#     -Root      "C:\\Users\\bhadk\\Documents\\CPCL\\ACM\\src" `
#     -Artifacts "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts" `
#     -All

param(
  [string]$Root      = "C:\\Users\\bhadk\\Documents\\CPCL\\ACM\\src",
  [string]$Artifacts = "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts",
  [string]$TrainCsv  = "",
  [string]$TestCsv   = "",
  [string]$Equip     = "",
  [switch]$All,
  [switch]$NoBrief,
  [switch]$ForceTrain
)

$ErrorActionPreference = "Stop"
function Die ($m){ Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
function Ok  ($m){ Write-Host $m -ForegroundColor Green }
function Info($m){ Write-Host $m -ForegroundColor Cyan }
function Step($m){ Write-Host "== $m ==" -ForegroundColor Yellow }

# Paths to pipeline components
$Core      = Join-Path $Root "acm_core_local_2.py"
$ScoreAgg  = Join-Path $Root "acm_score_local_2.py"
$ArtifactM = Join-Path $Root "report_main.py"
$Brief     = Join-Path $Root "acm_brief_local.py"

# Sanity checks
foreach($p in @($Core,$ScoreAgg,$ArtifactM,$Brief)){
  if(!(Test-Path $p)){ Die "Missing $p" }
}

# Ensure artifacts root exists
New-Item -ItemType Directory -Force -Path $Artifacts | Out-Null

function ArtifactsReady([string]$dir){
  $need = @(
    'acm_scaler.joblib',
    'acm_regimes.joblib',
    'acm_pca.joblib',
    'acm_tag_baselines.csv',
    'acm_manifest.json'
  )
  foreach($n in $need){ if(-not (Test-Path (Join-Path $dir $n))){ return $false } }
  return $true
}

function Run-OneEquipment {
  param(
    [string]$Train,
    [string]$Test,
    [string]$Name
  )

  if(!(Test-Path $Train)){ Die "Train CSV not found: $Train" }
  if(!(Test-Path $Test )){ Die "Test CSV not found:  $Test"  }

  if([string]::IsNullOrWhiteSpace($Name)){
    $bn = Split-Path $Train -Leaf
    if($bn -like "* TRAINING DATA.csv"){
      $Name = ($bn -replace ' TRAINING DATA\.csv$','')
    } else {
      $Name = [System.IO.Path]::GetFileNameWithoutExtension($bn)
    }
  }

  Info "Equipment: $Name"
  $equipOut = Join-Path $Artifacts $Name
  New-Item -ItemType Directory -Force -Path $equipOut | Out-Null

  # Ensure core writes locally
  $env:ACM_ART_DIR = $Artifacts

  Step "Clean artifacts (transient outputs only)"
  $toClear = @(
    'acm_scored_window.csv','acm_events.csv','acm_context_masks.csv','acm_drift.csv','acm_resampled.csv',
    'acm_tag_scores.csv','acm_equipment_score.csv','acm_report.html','brief.json','brief.md','llm_prompt.json'
  )
  foreach($f in $toClear){ $p = Join-Path $Artifacts $f; if(Test-Path $p){ Remove-Item $p -Force -ErrorAction SilentlyContinue } }

  if($ForceTrain -or -not (ArtifactsReady $Artifacts)){
    Step "Train"
    python $Core train --csv "$Train"; if($LASTEXITCODE){ Die "Train failed" }
  } else {
    Info "Reusing existing model artifacts; skipping Train. Use -ForceTrain to retrain."
  }

  Step "Score (window)"
  python $Core score --csv "$Test"; if($LASTEXITCODE){ Die "Score failed" }

  Step "Drift check"
  python $Core drift --csv "$Test"; if($LASTEXITCODE){ Die "Drift failed" }

  Step "Compute equipment & tag scores"
  $sc = Join-Path $Artifacts "acm_scored_window.csv"
  $dr = Join-Path $Artifacts "acm_drift.csv"
  $ev = Join-Path $Artifacts "acm_events.csv"
  python $ScoreAgg --scored_csv "$sc" --drift_csv "$dr" --events_csv "$ev"
  if($LASTEXITCODE){ Die "Score aggregation failed" }

  Step "Build Basic HTML Report"
  $env:ACM_ART_DIR = $Artifacts
  python $ArtifactM
  if($LASTEXITCODE){ Die "Report build failed" }

  if(-not $NoBrief){
    if(Test-Path (Join-Path $Artifacts "acm_scored_window.csv")){
      Copy-Item (Join-Path $Artifacts "acm_scored_window.csv") (Join-Path $Artifacts "scored.csv") -Force
    }
    if(Test-Path (Join-Path $Artifacts "acm_events.csv")){
      Copy-Item (Join-Path $Artifacts "acm_events.csv") (Join-Path $Artifacts "events.csv") -Force
    }
    if(Test-Path (Join-Path $Artifacts "acm_context_masks.csv")){
      Copy-Item (Join-Path $Artifacts "acm_context_masks.csv") (Join-Path $Artifacts "masks.csv") -Force
    }

    Step "LLM Brief (brief.json + brief.md)"
    python $Brief build --art_dir "$Artifacts" --equip "$Name"; if($LASTEXITCODE){ Die "LLM brief build failed" }

    Step "LLM Prompt (llm_prompt.json)"
    python $Brief prompt --brief (Join-Path $Artifacts "brief.json"); if($LASTEXITCODE){ Die "LLM prompt build failed" }
  } else {
    Info "Skipping LLM brief/prompt (NoBrief set)."
  }

  Step "Archive outputs"
  $copyList = @(
    "acm_scaler.joblib","acm_regimes.joblib","acm_pca.joblib","acm_h1_ar1.json","acm_manifest.json",
    "acm_train_diagnostics.csv","acm_tag_baselines.csv",
    "acm_scored_window.csv","acm_events.csv","acm_context_masks.csv","acm_drift.csv","acm_resampled.csv",
    "acm_tag_scores.csv","acm_equipment_score.csv",
    "acm_report.html",
    "brief.json","brief.md","llm_prompt.json"
  )
  foreach($f in $copyList){ $src = Join-Path $Artifacts $f; if(Test-Path $src){ Copy-Item $src (Join-Path $equipOut $f) -Force } }

  Ok "Saved -> $equipOut"
  $rep = Join-Path $equipOut "acm_report.html"
  if(Test-Path $rep){ Start-Process $rep }
}

if($All){
  $csvDir = Join-Path $Root "Dummy Data"
  if(!(Test-Path $csvDir)){ Die "Dummy Data folder not found: $csvDir" }
  $trainFiles = Get-ChildItem "$csvDir\* TRAINING DATA.csv" -File
  if($trainFiles.Count -eq 0){ Die "No * TRAINING DATA.csv found in $csvDir" }

  foreach($tr in $trainFiles){
    $equip = ($tr.BaseName -replace ' TRAINING DATA$','')
    $ts = Join-Path $csvDir "$equip TEST DATA.csv"
    if(!(Test-Path $ts)){
      Write-Warning "Missing TEST DATA for '$equip' at $ts"; continue
    }
    Run-OneEquipment -Train $tr.FullName -Test $ts -Name $equip
  }
  Ok "Batch run complete."
} else {
  if([string]::IsNullOrWhiteSpace($TrainCsv) -or [string]::IsNullOrWhiteSpace($TestCsv)){
    Die "Provide -TrainCsv and -TestCsv, or use -All"
  }
  Run-OneEquipment -Train $TrainCsv -Test $TestCsv -Name $Equip
}
