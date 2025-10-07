# run_acm_simple.ps1 â€” streamlined single-equipment run

#
# Streamlined single-equipment ACM pipeline runner.
# Steps: train -> score -> drift -> aggregate scores -> build HTML -> (optional) LLM brief + prompt -> archive
#
param(
  [string]$Root      = "C:\Users\bhadk\Documents\CPCL\ACM",
  [string]$Artifacts = "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts",
  [string]$TrainCsv  = "C:\Users\bhadk\Documents\CPCL\ACM\Dummy Data\FD FAN TRAINING DATA.csv",
  [string]$TestCsv   = "C:\Users\bhadk\Documents\CPCL\ACM\Dummy Data\FD FAN TEST DATA.csv",
  [string]$Equip     = "FD FAN",
  [switch]$NoBrief,
  [switch]$ForceTrain
)

$ErrorActionPreference = "Stop"
function Die ($m){ Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
function Step($m){ Write-Host "== $m ==" -ForegroundColor Yellow }
function Info($m){ Write-Host $m -ForegroundColor Cyan }
function Ok  ($m){ Write-Host $m -ForegroundColor Green }

# ---- Paths ----
$Core      = Join-Path $Root "acm_core_local_2.py"
$ScoreAgg  = Join-Path $Root "acm_score_local_2.py"
$ArtifactM = Join-Path $Root "report_main.py"          # main HTML builder (writes acm_report.html)
$Brief     = Join-Path $Root "acm_brief_local.py"

# ---- Sanity ----
foreach($p in @($Core,$ScoreAgg,$ArtifactM,$Brief)){ if(!(Test-Path $p)){ Die "Missing $p" } }
if(!(Test-Path $TrainCsv)){ Die "Train CSV not found: $TrainCsv" }
if(!(Test-Path $TestCsv )){ Die "Test CSV not found:  $TestCsv"  }

# ---- Prep ----
New-Item -ItemType Directory -Force -Path $Artifacts | Out-Null
$equipOut = Join-Path $Artifacts $Equip
New-Item -ItemType Directory -Force -Path $equipOut | Out-Null

# Clean only files in $Artifacts root (keep per-equipment folders)
Step "Clean artifacts (transient outputs only)"
# Keep model artifacts (joblib/json baselines); clear only transient outputs used by test/report
$toClear = @(
  'acm_scored_window.csv','acm_events.csv','acm_context_masks.csv','acm_drift.csv','acm_resampled.csv',
  'acm_tag_scores.csv','acm_equipment_score.csv','acm_report.html','brief.json','brief.md','llm_prompt.json'
)
foreach($f in $toClear){ $p = Join-Path $Artifacts $f; if(Test-Path $p){ Remove-Item $p -Force -ErrorAction SilentlyContinue } }

# ---- Pipeline ----
Info "Equipment: $Equip"

# Skip train if artifacts already exist and not forcing
function ArtifactsReady($dir){
  $need = @(
    "acm_scaler.joblib",
    "acm_regimes.joblib",
    "acm_pca.joblib",
    "acm_tag_baselines.csv",
    "acm_manifest.json"
  )
  foreach($n in $need){ if(-not (Test-Path (Join-Path $dir $n))){ return $false } }
  return $true
}

if($ForceTrain -or -not (ArtifactsReady $Artifacts)){
  Step "Train"
  python $Core train --csv "$TrainCsv"; if($LASTEXITCODE){ Die "Train failed" }
} else {
  Info "Reusing existing model artifacts; skipping Train. Use -ForceTrain to retrain."
}

Step "Score (window)"
python $Core score --csv "$TestCsv"; if($LASTEXITCODE){ Die "Score failed" }

Step "Drift check"
python $Core drift --csv "$TestCsv"; if($LASTEXITCODE){ Die "Drift failed" }

Step "Aggregate scores (equipment + tags)"
$sc = Join-Path $Artifacts "acm_scored_window.csv"
$dr = Join-Path $Artifacts "acm_drift.csv"
$ev = Join-Path $Artifacts "acm_events.csv"
python $ScoreAgg --scored_csv "$sc" --drift_csv "$dr" --events_csv "$ev"
if($LASTEXITCODE){ Die "Score aggregation failed" }

# Build report (report_main.py should internally call acm_artifact_local/report_charts)
# Build report (report_main.py writes acm_report.html in $Artifacts)
Step "Build HTML report"
$env:ACM_ART_DIR = $Artifacts
python $ArtifactM
if($LASTEXITCODE){ Die "Report build failed" }

if(-not $NoBrief){
  # Map filenames to what acm_brief_local.py expects
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
  # acm_brief_local.py supports subcommands: build | prompt
  python $Brief build --art_dir "$Artifacts" --equip "$Equip"
  if($LASTEXITCODE){ Die "LLM brief build failed" }

  Step "LLM Prompt (llm_prompt.json)"
  python $Brief prompt --brief (Join-Path $Artifacts "brief.json")
  if($LASTEXITCODE){ Die "LLM prompt build failed" }
}else{
  Info "Skipping LLM brief/prompt (NoBrief set)."
}

# ---- Archive essentials to per-equipment folder ----
Step "Archive outputs"
$copyList = @(
  # Models & manifests
  "acm_scaler.joblib","acm_regimes.joblib","acm_pca.joblib","acm_h1_ar1.json","acm_manifest.json",
  # Diagnostics & baselines
  "acm_train_diagnostics.csv","acm_tag_baselines.csv",
  # Scored window + extras
  "acm_scored_window.csv","acm_events.csv","acm_context_masks.csv","acm_drift.csv","acm_resampled.csv",
  # Scores
  "acm_tag_scores.csv","acm_equipment_score.csv",
  # Final report (report_main.py output name)
  "acm_report.html",
  # Briefing bundle
  "brief.json","brief.md","llm_prompt.json"
)
foreach($f in $copyList){
  $src = Join-Path $Artifacts $f
  if(Test-Path $src){ Copy-Item $src (Join-Path $equipOut $f) -Force }
}

Ok "Saved -> $equipOut"
$rep = Join-Path $equipOut "acm_report.html"
if(Test-Path $rep){ Start-Process $rep }
