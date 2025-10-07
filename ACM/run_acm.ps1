param(
  [string]$Root      = "C:\Users\bhadk\Documents\CPCL\ACM",
  [string]$Artifacts = "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts",
  [string]$TrainCsv  = "",
  [string]$TestCsv   = "",
  [string]$Equip     = "",
  [switch]$All,
  [switch]$NoBrief
)

$ErrorActionPreference = "Stop"
function Die($msg){ Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Ok($msg){ Write-Host $msg -ForegroundColor Green }
function Info($msg){ Write-Host $msg -ForegroundColor Cyan }
function Step($msg){ Write-Host "== $msg ==" -ForegroundColor Yellow }

$Core     = Join-Path $Root "acm_core_local_2.py"
$Artifact = Join-Path $Root "acm_artifact_local.py"
$ScoreAgg = Join-Path $Root "acm_score_local_2.py"
$Brief    = Join-Path $Root "acm_brief_local.py"

if (!(Test-Path $Core))     { Die "Missing $Core" }
if (!(Test-Path $Artifact)) { Die "Missing $Artifact" }
if (!(Test-Path $ScoreAgg)) { Die "Missing $ScoreAgg" }
if (!(Test-Path $Brief))    { Die "Missing $Brief" }

# Ensure artifacts root exists
New-Item -ItemType Directory -Force -Path $Artifacts | Out-Null

function Run-OneEquipment {
    param([string]$Train, [string]$Test, [string]$Name)

    if (!(Test-Path $Train)) { Die "Train CSV not found: $Train" }
    if (!(Test-Path $Test))  { Die "Test CSV not found:  $Test"  }

    if ([string]::IsNullOrWhiteSpace($Name)) {
        $Name = (Split-Path $Train -Leaf) -replace ' TRAINING DATA\.csv$',''
    }
    Info "Equipment: $Name"

    $equipOut = Join-Path $Artifacts $Name
    New-Item -ItemType Directory -Force -Path $equipOut | Out-Null

    Step "Clean artifacts (root)"
    # Only clear files in the artifacts root; keep subfolders like per-equipment archives
    Get-ChildItem -Path $Artifacts -File -Force -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

    Step "Train"
    python $Core train --csv "$Train"
    if ($LASTEXITCODE) { Die "Train failed" }

    Step "Score (window)"
    python $Core score --csv "$Test"
    if ($LASTEXITCODE) { Die "Score failed" }

    Step "Drift check"
    python $Core drift --csv "$Test"
    if ($LASTEXITCODE) { Die "Drift failed" }

    # Compute scores BEFORE report so EquipmentScore shows in KPIs
    Step "Compute equipment & tag scores"
    $sc = Join-Path $Artifacts "acm_scored_window.csv"
    $dr = Join-Path $Artifacts "acm_drift.csv"
    $ev = Join-Path $Artifacts "acm_events.csv"
    python $ScoreAgg --scored_csv "$sc" --drift_csv "$dr" --events_csv "$ev"
    if ($LASTEXITCODE) { Die "Score aggregation failed" }

    Step "Build HTML report"
    $mk = Join-Path $Artifacts "acm_context_masks.csv"
    python $Artifact --scored_csv "$sc" --drift_csv "$dr" --events_csv "$ev" --masks_csv "$mk"
    if ($LASTEXITCODE) { Die "Report build failed" }

    if (-not $NoBrief) {
        # Map current filenames to what acm_brief_local.py expects
        if (Test-Path (Join-Path $Artifacts "acm_scored_window.csv")) {
            Copy-Item (Join-Path $Artifacts "acm_scored_window.csv") (Join-Path $Artifacts "scored.csv") -Force
        }
        if (Test-Path (Join-Path $Artifacts "acm_events.csv")) {
            Copy-Item (Join-Path $Artifacts "acm_events.csv") (Join-Path $Artifacts "events.csv") -Force
        }
        if (Test-Path (Join-Path $Artifacts "acm_context_masks.csv")) {
            Copy-Item (Join-Path $Artifacts "acm_context_masks.csv") (Join-Path $Artifacts "masks.csv") -Force
        }

        Step "LLM Brief (brief.json + brief.md)"
        & python $Brief build --art_dir "$Artifacts" --equip "$Name"
        if ($LASTEXITCODE -ne 0) { Die "LLM brief build failed" }

        Step "LLM Prompt (llm_prompt.json)"
        & python $Brief prompt --brief (Join-Path "$Artifacts" "brief.json")
        if ($LASTEXITCODE -ne 0) { Die "LLM prompt build failed" }
    } else {
        Info "Skipping LLM brief/prompt (NoBrief set)."
    }

    # Archive outputs into per-equipment folder
    $copyList = @(
        # Models & manifests
        "acm_scaler.joblib","acm_regimes.joblib","acm_pca.joblib","acm_h1_ar1.json","acm_manifest.json",
        # Diagnostics & baselines
        "acm_train_diagnostics.csv","acm_tag_baselines.csv",
        # Scored window + extras
        "acm_scored_window.csv","acm_events.csv","acm_context_masks.csv","acm_drift.csv","acm_resampled.csv",
        # Scores
        "acm_tag_scores.csv","acm_equipment_score.csv",
        # Final report
        "acm_report.html",
        # Briefing bundle
        "brief.json","brief.md","llm_prompt.json"
    )
    foreach($f in $copyList){
        $src = Join-Path $Artifacts $f
        if(Test-Path $src){ Copy-Item $src (Join-Path $equipOut $f) -Force }
    }

    Ok "Saved → $equipOut"
    $rep = Join-Path $equipOut "acm_report.html"
    if (Test-Path $rep) { Start-Process $rep }
}

if ($All) {
    $csvDir = Join-Path $Root "Dummy Data"
    if (!(Test-Path $csvDir)) { Die "Dummy Data folder not found: $csvDir" }
    $trainFiles = Get-ChildItem "$csvDir\* TRAINING DATA.csv" -File
    if ($trainFiles.Count -eq 0) { Die "No * TRAINING DATA.csv found in $csvDir" }

    foreach($tr in $trainFiles){
        $equip = ($tr.BaseName -replace ' TRAINING DATA$','')
        $ts = Join-Path $csvDir "$equip TEST DATA.csv"
        if (!(Test-Path $ts)) { Write-Warning "Missing TEST DATA for '$equip' → $ts"; continue }
        Run-OneEquipment -Train $tr.FullName -Test $ts -Name $equip
    }
    Ok "Batch run complete."
}
else {
    if ([string]::IsNullOrWhiteSpace($TrainCsv) -or [string]::IsNullOrWhiteSpace($TestCsv)) {
        Die "Provide -TrainCsv and -TestCsv, or use -All"
    }
    Run-OneEquipment -Train $TrainCsv -Test $TestCsv -Name $Equip
}
