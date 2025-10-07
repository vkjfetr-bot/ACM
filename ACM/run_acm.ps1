param(
  [string]$Root    = "C:\Users\bhadk\Documents\CPCL\ACM",
  [string]$Artifacts = "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts",
  [string]$TrainCsv = "",
  [string]$TestCsv  = "",
  [string]$Equip    = "",
  [switch]$All      # batch over Dummy Data pairs
)

$ErrorActionPreference = "Stop"
function Die($msg){ Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Ok($msg){ Write-Host $msg -ForegroundColor Green }
function Info($msg){ Write-Host $msg -ForegroundColor Cyan }
function Step($msg){ Write-Host "== $msg ==" -ForegroundColor Yellow }

$Core     = Join-Path $Root "acm_core_local_2.py"
$Artifact = Join-Path $Root "acm_artifact_local.py"
$Score    = Join-Path $Root "acm_score_local_2.py"

if (!(Test-Path $Core))     { Die "Missing $Core" }
if (!(Test-Path $Artifact)) { Die "Missing $Artifact" }
if (!(Test-Path $Score))    { Die "Missing $Score" }

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
    
    Step "Clean artifacts"
    Get-ChildItem -Path $Artifacts -File -Force -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

    Step "Train"
    python $Core train --csv "$Train"; if ($LASTEXITCODE) { Die "Train failed" }

    Step "Score"
    python $Core score --csv "$Test"; if ($LASTEXITCODE) { Die "Score failed" }

    Step "Drift"
    python $Core drift --csv "$Test"; if ($LASTEXITCODE) { Die "Drift failed" }

    Step "Build report"
    $sc = Join-Path $Artifacts "acm_scored_window.csv"
    $dr = Join-Path $Artifacts "acm_drift.csv"
    $ev = Join-Path $Artifacts "acm_events.csv"
    $mk = Join-Path $Artifacts "acm_context_masks.csv"
    python $Artifact --scored_csv "$sc" --drift_csv "$dr" --events_csv "$ev" --masks_csv "$mk"
    if ($LASTEXITCODE) { Die "Artifact failed" }

    Step "Compute scores"
    python $Score --scored_csv "$sc" --drift_csv "$dr" --events_csv "$ev"
    if ($LASTEXITCODE) { Die "Score calc failed" }

    # Archive outputs into per-equipment folder
    $copyList = @(
        "acm_scaler.joblib","acm_regimes.joblib","acm_pca.joblib","acm_h1_models.joblib","acm_manifest.json",
        "acm_train_diagnostics.csv","acm_tag_baselines.csv","acm_scored_window.csv","acm_drift.csv",
        "acm_events.csv","acm_context_masks.csv",
        "acm_tag_scores.csv","acm_equipment_score.csv",
        "fused.png","heads.png","regime.png","acm_report.html"
    )
    foreach($f in $copyList){
        $src = Join-Path $Artifacts $f
        if(Test-Path $src){ Copy-Item $src (Join-Path $equipOut $f) -Force }
    }

    Ok "Saved → $equipOut"
    # Open report
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
