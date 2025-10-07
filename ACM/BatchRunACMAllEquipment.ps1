# ===== Batch run ACM for all equipment =====
$root   = "C:\Users\bhadk\Documents\CPCL\ACM"
$csvDir = "$root\Dummy Data"
$art    = "$root\acm_artifacts"

# ensure base artifacts dir exists
New-Item -ItemType Directory -Force -Path $art | Out-Null

# Find all training files and pair with test files
$trainFiles = Get-ChildItem "$csvDir\* TRAINING DATA.csv" -File

foreach ($train in $trainFiles) {
    $equip = ($train.BaseName -replace ' TRAINING DATA$','')
    $test  = Join-Path $csvDir "$equip TEST DATA.csv"

    if (!(Test-Path $test)) {
        Write-Warning "Missing TEST DATA for '$equip' → expected '$test'. Skipping."
        continue
    }

    Write-Host "=== Running: $equip ===" -ForegroundColor Cyan

    # 1) Train
    python "$root\acm_core_local_2.py" train --csv "$($train.FullName)"

    # 2) Score on test window
    python "$root\acm_core_local_2.py" score --csv "$test"

    # 3) Drift vs baseline
    python "$root\acm_core_local_2.py" drift --csv "$test"

    # 4) Artifacts (charts + HTML)
    python "$root\acm_artifact_local_2.py" --scored_csv "$art\acm_scored_window.csv" --drift_csv "$art\acm_drift.csv"

    # 5) Scores
    python "$root\acm_score_local_2.py" --scored_csv "$art\acm_scored_window.csv" --drift_csv "$art\acm_drift.csv"

    # ---- archive outputs under per-equipment folder ----
    $equipOut = Join-Path $art $equip
    New-Item -ItemType Directory -Force -Path $equipOut | Out-Null

    $toCopy = @(
        "acm_scaler.joblib","acm_regimes.joblib","acm_iforest.joblib","acm_manifest.json",
        "acm_train_diagnostics.csv","acm_tag_baselines.csv",
        "acm_scored_window.csv","acm_drift.csv",
        "signals.png","anomaly.png","regime.png","acm_report.html",
        "acm_tag_scores.csv","acm_equipment_score.csv"
    )

    foreach ($name in $toCopy) {
        $src = Join-Path $art $name
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $equipOut $name) -Force
        }
    }

    Write-Host "Saved → $equipOut" -ForegroundColor Green
}
