# ACM Detailed Guide

This document freezes the current Asset Condition Monitor (ACM) implementation. It explains the machine-learning workflow, lists all scripts and artifacts, provides run recipes, and records operational notes so the solution can be reproduced without diving into the code.

---

## 1. System Overview

ACM ingests historian exports (CSV/XLS), learns baseline behaviour from a training dataset, evaluates a test dataset for anomalies and drift, and produces both numeric outputs and human-facing artefacts (HTML report, analyst brief, LLM prompt). The core pipeline resides in `ACM/src/`.

### 1.1 Execution Flow

1. **Train (`acm_core_local_2.py train`)** – preprocess training data, engineer windowed features, learn operating regimes, fit AR(1) coefficients, compute drift baselines, and persist artifacts.
2. **Score (`acm_core_local_2.py score`)** – repeat feature pipeline on test data, apply trained models, compute H1/H2/H3 heads, fusions, context masks, anomaly episodes, and export results.
3. **Drift (`acm_core_local_2.py drift`)** – compare tag mean/σ vs baseline to produce drift rankings.
4. **Aggregate (`acm_score_local_2.py`)** – produce equipment and tag-level health scores from fused burden, regime stability, drift, and events.
5. **Report (`report_main.py`)** – render static HTML report using scored data and diagnostics.
6. **Brief & Prompt (`acm_brief_local.py`)** – generate concise analyst brief plus optional LLM prompt.
7. **Wrapper (`run_acm.ps1`)** – orchestrate the entire workflow, clean transient files, archive per-equipment outputs.

Outputs land in `ACM/acm_artifacts/` and are copied to `ACM/acm_artifacts/<Equip>/` for archival.

---

## 2. Machine-Learning Components

### 2.1 Time Alignment
* `ensure_time_index` identifies timestamp columns (case-insensitive). If absent, processing continues using existing index (warnings emitted).
* `resample_numeric` resamples to 1 min cadence (mean + interpolation) when timestamps available.

### 2.2 Feature Engineering (`build_feature_matrix`)
* Sliding windows (`window`=256, `stride`=64) across selected numeric tags.
* **Time-domain features** – mean, RMS, variance, kurtosis, skew, crest factor, linear slope (last half of window).
* **Frequency-domain features** – RFFT magnitudes (0–63 bins), spectral centroid, spectral flatness.
* `RobustScaler` fits on training features to reduce sensitivity to outliers; reused during scoring.

### 2.3 Regime Detection
* `auto_kmeans` sweeps k in `[k_min, k_max]` (default 2–6), computes silhouette scores, and selects the best KMeans model.
* Fitted model persisted to `acm_regimes.joblib`; used later to assign regime labels during scoring and reporting.

### 2.4 H1 Forecast-lite + AR(1)
* Rolling median baselines drive robust residuals per tag.
* AR(1) coefficients fitted when sufficient support exists; residuals converted to z-scores via MAD.
* Tag z-scores averaged and normalised by p95 to form H1 head.

### 2.5 H2 PCA Reconstruction
* PCA retains 90% variance. Reconstruction error is normalised by p95 and clipped [0,1].

### 2.6 H3 Embedding Drift
* Projects features into PCA space; cosine distance to rolling (≤50 samples) mean indicates drift severity.

### 2.7 Context, Corroboration, Fusion
* `detect_transients` flags slope / acceleration outliers; fusion down-weights masked periods.
* `corroboration_boost` scores simultaneous deviations across tag pairs.
* `change_point_signal` monitors rolling mean/std changes across tags.
* Fusion weighting: `0.45*H1 + 0.35*H2 + 0.35*H3 + 0.15*corroboration + 0.10*changepoint`, clipped [0,1].
* Episodes extracted where fused score exceeds `fused_tau` (0.7 default) with gap merge of 3 samples.

### 2.8 Drift Analysis
* Baseline statistics per tag saved from training. Drift scoring calculates z-score difference of means and ranks tags accordingly.

---

## 3. Scripts and Artefacts

| Script | Purpose | Key Inputs | Outputs |
|--------|---------|------------|---------|
| `acm_core_local_2.py` | Train / score / drift pipeline | CSV path, artifacts dir | scaler.joblib, acm_regimes.joblib, acm_pca.joblib, acm_h1_ar1.json, acm_scored_window.csv, acm_events.csv, acm_context_masks.csv, acm_resampled.csv, acm_drift.csv, diagnostics log |
| `acm_score_local_2.py` | Compute equipment & tag scores | scored CSV, drift CSV, events CSV | acm_equipment_score.csv, acm_tag_scores.csv |
| `report_main.py` | HTML report builder | Artifacts from above | acm_report.html |
| `acm_brief_local.py` | Analyst brief & LLM prompt | Artifacts directory, equipment name | brief.json, brief.md, llm_prompt.json |
| `run_acm.ps1` | Wrapper orchestrator | Root, artifacts path, CSVs, options | Full pipeline run + archived `<Artifacts>/<Equip>/` folder |

### Generated Artefacts
* `acm_scored_window.csv` – fused score, heads, regimes per timestamp.
* `acm_events.csv` – anomaly episodes (Start, End, PeakScore).
* `acm_context_masks.csv` – transient mask flags.
* `acm_drift.csv` – tag-level drift Z-scores.
* `acm_equipment_score.csv` / `acm_tag_scores.csv` – summarised health.
* `acm_report.html`, `brief.json`, `brief.md`, `llm_prompt.json` – human-readable outputs.
* `acm_resampled.csv`, `acm_train_diagnostics.csv` – supporting data for reporting.

---

## 4. Running ACM

### 4.1 Prerequisites
* Windows PowerShell 5.1+ or PowerShell 7.
* Python 3.9+ with packages: `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`.
* Input CSVs should include a timestamp column for best results.

### 4.2 Wrapper Commands

```powershell
# Full retrain + score (FD FAN demo)
powershell -ExecutionPolicy Bypass -File ACM\run_acm.ps1 `
    -Root      "C:\Users\bhadk\Documents\CPCL\ACM\src" `
    -Artifacts "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts" `
    -TrainCsv  "C:\Users\bhadk\Documents\CPCL\ACM\Dummy Data\FD FAN TRAINING DATA.csv" `
    -TestCsv   "C:\Users\bhadk\Documents\CPCL\ACM\Dummy Data\FD FAN TEST DATA.csv" `
    -Equip     "FD FAN" `
    -ForceTrain

# Score-only using existing models
powershell -ExecutionPolicy Bypass -File ACM\run_acm.ps1 `
    -Root      "C:\Users\bhadk\Documents\CPCL\ACM\src" `
    -Artifacts "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts" `
    -TestCsv   "C:\data\Asset123_latest.csv" `
    -Equip     "Asset123"

# Batch run for all demo equipment (pairs TRAINING/TEST files)
powershell -ExecutionPolicy Bypass -File ACM\run_acm.ps1 `
    -Root      "C:\Users\bhadk\Documents\CPCL\ACM\src" `
    -Artifacts "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts" `
    -All
```

> Keep `-ForceTrain` when switching equipment to guarantee artifacts align with tag sets.

### 4.3 Individual CLI Steps

```powershell
# Train
python ACM\src\acm_core_local_2.py train --csv "...\Train.csv"

# Score
python ACM\src\acm_core_local_2.py score --csv "...\Test.csv"

# Drift
python ACM\src\acm_core_local_2.py drift --csv "...\Test.csv"

# Aggregate
python ACM\src\acm_score_local_2.py `
    --scored_csv ACM\acm_artifacts\acm_scored_window.csv `
    --drift_csv  ACM\acm_artifacts\acm_drift.csv `
    --events_csv ACM\acm_artifacts\acm_events.csv

# Report
$env:ACM_ART_DIR = "C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts"
python ACM\src\report_main.py

# Brief + prompt
python ACM\src\acm_brief_local.py build --art_dir "C:\...\acm_artifacts" --equip "Asset XYZ"
python ACM\src\acm_brief_local.py prompt --brief "C:\...\acm_artifacts\brief.json"
```

---

## 5. Dummy Data Reference

* `ACM/Dummy Data/FD FAN ...` – includes timestamps, validated pipeline run.
* `ACM/Dummy Data/Gas Turbine ...` – raw demo lacking timestamps; clean versions are recommended for proper context/corroboration operations. If using raw files, add a timestamp column or accept limited functionality.

---

## 6. Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| `ERROR: Train CSV not found` | Path typo or quoting issue | Verify PowerShell quoting, ensure file exists |
| `[WARN] No timestamp column found` | CSV lacks timestamps | Provide cleaned CSV with datetime column to unlock full pipeline |
| `TypeError: Cannot compare dtypes int64 and datetime64` | Mixing timestamped and non-timestamped indices | Ensure both train and test datasets share datetime index |
| Missing report sections | Required files not generated | Confirm `acm_scored_window.csv`, `acm_events.csv`, `acm_drift.csv`, `acm_equipment_score.csv` exist |
| Equipment score unexpectedly low | High fused burden or event boost | Inspect `acm_scored_window.csv` and `acm_events.csv`; check H1/H2/H3 contributions |

---

## 7. Directory Layout (Frozen)

```
ACM/
├── Dummy Data/
├── docs/
│   ├── ACM Core Local Explanation
│   ├── ACM_ML_Theory_Guide.md
│   ├── ACM_Detailed_Guide.md   # this file
│   └── README_ACM_Core.md
├── acm_artifacts/              # overwritten per run
├── run_acm.ps1
└── src/
    ├── acm_core_local_2.py
    ├── acm_score_local_2.py
    ├── acm_brief_local.py
    ├── report_main.py
    ├── report_html.py
    ├── report_css.py
    └── report_charts.py
```

The entire directory is archived as `ACM_frozen.zip` in the repository root.

---

## 8. Change Control Checklist

1. Duplicate `ACM/` into a working copy before experimenting.
2. Apply code/data modifications and run FD FAN + cleaned Gas Turbine end-to-end.
3. Update this guide and other docs with behaviour changes.
4. Regenerate `ACM_frozen.zip`.
5. Commit changes, tag release, and communicate to stakeholders.

---

## 9. Contacts & Support

* **ML/Model owners:** responsible for H1/H2/H3, regimes, drift logic.
* **Reporting team:** maintains `report_main.py`, charts, HTML/CSS.
* **Operations:** ensures data availability, runtime environment, scheduling.

Keep this guide alongside the frozen package for future teams.
