# Changelog

All notable changes to this project are documented here. This file is maintained manually alongside meaningful commit messages and branches.

## [Unreleased]
- TBA

## [2025-10-08] ACMnxt v0.1.0 — Fast, modular baseline
- feat(acmnxt): New package under `acmnxt/` with CLI (train/score/report)
- feat(acmnxt): Native report with real charts (timeline, sampled tags, regimes, drift, event minis)
- feat(acmnxt): Score-only path; reuses PCA/scaler; 15-min loop friendly
- perf(acmnxt): Fast mode (PCA subsample, limited silhouette sample, downsampled DQ)
- chore(ps1): Wrapper supports separate Train CSV vs Score CSV (`-ScoreCsv`) + `-Fast`
- fix(report): tz-naive plotting and event context windows; clear legends
- chore(git): Add ignores for artifacts, joblibs, images, build outputs

## [2025-10-07] Brief/Report alignment and robustness
- feat(brief): Add `prompt` subcommand to `acm_brief_local.py` which reads `brief.json` and writes `llm_prompt.json` (simple chat payload).
- feat(brief): Make brief resilient to varying filenames and columns: reads `scored.csv`/`acm_scored_window.csv`, `events.csv`/`acm_events.csv`, `drift.csv`/`acm_drift.csv` and normalizes columns.
- fix(brief): Clean Markdown header and replace stray encoding artifacts.
- fix(report): Align `report_main.py` output to `acm_report.html` (was `acm_report_basic.html`).
- feat(report): Add Equipment Score KPI (reads `acm_equipment_score.csv`).
- perf(report): Compute Data Quality using `acm_resampled.csv` when present (smoother and more representative), fallback to scored frame.
- docs(report): Improve glossary (use τ and ≥) and wrap with module docstring.
- chore(report): Clean encoding artifacts in `report_html.py` and `report_charts.py` comments.
- chore(ps1): Add header comments to `run_acm_simple.ps1`; fix arrow artifact in `run_acm.ps1` output.

## Earlier history (from git log)
- 901e917 Added Codex, changed the ps file to run things
- 96dd2f9 Changes to brief generator and artefact generator
- 9fe2bde Added smaller report files instead of bigger artifact builder
- 8fae8af New Documentation to explain the software and the ML basics
- 19066eb New file to write the explanation of the script
- 052562d Broken report builder
- 0ada0e8 Report technically fixed and working.
- 9e7367c Inbetween fixing new report builder
- 2b5fa9a Added new Reporting Script. Currently facing issues.
- 00a18e7 Major Re-write. Changed the ML to improve H1
- 3f53059 Minor issue with H1 time computation
- 26435fb Major Revision to improve ML
- 34e581c Changed the local files to fix issues related to data quality and first successful run.
- 256f254 Made the scripts local and no changes in how the data is handled
- 352d00f Initial Commit
