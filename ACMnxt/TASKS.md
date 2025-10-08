ACMnxt — Detailed Implementation Tasklist

Phase 1 Working Rules (agreed)
- Single input CSV (FD FAN) for train/score; generalization later.
- No GitHub Actions/CI; local scripts only.
- Stay close to current ACM: reuse report/charts modules where practical.
- Keep types/docstrings light; prioritize running code and clarity.
- Artifacts layout close to current (equip folder, images subfolder when applicable).

Milestones
- M1: Scaffold + Data IO + DQ + heatmap (DONE)
- M2: Minimal features + H1/H2 + fused + events for FD FAN
- M3: Visualizations using existing ACM report modules + simple adapter
- M4: Optional clean-up, docs, and broader inputs

Global Engineering Tasks
- [ ] (P1) Deterministic RNG seed helper + simple TZ normalization.
- [ ] (P1) Minimal logging via prints; JSONL later.
- [ ] (P2) YAML-driven RunConfig; equipment overrides later.

Data I/O & Schemas (acmnxt/io)
- [ ] loaders.py: `read_table(path: str|Path) -> pd.DataFrame` supporting CSV/XLSX.
- [ ] loaders.py: `ensure_datetime_index(df, ts_col='Ts') -> pd.DataFrame`.
- [ ] writers.py: `write_parquet(df, path)`, `write_png(fig, path)`, `write_md/html`.
- [ ] schemas.py: `RunConfig`, `ScoreRow`, `Event` (Pydantic models).
- [ ] Tests: missing columns, bad encodings, timezone handling.

Cleaning & DQ (acmnxt/core)
- [ ] dq.py: `clean_time(df)` sort, drop dupes, enforce monotonic Ts.
- [ ] dq.py: `resample_numeric(df, rule='1min')` numeric-only resample.
- [ ] dq.py: `compute_dq(df)` flatlines, spikes, dropouts, NaN%.
- [ ] Heatmap and ranked tag issues; write dq.csv and PNG.
- [ ] Tests for partial tag availability (graceful fallback).

Tag Selection (acmnxt/core)
- [ ] select_tags(df, include, exclude, variance_floor, max_n) with logs.
- [ ] Stable output given global seed; unit tests.

Feature Engineering (acmnxt/core/features.py)
- [ ] build_features(df, cfg): rolling mean/std, deltas, z-scores (multi-window).
- [ ] Frequency features via Welch PSD bands, dominant freq, flatness.
- [ ] Context: lag features, AR(1) residuals; warmup NaNs masked (not dropped).
- [ ] Write features.parquet; unit tests on shapes and NaN masks.

Hypotheses (acmnxt/core)
- [ ] H1 (h1_ar1.py): AR(1) residual z-score in [0,1].
- [ ] H2 (h2_pca.py): fit_pca() persisted; score_h2() recon error normalized.
- [ ] H3 (h3_drift.py): embedding drift (e.g., PCA center distance or UMAP opt).
- [ ] Toggle any H* off; ensemble continues if one fails.
- [ ] Cache PCA & regimes artifacts; performance tests.

Regimes & Masks (acmnxt/core)
- [ ] regimes.py: auto-K KMeans with silhouette sweep; persist labels/artifacts.
- [ ] masks.py: maintenance windows, startup/shutdown, DQ-bad windows.
- [ ] Ensure no refit explosion; stable silhouette logs.

Fusion & Eventization (acmnxt/core)
- [ ] fusion.py: weighted geometric mean across active H*.
- [ ] events.py: MAD-based threshold; peak detection; merge to episodes.
- [ ] events.py: event stats (duration, max score, top tags); write events.csv.
- [ ] Non-overlapping, sorted events; unit tests.

Visualization (prefer reuse of ACM)
- [ ] Adapt or call ACM/report_charts.py from a thin adapter for timeline.
- [ ] Event panels later; start with timeline + DQ.

Report (reuse ACM where possible)
- [ ] Thin wrapper that feeds scores/events into existing ACM/report_html.py.
- [ ] Generate equip folder with report.html and images subfolder.

CLI & Scripts (simple, single-file)
- [ ] train.py: `--csv --equip --out-dir` produce dq.csv, features.parquet (later), scores.csv, events.csv.
- [ ] score.py: reuse trained artifacts (PCA) in same out-dir.
- [ ] report.py: call existing ACM report builder to emit report.html + images/.
- [ ] PowerShell wrappers for single run only in P1.

Performance Targets (Phase 1 practical)
- [ ] FD FAN training file runs comfortably on laptop hardware.
- [ ] Optimize only if noticeably slow.

Config Management (acmnxt/conf)
- [ ] default.yaml: global defaults; per-equip overrides (FD_FAN).
- [ ] Threshold helper: auto suggest cutoffs (quantile/MAD).
- [ ] Changing YAML must drive behavior without code changes.

Robustness & Errors
- [ ] Descriptive exceptions (missing tags, NaN floods). Partial failures tolerated.
- [ ] Unit tests per failure mode; ensemble resilience when H* disabled.

Testing (acmnxt/tests)
- [ ] Unit: IO, DQ, features, H1–H3, fusion, events.
- [ ] Golden: FD FAN fixture (copied from ACM/Dummy Data); image hashes.
- [ ] E2E: Full run -> report -> validate artifacts.

Logging & Artifacts
- [ ] JSONL: timings, tag counts, KMeans info, thresholds.
- [ ] run.json summary for external consumption.

Packaging & Release
- [ ] Local editable install only; no GitHub Actions.

Docs & Commenting
- [ ] Concise docstrings and in-code comments where helpful.
- [ ] Heavy docs and diagrams later.

Reuse from Existing ACM (safe to port)
- [ ] Charts: adapt `ACM/report_charts.py` → `acmnxt/vis/*`.
- [ ] HTML scaffolding: `ACM/report_html.py` → `acmnxt/report/build.py`.
- [ ] CSS: `ACM/report_css.py` → `acmnxt/report/assets/style.css`.
- [ ] FD Fan dummy data → `acmnxt/data/dummy/FD_FAN_*`.
- [ ] Report text/glossary from `ACM/report_main.py` where applicable.

Backlog Notes
- Prefer sklearn PCA + StandardScaler; persist via joblib.
- Consider simple drift: distance from reference centroid in PCA space.
- KMeans K sweep (2..8) with silhouette; jitter seed stable.
- MAD threshold tuned via config; merge peaks with gap rule (configurable).
