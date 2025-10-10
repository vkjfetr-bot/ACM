# ACMnxt - Development Plan & Detailed Task List (vNext)

## Ground Rules (from you)

* **Operationalization:** simple **CSV** logs per run (minimum info) + current JSONL timer log.
* **River:** implement **soon** (early phase), not a late stretch.
* **No venv**, **NO unit tests**.
* **LLM last**.
* **Wrapper script** to run everything first (one command).
* **Cold-Start & Data Availability logic** before LLM.
* **Dashboard payloads:** split into 2 - deliver **basic HTML** now; keep a **separate payload generator file (stub/empty)** for later.
* Keep/reuse **existing ACM core**; enhance only where needed.

---

## What's missing vs your older ACM flow (from `acm_core_local_2.py`) - Gaps to close

1. **Dynamic health index & guardrail logging**: events still use fixed `fused_tau`; there is no theta(t), no `thresholds.csv`, and no detection of sudden threshold shifts.
2. **DQ metrics**: no `dq.csv` (flatline%, spikes, dropout%, NaN% per tag).
3. **Cold-start/data-availability**: no phase gating (K=1, PCA rank checks, feature throttling) or retrain decision.
4. **Operator insight loop**: no per-event tag contributions, persistence classification, or `events_timeline.json` to answer "what, who, when".
5. **Autonomy guardrails**: no drift/volume gatekeeping, theta step alerts, runtime health monitoring, or rollback cache aligned with the new accountability pillar.
6. **River streaming**: no online KMeans/ADWIN adapter for scoring and drift triggers.
7. **Basic HTML analysis report**: not present.
8. **Dashboard payload generator**: not split; no stub that we can wire later.
9. **Evaluation/Refinement**: champion/challenger lanes not split yet; no seeded-scenario evaluation or guardrail-based promotion policy.
10. **Operational CSV + wrapper**: no single `run_summary.csv` with minimal fields and no `run_all` wrapper to orchestrate end-to-end.

We'll close these deliberately in phases below.

---

## Minimal File Map (simple, local-only)

```
ACM_V2/
  acm_core_local_2.py          # your current core (we'll extend, not break)
  acm_payloads.py              # NEW: payload generator (stub/empty now)
  acm_report_basic.py          # NEW: basic HTML analysis report (tables+charts)
  acm_observe.py               # NEW: operational + guardrail utilities (run CSV, guardrail log, runtime checks)
  acm_river.py                 # NEW: River adapters (online KMeans, ADWIN), drop-in
  acm_evaluate.py              # LATER: evaluator (kept lean)
  acm_refine.py                # LATER: refinement/promote (lean)
  scripts/
    run_acmnxt.ps1             # NEW: wrapper to call train -> score -> report (and later refine)
artifacts/<equip>/             # outputs live here
data/                          # inputs (SP export or CSV)
```

> We **will not** create virtual environments or unit tests. Everything runs directly with system Python + pip'd libs.

---

## Phase Plan (each phase leaves something visible)

### PHASE 0 - Wrapper + Basic Operationalization (CSV) **(do first)**

**Goal:** One command to run; minimal ops visibility.

**Tasks**

* [x] **Wrapper** `scripts/run_acmnxt.ps1`

  * `-Equip`, `-TrainCsv`, `-ScoreCsv`, `-ArtDir`, `-Minutes` (optional).
  * Calls: `python acm_core_local_2.py train`, then `score`, then `python acm_report_basic.py`.
* [x] **Operational CSV** `run_summary.csv` (append-only, 1 row per run) via `acm_observe.py`:

  * Fields: `run_id, ts_utc, equip, cmd(train/score), rows_in, tags, feat_rows, regimes, events, data_span_min, phase, k_selected, theta_p95, drift_flag, guardrail_state, theta_step_pct, latency_s, artifacts_age_min, status, err_msg`
* [x] Keep existing JSONL block timings - already in core; **don't remove**.

**Deliverables**

* [x] `scripts/run_acmnxt.ps1` runnable end-to-end.
* [x] `artifacts/<equip>/run_summary.csv` rows after each command.

---

### PHASE 1 - DQ Metrics + Dynamic Threshold Export

**Goal:** Produce `dq.csv` + dynamic theta(t) and store `thresholds.csv`.

**Tasks**

* [x] In `acm_core_local_2.py` **train** and **score** paths:

  * Compute per-tag DQ: `flatline_pct, dropout_pct, NaN_pct, spikes_pct` -> **`dq.csv`**.
  * Add `dynamic_threshold(fused_window, q=0.95, alpha=0.2)` -> compute **theta(t)** per scored index.
  * Export **`thresholds.csv`** with `ts, theta`.
  * Compare latest `theta` band to previous run -> compute `theta_step_pct`, stash to `run_summary.csv`, and append guardrail entry to `guardrail_log.jsonl` when a step exceeds tolerance.
  * In episodes, use **theta(t) for eventization** (still keep `fused_tau` as fallback).
* [x] Append `theta_p95` to `run_summary.csv` (median/95th of theta over window).

**Deliverables**

* [x] `artifacts/<equip>/dq.csv`, `thresholds.csv`, events built off theta(t).
* [ ] Timeline in HTML shows fused vs theta line; guardrail log captures any large theta jumps.

---

### PHASE 2 - Cold-Start & Data-Availability Logic (before LLM)

**Goal:** Auto-gate features/heads/regimes based on available data volume/rank.

**Tasks**

* [ ] **Phase detector** inside core:

  * `rows, tags, span_min, rank(X)` -> decide **phase**:

    * P0 `<1000` rows -> DQ + MAD/EWMA, **K=1**, no PCA/PSD
    * P1 `1k-5k` -> H1 + dynamic theta, **K=1**, no PCA
    * P2 `5k-20k` -> enable PCA (rank-aware) + H3; Kin{1..3}
    * P3 `>20k` -> full stack; Kin{2..6}
* [ ] **Regime gate**: only allow `K>1` if silhouette >= 0.15 and min-cluster-size >= 10% window.
* [ ] **PCA rank guard**: if rank < features, reduce components; if <2, skip H2/H3 safely.
* [ ] **Retrain decision** (very simple): retrain if `(now - last_train) >= 7d` OR `new_rows >= 0.5 * train_rows` OR `drift_flag`.
* [ ] When a phase downgrade or retrain deferral occurs, raise a guardrail entry with reason (`thin_data`, `low_rank`, etc.) and set `guardrail_state` accordingly.

**Deliverables**

* [x] `run_summary.csv` shows chosen phase & K, guardrail state, and system never stalls with thin data.

---

### PHASE 3 - Autonomy Guardrails + River Streaming (Early)

**Goal:** Instrument automation so it can be trusted (guardrails, health signals, rollback) while wiring in River streaming hooks.

**Tasks**

* [ ] **Guardrail instrumentation** in `acm_observe.py`:

  * Maintain `guardrail_log.jsonl` with entries for `theta_step`, `phase_downgrade`, `river_drift`, `thin_data`, `runtime_slow`, `artifact_stale`, etc.
  * Aggregate guardrail state (`ok | warn | alert`) per run and write to `run_summary.csv`.
  * Track scheduler runtime: capture wall-clock duration and SP latency -> populate `latency_s`.
  * Compute artifact freshness (minutes since last score output) -> populate `artifacts_age_min`.
  * Expose helper `ack_guardrail(run_id, type)` to mark items as acknowledged (CLI optional).
  * Emit `run_health.json` snapshot summarising latency, guardrail state, and latest acknowledgements.
* [ ] **Rollback cache**:

  * After each successful train, snapshot current champion (models, scalers, thresholds) to `models/rollback/last/`.
  * Provide `acm_observe.create_rollback_point()` and `restore_rollback_point()` wrappers (no auto-restore yet).
* [ ] **River adapters** (`acm_river.py`):

  * `RiverKMeansAdapter`: wraps `river.cluster.KMeans` (online partial updates).
  * `ADWINDrift`: wraps `river.drift.ADWIN` per fused score with `drift_update()` returning severity.
  * Persist minimal River state (JSON) to artifacts (centroids, counts, adwin deltas).
* [ ] **Score-path hooks**:

  * If River mode enabled, update ADWIN per fused window; on drift, set `drift_flag=1`, emit guardrail entry, and optionally trigger challenger retrain flag.
  * River KMeans predictions optional (parallel to batch ensemble); keep current ensemble default for now.

**Deliverables**

* [ ] `artifacts/<equip>/river_state.json`, `guardrail_log.jsonl`, and `run_health.json` written; `run_summary.csv` shows guardrail state when drift/latency occurs.

> We're **implementing River early** as requested, while also wiring the accountability hooks demanded by the updated vision.

---

### PHASE 4 - Reporting & Operator Insight Loop (HTML now, payload stub separate)

**Goal:** Answer "what's happening, will it continue, who caused it, when did it start" via reports while keeping dashboard payloads isolated for later.

**Tasks**

* [ ] **Core enrichments** (builds on Phase 1/2 work):

  * During eventization, compute persistence classification (`transient | persistent`) using fused slope + duration patience.
  * Capture per-head tag contributions (`H1`, `H2`, `H3`) -> attach to `events.csv` and emit `events_timeline.json`.
  * Include contributing-head list and persistence flag in `scores.csv` (via `per_tag_contrib` column or sidecar file).
* [ ] `acm_report_basic.py`:

  * Read `scores.csv`, `thresholds.csv`, `events.csv`, `dq.csv`, `guardrail_log.jsonl`, (optional) `regimes.csv`.
  * Build HTML: meta/guardrail callouts -> DQ heatmap -> timeline (fused vs theta, drift markers) -> event cards listing top tags, persistence, start/end, contributing heads.
  * Highlight current guardrail state and outstanding acknowledgements.
  * Save `report_<equip>.html`.
* [ ] **Split payloads**:

  * `acm_payloads.py` stays separate but now writes placeholder JSON structures (`timeline.json`, `events.json`, `dq.json`) with empty arrays ready for real data later.

**Deliverables**

* [ ] `report_<equip>.html` present with top tags/persistence callouts.
* [x] `events_timeline.json` plus placeholder `payload_*.json` files exist (structure only).

---

### PHASE 5 - Evaluation & Refinement (lean, same-host)

**Goal:** Keep lanes separate (evaluation offline, refinement online), **no tests**, minimal code.

**Tasks**

* [ ] `acm_evaluate.py`:

  * Inputs: `--equip`, `--range`.
  * Score with current champion; **optionally** score with challenger (if present).
  * KPIs (unsupervised): FAR/day, alert density/day, flip rate/hr, silhouette (if K>1), coverage, score volatility, persistence classification accuracy, theta stability (step %), drift response time.
  * Support **seeded scenario injection** (replay labelled anomalies from CSV) to approximate precision, recall, and detection delay.
  * Output: `eval_<ts>.json` + short HTML (table only) + guardrail summary block (pass/fail flags).
* [ ] `acm_refine.py`:

  * **Decide** retrain based on policy (Phase 2 guardrails + River drift severity).
  * Train challenger -> save under `models/challenger_<ts>/`.
  * Call evaluator; **promote** only if KPI guardrails pass and theta shift stays within tolerance (else log guardrail alert and keep champion).
  * On promotion, archive previous champion under `models/rollback/last/` and note change in `guardrail_log.jsonl`.
  * Update `models/champion/manifest.json` and emit `promotion_notice.json` for operators.

**Deliverables**

* [x] Evaluator outputs KPI + guardrail summaries; promotion writes manifest, rollback snapshot, and guardrail log entry.

---

### PHASE 6 - LLM (LAST)

**Goal:** Only after core + ops + River + cold-start are solid.

**Tasks**

* [ ] Generate `brief.md` and `brief.json` capturing current health (`I(t)` vs `theta(t)`), persistence outlook, top contributing tags, guardrail alerts.
* [ ] Structure `llm_prompt.json` (no network calls) with sections: context, current health, outlook, drivers, recommended actions.

**Deliverables**

* [x] Human-readable brief files per scoring window (or hourly) summarising health, outlook, drivers, guardrails.

---

## Concrete Edits to `acm_core_local_2.py` (surgical, not a rewrite)

1. **Operationalization CSV + guardrail fields**

   * After each `train/score`, call `acm_observe.write_run_summary(...)` to **append** one row to `run_summary.csv`.
   * Inputs: info you already print + `phase`, `theta_p95`, `drift_flag`, `guardrail_state`, `theta_step_pct`, `latency_s`, `artifacts_age_min`, `status/err`.

2. **DQ metrics**

   * Add `compute_dq(df, tags)` -> write `dq.csv`.
   * Keep light: flatline%, spikes (z>k), dropout% (NaNs), NaN%.

3. **Dynamic theta(t) + guardrail checks**

   * Add `dynamic_threshold(fused_series, q=0.95, alpha=0.2)` -> returns aligned Series.
   * In `score_window`, compute `theta_series`, save `thresholds.csv`, and **use theta** for episodes (replace fixed `fused_tau`; retain as fallback).
   * Compare latest theta band to prior run -> compute `theta_step_pct`, push to `run_summary.csv`, and append guardrail entry if tolerance exceeded.

4. **Event insight metrics**

   * Expand eventization outputs with persistence classification, contributing heads, and per-tag contribution vectors.
   * Write `events_timeline.json` plus any sidecar needed for `per_tag_contrib`.

5. **Cold-start gating**

   * At top of `train_core` and `score_window`, calculate `rows,tags,span,rank` -> set **phase** flags.
   * Conditionally skip PCA/PSD/regimes per phase, and force **K=1** as needed.

6. **River hooks**

   * If `RIVER_ENABLED` env/flag:

     * Update ADWIN each score (`adwin.update(fused_t)`), capture severity, and log guardrail entries on drift.
     * Optionally get River KMeans labels and log them (but keep current ensemble as default).

7. **Exports**

   * Already exporting `scored_window.csv`, `events.csv`.
   * Add `thresholds.csv`, `dq.csv`, `events_timeline.json`, `guardrail_log.jsonl`, and `run_health.json` (you have `manifest.json` for train; keep both).

8. **Error safety**

   * Wrap train/score blocks; in exception, log to `run_summary.csv` with `status="error"` and `err_msg`.

---

## `acm_observe.py` - Minimal CSV Writer (fields)

* File: `artifacts/<equip>/run_summary.csv`
* Columns (append one row per run):

  ```
  run_id, ts_utc, equip, cmd, rows_in, tags, feat_rows, regimes, events,
  data_span_min, phase, k_selected, theta_p95, drift_flag, guardrail_state,
  theta_step_pct, latency_s, artifacts_age_min, status, err_msg
  ```

* Guardrail acknowledgements persist in `guardrail_log.jsonl` (one entry per alert) and optional `run_health.json` snapshot.

---

## `scripts/run_acmnxt.ps1` - Wrapper Flow (first thing to exist)

1. `python acm_core_local_2.py train --csv $TrainCsv`
2. `python acm_core_local_2.py score --csv $ScoreCsv`
3. `python acm_report_basic.py --equip $Equip`
4. (Later) `python acm_refine.py --equip $Equip` (optional daily)

---

## What you'll see after Phase 1-4

* A **single PS1** command produces:

  * `dq.csv`, `scores.csv`, `thresholds.csv`, `events.csv` (and `events_timeline.json` once built)
  * `guardrail_log.jsonl` + `run_health.json` showing theta steps, drift, runtime health (pending River phase)
  * **run_summary.csv** with guardrail fields populated for each step
  * `report_<equip>.html` (timeline with theta + drift markers, event cards with top tags/persistence) [pending HTML implementation]

---

## Nice-to-have (kept out per constraints, but easy later)

* Composite-K upgrade (silhouette + separation - BIC) - you already have silhouette; add BIC and separation when needed.
* Robust HTML/CSS/JS payloads for Grafana - we left **`acm_payloads.py` stub** so wiring is straightforward when you're ready.
* Guardrail notification bridge into alert bus (PowerShell/REST) once ack workflow is defined.
* Persistence forecasting / hazard modelling to estimate time-to-alert based on fused trends.

---

### Quick sanity checklist (so we don't skip anything):

* [x] Basic ops CSV (run_summary with guardrail fields)
* [x] DQ metrics export
* [x] Dynamic theta(t) export + guardrail logging
* [ ] Cold-start & data-availability gating **before** LLM
* [x] Guardrail instrumentation (log, run_health, rollback)
* [ ] Evaluator guardrails + seeded scenario injection
* [ ] Operator insight loop (persistence, top tags, events_timeline)
* [ ] River adapters early (ADWIN + online KMeans)
* [ ] Basic HTML report now
* [x] Separate **payload generator** file (empty for now)
* [x] Wrapper to run everything
* [x] No venv, no unit tests
