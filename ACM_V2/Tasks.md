# ACMnxt — Development Plan & Detailed Task List (vNext)

## Ground Rules (from you)

* **Operationalization:** simple **CSV** logs per run (minimum info) + current JSONL timer log.
* **River:** implement **soon** (early phase), not a late stretch.
* **No venv**, **NO unit tests**.
* **LLM last**.
* **Wrapper script** to run everything first (one command).
* **Cold-Start & Data Availability logic** before LLM.
* **Dashboard payloads:** split into 2 — deliver **basic HTML** now; keep a **separate payload generator file (stub/empty)** for later.
* Keep/reuse **existing ACM core**; enhance only where needed.

---

## What’s missing vs your older ACM flow (from `acm_core_local_2.py`) — Gaps to close

1. **Dynamic threshold export**: events use fixed `fused_tau`; no θ(t) computation or `thresholds.csv`.
2. **DQ metrics**: no `dq.csv` (flatline%, spikes, dropout%, NaN% per tag).
3. **Cold-start/data-availability**: no phase gating (K=1, PCA rank checks, feature throttling) or retrain decision.
4. **River streaming**: no online KMeans/ADWIN adapter for scoring and drift triggers.
5. **Basic HTML analysis report**: not present.
6. **Dashboard payload generator**: not split; no stub that we can wire later.
7. **Evaluation/Refinement**: champion/challenger lanes not split yet (keep simple for now).
8. **Operational CSV**: no single **run_summary.csv** with minimal fields for quick ops view.
9. **Wrapper**: no `run_all` wrapper to orchestrate end-to-end.

We’ll close these deliberately in phases below.

---

## Minimal File Map (simple, local-only)

```
acmnxt/
  acm_core_local_2.py          # your current core (we’ll extend, not break)
  acm_payloads.py              # NEW: payload generator (stub/empty now)
  acm_report_basic.py          # NEW: basic HTML analysis report (tables+charts)
  acm_observe.py               # NEW: operationalization utilities (write CSVs)
  acm_river.py                 # NEW: River adapters (online KMeans, ADWIN), drop-in
  acm_evaluate.py              # LATER: evaluator (kept lean)
  acm_refine.py                # LATER: refinement/promote (lean)
  scripts/
    run_acmnxt.ps1             # NEW: wrapper to call train → score → report (and later refine)
artifacts/<equip>/             # outputs live here
data/                          # inputs (SP export or CSV)
```

> We **will not** create virtual environments or unit tests. Everything runs directly with system Python + pip’d libs.

---

## Phase Plan (each phase leaves something visible)

### PHASE 0 — Wrapper + Basic Operationalization (CSV) **(do first)**

**Goal:** One command to run; minimal ops visibility.

**Tasks**

* [ ] **Wrapper** `scripts/run_acmnxt.ps1`

  * `-Equip`, `-TrainCsv`, `-ScoreCsv`, `-ArtDir`, `-Minutes` (optional).
  * Calls: `python acm_core_local_2.py train`, then `score`, then `python acm_report_basic.py`.
* [ ] **Operational CSV** `run_summary.csv` (append-only, 1 row per run) via `acm_observe.py`:

  * Fields: `run_id, ts_utc, equip, cmd(train/score), rows_in, tags, feat_rows, regimes, events, data_span_min, phase, k_selected, theta_p95, drift_flag, status, err_msg`
* [ ] Keep existing JSONL block timings — already in core; **don’t remove**.

**Deliverables**

* ✅ `scripts/run_acmnxt.ps1` runnable end-to-end.
* ✅ `artifacts/<equip>/run_summary.csv` rows after each command.

---

### PHASE 1 — DQ Metrics + Dynamic Threshold Export

**Goal:** Produce `dq.csv` + dynamic θ(t) and store `thresholds.csv`.

**Tasks**

* [ ] In `acm_core_local_2.py` **train** and **score** paths:

  * Compute per-tag DQ: `flatline_pct, dropout_pct, NaN_pct, spikes_pct` → **`dq.csv`**.
  * Add `dynamic_threshold(fused_window, q=0.95, alpha=0.2)` → compute **θ(t)** per scored index.
  * Export **`thresholds.csv`** with `ts, theta`.
  * In episodes, use **θ(t) for eventization** (still keep `fused_tau` as fallback).
* [ ] Append `theta_p95` to `run_summary.csv` (median/95th of θ over window).

**Deliverables**

* ✅ `artifacts/<equip>/dq.csv`, `thresholds.csv`, events built off θ(t).
* ✅ Timeline in HTML shows fused vs θ line.

---

### PHASE 2 — Cold-Start & Data-Availability Logic (before LLM)

**Goal:** Auto-gate features/heads/regimes based on available data volume/rank.

**Tasks**

* [ ] **Phase detector** inside core:

  * `rows, tags, span_min, rank(X)` → decide **phase**:

    * P0 `<1000` rows → DQ + MAD/EWMA, **K=1**, no PCA/PSD
    * P1 `1k–5k` → H1 + dynamic θ, **K=1**, no PCA
    * P2 `5k–20k` → enable PCA (rank-aware) + H3; K∈{1..3}
    * P3 `>20k` → full stack; K∈{2..6}
* [ ] **Regime gate**: only allow `K>1` if silhouette ≥ 0.15 and min-cluster-size ≥ 10% window.
* [ ] **PCA rank guard**: if rank < features, reduce components; if <2, skip H2/H3 safely.
* [ ] **Retrain decision** (very simple): retrain if `(now - last_train) ≥ 7d` OR `new_rows ≥ 0.5 * train_rows` OR `drift_flag`.

**Deliverables**

* ✅ `run_summary.csv` shows chosen phase & K; system never stalls with thin data.

---

### PHASE 3 — **River** (Early) — Online Clustering + Drift

**Goal:** Add River adapters to prepare for true streaming; keep drop-in surface.

**Tasks**

* [ ] `acm_river.py`:

  * `RiverKMeansAdapter`: wraps `river.cluster.KMeans` (online partial updates).
  * `ADWINDrift`: wraps `river.drift.ADWIN` per fused score.
  * Common API: `.partial_fit(X)`, `.predict(X)`, `.drift_update(s)` → bool.
* [ ] In `score` path:

  * If **river mode enabled**, update ADWIN per fused; if drift → set `drift_flag=1` in `run_summary.csv`.
  * River KMeans predictions optional (parallel to batch ensemble); keep current ensemble default for now.
* [ ] Persist minimal River state (JSON) to artifacts (centroids, counts).

**Deliverables**

* ✅ `artifacts/<equip>/river_state.json` written; `run_summary.csv` marks `drift_flag` when ADWIN fires.

> We’re **implementing River early** as requested; full switch to online clustering remains optional until data cadence justifies it.

---

### PHASE 4 — Basic HTML Analysis Report (Dashboard now) + Payload Stub (empty)

**Goal:** Operators can **see** results; keep dashboard payload generator separate & empty.

**Tasks**

* [ ] `acm_report_basic.py`:

  * Read `scores.csv`, `thresholds.csv`, `events.csv`, `dq.csv`, (optional) `regimes.csv`.
  * Build a **simple HTML**: meta table → DQ table → timeline (PNG from matplotlib) → Top-N events table.
  * Save `report_<equip>.html`.
* [ ] **Split payloads**:

  * Create `acm_payloads.py` with **empty** functions (`build_timeline_payload()`, `build_events_payload()` etc.) → **write empty JSON** placeholders for now.
  * Ensure the report still renders from CSVs; payloads are not used yet.

**Deliverables**

* ✅ `report_<equip>.html` present (tables + charts).
* ✅ `payload_*.json` files exist but empty/placeholder.

---

### PHASE 5 — Evaluation & Refinement (lean, same-host)

**Goal:** Keep lanes separate (evaluation offline, refinement online), **no tests**, minimal code.

**Tasks**

* [ ] `acm_evaluate.py`:

  * Inputs: `--equip`, `--range`.
  * Score with current champion; **optionally** score with challenger (if present).
  * KPIs (unsupervised): FAR/day, alert density/day, flip rate/hr, silhouette (if K>1), coverage, score volatility.
  * Output: `eval_<ts>.json` + short HTML (table only).
* [ ] `acm_refine.py`:

  * **Decide** retrain based on simple policy (Phase 2).
  * Train challenger → save under `models/challenger_<ts>/`.
  * Call evaluator; **promote** if hard gates pass and soft score improves.
  * Update `models/champion/manifest.json`.

**Deliverables**

* ✅ Minimal evaluator/refine cycle working; promotion writes manifest.

---

### PHASE 6 — LLM (LAST)

**Goal:** Only after core + ops + River + cold-start are solid.

**Tasks**

* [ ] Generate `brief.md` and `brief.json` from events/θ/regimes/DQ.
* [ ] Structure `llm_prompt.json` (no network calls).

**Deliverables**

* ✅ Human-readable brief files per scoring window (or hourly).

---

## Concrete Edits to `acm_core_local_2.py` (surgical, not a rewrite)

1. **Operationalization CSV**

   * After each `train/score`, call `acm_observe.write_run_summary(...)` to **append** one row to `run_summary.csv`.
   * Inputs: info you already print + `phase`, `theta_p95`, `drift_flag`, `status/err`.

2. **DQ metrics**

   * Add `compute_dq(df, tags)` → write `dq.csv`.
   * Keep light: flatline%, spikes (z>k), dropout% (NaNs), NaN%.

3. **Dynamic θ(t)**

   * Add `dynamic_threshold(fused_series, q=0.95, alpha=0.2)` → returns aligned Series.
   * In `score_window`, compute `theta_series`, save `thresholds.csv` and **use θ** for episodes (replace fixed `fused_tau` thresholding; retain as fallback).

4. **Cold-start gating**

   * At top of `train_core` and `score_window`, calculate `rows,tags,span,rank` → set **phase** flags.
   * Conditionally skip PCA/PSD/regimes per phase, and force **K=1** as needed.

5. **River hooks**

   * If `RIVER_ENABLED` env/flag:

     * Update ADWIN each score (`adwin.update(fused_t)`).
     * Optionally get River KMeans labels and log them (but keep current ensemble as default).

6. **Exports**

   * Already exporting `scored_window.csv`, `events.csv`.
   * Add `thresholds.csv`, `dq.csv`, **and** minimal `run.json` meta (you have `manifest.json` for train; keep both).

7. **Error safety**

   * Wrap train/score blocks; in exception, log to `run_summary.csv` with `status="error"` and `err_msg`.

---

## `acm_observe.py` — Minimal CSV Writer (fields)

* File: `artifacts/<equip>/run_summary.csv`
* Columns (append one row per run):

  ```
  run_id, ts_utc, equip, cmd, rows_in, tags, feat_rows, regimes, events,
  data_span_min, phase, k_selected, theta_p95, drift_flag, status, err_msg
  ```

---

## `scripts/run_acmnxt.ps1` — Wrapper Flow (first thing to exist)

1. `python acm_core_local_2.py train --csv $TrainCsv`
2. `python acm_core_local_2.py score --csv $ScoreCsv`
3. `python acm_report_basic.py --equip $Equip`
4. (Later) `python acm_refine.py --equip $Equip` (optional daily)

---

## What you’ll see after Phase 1–4

* A **single PS1** command produces:

  * `dq.csv`, `scores.csv`, `thresholds.csv`, `events.csv`, `regimes.csv` (if eligible)
  * **run_summary.csv** with one line for each step
  * `report_<equip>.html` (basic tables + PNG timeline with θ line)

---

## Nice-to-have (kept out per constraints, but easy later)

* Composite-K upgrade (silhouette + separation − BIC) — you already have silhouette; add BIC and separation when needed.
* Robust HTML/CSS/JS payloads for Grafana — we left **`acm_payloads.py` stub** so wiring is straightforward when you’re ready.

---

### Quick sanity checklist (so we don’t skip anything):

* [x] Basic ops CSV (run_summary)
* [x] DQ metrics export
* [x] Dynamic θ(t) export and usage in events
* [x] Cold-start & data-availability gating **before** LLM
* [x] River adapters early (ADWIN + online KMeans)
* [x] Basic HTML report now
* [x] Separate **payload generator** file (empty for now)
* [x] Wrapper to run everything
* [x] No venv, no unit tests