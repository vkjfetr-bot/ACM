# 0) High-level pipeline (end-to-end)

**Ingest → Clean/DQ → Feature Bank → Regime Discovery → Regime-aware Anomaly Scoring → Event Grouping/Classification → Drift/Health → Brief/Explain → Persist → Report.**

Each stage is **stateless in code** but **stateful via artifacts** (per-equipment). Artifacts live under `acm_artifacts/<EQUIP_ID>/...`.

---

# 1) Data interfaces & artifacts

## 1.1 SQL Interfaces (assumed existing)

* `usp_GetEquipmentWindow(@equip, @t0, @t1)` → raw time series (wide format).
* `usp_WriteAnomalyEvent(@equip, @t0, @t1, @tau, @state, @score_type, @score, @ver)` → log events.
* `usp_WriteRegimeModels(@equip, @ver, @k, @centroids, @pca, @hmm)` → model blobs (varbinary(max)).
* `usp_WriteRunSummary(@equip, @run_id, @t0, @t1, @n_pts, @n_anoms, @dq_json, @lat_s)` → ops summary.

> If not present, create simple tables (names suggestive, adjust to your schema):

* `ACM.RunLog(run_id PK, equip, t0, t1, n_pts, n_anoms, dq_json NVARCHAR(MAX), latency_s, ver, created_utc)`
* `ACM.RegimeModel(equip PK, ver, k, pca VARBINARY(MAX), cluster_centers VARBINARY(MAX), hmm VARBINARY(MAX), created_utc)`
* `ACM.AnomalyEvent(event_id PK, equip, t0, t1, regime, score_type, score, tau, family_id NULL, top_tags NVARCHAR(4000), img_time VARBINARY(MAX), img_spec VARBINARY(MAX), created_utc)`
* `ACM.DriftLog(equip, ver, t, metric, value, threshold, action, created_utc)`

## 1.2 Artifact layout (on disk)

```
acm_artifacts/
  {EQUIP}/
    run_{YYYYMMDD_HHMMSS}/
      raw.parquet
      clean.parquet
      features.parquet
      dq.json
      regimes.json            # labels per sample, transitions
      regime_model.pkl        # {pca, kmeans|hdbscan, hmm}
      scores.csv              # point-, window-, and sequence-level
      anomalies.csv           # merged events w/ thresholds
      families.json           # clustered anomaly families
      drift.json              # ADWIN / PH alarms
      brief.json              # machine-readable
      brief.md                # operator brief
      report.html             # static report
      imgs/                   # charts
    model/
      regime_model_best.pkl   # last good model
      thresholds.json         # per-regime EVT or quantile τ
```

---

# 2) Module-by-module mapping (your codebase)

## 2.1 `acm_core_local_2.py` — Ingest → Clean/DQ → Feature Bank → Regimes

**Responsibilities**

* I/O with SQL → `raw_df`
* Time alignment, resample, outlier clamp, gap fill
* Data Quality (DQ) metrics
* Feature engineering
* Dimensionality reduction
* Change-point + clustering (regimes)
* HMM/HSMM smoothing
* Persist `{features, regimes, model}` to artifacts + DB

**Key functions (add/ensure)**

```python
def load_data(equip: str, t0: datetime, t1: datetime) -> pd.DataFrame: ...

def compute_dq(df: pd.DataFrame, key_tags: list[str]) -> dict:
    # flatline%, dropout%, spikes per tag; resample stats for report

def clean_and_align(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    # resample='5s/10s', interpolate='time', cap z-score>6, median filter optional

def build_features(df: pd.DataFrame, win='60s', step='10s') -> pd.DataFrame:
    # rolling mean/std, slope, iqr, spectral bands via STFT, AR residuals, cross-corr

def reduce_dim(X: np.ndarray, var=0.95) -> dict:
    # PCA fit (retain 95% variance). return {'pca', 'Xz'}

def segment_changes(Xz: np.ndarray, pen='aic') -> list[Tuple[s,e]]:
    # ruptures (Pelt/CostRbf) on latent; returns change-point segments

def cluster_regimes(Xz_seg: np.ndarray, algo='kmeans|hdbscan', k_auto=True) -> dict:
    # if kmeans: pick k via gap/silhouette/elbow. return labels + centers

def smooth_regimes(labels: np.ndarray) -> np.ndarray:
    # fit HMM/HSMM (Gaussian emissions on Xz_seg). return smoothed labels

def save_regime_model(path, pca, cluster_model, hmm, meta): ...  # pkl
```

**Default parameters (tunable)**

* Resample: `5–10s` (match historian cadence).
* Outlier clamp: per-tag `μ±6σ` after rolling mean; spikes heuristic: abs diff > `5σ`.
* STFT: 60-s windows, 50% overlap, 4–6 log bands (drop very high-f if not relevant).
* PCA: 95% variance.
* Change-points: `ruptures.Pelt(model="rbf", min_size=win/step, jump=1)`.
* Clustering:

  * Default `HDBSCAN(min_cluster_size≈3–5 mins of samples)` when regimes are irregular.
  * Else `KMeans(k∈[2..8])`, select by **Gap** statistic; guardrail with silhouette.
* HMM: 1st-order, enforce min-duration (HSMM or duration penalty). Transition constraints, e.g. `Idle→Start→Run→Stop`, disallow `Run→Idle` without passing through `Stop` if your process demands it.

**Outputs**

* `features.parquet` (index=time, columns=engineered)
* `regimes.json` (`{time: state_id}` + transitions list)
* `regime_model.pkl` (PCA, clusterer, HMM, metadata)
* Write `ACM.RegimeModel` (pickle as varbinary)

## 2.2 `acm_score_local_2.py` — Regime-aware anomaly scoring

**Responsibilities**

* Load `regime_model_best.pkl` (+ thresholds)
* For each regime: fit/refresh a light **detector** (AE / IF / KDE / One-Class SVM)
* Compute **point**, **window**, and **sequence** scores
* Combine to unified τ score; threshold via EVT/quantiles; merge into events

**Key functions**

```python
def fit_regime_detectors(features: pd.DataFrame, regimes: pd.Series) -> dict:
    # returns {regime_id: model} where model supports .score or .recon_error

def score_stream(features, regimes, models, seq_model=None) -> pd.DataFrame:
    # columns: ['pt_score','win_score','seq_score','tau','regime']

def calibrate_thresholds(scores, method='EVT', alpha=1e-3) -> dict:
    # EVT (GPD) on tail per regime or Q(0.995) quantiles

def merge_anomalies(scores, min_gap='2min', min_len='30s') -> pd.DataFrame:
    # run-length encoding on tau>τ; attach regime, peak τ, top contributing tags
```

**Models (sane defaults)**

* **Per-regime IsolationForest**: `n_estimators=200, max_samples='auto', contamination='auto'`.
* **Sequence model (optional)**: AR( p auto via AIC ) on key projections; residual z-score → `seq_score`.
* **Unified τ**: `tau = w1*pt + w2*win + w3*seq` (normalize scores to [0,1], start with `w=[0.5,0.3,0.2]`).
* **Threshold**: EVT (GPD) with α=0.001 per regime; backstop at Q99.5 if fit unstable.

**Outputs**

* `scores.csv` (time-indexed)
* `anomalies.csv` (`start,end,regime,peak_tau,score_breakdown,top_tags`)
* Upsert `ACM.AnomalyEvent` rows + optional base64 images placeholders (filled by `artifact` module).

## 2.3 `acm_artifact_local.py` — Event families, drift, images, HTML report

**Responsibilities**

* **Family clustering** of anomaly segments using latent embeddings from PCA/AE → HDBSCAN/KMeans → `families.json`.
* **Concept drift** monitors: ADWIN, Page-Hinkley, PSI/KL on feature marginals & regime distribution; emit `drift.json` + `ACM.DriftLog`.
* Generate **charts** via `report_charts.py` (PNG→base64) and compile a **plain** HTML report (tables + charts).
* (Already in your code) build time-series views + per-event snapshots; ensure f-strings are valid (no backslash in `{}`).

**Key functions**

```python
def cluster_anomaly_families(features, anomalies_df) -> dict:
    # collect windows around events, embed (PCA/AE), cluster → {event_id: family_id}

def detect_drift(features, regimes, ref_model) -> list[dict]:
    # ADWIN on tau, PH on projection mean, PSI on regime histogram

def build_report(run_dir, equip, windows, key_tags) -> Path:
    # calls report_charts to create trend PNGs; inline imgs; single clean HTML
```

**Gotchas (fixes you asked earlier)**

* Avoid `f"{…\"…\"…}"` nested quotes w/ backslashes in f-strings. Build with formatted variables **outside** strings first.
* Keep **report basics**: **no cards**, use **tables + charts** only; annotate anomalies and regimes on the trend plots; add **data snapshot** plots for sampled tags.

## 2.4 `report_charts.py` — Static matplotlib (already started)

**Responsibilities**

* Downsample helper (`rolling → stride`)
* Multi-tag trend with regime color bands + anomaly markers
* Spectral density panel if you keep STFT bands
* Per-event zoom plots
* DQ bar charts (flatline%, dropout%, spikes)
* Returns base64 `data:image/png` URIs

**Conventions**

* One chart per figure (no subplots if you don’t want), legible fonts, x-axis as time, shaded regimes `axvspan`.
* Marker policy: red dot for τ≥τ*, hollow for near-miss (τ*−δ).

## 2.5 `acm_brief_local.py` — Machine brief (for operators + LLM)

**Responsibilities**

* Produce **operator-first** summary (`brief.md`) + structured (`brief.json`)
* Include: time window, regimes found, health check (drift), top tags by SHAP/gradients per family, event list.

**Structure**

```json
{
  "equip": "FD FAN",
  "window": {"t0": "...", "t1": "..."},
  "regimes": [{"id":0,"name":"Idle","share":0.23}, ...],
  "detectors": {"r0":"IForest", "r1":"AE"},
  "thresholds": {"r0":0.9962, "r1":0.9948},
  "events": [
    {"id":"E-20251010-001","t0":"...","t1":"...","regime":1,"peak_tau":0.9991,
     "family":"F3","top_tags":["MOTOR_CURR","N1_SPEED","VIB_X"], "explain":"..."}
  ],
  "drift": [{"metric":"PSI_regime","value":0.31,"threshold":0.2,"action":"retrain"}]
}
```

## 2.6 `run_acm.ps1` — Orchestration

* Adds **modes**: `-Train`, `-Score`, `-Report`, `-All`
* Cleans run dir, timestamps run, sets env vars for paths.
* For **batch** across thousands of assets: `-All` reads a CSV of `{equip,t0,t1}` and loops safely with try/catch → writes `RunLog`.

---

# 3) Methods (unsupervised) — exact recipes

## 3.1 Data quality & cleaning

* **Flatline%** per tag: fraction of consecutive repeats > `N_flat=win/step * 3` (e.g., 3 windows).
* **Dropout%**: proportion of NaNs pre-interpolation.
* **Spike count**: `|x_t − med(x_{t−k..t+k})| > 5*mad`.
* **Clamp**: post-clean values to `μ±6σ` (robust μ,σ on sliding window).

**Trigger actions**

* If `Dropout%>20%` or `Flatline%>30%` for a key tag → degrade trust, **exclude from feature bank** for this run and write DQ warning.

## 3.2 Features

* **Stat**: mean, std, skew, kurt over 60-s rolling.
* **Trend**: first difference, linear slope over 60-s.
* **Robust range**: IQR, pct10/50/90.
* **Spectral**: STFT magnitude aggregated to bands (0–0.1, 0.1–0.3, 0.3–0.5 Nyquist).
* **AR residual**: fit AR(p) with AIC; residual z-score.
* **Cross-tag**: rolling corr with key driver (e.g., load).

## 3.3 Regime discovery

1. PCA on standardized features (retain 95% var).
2. Change-points on PCA-latent (rbf cost).
3. Cluster segment embeddings (mean of latent per segment):

   * If behavior is tidy, use **KMeans**, choose **k** via Gap (max Δ), guard with silhouette>0.25.
   * If messy or variable density, **HDBSCAN(min_cluster_size ≈ 3–5 min)**.
4. Smooth with **HMM/HSMM**; enforce min state duration (e.g., 30–90 s) and allowed transitions.

**Outputs**: state sequence `S_t`, transition graph, dwell time stats (used later for health).

## 3.4 Anomaly detection (regime-aware)

* **Detector per regime** (trained on in-regime normal windows):

  * Start with **IsolationForest**; optional swap to **AE** (2×Dense 128-64 bottleneck) if recon error is stable.
* **Context (sequence)**: AR residual or simple seq-AE on latent over 60-s windows.
* **Score fusion**: normalized to [0,1], weighted sum (`w=[0.5,0.3,0.2]`).
* **Thresholding**: per-regime EVT on top 5% scores (fit GPD), α=0.001.
* **Eventization**: run-length encode τ>τ* with `min_len=30 s`, `merge_gap=2 min`. Attach:

  * **Peak τ**, **regime**, **top contributing tags** (via permutation importance on detector or SHAP on AE).

## 3.5 Anomaly families (unsupervised classification)

* For each event, take a fixed **context window** (e.g., ±3 min), embed with **PCA latent** (or AE bottleneck).
* Cluster with **HDBSCAN** (nice for unknown family count).
* Compute **family prototypes** (centroid plots + median tag responses).
* Optionally add **semi-supervised consistency** later: if you label 5–10 events, train a thin classifier with **clean-teacher** consistency on heavy augmentations of windows (jitter, masking, time-warp) to solidify families (keeps labels light).

## 3.6 Drift & health

* **PSI** on regime distribution vs. reference week (`PSI>0.25` warn, `>0.5` alert).
* **ADWIN** on τ stream per regime (detects distribution shift).
* **Page-Hinkley** on PCA component means to detect slow creep.
* **Actions**: `warn → mark model stale`, `alert → auto-retrain` (see Scheduler).

## 3.7 Explainability

* For **IForest**: permutation importance per tag within events; rank tags used in split paths (approx via shadow variables).
* For **AE**: gradient × input or SHAP Kernel on low-dim latent features.
* Attach **top_tags** and a one-liner to each event & family.

---

# 4) Evaluation & KPIs (per run & rolling)

* **Precision proxy** (unsupervised): alert **load** (minutes in alarm / total) should be **< 2–5%** under steady normal operations.
* **Stability**: regime **dwell time CV**; should be stable across weeks unless process changes.
* **Drift alarms per week**: ideally low; spikes → inspect tag calibrations or upstream process changes.
* **Operator alignment**: % events acknowledged as “useful” (from elogbook feedback loop).
* **Latency**: ingestion→report wall time; target **P95 < 5 min** per asset per hour window.

---

# 5) Scaling to thousands of assets

* **Sharding**: partition by `hash(equip) % N_WORKERS`.
* **Cold vs. warm path**:

  * **Warm scoring** every `Δt=5 min` using cached `regime_model_best.pkl`.
  * **Cold retrain** nightly/weekly per asset or on drift alert.
* **Idempotency**: `run_id = equip + t0 + t1 + hash(ver)`; safe re-runs overwrite the same artifact dir.
* **Model registry** (lightweight): keep `model/regime_model_best.pkl` + `model/regime_model_{ver}.pkl`. Promote after drift-free week.
* **Fault tolerance**: try/catch per asset; write `RunLog` even on failure with `error_msg`.

---

# 6) Optional agentic layer (LangGraph/LangChain/DSPy) — if you want “always-on”

**Goal**: keep analyzer running, self-healing, and “explain on demand.”

### Nodes (LangGraph)

1. `Acquire` → SQL pull.
2. `CleanDQ` → DQ gate; if severe, yield “data issue” tool call.
3. `Feature` → features.parquet.
4. `Regime` → ensure `regime_model_best.pkl`, retrain if missing.
5. `Score` → compute τ, events.
6. `Family` → cluster anomalies, update prototypes.
7. `Drift` → monitors; on alert, branch to `Retrain`.
8. `Brief` → build brief.md/json.
9. `Persist` → DB writes + artifacts.
10. `Notify` (tool) → push summary to your MES/elogs.

**Memory**

* **Short-term** (working set): in-graph dict (window-scoped).
* **Long-term**:

  * **Procedural** in `/docs/*.md` (ops SOPs, thresholds policy).
  * **Episodic** in `ACM.AnomalyEvent`, `ACM.RunLog`.
  * **Semantic** in a simple `ACM.KB(key, json)` for tag definitions, equipment metadata.
* If you like **mem0/letta**, wrap DB read/write in their stores for retrieval (RAG over briefs + family prototypes).

**DSPy fit**

* Use DSPy to declaratively specify **prompts** for `Brief` generation and **tools** (SQL, file I/O), while keeping ML pieces in Python. Good for making the summarizer reliable and testable.

---

# 7) Configuration (single YAML/TOML)

Create `acm_config.yaml`:

```yaml
sampling: {period: "10s"}
features:
  window: "60s"
  step: "10s"
  spectral_bands: [ [0.0,0.1], [0.1,0.3], [0.3,0.5] ]
pca: {variance: 0.95}
segmentation: {model: "rbf", min_duration_s: 60}
clustering:
  algo: "hdbscan"        # or kmeans_auto
  k_range: [2,8]
  min_cluster_minutes: 3
hmm:
  min_state_seconds: 60
  allowed_transitions: [["Idle","Start"],["Start","Run"],["Run","Stop"],["Stop","Idle"]]
detectors:
  per_regime: "iforest"  # or "ae"
  fusion_weights: [0.5,0.3,0.2]
thresholds:
  method: "EVT"
  alpha: 0.001
eventization:
  min_len: "30s"
  merge_gap: "120s"
drift:
  psi_regime_warn: 0.25
  psi_regime_alert: 0.5
  adwin_delta: 0.002
report:
  key_tags: ["MOTOR_CURR","N1_SPEED","VIB_X","LOAD"]
```

---

# 8) Operator brief template (content rules)

Top section:

* Window + Equipment + Versions
* DQ summary (any tag excluded?)
* Regimes found (with % time)

Middle:

* Event table: `Start | End | Regime | Peak τ | Family | Top tags | Short note`
* Family table: `Family | Count | Prototype description | Similar past`
* Drift: `Metric | Value | Threshold | Action`

Bottom:

* “What to check” checklist from top tags (auto-generated, deterministic wording).
* Links: `RunLog`, `AnomalyEvent` rows.

---

# 9) Practical defaults (copy/paste into your code as constants)

* `RESAMPLE = '10s'`
* `FEATURE_WIN = '60s'`, `FEATURE_STEP = '10s'`
* `CP_MIN_SEG = 6` windows
* `HDBSCAN_MIN_CLUSTER_SIZE = 18–30` (≈3–5 min @10s)
* `IFOREST_N = 200`, `CONTAM='auto'`
* `EVT_ALPHA = 0.001`
* `EVENT_MIN_LEN = 30s`, `EVENT_MERGE_GAP = 120s`
* Drift: `PSI_WARN=0.25`, `PSI_ALERT=0.5`, `ADWIN_DELTA=0.002`

---

# 10) Run patterns

**Training / Refresh (nightly):**

1. `acm_core_local_2.py --train --equip "FD FAN" --lookback 14d`
2. `acm_score_local_2.py --calibrate --equip "FD FAN"`
3. `acm_artifact_local.py --families --drift --report`
4. Promote `regime_model_best.pkl` if drift clean.

**Online Scoring (every 5–10 min):**

1. `acm_core_local_2.py --prep --equip "FD FAN" --window 60min`
2. `acm_score_local_2.py --score --equip "FD FAN"`
3. `acm_artifact_local.py --report --brief --light`

Batch mode: `run_acm.ps1 -All -Window "60"` reads a CSV of assets and loops.

---

# 11) Quality guardrails (what to alert on)

* **No regimes** or **1 regime only** with silhouette < 0.1 → raise “regime unreliable”.
* **Threshold fit unstable** (EVT fails KS test p<0.05) → fallback to quantile and flag.
* **Top tags empty** for >30% events → run detector explainability self-test; suggest AE switch.
* **PSI_ALERT** or **ADWIN alarm** → auto-retrain schedule + send operator note.

---

# 12) Where each question you asked is answered

* **Detect anomalies (unsupervised)**: §3.4, §2.2 (models, τ, EVT, eventization).
* **Identify machine states**: §3.3, §2.1 (PCA + change-points + clustering + HMM).
* **Classify data (unsupervised)**: §3.5, §2.3 (families via HDBSCAN; optional semi-supervised consistency).
* **Do it all unsupervised (and scalable)**: §§3.*, 5, 6, 7, 10 show the full loop, artifacts, and ops.

---


Say the word and I’ll paste those in next.
