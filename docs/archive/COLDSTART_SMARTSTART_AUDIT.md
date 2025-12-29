# Cold Start and Smart Start Logic Audit

## Executive Summary

This document provides a comprehensive audit of ACM's cold start and smart start logic, analyzing its analytical correctness, model evolution mechanisms, and how the system solves unsupervised learning challenges.

**Overall Assessment: The cold start and smart start logic is analytically sound with some recommendations for improvement.**

---

## Table of Contents

1. [Cold Start Logic Analysis](#1-cold-start-logic-analysis)
2. [Smart Start Logic Analysis](#2-smart-start-logic-analysis)
3. [Model Evolution Analysis](#3-model-evolution-analysis)
4. [Unsupervised Learning Solution](#4-unsupervised-learning-solution)
5. [Analytical Correctness Assessment](#5-analytical-correctness-assessment)
6. [Recommendations](#6-recommendations)

---

## 1. Cold Start Logic Analysis

### 1.1 Overview

The cold start mode enables ACM to bootstrap anomaly detection capability from a single batch of operational data when no historical training set is available.

**Implementation Location:** 
- `core/output_manager.py` (`_load_data_from_sql()`)
- `core/smart_coldstart.py` (`SmartColdstart` class)
- `docs/COLDSTART_MODE.md` (documentation)

### 1.2 Data Splitting Strategy

**Current Implementation:**
```python
# core/output_manager.py, lines 944-961
if is_coldstart:
    # COLDSTART MODE: Split data for initial model training
    split_idx = int(len(df_all) * cold_start_split_ratio)
    train_raw = df_all.iloc[:split_idx].copy()
    score_raw = df_all.iloc[split_idx:].copy()
else:
    # REGULAR BATCH MODE: Use ALL data for scoring
    train_raw = pd.DataFrame()  # Empty train, will be loaded from baseline_buffer
    score_raw = df_all.copy()
```

**Split Ratio:** 60% train / 40% test (configurable via `cold_start_split_ratio`)

### 1.3 Analytical Correctness Assessment

#### Strengths

1. **Temporal Order Preservation**: The split maintains chronological order (first 60% for training), which is correct for time-series anomaly detection. This prevents future data leakage.

2. **Configurable Split Ratio**: The ratio is configurable (0.1 to 0.9), allowing adaptation to different data volumes.

3. **Minimum Sample Validation**: The code validates sufficient samples exist before proceeding:
   ```python
   if len(train_raw) < min_train_samples:
       Console.warn(f"[DATA] Training data ({len(train_raw)} rows) is below recommended minimum")
   ```

4. **Feature Alignment**: Common columns between train and score are properly identified and aligned.

#### Potential Issues

1. **Stationarity Assumption**: The 60/40 split assumes the first 60% of data is representative of normal operating conditions. If the equipment started in an abnormal state, the models will learn the wrong baseline.

   **Recommendation:** Add an option for variance-based or regime-based filtering of training data.

2. **Fixed Split Position**: The split is always at the 60% mark chronologically. This may not align with actual operating regime transitions.

   **Recommendation:** Consider adaptive splitting based on detected regime changes.

3. **No Shuffling Option**: For non-sequential data (rare in industrial settings), there's no shuffle option.

### 1.4 Cold Start Detection Logic

**Implementation:**
```python
# core/smart_coldstart.py, lines 63-119
def check_status(self, required_rows: int = 500, tick_minutes: Optional[int] = None) -> ColdstartState:
    # Auto-detect tick_minutes from data cadence if not provided
    if tick_minutes is None:
        table_name = f"{self.equip_name}_Data"
        data_cadence_seconds = self.detect_data_cadence(table_name)
```

**Cadence Detection:**
```python
# Lines 126-181
def detect_data_cadence(self, table_name: str, sample_hours: int = 24) -> Optional[int]:
    # Get sample of timestamps to detect cadence
    query = f"""
    SELECT TOP 100 EntryDateTime
    FROM dbo.{table_name}
    ORDER BY EntryDateTime
    """
    # Calculate intervals between consecutive timestamps
    intervals = []
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i-1]).total_seconds()
        if delta > 0:  # Skip duplicates
            intervals.append(delta)
    # Find most common interval (mode)
    most_common_interval = interval_counts.most_common(1)[0][0]
```

**Assessment:** The cadence detection using mode of timestamp intervals is analytically sound and robust to outliers.

---

## 2. Smart Start Logic Analysis

### 2.1 Overview

Smart start extends cold start with intelligent retry logic, window expansion, and progress tracking across multiple job runs.

**Implementation Location:** `core/smart_coldstart.py`

### 2.2 Window Expansion Strategy

**Implementation:**
```python
# core/smart_coldstart.py, lines 183-254
def calculate_optimal_window(self, 
                              current_window_end: datetime,
                              required_rows: int = 500,
                              data_cadence_seconds: Optional[int] = None) -> Tuple[datetime, datetime]:
    # Calculate how many minutes needed to get required_rows
    cadence_minutes = data_cadence_seconds / 60
    required_minutes = required_rows * cadence_minutes
    
    # Add 20% buffer for safety
    required_minutes = int(required_minutes * 1.2)
    
    # For coldstart, get the EARLIEST data available
    query = f"SELECT MIN(EntryDateTime) FROM {table_name}"
```

**Key Features:**
1. Uses earliest available data (not most recent)
2. 20% buffer added for safety
3. Falls back to lookback from current time if no data found

### 2.3 Retry Logic with Exponential Window Expansion

**Implementation:**
```python
# Lines 256-406
def load_with_retry(self, output_manager, cfg, initial_start, initial_end, 
                    max_attempts: int = 3, historical_replay: bool = False):
    for attempt in range(1, max_attempts + 1):
        try:
            train, score, meta = output_manager._load_data_from_sql(...)
            rows_loaded = len(train) + len(score)
            
            if rows_loaded >= min_rows:
                return train, score, meta, True
            else:
                # Expand window for next attempt
                if attempt < max_attempts:
                    window_size = (attempt_end - attempt_start).total_seconds() / 60
                    new_window_size = window_size * 2  # Exponential expansion
                    if historical_replay:
                        attempt_end = attempt_start + timedelta(minutes=new_window_size)
                    else:
                        attempt_start = attempt_end - timedelta(minutes=new_window_size)
```

### 2.4 Analytical Correctness Assessment

#### Strengths

1. **Exponential Backoff**: Window doubling is efficient - it quickly expands to find sufficient data without excessive iterations.

2. **Historical Replay Mode**: Correctly expands forward in time during historical replay, backward during live mode.

3. **Progress Tracking**: Progress is persisted to database via stored procedures:
   ```python
   self._update_progress(
       rows_received=rows_loaded,
       data_start=attempt_start,
       data_end=attempt_end,
       success=(rows_loaded >= min_rows)
   )
   ```

4. **Graceful Degradation**: Returns `(None, None, None, False)` when insufficient data, allowing the batch to be marked as NOOP rather than failing.

#### Potential Issues

1. **Max Attempts Fixed at 3**: For very sparse data, 3 attempts may not be enough. Window expansion: 1x -> 2x -> 4x covers only 7x original window.

   **Recommendation:** Make `max_attempts` configurable with a sensible upper bound.

2. **No Upper Bound on Window Size**: Theoretical risk of querying years of data if available.

   **Recommendation:** Add `max_window_hours` configuration parameter.

3. **Stored Procedure Dependency**: The retry logic depends on `usp_ACM_CheckColdstartStatus` and `usp_ACM_UpdateColdstartProgress` stored procedures existing.

---

## 3. Model Evolution Analysis

### 3.1 Overview

ACM implements continuous learning where models evolve with each batch rather than remaining static after initial training.

**Evidence from MODEL_EVOLUTION_PROOF.md:**
- 20 model versions created in ~30 minutes
- 122 total model artifacts saved to SQL ModelRegistry
- Model sizes vary (proving parameter changes, not static copies)
- Forecast state evolved from v157 to v171 (14 versions)

### 3.2 Model Evolution Mechanisms

#### 3.2.1 Version Tracking

**Implementation:** `core/model_persistence.py`
```python
@dataclass
class ForecastState:
    equip_id: int
    state_version: int  # Incremental version number
    model_type: str
    model_params: Dict[str, Any]
    last_retrain_time: str
    training_data_hash: str
```

#### 3.2.2 Continuous Learning Architecture

**Implementation:** `core/acm_main.py`, `docs/CONTINUOUS_LEARNING.md`

```python
# Batch mode detection
BATCH_MODE = _batch_mode()  # os.getenv("ACM_BATCH_MODE", "0") == "1"
CONTINUOUS_LEARNING = _continuous_learning_enabled(cfg, BATCH_MODE)

# Force retraining in continuous learning mode
force_retraining = BATCH_MODE and CONTINUOUS_LEARNING
```

### 3.3 Model Update Frequency Control

**Configuration:**
```python
model_update_interval = int(cl_cfg.get("model_update_interval", 1))
threshold_update_interval = int(cl_cfg.get("threshold_update_interval", 1))
```

**Decision Logic:**
```python
# Models retrain when: batch_num % model_update_interval == 0
# Thresholds update when: batch_num % threshold_update_interval == 0
```

### 3.4 Analytical Correctness Assessment

#### Strengths

1. **Incremental Versioning**: Each successful run increments model version, creating an audit trail.

2. **Config Signature Validation**: Models are invalidated when configuration changes:
   ```python
   def _compute_config_signature(cfg: Dict[str, Any]) -> str:
       relevant_keys = ["models", "features", "preprocessing", "detectors", 
                        "thresholds", "fusion", "regimes", "episodes", "drift"]
       config_subset = {k: cfg.get(k) for k in relevant_keys if k in cfg}
       return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]
   ```

3. **Data Hash Tracking**: Training data hash is stored to detect significant data changes:
   ```python
   training_data_hash: str  # Hash of training window for change detection
   ```

4. **Adaptive Model Selection**: Models like GMM and PCA auto-select optimal parameters:
   ```python
   # GMM: BIC-driven component selection
   if self.gmm_cfg.get("enable_bic_search", True) and safe_k > 2:
       bics = []
       for k_test in k_range:
           gm_test = GaussianMixture(n_components=k_test, ...)
           bics.append(gm_test.bic(Xs))
       safe_k = k_range[np.argmin(bics)]
   ```

#### Model Size Variations (Proof of Real Evolution)

From MODEL_EVOLUTION_PROOF.md:
```
IsolationForest Model:
  v200: 473,305 bytes
  v199: 474,009 bytes  <- Different
  v198: 482,809 bytes  <- Different

GMM Model:
  v200: 9,680 bytes
  v197: 8,824 bytes    <- Fewer components
  v190: 7,952 bytes    <- Fewer components

PCA Model:
  v200: 5,375 bytes
  v197: 4,927 bytes    <- Fewer components
```

---

## 4. Unsupervised Learning Solution

### 4.1 Challenge

Industrial anomaly detection faces the classic unsupervised learning problem:
- No labeled anomalies for training
- Must distinguish normal from abnormal without ground truth
- Equipment behavior evolves over time

### 4.2 ACM's Solution: Multi-Detector Ensemble with Adaptive Calibration

#### 4.2.1 Detector Diversity

ACM uses 6+ complementary unsupervised detectors:

| Detector | Type | What It Catches |
|----------|------|-----------------|
| PCA (SPE) | Subspace | Variance loss, correlation breaks |
| PCA (T2) | Subspace | Distance from normal within principal subspace |
| Mahalanobis | Statistical | Multivariate distance from mean |
| IsolationForest | Density | Isolated points in feature space |
| GMM | Clustering | Low density regions |
| OMR | Reconstruction | Multivariate reconstruction error |
| AR1 | Time-Series | Temporal prediction residuals |

**Implementation:** `core/correlation.py`, `core/outliers.py`, `core/omr.py`, `core/forecasting.py`

#### 4.2.2 Self-Calibrating Z-Scores

Each detector output is calibrated to z-scores using training data statistics:

```python
# core/fuse.py - ScoreCalibrator
class ScoreCalibrator:
    def fit(self, scores: np.ndarray) -> "ScoreCalibrator":
        self.mu = float(np.nanmean(scores))
        self.sigma = float(np.nanstd(scores))
        if self.sigma < 1e-9:
            self.sigma = 1.0
        return self
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        z = (scores - self.mu) / self.sigma
        return np.clip(z, -self.max_z, self.max_z)
```

**Key Insight:** By calibrating each detector to z-scores on training data, ACM establishes what "normal" means for each detection method without requiring labeled data.

#### 4.2.3 Weighted Fusion

Detector z-scores are combined with configurable weights:

```python
# core/fuse.py
def fuse(self, streams: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> np.ndarray:
    weights = cfg.get("fusion", {}).get("weights", {})
    fused = np.zeros(n_rows, dtype=np.float32)
    total_weight = 0.0
    for name, z_scores in streams.items():
        w = float(weights.get(name, 1.0))
        fused += w * z_scores
        total_weight += w
    return fused / max(total_weight, 1e-9)
```

#### 4.2.4 Auto-Tuned Fusion Weights

Weights are auto-tuned using episode separability metrics:

```python
# core/fuse.py - tune_detector_weights()
# Strategy:
# - Split data into train/validation folds
# - For each detector, compute episode separability metrics:
#   * Defect episode detection rate
#   * False positive rate
#   * Mean separation (difference between defect and normal z-scores)
# - Convert metrics to weights using configurable softmax with priors
# - Blend with existing weights using learning rate
```

**Key Insight:** Without labeled anomalies, ACM uses detected episodes as pseudo-labels for weight tuning. Detectors that better separate episodes from normal data get higher weights.

#### 4.2.5 Regime-Aware Detection

Operating regimes are discovered unsupervised via clustering:

```python
# core/regimes.py
# 1. Build feature basis (PCA scores + optional raw tags)
# 2. Auto-select k via silhouette/Calinski-Harabasz scores
# 3. Cluster with MiniBatchKMeans
# 4. Smooth labels to prevent flapping
# 5. Assign health state per regime based on fused z-score distribution
```

**Key Insight:** Per-regime thresholds prevent false positives in high-variance but normal operating modes.

### 4.3 How Unsupervised Learning is Solved

1. **Assumption of Normality During Cold Start**: The first batch is assumed to be predominantly normal. This is standard practice in industrial anomaly detection.

2. **Statistical Baseline Establishment**: Training data statistics (mean, std, covariance, PCA subspace) define "normal".

3. **Distance-Based Anomaly Detection**: Points far from the established baseline (high z-score) are anomalous.

4. **Multi-View Consensus**: Multiple detectors reduce false positives. Only points flagged by multiple methods trigger alerts.

5. **Continuous Adaptation**: Models evolve with data, preventing concept drift from causing false positives.

6. **Episode-Based Validation**: Sustained anomalies (episodes) provide pseudo-labels for self-tuning.

---

## 5. Analytical Correctness Assessment

### 5.1 Cold Start Split (60/40)

**Is it analytically correct?**

**Yes, with caveats:**

1. **Temporal integrity is maintained**: No future leakage.

2. **60% is sufficient for stable estimates**: For most detectors (PCA, GMM, IForest), 60% of typical batch sizes (500+ rows) provides enough samples.

3. **40% validation is meaningful**: Enough data to validate model performance.

**Caveats:**
- Assumes first 60% is representative of normal operation
- Fixed ratio may not be optimal for all data volumes

### 5.2 Window Expansion Strategy

**Is it analytically correct?**

**Yes:**

1. **Exponential backoff is efficient**: 2x expansion reaches large windows quickly.

2. **Using earliest data is correct**: Ensures widest temporal coverage for initial training.

3. **Progress tracking prevents redundant work**: Database state persists across job restarts.

### 5.3 Model Evolution

**Is it analytically correct?**

**Yes:**

1. **Continuous retraining on accumulated data**: Models learn from growing dataset.

2. **Adaptive parameters**: GMM components, PCA dimensions, etc. adjust to data characteristics.

3. **Version tracking**: Enables rollback if needed.

### 5.4 Unsupervised Learning Approach

**Is it analytically correct?**

**Yes, it follows established best practices:**

1. **Ensemble methods**: Combining multiple detectors is proven to reduce false positives.

2. **Z-score calibration**: Standard statistical normalization.

3. **Regime-aware thresholds**: Prevents false positives in multi-mode operation.

4. **Continuous adaptation**: Addresses concept drift.

---

## 6. Recommendations

### 6.1 High Priority

1. **Add Configurable Max Window Size**
   
   Current code has no upper bound on window expansion, risking very large queries.
   
   ```python
   # Recommended addition to smart_coldstart.py
   max_window_hours = float(cfg.get("data", {}).get("max_coldstart_window_hours", 168))  # 7 days default
   if required_minutes / 60 > max_window_hours:
       Console.warn(f"[COLDSTART] Window size limited to {max_window_hours} hours")
       required_minutes = max_window_hours * 60
   ```

2. **Add Anomaly Filtering for Cold Start Training Data**
   
   If first 60% contains anomalies, models learn wrong baseline.
   
   ```python
   # Recommended: Add IQR-based outlier filtering on cold start training data
   def filter_extreme_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
       q1 = df.quantile(0.25)
       q3 = df.quantile(0.75)
       iqr = q3 - q1
       return df[~((df < (q1 - threshold * iqr)) | (df > (q3 + threshold * iqr))).any(axis=1)]
   ```

### 6.2 Medium Priority

3. **Make Max Retry Attempts Configurable**
   
   ```python
   # In config_table.csv
   # EquipID,Category,ParamPath,ParamValue,ValueType
   # 0,data,max_coldstart_attempts,5,int
   ```

4. **Add Regime-Based Training Data Selection**
   
   For multi-regime equipment, consider training only on the "healthiest" detected regime.

5. **Add Cold Start Quality Metrics**
   
   Track and report:
   - Training data variance
   - Number of low-variance features dropped
   - Regime stability during cold start window

### 6.3 Low Priority (Future Enhancements)

6. **Adaptive Split Ratio**
   
   Adjust split ratio based on data volume:
   ```python
   if len(df_all) > 5000:
       split_ratio = 0.8  # More data = more for training
   elif len(df_all) < 500:
       split_ratio = 0.5  # Less data = more for validation
   ```

7. **Cross-Validation for Cold Start**
   
   Use k-fold cross-validation during cold start to get confidence intervals on model quality.

8. **Transfer Learning from Similar Equipment**
   
   Initialize models with weights from similar equipment to accelerate cold start.

---

## Appendix A: Code References

| Component | File | Key Functions |
|-----------|------|---------------|
| Cold Start Split | `core/output_manager.py` | `_load_data_from_sql()` |
| Smart Coldstart | `core/smart_coldstart.py` | `SmartColdstart.load_with_retry()` |
| Cadence Detection | `core/smart_coldstart.py` | `detect_data_cadence()` |
| Model Persistence | `core/model_persistence.py` | `save_forecast_state()`, `load_forecast_state()` |
| PCA Detector | `core/correlation.py` | `PCASubspaceDetector` |
| GMM Detector | `core/outliers.py` | `GMMDetector` |
| Fusion | `core/fuse.py` | `Fuser.fuse()`, `tune_detector_weights()` |
| Regimes | `core/regimes.py` | `build_regime_model()` |

## Appendix B: Configuration Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `cold_start_split_ratio` | `data.cold_start_split_ratio` | 0.6 | Train/test split ratio |
| `min_train_samples` | `data.min_train_samples` | 500 | Minimum rows for training |
| `model_update_interval` | `continuous_learning.model_update_interval` | 1 | Batches between model retraining |
| `threshold_update_interval` | `continuous_learning.threshold_update_interval` | 1 | Batches between threshold updates |
| `max_attempts` | SmartColdstart | 3 | Max retry attempts |

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Author:** Copilot Audit
