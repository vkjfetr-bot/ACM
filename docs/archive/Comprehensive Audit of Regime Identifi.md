# Comprehensive Audit of Regime Identification Script

---

## 1. Architecture & Design

### Strengths
- **Modular separation**: Clear distinction between feature basis construction, model fitting, prediction, and persistence
- **Auto-k selection**: Automated cluster number optimization using silhouette scores prevents manual tuning
- **Dual smoothing approach**: Combines label smoothing (median filter) with transition enforcement (min dwell time)
- **Flexible feature basis**: Supports PCA components + raw sensor tags for dimensionally-aware clustering
- **Health classification**: Maps regimes to operational states (healthy/suspect/critical) using fused scores

### Architectural Concerns

~~**1. Tight Coupling to Legacy Path**~~ (fixed: legacy label path disabled unless allow_legacy_label=True)
```python
def label(score_df, ctx: Dict[str, Any], score_out: Dict[str, Any], cfg: Dict[str, Any]):
    basis_train: Optional[pd.DataFrame] = ctx.get("regime_basis_train")
    # ... modern path
    return _legacy_label(score_df, ctx, out, cfg)  # Fallback
```
- Maintains deprecated `_legacy_label()` for backward compatibility
- Legacy path lacks health labeling, smoothing, and quality metrics
- **Risk**: Production systems may silently fall back to inferior legacy mode

~~**2. State Machine vs. Vectorized Operations**~~ (fixed: transient detection now vectorized with dilated masks)
```python
class _StateMachine:
    def update(self, roc_value: float, changed: bool) -> str:
        # Iterative processing of each sample
```
- Transient detection uses row-by-row iteration (O(n))
- Could be vectorized for 10-100x speedup on large datasets
- **Impact**: Becomes bottleneck for high-frequency data (>10k samples)

~~**3. Configuration Schema Validation**~~ (fixed: expanded schema covers smoothing, transient, health thresholds)
```python
_REGIME_CONFIG_SCHEMA = {
    "regimes.auto_k.k_min": (int, 2, 20, "Minimum clusters"),
    # Only 4 parameters validated
}
```
- Validates only 4 of 20+ configuration parameters
- Missing validation: smoothing, transient detection, health thresholds
- **Risk**: Invalid configs cause runtime failures instead of startup errors

---

## 2. Correctness & Logic Issues

### Critical Issues

~~**1. Degenerate k=1 Handling**~~ (fixed: k<2 now rejected, k=1 fallback removed)
```python
def _fit_kmeans_scaled(...) -> Tuple[...]:
    for k in range(max(2, k_min), k_max + 1):
        # Loop starts at k=2
    
    if best_model_eval is None:
        fallback_k = max(1, min(k_min, n_samples))  # Can return k=1
```
- Auto-k explicitly avoids k=1 in loop
- Fallback allows k=1 when all candidates fail
- **Problem**: KMeans with k=1 is meaningless (all samples in one cluster)
- **Fix**: Enforce `fallback_k = max(2, ...)` and mark as quality failure

~~**2. Transition Smoothing Priority Logic**~~ (fixed: continuity prioritized over health rank)
```python
def _candidate_score(label: int, segment_start: int, segment_end: int) -> Tuple[int, int]:
    health_rank = _HEALTH_PRIORITY.get(health, _HEALTH_PRIORITY["unknown"])
    # Returns (health_rank, -run_length)
```
- Prioritizes healthier states when collapsing short segments
- **Problem**: A 2-sample "healthy" regime between 100-sample "critical" regimes gets expanded
- **Expected**: Should preserve continuity (adjacent label) over health status
- **Fix**: Return `(-run_length, health_rank)` to prioritize persistence

~~**3. Duration Estimation Edge Cases**~~ (fixed: safe fallback for last sample duration)
```python
def _compute_sample_durations(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        diffs = np.diff(values).astype(np.float64) / 1e9
        durations[:-1] = np.where(np.isfinite(diffs) & (diffs >= 0), diffs, fallback)
        durations[-1] = fallback if fallback > 0 else (diffs[-1] if diffs.size else 0.0)
```
- Last sample duration uses `fallback` or `diffs[-1]`
- **Problem**: If `diffs[-1]` is negative or inf, duration becomes invalid
- **Impact**: Metrics like `dwell_seconds` and `stability_score` corrupted
- **Fix**: `durations[-1] = fallback if fallback > 0 and np.isfinite(fallback) else 0.0`

~~**4. Alignment Dimension Mismatch**~~ (fixed: logs once and records skip reason/shape)
```python
def align_regime_labels(new_model: RegimeModel, prev_model: RegimeModel) -> RegimeModel:
    if new_centers.shape[1] != prev_centers.shape[1]:
        Console.warn(f"Feature dimension mismatch...")
        return new_model  # Returns unaligned model
```
- Skips alignment when feature spaces differ
- **Problem**: Next run will retry alignment with same mismatch
- **Fix**: Store alignment attempt in metadata to prevent repeated warnings

### Moderate Issues

~~**5. Silhouette Sampling Bias**~~ (fixed: stratified pre-cluster sampling for auto-k evaluation)
```python
if n_samples > max_eval_samples:
    eval_idx = rng.choice(n_samples, size=max_eval_samples, replace=False)
    X_eval = X_scaled[eval_idx]
```
- Random sampling for large datasets
- **Problem**: May miss rare regimes or boundary regions
- **Better**: Stratified sampling by preliminary cluster labels

~~**6. Health Label Race Condition**~~ (fixed: smoothing no longer uses pre-health maps; health computed after)
```python
def smooth_transitions(..., health_map: Optional[Dict[int, str]] = None):
    # Uses health_map during smoothing
    
def update_health_labels(...) -> Dict[int, Dict[str, Any]]:
    # Computes health labels after smoothing
```
- Smoothing can use health labels that don't exist yet
- **Impact**: First run uses empty `health_map`, second run uses stale labels
- **Fix**: Document that smoothing should happen before `update_health_labels()`

---

## 3. Data Quality & Robustness

### Strengths
- **Comprehensive imputation**: `_finite_impute_inplace()` handles NaN, inf, -inf
- **Robust scaling**: `_robust_scale_clip()` uses IQR instead of std for outlier resistance
- **Input validation**: `_validate_regime_inputs()` checks for empty data, non-numeric columns, low variance

### Weaknesses

~~**7. Implicit Feature Scaling**~~ (fixed: only non-PCA raw tags scaled; PCA columns untouched)
```python
def build_feature_basis(...):
    basis_scaler = StandardScaler()
    basis_scaler.fit(train_basis.values)
    # Scales entire basis (PCA + raw tags)
```
- PCA scores already scaled by PCA process
- Raw sensor tags have arbitrary units
- **Problem**: Double-scaling PCA components, under-scaling raw tags
- **Fix**: Scale only raw tags, not PCA scores

~~**8. Zero-Variance Detection**~~ (fixed: relative-variance check added alongside absolute threshold)
```python
def _validate_regime_inputs(df: pd.DataFrame, name: str = "train_basis") -> List[str]:
    variances = numeric.var(axis=0)
    low_var_cols = variances[variances <= 1e-6].index.tolist()
```
- Uses absolute threshold (1e-6) regardless of feature scale
- **Problem**: Scaled features have variance ≈ 1.0, this catches nothing
- **Fix**: Apply before scaling, or use relative threshold (var/mean_var < 0.01)

~~**9. Missing Data Propagation**~~ (fixed: invalid timestamps now logged before dropping)
```python
def _read_scores_csv(p: Path) -> pd.DataFrame:
    df["timestamp"] = _to_datetime_mixed(df["timestamp"])
    return df[~df.index.isna()]  # Drops invalid timestamps
```
- Silently drops rows with unparseable timestamps
- **Problem**: Misaligned regime labels vs. original data length
- **Fix**: Log dropped row count, validate alignment in calling code

---

## 4. Performance & Scalability

### Bottlenecks

~~**10. Repeated Memory Copies**~~ (fixed: _finite_impute_inplace avoids copies when already float32 contiguous)
```python
def _finite_impute_inplace(X: np.ndarray) -> np.ndarray:
    X = _as_f32(X)  # Copy to float32
    # ... modifications
    return X
```
- Despite "inplace" in name, creates copy via `_as_f32()`
- Called 4+ times per scoring run
- **Impact**: 4x memory overhead for large datasets
- **Fix**: Accept pre-allocated buffer or rename to `_finite_impute()`

~~**11. Quadratic Transition Counting**~~ (fixed: transition counts aggregated via Counter)
```python
for seg_idx, (label_value, start_idx, end_idx) in enumerate(segments):
    if seg_idx > 0:
        prev_label, _, _ = segments[seg_idx - 1]  # O(1) lookup
        # But builds transition dict incrementally
```
- Linear in segments (good), but repeated dict updates
- **Better**: Use `Counter` from collections, update once

~~**12. Unoptimized Smoothing**~~ (fixed: stride-based rolling mode fallback)
```python
def smooth_labels(labels: np.ndarray, passes: int = 1, window: Optional[int] = None):
    for _ in range(iterations):
        for i in range(1, len(smoothed) - 1):  # O(n*passes)
```
- Fallback loop is O(n) per pass when SciPy unavailable
- **Problem**: SciPy failure message suggests this is expected in some deployments
- **Fix**: Implement vectorized mode-finding (moving window with bincount)

~~**13. JSON Serialization Overhead**~~ (fixed: uses orjson when available for metadata writes)
```python
def save_regime_model(model: RegimeModel, models_dir: Path):
    metadata = {
        "stats": {
            str(k): {kk: (float(vv) if isinstance(vv, ...) else str(vv)) ...}
        }
    }
```
- Recursively converts all values to JSON-safe types
- **Impact**: Large stats dicts (1000+ regimes) serialize slowly
- **Fix**: Use `orjson` or `msgpack` for 5-10x speedup

---

## 5. Operational & Production Concerns

### Error Handling

~~**14. Silent Persistence Failures**~~ (fixed: save path now calls `_persist_regime_error`)
```python
def _persist_regime_error(e: Exception, models_dir: Path):
    err_file = models_dir / "regime_persist.errors.txt"
    with err_file.open("w", encoding="utf-8") as f:
        f.write(f"Error type: {type(e).__name__}\n\n{traceback.format_exc()}")
```
- Function defined but never called in the script
- Actual save failures raise exceptions without fallback
- **Fix**: Wrap `save_regime_model()` with try/except calling this handler

~~**15. Model Version Mismatch Handling**~~ (fixed: raises explicit ModelVersionMismatch)
```python
def load_regime_model(models_dir: Path) -> Optional[RegimeModel]:
    version = meta.get("model_version")
    if version and version != REGIME_MODEL_VERSION:
        Console.warn(f"Cached model version {version} mismatches...")
        return None
```
- Returns None on version mismatch
- **Problem**: Calling code may not expect None, causing AttributeError later
- **Fix**: Raise `ModelVersionError` with upgrade instructions

~~**16. Hash Collision Risk**~~ (fixed: deterministic md5-based basis hash when missing)
```python
def fit_regime_model(..., train_hash: Optional[int], ...):
    regime_model.train_hash = train_hash
```
- Uses Python's builtin `hash()` for data fingerprinting
- **Problem**: Non-deterministic across processes (randomized in Python 3.3+)
- **Fix**: Use `hashlib.md5()` or `xxhash` for stable hashing

### Observability

~~**17. Quality Metric Explosion**~~ (fixed: adds aggregated regime_quality_score 0-100)
```python
out["regime_quality_ok"] = quality_ok
out["regime_quality_notes"] = list(regime_model.meta.get("quality_notes", []))
out["regime_sweep_scores"] = list(regime_model.meta.get("quality_sweep", []))
```
- Emits 10+ quality metrics per run
- No aggregation or severity levels
- **Problem**: Monitoring systems struggle with high-cardinality metrics
- **Fix**: Add `regime_quality_score` (0-100) combining all factors

~~**18. Missing Convergence Metrics**~~ (fixed: inertia and n_iter captured in meta)
```python
best_model = MiniBatchKMeans(
    n_clusters=best_k,
    batch_size=...,
    n_init=20,
    random_state=random_state,
)
best_model.fit(X_scaled)
```
- No tracking of convergence iterations or inertia
- **Problem**: Can't detect if KMeans failed to converge
- **Fix**: Check `best_model.n_iter_` and `best_model.inertia_`

---

## 6. Testing & Maintainability

### Test Coverage Gaps

~~**19. Edge Case Validation**~~ (fixed: type-coercing _cfg_get protects against wrong types)
```python
def _cfg_get(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
```
- No validation that retrieved value matches expected type
- **Missing tests**:
  - Config with string value for int parameter
  - Nested config with missing intermediate keys
  - List values where scalars expected

~~**20. Smoothing Integration Tests**~~ (fixed: added regression tests for smoothing + dwell order/length)
```python
labels = smooth_labels(labels, passes=passes)
labels = smooth_transitions(labels, timestamps=ts_pred, ...)
```
- Two-stage smoothing with order dependency
- **Missing tests**:
  - Verify smoothing doesn't create new regime IDs
  - Verify total sample count unchanged
  - Verify smoothing commutes (order independence)

### Code Duplication

~~**21. Redundant Feature Scaling**~~ (fixed: skip second scaling when basis is already normalized)
```python
# In build_feature_basis():
basis_scaler.fit(train_basis.values)
train_basis = pd.DataFrame(basis_scaler.transform(train_basis.values), ...)

# In fit_regime_model():
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
- Feature basis scaled, then re-scaled in model fitting
- **Problem**: Loses interpretability, compounds numerical errors
- **Fix**: Pass `pre_scaled=True` flag to `fit_regime_model()`

~~**22. Persistence Logic Duplication**~~ (fixed: shared metadata helper used for file + state persistence)
```python
def save_regime_model(model: RegimeModel, models_dir: Path):
    # Saves to joblib + json

def regime_model_to_state(model: RegimeModel, ...):
    # Saves to database-compatible state
```
- Two separate serialization paths
- **Risk**: Divergence between file-based and DB-based persistence
- **Fix**: Make `regime_model_to_state()` call `save_regime_model()` internally

---

## 7. Security & Privacy

### Low-Risk Findings

~~**23. Path Traversal (Low Risk)**~~ (fixed: rejects episode/score paths outside workspace)
```python
def _read_episodes_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame(columns=["start_ts", "end_ts"])
    df = pd.read_csv(p, ...)
```
- Accepts `Path` objects from caller without validation
- **Scenario**: Attacker-controlled `ctx.run_dir` could read arbitrary files
- **Mitigation**: System design likely restricts `ctx.run_dir` to safe locations
- **Fix**: Add `assert p.is_relative_to(safe_base_dir)`

~~**24. Resource Exhaustion (Medium Risk)**~~ (fixed: auto-k sweep capped via max_models budget)
```python
def _fit_kmeans_scaled(...):
    for k in range(k_min, k_max + 1):
        km_eval = MiniBatchKMeans(...)
        km_eval.fit(X_eval)  # Unbounded k_max
```
- Config allows `k_max=40` (default)
- **Problem**: O(k) model fitting, each with O(n) complexity
- **Impact**: 40 iterations * 10k samples = 400k KMeans iterations
- **Fix**: Add timeout or total computation budget

---

## 8. Priority Recommendations

### Critical (Fix Immediately)
~~1. **Enforce k≥2 in all paths** - Prevents degenerate single-cluster models~~
~~2. **Fix transition smoothing priority** - Restore temporal continuity~~
~~3. **Validate duration computation** - Prevent metric corruption~~
~~4. **Call `_persist_regime_error()` in save path** - Enable failure diagnosis~~

### High (Fix Before Production)
5. **Separate PCA and raw tag scaling** - Correct feature weighting
6. **Vectorize transient state machine** - 10-100x speedup
7. **Add model version migration** - Prevent silent downgrades
8. **Use deterministic hashing** - Enable distributed caching

### Medium (Technical Debt)
9. **Consolidate persistence paths** - Reduce maintenance burden
10. **Add convergence monitoring** - Detect KMeans failures
11. **Implement stratified auto-k sampling** - Better rare regime detection
12. **Expand config validation** - Catch errors at startup

### Low (Nice to Have)
13. **Vectorize smoothing fallback** - Eliminate SciPy dependency
14. **Add quality score aggregation** - Simplify monitoring
15. **Optimize JSON serialization** - Faster save/load
16. **Add integration tests** - Verify smoothing correctness

---

## 9. Code Quality Score

| Category | Score | Notes |
|----------|-------|-------|
| **Architecture** | 7/10 | Modular design, but legacy coupling |
| **Correctness** | 6/10 | Multiple logic errors in edge cases |
| **Performance** | 6/10 | Bottlenecks in state machine, I/O |
| **Robustness** | 8/10 | Good input validation, needs error handling |
| **Maintainability** | 7/10 | Clear structure, some duplication |
| **Testing** | 4/10 | Missing edge case and integration tests |
| **Security** | 8/10 | Low-risk issues only |
| **Documentation** | 6/10 | Docstrings present, missing architecture docs |

**Overall: 6.5/10** - Solid foundation with critical correctness issues requiring immediate attention.
