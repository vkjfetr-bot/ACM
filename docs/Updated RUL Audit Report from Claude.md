# Updated RUL Audit Report from Claude

## Critical Issues Found

### 1. **AR(1) Model - Incorrect Variance Calculation**
**Location**: `AR1Model.fit()` lines 500-510

```python
# Current (INCORRECT):
cov = float(np.dot(yc[1:], yc[:-1]))
var = float(np.dot(yc[:-1], yc[:-1]))
self.phi = np.clip(cov / (var + 1e-9), -0.99, 0.99)
```

**Issues**:
- Covariance calculation is missing division by sample size
- Variance should be `np.var(yc[:-1])` or normalized dot product
- This leads to scale-dependent phi estimates

**Correct approach**:
```python
n = len(yc) - 1
cov = np.sum(yc[1:] * yc[:-1]) / n
var = np.var(yc[:-1])
self.phi = np.clip(cov / (var + 1e-9), -0.99, 0.99)
```

### 2. **AR(1) Forecast Variance - Mathematically Incorrect**
**Location**: `AR1Model.predict()` lines 535-542

```python
# Current (INCORRECT):
if abs(self.phi) < 0.999:
    var_mult = (1 - self.phi ** (2 * h)) / (1 - self.phi**2 + 1e-9)
else:
    var_mult = h
```

**Issues**:
- Formula is correct for variance of h-step ahead prediction in AR(1) **without drift**
- With drift term, variance grows faster (linear + AR component)
- Should account for drift uncertainty

**Correct approach**:
```python
# AR(1) component variance
if abs(self.phi) < 0.999:
    var_ar = (1 - self.phi ** (2 * h)) / (1 - self.phi**2 + 1e-9)
else:
    var_ar = h

# Drift adds quadratic uncertainty growth
var_drift = (h * self.step_sec / 3600.0) ** 2 * 0.1  # Drift uncertainty
var_mult = var_ar + var_drift
```

### 3. **Exponential Model - Inappropriate Offset Calculation**
**Location**: `ExponentialDegradationModel.fit()` line 567

```python
offset = float(np.min(y)) - 1.0  # Stabilize for log
```

**Issues**:
- Arbitrary `-1.0` is not statistically justified
- If `min(y) = 71`, offset = 70, meaning you're modeling `exp(decay) + 70`
- This assumes degradation asymptotes to 70, which contradicts threshold=70
- Should estimate offset from data or set to 0

**Correct approach**:
```python
# Option 1: Estimate asymptotic level from late-stage data
offset = float(np.percentile(y[-10:], 5))  # 5th percentile of recent values

# Option 2: Set to zero (pure exponential decay)
offset = 0.0

# Option 3: Estimate via nonlinear optimization
```

### 4. **Weibull Model - Power Law Misspecification**
**Location**: `WeibullInspiredModel.fit()` lines 636-665

**Issues**:
- Model is `h(t) = h0 - k * t^β` (degradation form)
- True Weibull reliability: `R(t) = exp(-(t/λ)^β)`
- Current form can produce negative health (no lower bound)
- Shape parameter β optimization is good, but model form is questionable

**Recommendation**:
```python
# Option 1: Add lower bound constraint
forecast = np.maximum(self.h0 - self.k * (t**self.beta), 0.0)

# Option 2: Use true Weibull hazard rate
# λ(t) = (β/η) * (t/η)^(β-1)
# More complex but theoretically sound
```

### 5. **Ensemble Variance Calculation - Double Counting Risk**
**Location**: `RULModel.forecast()` lines 779-784

```python
ensemble_var_pred = np.sum(weights[:, None] * (stds**2), axis=0)
disagreement = predictions - ensemble_mean[None, :]
ensemble_var_disagreement = np.sum(weights[:, None] * (disagreement**2), axis=0)
ensemble_var = ensemble_var_pred + ensemble_var_disagreement
```

**Issue**:
- Adding prediction variance + disagreement variance can **double-count** uncertainty
- If models disagree due to inherent uncertainty, this is already in their `stds`
- Standard ensemble variance: `Var = E[Var] + Var[E]` where:
  - `E[Var]` = average prediction variance (already have)
  - `Var[E]` = variance of means (disagreement)
  
**This is actually CORRECT** for the law of total variance, but need to verify models' `stds` don't already include disagreement.

### 6. **Normal CDF Approximation - Accuracy Concerns**
**Location**: `norm_cdf()` lines 65-67

```python
def norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(z * np.sqrt(2.0 / np.pi)))
```

**Issues**:
- This is a rough approximation, max error ~0.0001
- For critical safety systems, use `scipy.stats.norm.cdf()`
- Current approx: Φ(z) ≈ 0.5(1 + tanh(z√(2/π)))

**Impact**: Low for most cases, but consider using exact CDF.

### 7. **Failure Probability Calculation - Missing Hazard Rate**
**Location**: `compute_failure_distribution()` lines 796-823

```python
z = (threshold - health_mean) / (health_std + 1e-9)
failure_prob = norm_cdf(z)
```

**Issue**:
- This computes `P(health(t) < threshold)` assuming Gaussian health
- **NOT** the hazard rate or survival probability
- For RUL, you want cumulative failure probability, not instantaneous
- Should compute: `P(failure before time t) = P(first crossing before t)`

**Correct approach for first-passage time**:
```python
# For AR(1) with drift crossing a barrier, use first-passage distribution
# Approximate: P(T_fail < t) ≈ Φ((threshold - μ(t)) / σ(t))
# But this ignores path dependency (could cross, then recover)
```

This is a **major theoretical issue** - the current method treats each time point independently, ignoring the sequential nature of degradation.

### 8. **Confidence Calculation - Ad-hoc Weighting**
**Location**: `compute_confidence()` lines 1166-1227

```python
confidence *= ci_confidence  # Factor 1
confidence *= 0.7 + 0.3 * agreement_score  # Factor 2
confidence *= calibration_score  # Factor 3
confidence *= quality_mult  # Factor 4
```

**Issues**:
- Multiplicative combination has no statistical justification
- Why `0.7 + 0.3 * agreement`? Why not `agreement` alone?
- Calibration score is binary (1.0 or 0.7) - too coarse
- Quality multipliers are arbitrary

**Better approach**:
```python
# Use weighted sum with normalized factors
conf = (0.4 * ci_confidence + 
        0.3 * agreement_score + 
        0.2 * calibration_score + 
        0.1 * quality_mult)
```

### 9. **RUL Multipath - Flawed Logic**
**Location**: `compute_rul_multipath()` lines 856-932

```python
# Path 1: Trajectory crossing (mean forecast < threshold)
# Path 2: Hazard crossing (failure probability >= 50%)
rul_final = min(available_ruls)  # Take minimum
```

**Issues**:
- Path 1 uses **mean** crossing (50th percentile)
- Path 2 uses **50% failure probability** (also 50th percentile)
- These should give **identical** results for Gaussian forecast!
- Only differ due to approximation errors
- Path 3 (energy) is unused

**Why this happens**:
```
P(health < threshold) = 0.5 
⟺ Φ((threshold - μ) / σ) = 0.5
⟺ (threshold - μ) / σ = 0
⟺ μ = threshold
```

So trajectory and hazard crossings occur at the same time!

**Recommendation**: Use different percentiles or methods:
- Path 1: Mean crossing (50th percentile) - "expected" RUL
- Path 2: 90th percentile crossing - "conservative" RUL  
- Path 3: 10th percentile crossing - "optimistic" RUL

### 10. **Learning State - Not Actually Used for Learning**
**Location**: `LearningState` class and `run_rul()` function

**Issue**:
- Learning state is loaded and saved, but **never updated**
- Weights remain at default values
- No actual online learning occurs (despite `enable_online_learning` flag)
- Lines 1080-1096 show the update logic exists, but it's never called

**Missing**: Actual learning update after new observations arrive.

---

## Statistical Correctness Issues

### 11. **No Handling of Censored Data**
If equipment is maintained before failure, all observations are **right-censored**. Standard regression assumes complete failure observations, leading to:
- Overestimation of RUL
- Underestimation of failure risk

**Fix**: Use survival analysis with censoring indicators.

### 12. **Independence Assumption Violation**
All three models assume:
- Independent residuals (AR1 addresses this)
- Stationary covariance structure
- No regime changes

Real equipment has:
- Maintenance interventions (sudden health jumps)
- Operating condition changes
- Seasonal patterns

**Fix**: Add change-point detection or regime-switching models.

### 13. **No Model Selection Criteria**
Current code fits all three models and ensembles them. Should:
- Use AIC/BIC for model selection
- Check residual diagnostics
- Validate stationarity assumptions (Augmented Dickey-Fuller test)

---

## Recommendations

### High Priority
1. **Fix AR(1) covariance calculation** (critical math error)
2. **Fix failure probability to account for first-passage time** (major theoretical issue)
3. **Implement actual online learning** (currently dormant)
4. **Add lower bounds to Weibull model** (prevents negative health)

### Medium Priority
5. **Revise confidence calculation** (use principled weighting)
6. **Differentiate multipath methods** (currently redundant)
7. **Add censoring support** (for real-world applicability)

### Low Priority
8. **Use exact normal CDF** (scipy instead of tanh approximation)
9. **Add model diagnostics** (AIC, residual plots)
10. **Implement change-point detection** (for maintenance events)

---

## Code Quality Issues

- **Line 1080**: `import os` is used but not imported at module level
- **Lines 500-542**: AR1 drift calculation mixes time scales (seconds vs hours) - verify units
- **Lines 1166-1227**: Magic numbers everywhere (0.7, 0.3, 0.8, etc.) - should be config parameters

