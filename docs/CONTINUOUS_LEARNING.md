# Continuous Learning Architecture

## Overview

ACM implements a **continuous learning architecture** where models and thresholds evolve as data accumulates batch after batch. This is fundamentally different from traditional machine learning systems that train once and score many times.

## Architecture Philosophy

### Traditional ML Systems (Train/Test Split)
```
┌─────────────────────────────────────────────────────┐
│  Batch 0: Train models → Calculate thresholds      │
│  Cache models ✓                                     │
├─────────────────────────────────────────────────────┤
│  Batch 1-N: Load cached models → Score only        │
│  No retraining, No threshold updates               │
└─────────────────────────────────────────────────────┘
```

### ACM Continuous Learning
```
┌─────────────────────────────────────────────────────┐
│  Every Batch:                                       │
│  1. Validate current data against existing model   │
│  2. Retrain models on accumulated data (train+score)│
│  3. Recalculate thresholds on growing dataset      │
│  4. Update persisted models                         │
│  5. Move to next batch                              │
└─────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. **No Fixed Train/Test Split**
- **`train`**: Accumulated historical data (grows each batch)
- **`score`**: Current batch window to analyze
- Each batch adds to the accumulated dataset
- Models continuously learn from all data seen so far

### 2. **Sliding Window Learning**
- Coldstart phase: Accumulate minimum required data
- After coldstart: Each batch = analysis + retraining
- Models evolve as patterns emerge in accumulated data
- Thresholds adapt as distribution statistics stabilize

### 3. **Batch Number Tracking**
- `sql_batch_runner.py` sets `ACM_BATCH_NUM` environment variable
- Enables frequency control: update every N batches instead of every batch
- Balances computation cost vs. adaptation speed

## Implementation Details

### Batch Mode Detection

**Environment Variables (set by `sql_batch_runner.py`):**
```python
ACM_BATCH_MODE=1       # Indicates running under batch runner
ACM_BATCH_NUM=71       # Current batch number (0-indexed)
```

**Detection Logic (`acm_main.py`):**
```python
def _batch_mode() -> bool:
    """Detect if running under sql_batch_runner (continuous learning mode)."""
    return bool(os.getenv("ACM_BATCH_MODE", "0") == "1")

def _continuous_learning_enabled(cfg: Dict[str, Any], batch_mode: bool) -> bool:
    """Check if continuous learning is enabled for this run."""
    if batch_mode:
        return cfg.get("continuous_learning", {}).get("enabled", True)
    return cfg.get("continuous_learning", {}).get("enabled", False)
```

### Model Retraining Logic

**Cache Disabling:**
```python
# Line ~660 in acm_main.py
force_retraining = BATCH_MODE and CONTINUOUS_LEARNING
use_cache = cfg.get("models", {}).get("use_cache", True) and not force_retraining

if force_retraining:
    Console.info("[MODEL] Continuous learning enabled - models will retrain on accumulated data")
```

**Model Fitting (Lines 1535-1590):**
- If `use_cache=False`, all detectors refit on accumulated `train` data
- Models: AR1, PCA, Mahalanobis, IForest, GMM, OMR
- Regimes: GMM clustering on accumulated basis features
- Calibrators: Fit on accumulated detector scores

### Threshold Calculation

**Standalone Function:**
```python
def _calculate_adaptive_thresholds(
    fused_scores: np.ndarray,
    cfg: Dict[str, Any],
    equip_id: int,
    output_manager: Optional[Any],
    train_index: Optional[pd.Index] = None,
    regime_labels: Optional[np.ndarray] = None,
    regime_quality_ok: bool = False
) -> Dict[str, Any]:
    """Calculate adaptive thresholds from fused z-scores."""
```

**Execution Logic (After Fusion):**
```python
# Line ~2520 in acm_main.py
with T.section("thresholds.adaptive"):
    should_update_thresholds = False
    
    # Update conditions:
    is_first_threshold_calc = coldstart_complete and not hasattr(cfg, '_thresholds_calculated')
    interval_reached = (batch_num % threshold_update_interval == 0)
    
    if is_first_threshold_calc:
        should_update_thresholds = True
    elif CONTINUOUS_LEARNING and interval_reached:
        should_update_thresholds = True
    
    if should_update_thresholds:
        # Calculate on accumulated data (train + score)
        accumulated_data = pd.concat([train, score], axis=0)
        accumulated_fused = calculate_fusion_on_accumulated_data(...)
        _calculate_adaptive_thresholds(...)
```

### Frequency Control

**Configuration Options:**
```csv
# configs/config_table.csv
0,continuous_learning,model_update_interval,1,int,...
0,continuous_learning,threshold_update_interval,1,int,...
```

**Usage Examples:**
- `model_update_interval=1`: Retrain every batch (maximum adaptation)
- `model_update_interval=5`: Retrain every 5 batches (reduced computation)
- `threshold_update_interval=10`: Update thresholds every 10 batches

**Decision Logic:**
```python
# Models
if batch_num % model_update_interval == 0:
    force_retraining = True  # Retrain models
else:
    use_cache = True  # Load cached models

# Thresholds
if batch_num % threshold_update_interval == 0:
    recalculate_thresholds()  # Update thresholds
else:
    skip_threshold_update()  # Use existing thresholds
```

## Configuration

### Enable/Disable Continuous Learning

**Global Config (`configs/config_table.csv`):**
```csv
EquipID,Category,ParamPath,ParamValue,ValueType,...
0,continuous_learning,enabled,True,bool,...
```

**Per-Equipment Override:**
```csv
1,continuous_learning,enabled,False,bool,...  # FD_FAN: disable
2,continuous_learning,enabled,True,bool,...   # GAS_TURBINE: enable
```

### Update Intervals

**Aggressive (High Computation):**
```csv
0,continuous_learning,model_update_interval,1,int,...
0,continuous_learning,threshold_update_interval,1,int,...
```
- Models retrain every batch
- Thresholds update every batch
- Maximum adaptation speed
- Highest computation cost

**Balanced (Medium Computation):**
```csv
0,continuous_learning,model_update_interval,5,int,...
0,continuous_learning,threshold_update_interval,3,int,...
```
- Models retrain every 5 batches
- Thresholds update every 3 batches
- Good balance between adaptation and cost

**Conservative (Low Computation):**
```csv
0,continuous_learning,model_update_interval,10,int,...
0,continuous_learning,threshold_update_interval,10,int,...
```
- Models retrain every 10 batches
- Thresholds update every 10 batches
- Slower adaptation, lower cost

## SQL Integration

### ACM_ThresholdMetadata Table

**Schema:**
```sql
CREATE TABLE ACM_ThresholdMetadata (
    ThresholdID INT IDENTITY(1,1) PRIMARY KEY,
    EquipID INT NOT NULL,
    ThresholdType NVARCHAR(50) NOT NULL,
    ThresholdValue FLOAT NOT NULL,
    CalculationMethod NVARCHAR(100),
    SampleCount INT,
    TrainStart DATETIME2,
    TrainEnd DATETIME2,
    ConfigSignature NVARCHAR(100),
    Notes NVARCHAR(MAX),
    CreatedAt DATETIME2 DEFAULT GETDATE()
);
```

**Population:**
- Every batch (or at intervals): Write new threshold rows
- `EquipID`: Links to ACM_Equipment table
- `ThresholdType`: 'fused_alert_z', 'fused_warn_z'
- `ThresholdValue`: Calculated threshold (global or per-regime dict)
- `SampleCount`: Number of accumulated samples used
- `TrainStart/TrainEnd`: Timestamp range of accumulated data
- `ConfigSignature`: MD5 hash of threshold config
- `CreatedAt`: Timestamp of calculation

**Grafana Dashboard:**
- Panel 30: Threshold comparison (FusedZ vs adaptive vs hardcoded 3.0)
- Panel 31: Active threshold metadata (current values)
- Panel 32: Threshold evolution timeline (historical changes)

## Workflow Example

### Batch Processing Flow

**Coldstart (Batch 0):**
```
1. Accumulate minimum required data (50+ rows)
2. Fit models on accumulated data
3. Calculate initial thresholds
4. Save models and thresholds
5. Mark coldstart_complete = True
```

**Continuous Learning (Batch 1-N):**
```
For each batch:
  1. Load accumulated data (train) + current batch (score)
  2. Check if retraining due: batch_num % model_update_interval == 0
  3. If due:
     - Retrain all models on accumulated data
     - Save updated models
  4. Score current batch with models
  5. Calculate fusion on accumulated data (train + score)
  6. Check if threshold update due: batch_num % threshold_update_interval == 0
  7. If due:
     - Calculate adaptive thresholds on accumulated fusion
     - Write to ACM_ThresholdMetadata
     - Update cfg with new thresholds
  8. Use thresholds for health labels
  9. Move to next batch
```

### Example: 100 Batches with Intervals

**Config:**
```
model_update_interval = 5
threshold_update_interval = 3
```

**Execution:**
```
Batch 0: Coldstart → Fit models, Calculate thresholds
Batch 1: Score only (use cached models and thresholds)
Batch 2: Score only
Batch 3: Score + Update thresholds
Batch 4: Score only
Batch 5: Score + Retrain models
Batch 6: Score + Update thresholds
Batch 7: Score only
Batch 8: Score only
Batch 9: Score + Update thresholds
Batch 10: Score + Retrain models + Update thresholds
...
```

## Monitoring

### Console Logs

**Continuous Learning Enabled:**
```
[CFG] batch_mode=True  |  continuous_learning=True
[CFG] model_update_interval=1  |  threshold_update_interval=1
[MODEL] Continuous learning enabled - models will retrain on accumulated data
```

**Threshold Updates:**
```
[THRESHOLD] Update interval reached (batch 5, interval=5)
[THRESHOLD] Calculating thresholds on accumulated data (train + score)
[THRESHOLD] Global thresholds: alert=1.234, warn=0.617 (method=mad, conf=p99.7)
```

**Threshold Skips:**
```
[THRESHOLD] Skipping threshold update (batch 6, next update at batch 10)
```

### SQL Queries

**Check Threshold Updates:**
```sql
SELECT 
    ThresholdID,
    ThresholdType,
    ThresholdValue,
    SampleCount,
    CreatedAt
FROM ACM_ThresholdMetadata
WHERE EquipID = 1
ORDER BY CreatedAt DESC;
```

**Monitor Update Frequency:**
```sql
SELECT 
    ThresholdType,
    COUNT(*) AS UpdateCount,
    MIN(CreatedAt) AS FirstUpdate,
    MAX(CreatedAt) AS LastUpdate,
    DATEDIFF(MINUTE, MIN(CreatedAt), MAX(CreatedAt)) / COUNT(*) AS AvgMinutesBetweenUpdates
FROM ACM_ThresholdMetadata
WHERE EquipID = 1
GROUP BY ThresholdType;
```

## Performance Considerations

### Computation Cost

**Every Batch Retraining (interval=1):**
- **Time**: 30-60 seconds per batch (vs 5-10 seconds score-only)
- **CPU**: High utilization during model fitting
- **Memory**: Accumulated data grows linearly
- **Use Case**: Critical equipment requiring maximum adaptation

**Interval-Based Retraining (interval=5-10):**
- **Time**: Average 10-15 seconds per batch
- **CPU**: Spikes every N batches
- **Memory**: Same accumulation, periodic cleanup possible
- **Use Case**: Standard equipment with stable patterns

### Optimization Strategies

**1. Adaptive Intervals:**
```python
# Increase interval if patterns stable
if drift_score < 0.5:
    model_update_interval = 10
else:
    model_update_interval = 1
```

**2. Selective Retraining:**
```python
# Only retrain detectors with high error
if ar1_error > threshold:
    retrain_ar1()
if pca_error > threshold:
    retrain_pca()
```

**3. Parallel Batch Processing:**
```python
# Process multiple equipment in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_batch, equip) for equip in equipment_list]
```

## Troubleshooting

### Issue: Thresholds Not Updating

**Symptoms:**
- `ACM_ThresholdMetadata` table empty or no new rows
- Dashboard shows "No data"

**Diagnosis:**
```python
# Check batch mode detection
echo $env:ACM_BATCH_MODE  # Should be "1"

# Check continuous learning enabled
grep "continuous_learning.enabled" configs/config_table.csv  # Should be "True"

# Check logs for threshold section
grep "\[THRESHOLD\]" logs/acm.log
```

**Solutions:**
1. Ensure `sql_batch_runner.py` sets `ACM_BATCH_MODE=1`
2. Verify `continuous_learning.enabled=True` in config
3. Check `threshold_update_interval` not too large
4. Verify `output_manager.sql_client` is connected

### Issue: Models Not Retraining

**Symptoms:**
- Logs show "Using cached detectors from previous training run"
- Models not adapting to new data patterns

**Diagnosis:**
```python
# Check force_retraining flag
grep "force_retraining" logs/acm.log

# Check use_cache logic
grep "use_cache" logs/acm.log
```

**Solutions:**
1. Set `continuous_learning.enabled=True`
2. Reduce `model_update_interval` (try 1)
3. Delete `.sql_batch_progress.json` to reset coldstart
4. Clear model cache manually

### Issue: High Computation Time

**Symptoms:**
- Batches taking > 60 seconds each
- CPU utilization > 80%
- Batch processing stalling

**Diagnosis:**
```python
# Check timing sections
grep "section.*elapsed" logs/acm.log

# Identify slow detectors
grep "fit\." logs/acm.log
```

**Solutions:**
1. Increase `model_update_interval` to 5-10
2. Increase `threshold_update_interval` to 5-10
3. Disable heavy detectors (GMM, OMR) if not needed
4. Reduce PCA components or IForest estimators
5. Enable Polars backend for fast_features

## Future Enhancements

### 1. Drift-Triggered Retraining
- Monitor drift score continuously
- Trigger retraining when drift exceeds threshold
- Reset intervals after major drift events

### 2. Forgetting Mechanisms
- Exponential decay weighting for old data
- Sliding window with max accumulation size
- Adaptive forgetting based on stationarity

### 3. Multi-Stage Learning
- Fast adaptation (small interval) during unstable periods
- Slow adaptation (large interval) during stable periods
- Automatic interval adjustment based on stability metrics

### 4. Distributed Batch Processing
- Process multiple equipment in parallel
- Shared model cache across workers
- Coordinated threshold updates

## References

- **Analytics Backbone.md**: Detector architecture and fusion logic
- **COLDSTART_MODE.md**: Initial data accumulation phase
- **BATCH_PROCESSING.md**: sql_batch_runner workflow
- **CHANGELOG.md**: Implementation history
- **Task Backlog.md**: Continuous learning implementation tasks
