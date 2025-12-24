#!/usr/bin/env python3
"""Add _build_sensor_context helper function."""

# Read the file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the location to insert (before _update_baseline_buffer function)
insert_marker = "def _update_baseline_buffer("

# New helper function
new_code = '''def _build_sensor_context(
    train_numeric: pd.DataFrame,
    score_numeric: pd.DataFrame,
    frame: pd.DataFrame,
    omr_contributions_data: Optional[Dict[str, Any]],
    regime_model: Any,
    equip: str,
    T: Timer,
) -> Optional[Dict[str, Any]]:
    """Build sensor analytics context for downstream consumers.
    
    Calculates z-scores, percentiles, and other statistics for sensors
    to support visualization and episode attribution.
    
    Args:
        train_numeric: Training data for computing baseline statistics.
        score_numeric: Scoring data for current window.
        frame: Fused scores frame.
        omr_contributions_data: OMR detector contributions data.
        regime_model: Regime model for metadata.
        equip: Equipment name for logging.
        T: Timer for profiling.
    
    Returns:
        Dictionary with sensor context or None if failed.
    """
    sensor_context: Optional[Dict[str, Any]] = None
    
    with T.section("sensor.context"):
        try:
            if (
                isinstance(train_numeric, pd.DataFrame)
                and isinstance(score_numeric, pd.DataFrame)
                and len(train_numeric)
                and len(score_numeric)
            ):
                common_cols = [col for col in score_numeric.columns if col in train_numeric.columns]
                if common_cols:
                    train_baseline = train_numeric[common_cols].copy()
                    score_baseline = score_numeric[common_cols].copy()
                    train_mean = train_baseline.mean()
                    # Prevent division by zero with safe epsilon fallback
                    train_std = train_baseline.std()
                    train_std = train_std.replace(0.0, np.nan).fillna(1e-10)
                    valid_cols = train_std[train_std > 1e-10].index.tolist()
                    if valid_cols:
                        train_mean = train_mean[valid_cols]
                        train_std = train_std[valid_cols]
                        score_baseline = score_baseline[valid_cols]
                        score_aligned = score_baseline.reindex(frame.index)
                        score_aligned = score_aligned.apply(pd.to_numeric, errors="coerce")
                        sensor_z = (score_aligned - train_mean) / train_std
                        sensor_z = sensor_z.replace([np.inf, -np.inf], np.nan)
                        sensor_context = {
                            "values": score_aligned,
                            "z_scores": sensor_z,
                            "train_mean": train_mean,
                            "train_std": train_std,
                            "train_p95": train_baseline[valid_cols].quantile(0.95),
                            "train_p05": train_baseline[valid_cols].quantile(0.05),
                            "omr_contributions": omr_contributions_data,
                            "regime_meta": regime_model.meta if regime_model else {}
                        }
        except Exception as sensor_ctx_err:
            Console.warn(f"Failed to build sensor analytics context: {sensor_ctx_err}", component="SENSOR",
                         equip=equip, error_type=type(sensor_ctx_err).__name__, error=str(sensor_ctx_err)[:200])
            sensor_context = None
    
    return sensor_context


'''

# Check if already added
if "def _build_sensor_context(" in content:
    print("_build_sensor_context already exists, skipping")
else:
    # Insert before _update_baseline_buffer
    if insert_marker in content:
        content = content.replace(insert_marker, new_code + insert_marker)
        
        with open("core/acm_main.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("SUCCESS: Added _build_sensor_context helper function")
    else:
        print("ERROR: Could not find insertion marker")
