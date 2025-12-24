#!/usr/bin/env python3
"""Replace the sensor context section in main() with _build_sensor_context() call."""

with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# The old code block to replace
old_code = '''        sensor_context: Optional[Dict[str, Any]] = None
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
                        # CRITICAL FIX #5: Prevent division by zero with safe epsilon fallback
                        train_std = train_baseline.std()
                        train_std = train_std.replace(0.0, np.nan).fillna(1e-10)  # Safe fallback
                        valid_cols = train_std[train_std > 1e-10].index.tolist()  # Only truly valid columns
                        if valid_cols:
                            train_mean = train_mean[valid_cols]
                            train_std = train_std[valid_cols]
                            score_baseline = score_baseline[valid_cols]
                            score_aligned = score_baseline.reindex(frame.index)
                            score_aligned = score_aligned.apply(pd.to_numeric, errors="coerce")
                            sensor_z = (score_aligned - train_mean) / train_std
                            sensor_z = sensor_z.replace([np.inf, -np.inf], np.nan)
                            # Ensure alignment with scoring frame for downstream joins
                            sensor_context = {
                                "values": score_aligned,
                                "z_scores": sensor_z,
                                "train_mean": train_mean,
                                "train_std": train_std,
                                "train_p95": train_baseline[valid_cols].quantile(0.95),
                                "train_p05": train_baseline[valid_cols].quantile(0.05),
                                "omr_contributions": omr_contributions_data,  # Add OMR contributions for visualization
                                "regime_meta": regime_model.meta if regime_model else {}  # Add regime model metadata for chart subtitles
                            }
            except Exception as sensor_ctx_err:
                Console.warn(f"Failed to build sensor analytics context: {sensor_ctx_err}", component="SENSOR",
                             equip=equip, error_type=type(sensor_ctx_err).__name__, error=str(sensor_ctx_err)[:200])
                sensor_context = None'''

# New code with helper function call
new_code = '''        sensor_context = _build_sensor_context(
            train_numeric=train_numeric, score_numeric=score_numeric, frame=frame,
            omr_contributions_data=omr_contributions_data, regime_model=regime_model,
            equip=equip, T=T,
        )'''

if old_code in content:
    content = content.replace(old_code, new_code)
    
    with open("core/acm_main.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    old_lines = len(old_code.split('\n'))
    new_lines = len(new_code.split('\n'))
    print(f"SUCCESS: Replaced sensor context section ({old_lines} lines -> {new_lines} lines)")
    print(f"         Removed {old_lines - new_lines} lines from main()")
else:
    print("ERROR: Could not find old_code block to replace")
    if "sensor_context: Optional[Dict" in content:
        print("Found section start, but full block doesn't match")
    else:
        print("Section start not found")
