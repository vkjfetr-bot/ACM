#!/usr/bin/env python3
"""Replace the drift detection section in main() with _compute_drift_detection() call."""

with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# The old code block to replace (from ===== 7) Drift ===== to frame["alert_mode"] = "FAULT")
old_code = '''        # ===== 7) Drift =====
        with T.section("drift"):
            score_out["frame"] = frame # type: ignore
            score_out = drift.compute(score, score_out, cfg)
            frame = score_out["frame"]

        # DRIFT-01: Multi-Feature Drift Detection (replaces simple P95 threshold)
        # Combines drift trend, fused level, and regime volatility with hysteresis to distinguish
        # gradual drift (requires retraining) from transient faults (does not require retraining)
        drift_col = "cusum_z" if "cusum_z" in frame.columns else ("drift_z" if "drift_z" in frame.columns else None)
        
        # Retrieve multi-feature drift configuration
        drift_cfg = (cfg or {}).get("drift", {})
        multi_feat_cfg = drift_cfg.get("multi_feature", {})
        multi_feat_enabled = bool(multi_feat_cfg.get("enabled", False))
        
        if drift_col is not None:
            try:
                drift_array = frame[drift_col].to_numpy(dtype=np.float32)
                
                if multi_feat_enabled:
                    # DRIFT-01: Multi-feature logic with hysteresis
                    # Configuration parameters
                    trend_window = int(multi_feat_cfg.get("trend_window", 20))
                    trend_threshold = float(multi_feat_cfg.get("trend_threshold", 0.05))  # Slope per sample
                    fused_drift_min = float(multi_feat_cfg.get("fused_drift_min", 2.0))    # P95 min for drift
                    fused_drift_max = float(multi_feat_cfg.get("fused_drift_max", 5.0))    # P95 max for drift
                    regime_volatility_max = float(multi_feat_cfg.get("regime_volatility_max", 0.3))
                    hysteresis_on = float(multi_feat_cfg.get("hysteresis_on", 3.0))         # Turn ON drift alert
                    hysteresis_off = float(multi_feat_cfg.get("hysteresis_off", 1.5))       # Turn OFF drift alert
                    
                    # Compute features
                    drift_trend = _compute_drift_trend(drift_array, window=trend_window)
                    fused_p95 = float(np.nanpercentile(frame["fused"].to_numpy(dtype=np.float32), 95)) if "fused" in frame.columns else 0.0
                    
                    # Compute regime volatility if regime labels exist
                    regime_volatility = 0.0
                    if "regime_label" in frame.columns and regime_quality_ok:
                        regime_labels = frame["regime_label"].to_numpy()
                        regime_volatility = _compute_regime_volatility(regime_labels, window=trend_window)
                    
                    # Composite rule: DRIFT if all conditions met
                    # 1. Positive drift trend (sustained upward movement)
                    # 2. Fused level in drift range (not too high = fault, not too low = normal)
                    # 3. Low regime volatility (stable operating conditions)
                    drift_p95 = float(np.nanpercentile(drift_array, 95))
                    
                    # Get previous alert_mode (if exists) for hysteresis
                    # In first run or if unavailable, assume "FAULT"
                    prev_alert_mode = "FAULT"
                    # Note: In production, could load from prior run's final frame or maintain state
                    
                    # Apply hysteresis: different thresholds for turning ON vs OFF
                    is_drift_condition = (
                        abs(drift_trend) > trend_threshold and       # Sustained trend (up or down)
                        fused_drift_min <= fused_p95 <= fused_drift_max and  # In drift severity range
                        regime_volatility < regime_volatility_max     # Stable regime
                    )
                    
                    # Hysteresis logic
                    if prev_alert_mode == "DRIFT":
                        # Currently in DRIFT: turn OFF if drift_p95 drops below hysteresis_off
                        alert_mode = "DRIFT" if drift_p95 > hysteresis_off else "FAULT"
                    else:
                        # Currently in FAULT: turn ON if drift_p95 exceeds hysteresis_on AND conditions met
                        alert_mode = "DRIFT" if (drift_p95 > hysteresis_on and is_drift_condition) else "FAULT"
                    
                    frame["alert_mode"] = alert_mode
                    Console.info(
                        f"Multi-feature: {drift_col} P95={drift_p95:.3f}, trend={drift_trend:.4f}, "
                        f"fused_P95={fused_p95:.3f}, regime_vol={regime_volatility:.3f} -> {alert_mode}",
                        component="DRIFT"
                    )
                else:
                    # Fallback to legacy simple threshold (CFG-06)
                    drift_p95 = float(np.nanpercentile(drift_array, 95))
                    drift_threshold = float(drift_cfg.get("p95_threshold", 2.0))
                    frame["alert_mode"] = "DRIFT" if drift_p95 > drift_threshold else "FAULT"
                    Console.info(f"{drift_col} P95={drift_p95:.3f} (threshold={drift_threshold:.1f}) -> alert_mode={frame['alert_mode'].iloc[-1]}", component="DRIFT")
            except Exception as e:
                Console.warn(f"Detection failed: {e}", component="DRIFT",
                             equip=equip, error_type=type(e).__name__, error=str(e)[:200])
                frame["alert_mode"] = "FAULT"
        else:
            frame["alert_mode"] = "FAULT"'''

# New code with helper function call
new_code = '''        # ===== 7) Drift =====
        drift_result = _compute_drift_detection(
            frame=frame, score=score, score_out=score_out,
            regime_quality_ok=regime_quality_ok, cfg=cfg, equip=equip, T=T,
        )
        frame = drift_result.frame
        score_out = drift_result.score_out'''

if old_code in content:
    content = content.replace(old_code, new_code)
    
    with open("core/acm_main.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    old_lines = len(old_code.split('\n'))
    new_lines = len(new_code.split('\n'))
    print(f"SUCCESS: Replaced drift section ({old_lines} lines -> {new_lines} lines)")
    print(f"         Removed {old_lines - new_lines} lines from main()")
else:
    print("ERROR: Could not find old_code block to replace")
    print("Searching for partial match...")
    if "# ===== 7) Drift =====" in content:
        print("Found section header, but full block doesn't match")
    else:
        print("Section header not found")
