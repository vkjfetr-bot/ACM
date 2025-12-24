#!/usr/bin/env python3
"""Add DriftResult dataclass and _compute_drift_detection helper function."""

# Read the file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the location to insert (before ScoreResult dataclass)
insert_marker = "@dataclass\nclass ScoreResult:"

# New code to insert
new_code = '''@dataclass
class DriftResult:
    """Result of drift detection."""
    frame: pd.DataFrame  # Frame with drift columns and alert_mode
    score_out: Dict[str, Any]  # Updated score_out dict


def _compute_drift_detection(
    frame: pd.DataFrame,
    score: pd.DataFrame,
    score_out: Dict[str, Any],
    regime_quality_ok: bool,
    cfg: Dict[str, Any],
    equip: str,
    T: Timer,
) -> DriftResult:
    """Compute drift detection and alert mode.
    
    Uses multi-feature drift detection if enabled, otherwise falls back
    to simple P95 threshold.
    
    Args:
        frame: Frame with fused scores.
        score: Original score data.
        score_out: Score output dict from regimes.
        regime_quality_ok: Whether regime quality is acceptable.
        cfg: Configuration dictionary.
        equip: Equipment name.
        T: Timer for profiling.
    
    Returns:
        DriftResult with updated frame and score_out.
    """
    with T.section("drift"):
        score_out["frame"] = frame
        score_out = drift.compute(score, score_out, cfg)
        frame = score_out["frame"]

    # DRIFT-01: Multi-Feature Drift Detection
    drift_col = "cusum_z" if "cusum_z" in frame.columns else ("drift_z" if "drift_z" in frame.columns else None)
    
    drift_cfg = (cfg or {}).get("drift", {})
    multi_feat_cfg = drift_cfg.get("multi_feature", {})
    multi_feat_enabled = bool(multi_feat_cfg.get("enabled", False))
    
    if drift_col is not None:
        try:
            drift_array = frame[drift_col].to_numpy(dtype=np.float32)
            
            if multi_feat_enabled:
                trend_window = int(multi_feat_cfg.get("trend_window", 20))
                trend_threshold = float(multi_feat_cfg.get("trend_threshold", 0.05))
                fused_drift_min = float(multi_feat_cfg.get("fused_drift_min", 2.0))
                fused_drift_max = float(multi_feat_cfg.get("fused_drift_max", 5.0))
                regime_volatility_max = float(multi_feat_cfg.get("regime_volatility_max", 0.3))
                hysteresis_on = float(multi_feat_cfg.get("hysteresis_on", 3.0))
                hysteresis_off = float(multi_feat_cfg.get("hysteresis_off", 1.5))
                
                drift_trend = _compute_drift_trend(drift_array, window=trend_window)
                fused_p95 = float(np.nanpercentile(frame["fused"].to_numpy(dtype=np.float32), 95)) if "fused" in frame.columns else 0.0
                
                regime_volatility = 0.0
                if "regime_label" in frame.columns and regime_quality_ok:
                    regime_labels = frame["regime_label"].to_numpy()
                    regime_volatility = _compute_regime_volatility(regime_labels, window=trend_window)
                
                drift_p95 = float(np.nanpercentile(drift_array, 95))
                prev_alert_mode = "FAULT"
                
                is_drift_condition = (
                    abs(drift_trend) > trend_threshold and
                    fused_drift_min <= fused_p95 <= fused_drift_max and
                    regime_volatility < regime_volatility_max
                )
                
                if prev_alert_mode == "DRIFT":
                    alert_mode = "DRIFT" if drift_p95 > hysteresis_off else "FAULT"
                else:
                    alert_mode = "DRIFT" if (drift_p95 > hysteresis_on and is_drift_condition) else "FAULT"
                
                frame["alert_mode"] = alert_mode
                Console.info(
                    f"Multi-feature: {drift_col} P95={drift_p95:.3f}, trend={drift_trend:.4f}, "
                    f"fused_P95={fused_p95:.3f}, regime_vol={regime_volatility:.3f} -> {alert_mode}",
                    component="DRIFT"
                )
            else:
                drift_p95 = float(np.nanpercentile(drift_array, 95))
                drift_threshold = float(drift_cfg.get("p95_threshold", 2.0))
                frame["alert_mode"] = "DRIFT" if drift_p95 > drift_threshold else "FAULT"
                Console.info(f"{drift_col} P95={drift_p95:.3f} (threshold={drift_threshold:.1f}) -> alert_mode={frame['alert_mode'].iloc[-1]}", component="DRIFT")
        except Exception as e:
            Console.warn(f"Detection failed: {e}", component="DRIFT",
                         equip=equip, error_type=type(e).__name__, error=str(e)[:200])
            frame["alert_mode"] = "FAULT"
    else:
        frame["alert_mode"] = "FAULT"
    
    return DriftResult(frame=frame, score_out=score_out)


'''

# Check if already added
if "class DriftResult:" in content:
    print("DriftResult already exists, skipping")
else:
    # Insert before ScoreResult
    if insert_marker in content:
        content = content.replace(insert_marker, new_code + insert_marker)
        
        with open("core/acm_main.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("SUCCESS: Added DriftResult and _compute_drift_detection helper")
    else:
        print("ERROR: Could not find insertion marker")
