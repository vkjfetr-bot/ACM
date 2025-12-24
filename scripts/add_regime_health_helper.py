#!/usr/bin/env python
"""Add regime health labeling helper function to acm_main.py."""

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find insertion point - before main()
    insert_marker = "\ndef main() -> None:"
    insert_idx = content.find(insert_marker)
    
    if insert_idx == -1:
        print("ERROR: Could not find insertion point")
        return
    
    helper_code = '''

# ---------- REGIME-02: Regime health labeling helper ----------
def _apply_regime_health_labels(
    frame: pd.DataFrame,
    score: pd.DataFrame,
    regime_model: Optional[Any],
    regime_quality_ok: bool,
    output_manager: Any,
    cfg: Dict[str, Any],
    equip: str,
    T: Any,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    """
    Apply regime health labels and detect transient states.
    
    Returns:
        Tuple of (updated frame, regime_stats dict)
    """
    Console.info("Starting regime health labeling and transient detection...", component="REGIME")
    regime_stats: Dict[int, Dict[str, float]] = {}
    
    # Handle low quality regimes
    if not regime_quality_ok and "regime_label" in frame.columns:
        Console.warn("Per-regime thresholds disabled (quality low).", component="REGIME", equip=equip)
        frame["regime_state"] = "unknown"
    
    # Apply health labels if model available
    if regime_model is not None and regime_quality_ok and "regime_label" in frame.columns and "fused" in frame.columns:
        try:
            regime_stats = regimes.update_health_labels(
                regime_model, 
                frame["regime_label"].to_numpy(copy=False), 
                frame["fused"], 
                cfg
            )
            frame["regime_state"] = frame["regime_label"].map(
                lambda x: regime_model.health_labels.get(int(x), "unknown")
            )
            summary_df = regimes.build_summary_dataframe(regime_model)
            if not summary_df.empty:
                output_manager.write_dataframe(summary_df, "regime_summary")
        except Exception as e:
            Console.warn(f"Health labelling skipped: {e}", component="REGIME",
                         equip=equip, error_type=type(e).__name__, error=str(e)[:200])
    
    # Ensure regime_state column exists
    if "regime_label" in frame.columns and "regime_state" not in frame.columns:
        frame["regime_state"] = frame["regime_label"].map(lambda _: "unknown")
    
    # Transient state detection
    if "regime_label" in frame.columns:
        with T.section("regimes.transient_detection"):
            try:
                transient_states = regimes.detect_transient_states(
                    data=score,
                    regime_labels=frame["regime_label"].to_numpy(copy=False),
                    cfg=cfg
                )
                frame["transient_state"] = transient_states
                
                transient_counts = frame["transient_state"].value_counts().to_dict() if "transient_state" in frame.columns else {}
                Console.info(f"Distribution: {transient_counts}", component="TRANSIENT")
            except Exception as trans_e:
                Console.warn(f"Detection failed: {trans_e}", component="TRANSIENT",
                             equip=equip, error_type=type(trans_e).__name__, error=str(trans_e)[:200])
                frame["transient_state"] = "unknown"
    
    # Log stats
    if regime_stats:
        state_counts = frame["regime_state"].value_counts().to_dict() if "regime_state" in frame.columns else {}
        Console.info(f"state histogram {state_counts}", component="REGIME")
    elif not regime_quality_ok:
        Console.warn("Clustering quality below threshold; per-regime thresholds disabled.", component="REGIME", equip=equip)
    
    return frame, regime_stats


'''
    
    new_content = content[:insert_idx] + helper_code + content[insert_idx:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("SUCCESS: Added regime health labeling helper function")
    print("  - _apply_regime_health_labels()")

if __name__ == "__main__":
    main()
