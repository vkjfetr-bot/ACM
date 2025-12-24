#!/usr/bin/env python
"""Add continuous learning thresholds helper function to acm_main.py."""

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

# ---------- THRESHOLD-01: Continuous learning threshold update helper ----------
def _update_adaptive_thresholds(
    train: pd.DataFrame,
    score: pd.DataFrame,
    train_frame: pd.DataFrame,
    frame: pd.DataFrame,
    present: Dict[str, np.ndarray],
    weights: Dict[str, float],
    train_regime_labels: Optional[np.ndarray],
    score_regime_labels: Optional[np.ndarray],
    regime_quality_ok: bool,
    cfg: Dict[str, Any],
    equip_id: int,
    output_manager: Any,
    CONTINUOUS_LEARNING: bool,
    coldstart_complete: bool,
    threshold_update_interval: int,
    equip: str,
    T: Any,
) -> bool:
    """
    Calculate adaptive thresholds on accumulated or training data.
    
    Returns:
        True if thresholds were updated, False otherwise.
    """
    # Determine if we should recalculate thresholds this batch
    should_update_thresholds = False
    
    is_first_threshold_calc = coldstart_complete and not hasattr(cfg, '_thresholds_calculated')
    batch_num = cfg.get("runtime", {}).get("batch_num", 0)
    interval_reached = (batch_num % threshold_update_interval == 0) if threshold_update_interval > 0 else True
    
    if is_first_threshold_calc:
        should_update_thresholds = True
        Console.info("First threshold calculation after coldstart", component="THRESHOLD")
    elif CONTINUOUS_LEARNING and interval_reached:
        should_update_thresholds = True
        Console.info(f"Update interval reached (batch {batch_num}, interval={threshold_update_interval})", component="THRESHOLD")
    elif not CONTINUOUS_LEARNING and not hasattr(cfg, '_thresholds_calculated'):
        should_update_thresholds = True
        Console.info("Single threshold calculation (non-continuous mode)", component="THRESHOLD")
    
    if not should_update_thresholds:
        next_update_batch = (batch_num // threshold_update_interval + 1) * threshold_update_interval if threshold_update_interval > 0 else batch_num
        Console.info(f"Skipping threshold update (batch {batch_num}, next update at batch {next_update_batch})", component="THRESHOLD")
        return False
    
    # Calculate thresholds
    if CONTINUOUS_LEARNING and not train.empty and not score.empty:
        return _calculate_accumulated_thresholds(
            train=train, score=score, train_frame=train_frame, frame=frame,
            present=present, weights=weights,
            train_regime_labels=train_regime_labels, score_regime_labels=score_regime_labels,
            regime_quality_ok=regime_quality_ok, cfg=cfg, equip_id=equip_id,
            output_manager=output_manager, equip=equip,
        )
    else:
        return _calculate_train_only_thresholds(
            train=train, train_frame=train_frame, cfg=cfg, equip_id=equip_id,
            output_manager=output_manager, regime_quality_ok=regime_quality_ok, equip=equip,
        )


def _calculate_accumulated_thresholds(
    train: pd.DataFrame,
    score: pd.DataFrame,
    train_frame: pd.DataFrame,
    frame: pd.DataFrame,
    present: Dict[str, np.ndarray],
    weights: Dict[str, float],
    train_regime_labels: Optional[np.ndarray],
    score_regime_labels: Optional[np.ndarray],
    regime_quality_ok: bool,
    cfg: Dict[str, Any],
    equip_id: int,
    output_manager: Any,
    equip: str,
) -> bool:
    """Calculate thresholds on accumulated train + score data."""
    Console.info("Calculating thresholds on accumulated data (train + score)", component="THRESHOLD")
    accumulated_data = pd.concat([train, score], axis=0)
    
    # Build detector scores dict for accumulated data
    accumulated_present = {}
    for detector_name in present.keys():
        if detector_name in train_frame.columns and detector_name in frame.columns:
            train_scores = train_frame[detector_name].to_numpy(copy=False)
            score_scores = frame[detector_name].to_numpy(copy=False)
            accumulated_present[detector_name] = np.concatenate([train_scores, score_scores])
        elif detector_name in frame.columns:
            accumulated_present[detector_name] = frame[detector_name].to_numpy(copy=False)
    
    if not accumulated_present:
        return False
    
    try:
        # Build accumulated regime labels
        accumulated_regime_labels = None
        if regime_quality_ok and train_regime_labels is not None and score_regime_labels is not None:
            accumulated_regime_labels = np.concatenate([train_regime_labels, score_regime_labels])
        
        accumulated_fused, _ = fuse.combine(
            accumulated_present, weights, cfg,
            original_features=accumulated_data,
            regime_labels=accumulated_regime_labels
        )
        accumulated_fused_np = np.asarray(accumulated_fused, dtype=np.float32).reshape(-1)
        
        # Get regime labels from frame columns if available
        if regime_quality_ok:
            if "regime_label" in train.columns and "regime_label" in frame.columns:
                train_regimes = train["regime_label"].to_numpy(copy=False)
                score_regimes = frame.get("regime_label")
                if score_regimes is not None:
                    score_regimes = score_regimes.to_numpy(copy=False)
                    accumulated_regime_labels = np.concatenate([train_regimes, score_regimes])
        
        _calculate_adaptive_thresholds(
            fused_scores=accumulated_fused_np,
            cfg=cfg,
            equip_id=equip_id,
            output_manager=output_manager,
            train_index=accumulated_data.index,
            regime_labels=accumulated_regime_labels,
            regime_quality_ok=regime_quality_ok
        )
        cfg._thresholds_calculated = True
        return True
        
    except Exception as acc_e:
        Console.warn(f"Failed to calculate on accumulated data: {acc_e}", component="THRESHOLD",
                     equip=equip, error=str(acc_e)[:200])
        Console.warn("Falling back to train-only calculation", component="THRESHOLD", equip=equip)
        return _calculate_train_only_thresholds(
            train=train, train_frame=train_frame, cfg=cfg, equip_id=equip_id,
            output_manager=output_manager, regime_quality_ok=regime_quality_ok, equip=equip,
        )


def _calculate_train_only_thresholds(
    train: pd.DataFrame,
    train_frame: pd.DataFrame,
    cfg: Dict[str, Any],
    equip_id: int,
    output_manager: Any,
    regime_quality_ok: bool,
    equip: str,
) -> bool:
    """Calculate thresholds on training data only."""
    Console.info("Calculating thresholds on training data only", component="THRESHOLD")
    if train.empty or "fused" not in train_frame.columns:
        Console.warn("No train data available for threshold calculation", component="THRESHOLD", equip=equip)
        return False
    
    train_fused_np = train_frame["fused"].to_numpy(copy=False)
    train_regime_labels = train["regime_label"].to_numpy(copy=False) if "regime_label" in train.columns else None
    
    _calculate_adaptive_thresholds(
        fused_scores=train_fused_np,
        cfg=cfg,
        equip_id=equip_id,
        output_manager=output_manager,
        train_index=train.index,
        regime_labels=train_regime_labels,
        regime_quality_ok=regime_quality_ok
    )
    cfg._thresholds_calculated = True
    return True


'''
    
    new_content = content[:insert_idx] + helper_code + content[insert_idx:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("SUCCESS: Added continuous learning threshold helper functions")
    print("  - _update_adaptive_thresholds()")
    print("  - _calculate_accumulated_thresholds()")
    print("  - _calculate_train_only_thresholds()")

if __name__ == "__main__":
    main()
