#!/usr/bin/env python
"""Add auto-tune fusion weights helper to acm_main.py."""
import re

# Read the current file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find insertion point - after _record_fusion_observability function
insert_marker = "def _run_autonomous_tuning("
insert_pos = content.find(insert_marker)

if insert_pos == -1:
    print("ERROR: Could not find _run_autonomous_tuning function")
    exit(1)

# New helper function to add
new_helper = '''

@dataclass
class AutoTuneResult:
    """Result of auto-tuning fusion weights."""
    weights: Dict[str, float]
    fusion_weights_used: Dict[str, float]
    tuning_diagnostics: Optional[Dict[str, Any]]
    tuning_enabled: bool


def _auto_tune_fusion_weights(
    present: Dict[str, np.ndarray],
    weights: Dict[str, float],
    previous_weights: Optional[Dict[str, float]],
    score: pd.DataFrame,
    run_dir: Path,
    output_manager: Any,
    sql_client: Optional[Any],
    equip_id: int,
    run_id: str,
    equip: str,
    cfg: Dict[str, Any],
    SQL_MODE: bool,
    T: Any,
) -> AutoTuneResult:
    """
    Auto-tune detector fusion weights for optimal episode separability.
    
    DET-06: Tune weights based on episode separability (not circular correlation).
    
    Args:
        present: Dictionary of detector z-score streams
        weights: Current fusion weights
        previous_weights: Previous weights from warm-start (optional)
        score: Score dataframe with original features
        run_dir: Run directory for saving diagnostics
        output_manager: Output manager for writing diagnostics
        sql_client: SQL client (for SQL mode)
        equip_id: Equipment ID
        run_id: Run ID
        equip: Equipment name
        cfg: Configuration dictionary
        SQL_MODE: Whether running in SQL mode
        T: Timer context
        
    Returns:
        AutoTuneResult with tuned weights and diagnostics
    """
    tuned_weights = None
    tuning_diagnostics = None
    fusion_weights_used = dict(weights)
    
    Console.info("Starting detector weight auto-tuning...", component="FUSE")
    with T.section("fusion.auto_tune"):
        try:
            # First fusion pass with current weights to get baseline
            fused_baseline, _ = fuse.combine(present, weights, cfg, original_features=score, regime_labels=None)
            fused_baseline_np = np.asarray(fused_baseline, dtype=np.float32).reshape(-1)
            
            # Tune weights based on episode separability
            tuned_weights, tuning_diagnostics = fuse.tune_detector_weights(
                streams=present,
                fused=fused_baseline_np,
                current_weights=weights,
                cfg=cfg
            )
            
            # Use tuned weights if tuning was enabled
            if tuning_diagnostics.get("enabled"):
                weights = tuned_weights
                fusion_weights_used = dict(tuned_weights)
                Console.info("Using auto-tuned weights for final fusion", component="TUNE")
                
                # Save tuning diagnostics to file/SQL
                if tuning_diagnostics and run_dir:
                    # Add warm-start metadata
                    tuning_diagnostics["warm_started"] = previous_weights is not None
                    if previous_weights:
                        tuning_diagnostics["previous_weights"] = previous_weights
                    
                    _save_tuning_diagnostics(
                        tuning_diagnostics=tuning_diagnostics,
                        fusion_weights_used=fusion_weights_used,
                        run_dir=run_dir,
                        output_manager=output_manager,
                        sql_client=sql_client if SQL_MODE else None,
                        equip_id=int(equip_id),
                        run_id=run_id,
                        equip=equip,
                        SQL_MODE=SQL_MODE,
                    )
                
                return AutoTuneResult(
                    weights=weights,
                    fusion_weights_used=fusion_weights_used,
                    tuning_diagnostics=tuning_diagnostics,
                    tuning_enabled=True,
                )
            else:
                return AutoTuneResult(
                    weights=weights,
                    fusion_weights_used=fusion_weights_used,
                    tuning_diagnostics=tuning_diagnostics,
                    tuning_enabled=False,
                )
        except Exception as tune_e:
            Console.warn(f"Weight auto-tuning failed: {tune_e}", component="TUNE",
                         equip=equip, error_type=type(tune_e).__name__, error=str(tune_e)[:200])
            return AutoTuneResult(
                weights=weights,
                fusion_weights_used=fusion_weights_used,
                tuning_diagnostics=None,
                tuning_enabled=False,
            )


'''

# Insert the new helper before _run_autonomous_tuning
new_content = content[:insert_pos] + new_helper + content[insert_pos:]

# Write the updated file
with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("SUCCESS: Added auto-tune fusion weights helper:")
print("  - AutoTuneResult dataclass")
print("  - _auto_tune_fusion_weights()")
