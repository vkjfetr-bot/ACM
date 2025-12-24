#!/usr/bin/env python
"""Add regime state loading and detector enabled flags helpers to acm_main.py."""
import re

# Read the current file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find insertion point - after _apply_local_detector_cache function
insert_marker = "def _run_autonomous_tuning("
insert_pos = content.find(insert_marker)

if insert_pos == -1:
    print("ERROR: Could not find _run_autonomous_tuning function")
    exit(1)

# New helper functions to add
new_helpers = '''

@dataclass
class DetectorEnabledFlags:
    """Flags indicating which detectors are enabled based on fusion weights."""
    ar1_enabled: bool
    pca_enabled: bool
    iforest_enabled: bool
    gmm_enabled: bool
    omr_enabled: bool
    disabled_detectors: List[str]


def _check_detector_enabled_flags(cfg: Dict[str, Any]) -> DetectorEnabledFlags:
    """
    Check fusion weights to determine which detectors are enabled (lazy evaluation).
    
    PERF-03: Skip disabled detectors based on fusion weight configuration.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        DetectorEnabledFlags with enabled status for each detector
    """
    fusion_cfg = (cfg or {}).get("fusion", {})
    fusion_weights = fusion_cfg.get("weights", {})
    
    ar1_enabled = fusion_weights.get("ar1_z", 0.0) > 0
    pca_enabled = fusion_weights.get("pca_spe_z", 0.0) > 0 or fusion_weights.get("pca_t2_z", 0.0) > 0
    iforest_enabled = fusion_weights.get("iforest_z", 0.0) > 0
    gmm_enabled = fusion_weights.get("gmm_z", 0.0) > 0
    omr_enabled = fusion_weights.get("omr_z", 0.0) > 0
    
    disabled_detectors = []
    if not ar1_enabled: disabled_detectors.append("ar1")
    if not pca_enabled: disabled_detectors.append("pca")
    if not iforest_enabled: disabled_detectors.append("iforest")
    if not gmm_enabled: disabled_detectors.append("gmm")
    if not omr_enabled: disabled_detectors.append("omr")
    
    if disabled_detectors:
        Console.info(f"Lazy evaluation: skipping disabled detectors: {', '.join(disabled_detectors)}", component="PERF")
    
    return DetectorEnabledFlags(
        ar1_enabled=ar1_enabled,
        pca_enabled=pca_enabled,
        iforest_enabled=iforest_enabled,
        gmm_enabled=gmm_enabled,
        omr_enabled=omr_enabled,
        disabled_detectors=disabled_detectors,
    )


@dataclass
class RegimeStateLoadResult:
    """Result of loading regime state from persistence."""
    regime_model: Optional[Any]
    regime_state: Optional[Any]
    regime_state_version: int


def _load_regime_state_from_persistence(
    art_root: str,
    equip: str,
    equip_id: int,
    sql_client: Optional[Any],
    stable_models_dir: Path,
    SQL_MODE: bool,
    dual_mode: bool,
) -> RegimeStateLoadResult:
    """
    Load cached regime model from RegimeState (new system) or joblib persistence.
    
    REGIME-STATE-01: Try loading from RegimeState first (enables continuity),
    then fallback to old joblib persistence.
    
    Args:
        art_root: Artifact root path
        equip: Equipment name
        equip_id: Equipment ID
        sql_client: SQL client (for SQL/dual modes)
        stable_models_dir: Path to stable models directory
        SQL_MODE: Whether running in SQL mode
        dual_mode: Whether running in dual mode
        
    Returns:
        RegimeStateLoadResult with regime_model and state info
    """
    regime_model = None
    regime_state = None
    regime_state_version = 0
    
    try:
        # REGIME-STATE-01: Try loading from RegimeState first (enables continuity)
        from core.model_persistence import load_regime_state
        regime_state = load_regime_state(
            artifact_root=Path(art_root),
            equip=equip,
            equip_id=equip_id if SQL_MODE or dual_mode else None,
            sql_client=sql_client if SQL_MODE or dual_mode else None
        )
        
        if regime_state is not None and regime_state.quality_ok:
            Console.info(f"Loaded state v{regime_state.state_version}: K={regime_state.n_clusters}, silhouette={regime_state.silhouette_score:.3f}", component="REGIME_STATE")
            regime_state_version = regime_state.state_version
            
            # Reconstruct RegimeModel from state (will be validated against current basis later)
            # Note: Feature columns will be set from regime_basis_train when available
            regime_model = "STATE_LOADED"  # Placeholder, will be reconstructed in label()
        else:
            # Fallback to old joblib persistence
            regime_model = regimes.load_regime_model(stable_models_dir)
            if regime_model is not None:
                Console.info(f"Loaded cached regime model (legacy): K={regime_model.kmeans.n_clusters}, hash={regime_model.train_hash}", component="REGIME")
    except Exception as e:
        Console.warn(f"Failed to load cached regime state/model: {e}", component="REGIME",
                     equip=equip, error_type=type(e).__name__, error=str(e)[:200])
        regime_model = None
        regime_state = None
        regime_state_version = 0
    
    return RegimeStateLoadResult(
        regime_model=regime_model,
        regime_state=regime_state,
        regime_state_version=regime_state_version,
    )


'''

# Insert the new helpers before _run_autonomous_tuning
new_content = content[:insert_pos] + new_helpers + content[insert_pos:]

# Write the updated file
with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("SUCCESS: Added regime state and detector enabled helpers:")
print("  - DetectorEnabledFlags dataclass")
print("  - _check_detector_enabled_flags()")
print("  - RegimeStateLoadResult dataclass")
print("  - _load_regime_state_from_persistence()")
