#!/usr/bin/env python3
"""Add RegimeLabelResult dataclass and _label_regimes helper function."""

# Read the file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the location to insert (before ScoreResult dataclass)
insert_marker = "@dataclass\nclass ScoreResult:"

# New code to insert
new_code = '''@dataclass
class RegimeLabelResult:
    """Result of regime labeling."""
    frame: pd.DataFrame
    regime_model: Optional[Any]
    train_regime_labels: Optional[np.ndarray]
    score_regime_labels: Optional[np.ndarray]
    regime_quality_ok: bool
    regime_model_was_trained: bool
    regime_out: Dict[str, Any]


def _label_regimes(
    score: pd.DataFrame,
    frame: pd.DataFrame,
    train: pd.DataFrame,
    train_numeric: Optional[pd.DataFrame],
    regime_model: Optional[Any],
    regime_state: Optional[Any],
    regime_state_version: int,
    regime_basis: RegimeBasisResult,
    equip: str,
    equip_id: int,
    cfg: Dict[str, Any],
    T: Timer,
    art_root: str,
    sql_client: Optional[Any],
    SQL_MODE: bool,
    dual_mode: bool,
) -> RegimeLabelResult:
    """Label data with regimes and save regime state if trained.
    
    Args:
        score: Score data.
        frame: Frame to update with regime labels.
        train: Training data.
        train_numeric: Raw numeric training data.
        regime_model: Regime model (may be "STATE_LOADED" string or actual model).
        regime_state: Loaded regime state for reconstruction.
        regime_state_version: Current regime state version number.
        regime_basis: Result from _build_regime_basis.
        equip: Equipment name.
        equip_id: Equipment ID.
        cfg: Configuration dictionary.
        T: Timer for profiling.
        art_root: Artifact root path.
        sql_client: SQL client for persistence.
        SQL_MODE: Whether in SQL mode.
        dual_mode: Whether in dual-write mode.
    
    Returns:
        RegimeLabelResult with updated frame, model, labels, and quality flags.
    """
    train_regime_labels = None
    score_regime_labels = None
    regime_model_was_trained = False
    regime_out: Dict[str, Any] = {}
    
    regime_basis_train = regime_basis.regime_basis_train
    regime_basis_score = regime_basis.regime_basis_score
    regime_basis_meta = regime_basis.regime_basis_meta
    regime_basis_hash = regime_basis.regime_basis_hash
    
    with T.section("regimes.label"):
        # REGIME-STATE-02: Reconstruct model from loaded state if available
        if regime_model == "STATE_LOADED" and regime_state is not None and regime_basis_train is not None:
            try:
                regime_model = regimes.regime_state_to_model(
                    state=regime_state,
                    feature_columns=list(regime_basis_train.columns),
                    raw_tags=list(train_numeric.columns) if train_numeric is not None else [],
                    train_hash=regime_basis_hash
                )
                Console.info(f"Reconstructed RegimeModel from state v{regime_state.state_version}", component="REGIME_STATE")
            except Exception as e:
                Console.warn(f"Failed to reconstruct model from state: {e}", component="REGIME_STATE",
                             equip=equip, state_version=regime_state.state_version, error=str(e)[:200])
                regime_model = None
        
        regime_ctx: Dict[str, Any] = {
            "regime_basis_train": regime_basis_train,
            "regime_basis_score": regime_basis_score,
            "basis_meta": regime_basis_meta,
            "regime_model": regime_model if regime_model != "STATE_LOADED" else None,
            "regime_basis_hash": regime_basis_hash,
            "X_train": train,
        }
        regime_out = regimes.label(score, regime_ctx, {"frame": frame}, cfg)
        frame = regime_out.get("frame", frame)
        new_regime_model = regime_out.get("regime_model", regime_model)
        
        # Check if model was retrained
        if new_regime_model is not regime_model and new_regime_model is not None:
            regime_model_was_trained = True
            regime_model = new_regime_model
        
        score_regime_labels = regime_out.get("regime_labels")
        train_regime_labels = regime_out.get("regime_labels_train")
        regime_quality_ok = bool(regime_out.get("regime_quality_ok", True))
        
        if train_regime_labels is None and regime_model is not None and regime_basis_train is not None:
            train_regime_labels = regimes.predict_regime(regime_model, regime_basis_train)
        if score_regime_labels is None and regime_model is not None and regime_basis_score is not None:
            score_regime_labels = regimes.predict_regime(regime_model, regime_basis_score)
        
        # Record regime for Prometheus/Grafana observability
        if score_regime_labels is not None and len(score_regime_labels) > 0:
            current_regime_id = int(score_regime_labels[-1]) if hasattr(score_regime_labels[-1], '__int__') else 0
            regime_label = ""
            if regime_model is not None and hasattr(regime_model, 'cluster_labels_'):
                try:
                    regime_label = regime_model.cluster_labels_.get(current_regime_id, f"regime_{current_regime_id}")
                except Exception:
                    regime_label = f"regime_{current_regime_id}"
            record_regime(equip, current_regime_id, regime_label)
        
        # REGIME-STATE-03: Save regime state if model was trained
        if regime_model_was_trained and regime_model is not None:
            try:
                from core.model_persistence import save_regime_state
                import hashlib
                
                # Generate config hash for change detection
                regime_cfg_str = str(cfg.get("regimes", {}))
                config_hash = hashlib.sha256(regime_cfg_str.encode()).hexdigest()[:16]
                
                # Convert model to state
                new_state = regimes.regime_model_to_state(
                    model=regime_model,
                    equip_id=equip_id,
                    state_version=regime_state_version + 1,
                    config_hash=config_hash,
                    regime_basis_hash=str(regime_basis_hash) if regime_basis_hash else ""
                )
                
                # Save state
                save_regime_state(
                    state=new_state,
                    artifact_root=Path(art_root),
                    equip=equip,
                    sql_client=sql_client if SQL_MODE or dual_mode else None
                )
                
                regime_state_version = new_state.state_version
                Console.info(f"Saved state v{regime_state_version}: K={new_state.n_clusters}, quality_ok={new_state.quality_ok}", component="REGIME_STATE")
            except Exception as e:
                Console.warn(f"Failed to save regime state: {e}", component="REGIME_STATE",
                             equip=equip, error_type=type(e).__name__, error=str(e)[:200])
    
    return RegimeLabelResult(
        frame=frame,
        regime_model=regime_model,
        train_regime_labels=train_regime_labels,
        score_regime_labels=score_regime_labels,
        regime_quality_ok=regime_quality_ok,
        regime_model_was_trained=regime_model_was_trained,
        regime_out=regime_out,
    )


'''

# Check if already added
if "class RegimeLabelResult:" in content:
    print("RegimeLabelResult already exists, skipping")
else:
    # Insert before ScoreResult
    if insert_marker in content:
        content = content.replace(insert_marker, new_code + insert_marker)
        
        with open("core/acm_main.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("SUCCESS: Added RegimeLabelResult and _label_regimes helper")
    else:
        print("ERROR: Could not find insertion marker")
