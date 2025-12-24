#!/usr/bin/env python
"""Add model persistence save helper function to acm_main.py."""

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

# ---------- MODEL-02: Model persistence save helper ----------
def _save_trained_models(
    ar1_detector: Optional[Any],
    pca_detector: Optional[Any],
    iforest_detector: Optional[Any],
    gmm_detector: Optional[Any],
    omr_detector: Optional[Any],
    regime_model: Optional[Any],
    regime_quality_ok: bool,
    train: pd.DataFrame,
    cfg: Dict[str, Any],
    equip: str,
    art_root: str,
    sql_client: Optional[Any],
    equip_id: int,
    run_id: Optional[str],
    SQL_MODE: bool,
    dual_mode: bool,
    col_meds: Optional[Dict[str, float]],
    T: Any,
) -> Optional[int]:
    """
    Save trained models to persistence layer with versioning.
    
    Returns:
        Model version number if saved successfully, None otherwise.
    """
    try:
        from core.model_persistence import ModelVersionManager, create_model_metadata
        
        model_manager = ModelVersionManager(
            equip=equip, 
            artifact_root=Path(art_root),
            sql_client=sql_client if SQL_MODE or dual_mode else None,
            equip_id=equip_id if SQL_MODE or dual_mode else None
        )
        
        # Collect all models
        models_to_save = {
            "ar1_params": {"phimap": ar1_detector.phimap, "sdmap": ar1_detector.sdmap} if ar1_detector and hasattr(ar1_detector, 'phimap') else None,
            "pca_model": pca_detector.pca if pca_detector and hasattr(pca_detector, 'pca') else None,
            "iforest_model": iforest_detector.model if iforest_detector and hasattr(iforest_detector, 'model') else None,
            "gmm_model": gmm_detector.model if gmm_detector and hasattr(gmm_detector, 'model') else None,
            "omr_model": omr_detector.to_dict() if omr_detector and omr_detector._is_fitted else None,
            "regime_model": regime_model.model if regime_model and hasattr(regime_model, 'model') else None,
            "feature_medians": col_meds
        }
        
        # Calculate training duration from timing sections
        training_duration_s = _calculate_training_duration(T)
        
        # Create metadata
        regime_quality_metrics = {
            "quality_ok": regime_quality_ok,
            "n_regimes": regime_model.model.n_clusters if regime_model and hasattr(regime_model, 'model') else 0
        }
        
        with T.section("models.persistence.metadata"):
            metadata = create_model_metadata(
                config_signature=cfg.get("_signature", "unknown"),
                train_data=train,
                models_dict=models_to_save,
                regime_quality=regime_quality_metrics,
                training_duration_s=training_duration_s
            )
        
        # Save models
        model_version = model_manager.save_models(
            models=models_to_save,
            metadata=metadata
        )
        
        Console.info(f"Saved all trained models to version v{model_version}", component="MODEL")
        return model_version
        
    except Exception as e:
        import traceback
        Console.warn(f"Failed to save models: {e}", component="MODEL",
                     equip=equip, run_id=run_id, error_type=type(e).__name__, error=str(e)[:500])
        traceback.print_exc()
        return None


def _calculate_training_duration(T: Any) -> Optional[float]:
    """Calculate total training duration from timer sections."""
    try:
        fit_sections = ["fit.ar1", "fit.pca_subspace", "fit.iforest", "fit.gmm", "fit.omr", "regimes.fit"]
        total_fit_time = sum([T.timings.get(sec, {}).get("elapsed", 0.0) for sec in fit_sections])
        if total_fit_time > 0:
            return total_fit_time
    except Exception:
        pass
    return None


'''
    
    new_content = content[:insert_idx] + helper_code + content[insert_idx:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("SUCCESS: Added model persistence save helper functions")
    print("  - _save_trained_models()")
    print("  - _calculate_training_duration()")

if __name__ == "__main__":
    main()
