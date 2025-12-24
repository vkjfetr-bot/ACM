#!/usr/bin/env python
"""Add model quality and persistence helper functions to acm_main.py."""

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

# ---------- MODEL-01: Model quality assessment helper ----------
@dataclass
class QualityAssessmentResult:
    """Result from model quality assessment."""
    force_retrain: bool
    cached_models_invalidated: bool
    retrain_reason: Optional[str]
    ar1_detector: Optional[Any] = None
    pca_detector: Optional[Any] = None
    iforest_detector: Optional[Any] = None
    gmm_detector: Optional[Any] = None
    pca_train_spe: Optional[np.ndarray] = None
    pca_train_t2: Optional[np.ndarray] = None


def _assess_model_quality_and_retrain(
    cached_models: Optional[Dict[str, Any]],
    cached_manifest: Optional[Dict[str, Any]],
    score_out: Dict[str, Any],
    regime_quality_ok: bool,
    train: pd.DataFrame,
    cfg: Dict[str, Any],
    equip: str,
    SQL_MODE: bool,
    ar1_enabled: bool,
    pca_enabled: bool,
    iforest_enabled: bool,
    gmm_enabled: bool,
    T: Any,
) -> QualityAssessmentResult:
    """
    Assess model quality and trigger retraining if needed.
    
    Returns:
        QualityAssessmentResult with updated detectors if retraining occurred.
    """
    force_retrain = False
    retrain_reason = None
    
    # Return early if no cached models or auto_retrain disabled
    if not cached_models or not cfg.get("models", {}).get("auto_retrain", True):
        return QualityAssessmentResult(
            force_retrain=False,
            cached_models_invalidated=False,
            retrain_reason=None,
        )
    
    try:
        # Get regime quality metrics
        regime_quality_metrics = {
            "silhouette": score_out.get("silhouette", 0.0),
            "quality_ok": regime_quality_ok
        }
        
        # Check config changed
        config_changed = False
        if cached_manifest:
            cached_sig = cached_manifest.get("config_signature", "")
            current_sig = cfg.get("_signature", "unknown")
            config_changed = (cached_sig != current_sig)
        
        # Get auto_retrain config
        auto_retrain_cfg = cfg.get("models", {}).get("auto_retrain", {})
        if isinstance(auto_retrain_cfg, bool):
            auto_retrain_cfg = {}
        
        # Check model age trigger
        model_age_trigger = False
        if SQL_MODE and cached_manifest:
            model_age_trigger = _check_model_age_trigger(
                cached_manifest=cached_manifest,
                auto_retrain_cfg=auto_retrain_cfg,
                equip=equip,
            )
        
        # Check regime quality trigger
        regime_quality_trigger = False
        if SQL_MODE:
            min_silhouette = auto_retrain_cfg.get("min_regime_quality", 0.3)
            current_silhouette = regime_quality_metrics.get("silhouette", 0.0)
            if not regime_quality_ok or current_silhouette < min_silhouette:
                Console.warn(f"Regime quality degraded (silhouette={current_silhouette:.3f} < {min_silhouette}) - forcing retraining", 
                             component="MODEL", equip=equip, silhouette=round(current_silhouette, 3), min_silhouette=min_silhouette)
                regime_quality_trigger = True
        
        # Aggregate triggers
        if config_changed:
            Console.warn("Config changed - forcing retraining", component="MODEL", equip=equip)
            force_retrain = True
            retrain_reason = "config_changed"
        elif model_age_trigger:
            force_retrain = True
            retrain_reason = "model_age"
        elif regime_quality_trigger:
            force_retrain = True
            retrain_reason = "regime_quality"
        
        # If retraining needed, fit new detectors
        if force_retrain:
            Console.info("Re-fitting detectors due to forced retraining...", component="MODEL")
            record_model_refit(equip, reason=retrain_reason, detector="all")
            
            ar1_detector = pca_detector = iforest_detector = gmm_detector = None
            pca_train_spe = pca_train_t2 = None
            
            if ar1_enabled:
                ar1_detector = AR1Detector(ar1_cfg=(cfg.get("models", {}).get("ar1", {}) or {})).fit(train)
            if pca_enabled:
                pca_cfg = (cfg.get("models", {}).get("pca", {}) or {})
                pca_detector = correlation.PCASubspaceDetector(pca_cfg=pca_cfg).fit(train)
                pca_train_spe, pca_train_t2 = pca_detector.score(train)
            if iforest_enabled:
                iforest_cfg = (cfg.get("models", {}).get("iforest", {}) or {})
                iforest_detector = outliers.IsolationForestDetector(iforest_cfg=iforest_cfg).fit(train)
            if gmm_enabled:
                gmm_cfg = (cfg.get("models", {}).get("gmm", {}) or {})
                gmm_detector = outliers.GaussianMixtureDetector(gmm_cfg=gmm_cfg).fit(train)
            
            return QualityAssessmentResult(
                force_retrain=True,
                cached_models_invalidated=True,
                retrain_reason=retrain_reason,
                ar1_detector=ar1_detector,
                pca_detector=pca_detector,
                iforest_detector=iforest_detector,
                gmm_detector=gmm_detector,
                pca_train_spe=pca_train_spe,
                pca_train_t2=pca_train_t2,
            )
        
    except Exception as e:
        Console.warn(f"Quality assessment failed: {e}", component="MODEL",
                     equip=equip, error_type=type(e).__name__, error=str(e)[:200])
    
    return QualityAssessmentResult(
        force_retrain=False,
        cached_models_invalidated=False,
        retrain_reason=None,
    )


def _check_model_age_trigger(
    cached_manifest: Dict[str, Any],
    auto_retrain_cfg: Dict[str, Any],
    equip: str,
) -> bool:
    """Check if model age exceeds the configured limit."""
    created_at_str = cached_manifest.get("created_at")
    if not created_at_str:
        return False
    
    try:
        from datetime import datetime
        created_at = datetime.fromisoformat(created_at_str)
        model_age_hours = (datetime.now() - created_at).total_seconds() / 3600
        max_age_hours = auto_retrain_cfg.get("max_model_age_hours", 720)  # 30 days default
        
        if model_age_hours > max_age_hours:
            Console.warn(f"Model age {model_age_hours:.1f}h exceeds limit {max_age_hours}h - forcing retraining",
                         component="MODEL", equip=equip, model_age_hours=round(model_age_hours, 1), max_age_hours=max_age_hours)
            return True
    except Exception as age_e:
        Console.warn(f"Failed to check model age: {age_e}", component="MODEL",
                     equip=equip, error=str(age_e)[:200])
    
    return False


'''
    
    new_content = content[:insert_idx] + helper_code + content[insert_idx:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("SUCCESS: Added model quality assessment helper functions")
    print("  - QualityAssessmentResult dataclass")
    print("  - _assess_model_quality_and_retrain()")
    print("  - _check_model_age_trigger()")

if __name__ == "__main__":
    main()
