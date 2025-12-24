#!/usr/bin/env python
"""Replace Model Quality Assessment section with helper function call."""

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the section start and end markers
    start_marker = """        # ===== Model Quality Assessment: Check if retraining needed =====
        # This happens AFTER first scoring so we can evaluate cached model performance
        force_retrain = False
        quality_report = None
        
        if cached_models and cfg.get("models", {}).get("auto_retrain", True):
            with T.section("models.quality_check"):
                try:
                    from core.model_evaluation import assess_model_quality
                    
                    # Build temporary episodes for quality check (before fusion/episodes)
                    temp_episodes = pd.DataFrame()  # Will be populated after fusion
                    
                    # Get regime quality metrics
                    regime_quality_metrics = {
                        "silhouette": score_out.get("silhouette", 0.0),
                        "quality_ok": regime_quality_ok
                    }
                    
                    # Assess quality (will do full assessment after fusion, but check config now)
                    config_changed = False
                    if cached_manifest:
                        cached_sig = cached_manifest.get("config_signature", "")
                        current_sig = cfg.get("_signature", "unknown")
                        config_changed = (cached_sig != current_sig)
                    
                    # Get auto_retrain config for SQL mode data-driven triggers
                    auto_retrain_cfg = cfg.get("models", {}).get("auto_retrain", {})
                    if isinstance(auto_retrain_cfg, bool):
                        auto_retrain_cfg = {}  # Convert legacy boolean to dict
                    
                    # Check model age (SQL mode temporal validation)
                    model_age_trigger = False
                    if SQL_MODE and cached_manifest:
                        created_at_str = cached_manifest.get("created_at")
                        if created_at_str:
                            try:
                                from datetime import datetime
                                created_at = datetime.fromisoformat(created_at_str)
                                model_age_hours = (datetime.now() - created_at).total_seconds() / 3600
                                max_age_hours = auto_retrain_cfg.get("max_model_age_hours", 720)  # 30 days default
                                
                                if model_age_hours > max_age_hours:
                                    Console.warn(f"Model age {model_age_hours:.1f}h exceeds limit {max_age_hours}h - forcing retraining", component="MODEL",
                                                 equip=equip, model_age_hours=round(model_age_hours, 1), max_age_hours=max_age_hours)
                                    model_age_trigger = True
                            except Exception as age_e:
                                Console.warn(f"Failed to check model age: {age_e}", component="MODEL",
                                             equip=equip, error=str(age_e)[:200])
                    
                    # Check regime quality (SQL mode data-driven trigger)
                    regime_quality_trigger = False
                    if SQL_MODE:
                        min_silhouette = auto_retrain_cfg.get("min_regime_quality", 0.3)
                        current_silhouette = regime_quality_metrics.get("silhouette", 0.0)
                        if not regime_quality_ok or current_silhouette < min_silhouette:
                            Console.warn(f"Regime quality degraded (silhouette={current_silhouette:.3f} < {min_silhouette}) - forcing retraining", component="MODEL",
                                         equip=equip, silhouette=round(current_silhouette, 3), min_silhouette=min_silhouette)
                            regime_quality_trigger = True
                    
                    # Aggregate triggers
                    if config_changed:
                        Console.warn(f"Config changed - forcing retraining", component="MODEL",
                                     equip=equip)
                        force_retrain = True
                    elif model_age_trigger or regime_quality_trigger:
                        force_retrain = True
                    
                    # Invalidate cached models if retraining required (MHAL removed v9.1.0)
                    if force_retrain:
                        cached_models = None
                        ar1_detector = pca_detector = iforest_detector = gmm_detector = None
                        # Re-fit detectors immediately after invalidation
                        Console.info("Re-fitting detectors due to forced retraining...", component="MODEL")
                        
                        # Determine retrain reason for observability
                        retrain_reason = "config_changed" if config_changed else (
                            "model_age" if model_age_trigger else (
                                "regime_quality" if regime_quality_trigger else "forced"
                            )
                        )
                        record_model_refit(equip, reason=retrain_reason, detector="all")
                        
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
                        
                except Exception as e:
                    Console.warn(f"Quality assessment failed: {e}", component="MODEL",
                                 equip=equip, error_type=type(e).__name__, error=str(e)[:200])

        # ===== Model Persistence: Save trained models with versioning ====="""

    end_marker = "        # ===== Model Persistence: Save trained models with versioning ====="
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("ERROR: Could not find start marker")
        return
    
    end_idx = content.find(end_marker, start_idx + 100)
    if end_idx == -1:
        print("ERROR: Could not find end marker")
        return
    
    # Calculate old section
    old_section = content[start_idx:end_idx]
    old_lines = old_section.count('\n') + 1
    
    new_section = """        # ===== Model Quality Assessment: Check if retraining needed =====
        with T.section("models.quality_check"):
            quality_result = _assess_model_quality_and_retrain(
                cached_models=cached_models,
                cached_manifest=cached_manifest if 'cached_manifest' in locals() else None,
                score_out=score_out,
                regime_quality_ok=regime_quality_ok,
                train=train,
                cfg=cfg,
                equip=equip,
                SQL_MODE=SQL_MODE,
                ar1_enabled=ar1_enabled,
                pca_enabled=pca_enabled,
                iforest_enabled=iforest_enabled,
                gmm_enabled=gmm_enabled,
                T=T,
            )
            force_retrain = quality_result.force_retrain
            if quality_result.cached_models_invalidated:
                cached_models = None
                ar1_detector = quality_result.ar1_detector
                pca_detector = quality_result.pca_detector
                iforest_detector = quality_result.iforest_detector
                gmm_detector = quality_result.gmm_detector
                pca_train_spe = quality_result.pca_train_spe
                pca_train_t2 = quality_result.pca_train_t2

        # ===== Model Persistence: Save trained models with versioning ====="""
    
    new_lines = new_section.count('\n') + 1
    
    # Perform replacement
    new_content = content[:start_idx] + new_section + content[end_idx:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"SUCCESS: Replaced Model Quality Assessment section ({old_lines} lines -> {new_lines} lines)")
    print(f"Removed {old_lines - new_lines} lines from main()")

if __name__ == "__main__":
    main()
