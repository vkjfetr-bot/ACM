#!/usr/bin/env python
"""Add detector reconstruction helper functions to acm_main.py."""
import re

# Read the current file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find insertion point - after _load_models_from_persistence function
insert_marker = "def _run_autonomous_tuning("
insert_pos = content.find(insert_marker)

if insert_pos == -1:
    print("ERROR: Could not find _run_autonomous_tuning function")
    exit(1)

# New helper functions to add
new_helpers = '''

@dataclass
class DetectorReconstructResult:
    """Result of reconstructing detectors from cached models."""
    ar1_detector: Optional[Any]
    pca_detector: Optional[Any]
    iforest_detector: Optional[Any]
    gmm_detector: Optional[Any]
    omr_detector: Optional[Any]
    regime_model: Optional[Any]
    regime_quality_ok: bool
    col_meds: Optional[Dict[str, float]]
    success: bool


def _reconstruct_detectors_from_cache(
    cached_models: Dict[str, Any],
    cached_manifest: Optional[Dict[str, Any]],
    cfg: Dict[str, Any],
    equip: str,
) -> DetectorReconstructResult:
    """
    Reconstruct detector objects from cached model parameters.
    
    Args:
        cached_models: Dictionary of cached model parameters
        cached_manifest: Model manifest with metadata
        cfg: Configuration dictionary
        equip: Equipment name
        
    Returns:
        DetectorReconstructResult with reconstructed detectors
    """
    ar1_detector = pca_detector = iforest_detector = gmm_detector = omr_detector = None
    regime_model = None
    regime_quality_ok = True
    col_meds = None
    
    try:
        # Reconstruct detector objects from cached models
        # Note: We need to pass empty configs since we're loading pre-trained models
        if "ar1_params" in cached_models and cached_models["ar1_params"]:
            ar1_detector = AR1Detector(ar1_cfg={})
            ar1_detector.phimap = cached_models["ar1_params"]["phimap"]
            ar1_detector.sdmap = cached_models["ar1_params"]["sdmap"]
            ar1_detector._is_fitted = True
        
        if "pca_model" in cached_models and cached_models["pca_model"]:
            pca_detector = correlation.PCASubspaceDetector(pca_cfg={})
            pca_detector.pca = cached_models["pca_model"]
            pca_detector._is_fitted = True
        
        # NOTE: MHAL removed v9.1.0 - redundant with PCA-T2
        
        if "iforest_model" in cached_models and cached_models["iforest_model"]:
            iforest_detector = outliers.IsolationForestDetector(if_cfg={})
            iforest_detector.model = cached_models["iforest_model"]
            iforest_detector._is_fitted = True
        
        if "gmm_model" in cached_models and cached_models["gmm_model"]:
            gmm_detector = outliers.GMMDetector(gmm_cfg={})
            gmm_detector.model = cached_models["gmm_model"]
            gmm_detector._is_fitted = True
        
        if "omr_model" in cached_models and cached_models["omr_model"]:
            omr_cfg = (cfg.get("models", {}).get("omr", {}) or {})
            omr_detector = OMRDetector.from_dict(cached_models["omr_model"], cfg=omr_cfg)
        
        if "regime_model" in cached_models and cached_models["regime_model"]:
            from core.regimes import RegimeModel
            regime_model = RegimeModel()
            regime_model.model = cached_models["regime_model"]
            if cached_manifest:
                regime_quality_ok = cached_manifest.get("models", {}).get("regimes", {}).get("quality", {}).get("quality_ok", True)
            # CRITICAL FIX: Validate regime model compatibility
            if regime_model.model is None:
                Console.warn("Cached regime model is None; discarding.", component="REGIME", equip=equip)
                regime_model = None
        
        if "feature_medians" in cached_models and cached_models["feature_medians"] is not None:
            col_meds = cached_models["feature_medians"]
        
        # Validate all critical models loaded (MHAL removed v9.1.0)
        if all([ar1_detector, pca_detector, iforest_detector]):
            Console.info("Successfully loaded all models from cache", component="MODEL")
            return DetectorReconstructResult(
                ar1_detector=ar1_detector,
                pca_detector=pca_detector,
                iforest_detector=iforest_detector,
                gmm_detector=gmm_detector,
                omr_detector=omr_detector,
                regime_model=regime_model,
                regime_quality_ok=regime_quality_ok,
                col_meds=col_meds,
                success=True,
            )
        else:
            missing = []
            if not ar1_detector: missing.append("ar1")
            if not pca_detector: missing.append("pca")
            if not iforest_detector: missing.append("iforest")
            Console.warn(f"Incomplete model cache, missing: {missing}, retraining required", component="MODEL",
                         equip=equip, missing_models=missing)
            return DetectorReconstructResult(
                ar1_detector=None,
                pca_detector=None,
                iforest_detector=None,
                gmm_detector=None,
                omr_detector=None,
                regime_model=None,
                regime_quality_ok=regime_quality_ok,
                col_meds=col_meds,
                success=False,
            )
            
    except Exception as e:
        import traceback
        Console.warn(f"Failed to reconstruct detectors from cache: {e}", component="MODEL",
                     equip=equip, error_type=type(e).__name__, error=str(e)[:500])
        Console.warn(f"Traceback: {traceback.format_exc()}", component="MODEL", equip=equip)
        return DetectorReconstructResult(
            ar1_detector=None,
            pca_detector=None,
            iforest_detector=None,
            gmm_detector=None,
            omr_detector=None,
            regime_model=None,
            regime_quality_ok=True,
            col_meds=None,
            success=False,
        )


@dataclass
class LocalCacheApplyResult:
    """Result of applying local detector cache."""
    ar1_detector: Optional[Any]
    pca_detector: Optional[Any]
    iforest_detector: Optional[Any]
    gmm_detector: Optional[Any]
    regime_model: Optional[Any]
    regime_basis_hash: Optional[str]
    regime_quality_ok: bool
    cache_valid: bool


def _apply_local_detector_cache(
    detector_cache: Dict[str, Any],
    equip: str,
) -> LocalCacheApplyResult:
    """
    Apply local detector cache (legacy joblib cache fallback).
    
    Args:
        detector_cache: Dictionary of cached detector objects
        equip: Equipment name
        
    Returns:
        LocalCacheApplyResult with detectors and validation status
    """
    ar1_detector = detector_cache.get("ar1")
    pca_detector = detector_cache.get("pca")
    iforest_detector = detector_cache.get("iforest")
    gmm_detector = detector_cache.get("gmm")
    regime_model = detector_cache.get("regime_model")
    cached_regime_hash = detector_cache.get("regime_basis_hash")
    regime_quality_ok = bool(detector_cache.get("regime_quality_ok", True))
    
    if regime_model is not None:
        regime_model.meta["quality_ok"] = regime_quality_ok
    
    # MHAL removed v9.1.0 - redundant with PCA-T2
    if not all([ar1_detector, pca_detector, iforest_detector]):
        Console.warn("Cached detectors incomplete; re-fitting this run.", component="MODEL", equip=equip)
        return LocalCacheApplyResult(
            ar1_detector=None,
            pca_detector=None,
            iforest_detector=None,
            gmm_detector=None,
            regime_model=None,
            regime_basis_hash=None,
            regime_quality_ok=regime_quality_ok,
            cache_valid=False,
        )
    else:
        Console.info("Using cached detectors from previous training run.", component="MODEL")
        if regime_model is not None and cached_regime_hash is not None:
            regime_model.train_hash = cached_regime_hash
        return LocalCacheApplyResult(
            ar1_detector=ar1_detector,
            pca_detector=pca_detector,
            iforest_detector=iforest_detector,
            gmm_detector=gmm_detector,
            regime_model=regime_model,
            regime_basis_hash=cached_regime_hash,
            regime_quality_ok=regime_quality_ok,
            cache_valid=True,
        )


'''

# Insert the new helpers before _run_autonomous_tuning
new_content = content[:insert_pos] + new_helpers + content[insert_pos:]

# Write the updated file
with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("SUCCESS: Added detector reconstruction helper functions:")
print("  - DetectorReconstructResult dataclass")
print("  - _reconstruct_detectors_from_cache()")
print("  - LocalCacheApplyResult dataclass")
print("  - _apply_local_detector_cache()")
