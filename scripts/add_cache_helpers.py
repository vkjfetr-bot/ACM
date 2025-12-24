#!/usr/bin/env python
"""Add model cache loading and validation helper functions to acm_main.py."""
import re

# Read the current file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find insertion point - after _apply_regime_health_labels function
# Look for the function and find its end
insert_marker = "def _run_autonomous_tuning("
insert_pos = content.find(insert_marker)

if insert_pos == -1:
    print("ERROR: Could not find _run_autonomous_tuning function")
    exit(1)

# New helper functions to add
new_helpers = '''

# ---------------------------------------------------------------------------
# Model Cache Loading and Validation Helpers (extracted for clarity)
# ---------------------------------------------------------------------------


@dataclass
class LocalCacheValidationResult:
    """Result of local detector cache validation."""
    cache_valid: bool
    detector_cache: Optional[Dict[str, Any]]
    invalidation_reasons: List[str]


def _validate_local_detector_cache(
    model_cache_path: Path,
    current_train_columns: List[str],
    train_feature_hash: str,
    config_signature: str,
    equip: str,
    T: Any,
) -> LocalCacheValidationResult:
    """
    Validate local detector cache from joblib file.
    
    Checks that columns, train hash, and config signature all match.
    Returns validation result with cache bundle if valid.
    """
    import joblib
    
    try:
        cached_bundle = joblib.load(model_cache_path)
        cached_cols = cached_bundle.get("train_columns")
        cached_hash = cached_bundle.get("train_hash")
        cached_cfg_sig = cached_bundle.get("config_signature")
        
        # Validate cache: columns, train data hash, and config signature must match
        with T.section("models.cache_local.validate"):
            cols_match = (cached_cols == current_train_columns)
            hash_match = (cached_hash is None or cached_hash == train_feature_hash)
            cfg_match = (cached_cfg_sig is None or cached_cfg_sig == config_signature)
        
        if cols_match and hash_match and cfg_match:
            Console.info(f"Reusing cached detectors from {model_cache_path}", component="MODEL")
            Console.info(f"Cache validated: config_sig={config_signature[:8]}...", component="MODEL")
            return LocalCacheValidationResult(
                cache_valid=True,
                detector_cache=cached_bundle,
                invalidation_reasons=[],
            )
        else:
            reasons = []
            if not cols_match:
                reasons.append("columns changed")
            if not hash_match:
                reasons.append(f"train data changed ({cached_hash[:8] if cached_hash else 'none'} -> {train_feature_hash[:8]})")
            if not cfg_match:
                reasons.append(f"config changed ({cached_cfg_sig[:8] if cached_cfg_sig else 'none'} -> {config_signature[:8]})")
            Console.warn(f"Cache invalidated ({', '.join(reasons)}); re-fitting.", component="MODEL",
                         equip=equip, invalidation_reasons=reasons[:5])
            return LocalCacheValidationResult(
                cache_valid=False,
                detector_cache=None,
                invalidation_reasons=reasons,
            )
    except Exception as e:
        Console.warn(f"Failed to load cached detectors: {e}", component="MODEL",
                     equip=equip, error_type=type(e).__name__, error=str(e)[:200])
        return LocalCacheValidationResult(
            cache_valid=False,
            detector_cache=None,
            invalidation_reasons=[f"Load error: {e}"],
        )


def _infer_equip_id(
    meta: Any,
    cfg: Dict[str, Any],
    SQL_MODE: bool,
    dual_mode: bool,
    equip_id: int,
    T: Any,
) -> int:
    """
    Infer EquipID from meta or config for non-SQL modes.
    
    CRITICAL: In SQL_MODE, equip_id is already set by _sql_start_run() from stored procedure.
    Do NOT override it! Only infer for file/dual modes.
    
    Args:
        meta: Metadata object that may contain equip_id
        cfg: Configuration dictionary
        SQL_MODE: Whether running in SQL mode
        dual_mode: Whether running in dual mode
        equip_id: Current equip_id value (from SQL mode or 0)
        T: Timer context
        
    Returns:
        Inferred equip_id (positive integer in SQL/dual modes)
        
    Raises:
        RuntimeError: If equip_id is invalid in SQL/dual modes
    """
    if not SQL_MODE:
        with T.section("data.equip_id_infer"):
            try:
                equip_id = int(getattr(meta, "equip_id", 0) or 0)
            except Exception:
                equip_id = 0
            
            # For dual mode, try config fallback if meta didn't provide it
            if dual_mode and equip_id == 0:
                equip_id_cfg = cfg.get("runtime", {}).get("equip_id", equip_id)
                try:
                    equip_id = int(equip_id_cfg)
                except Exception:
                    equip_id = 0
    
    # Validate equip_id for SQL/dual modes
    if (SQL_MODE or dual_mode) and equip_id <= 0:
        raise RuntimeError(
            f"EquipID is required and must be a positive integer in SQL/dual mode. "
            f"Current value: {equip_id}. In SQL mode, this should come from _sql_start_run(). "
            f"In dual mode, set runtime.equip_id in config OR ensure load_data provides it."
        )
    
    return equip_id


@dataclass
class PersistenceLoadResult:
    """Result of loading models from persistence layer."""
    cached_models: Optional[Dict[str, Any]]
    cached_manifest: Optional[Dict[str, Any]]
    load_successful: bool
    invalid_reasons: List[str]


def _load_models_from_persistence(
    train: pd.DataFrame,
    cfg: Dict[str, Any],
    equip: str,
    art_root: str,
    sql_client: Optional[Any],
    equip_id: int,
    SQL_MODE: bool,
    dual_mode: bool,
    T: Any,
) -> PersistenceLoadResult:
    """
    Load cached models from the persistence layer (ModelVersionManager).
    
    Validates cache against current config signature and sensors.
    
    Args:
        train: Training dataframe for sensor validation
        cfg: Configuration dictionary
        equip: Equipment name
        art_root: Artifact root path
        sql_client: SQL client (for SQL/dual modes)
        equip_id: Equipment ID
        SQL_MODE: Whether running in SQL mode
        dual_mode: Whether running in dual mode
        T: Timer context
        
    Returns:
        PersistenceLoadResult with cached_models and manifest if valid
    """
    from core.model_persistence import ModelVersionManager
    
    try:
        model_manager = ModelVersionManager(
            equip=equip,
            artifact_root=Path(art_root),
            sql_client=sql_client if SQL_MODE or dual_mode else None,
            equip_id=equip_id if SQL_MODE or dual_mode else None
        )
        cached_models, cached_manifest = model_manager.load_models()
        
        if cached_models and cached_manifest:
            # Validate cache
            current_config_sig = cfg.get("_signature", "unknown")
            current_sensors = list(train.columns) if hasattr(train, 'columns') else []
            
            with T.section("models.persistence.validate"):
                is_valid, invalid_reasons = model_manager.check_model_validity(
                    manifest=cached_manifest,
                    current_config_signature=current_config_sig,
                    current_sensors=current_sensors
                )
            
            if is_valid:
                # Enhanced logging for cached model acceptance
                Console.info(f"Using cached models from v{cached_manifest['version']}", component="MODEL")
                Console.info(f"Cache created: {cached_manifest.get('created_at', 'unknown')}", component="MODEL")
                Console.info(f"Config signature: {current_config_sig[:16]}... (unchanged)", component="MODEL")
                Console.info(f"Sensor count: {len(current_sensors)} (matching cached)", component="MODEL")
                if 'created_at' in cached_manifest:
                    from datetime import datetime
                    try:
                        created_at = datetime.fromisoformat(cached_manifest['created_at'])
                        age_hours = (datetime.now() - created_at).total_seconds() / 3600
                        Console.info(f"Model age: {age_hours:.1f}h ({age_hours/24:.1f}d)", component="MODEL")
                    except Exception:
                        pass
                
                return PersistenceLoadResult(
                    cached_models=cached_models,
                    cached_manifest=cached_manifest,
                    load_successful=True,
                    invalid_reasons=[],
                )
            else:
                # Enhanced logging for retrain trigger reasons
                Console.warn(f"Cached models invalid, retraining required:", component="MODEL",
                             equip=equip, invalid_reason_count=len(invalid_reasons))
                for reason in invalid_reasons:
                    Console.warn(f"- {reason}", component="MODEL", equip=equip)
                return PersistenceLoadResult(
                    cached_models=None,
                    cached_manifest=None,
                    load_successful=False,
                    invalid_reasons=invalid_reasons,
                )
        else:
            return PersistenceLoadResult(
                cached_models=None,
                cached_manifest=None,
                load_successful=False,
                invalid_reasons=["No cached models found"],
            )
    except Exception as e:
        Console.warn(f"Failed to load cached models: {e}", component="MODEL",
                     equip=equip, error_type=type(e).__name__, error=str(e)[:200])
        return PersistenceLoadResult(
            cached_models=None,
            cached_manifest=None,
            load_successful=False,
            invalid_reasons=[f"Load error: {e}"],
        )


'''

# Insert the new helpers before _run_autonomous_tuning
new_content = content[:insert_pos] + new_helpers + content[insert_pos:]

# Write the updated file
with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("SUCCESS: Added cache loading helper functions:")
print("  - LocalCacheValidationResult dataclass")
print("  - _validate_local_detector_cache()")
print("  - _infer_equip_id()")
print("  - PersistenceLoadResult dataclass")
print("  - _load_models_from_persistence()")
