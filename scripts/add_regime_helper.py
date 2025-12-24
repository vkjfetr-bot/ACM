#!/usr/bin/env python3
"""Add RegimeBasisResult dataclass and _build_regime_basis helper function."""

import re

# Read the file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the location to insert (before ScoreResult dataclass)
insert_marker = "@dataclass\nclass ScoreResult:"

# New code to insert
new_code = '''@dataclass
class RegimeBasisResult:
    """Result of building regime feature basis."""
    regime_basis_train: Optional[pd.DataFrame]
    regime_basis_score: Optional[pd.DataFrame]
    regime_basis_meta: Dict[str, Any]
    regime_basis_hash: Optional[int]


def _build_regime_basis(
    train: pd.DataFrame,
    score: pd.DataFrame,
    train_numeric: Optional[pd.DataFrame],
    score_numeric: Optional[pd.DataFrame],
    pca_detector: Optional[Any],
    regime_model: Optional[Any],
    cfg: Dict[str, Any],
    equip: str,
) -> Tuple[RegimeBasisResult, Optional[Any]]:
    """Build feature basis for regime detection and validate cached model.
    
    Args:
        train: Engineered training features.
        score: Engineered scoring features.
        train_numeric: Raw numeric training data.
        score_numeric: Raw numeric scoring data.
        pca_detector: Fitted PCA detector.
        regime_model: Cached regime model (may be invalidated).
        cfg: Configuration dictionary.
        equip: Equipment name for logging.
    
    Returns:
        Tuple of (RegimeBasisResult, updated_regime_model).
        If regime_model is invalidated, returns None for updated_regime_model.
    """
    try:
        regime_basis_train, regime_basis_score, regime_basis_meta = regimes.build_feature_basis(
            train_features=train,
            score_features=score,
            raw_train=train_numeric,
            raw_score=score_numeric,
            pca_detector=pca_detector,
            cfg=cfg,
        )
        regime_basis_hash = int(pd.util.hash_pandas_object(regime_basis_train, index=True).sum())
    except Exception as e:
        Console.warn(f"Failed to build regime feature basis: {e}", component="REGIME",
                     equip=equip, error_type=type(e).__name__, error=str(e)[:200])
        regime_basis_train = None
        regime_basis_score = None
        regime_basis_meta = {}
        regime_basis_hash = None

    # Validate cached regime model against new basis
    updated_regime_model = regime_model
    if regime_model is not None:
        if (
            regime_basis_train is None
            or regime_model.feature_columns != list(regime_basis_train.columns)
            or (regime_basis_hash is not None and regime_model.train_hash != regime_basis_hash)
        ):
            Console.warn("Cached regime model mismatch; will refit.", component="REGIME",
                         equip=equip)
            updated_regime_model = None

    result = RegimeBasisResult(
        regime_basis_train=regime_basis_train,
        regime_basis_score=regime_basis_score,
        regime_basis_meta=regime_basis_meta,
        regime_basis_hash=regime_basis_hash,
    )
    return result, updated_regime_model


'''

# Check if already added
if "class RegimeBasisResult:" in content:
    print("RegimeBasisResult already exists, skipping")
else:
    # Insert before ScoreResult
    if insert_marker in content:
        content = content.replace(insert_marker, new_code + insert_marker)
        
        with open("core/acm_main.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("SUCCESS: Added RegimeBasisResult and _build_regime_basis helper")
    else:
        print("ERROR: Could not find insertion marker")
