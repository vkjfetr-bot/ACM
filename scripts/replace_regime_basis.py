#!/usr/bin/env python3
"""Replace the regime basis building section with _build_regime_basis call."""

with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Old pattern to find - the regime basis building section
old_pattern = '''        try:
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

        if regime_model is not None:
            if (
                regime_basis_train is None
                or regime_model.feature_columns != list(regime_basis_train.columns)
                or (regime_basis_hash is not None and regime_model.train_hash != regime_basis_hash)
            ):
                Console.warn("Cached regime model mismatch; will refit.", component="REGIME",
                             equip=equip)
                regime_model = None'''

# New replacement
new_replacement = '''        # Build regime feature basis and validate cached model
        regime_basis, regime_model = _build_regime_basis(
            train=train,
            score=score,
            train_numeric=train_numeric,
            score_numeric=score_numeric,
            pca_detector=pca_detector,
            regime_model=regime_model,
            cfg=cfg,
            equip=equip,
        )
        regime_basis_train = regime_basis.regime_basis_train
        regime_basis_score = regime_basis.regime_basis_score
        regime_basis_meta = regime_basis.regime_basis_meta
        regime_basis_hash = regime_basis.regime_basis_hash'''

if old_pattern in content:
    content = content.replace(old_pattern, new_replacement)
    with open("core/acm_main.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    # Calculate savings
    old_chars = len(old_pattern)
    new_chars = len(new_replacement)
    print(f"SUCCESS: Replaced regime basis building section")
    print(f"Original: {old_chars} chars, New: {new_chars} chars")
    print(f"Removed: {old_chars - new_chars} chars")
else:
    print("ERROR: Could not find old pattern to replace")
    # Debug: show what we're looking for
    if "regimes.build_feature_basis" in content:
        print("Found build_feature_basis but pattern didn't match exactly")
    else:
        print("build_feature_basis not found in file")
