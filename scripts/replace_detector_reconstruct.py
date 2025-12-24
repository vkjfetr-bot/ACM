#!/usr/bin/env python
"""Replace detector reconstruction sections with helper calls."""
import re

# Read the current file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the start of the cached_models section
start_marker = '        if cached_models:\n            with T.section("models.persistence.rebuild"):\n                # Load from new persistence system\n                try:'
start_pos = content.find(start_marker)

if start_pos == -1:
    print("ERROR: Could not find start of cached_models section")
    exit(1)

# Find the end marker - before "# Create detector state for fitting"
end_marker = "\n        # Create detector state for fitting"
end_pos = content.find(end_marker, start_pos)

if end_pos == -1:
    print("ERROR: Could not find end of detector reconstruction section")
    exit(1)

# Extract the old code
old_code = content[start_pos:end_pos]
print(f"Found old code: {len(old_code)} characters")

# New code to replace it with
new_code = '''        if cached_models:
            with T.section("models.persistence.rebuild"):
                reconstruct_result = _reconstruct_detectors_from_cache(
                    cached_models=cached_models,
                    cached_manifest=cached_manifest,
                    cfg=cfg,
                    equip=equip,
                )
                if reconstruct_result.success:
                    ar1_detector = reconstruct_result.ar1_detector
                    pca_detector = reconstruct_result.pca_detector
                    iforest_detector = reconstruct_result.iforest_detector
                    gmm_detector = reconstruct_result.gmm_detector
                    omr_detector = reconstruct_result.omr_detector
                    regime_model = reconstruct_result.regime_model
                    regime_quality_ok = reconstruct_result.regime_quality_ok
                    if reconstruct_result.col_meds is not None:
                        col_meds = reconstruct_result.col_meds
                else:
                    ar1_detector = pca_detector = iforest_detector = gmm_detector = None
        
        elif detector_cache:
            with T.section("models.cache_local.apply"):
                local_result = _apply_local_detector_cache(
                    detector_cache=detector_cache,
                    equip=equip,
                )
                if local_result.cache_valid:
                    ar1_detector = local_result.ar1_detector
                    pca_detector = local_result.pca_detector
                    iforest_detector = local_result.iforest_detector
                    gmm_detector = local_result.gmm_detector
                    regime_model = local_result.regime_model
                    regime_basis_hash = local_result.regime_basis_hash
                    regime_quality_ok = local_result.regime_quality_ok
                else:
                    detector_cache = None
                    regime_model = None'''

# Replace the old code with new code
new_content = content[:start_pos] + new_code + content[end_pos:]

# Write the updated file
with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(new_content)

old_lines = old_code.count('\n')
new_lines = new_code.count('\n')
print(f"SUCCESS: Replaced detector reconstruction sections")
print(f"  Old: {old_lines} lines -> New: {new_lines} lines (-{old_lines - new_lines} lines)")
