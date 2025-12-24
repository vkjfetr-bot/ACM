#!/usr/bin/env python
"""Replace detector enabled flags and regime state loading with helper calls."""
import re

# Read the current file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the start of the detector enabled section
start_marker = "        # PERF-03: Check fusion weights to skip disabled detectors (lazy evaluation)\n"
start_pos = content.find(start_marker)

if start_pos == -1:
    print("ERROR: Could not find start of detector enabled section")
    exit(1)

# Find the end marker - before detector reconstruction
end_marker = "\n        # Try loading from cache (either old detector_cache or new persistence system)"
end_pos = content.find(end_marker, start_pos)

if end_pos == -1:
    print("ERROR: Could not find end of regime state loading section")
    exit(1)

# Extract the old code
old_code = content[start_pos:end_pos]
print(f"Found old code: {len(old_code)} characters")

# New code to replace it with
new_code = '''        # PERF-03: Check fusion weights to skip disabled detectors (lazy evaluation)
        enabled_flags = _check_detector_enabled_flags(cfg)
        ar1_enabled = enabled_flags.ar1_enabled
        pca_enabled = enabled_flags.pca_enabled
        iforest_enabled = enabled_flags.iforest_enabled
        gmm_enabled = enabled_flags.gmm_enabled
        omr_enabled = enabled_flags.omr_enabled
        
        # Try loading cached regime model from RegimeState or joblib persistence
        regime_state_result = _load_regime_state_from_persistence(
            art_root=art_root,
            equip=equip,
            equip_id=equip_id,
            sql_client=sql_client if SQL_MODE or dual_mode else None,
            stable_models_dir=stable_models_dir,
            SQL_MODE=SQL_MODE,
            dual_mode=dual_mode,
        )
        regime_model = regime_state_result.regime_model
        regime_state = regime_state_result.regime_state
        regime_state_version = regime_state_result.regime_state_version'''

# Replace the old code with new code
new_content = content[:start_pos] + new_code + content[end_pos:]

# Write the updated file
with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(new_content)

old_lines = old_code.count('\n')
new_lines = new_code.count('\n')
print(f"SUCCESS: Replaced detector enabled flags and regime state loading")
print(f"  Old: {old_lines} lines -> New: {new_lines} lines (-{old_lines - new_lines} lines)")
