#!/usr/bin/env python
"""
Cleanup over-engineered patterns in acm_main.py:
1. Remove dead no-op functions (_start_heartbeat, _apply_module_overrides)
2. Inline trivial wrappers (_sql_mode, _batch_mode, _continuous_learning_enabled)
3. Remove heartbeat usage patterns
4. Merge tuning functions into parent
"""

import re

# Read the current file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

original_lines = content.count('\n')

# === 1. Remove _start_heartbeat function definition ===
# Pattern: def _start_heartbeat ... return _NoOpHeartbeat()
heartbeat_pattern = r'''def _start_heartbeat\(enabled: bool, \*args, \*\*kwargs\):
    # No-op heartbeat for now; can be extended to ACMLog progress if needed
    class _NoOpHeartbeat:
        def stop\(self\):
            return
    return _NoOpHeartbeat\(\)


'''
content = re.sub(heartbeat_pattern, '', content)
print("1. Removed _start_heartbeat function definition")

# === 2. Remove _apply_module_overrides function definition ===
override_pattern = r'''def _apply_module_overrides\(entries\):
    # ACMLog does not support per-module levels yet; skip or implement if needed
    pass


'''
content = re.sub(override_pattern, '', content)
print("2. Removed _apply_module_overrides function definition")

# === 3. Remove heartbeat usage in _load_pipeline_data ===
# Pattern: hb = _start_heartbeat(...) ... finally: hb.stop()
heartbeat_usage1 = r'''    hb = _start_heartbeat\(
        heartbeat_on,
        "Loading data \(read -> parse ts -> sort -> resample -> interpolate\)",
        next_hint="build features",
        eta_hint=eta_load,
    \)
    
    try:
        with'''
content = re.sub(heartbeat_usage1, '    with', content)

# Remove the corresponding finally: hb.stop()
finally_pattern1 = r'''        
    finally:
        hb\.stop\(\)


def _apply_adaptive_baseline'''
content = re.sub(finally_pattern1, '\n\ndef _apply_adaptive_baseline', content)
print("3. Removed heartbeat usage in _load_pipeline_data")

# === 4. Remove heartbeat usage in main() scoring section ===
heartbeat_usage2 = r'''        hb = _start_heartbeat\(
            heartbeat_on,
            "Scoring heads \(AR1, Correlation, Outliers\)",
            next_hint="calibration",
            eta_hint=eta_score,
        \)
        
        # Score all enabled detectors'''
content = re.sub(heartbeat_usage2, '        # Score all enabled detectors', content)

# Remove hb.stop() after scoring
hb_stop_pattern = r'''        hb\.stop\(\)
        
        # === Fused health zone ==='''
content = re.sub(hb_stop_pattern, '\n        # === Fused health zone ===', content)
print("4. Removed heartbeat usage in main() scoring section")

# === 5. Replace function calls with inline code ===
# Replace SQL_MODE = _sql_mode(cfg) with inline
old_mode_lines = '''    # === Mode Detection ===
    SQL_MODE = _sql_mode(cfg)
    BATCH_MODE = _batch_mode()
    CONTINUOUS_LEARNING = _continuous_learning_enabled(cfg, BATCH_MODE)'''

new_mode_lines = '''    # === Mode Detection ===
    # SQL-only mode: always true unless ACM_FORCE_FILE_MODE env var is set
    SQL_MODE = os.getenv("ACM_FORCE_FILE_MODE", "0") != "1"
    # Batch mode: detect if running under sql_batch_runner
    BATCH_MODE = os.getenv("ACM_BATCH_MODE", "0") == "1"
    # Continuous learning: enabled by default in batch mode
    cl_enabled_cfg = cfg.get("continuous_learning", {}).get("enabled", BATCH_MODE)
    CONTINUOUS_LEARNING = cl_enabled_cfg if BATCH_MODE else False'''

content = content.replace(old_mode_lines, new_mode_lines)
print("5. Inlined _sql_mode, _batch_mode, _continuous_learning_enabled")

# === 6. Remove the now-unused function definitions ===
sql_mode_def = r'''# =======================
# SQL helpers \(local\)
# =======================
def _sql_mode\(cfg: Dict\[str, Any\]\) -> bool:
    """SQL-only mode: use SQL backend unless ACM_FORCE_FILE_MODE env var is set\."""
    force_file_mode = os\.getenv\("ACM_FORCE_FILE_MODE", "0"\) == "1"
    if force_file_mode:
        return False
    return True

def _batch_mode\(\) -> bool:
    """Detect if running under sql_batch_runner \(continuous learning mode\)\."""
    return bool\(os\.getenv\("ACM_BATCH_MODE", "0"\) == "1"\)

def _continuous_learning_enabled\(cfg: Dict\[str, Any\], batch_mode: bool\) -> bool:
    """Check if continuous learning is enabled for this run\."""
    # In batch mode, default to continuous learning unless explicitly disabled
    if batch_mode:
        return cfg\.get\("continuous_learning", \{\}\)\.get\("enabled", True\)
    # In single-run mode, default to disabled
    return cfg\.get\("continuous_learning", \{\}\)\.get\("enabled", False\)

'''
content = re.sub(sql_mode_def, '', content)
print("6. Removed unused _sql_mode, _batch_mode, _continuous_learning_enabled definitions")

# === 7. Remove _ensure_dir function and inline ===
ensure_dir_def = r'''def _ensure_dir\(p: Path\) -> None:
    p\.mkdir\(parents=True, exist_ok=True\)

'''
content = re.sub(ensure_dir_def, '', content)
print("7. Removed _ensure_dir (can use p.mkdir(parents=True, exist_ok=True) inline)")

# === 8. Add 'os' import if not present (needed for getenv) ===
if "import os" not in content and "from os import" not in content:
    # Add after other stdlib imports
    content = content.replace(
        "import sys\n",
        "import os\nimport sys\n"
    )
    print("8. Added 'import os' for getenv calls")

# Write the result
with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(content)

final_lines = content.count('\n')
print(f"\n=== CLEANUP COMPLETE ===")
print(f"Lines: {original_lines} -> {final_lines} (removed {original_lines - final_lines})")
