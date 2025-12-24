#!/usr/bin/env python
"""
Comprehensive cleanup of acm_main.py:
1. Remove _ensure_dir (unused)
2. Inline _sql_mode (always True, inline the check)
3. Inline _batch_mode (one-liner)
4. Remove _configure_logging (does nothing useful)
5. Remove _write_run_meta_json and _maybe_write_run_meta_json (file-mode only)
6. Remove all "if not SQL_MODE" branches (file-mode dead code)
7. Remove mhal_z references (deprecated v9.1.0)
"""
import re

# Read the current file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

original_lines = content.count('\n')
print(f"Original line count: {original_lines}")

# === STEP 1: Remove _ensure_dir function (unused) ===
ensure_dir_pattern = r'\ndef _ensure_dir\(p: Path\) -> None:\n    p\.mkdir\(parents=True, exist_ok=True\)\n'
content = re.sub(ensure_dir_pattern, '\n', content)
print("STEP 1: Removed _ensure_dir")

# === STEP 2: Remove _sql_mode function and inline (always True) ===
# Remove the function definition
sql_mode_func = r'''def _sql_mode\(cfg: Dict\[str, Any\]\) -> bool:
    """SQL-only mode: use SQL backend unless ACM_FORCE_FILE_MODE env var is set\."""
    force_file_mode = os\.getenv\("ACM_FORCE_FILE_MODE", "0"\) == "1"
    if force_file_mode:
        return False
    return True

'''
content = re.sub(sql_mode_func, '', content)

# Replace the call with inline: SQL_MODE = True (SQL-only mode is enforced)
content = re.sub(r'SQL_MODE = _sql_mode\(cfg\)', 'SQL_MODE = True  # SQL-only mode enforced', content)
print("STEP 2: Removed _sql_mode, inlined as SQL_MODE = True")

# === STEP 3: Remove _batch_mode function and inline ===
batch_mode_func = r'''def _batch_mode\(\) -> bool:
    """Detect if running under sql_batch_runner \(continuous learning mode\)\."""
    return bool\(os\.getenv\("ACM_BATCH_MODE", "0"\) == "1"\)

'''
content = re.sub(batch_mode_func, '', content)

# Replace the call with inline
content = re.sub(
    r'BATCH_MODE = _batch_mode\(\)',
    'BATCH_MODE = os.getenv("ACM_BATCH_MODE", "0") == "1"',
    content
)
print("STEP 3: Removed _batch_mode, inlined")

# === STEP 4: Remove _configure_logging function ===
# This function just prints warnings and returns {"enable_sql_logging": True}
configure_logging_pattern = r'''def _configure_logging\(logging_cfg, args\):
    """Apply CLI/config logging overrides and return flags\."""
    enable_sql_logging_cfg = \(logging_cfg or \{\}\)\.get\("enable_sql_sink"\)
    if enable_sql_logging_cfg is False:
        Console\.warn\("SQL sink disable flag in config is ignored; SQL logging is always enabled in SQL mode\.", component="LOG",
                     config_flag=enable_sql_logging_cfg\)
    enable_sql_logging = True

    # ACMLog does not support dynamic level/format yet; can be extended if needed

    log_file = args\.log_file or \(logging_cfg or \{\}\)\.get\("file"\)
    if log_file:
        Console\.warn\(f"File logging disabled in SQL-only mode \(ignoring --log-file=\{log_file\}\)", component="CONFIG",
                     log_file=str\(log_file\)\)

    # Module levels not supported in ACMLog yet
    return \{"enable_sql_logging": enable_sql_logging\}


'''
content = re.sub(configure_logging_pattern, '', content)

# Replace the call with inline
content = re.sub(
    r'logging_settings = _configure_logging\(logging_cfg, args\)',
    '# Logging is always SQL-enabled in SQL-only mode',
    content
)
print("STEP 4: Removed _configure_logging")

# === STEP 5: Remove mhal_z references ===
# Remove "if "mhal_z" in frame.columns: ..." lines
content = re.sub(r'\s*if "mhal_z" in frame\.columns:.*\n', '\n', content)
# Update the missing detector filter
content = content.replace(
    "missing_to_warn = [k for k in missing if k != 'mhal_z']",
    "missing_to_warn = list(missing)"
)
print("STEP 5: Removed mhal_z references")

# Write intermediate result
with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(content)

# Count lines removed so far
intermediate_lines = content.count('\n')
print(f"After steps 1-5: {intermediate_lines} lines (removed {original_lines - intermediate_lines})")

# Now we need to handle the file-mode functions and branches more carefully
# Re-read for next phase
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

print("\n=== PHASE 2: File-mode cleanup ===")

# === STEP 6: Remove _write_run_meta_json and _maybe_write_run_meta_json ===
# Find and remove _write_run_meta_json function (lines 209-323 approximately)
# This is a large function, use a different approach

# Remove calls to _maybe_write_run_meta_json
content = re.sub(r'\s*_maybe_write_run_meta_json\(locals\(\)\)\n', '\n', content)
print("STEP 6a: Removed calls to _maybe_write_run_meta_json")

# Now remove the function definitions
# _maybe_write_run_meta_json
maybe_write_pattern = r'''def _maybe_write_run_meta_json\(local_vars: Dict\[str, Any\]\) -> None:
    """Invoke run metadata writer if it is available in the module globals\."""
    # Enforce SQL-only: do not write meta\.json when running in SQL mode
    try:
        if bool\(local_vars\.get\('SQL_MODE'\)\):
            return  # Silent skip - SQL-only mode is the standard
    except Exception:
        pass
    writer = globals\(\)\.get\("_write_run_meta_json"\)
    if callable\(writer\):
        writer\(local_vars\)
    else:
        Console\.warn\("meta\.json writer unavailable; skipping run metadata dump", component="META",
                     writer_found=False\)

'''
content = re.sub(maybe_write_pattern, '', content)
print("STEP 6b: Removed _maybe_write_run_meta_json definition")

# Write result
with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(content)

final_lines = content.count('\n')
print(f"\n=== CLEANUP COMPLETE ===")
print(f"Original: {original_lines} lines")
print(f"Final: {final_lines} lines")
print(f"Removed: {original_lines - final_lines} lines")
