#!/usr/bin/env python
"""Replace model persistence loading section with helper call."""
import re

# Read the current file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the start of the section
start_marker = "        if use_cache and detector_cache is None:\n            with T.section(\"models.persistence.load\"):\n                try:"
start_pos = content.find(start_marker)

if start_pos == -1:
    print("ERROR: Could not find start of model persistence loading section")
    exit(1)

# Find the end marker - the line after the finally block
end_marker = "\n        # Initialize detector variables"
end_pos = content.find(end_marker, start_pos)

if end_pos == -1:
    print("ERROR: Could not find end of model persistence loading section")
    exit(1)

# Extract the old code
old_code = content[start_pos:end_pos]
print(f"Found old code: {len(old_code)} characters")

# New code to replace it with
new_code = '''        if use_cache and detector_cache is None:
            with T.section("models.persistence.load"):
                persist_result = _load_models_from_persistence(
                    train=train,
                    cfg=cfg,
                    equip=equip,
                    art_root=art_root,
                    sql_client=sql_client if SQL_MODE or dual_mode else None,
                    equip_id=equip_id,
                    SQL_MODE=SQL_MODE,
                    dual_mode=dual_mode,
                    T=T,
                )
                cached_models = persist_result.cached_models
                cached_manifest = persist_result.cached_manifest'''

# Replace the old code with new code
new_content = content[:start_pos] + new_code + content[end_pos:]

# Write the updated file
with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(new_content)

old_lines = old_code.count('\n')
new_lines = new_code.count('\n')
print(f"SUCCESS: Replaced model persistence loading section")
print(f"  Old: {old_lines} lines -> New: {new_lines} lines (-{old_lines - new_lines} lines)")
