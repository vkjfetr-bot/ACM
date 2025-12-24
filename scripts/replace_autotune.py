#!/usr/bin/env python
"""Replace auto-tune fusion weights section with helper call."""
import re

# Read the current file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the start of the auto-tune section
start_marker = '''            # DET-06: Auto-tune weights before fusion
            tuned_weights = None
            tuning_diagnostics = None
            Console.info("Starting detector weight auto-tuning...", component="FUSE")
            with T.section("fusion.auto_tune"):'''
start_pos = content.find(start_marker)

if start_pos == -1:
    print("ERROR: Could not find start of auto-tune section")
    exit(1)

# Find the end marker 
end_marker = "\n            # Calculate fusion on TRAIN data for threshold calculation later"
end_pos = content.find(end_marker, start_pos)

if end_pos == -1:
    print("ERROR: Could not find end of auto-tune section")
    exit(1)

# Extract the old code
old_code = content[start_pos:end_pos]
print(f"Found old code: {len(old_code)} characters")

# New code to replace it with
new_code = '''            # DET-06: Auto-tune weights before fusion
            autotune_result = _auto_tune_fusion_weights(
                present=present,
                weights=weights,
                previous_weights=previous_weights,
                score=score,
                run_dir=run_dir,
                output_manager=output_manager,
                sql_client=sql_client if SQL_MODE else None,
                equip_id=int(equip_id),
                run_id=run_id,
                equip=equip,
                cfg=cfg,
                SQL_MODE=SQL_MODE,
                T=T,
            )
            weights = autotune_result.weights
            fusion_weights_used = autotune_result.fusion_weights_used
            tuning_diagnostics = autotune_result.tuning_diagnostics'''

# Replace the old code with new code
new_content = content[:start_pos] + new_code + content[end_pos:]

# Write the updated file
with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(new_content)

old_lines = old_code.count('\n')
new_lines = new_code.count('\n')
print(f"SUCCESS: Replaced auto-tune fusion weights section")
print(f"  Old: {old_lines} lines -> New: {new_lines} lines (-{old_lines - new_lines} lines)")
