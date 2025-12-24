#!/usr/bin/env python3
"""Replace run stats and culprits sections with helper call."""

with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find start - "# Optional compact RunStats"
start_marker = "# Optional compact RunStats (best-effort)"
start_idx = content.find(start_marker)

if start_idx == -1:
    print("ERROR: Could not find start marker")
    exit(1)

# Find end - "if reuse_models and cache_payload:"
end_marker = "if reuse_models and cache_payload:"
end_idx = content.find(end_marker, start_idx)

if end_idx == -1:
    print("ERROR: Could not find end marker")
    exit(1)

# Extract old code
old_code = content[start_idx:end_idx]

# New replacement code
new_code = '''# Run stats and episode culprits
        _write_run_stats_and_culprits(
            frame=frame, episodes=episodes, meta=meta, output_manager=output_manager,
            sql_client=sql_client, equip_id=equip_id, run_id=run_id, rows_read=rows_read,
            anomaly_count=anomaly_count, win_start=win_start, win_end=win_end,
            equip=equip, T=T,
        )

        '''

# Replace
content = content[:start_idx] + new_code + content[end_idx:]

with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(content)

old_lines = old_code.count('\n')
new_lines = new_code.count('\n')
print(f"SUCCESS: Replaced run stats/culprits section ({old_lines} lines -> {new_lines} lines)")
print(f"         Removed {old_lines - new_lines} lines from main()")
