#!/usr/bin/env python3
"""Replace SQL artifact writing section by position markers."""

with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the start marker
start_marker = "# === SQL-SPECIFIC ARTIFACT WRITING (BATCHED TRANSACTION) ==="
start_idx = content.find(start_marker)

if start_idx == -1:
    print("ERROR: Could not find start marker")
    exit(1)

# Find where the section ends - it ends at "# Aggregate row counts for finalize"
end_marker = "# Aggregate row counts for finalize"
end_idx = content.find(end_marker, start_idx)

if end_idx == -1:
    print("ERROR: Could not find end marker")
    exit(1)

# Find the end of the rows_written line (the line after end_marker)
# Go to end of line containing "rows_written = int(...)"
rows_written_line = content.find("rows_written = int(", end_idx)
if rows_written_line == -1:
    print("ERROR: Could not find rows_written line")
    exit(1)

# Find the end of that line
end_of_section = content.find("\n", rows_written_line) + 1

# Get the old code
old_code = content[start_idx:end_of_section]

# New replacement code
new_code = '''# === SQL-SPECIFIC ARTIFACT WRITING ===
        sql_artifact_result = _write_sql_artifacts(
            frame=frame, episodes=episodes, train=train, pca_detector=pca_detector,
            spe_p95_train=spe_p95_train, t2_p95_train=t2_p95_train,
            output_manager=output_manager, sql_client=sql_client, equip_id=equip_id,
            run_id=run_id, cfg=cfg, T=T, equip=equip,
        )
        rows_written = sql_artifact_result.total_rows
'''

# Replace - need to preserve proper indentation (8 spaces for method body)
content = content[:start_idx] + new_code + content[end_of_section:]

with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(content)

old_lines = old_code.count('\n')
new_lines = new_code.count('\n')
print(f"SUCCESS: Replaced SQL artifacts section ({old_lines} lines -> {new_lines} lines)")
print(f"         Removed {old_lines - new_lines} lines from main()")
