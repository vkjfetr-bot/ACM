#!/usr/bin/env python
"""
Remove all 'if not SQL_MODE' branches from acm_main.py.
Since SQL_MODE is always True, these branches are dead code.
"""
import re

with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

original_lines = content.count('\n')
print(f"Original line count: {original_lines}")

# Strategy: Find 'if not SQL_MODE:' blocks and remove them
# This is complex because we need to handle indentation properly

# Pattern 1: Simple if not SQL_MODE blocks with no else
# These just need the entire if block removed

# Pattern 2: elif not SQL_MODE: blocks - remove the elif and its contents
# Pattern 3: if not SQL_MODE and (...): - remove the block

lines = content.split('\n')
result = []
skip_until_dedent = False
skip_indent_level = 0
i = 0

while i < len(lines):
    line = lines[i]
    
    # Check if we're in skip mode
    if skip_until_dedent:
        # Check if this line is dedented past our skip level
        stripped = line.lstrip()
        if stripped and not stripped.startswith('#'):
            current_indent = len(line) - len(stripped)
            if current_indent <= skip_indent_level:
                skip_until_dedent = False
                # Don't skip this line, process it normally
            else:
                i += 1
                continue  # Skip this line
        elif not stripped:
            # Empty line while skipping - might be in the block
            i += 1
            continue
        else:
            i += 1
            continue
    
    # Check for 'if not SQL_MODE:' pattern
    stripped = line.lstrip()
    if 'if not SQL_MODE:' in stripped or 'if not SQL_MODE and' in stripped:
        # This is a file-mode only block
        current_indent = len(line) - len(stripped)
        skip_indent_level = current_indent
        skip_until_dedent = True
        print(f"Removing 'if not SQL_MODE' block at line {i+1}")
        i += 1
        continue
    
    # Check for 'elif not SQL_MODE:' pattern
    if 'elif not SQL_MODE:' in stripped:
        current_indent = len(line) - len(stripped)
        skip_indent_level = current_indent
        skip_until_dedent = True
        print(f"Removing 'elif not SQL_MODE' block at line {i+1}")
        i += 1
        continue
    
    result.append(line)
    i += 1

new_content = '\n'.join(result)

# Also clean up any orphaned comments about file mode
new_content = re.sub(r'\n\s*# File mode:.*\n', '\n', new_content)
new_content = re.sub(r'\n\s*# ACM-CSV-\d+:.*file mode.*\n', '\n', new_content, flags=re.IGNORECASE)

# Clean up double blank lines
new_content = re.sub(r'\n{3,}', '\n\n', new_content)

with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(new_content)

final_lines = new_content.count('\n')
print(f"\n=== CLEANUP COMPLETE ===")
print(f"Original: {original_lines} lines")
print(f"Final: {final_lines} lines")
print(f"Removed: {original_lines - final_lines} lines")
