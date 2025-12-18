#!/usr/bin/env python3
"""
Migrate Console calls from old format to new structured format.

OLD: Console.info("[DATA] Loading 5000 rows")
NEW: Console.info("Loading 5000 rows", component="DATA")

This script transforms all Console.info/warn/error/ok/debug calls
to use the component parameter instead of embedding [COMPONENT] in the message.
"""
import re
import sys
from pathlib import Path

# Pattern to match Console calls with [COMPONENT] in message
# Matches: Console.info("[COMPONENT] message") or Console.info(f"[COMPONENT] message")
CONSOLE_PATTERN = re.compile(
    r'''(Console\.(info|warn|warning|error|ok|debug)\()'''  # Console.method(
    r'''(f?["\'])'''                                         # f" or " or f' or '
    r'''\[([A-Z0-9_:-]+)\]\s*'''                              # [COMPONENT] 
    r'''(.+?)'''                                              # message content
    r'''(["\'])'''                                            # closing quote
    r'''(\s*,\s*skip_loki\s*=\s*(?:True|False))?'''           # optional skip_loki
    r'''(\s*\))'''                                            # closing paren or more args
)

def transform_line(line: str) -> str:
    """Transform a single line, handling Console calls with [COMPONENT]."""
    
    def replacer(match):
        prefix = match.group(1)        # Console.info(
        method = match.group(2)        # info/warn/error/ok/debug
        quote_start = match.group(3)   # f" or " or f' or '
        component = match.group(4)     # COMPONENT
        message = match.group(5)       # rest of message
        quote_end = match.group(6)     # closing quote
        skip_loki = match.group(7) or ""  # optional skip_loki
        suffix = match.group(8)        # closing paren
        
        # Build new format
        new_call = f'{prefix}{quote_start}{message}{quote_end}, component="{component}"{skip_loki}{suffix}'
        return new_call
    
    return CONSOLE_PATTERN.sub(replacer, line)


def process_file(filepath: Path, dry_run: bool = True) -> int:
    """Process a single file, return count of changes."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"ERROR reading {filepath}: {e}")
        return 0
    
    lines = content.split('\n')
    new_lines = []
    changes = 0
    
    for i, line in enumerate(lines, 1):
        new_line = transform_line(line)
        if new_line != line:
            changes += 1
            if dry_run:
                print(f"  L{i}: {line.strip()}")
                print(f"    -> {new_line.strip()}")
        new_lines.append(new_line)
    
    if not dry_run and changes > 0:
        filepath.write_text('\n'.join(new_lines), encoding='utf-8')
    
    return changes


def main():
    dry_run = "--apply" not in sys.argv
    
    if dry_run:
        print("DRY RUN MODE - use --apply to make changes")
        print("=" * 60)
    
    # Find all Python files
    root = Path(__file__).parent.parent
    patterns = ["core/*.py", "scripts/*.py", "scripts/sql/*.py", "scripts/archive/*.py", "utils/*.py", "tests/*.py"]
    
    total_changes = 0
    files_changed = 0
    
    for pattern in patterns:
        for filepath in root.glob(pattern):
            if filepath.name == "migrate_console_calls.py":
                continue
            if filepath.name == "observability.py":
                continue  # Don't modify the Console class itself
                
            changes = process_file(filepath, dry_run)
            if changes > 0:
                files_changed += 1
                total_changes += changes
                print(f"{filepath.name}: {changes} changes")
    
    print("=" * 60)
    print(f"Total: {total_changes} changes in {files_changed} files")
    
    if dry_run:
        print("\nRun with --apply to make changes")


if __name__ == "__main__":
    main()
