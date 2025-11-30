"""
Temporary script to remove chart generation code from output_manager.py.
Phase 1 of refactor/output-manager-bloat-removal.
"""
import sys
from pathlib import Path

def remove_chart_code():
    """Remove lines 2754-3728 (chart generation code)."""
    file_path = Path(r"c:\Users\bhadk\Documents\ACM V8 SQL\ACM\core\output_manager.py")
    
    # Read all lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Original file: {len(lines)} lines")
    
    # Find the section markers
    start_marker_line = None
    end_marker_line = None
    
    for i, line in enumerate(lines):
        if "# ==================== INDIVIDUAL TABLE GENERATORS ====================" in line:
            if start_marker_line is None:
                start_marker_line = i
                print(f"Found first section marker at line {i+1}")
            else:
                end_marker_line = i
                print(f"Found second section marker at line {i+1}")
                break
    
    if start_marker_line is None or end_marker_line is None:
        print("ERROR: Could not find both section markers")
        return 1
    
    # Remove everything between the markers (keep first marker, remove second)
    new_lines = lines[:start_marker_line+1] + ['\n'] + lines[end_marker_line+1:]
    
    print(f"Removing lines {start_marker_line+2} to {end_marker_line+1}")
    print(f"Removed {end_marker_line - start_marker_line} lines")
    print(f"New file: {len(new_lines)} lines")
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("✓ Chart generation code removed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(remove_chart_code())
