"""
Phase 2: Remove CSV writing infrastructure from output_manager.py.
Part of refactor/output-manager-bloat-removal.
"""
from pathlib import Path


def remove_csv_methods():
    """Remove batch_write_csvs and _write_csv_optimized methods."""
    file_path = Path(r"c:\Users\bhadk\Documents\ACM V8 SQL\ACM\core\output_manager.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Original file: {len(lines)} lines")
    
    # Find and remove batch_write_csvs (lines ~2036-2070)
    batch_write_start = None
    batch_write_end = None
    
    for i, line in enumerate(lines):
        if "def batch_write_csvs(self" in line:
            batch_write_start = i
            print(f"Found batch_write_csvs at line {i+1}")
        
        if batch_write_start is not None and batch_write_end is None:
            # Find the next method definition
            if i > batch_write_start and line.strip().startswith("def ") and "batch_write_csvs" not in line:
                batch_write_end = i - 1
                # Backtrack to remove empty lines
                while batch_write_end > batch_write_start and not lines[batch_write_end].strip():
                    batch_write_end -= 1
                print(f"batch_write_csvs ends at line {batch_write_end+1}")
                break
    
    if batch_write_start is None:
        print("batch_write_csvs not found")
        return 1
    
    # Remove batch_write_csvs
    print(f"Removing lines {batch_write_start+1} to {batch_write_end+1} ({batch_write_end - batch_write_start + 1} lines)")
    lines_removed = batch_write_end - batch_write_start + 1
    new_lines = lines[:batch_write_start] + lines[batch_write_end+1:]
    
    # Now find and remove _write_csv_optimized
    lines = new_lines
    csv_write_start = None
    csv_write_end = None
    
    for i, line in enumerate(lines):
        if "def _write_csv_optimized(self" in line:
            csv_write_start = i
            print(f"Found _write_csv_optimized at line {i+1}")
        
        if csv_write_start is not None and csv_write_end is None:
            # Find the next method definition
            if i > csv_write_start and line.strip().startswith("def ") and "_write_csv_optimized" not in line:
                csv_write_end = i - 1
                # Backtrack to remove empty lines
                while csv_write_end > csv_write_start and not lines[csv_write_end].strip():
                    csv_write_end -= 1
                print(f"_write_csv_optimized ends at line {csv_write_end+1}")
                break
    
    if csv_write_start is None:
        print("_write_csv_optimized not found")
        return 1
    
    # Remove _write_csv_optimized
    print(f"Removing lines {csv_write_start+1} to {csv_write_end+1} ({csv_write_end - csv_write_start + 1} lines)")
    lines_removed += csv_write_end - csv_write_start + 1
    new_lines = lines[:csv_write_start] + lines[csv_write_end+1:]
    
    print(f"Total lines removed: {lines_removed}")
    print(f"New file: {len(new_lines)} lines")
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("✓ CSV writing methods removed successfully")
    return 0


def remove_csv_files_from_batch():
    """Remove csv_files field from OutputBatch dataclass."""
    file_path = Path(r"c:\Users\bhadk\Documents\ACM V8 SQL\ACM\core\output_manager.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and remove the csv_files line
    if "csv_files: Dict[Path, pd.DataFrame] = field(default_factory=dict)" in content:
        content = content.replace(
            "    csv_files: Dict[Path, pd.DataFrame] = field(default_factory=dict)\n",
            ""
        )
        print("✓ Removed csv_files field from OutputBatch")
    else:
        print("csv_files field not found in OutputBatch")
        return 1
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return 0


if __name__ == "__main__":
    import sys
    
    print("=== Phase 2: Remove CSV Writing Infrastructure ===\n")
    
    print("Step 1: Remove CSV writing methods...")
    result1 = remove_csv_methods()
    
    print("\nStep 2: Remove csv_files from OutputBatch...")
    result2 = remove_csv_files_from_batch()
    
    if result1 == 0 and result2 == 0:
        print("\n✓ Phase 2 complete!")
        sys.exit(0)
    else:
        print("\n✗ Phase 2 failed")
        sys.exit(1)
