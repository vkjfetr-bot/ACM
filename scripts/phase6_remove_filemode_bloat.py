"""
Phase 6: Remove remaining file-mode bloat.
- Remove unused RULConfig/AR1 import
- Remove deprecated _to_naive wrapper functions (replace with direct calls)
- Remove JSON writing methods (file-mode only)
- Remove json_files from OutputBatch
- Clean up flush() method references to removed features
"""
from pathlib import Path
import re


def remove_rul_import():
    """Remove unused RULConfig import block."""
    file_path = Path(r"c:\Users\bhadk\Documents\ACM V8 SQL\ACM\core\output_manager.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove the entire try-except block for RULConfig
    old_block = """# Optional import for reusing AR(1) forecast helper in per-sensor forecasting
try:  # pragma: no cover - defensive import
    from core.rul_engine import RULConfig  # type: ignore
    # Note: _simple_ar1_forecast was a helper from old rul_estimator - can be refactored later
    _simple_ar1_forecast = None  # type: ignore  # TODO: migrate to new rul_engine if needed
except Exception:  # pragma: no cover
    RULConfig = None  # type: ignore
    _simple_ar1_forecast = None  # type: ignore

"""
    
    if old_block in content:
        content = content.replace(old_block, "")
        print("✓ Removed RULConfig/AR1 import block (10 lines)")
    else:
        print("! RULConfig import not found (may have changed)")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return 0


def replace_to_naive_calls():
    """Replace _to_naive calls with normalize_timestamp_scalar."""
    file_path = Path(r"c:\Users\bhadk\Documents\ACM V8 SQL\ACM\core\output_manager.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace _to_naive with normalize_timestamp_scalar
    content = content.replace('_to_naive(', 'normalize_timestamp_scalar(')
    
    # Replace _to_naive_series with normalize_timestamp_series
    content = content.replace('_to_naive_series(', 'normalize_timestamp_series(')
    
    print("✓ Replaced _to_naive calls with normalize_timestamp_* functions")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return 0


def remove_to_naive_functions():
    """Remove the deprecated _to_naive wrapper functions."""
    file_path = Path(r"c:\Users\bhadk\Documents\ACM V8 SQL\ACM\core\output_manager.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove _to_naive function
    old_func = """
# Safe datetime cast helpers - local time policy
def _to_naive(ts) -> Optional[pd.Timestamp]:
    \"\"\"
    DEPRECATED: Use utils.timestamp_utils.normalize_timestamp_scalar instead.
    Maintained for backward compatibility.
    \"\"\"
    return normalize_timestamp_scalar(ts)

def _to_naive_series(idx_or_series: Union[pd.Index, pd.Series]) -> pd.Series:
    \"\"\"
    DEPRECATED: Use utils.timestamp_utils.normalize_timestamp_series instead.
    Maintained for backward compatibility.
    \"\"\"
    return normalize_timestamp_series(idx_or_series)
"""
    
    if old_func in content:
        content = content.replace(old_func, "")
        print("✓ Removed deprecated _to_naive wrapper functions (15 lines)")
    else:
        print("! Deprecated functions not found (may have changed)")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return 0


def remove_json_methods():
    """Remove write_json and write_jsonl methods."""
    file_path = Path(r"c:\Users\bhadk\Documents\ACM V8 SQL\ACM\core\output_manager.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Original file: {len(lines)} lines")
    
    # Find write_json
    json_start = None
    json_end = None
    
    for i, line in enumerate(lines):
        if "def write_json(self, data:" in line:
            json_start = i
            print(f"Found write_json at line {i+1}")
        
        if json_start is not None and json_end is None:
            # Find write_jsonl (it follows write_json)
            if "def write_jsonl(self" in line:
                # Continue to find the end of write_jsonl
                continue
            # Find the next method or section marker
            if i > json_start + 5 and line.strip().startswith("# ===="):
                json_end = i - 1
                while json_end > json_start and not lines[json_end].strip():
                    json_end -= 1
                print(f"JSON methods end at line {json_end+1}")
                break
    
    if json_start is None:
        print("write_json not found")
        return 1
    
    # Remove the methods
    lines_removed = json_end - json_start + 1
    print(f"Removing lines {json_start+1} to {json_end+1} ({lines_removed} lines)")
    new_lines = lines[:json_start] + lines[json_end+1:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"New file: {len(new_lines)} lines")
    print("✓ Removed write_json and write_jsonl methods")
    
    return 0


def remove_json_files_field():
    """Remove json_files field from OutputBatch."""
    file_path = Path(r"c:\Users\bhadk\Documents\ACM V8 SQL\ACM\core\output_manager.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove json_files field
    if "json_files: Dict[Path, Dict[str, Any]] = field(default_factory=dict)" in content:
        content = content.replace(
            "    json_files: Dict[Path, Dict[str, Any]] = field(default_factory=dict)\n",
            ""
        )
        print("✓ Removed json_files field from OutputBatch")
    else:
        print("! json_files field not found")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return 0


def clean_flush_method():
    """Remove CSV/JSON references from flush() method."""
    file_path = Path(r"c:\Users\bhadk\Documents\ACM V8 SQL\ACM\core\output_manager.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove CSV and JSON flush code
    old_flush_body = """        with self._batch_lock:
            # OM-CSV-02: Skip CSV writes in SQL-only mode
            if self._current_batch.csv_files and not self.sql_only_mode:
                self.batch_write_csvs(self._current_batch.csv_files)
                self._current_batch.csv_files.clear()
            
            if self._current_batch.json_files and not self.sql_only_mode:
                for path, data in self._current_batch.json_files.items():
                    self.write_json(data, path)
                self._current_batch.json_files.clear()
            
            # Reset batch for next accumulation
            self._current_batch = OutputBatch()"""
    
    new_flush_body = """        with self._batch_lock:
            # Reset batch for next accumulation
            self._current_batch = OutputBatch()"""
    
    if old_flush_body in content:
        content = content.replace(old_flush_body, new_flush_body)
        print("✓ Cleaned flush() method (removed CSV/JSON code)")
    else:
        print("! Flush method body not found (may have changed)")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return 0


if __name__ == "__main__":
    import sys
    
    print("=== Phase 6: Remove Remaining File-Mode Bloat ===\n")
    
    print("Step 1: Remove RULConfig/AR1 import...")
    result1 = remove_rul_import()
    
    print("\nStep 2: Replace _to_naive calls...")
    result2 = replace_to_naive_calls()
    
    print("\nStep 3: Remove deprecated _to_naive functions...")
    result3 = remove_to_naive_functions()
    
    print("\nStep 4: Remove JSON writing methods...")
    result4 = remove_json_methods()
    
    print("\nStep 5: Remove json_files field...")
    result5 = remove_json_files_field()
    
    print("\nStep 6: Clean flush() method...")
    result6 = clean_flush_method()
    
    if all(r == 0 for r in [result1, result2, result3, result4, result5, result6]):
        print("\n✓ Phase 6 complete!")
        sys.exit(0)
    else:
        print("\n⚠ Phase 6 completed with warnings")
        sys.exit(0)
