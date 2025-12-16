"""
Phase 5: Remove _generate_schema_descriptor helper method.
Part of refactor/output-manager-bloat-removal.
"""
from pathlib import Path


def remove_schema_descriptor():
    """Remove _generate_schema_descriptor method and its usage."""
    file_path = Path(r"c:\Users\bhadk\Documents\ACM V8 SQL\ACM\core\output_manager.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Original file: {len(lines)} lines")
    
    # Find the method
    method_start = None
    method_end = None
    
    for i, line in enumerate(lines):
        if "def _generate_schema_descriptor(self" in line:
            method_start = i
            print(f"Found _generate_schema_descriptor at line {i+1}")
        
        if method_start is not None and method_end is None:
            # Find the next method or end of class
            if i > method_start and (line.strip().startswith("def ") and "_generate_schema_descriptor" not in line):
                method_end = i - 1
                # Backtrack to remove empty lines
                while method_end > method_start and not lines[method_end].strip():
                    method_end -= 1
                print(f"_generate_schema_descriptor ends at line {method_end+1}")
                break
    
    if method_start is None:
        print("_generate_schema_descriptor not found")
        return 1
    
    # Remove the method
    lines_removed = method_end - method_start + 1
    print(f"Removing lines {method_start+1} to {method_end+1} ({lines_removed} lines)")
    new_lines = lines[:method_start] + lines[method_end+1:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"New file: {len(new_lines)} lines")
    print("✓ _generate_schema_descriptor removed successfully")
    
    # Now remove the usage (the try-except block that calls it)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and remove the schema descriptor generation block
    old_block = """                # OUT-20: Generate schema descriptor JSON after all tables are written
                if not self.sql_only_mode:
                    try:
                        schema_descriptor = self._generate_schema_descriptor(tables_dir)
                        schema_path = tables_dir / "schema_descriptor.json"
                        with open(schema_path, 'w') as f:
                            import json
                            json.dump(schema_descriptor, f, indent=2)
                        Console.info(f"[ANALYTICS] Generated schema descriptor: {schema_path}")
                    except Exception as e:
                        Console.warn(f"[ANALYTICS] Failed to generate schema descriptor: {e}")
                    """
    
    if old_block in content:
        content = content.replace(old_block + "\n", "")
        print("✓ Removed schema descriptor generation call")
    else:
        print("! Schema descriptor call not found (may have changed)")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return 0


if __name__ == "__main__":
    import sys
    
    print("=== Phase 5: Remove Schema Descriptor Helper ===\n")
    result = remove_schema_descriptor()
    
    if result == 0:
        print("\n✓ Phase 5 complete!")
        sys.exit(0)
    else:
        print("\n✗ Phase 5 failed")
        sys.exit(1)
