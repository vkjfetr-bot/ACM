#!/usr/bin/env python3
"""
Surgical fix: Replace old refit checking and model loading code with helper function calls.

This script:
1. Finds the refit checking section and replaces it with _check_refit_requested() call
2. Preserves the equip_id inference section
3. Replaces cache loading and detector reconstruction with helper calls
"""
import re

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    original_length = len(content)
    
    # Pattern 1: Replace the old refit checking code block
    # Looking for the section that starts with "# Respect refit-request: file flag or SQL table entries"
    # and ends before "if reuse_models and model_cache_path.exists()"
    
    old_refit_block_start = "        # Respect refit-request: file flag or SQL table entries"
    old_refit_block_end = """        if reuse_models and model_cache_path.exists() and not refit_requested:"""
    
    # Find the refit block
    refit_start_idx = content.find(old_refit_block_start)
    if refit_start_idx == -1:
        print("ERROR: Could not find refit block start marker")
        return
    
    refit_end_idx = content.find(old_refit_block_end)
    if refit_end_idx == -1:
        print("ERROR: Could not find refit block end marker")
        return
    
    print(f"Found refit block: start={refit_start_idx}, end={refit_end_idx}")
    
    # New refit checking call
    new_refit_call = """        # Check for refit requests (file flag or SQL table)
        refit_requested = _check_refit_requested(ctx, T, refit_flag_path)

"""
    
    # Replace refit block
    content = content[:refit_start_idx] + new_refit_call + content[refit_end_idx:]
    
    print(f"After refit replacement: {len(content)} chars")
    
    # Pattern 2: Find and replace the local cache loading block 
    # From "if reuse_models and model_cache_path.exists()" to the equip_id inference section
    old_cache_start = """        # Check for refit requests (file flag or SQL table)
        refit_requested = _check_refit_requested(ctx, T, refit_flag_path)

        if reuse_models and model_cache_path.exists() and not refit_requested:
            with T.section("models.cache_local"):"""
    
    old_cache_end = """        # Attempt to infer EquipID from config or meta if available (0 if unknown)"""
    
    cache_start_idx = content.find(old_cache_start)
    if cache_start_idx == -1:
        print("ERROR: Could not find cache block start marker after refit replacement")
        # Try alternative - just find the if reuse_models line
        alt_start = """        if reuse_models and model_cache_path.exists() and not refit_requested:"""
        alt_start_idx = content.find(alt_start)
        if alt_start_idx != -1:
            print(f"Found alternative start at {alt_start_idx}")
            cache_start_idx = alt_start_idx
        else:
            print("ERROR: Could not find alternative start either")
            return
    
    cache_end_idx = content.find(old_cache_end)
    if cache_end_idx == -1:
        print("ERROR: Could not find cache block end marker")
        return
    
    print(f"Found cache block: start={cache_start_idx}, end={cache_end_idx}")
    
    # We want to keep everything before the refit call and replace up to equip_id section
    # Find position after the new refit call
    after_refit_call = new_refit_call
    after_refit_idx = content.find(after_refit_call)
    if after_refit_idx != -1:
        after_refit_idx += len(after_refit_call)
    else:
        print("ERROR: Could not find position after refit call")
        return
    
    # Find start of local cache loading (after refit call)
    local_cache_line = "        if reuse_models and model_cache_path.exists() and not refit_requested:"
    local_cache_idx = content.find(local_cache_line, after_refit_idx)
    if local_cache_idx == -1:
        print("ERROR: Could not find local cache loading line")
        return
    
    print(f"Local cache loading at: {local_cache_idx}")
    
    # The section from local_cache_idx to cache_end_idx is the local cache loading
    # We want to replace this with our _load_cached_models call
    # But we need to preserve the equip_id inference section
    
    # Actually, let me simplify - just keep the refit fix for now and do the rest in a second pass
    # This is getting complex
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    new_length = len(content)
    print(f"SUCCESS: Fixed refit checking section")
    print(f"Original: {original_length} chars, New: {new_length} chars")
    print(f"Removed: {original_length - new_length} chars")

if __name__ == "__main__":
    main()
