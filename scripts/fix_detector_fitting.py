#!/usr/bin/env python3
"""
Phase 2: Replace old detector fitting code with helper function calls.

This script finds and replaces the detector fitting section in main() with a call
to the new _fit_detectors() helper function.
"""
import re

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    original_length = len(content)
    lines = content.split('\n')
    
    # Find the detector fitting section
    # Starts with: "# Fit models if not loaded from cache"
    # Ends with: "Console.info(f"All detectors fitted..."
    
    fit_start_marker = "        # Fit models if not loaded from cache (MHAL removed v9.1.0)"
    fit_end_marker = '                Console.info(f"All detectors fitted in {time.perf_counter()-fit_start_time:.2f}s", component="MODEL")'
    
    # Find start line
    start_line = None
    for i, line in enumerate(lines):
        if fit_start_marker in line:
            start_line = i
            break
    
    if start_line is None:
        print("ERROR: Could not find detector fitting start marker")
        return
    
    print(f"Found fitting start at line {start_line + 1}")
    
    # Find end line (after the start)
    end_line = None
    for i in range(start_line, len(lines)):
        if fit_end_marker in lines[i]:
            end_line = i
            break
    
    if end_line is None:
        print("ERROR: Could not find detector fitting end marker")
        return
    
    print(f"Found fitting end at line {end_line + 1}")
    
    # The section we want to replace is from start_line to end_line (inclusive)
    # We also need to keep the validation section that follows
    
    # New code to insert
    new_fit_code = '''        # Create detector state for fitting
        detector_state = DetectorState(
            ar1_detector=ar1_detector,
            pca_detector=pca_detector,
            iforest_detector=iforest_detector,
            gmm_detector=gmm_detector,
            omr_detector=omr_detector,
            pca_train_spe=pca_train_spe,
            pca_train_t2=pca_train_t2,
            ar1_enabled=ar1_enabled,
            pca_enabled=pca_enabled,
            iforest_enabled=iforest_enabled,
            gmm_enabled=gmm_enabled,
            omr_enabled=omr_enabled,
        )
        
        # Fit detectors that are not already loaded from cache
        fit_result = _fit_detectors(
            train=train,
            ctx=ctx,
            T=T,
            detector_state=detector_state,
            output_manager=output_manager,
            tables_dir=tables_dir,
            run_dir=run_dir,
        )
        ar1_detector = fit_result.ar1_detector
        pca_detector = fit_result.pca_detector
        iforest_detector = fit_result.iforest_detector
        gmm_detector = fit_result.gmm_detector
        omr_detector = fit_result.omr_detector
        pca_train_spe = fit_result.pca_train_spe
        pca_train_t2 = fit_result.pca_train_t2'''
    
    # Construct new content
    new_lines = lines[:start_line] + new_fit_code.split('\n') + lines[end_line + 1:]
    new_content = '\n'.join(new_lines)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    new_length = len(new_content)
    lines_removed = end_line - start_line + 1
    lines_added = len(new_fit_code.split('\n'))
    
    print(f"SUCCESS: Replaced detector fitting section")
    print(f"Original: {original_length} chars, New: {new_length} chars")
    print(f"Lines removed: {lines_removed}, Lines added: {lines_added}")
    print(f"Net change: {lines_added - lines_removed} lines")

if __name__ == "__main__":
    main()
