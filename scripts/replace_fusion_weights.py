#!/usr/bin/env python3
"""Replace the fusion weight preparation section with helper calls."""

with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Old pattern - from start of fusion to just before auto-tune
old_start = '''        # ===== 6) Fusion + episodes =====
        with T.section("fusion"): # type: ignore
            # Active detectors: PCA-SPE, PCA-TÂ², AR1, IForest, GMM, OMR
            default_w = {
                "pca_spe_z": 0.30,
                "pca_t2_z": 0.20,
                "ar1_z": 0.20,
                "iforest_z": 0.15,
                "gmm_z": 0.05,
                "omr_z": 0.10,
            }
            weights = (cfg or {}).get("fusion", {}).get("weights", default_w)
            fusion_weights_used = dict(weights)
            avail = set(frame.columns)
            
            # CRITICAL FIX #10: Validate weights keys against available detectors BEFORE fusion
            # This prevents KeyError crashes from misconfigured weights
            invalid_keys = [k for k in weights.keys() if not k.endswith('_z')]
            if invalid_keys:
                raise ValueError(f"[FUSE] Invalid detector keys in fusion.weights: {invalid_keys}. "
                               f"All keys must end with '_z' (e.g., 'ar1_z', 'pca_spe_z')")
            
            missing = [k for k in weights.keys() if k not in avail]
            # Filter out permanently unimplemented detectors from warnings
            missing_to_warn = [k for k in missing if k not in {'mhal_z', 'river_hst_z'}]
            if missing_to_warn:
                Console.warn(f"Ignoring missing streams: {missing_to_warn}; available={sorted([c for c in avail if c.endswith('_z')])}", component="FUSE",
                             equip=equip, missing_streams=missing_to_warn)
            present = {k: frame[k].to_numpy(copy=False) for k in weights.keys() if k in avail}
            if not present:
                raise RuntimeError("[FUSE] No valid input streams for fusion. Check your fusion.weights keys or ensure detectors are enabled.")
            
            # FUSE-11: Dynamic weight normalization for present streams only
            # Remove weights for missing detectors and renormalize
            if missing:
                # Build new weights dict with only present detectors
                present_weights = {k: weights[k] for k in present.keys()}
                total_present = sum(present_weights.values())
                
                if total_present > 0:
                    # Renormalize to sum to 1.0
                    weights = {k: v / total_present for k, v in present_weights.items()}
                    Console.info(f"Dynamic normalization: {len(missing)} detector(s) absent, renormalized {len(weights)} weights", component="FUSE")
                else:
                    # All weights were zero - use equal weighting
                    equal_weight = 1.0 / len(present)
                    weights = {k: equal_weight for k in present.keys()}
                    Console.warn(f"All configured weights were 0.0, using equal weighting ({equal_weight:.3f} each)", component="FUSE",
                                 equip=equip, equal_weight=round(equal_weight, 3))
                
                fusion_weights_used = dict(weights)
            
            # FUSE-10: Load previous weights for warm-start (persistent learning)
            previous_weights = None
            if run_dir:
                try:
                    # Look for previous weight_tuning.json in equipment's artifact folder
                    equipment_artifact_root = run_dir.parent
                    prev_runs = sorted([d for d in equipment_artifact_root.glob("run_*") if d.is_dir()], reverse=True)
                    
                    # Skip current run, look for most recent previous run
                    for prev_run_dir in prev_runs[1:] if len(prev_runs) > 1 else []:
                        prev_tune_path = prev_run_dir / "tables" / "weight_tuning.json"
                        if prev_tune_path.exists():
                            with open(prev_tune_path, 'r') as f:
                                prev_data = json.load(f)
                                previous_weights = prev_data.get("tuned_weights", {})
                                Console.info(f"Loaded previous weights from {prev_tune_path.name}", component="TUNE")
                                break
                except Exception as load_e:
                    Console.warn(f"Failed to load previous weights: {load_e}", component="TUNE",
                                 equip=equip, error=str(load_e)[:200])
            
            # Use previous weights as starting point if available
            if previous_weights:
                # Warm-start: blend config weights with learned weights (favor learned)
                warm_start_lr = float((cfg or {}).get("fusion", {}).get("auto_tune", {}).get("warm_start_lr", 0.7))
                weights = {k: (1 - warm_start_lr) * weights.get(k, 0.0) + warm_start_lr * previous_weights.get(k, weights.get(k, 0.0)) 
                          for k in weights.keys()}
                # Normalize
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in weights.items()}
                Console.info(f"Warm-start blending: warm_start_lr={warm_start_lr:.2f}", component="TUNE")
            
            # DET-06: Auto-tune weights before fusion'''

old_end = '''            # DET-06: Auto-tune weights before fusion'''

# Find the section
start_idx = content.find(old_start)
if start_idx == -1:
    print("ERROR: Could not find start pattern")
    print("Looking for pattern starting with: '===== 6) Fusion + episodes'")
    if "===== 6) Fusion" in content:
        print("Found '===== 6) Fusion' but full pattern didn't match")
    exit(1)

end_idx = content.find(old_end)
if end_idx == -1:
    print("ERROR: Could not find end pattern")
    exit(1)

# Include end pattern in replacement
end_idx = end_idx + len(old_end)

old_section = content[start_idx:end_idx]

# New replacement
new_replacement = '''        # ===== 6) Fusion + episodes =====
        with T.section("fusion"): # type: ignore
            # Prepare fusion weights (validation, normalization, warm-start)
            present, weights, previous_weights = _prepare_fusion_weights(
                frame=frame,
                cfg=cfg,
                equip=equip,
                run_dir=run_dir,
            )
            fusion_weights_used = dict(weights)
            
            # DET-06: Auto-tune weights before fusion'''

content = content[:start_idx] + new_replacement + content[end_idx:]

with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(content)

# Calculate savings
old_chars = len(old_section)
new_chars = len(new_replacement)
old_lines = old_section.count('\n')
new_lines = new_replacement.count('\n')
print(f"SUCCESS: Replaced fusion weight preparation section")
print(f"Original: {old_chars} chars, {old_lines} lines")
print(f"New: {new_chars} chars, {new_lines} lines")
print(f"Removed: {old_chars - new_chars} chars, {old_lines - new_lines} lines")
