#!/usr/bin/env python3
"""Replace the calibration section with _calibrate_scores call."""

with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Old pattern - find start and end
old_start = '''        # ===== 5) Calibrate -> pca_spe_z, pca_t2_z, ar1_z, iforest_z, gmm_z, omr_z =====
        # CRITICAL FIX: Fit calibrators on TRAIN data, transform SCORE data
        with T.section("calibrate"):'''

old_end = '''                Console.info(f"Wrote thresholds table with {len(threshold_rows)} rows -> acm_thresholds", component="CAL")

        # ===== 6) Fusion + episodes ====='''

# Find the section
start_idx = content.find(old_start)
end_idx = content.find(old_end)

if start_idx == -1:
    print("ERROR: Could not find start pattern")
    exit(1)
if end_idx == -1:
    print("ERROR: Could not find end pattern")
    exit(1)

# Calculate positions - we want to keep the "===== 6)" marker
old_section_end = old_end.find("\n\n        # ===== 6)")
if old_section_end == -1:
    old_section_end = len(old_end)
else:
    old_section_end = end_idx + old_section_end

old_section = content[start_idx:old_section_end]

# New replacement
new_replacement = '''        # ===== 5) Calibrate -> pca_spe_z, pca_t2_z, ar1_z, iforest_z, gmm_z, omr_z =====
        # CRITICAL FIX: Fit calibrators on TRAIN data, transform SCORE data
        use_per_regime = (cfg.get("fusion", {}) or {}).get("per_regime", False)
        cal_result = _calibrate_scores(
            frame=frame,
            train=train,
            train_regime_labels=train_regime_labels,
            score_regime_labels=score_regime_labels,
            regime_quality_ok=regime_quality_ok,
            ar1_detector=ar1_detector,
            pca_detector=pca_detector,
            iforest_detector=iforest_detector,
            gmm_detector=gmm_detector,
            omr_detector=omr_detector,
            omr_enabled=omr_enabled,
            pca_train_spe=pca_train_spe,
            pca_train_t2=pca_train_t2,
            cfg=cfg,
            equip=equip,
            output_manager=output_manager,
            T=T,
        )
        frame = cal_result.frame
        train_frame = cal_result.train_frame
        calibrators = cal_result.calibrators
        spe_p95_train = cal_result.spe_p95_train
        t2_p95_train = cal_result.t2_p95_train
        pca_train_spe_z = cal_result.pca_train_spe_z
        pca_train_t2_z = cal_result.pca_train_t2_z
        quality_ok = bool(use_per_regime and regime_quality_ok and train_regime_labels is not None and score_regime_labels is not None)
'''

content = content[:start_idx] + new_replacement + content[old_section_end:]

with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(content)

# Calculate savings
old_chars = len(old_section)
new_chars = len(new_replacement)
old_lines = old_section.count('\n')
new_lines = new_replacement.count('\n')
print(f"SUCCESS: Replaced calibration section")
print(f"Original: {old_chars} chars, {old_lines} lines")
print(f"New: {new_chars} chars, {new_lines} lines")
print(f"Removed: {old_chars - new_chars} chars, {old_lines - new_lines} lines")
