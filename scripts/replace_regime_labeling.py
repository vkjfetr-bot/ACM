#!/usr/bin/env python3
"""Replace the regime labeling section with _label_regimes call."""

with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Old pattern - the entire regime labeling section
old_start = '''        # ===== 4) Regimes (Run before calibration to enable regime-aware thresholds) =====
        train_regime_labels = None
        score_regime_labels = None
        regime_model_was_trained = False
        
        with T.section("regimes.label"):'''

old_end = '''        score_out = regime_out
        regime_quality_ok = bool(regime_out.get("regime_quality_ok", True))'''

# Find the section
start_idx = content.find(old_start)
end_idx = content.find(old_end)

if start_idx == -1:
    print("ERROR: Could not find start pattern")
    exit(1)
if end_idx == -1:
    print("ERROR: Could not find end pattern")
    exit(1)

# Include the end pattern
end_idx = end_idx + len(old_end)

old_section = content[start_idx:end_idx]

# New replacement
new_replacement = '''        # ===== 4) Regimes (Run before calibration to enable regime-aware thresholds) =====
        regime_label_result = _label_regimes(
            score=score,
            frame=frame,
            train=train,
            train_numeric=train_numeric,
            regime_model=regime_model,
            regime_state=regime_state,
            regime_state_version=regime_state_version,
            regime_basis=regime_basis,
            equip=equip,
            equip_id=equip_id,
            cfg=cfg,
            T=T,
            art_root=art_root,
            sql_client=sql_client,
            SQL_MODE=SQL_MODE,
            dual_mode=dual_mode,
        )
        frame = regime_label_result.frame
        regime_model = regime_label_result.regime_model
        train_regime_labels = regime_label_result.train_regime_labels
        score_regime_labels = regime_label_result.score_regime_labels
        regime_quality_ok = regime_label_result.regime_quality_ok
        regime_model_was_trained = regime_label_result.regime_model_was_trained
        regime_out = regime_label_result.regime_out
        score_out = regime_out'''

content = content[:start_idx] + new_replacement + content[end_idx:]

with open("core/acm_main.py", "w", encoding="utf-8") as f:
    f.write(content)

# Calculate savings
old_chars = len(old_section)
new_chars = len(new_replacement)
old_lines = old_section.count('\n')
new_lines = new_replacement.count('\n')
print(f"SUCCESS: Replaced regime labeling section")
print(f"Original: {old_chars} chars, {old_lines} lines")
print(f"New: {new_chars} chars, {new_lines} lines")
print(f"Removed: {old_chars - new_chars} chars, {old_lines - new_lines} lines")
