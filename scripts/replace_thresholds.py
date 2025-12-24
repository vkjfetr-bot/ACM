#!/usr/bin/env python
"""Replace adaptive thresholds section with helper function call."""

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the start and end of the section
    start_marker = """            # ===== CONTINUOUS LEARNING: Calculate adaptive thresholds on accumulated data =====
            # This runs AFTER fusion, using combined train+score data (or just train if not continuous learning)
            # Frequency control: Only recalculate if update interval reached or first run
            with T.section("thresholds.adaptive"):"""
    
    end_marker = """        Console.info("Starting regime health labeling and transient detection...", component="REGIME")"""
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1:
        print("ERROR: Could not find start marker")
        return
    if end_idx == -1:
        print("ERROR: Could not find end marker")
        return
    
    old_section = content[start_idx:end_idx]
    old_lines = old_section.count('\n') + 1
    
    new_section = """            # ===== CONTINUOUS LEARNING: Calculate adaptive thresholds on accumulated data =====
            with T.section("thresholds.adaptive"):
                _update_adaptive_thresholds(
                    train=train,
                    score=score,
                    train_frame=train_frame,
                    frame=frame,
                    present=present,
                    weights=weights,
                    train_regime_labels=train_regime_labels,
                    score_regime_labels=score_regime_labels,
                    regime_quality_ok=regime_quality_ok,
                    cfg=cfg,
                    equip_id=int(equip_id),
                    output_manager=output_manager,
                    CONTINUOUS_LEARNING=CONTINUOUS_LEARNING,
                    coldstart_complete=coldstart_complete,
                    threshold_update_interval=threshold_update_interval,
                    equip=equip,
                    T=T,
                )

"""
    
    new_lines = new_section.count('\n') + 1
    
    # Replace
    new_content = content[:start_idx] + new_section + content[end_idx:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"SUCCESS: Replaced adaptive thresholds section ({old_lines} lines -> {new_lines} lines)")
    print(f"Removed {old_lines - new_lines} lines from main()")

if __name__ == "__main__":
    main()
