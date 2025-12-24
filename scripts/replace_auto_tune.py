#!/usr/bin/env python
"""Replace autonomous parameter tuning section with helper function call."""

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the section by markers
    start_marker = "        # ===== Autonomous Parameter Tuning: Update config based on quality ====="
    end_marker = "        if reuse_models:"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print(f"ERROR: Could not find markers. start={start_idx}, end={end_idx}")
        return
    
    old_section = content[start_idx:end_idx]
    old_lines = old_section.count('\n') + 1
    
    # New replacement code
    new_section = """        # ===== Autonomous Parameter Tuning: Update config based on quality =====
        tuning_actions = _run_autonomous_tuning(
            frame=frame,
            episodes=episodes,
            score_out=score_out,
            regime_quality_ok=regime_quality_ok,
            cached_manifest=cached_manifest if 'cached_manifest' in locals() else None,
            cfg=cfg,
            sql_client=sql_client if 'sql_client' in locals() else None,
            equip_id=int(equip_id),
            run_id=run_id,
            equip=equip,
            refit_flag_path=refit_flag_path if 'refit_flag_path' in locals() else run_dir / ".refit",
            SQL_MODE=SQL_MODE,
        )

"""
    
    new_lines = new_section.count('\n') + 1
    
    # Replace the section
    new_content = content[:start_idx] + new_section + content[end_idx:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"SUCCESS: Replaced autonomous tuning section ({old_lines} lines -> {new_lines} lines)")
    print(f"Removed {old_lines - new_lines} lines from main()")

if __name__ == "__main__":
    main()
