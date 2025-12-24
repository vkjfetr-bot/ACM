#!/usr/bin/env python
"""Replace persist file artifacts section with helper function call."""

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the section by markers
    start_marker = "        if SQL_MODE or file_mode_enabled:\n            # ---------- FILE/SQL persistence for scores ----------"
    end_marker = "            # === COMPREHENSIVE ANALYTICS GENERATION ==="
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print(f"ERROR: Could not find markers. start={start_idx}, end={end_idx}")
        # Try alternative markers
        start_marker2 = "        if SQL_MODE or file_mode_enabled:"
        start_idx = content.find(start_marker2)
        print(f"Alternative start: {start_idx}")
        return
    
    old_section = content[start_idx:end_idx]
    old_lines = old_section.count('\n') + 1
    
    # New replacement code
    new_section = """        if SQL_MODE or file_mode_enabled:
            # Persist file artifacts (scores, episodes, schema, culprits, etc)
            _persist_file_artifacts(
                frame=frame,
                episodes=episodes,
                meta=meta,
                run_dir=run_dir,
                models_dir=models_dir,
                stable_models_dir=stable_models_dir if 'stable_models_dir' in locals() else None,
                output_manager=output_manager,
                model_cache_path=model_cache_path if 'model_cache_path' in locals() else run_dir / "detectors.pkl",
                regime_model=regime_model if 'regime_model' in locals() else None,
                regime_quality_ok=regime_quality_ok if 'regime_quality_ok' in locals() else False,
                cache_payload=cache_payload if 'cache_payload' in locals() else None,
                equip=equip,
                run_id=run_id,
                cfg=cfg,
                T=T,
                SQL_MODE=SQL_MODE,
                dual_mode=dual_mode,
            )

"""
    
    new_lines = new_section.count('\n') + 1
    
    # Replace the section
    new_content = content[:start_idx] + new_section + "            " + content[end_idx:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"SUCCESS: Replaced persist artifacts section ({old_lines} lines -> {new_lines} lines)")
    print(f"Removed {old_lines - new_lines} lines from main()")

if __name__ == "__main__":
    main()
