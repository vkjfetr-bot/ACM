#!/usr/bin/env python
"""Replace comprehensive analytics section with helper function call."""

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the section by markers
    start_marker = "            # === COMPREHENSIVE ANALYTICS GENERATION ==="
    end_marker = "            # Legacy enhanced_forecasting removed - ForecastEngine (v10) handles all forecasting"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print(f"ERROR: Could not find markers. start={start_idx}, end={end_idx}")
        return
    
    # Include the end marker line
    end_idx = content.find('\n', end_idx) + 1
    
    old_section = content[start_idx:end_idx]
    old_lines = old_section.count('\n') + 1
    
    # New replacement code
    new_section = """            # === COMPREHENSIVE ANALYTICS GENERATION ===
            try:
                analytics_result = _run_comprehensive_analytics(
                    frame=frame,
                    episodes=episodes,
                    run_dir=run_dir,
                    output_manager=output_manager,
                    regime_model=regime_model if 'regime_model' in locals() else None,
                    fusion_weights_used=fusion_weights_used if 'fusion_weights_used' in locals() else None,
                    sensor_context=sensor_context if 'sensor_context' in locals() else None,
                    equip_id=equip_id,
                    run_id=run_id,
                    equip=equip,
                    cfg=cfg,
                    T=T,
                    SQL_MODE=SQL_MODE,
                )
            except Exception as e:
                Console.warn(f"Output generation failed: {e}", component="OUTPUTS",
                             equip=equip, run_id=run_id, error_type=type(e).__name__, error=str(e)[:500])

            # File mode path exits here (finally still runs, no SQL finalize executed).
            run_completion_time = datetime.now()
            _maybe_write_run_meta_json(locals())

"""
    
    new_lines = new_section.count('\n') + 1
    
    # Replace the section
    new_content = content[:start_idx] + new_section + content[end_idx:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"SUCCESS: Replaced comprehensive analytics section ({old_lines} lines -> {new_lines} lines)")
    print(f"Removed {old_lines - new_lines} lines from main()")

if __name__ == "__main__":
    main()
