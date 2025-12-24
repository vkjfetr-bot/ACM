#!/usr/bin/env python
"""Replace run metadata writing section in finally block with helper function call."""

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the run metadata section by markers
    start_marker = "        # === WRITE RUN METADATA TO ACM_RUNS (ALWAYS, EVEN FOR NOOP) ==="
    end_marker = "        # Always finalize and close SQL in SQL mode"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print(f"ERROR: Could not find markers. start={start_idx}, end={end_idx}")
        return
    
    old_section = content[start_idx:end_idx]
    old_lines = old_section.count('\n') + 1
    
    # New replacement code
    new_section = """        # === WRITE RUN METADATA TO ACM_RUNS (ALWAYS, EVEN FOR NOOP) ===
        run_metadata_result = None
        if SQL_MODE and sql_client and run_id:  # type: ignore
            run_ctx = FinalRunContext(
                frame=frame if 'frame' in locals() and isinstance(frame, pd.DataFrame) else None,
                train=train if 'train' in locals() and isinstance(train, pd.DataFrame) else None,
                score=score if 'score' in locals() and isinstance(score, pd.DataFrame) else None,
                episodes=episodes if 'episodes' in locals() and isinstance(episodes, pd.DataFrame) else None,
                meta=meta if 'meta' in locals() else None,
                score_regime_labels=score_regime_labels if 'score_regime_labels' in locals() else None,
                tables_dir=tables_dir if 'tables_dir' in locals() else None,
                refit_flag_path=refit_flag_path if 'refit_flag_path' in locals() else None,
                quality_ok=quality_ok if 'quality_ok' in locals() else False,
                use_per_regime=use_per_regime if 'use_per_regime' in locals() else False,
            )
            run_metadata_result = _write_final_run_metadata(
                ctx=run_ctx,
                sql_client=sql_client,
                run_id=run_id,
                equip_id=int(equip_id) if 'equip_id' in locals() else 0,
                equip=equip if 'equip' in locals() else "UNKNOWN",
                run_start_time=run_start_time,
                config_signature=config_signature if 'config_signature' in locals() else "UNKNOWN",
                rows_read=rows_read if 'rows_read' in locals() else 0,
                outcome=outcome,
                err_json=err_json if 'err_json' in locals() else None,
                SQL_MODE=SQL_MODE,
            )
        
"""
    
    new_lines = new_section.count('\n') + 1
    
    # Replace the section
    new_content = content[:start_idx] + new_section + content[end_idx:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"SUCCESS: Replaced run metadata section ({old_lines} lines -> {new_lines} lines)")
    print(f"Removed {old_lines - new_lines} lines from main()")

if __name__ == "__main__":
    main()
