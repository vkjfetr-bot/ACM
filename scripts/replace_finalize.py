#!/usr/bin/env python
"""Replace SQL finalize and OTEL metrics section in finally block with helper function call."""

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the SQL finalize section by markers
    start_marker = "        # Always finalize and close SQL in SQL mode"
    end_marker = "        # Close OpenTelemetry root span"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print(f"ERROR: Could not find markers. start={start_idx}, end={end_idx}")
        return
    
    old_section = content[start_idx:end_idx]
    old_lines = old_section.count('\n') + 1
    
    # New replacement code
    new_section = """        # Always finalize and close SQL in SQL mode
        if SQL_MODE and sql_client and run_id:  # type: ignore
            _finalize_sql_and_record_metrics(
                sql_client=sql_client,
                run_id=run_id,
                outcome=outcome,
                rows_read=rows_read if 'rows_read' in locals() else 0,
                rows_written=rows_written if 'rows_written' in locals() else 0,
                err_json=err_json if 'err_json' in locals() else None,
                run_start_time=run_start_time if 'run_start_time' in locals() else None,
                run_metadata_result=run_metadata_result,
                equip=equip if 'equip' in locals() else "UNKNOWN",
                output_manager=output_manager if 'output_manager' in locals() else None,
                _OBSERVABILITY_AVAILABLE=_OBSERVABILITY_AVAILABLE,
            )
        
"""
    
    new_lines = new_section.count('\n') + 1
    
    # Replace the section
    new_content = content[:start_idx] + new_section + content[end_idx:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"SUCCESS: Replaced SQL finalize section ({old_lines} lines -> {new_lines} lines)")
    print(f"Removed {old_lines - new_lines} lines from main()")

if __name__ == "__main__":
    main()
