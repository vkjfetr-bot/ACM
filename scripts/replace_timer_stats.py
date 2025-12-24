#!/usr/bin/env python
"""Replace finally block timer stats section with helper function call."""

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the timer stats section in finally block
    old_section = '''    finally:
        # === PERF: LOG TIMER STATS ===
        # Note: Timer class uses 'totals' attribute, not 'timings'
        if 'T' in locals() and hasattr(T, 'totals') and T.totals:
            try:
                # Log a summary of all timed sections (console-only, not to Loki)
                Console.section("Performance Summary")
                total_time = T.totals.get("total_run", 0.0)
                for section, duration in T.totals.items():
                    Console.status(f"{section}: {duration:.4f}s")
                    # Also emit to Loki for Grafana timer panel
                    pct = (duration / total_time * 100) if total_time > 0 else 0
                    log_timer(section=section, duration_s=duration, pct=pct, total_s=total_time)
                
                # Write detailed timer stats to SQL (if available)
                if SQL_MODE and sql_client and run_id:
                     try:
                         batch_num = int(cfg.get("runtime", {}).get("batch_num", 0)) if 'cfg' in locals() else 0
                         write_timer_stats(
                             sql_client=sql_client,
                             run_id=run_id,
                             equip_id=int(equip_id) if 'equip_id' in locals() else 0,
                             batch_num=batch_num,
                             timings=T.totals
                         )
                     except Exception as timer_err:
                         Console.warn(f"Failed to write timer stats from main: {timer_err}", component="PERF",
                                      equip=equip, run_id=run_id, error=str(timer_err)[:200])
            except Exception:
                pass'''
    
    new_section = '''    finally:
        # === PERF: LOG TIMER STATS ===
        if 'T' in locals() and hasattr(T, 'totals') and T.totals:
            _log_timer_stats(
                T=T,
                sql_client=sql_client if 'sql_client' in locals() else None,
                run_id=run_id if 'run_id' in locals() else None,
                equip_id=int(equip_id) if 'equip_id' in locals() else 0,
                cfg=cfg if 'cfg' in locals() else {},
                equip=equip if 'equip' in locals() else "UNKNOWN",
                SQL_MODE=SQL_MODE if 'SQL_MODE' in locals() else False,
            )'''
    
    if old_section not in content:
        print("ERROR: Could not find the timer stats section in finally block")
        print("Trying to find it manually...")
        
        # Try to find by line markers
        lines = content.split('\n')
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if "finally:" in line and "# === PERF: LOG TIMER STATS ===" not in line:
                # Check next line
                if i + 1 < len(lines) and "# === PERF: LOG TIMER STATS ===" in lines[i + 1]:
                    start_idx = i
            if start_idx and "if sql_log_sink:" in line:
                end_idx = i
                break
        
        if start_idx and end_idx:
            old_lines = lines[start_idx:end_idx]
            old_section = '\n'.join(old_lines)
            print(f"Found section from line {start_idx} to {end_idx}")
            content = '\n'.join(lines[:start_idx]) + '\n' + new_section + '\n\n' + '\n'.join(lines[end_idx:])
        else:
            print("Could not locate section")
            return
    else:
        content = content.replace(old_section, new_section)
    
    # Count lines saved
    old_lines = old_section.count('\n') + 1
    new_lines = new_section.count('\n') + 1
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"SUCCESS: Replaced timer stats section ({old_lines} lines -> {new_lines} lines)")
    print(f"Removed {old_lines - new_lines} lines from main()")

if __name__ == "__main__":
    main()
