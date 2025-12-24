#!/usr/bin/env python
"""Add _log_timer_stats helper function to acm_main.py"""
import re

HELPER_CODE = '''

def _log_timer_stats(
    T: "Timer",
    sql_client: Optional["SQLClient"],
    run_id: Optional[str],
    equip_id: int,
    cfg: Dict[str, Any],
    equip: str,
    SQL_MODE: bool,
) -> None:
    """
    Log timer statistics to console and optionally write to SQL.
    
    Console output goes to section headers (not Loki), SQL gets detailed records.
    
    Args:
        T: Timer instance with totals attribute
        sql_client: SQL client for writing stats
        run_id: Current run ID
        equip_id: Equipment ID
        cfg: Configuration dictionary
        equip: Equipment name for logging
        SQL_MODE: Whether SQL mode is enabled
    """
    from core.observability import Console, log_timer
    from core.run_metadata_writer import write_timer_stats
    
    if not hasattr(T, 'totals') or not T.totals:
        return
    
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
                batch_num = int(cfg.get("runtime", {}).get("batch_num", 0))
                write_timer_stats(
                    sql_client=sql_client,
                    run_id=run_id,
                    equip_id=equip_id,
                    batch_num=batch_num,
                    timings=T.totals
                )
            except Exception as timer_err:
                Console.warn(
                    f"Failed to write timer stats from main: {timer_err}",
                    component="PERF",
                    equip=equip,
                    run_id=run_id,
                    error=str(timer_err)[:200]
                )
    except Exception:
        pass  # Timer logging should never break the run

'''

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the insertion point - after _write_run_stats_and_culprits function
    pattern = r"(def _write_run_stats_and_culprits\([\s\S]*?^    Console\.info\(f\"Wrote run stats.*?component=\"RUN_STATS\"\)\n)"
    match = re.search(pattern, content, re.MULTILINE)
    
    if not match:
        # Try alternative - find end of the function by looking for next def or end of indented block
        pattern2 = r"(def _write_run_stats_and_culprits\([\s\S]*?(?=\ndef |\nclass |\n# ===))"
        match = re.search(pattern2, content)
    
    if not match:
        print("ERROR: Could not find _write_run_stats_and_culprits function")
        return
    
    insert_pos = match.end()
    
    # Check if helper already exists
    if "_log_timer_stats" in content:
        print("WARNING: _log_timer_stats already exists")
        return
    
    # Insert the helper
    new_content = content[:insert_pos] + HELPER_CODE + content[insert_pos:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("SUCCESS: Added _log_timer_stats helper function")

if __name__ == "__main__":
    main()
