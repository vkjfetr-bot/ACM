#!/usr/bin/env python
"""Add _finalize_sql_and_record_metrics helper function to acm_main.py"""
import re

HELPER_CODE = '''

def _finalize_sql_and_record_metrics(
    sql_client: "SQLClient",
    run_id: str,
    outcome: str,
    rows_read: int,
    rows_written: int,
    err_json: Optional[str],
    run_start_time: datetime,
    run_metadata_result: Optional[Dict[str, Any]],
    equip: str,
    output_manager: Optional["OutputManager"],
    _OBSERVABILITY_AVAILABLE: bool,
) -> None:
    """
    Finalize SQL run and record OTEL metrics.
    
    Calls _sql_finalize_run to update ACM_Runs, then records OTEL metrics
    for Prometheus. Closes OutputManager and SQL client on completion.
    
    Args:
        sql_client: SQL client for finalization
        run_id: Current run ID
        outcome: Run outcome (OK, NOOP, FAIL)
        rows_read: Rows read during run
        rows_written: Rows written during run
        err_json: Error JSON if failed
        run_start_time: When run started
        run_metadata_result: Metadata dict from _write_final_run_metadata
        equip: Equipment name
        output_manager: Output manager to close
        _OBSERVABILITY_AVAILABLE: Whether observability is available
    """
    from core.observability import Console
    from core.metrics import (
        record_run,
        record_batch_processed,
        record_health_score,
        record_error,
    )
    
    try:
        _sql_finalize_run(
            sql_client,
            run_id=run_id,
            outcome=outcome,
            rows_read=rows_read,
            rows_written=rows_written,
            err_json=err_json
        )
        Console.info(
            f"Finalized RunID={run_id} outcome={outcome} rows_in={rows_read} rows_out={rows_written}",
            component="RUN"
        )
        
        # Record OTEL metrics for Prometheus
        if _OBSERVABILITY_AVAILABLE and run_start_time:
            try:
                # Get completion time from run_metadata_result or use now
                run_completion_time = (
                    run_metadata_result.get("_completion_time")
                    if run_metadata_result else None
                ) or datetime.now()
                
                duration_seconds = (run_completion_time - run_start_time).total_seconds()
                
                # Record run outcome (counter + histogram)
                record_run(equip, outcome or "OK", duration_seconds)
                
                # Record batch processing
                record_batch_processed(
                    equipment=equip,
                    duration_seconds=duration_seconds,
                    rows=rows_read,
                    status=outcome.lower() if outcome else "unknown"
                )
                
                # Record health score if available
                if run_metadata_result and run_metadata_result.get("avg_health_index") is not None:
                    record_health_score(equip, float(run_metadata_result.get("avg_health_index", 0)))
                
                # Record error if run failed
                if outcome == "FAIL":
                    error_type = type(err_json).__name__ if err_json else "unknown"
                    error_msg = str(err_json) if err_json else "Run failed"
                    record_error(equip, error_msg, error_type)
                    
            except Exception as metric_err:
                Console.debug(f"Metrics recording failed (non-fatal): {metric_err}", component="OTEL")
                
    except Exception as fe:
        Console.error(
            f"Finalize failed (finally): {fe}",
            component="RUN",
            equip=equip,
            run_id=run_id,
            error_type=type(fe).__name__,
            error=str(fe)[:200]
        )
    finally:
        # CRITICAL FIX: Close OutputManager to prevent connection leaks
        try:
            if output_manager:
                output_manager.close()
        except Exception:
            pass
        try:
            getattr(sql_client, "close", lambda: None)()
        except Exception:
            pass

'''

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the insertion point - after _write_final_run_metadata function
    pattern = r"(def _write_final_run_metadata\([\s\S]*?return None\n)"
    match = re.search(pattern, content)
    
    if not match:
        print("ERROR: Could not find _write_final_run_metadata function")
        return
    
    insert_pos = match.end()
    
    # Check if helper already exists
    if "def _finalize_sql_and_record_metrics" in content:
        print("WARNING: _finalize_sql_and_record_metrics already exists")
        return
    
    # Insert the helper
    new_content = content[:insert_pos] + HELPER_CODE + content[insert_pos:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("SUCCESS: Added _finalize_sql_and_record_metrics helper function")

if __name__ == "__main__":
    main()
