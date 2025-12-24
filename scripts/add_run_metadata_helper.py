#!/usr/bin/env python
"""Add _write_final_run_metadata helper function to acm_main.py"""
import re

HELPER_CODE = '''
@dataclass
class FinalRunContext:
    """Context for writing final run metadata in finally block."""
    frame: Optional[pd.DataFrame]
    train: Optional[pd.DataFrame]
    score: Optional[pd.DataFrame]
    episodes: Optional[pd.DataFrame]
    meta: Optional[Any]
    score_regime_labels: Optional[Any]
    tables_dir: Optional[Path]
    refit_flag_path: Optional[Path]
    quality_ok: bool
    use_per_regime: bool


def _write_final_run_metadata(
    ctx: FinalRunContext,
    sql_client: "SQLClient",
    run_id: str,
    equip_id: int,
    equip: str,
    run_start_time: datetime,
    config_signature: str,
    rows_read: int,
    outcome: str,
    err_json: Optional[str],
    SQL_MODE: bool,
) -> Optional[Dict[str, Any]]:
    """
    Write run metadata to ACM_Runs table.
    
    This must happen for ALL runs (OK, NOOP, FAIL) for proper audit trail.
    
    Args:
        ctx: Context with frames and metadata objects
        sql_client: SQL client for writing
        run_id: Current run ID
        equip_id: Equipment ID
        equip: Equipment name
        run_start_time: When run started
        config_signature: Config hash
        rows_read: Number of rows read
        outcome: Run outcome (OK, NOOP, FAIL)
        err_json: Error JSON string if failed
        SQL_MODE: Whether SQL mode is enabled
        
    Returns:
        run_metadata dict if successful, None otherwise
    """
    from core.observability import Console
    from core.run_metadata_writer import (
        write_run_metadata,
        extract_run_metadata_from_scores,
        extract_data_quality_score,
    )
    from core.metrics import record_data_quality
    
    run_completion_time = datetime.now()
    run_metadata: Dict[str, Any] = {}
    
    try:
        # Extract health metrics from scores if available
        if ctx.frame is not None and isinstance(ctx.frame, pd.DataFrame) and len(ctx.frame) > 0:
            per_regime_enabled = bool(ctx.quality_ok and ctx.use_per_regime)
            regime_count = len(set(ctx.score_regime_labels)) if ctx.score_regime_labels is not None else 0
            run_metadata = extract_run_metadata_from_scores(
                ctx.frame,
                per_regime_enabled=per_regime_enabled,
                regime_count=regime_count
            )
            data_quality_path = ctx.tables_dir / "data_quality.csv" if ctx.tables_dir else None
            data_quality_score = extract_data_quality_score(
                data_quality_path=data_quality_path,
                sql_client=sql_client if SQL_MODE else None,
                run_id=run_id,
                equip_id=equip_id
            )
            
            # Record data quality for Prometheus/Grafana observability
            record_data_quality(
                equipment=equip,
                quality_score=float(data_quality_score) if data_quality_score else 0.0,
            )
        else:
            # NOOP or failed run - use defaults
            run_metadata = {
                "health_status": "UNKNOWN",
                "avg_health_index": None,
                "min_health_index": None,
                "max_fused_z": None
            }
            data_quality_score = 0.0
        
        # Get kept columns list
        kept_cols_str = ",".join(getattr(ctx.meta, "kept_cols", [])) if ctx.meta else ""
        
        # Calculate row counts
        train_row_count = len(ctx.train) if ctx.train is not None and isinstance(ctx.train, pd.DataFrame) else 0
        score_row_count = (
            len(ctx.frame) if ctx.frame is not None and isinstance(ctx.frame, pd.DataFrame) else (
                len(ctx.score) if ctx.score is not None and isinstance(ctx.score, pd.DataFrame) else rows_read
            )
        )
        episode_count = len(ctx.episodes) if ctx.episodes is not None and isinstance(ctx.episodes, pd.DataFrame) else 0
        
        # Check refit request
        refit_requested = False
        if ctx.refit_flag_path is not None:
            if isinstance(ctx.refit_flag_path, str):
                refit_requested = Path(ctx.refit_flag_path).exists()
            elif isinstance(ctx.refit_flag_path, Path):
                refit_requested = ctx.refit_flag_path.exists()
        
        # Write run metadata
        write_run_metadata(
            sql_client=sql_client,
            run_id=run_id,
            equip_id=equip_id,
            equip_name=equip,
            started_at=run_start_time,
            completed_at=run_completion_time,
            config_signature=config_signature,
            train_row_count=train_row_count,
            score_row_count=score_row_count,
            episode_count=episode_count,
            health_status=run_metadata.get("health_status", "UNKNOWN"),
            avg_health_index=run_metadata.get("avg_health_index"),
            min_health_index=run_metadata.get("min_health_index"),
            max_fused_z=run_metadata.get("max_fused_z"),
            data_quality_score=data_quality_score,
            refit_requested=refit_requested,
            kept_columns=kept_cols_str,
            error_message=err_json if outcome == "FAIL" else None
        )
        Console.info(
            f"Wrote run metadata to ACM_Runs for RunID={run_id} (outcome={outcome})",
            component="RUN_META"
        )
        
        # Store completion time in run_metadata for OTEL metrics
        run_metadata["_completion_time"] = run_completion_time
        return run_metadata
        
    except Exception as meta_err:
        Console.warn(
            f"Failed to write ACM_Runs metadata: {meta_err}",
            component="RUN_META",
            equip=equip,
            run_id=run_id,
            error=str(meta_err)[:200]
        )
        return None

'''

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the insertion point - after _log_timer_stats function
    pattern = r"(def _log_timer_stats\([\s\S]*?pass  # Timer logging should never break the run\n)"
    match = re.search(pattern, content)
    
    if not match:
        print("ERROR: Could not find _log_timer_stats function")
        return
    
    insert_pos = match.end()
    
    # Check if helper already exists
    if "def _write_final_run_metadata" in content:
        print("WARNING: _write_final_run_metadata already exists")
        return
    
    # Insert the helper
    new_content = content[:insert_pos] + HELPER_CODE + content[insert_pos:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("SUCCESS: Added FinalRunContext dataclass and _write_final_run_metadata helper function")

if __name__ == "__main__":
    main()
