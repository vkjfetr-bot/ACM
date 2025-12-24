#!/usr/bin/env python
"""Add _persist_file_artifacts helper function to acm_main.py"""
import re

HELPER_CODE = '''

def _persist_file_artifacts(
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    meta: "DataMeta",
    run_dir: Path,
    models_dir: Path,
    stable_models_dir: Optional[Path],
    output_manager: "OutputManager",
    model_cache_path: Path,
    regime_model: Optional[Any],
    regime_quality_ok: bool,
    cache_payload: Optional[Any],
    equip: str,
    run_id: str,
    cfg: Dict[str, Any],
    T: "Timer",
    SQL_MODE: bool,
    dual_mode: bool,
) -> None:
    """
    Persist file-based artifacts (scores, episodes, schema, culprits, etc).
    
    Handles all file-based persistence including schema.json generation,
    score stream, regime model saving, episodes, culprits, and detector caching.
    Most operations are skipped in SQL-only mode.
    
    Args:
        frame: Score frame with detector outputs
        episodes: Detected anomaly episodes
        meta: Data metadata with kept_cols
        run_dir: Run output directory
        models_dir: Models directory
        stable_models_dir: Stable models cache directory
        output_manager: Output manager for writes
        model_cache_path: Path for detector cache
        regime_model: Trained regime model
        regime_quality_ok: Whether regime model quality is OK
        cache_payload: Detector cache payload
        equip: Equipment name
        run_id: Current run ID
        cfg: Configuration dictionary
        T: Timer instance
        SQL_MODE: Whether SQL mode is enabled
        dual_mode: Whether dual-write mode is enabled
    """
    from core.observability import Console
    from core.metrics import record_episode
    import core.regimes as regimes
    
    with T.section("persist"):
        if not SQL_MODE:
            out_log = run_dir / "run.jsonl"
        
        # Write scores with schema
        with T.section("persist.write_scores"):
            try:
                output_manager.write_scores(frame, run_dir, enable_sql=(SQL_MODE or dual_mode))
                
                # Generate schema.json for file mode
                if not SQL_MODE:
                    _write_schema_json(frame, run_dir, equip)
                    
            except Exception as we:
                Console.warn(
                    f"Failed to write scores via OutputManager: {we}",
                    component="IO",
                    equip=equip,
                    run_id=run_id,
                    error=str(we)[:200]
                )
        
        # Skip filesystem persistence in SQL-only mode
        if not SQL_MODE:
            with T.section("persist.write_score_stream"):
                try:
                    stream = frame.copy().reset_index().rename(columns={"index": "ts"})
                    output_manager.write_dataframe(stream, models_dir / "score_stream.csv")
                except Exception as se:
                    Console.warn(
                        f"Failed to write score_stream via OutputManager: {se}",
                        component="IO",
                        equip=equip,
                        run_id=run_id,
                        error=str(se)[:200]
                    )
            
            if regime_model is not None:
                with T.section("persist.regime_model"):
                    try:
                        regimes.save_regime_model(regime_model, models_dir)
                    except Exception as e:
                        Console.warn(
                            f"Failed to persist regime model: {e}",
                            component="REGIME",
                            equip=equip,
                            error=str(e)[:200]
                        )
                    
                    # Promote to stable cache if quality OK
                    promote_dir = None
                    if stable_models_dir and stable_models_dir != models_dir:
                        promote_dir = stable_models_dir
                    
                    if promote_dir and regime_quality_ok:
                        try:
                            regimes.save_regime_model(regime_model, promote_dir)
                            Console.info(f"Promoted regime model to {promote_dir}", component="REGIME")
                        except Exception as promote_exc:
                            Console.warn(
                                f"Failed to promote regime model to stable cache: {promote_exc}",
                                component="REGIME",
                                equip=equip,
                                error=str(promote_exc)[:200]
                            )
                    elif promote_dir and not regime_quality_ok:
                        Console.info("Skipping stable regime cache update because quality_ok=False", component="REGIME")
        
        # Write episodes
        with T.section("persist.write_episodes"):
            try:
                output_manager.write_episodes(episodes, run_dir, enable_sql=dual_mode)
                episode_count = len(episodes) if episodes is not None else 0
                if episode_count > 0:
                    record_episode(equip, count=episode_count, severity="info")
            except Exception as ee:
                Console.warn(
                    f"Failed to write episodes via OutputManager: {ee}",
                    component="IO",
                    equip=equip,
                    run_id=run_id,
                    error=str(ee)[:200]
                )
        
        # Write culprits.jsonl (file mode only)
        with T.section("persist.write_culprits"):
            if not SQL_MODE:
                _write_culprits_jsonl(episodes, run_dir, equip)
        
        # Write runlog (file mode only)
        with T.section("persist.write_runlog"):
            if not SQL_MODE:
                out_log = run_dir / "run.jsonl"
                with out_log.open("w", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "equip": equip,
                        "rows": int(len(frame)),
                        "kept_cols": meta.kept_cols
                    }, ensure_ascii=False) + "\\n")
        
        # Cache detectors
        if cache_payload:
            with T.section("persist.cache_detectors"):
                try:
                    joblib.dump(cache_payload, model_cache_path)
                    Console.info(f"Cached detectors to {model_cache_path}", component="MODEL")
                except Exception as e:
                    Console.warn(
                        f"Failed to cache detectors: {e}",
                        component="MODEL",
                        equip=equip,
                        cache_path=str(model_cache_path),
                        error=str(e)[:200]
                    )
    
    # Log file mode summary
    if not SQL_MODE:
        Console.info(
            f"rows={len(frame)} heads={','.join([c for c in frame.columns if c.endswith('_z')])} episodes={len(episodes)}",
            component="OK"
        )
        Console.info(f"{run_dir / 'scores.csv'}", component="ART")
        Console.info(f"{run_dir / 'episodes.csv'}", component="ART")


def _write_schema_json(frame: pd.DataFrame, run_dir: Path, equip: str) -> None:
    """Write schema.json descriptor for scores.csv."""
    from core.observability import Console
    
    try:
        schema_path = run_dir / "schema.json"
        schema_dict = {
            "file": "scores.csv",
            "description": "ACM anomaly scores with detector outputs and fusion results",
            "timestamp_column": "index" if frame.index.name is None else frame.index.name,
            "columns": []
        }
        
        # Document each column
        for col in frame.columns:
            col_info = {
                "name": str(col),
                "dtype": str(frame[col].dtype),
                "nullable": bool(frame[col].isnull().any())
            }
            
            # Semantic descriptions
            if col.endswith("_raw"):
                col_info["description"] = f"Raw anomaly score from {col.replace('_raw', '')} detector"
            elif col.endswith("_z"):
                col_info["description"] = f"Calibrated z-score from {col.replace('_z', '')} detector"
            elif col == "fused":
                col_info["description"] = "Weighted fusion of all detector z-scores"
            elif col == "alert_level":
                col_info["description"] = "Alert severity: NORMAL, CAUTION, or FAULT"
            elif col == "alert_mode":
                col_info["description"] = "Alert mode based on threshold exceedance"
            elif col == "regime_label":
                col_info["description"] = "Operating regime cluster label (0-based)"
            elif col == "regime_state":
                col_info["description"] = "Regime health state: healthy, suspect, or critical"
            elif col == "episode_id":
                col_info["description"] = "Episode identifier for anomaly periods (NaN outside episodes)"
            else:
                col_info["description"] = f"Column {col}"
            
            schema_dict["columns"].append(col_info)
        
        with schema_path.open("w", encoding="utf-8") as sf:
            json.dump(schema_dict, sf, indent=2, ensure_ascii=False)
        Console.info(f"Schema written to {schema_path}", component="ART", columns=len(schema_dict.get('columns', [])))
        
    except Exception as se:
        Console.warn(f"Failed to write schema.json: {se}", component="IO", equip=equip, error=str(se)[:200])


def _write_culprits_jsonl(episodes: pd.DataFrame, run_dir: Path, equip: str) -> None:
    """Write culprits.jsonl for episode-level attribution."""
    from core.observability import Console
    
    try:
        culprits_path = run_dir / "culprits.jsonl"
        with culprits_path.open("w", encoding="utf-8") as cj:
            for _, row in episodes.iterrows():
                start_ts_val = row.get("start_ts")
                end_ts_val = row.get("end_ts")
                start_ts_str = None
                end_ts_str = None
                
                if pd.notna(start_ts_val):
                    start_dt = pd.to_datetime(start_ts_val, errors="coerce")
                    if pd.notna(start_dt):
                        start_ts_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(end_ts_val):
                    end_dt = pd.to_datetime(end_ts_val, errors="coerce")
                    if pd.notna(end_dt):
                        end_ts_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                
                rec = {
                    "start_ts": start_ts_str,
                    "end_ts": end_ts_str,
                    "duration_hours": float(row.get("duration_hours", np.nan)) if pd.notna(row.get("duration_hours", np.nan)) else None,
                    "culprits": row.get("culprits", ""),
                    "method": "episode_primary_detector"
                }
                cj.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        
        Console.info(f"Culprits written to {culprits_path}", component="ART", episodes=len(episodes))
        
    except Exception as ce:
        Console.warn(f"Failed to write culprits.jsonl: {ce}", component="IO", equip=equip, error=str(ce)[:200])

'''

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the insertion point - after _run_forecast_engine function
    pattern = r'(def _run_forecast_engine\([\s\S]*?return \{"success": False, "error": str\(e\)\}\n)'
    match = re.search(pattern, content)
    
    if not match:
        print("ERROR: Could not find _run_forecast_engine function")
        return
    
    insert_pos = match.end()
    
    # Check if helper already exists
    if "def _persist_file_artifacts" in content:
        print("WARNING: _persist_file_artifacts already exists")
        return
    
    # Insert the helper
    new_content = content[:insert_pos] + HELPER_CODE + content[insert_pos:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("SUCCESS: Added _persist_file_artifacts and related helper functions")

if __name__ == "__main__":
    main()
