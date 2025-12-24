#!/usr/bin/env python3
"""Add _write_run_stats helper function."""

# Read the file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the location to insert (before SQLArtifactResult dataclass)
insert_marker = "@dataclass\nclass SQLArtifactResult:"

# New helper function
new_code = '''def _write_run_stats_and_culprits(
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    meta: Any,
    output_manager: Any,
    sql_client: Any,
    equip_id: int,
    run_id: str,
    rows_read: int,
    anomaly_count: int,
    win_start: Optional[datetime],
    win_end: Optional[datetime],
    equip: str,
    T: Timer,
) -> None:
    """Write run stats and episode culprits to SQL.
    
    Args:
        frame: Fused scores frame.
        episodes: Normalized episodes DataFrame.
        meta: Metadata object with kept_cols.
        output_manager: OutputManager instance.
        sql_client: SQL client instance.
        equip_id: Equipment ID.
        run_id: Run identifier.
        rows_read: Number of rows read.
        anomaly_count: Number of anomalies.
        win_start: Window start datetime.
        win_end: Window end datetime.
        equip: Equipment name for logging.
        T: Timer for profiling.
    """
    # Run Stats
    with T.section("sql.run_stats"):
        try:
            if sql_client and run_id and win_start is not None and win_end is not None:
                drift_p95 = None
                if "drift_z" in frame.columns:
                    drift_p95 = float(np.nanpercentile(frame["drift_z"].to_numpy(dtype=np.float32), 95))
                recon_rmse = None
                sensors_kept = len(getattr(meta, "kept_cols", []))
                cadence_ok_pct = float(getattr(meta, "cadence_ok", 1.0)) * 100.0 if hasattr(meta, "cadence_ok") else None

                output_manager.write_run_stats({
                    "RunID": run_id,
                    "EquipID": int(equip_id),
                    "WindowStartEntryDateTime": win_start,
                    "WindowEndEntryDateTime": win_end,
                    "SamplesIn": rows_read,
                    "SamplesKept": rows_read,
                    "SensorsKept": sensors_kept,
                    "CadenceOKPct": cadence_ok_pct,
                    "DriftP95": drift_p95,
                    "ReconRMSE": recon_rmse,
                    "AnomalyCount": anomaly_count
                })
        except Exception as e:
            Console.warn(f"RunStats not recorded: {e}", component="RUN",
                         equip=equip, run_id=run_id, error=str(e)[:200])

    # Episode Culprits
    with T.section("sql.culprits"):
        try:
            if sql_client and run_id and isinstance(episodes, pd.DataFrame) and len(episodes) > 0:
                write_episode_culprits_enhanced(
                    sql_client=sql_client,
                    run_id=run_id,
                    episodes=episodes,
                    scores_df=frame,
                    equip_id=equip_id
                )
                Console.info(f"Successfully wrote episode culprits to ACM_EpisodeCulprits for RunID={run_id}", component="CULPRITS")
        except Exception as e:
            Console.warn(f"Failed to write ACM_EpisodeCulprits: {e}", component="CULPRITS",
                         equip=equip, run_id=run_id, error=str(e))


'''

# Check if already added
if "def _write_run_stats_and_culprits(" in content:
    print("_write_run_stats_and_culprits already exists, skipping")
else:
    # Insert before SQLArtifactResult
    if insert_marker in content:
        content = content.replace(insert_marker, new_code + insert_marker)
        
        with open("core/acm_main.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("SUCCESS: Added _write_run_stats_and_culprits helper function")
    else:
        print("ERROR: Could not find insertion marker")
