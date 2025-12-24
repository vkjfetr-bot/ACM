#!/usr/bin/env python3
"""Add _update_baseline_buffer helper function."""

# Read the file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the location to insert (before EpisodeNormResult dataclass)
insert_marker = "@dataclass\nclass EpisodeNormResult:"

# New helper function
new_code = '''def _update_baseline_buffer(
    score_numeric: pd.DataFrame,
    coldstart_complete: bool,
    equip: str,
    equip_id: int,
    stable_models_dir: Path,
    sql_client: Any,
    SQL_MODE: bool,
    cfg: Dict[str, Any],
    T: Timer,
) -> None:
    """Update rolling baseline buffer with latest score data.
    
    Writes to either file (baseline_buffer.csv) or SQL (ACM_BaselineBuffer)
    depending on mode. Implements smart refresh to avoid writing every run.
    
    Args:
        score_numeric: Numeric score data to append to buffer.
        coldstart_complete: Whether coldstart has completed.
        equip: Equipment name.
        equip_id: Equipment ID.
        stable_models_dir: Path to stable models directory.
        sql_client: SQL client for database access.
        SQL_MODE: Whether SQL mode is active.
        cfg: Configuration dictionary.
        T: Timer for profiling.
    """
    with T.section("baseline.buffer_write"):
        try:
            baseline_cfg = (cfg.get("runtime", {}) or {}).get("baseline", {}) or {}
            window_hours = float(baseline_cfg.get("window_hours", 72))
            max_points = int(baseline_cfg.get("max_points", 100000))
            refresh_interval = int(baseline_cfg.get("refresh_interval_batches", 10))
            
            # Determine if we should write baseline buffer this run
            should_write_buffer = False
            write_reason = ""
            recent_run_count = 0
            
            if not coldstart_complete:
                should_write_buffer = True
                write_reason = "coldstart"
            else:
                try:
                    with sql_client.cursor() as cur:
                        run_count_result = cur.execute(
                            "SELECT COUNT(*) FROM ACM_Runs WHERE EquipID = ? AND CreatedAt > DATEADD(DAY, -7, GETDATE())",
                            (int(equip_id),)
                        ).fetchone()
                        recent_run_count = run_count_result[0] if run_count_result else 0
                        if recent_run_count == 0 or (recent_run_count % refresh_interval == 0):
                            should_write_buffer = True
                            write_reason = f"periodic_refresh (batch {recent_run_count})"
                except Exception:
                    should_write_buffer = True
                    write_reason = "fallback"
            
            if not should_write_buffer:
                batches_until_refresh = refresh_interval - (recent_run_count % refresh_interval) if refresh_interval > 0 else 0
                Console.info(f"Skipping buffer write (models exist, next refresh in {batches_until_refresh} batches)", component="BASELINE")
                return
            
            if not isinstance(score_numeric, pd.DataFrame) or len(score_numeric) == 0:
                return
            
            to_append = score_numeric.copy()
            try:
                idx_local = pd.DatetimeIndex(to_append.index).tz_localize(None)
            except Exception:
                idx_local = pd.DatetimeIndex(to_append.index)
            to_append.index = idx_local
            
            if not SQL_MODE:
                # File mode - write to baseline_buffer.csv
                buffer_path = stable_models_dir / "baseline_buffer.csv"
                if buffer_path.exists():
                    try:
                        prev = pd.read_csv(buffer_path, index_col=0, parse_dates=True)
                    except Exception:
                        prev = pd.DataFrame()
                    common = [c for c in prev.columns if c in to_append.columns]
                    if common:
                        prev = prev[common]
                        to_append = to_append[common]
                    combined = pd.concat([prev, to_append], axis=0)
                else:
                    combined = to_append

                try:
                    norm_idx = pd.to_datetime(combined.index, errors="coerce")
                except Exception:
                    norm_idx = pd.DatetimeIndex(combined.index)
                combined.index = norm_idx
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()

                if len(combined):
                    last_ts = pd.to_datetime(combined.index.max())
                    if window_hours and window_hours > 0:
                        cutoff = last_ts - pd.Timedelta(hours=window_hours)
                        combined = combined[combined.index >= cutoff]
                    if max_points and max_points > 0 and len(combined) > max_points:
                        combined = combined.iloc[-max_points:]

                combined.to_csv(buffer_path, index=True, date_format="%Y-%m-%d %H:%M:%S")
                Console.info(f"Updated baseline_buffer.csv: rows={len(combined)} cols={len(combined.columns)}", component="BASELINE")
            
            elif sql_client and SQL_MODE:
                # SQL mode - write to ACM_BaselineBuffer
                try:
                    to_append_reset = to_append.reset_index()
                    ts_col = to_append_reset.columns[0]
                    
                    long_df = to_append_reset.melt(
                        id_vars=[ts_col],
                        var_name='SensorName',
                        value_name='SensorValue'
                    )
                    
                    long_df = long_df.dropna(subset=['SensorValue'])
                    long_df['EquipID'] = int(equip_id)
                    long_df['DataQuality'] = None
                    long_df[ts_col] = pd.to_datetime(long_df[ts_col]).dt.tz_localize(None)
                    long_df = long_df.rename(columns={ts_col: 'Timestamp'})
                    long_df = long_df[['EquipID', 'Timestamp', 'SensorName', 'SensorValue', 'DataQuality']]
                    
                    if len(long_df) > 0:
                        baseline_records = list(long_df.itertuples(index=False, name=None))
                        
                        insert_sql = """
                        INSERT INTO dbo.ACM_BaselineBuffer (EquipID, Timestamp, SensorName, SensorValue, DataQuality)
                        VALUES (?, ?, ?, ?, ?)
                        """
                        with sql_client.cursor() as cur:
                            cur.fast_executemany = True
                            cur.executemany(insert_sql, baseline_records)
                        sql_client.conn.commit()
                        Console.info(f"Wrote {len(baseline_records)} records to ACM_BaselineBuffer ({write_reason})", component="BASELINE")
                    
                        try:
                            with sql_client.cursor() as cur:
                                cur.execute("EXEC dbo.usp_CleanupBaselineBuffer @EquipID=?, @RetentionHours=?, @MaxRowsPerEquip=?",
                                          (int(equip_id), int(window_hours), max_points))
                            sql_client.conn.commit()
                        except Exception as cleanup_err:
                            Console.warn(f"Cleanup procedure failed: {cleanup_err}", component="BASELINE",
                                         equip=equip, equip_id=equip_id, error=str(cleanup_err)[:200])
                except Exception as sql_err:
                    Console.warn(f"SQL write to ACM_BaselineBuffer failed: {sql_err}", component="BASELINE",
                                 equip=equip, equip_id=equip_id, error=str(sql_err)[:200])
                    try:
                        sql_client.conn.rollback()
                    except:
                        pass
        except Exception as be:
            Console.warn(f"Baseline buffer update failed: {be}", component="BASELINE",
                         equip=equip, error_type=type(be).__name__, error=str(be)[:200])


'''

# Check if already added
if "def _update_baseline_buffer(" in content:
    print("_update_baseline_buffer already exists, skipping")
else:
    # Insert before EpisodeNormResult
    if insert_marker in content:
        content = content.replace(insert_marker, new_code + insert_marker)
        
        with open("core/acm_main.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("SUCCESS: Added _update_baseline_buffer helper function")
    else:
        print("ERROR: Could not find insertion marker")
