#!/usr/bin/env python3
"""Replace the baseline buffer section in main() with _update_baseline_buffer() call."""

with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# The old code block to replace (full baseline buffer section)
old_code = '''        # ===== Rolling Baseline Buffer: Update with latest raw SCORE =====
        # ACM-CSV-01: Separate file-mode and SQL-mode baseline writes
        # OPTIMIZATION v10.2.1: Smart skip + vectorized writes for 100x speedup
        with T.section("baseline.buffer_write"):
          try:
            baseline_cfg = (cfg.get("runtime", {}) or {}).get("baseline", {}) or {}
            window_hours = float(baseline_cfg.get("window_hours", 72))
            max_points = int(baseline_cfg.get("max_points", 100000))
            # Smart refresh: only write every N batches (default 10) unless coldstart
            refresh_interval = int(baseline_cfg.get("refresh_interval_batches", 10))
            
            # Determine if we should write baseline buffer this run
            # Write if: (1) coldstart just completed, (2) periodic refresh, or (3) first run
            should_write_buffer = False
            write_reason = ""
            recent_run_count = 0  # Initialize for skip message
            if not coldstart_complete:
                # Coldstart in progress - always write to build baseline
                should_write_buffer = True
                write_reason = "coldstart"
            else:
                # Models exist - check periodic refresh
                # Use tick_minutes as a proxy for batch count (stored in run metadata)
                try:
                    with sql_client.cursor() as cur:
                        # Count recent runs to determine batch number for this equipment
                        run_count_result = cur.execute(
                            "SELECT COUNT(*) FROM ACM_Runs WHERE EquipID = ? AND CreatedAt > DATEADD(DAY, -7, GETDATE())",
                            (int(equip_id),)
                        ).fetchone()
                        recent_run_count = run_count_result[0] if run_count_result else 0
                        # Write on first run or every refresh_interval batches
                        if recent_run_count == 0 or (recent_run_count % refresh_interval == 0):
                            should_write_buffer = True
                            write_reason = f"periodic_refresh (batch {recent_run_count})"
                except Exception:
                    # On error, default to writing (safe fallback)
                    should_write_buffer = True
                    write_reason = "fallback"
            
            if not should_write_buffer:
                batches_until_refresh = refresh_interval - (recent_run_count % refresh_interval) if refresh_interval > 0 else 0
                Console.info(f"Skipping buffer write (models exist, next refresh in {batches_until_refresh} batches)", component="BASELINE")
            elif isinstance(score_numeric, pd.DataFrame) and len(score_numeric):
                to_append = score_numeric.copy()
                # Normalize index to local naive timestamps
                try:
                    idx_local = pd.DatetimeIndex(to_append.index).tz_localize(None)
                except Exception:
                    idx_local = pd.DatetimeIndex(to_append.index)
                to_append.index = idx_local
                
                if not SQL_MODE:
                    # ACM-CSV-01: File mode - write to baseline_buffer.csv
                    buffer_path = stable_models_dir / "baseline_buffer.csv"
                    if buffer_path.exists():
                        try:
                            prev = pd.read_csv(buffer_path, index_col=0, parse_dates=True)
                        except Exception:
                            prev = pd.DataFrame()
                        # Keep only common columns to avoid drift issues
                        common = [c for c in prev.columns if c in to_append.columns]
                        if common:
                            prev = prev[common]
                            to_append = to_append[common]
                        combined = pd.concat([prev, to_append], axis=0)
                    else:
                        combined = to_append

                    # Normalize index, drop dups, sort
                    try:
                        norm_idx = pd.to_datetime(combined.index, errors="coerce")
                    except Exception:
                        norm_idx = pd.DatetimeIndex(combined.index)
                    combined.index = norm_idx
                    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

                    # Apply retention policy
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
                    # ACM-CSV-01: SQL mode - write to ACM_BaselineBuffer
                    # OPTIMIZATION v10.2.1: Vectorized pandas melt (100x faster than Python loops)
                    try:
                        # Reset index to make timestamp a column for melt
                        to_append_reset = to_append.reset_index()
                        ts_col = to_append_reset.columns[0]  # First column is the timestamp index
                        
                        # Vectorized wide-to-long transformation using pandas melt
                        # This replaces the O(n*m) Python loop with a single vectorized operation
                        long_df = to_append_reset.melt(
                            id_vars=[ts_col],
                            var_name='SensorName',
                            value_name='SensorValue'
                        )
                        
                        # Drop NaN values and add EquipID
                        long_df = long_df.dropna(subset=['SensorValue'])
                        long_df['EquipID'] = int(equip_id)
                        long_df['DataQuality'] = None
                        
                        # Ensure timestamp is naive datetime
                        long_df[ts_col] = pd.to_datetime(long_df[ts_col]).dt.tz_localize(None)
                        
                        # Rename timestamp column for consistency
                        long_df = long_df.rename(columns={ts_col: 'Timestamp'})
                        
                        # Reorder columns to match INSERT statement
                        long_df = long_df[['EquipID', 'Timestamp', 'SensorName', 'SensorValue', 'DataQuality']]
                        
                        if len(long_df) > 0:
                            # Convert to list of tuples for executemany (in-memory, no temp files)
                            baseline_records = list(long_df.itertuples(index=False, name=None))
                            
                            # Bulk insert with fast_executemany
                            insert_sql = """
                            INSERT INTO dbo.ACM_BaselineBuffer (EquipID, Timestamp, SensorName, SensorValue, DataQuality)
                            VALUES (?, ?, ?, ?, ?)
                            """
                            with sql_client.cursor() as cur:
                                cur.fast_executemany = True
                                cur.executemany(insert_sql, baseline_records)
                            sql_client.conn.commit()
                            Console.info(f"Wrote {len(baseline_records)} records to ACM_BaselineBuffer ({write_reason})", component="BASELINE")
                        
                            # Run cleanup procedure to maintain retention policy
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
                         equip=equip, error_type=type(be).__name__, error=str(be)[:200])'''

# New code with helper function call
new_code = '''        # ===== Rolling Baseline Buffer: Update with latest raw SCORE =====
        _update_baseline_buffer(
            score_numeric=score_numeric, coldstart_complete=coldstart_complete,
            equip=equip, equip_id=equip_id, stable_models_dir=stable_models_dir,
            sql_client=sql_client, SQL_MODE=SQL_MODE, cfg=cfg, T=T,
        )'''

if old_code in content:
    content = content.replace(old_code, new_code)
    
    with open("core/acm_main.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    old_lines = len(old_code.split('\n'))
    new_lines = len(new_code.split('\n'))
    print(f"SUCCESS: Replaced baseline buffer section ({old_lines} lines -> {new_lines} lines)")
    print(f"         Removed {old_lines - new_lines} lines from main()")
else:
    print("ERROR: Could not find old_code block to replace")
    if "===== Rolling Baseline Buffer" in content:
        print("Found section header, but full block doesn't match")
    else:
        print("Section header not found")
