#!/usr/bin/env python3
"""Replace SQL artifact writing section in main() with _write_sql_artifacts() call."""

with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# The old code block - this is the SQL SPECIFIC ARTIFACT WRITING section
# We need to find where it starts and ends

old_code = '''        # === SQL-SPECIFIC ARTIFACT WRITING (BATCHED TRANSACTION) ===
        # Batch all SQL writes in a single transaction to prevent connection pool exhaustion
        if sql_client:
            with T.section("sql.batch_writes"):
                try:
                    Console.info("Starting batched artifact writes...", component="SQL")
                    
                    # 1) ScoresTS: write fused + calibrated z streams (as sensors)
                    rows_scores = 0
                    out_scores_wide = pd.DataFrame(index=frame.index)
                    # Name them explicitly to keep clarity in Grafana
                    if "fused" in frame.columns:       out_scores_wide["ACM_fused"] = frame["fused"]
                    if "pca_spe_z" in frame.columns:   out_scores_wide["ACM_pca_spe_z"] = frame["pca_spe_z"]
                    if "pca_t2_z" in frame.columns:    out_scores_wide["ACM_pca_t2_z"] = frame["pca_t2_z"]
                    if "mhal_z" in frame.columns:      out_scores_wide["ACM_mhal_z"] = frame["mhal_z"]
                    if "iforest_z" in frame.columns:   out_scores_wide["ACM_iforest_z"] = frame["iforest_z"]
                    if "gmm_z" in frame.columns:       out_scores_wide["ACM_gmm_z"] = frame["gmm_z"]
                    if "river_hst_z" in frame.columns: out_scores_wide["ACM_river_hst_z"] = frame["river_hst_z"]

                    if len(out_scores_wide.columns):
                        with T.section("sql.scores.melt"):
                            long_scores = output_manager.melt_scores_long(out_scores_wide, equip_id=equip_id, run_id=run_id or "", source="ACM")
                        with T.section("sql.scores.write"):
                            rows_scores = output_manager.write_scores_ts(long_scores, run_id or "")
                        
                except Exception as e:
                    Console.warn(f"Batched SQL writes failed, continuing with individual writes: {e}", component="SQL",
                                 equip=equip, run_id=run_id, error_type=type(e).__name__, error=str(e)[:200])
                    rows_scores = 0
        else:
            Console.info("No SQL client available, skipping SQL writes", component="SQL")
            rows_scores = 0
            
        # Fallback to individual writes if batching not available
        if sql_client and rows_scores == 0:
            # 1) ScoresTS: write fused + calibrated z streams (as sensors)
            with T.section("sql.scores.individual"):
                rows_scores = 0
                try:
                    out_scores_wide = pd.DataFrame(index=frame.index)
                    # Name them explicitly to keep clarity in Grafana
                    if "fused" in frame.columns:       out_scores_wide["ACM_fused"] = frame["fused"]
                    if "pca_spe_z" in frame.columns:   out_scores_wide["ACM_pca_spe_z"] = frame["pca_spe_z"]
                    if "pca_t2_z" in frame.columns:    out_scores_wide["ACM_pca_t2_z"] = frame["pca_t2_z"]
                    if "mhal_z" in frame.columns:      out_scores_wide["ACM_mhal_z"] = frame["mhal_z"]
                    if "iforest_z" in frame.columns:   out_scores_wide["ACM_iforest_z"] = frame["iforest_z"]
                    if "gmm_z" in frame.columns:       out_scores_wide["ACM_gmm_z"] = frame["gmm_z"]
                    if "river_hst_z" in frame.columns: out_scores_wide["ACM_river_hst_z"] = frame["river_hst_z"]

                    if len(out_scores_wide.columns):
                        with T.section("sql.scores.melt"):
                            long_scores = output_manager.melt_scores_long(out_scores_wide, equip_id=equip_id, run_id=run_id or "", source="ACM")
                        with T.section("sql.scores.write"):
                            rows_scores = output_manager.write_scores_ts(long_scores, run_id or "")
                except Exception as e:
                    Console.warn(f"ScoresTS write skipped: {e}", component="SQL",
                                 equip=equip, run_id=run_id, error=str(e)[:200])

        # 2) DriftTS (if drift_z exists) â€" method from config
        rows_drift = 0
        with T.section("sql.drift"):
            try:
                if "drift_z" in frame.columns:
                    drift_method = (cfg.get("drift", {}) or {}).get("method", "CUSUM") # type: ignore
                    df_drift = pd.DataFrame({
                        "EntryDateTime": pd.to_datetime(frame.index),
                        "EquipID": int(equip_id),
                        "DriftZ": frame["drift_z"].astype(np.float32),
                        "Method": drift_method,
                        "RunID": run_id or ""
                    })
                    rows_drift = output_manager.write_drift_ts(df_drift, run_id or "")
            except Exception as e:
                Console.warn(f"DriftTS write skipped: {e}", component="SQL",
                             equip=equip, run_id=run_id, error=str(e)[:200])

        # 3) AnomalyEvents (from episodes)
        rows_events = 0
        with T.section("sql.events"):
            try:
                if len(episodes):
                    df_events = pd.DataFrame({
                        "EquipID": int(equip_id),
                        "start_ts": episodes["start_ts"],
                        "end_ts": episodes["end_ts"],
                        "severity": episodes.get("severity", pd.Series(["info"]*len(episodes))),
                        "Detector": "FUSION",
                        "Score": episodes.get("score", np.nan),
                        "ContributorsJSON": episodes.get("culprits", "{}"),
                        "RunID": run_id or ""
                    })
                    rows_events = output_manager.write_anomaly_events(df_events, run_id or "")
            except Exception as e:
                Console.warn(f"AnomalyEvents write skipped: {e}", component="SQL",
                             equip=equip, run_id=run_id, error=str(e)[:200])

        # 4) RegimeEpisodes
        rows_regimes = 0
        with T.section("sql.regimes"):
            try:
                if len(episodes):
                    df_reg = pd.DataFrame({
                        "EquipID": int(equip_id),
                        "StartEntryDateTime": episodes["start_ts"],
                        "EndEntryDateTime": episodes["end_ts"],
                        "RegimeLabel": episodes.get("regime", pd.Series([""]*len(episodes))),
                        "Confidence": np.nan,
                        "RunID": run_id or ""
                    })
                    rows_regimes = output_manager.write_regime_episodes(df_reg, run_id or "")
            except Exception as e:
                Console.warn(f"RegimeEpisodes write skipped: {e}", component="SQL",
                             equip=equip, run_id=run_id, error=str(e)[:200])

        # 5) PCA Model / Loadings / Metrics
        rows_pca_model = rows_pca_load = rows_pca_metrics = 0
        with T.section("sql.pca"):
            try:
                now_utc = pd.Timestamp.now()
                pca_model = getattr(pca_detector, "pca", None) # type: ignore

                # PCA Model row (TRAIN window used)
                var_ratio = getattr(pca_model, "explained_variance_ratio_", None)
                var_json = json.dumps(var_ratio.tolist()) if var_ratio is not None else "[]"
                
                # Use TRAIN-based thresholds computed earlier in calibration section
                # (spe_p95_train, t2_p95_train are already available from TRAIN data)

                # Capture actual scaler type from PCA detector
                scaler_name = pca_detector.scaler.__class__.__name__ if hasattr(pca_detector, 'scaler') else "StandardScaler"
                scaler_params = {}
                if hasattr(pca_detector, 'scaler'):
                    scaler_params["with_mean"] = getattr(pca_detector.scaler, 'with_mean', True)
                    scaler_params["with_std"] = getattr(pca_detector.scaler, 'with_std', True)
                else:
                    scaler_params = {"with_mean": True, "with_std": True}
                
                scaling_spec = json.dumps({"scaler": scaler_name, **scaler_params})
                model_row = {
                    "RunID": run_id or "",
                    "EquipID": int(equip_id),
                    "EntryDateTime": now_utc,
                    "NComponents": int(getattr(pca_model, "n_components_", getattr(pca_model, "n_components", 0))),
                    "TargetVar": json.dumps({"SPE_P95_train": spe_p95_train, "T2_P95_train": t2_p95_train}),
                    "VarExplainedJSON": var_json,
                    "ScalingSpecJSON": scaling_spec,
                    "ModelVersion": cfg.get("runtime", {}).get("version", "v5.0.0"),
                    "TrainStartEntryDateTime": train.index.min() if len(train.index) else None,
                    "TrainEndEntryDateTime": train.index.max() if len(train.index) else None
                }
                rows_pca_model = output_manager.write_pca_model(model_row)

                # PCA Loadings
                comps = getattr(pca_model, "components_", None)
                if comps is not None and hasattr(train, "columns"):
                    load_rows = []
                    for k in range(comps.shape[0]):
                        for j, sensor in enumerate(train.columns):
                            load_rows.append({
                                "RunID": run_id or "",
                                "EntryDateTime": now_utc,
                                "ComponentNo": int(k + 1),
                                "Sensor": str(sensor),
                                "Loading": float(comps[k, j])
                            })
                    df_load = pd.DataFrame(load_rows)
                    rows_pca_load = output_manager.write_pca_loadings(df_load, run_id or "")

                # PCA Metrics
                spe_p95 = float(np.nanpercentile(frame["pca_spe"].to_numpy(dtype=np.float32), 95)) if "pca_spe" in frame.columns else None
                t2_p95  = float(np.nanpercentile(frame["pca_t2"].to_numpy(dtype=np.float32),  95)) if "pca_t2"  in frame.columns else None

                var90_n = None
                if var_ratio is not None:
                    csum = np.cumsum(var_ratio)
                    var90_n = int(np.searchsorted(csum, 0.90) + 1)
                df_metrics = pd.DataFrame([{
                    "RunID": run_id or "",
                    "EntryDateTime": now_utc,
                    "Var90_N": var90_n,
                    "ReconRMSE": None,
                    "P95_ReconRMSE": spe_p95, # This is the score-based P95 SPE
                    "Notes": json.dumps({"SPE_P95_score": spe_p95, "T2_P95_score": t2_p95})
                }])
                rows_pca_metrics = output_manager.write_pca_metrics(df_metrics, run_id or "")
            except Exception as e:
                Console.warn(f"PCA artifacts write skipped: {e}", component="SQL",
                             equip=equip, run_id=run_id, error=str(e)[:200])

        # Aggregate row counts for finalize
        rows_written = int(rows_scores + rows_drift + rows_events + rows_regimes + rows_pca_model + rows_pca_load + rows_pca_metrics)'''

# New code with helper function call
new_code = '''        # === SQL-SPECIFIC ARTIFACT WRITING ===
        sql_artifact_result = _write_sql_artifacts(
            frame=frame, episodes=episodes, train=train, pca_detector=pca_detector,
            spe_p95_train=spe_p95_train, t2_p95_train=t2_p95_train,
            output_manager=output_manager, sql_client=sql_client, equip_id=equip_id,
            run_id=run_id, cfg=cfg, T=T, equip=equip,
        )
        rows_written = sql_artifact_result.total_rows'''

if old_code in content:
    content = content.replace(old_code, new_code)
    
    with open("core/acm_main.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    old_lines = len(old_code.split('\n'))
    new_lines = len(new_code.split('\n'))
    print(f"SUCCESS: Replaced SQL artifacts section ({old_lines} lines -> {new_lines} lines)")
    print(f"         Removed {old_lines - new_lines} lines from main()")
else:
    print("ERROR: Could not find old_code block to replace")
    if "=== SQL-SPECIFIC ARTIFACT WRITING" in content:
        print("Found section header, but full block doesn't match")
        # Try to find the issue
        idx = content.find("=== SQL-SPECIFIC ARTIFACT WRITING")
        print(f"Section at char {idx}")
    else:
        print("Section header not found")
