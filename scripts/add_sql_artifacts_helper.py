#!/usr/bin/env python3
"""Add SQLArtifactResult dataclass and _write_sql_artifacts helper function."""

# Read the file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the location to insert (before _build_sensor_context function)
insert_marker = "def _build_sensor_context("

# New code to insert
new_code = '''@dataclass
class SQLArtifactResult:
    """Result of SQL artifact writing."""
    rows_scores: int = 0
    rows_drift: int = 0
    rows_events: int = 0
    rows_regimes: int = 0
    rows_pca_model: int = 0
    rows_pca_load: int = 0
    rows_pca_metrics: int = 0
    
    @property
    def total_rows(self) -> int:
        return int(self.rows_scores + self.rows_drift + self.rows_events + 
                   self.rows_regimes + self.rows_pca_model + self.rows_pca_load + 
                   self.rows_pca_metrics)


def _write_sql_artifacts(
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    train: pd.DataFrame,
    pca_detector: Any,
    spe_p95_train: float,
    t2_p95_train: float,
    output_manager: Any,
    sql_client: Any,
    equip_id: int,
    run_id: str,
    cfg: Dict[str, Any],
    T: Timer,
    equip: str,
) -> SQLArtifactResult:
    """Write SQL artifacts (scores, drift, events, regimes, PCA).
    
    Consolidates all SQL artifact writing into a single function.
    
    Args:
        frame: Fused scores frame.
        episodes: Normalized episodes DataFrame.
        train: Training data.
        pca_detector: PCA detector instance.
        spe_p95_train: SPE P95 threshold from training.
        t2_p95_train: T2 P95 threshold from training.
        output_manager: OutputManager instance.
        sql_client: SQL client instance.
        equip_id: Equipment ID.
        run_id: Run identifier.
        cfg: Configuration dictionary.
        T: Timer for profiling.
        equip: Equipment name for logging.
    
    Returns:
        SQLArtifactResult with row counts for each artifact type.
    """
    result = SQLArtifactResult()
    
    if not sql_client:
        Console.info("No SQL client available, skipping SQL writes", component="SQL")
        return result
    
    # 1) ScoresTS: write fused + calibrated z streams
    with T.section("sql.batch_writes"):
        try:
            Console.info("Starting batched artifact writes...", component="SQL")
            
            out_scores_wide = pd.DataFrame(index=frame.index)
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
                    result.rows_scores = output_manager.write_scores_ts(long_scores, run_id or "")
                    
        except Exception as e:
            Console.warn(f"Batched SQL writes failed, continuing with individual writes: {e}", component="SQL",
                         equip=equip, run_id=run_id, error_type=type(e).__name__, error=str(e)[:200])
    
    # Fallback to individual writes if batching not available
    if result.rows_scores == 0:
        with T.section("sql.scores.individual"):
            try:
                out_scores_wide = pd.DataFrame(index=frame.index)
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
                        result.rows_scores = output_manager.write_scores_ts(long_scores, run_id or "")
            except Exception as e:
                Console.warn(f"ScoresTS write skipped: {e}", component="SQL",
                             equip=equip, run_id=run_id, error=str(e)[:200])

    # 2) DriftTS
    with T.section("sql.drift"):
        try:
            if "drift_z" in frame.columns:
                drift_method = (cfg.get("drift", {}) or {}).get("method", "CUSUM")
                df_drift = pd.DataFrame({
                    "EntryDateTime": pd.to_datetime(frame.index),
                    "EquipID": int(equip_id),
                    "DriftZ": frame["drift_z"].astype(np.float32),
                    "Method": drift_method,
                    "RunID": run_id or ""
                })
                result.rows_drift = output_manager.write_drift_ts(df_drift, run_id or "")
        except Exception as e:
            Console.warn(f"DriftTS write skipped: {e}", component="SQL",
                         equip=equip, run_id=run_id, error=str(e)[:200])

    # 3) AnomalyEvents
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
                result.rows_events = output_manager.write_anomaly_events(df_events, run_id or "")
        except Exception as e:
            Console.warn(f"AnomalyEvents write skipped: {e}", component="SQL",
                         equip=equip, run_id=run_id, error=str(e)[:200])

    # 4) RegimeEpisodes
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
                result.rows_regimes = output_manager.write_regime_episodes(df_reg, run_id or "")
        except Exception as e:
            Console.warn(f"RegimeEpisodes write skipped: {e}", component="SQL",
                         equip=equip, run_id=run_id, error=str(e)[:200])

    # 5) PCA Model / Loadings / Metrics
    with T.section("sql.pca"):
        try:
            now_utc = pd.Timestamp.now()
            pca_model = getattr(pca_detector, "pca", None)

            var_ratio = getattr(pca_model, "explained_variance_ratio_", None)
            var_json = json.dumps(var_ratio.tolist()) if var_ratio is not None else "[]"
            
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
            result.rows_pca_model = output_manager.write_pca_model(model_row)

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
                result.rows_pca_load = output_manager.write_pca_loadings(df_load, run_id or "")

            # PCA Metrics
            spe_p95 = float(np.nanpercentile(frame["pca_spe"].to_numpy(dtype=np.float32), 95)) if "pca_spe" in frame.columns else None
            t2_p95 = float(np.nanpercentile(frame["pca_t2"].to_numpy(dtype=np.float32), 95)) if "pca_t2" in frame.columns else None

            var90_n = None
            if var_ratio is not None:
                csum = np.cumsum(var_ratio)
                var90_n = int(np.searchsorted(csum, 0.90) + 1)
            df_metrics = pd.DataFrame([{
                "RunID": run_id or "",
                "EntryDateTime": now_utc,
                "Var90_N": var90_n,
                "ReconRMSE": None,
                "P95_ReconRMSE": spe_p95,
                "Notes": json.dumps({"SPE_P95_score": spe_p95, "T2_P95_score": t2_p95})
            }])
            result.rows_pca_metrics = output_manager.write_pca_metrics(df_metrics, run_id or "")
        except Exception as e:
            Console.warn(f"PCA artifacts write skipped: {e}", component="SQL",
                         equip=equip, run_id=run_id, error=str(e)[:200])
    
    return result


'''

# Check if already added
if "class SQLArtifactResult:" in content:
    print("SQLArtifactResult already exists, skipping")
else:
    # Insert before _build_sensor_context
    if insert_marker in content:
        content = content.replace(insert_marker, new_code + insert_marker)
        
        with open("core/acm_main.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("SUCCESS: Added SQLArtifactResult and _write_sql_artifacts helper")
    else:
        print("ERROR: Could not find insertion marker")
