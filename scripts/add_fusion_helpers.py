#!/usr/bin/env python
"""Add fusion pipeline helper functions to acm_main.py."""

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find insertion point - before main()
    insert_marker = "\ndef main() -> None:"
    
    insert_idx = content.find(insert_marker)
    
    if insert_idx == -1:
        print("ERROR: Could not find insertion point")
        return
    
    helper_code = '''

# ---------- FUSION-01: _run_fusion_pipeline extracted helper ----------
@dataclass
class FusionPipelineResult:
    """Result from fusion pipeline including episodes and thresholds."""
    frame: pd.DataFrame
    episodes: pd.DataFrame
    train_frame: pd.DataFrame
    fusion_weights_used: Dict[str, float]
    tuning_diagnostics: Optional[Dict[str, Any]]
    episode_count: int


def _save_tuning_diagnostics(
    tuning_diagnostics: Dict[str, Any],
    fusion_weights_used: Dict[str, float],
    run_dir: Path,
    output_manager: Any,
    sql_client: Optional[Any],
    equip_id: int,
    run_id: Optional[str],
    equip: str,
    SQL_MODE: bool,
) -> None:
    """Save tuning diagnostics to file and/or SQL."""
    try:
        tables_dir = run_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        tune_path = tables_dir / "weight_tuning.json"
        
        # Add timestamp and metadata
        tuning_diagnostics["timestamp"] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if not SQL_MODE:
            with open(tune_path, 'w') as f:
                json.dump(tuning_diagnostics, f, indent=2)
            Console.info(f"Saved tuning diagnostics -> {tune_path}", component="TUNE")
        
        # Build fusion metrics rows
        metrics_rows = []
        for detector_name, weight in fusion_weights_used.items():
            det_metrics = tuning_diagnostics.get("detector_metrics", {}).get(detector_name, {})
            metrics_rows.append({
                "detector_name": detector_name,
                "weight": weight,
                "n_samples": det_metrics.get("n_samples", 0),
                "quality_score": det_metrics.get("quality_score", 0.0),
                "tuning_method": tuning_diagnostics.get("method", "unknown"),
                "timestamp": tuning_diagnostics["timestamp"]
            })
        
        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            output_manager.write_dataframe(metrics_df, "fusion_metrics")
            
            # SQL mode: ACM_RunMetrics bulk insert
            if sql_client and SQL_MODE:
                _write_fusion_metrics_to_sql(
                    sql_client=sql_client,
                    run_id=run_id,
                    equip_id=equip_id,
                    metrics_rows=metrics_rows,
                    equip=equip,
                )
    except Exception as save_e:
        Console.warn(f"Failed to save diagnostics: {save_e}", component="TUNE",
                     equip=equip, error=str(save_e)[:200])


def _write_fusion_metrics_to_sql(
    sql_client: Any,
    run_id: str,
    equip_id: int,
    metrics_rows: List[Dict[str, Any]],
    equip: str,
) -> None:
    """Write fusion metrics to ACM_RunMetrics SQL table."""
    timestamp_now = pd.Timestamp.now()
    insert_records = []
    
    for row in metrics_rows:
        insert_records.append((
            run_id, int(equip_id),
            f"fusion.weight.{row['detector_name']}",
            float(row['weight']), timestamp_now
        ))
        insert_records.append((
            run_id, int(equip_id),
            f"fusion.quality.{row['detector_name']}",
            float(row['quality_score']), timestamp_now
        ))
        insert_records.append((
            run_id, int(equip_id),
            f"fusion.n_samples.{row['detector_name']}",
            float(row['n_samples']), timestamp_now
        ))
    
    insert_sql = """
        INSERT INTO dbo.ACM_RunMetrics 
        (RunID, EquipID, MetricName, MetricValue, Timestamp)
        VALUES (?, ?, ?, ?, ?)
    """
    try:
        with sql_client.cursor() as cur:
            cur.fast_executemany = True
            cur.executemany(insert_sql, insert_records)
        sql_client.conn.commit()
        Console.info(f"Saved fusion metrics -> SQL:ACM_RunMetrics ({len(insert_records)} records)", component="TUNE")
    except Exception as sql_e:
        Console.warn(f"Failed to write fusion metrics to SQL: {sql_e}", component="TUNE",
                     equip=equip, run_id=run_id, error=str(sql_e)[:200])


def _calculate_train_fusion(
    train: pd.DataFrame,
    train_frame: pd.DataFrame,
    present: Dict[str, np.ndarray],
    weights: Dict[str, float],
    train_regime_labels: Optional[np.ndarray],
    cfg: Dict[str, Any],
    equip: str,
) -> pd.DataFrame:
    """Calculate fusion on TRAIN data for threshold baseline."""
    # Build present dict for train data
    train_present = {}
    for detector_name in present.keys():
        if detector_name in train_frame.columns:
            train_present[detector_name] = train_frame[detector_name].to_numpy(copy=False)
    
    # Calculate fusion on train data
    if train_present and not train.empty:
        try:
            train_fused, _ = fuse.combine(
                train_present, weights, cfg, 
                original_features=train, 
                regime_labels=train_regime_labels
            )
            train_fused_np = np.asarray(train_fused, dtype=np.float32).reshape(-1)
            train_frame["fused"] = train_fused_np
        except Exception as train_fuse_e:
            Console.warn(f"Failed to calculate train fusion: {train_fuse_e}", component="FUSE",
                         equip=equip, error_type=type(train_fuse_e).__name__, error=str(train_fuse_e)[:200])
            train_frame["fused"] = np.zeros(len(train))
    
    return train_frame


def _run_final_fusion(
    present: Dict[str, np.ndarray],
    weights: Dict[str, float],
    score: pd.DataFrame,
    frame: pd.DataFrame,
    score_regime_labels: Optional[np.ndarray],
    cfg: Dict[str, Any],
    equip: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Run final fusion and episode detection.
    
    Returns:
        Tuple of (frame with fused column, episodes DataFrame, episode count)
    """
    Console.info("Computing final fusion and detecting episodes...", component="FUSE")
    
    fused, episodes = fuse.combine(
        present, weights, cfg, 
        original_features=score, 
        regime_labels=score_regime_labels
    )
    fused_np = np.asarray(fused, dtype=np.float32).reshape(-1)
    
    if fused_np.shape[0] != len(frame.index):
        raise RuntimeError(f"[FUSE] Fused length {fused_np.shape[0]} != frame length {len(frame.index)}")
    
    frame["fused"] = fused_np
    episode_count = len(episodes)
    Console.info(f"Detected {episode_count} anomaly episodes", component="FUSE")
    
    return frame, episodes, episode_count


def _record_fusion_observability(
    frame: pd.DataFrame,
    episode_count: int,
    equip: str,
) -> None:
    """Record detector scores and episode metrics for Prometheus/Grafana."""
    # Build scores dict from latest row
    fused_col = frame["fused"].to_numpy(copy=False) if "fused" in frame.columns else np.array([0.0])
    detector_scores_dict = {"fused_z": float(fused_col[-1]) if len(fused_col) > 0 else 0.0}
    
    for detector_name in ["ar1_z", "pca_spe_z", "pca_t2_z", "iforest_z", "gmm_z", "omr_z"]:
        if detector_name in frame.columns:
            col_data = frame[detector_name].to_numpy(copy=False)
            detector_scores_dict[detector_name] = float(col_data[-1]) if len(col_data) > 0 else 0.0
    
    record_detector_scores(equip, detector_scores_dict)
    
    if episode_count > 0:
        record_episode(equip, count=episode_count, severity="warning")


'''
    
    new_content = content[:insert_idx] + helper_code + content[insert_idx:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("SUCCESS: Added fusion pipeline helper functions")
    print("  - FusionPipelineResult dataclass")
    print("  - _save_tuning_diagnostics()")
    print("  - _write_fusion_metrics_to_sql()")
    print("  - _calculate_train_fusion()")
    print("  - _run_final_fusion()")
    print("  - _record_fusion_observability()")

if __name__ == "__main__":
    main()
