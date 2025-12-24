#!/usr/bin/env python
"""Add _run_autonomous_tuning helper function to acm_main.py"""
import re

HELPER_CODE = '''

def _run_autonomous_tuning(
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    score_out: Dict[str, Any],
    regime_quality_ok: bool,
    cached_manifest: Optional[Any],
    cfg: Dict[str, Any],
    sql_client: Optional["SQLClient"],
    equip_id: int,
    run_id: str,
    equip: str,
    refit_flag_path: Path,
    SQL_MODE: bool,
) -> List[str]:
    """
    Run autonomous parameter tuning based on model quality assessment.
    
    Evaluates model quality and triggers retraining if needed. Also performs
    auto-tuning of parameters (clip_z, k_sigma, k_max) based on specific issues.
    
    Args:
        frame: Score frame with detector outputs
        episodes: Detected anomaly episodes
        score_out: Score output dict with metrics
        regime_quality_ok: Whether regime model quality is OK
        cached_manifest: Cached model manifest
        cfg: Configuration dictionary
        sql_client: SQL client for writes
        equip_id: Equipment ID
        run_id: Current run ID
        equip: Equipment name
        refit_flag_path: Path for refit flag file
        SQL_MODE: Whether SQL mode is enabled
        
    Returns:
        List of tuning action strings
    """
    from core.observability import Console
    from core.model_evaluation import assess_model_quality
    from core.config_history_writer import log_auto_tune_changes
    
    if not cfg.get("models", {}).get("auto_tune", True):
        return []
    
    tuning_actions: List[str] = []
    
    try:
        # Build regime quality metrics
        regime_quality_metrics = {
            "silhouette": score_out.get("silhouette", 0.0),
            "quality_ok": regime_quality_ok
        }
        
        # Perform full quality assessment
        should_retrain, reasons, quality_report = assess_model_quality(
            scores=frame,
            episodes=episodes,
            regime_quality=regime_quality_metrics,
            cfg=cfg,
            cached_manifest=cached_manifest
        )
        
        # Check additional retrain triggers
        auto_retrain_cfg = cfg.get("models", {}).get("auto_retrain", {})
        if isinstance(auto_retrain_cfg, bool):
            auto_retrain_cfg = {}
        
        # Anomaly rate trigger
        anomaly_rate_trigger = False
        current_anomaly_rate = 0.0
        anomaly_metrics = quality_report.get("metrics", {}).get("anomaly_metrics", {})
        current_anomaly_rate = anomaly_metrics.get("anomaly_rate", 0.0)
        max_anomaly_rate = auto_retrain_cfg.get("max_anomaly_rate", 0.25)
        if current_anomaly_rate > max_anomaly_rate:
            anomaly_rate_trigger = True
            if not should_retrain:
                reasons = []
            reasons.append(f"anomaly_rate={current_anomaly_rate:.2%} > {max_anomaly_rate:.2%}")
            Console.warn(
                f"Anomaly rate {current_anomaly_rate:.2%} exceeds threshold {max_anomaly_rate:.2%}",
                component="RETRAIN-TRIGGER",
                equip=equip,
                anomaly_rate=round(current_anomaly_rate, 4),
                threshold=max_anomaly_rate
            )
        
        # Drift score trigger
        drift_score_trigger = False
        drift_score = quality_report.get("metrics", {}).get("drift_score", 0.0)
        max_drift_score = auto_retrain_cfg.get("max_drift_score", 2.0)
        if drift_score > max_drift_score:
            drift_score_trigger = True
            if not should_retrain and not anomaly_rate_trigger:
                reasons = []
            reasons.append(f"drift_score={drift_score:.2f} > {max_drift_score:.2f}")
            Console.warn(
                f"Drift score {drift_score:.2f} exceeds threshold {max_drift_score:.2f}",
                component="RETRAIN-TRIGGER",
                equip=equip,
                drift_score=round(drift_score, 2),
                threshold=max_drift_score
            )
        
        # Aggregate triggers
        needs_retraining = should_retrain or anomaly_rate_trigger or drift_score_trigger
        
        if not needs_retraining:
            Console.info("Model quality acceptable, no tuning needed", component="AUTO-TUNE")
            return []
        
        Console.warn(
            f"Quality degradation detected: {', '.join(reasons)}",
            component="AUTO-TUNE",
            equip=equip,
            reason_count=len(reasons)
        )
        
        # Auto-tune parameters
        detector_quality = quality_report.get("metrics", {}).get("detector_quality", {})
        regime_metrics = quality_report.get("metrics", {}).get("regime_metrics", {})
        
        # Issue 1: High detector saturation -> Increase clip_z
        tuning_actions.extend(
            _tune_clip_z(detector_quality, cfg)
        )
        
        # Issue 2: High anomaly rate -> Increase k_sigma
        tuning_actions.extend(
            _tune_k_sigma(anomaly_metrics, cfg)
        )
        
        # Issue 3: Low regime quality -> Increase k_max
        tuning_actions.extend(
            _tune_k_max(regime_metrics, cfg)
        )
        
        if tuning_actions:
            Console.info(
                f"Applied {len(tuning_actions)} parameter adjustments: {', '.join(tuning_actions)}",
                component="AUTO-TUNE"
            )
            Console.info("Retraining required on next run to apply changes", component="AUTO-TUNE")
            
            # Log config changes to SQL
            if sql_client and run_id:
                try:
                    trigger_refit_on_tune = auto_retrain_cfg.get("on_tuning_change", False)
                    log_auto_tune_changes(
                        sql_client=sql_client,
                        equip_id=int(equip_id),
                        tuning_actions=tuning_actions,
                        run_id=run_id,
                        trigger_refit=trigger_refit_on_tune
                    )
                    if trigger_refit_on_tune:
                        Console.info(
                            "on_tuning_change=True: refit request created to apply config changes",
                            component="AUTO-TUNE"
                        )
                except Exception as log_err:
                    Console.warn(
                        f"Failed to log auto-tune changes: {log_err}",
                        component="CONFIG_HIST",
                        equip=equip,
                        error=str(log_err)[:200]
                    )
        else:
            Console.info("No automatic parameter adjustments available", component="AUTO-TUNE")
        
        # Persist refit marker/request
        _persist_refit_request(
            reasons=reasons,
            anomaly_rate_trigger=anomaly_rate_trigger,
            current_anomaly_rate=current_anomaly_rate,
            drift_score_trigger=drift_score_trigger,
            drift_score=drift_score,
            regime_metrics=regime_metrics,
            refit_flag_path=refit_flag_path,
            sql_client=sql_client,
            equip_id=equip_id,
            equip=equip,
            SQL_MODE=SQL_MODE,
        )
        
    except Exception as e:
        Console.warn(
            f"Autonomous tuning failed: {e}",
            component="AUTO-TUNE",
            equip=equip,
            error_type=type(e).__name__,
            error=str(e)[:200]
        )
    
    return tuning_actions


def _tune_clip_z(detector_quality: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    """Tune clip_z parameter based on detector saturation."""
    from core.observability import Console
    
    if detector_quality.get("max_saturation_pct", 0) <= 5.0:
        return []
    
    self_tune_cfg = cfg.get("thresholds", {}).get("self_tune", {})
    try:
        current_clip_z = float(self_tune_cfg.get("clip_z", 12.0))
    except (TypeError, ValueError):
        current_clip_z = 12.0
    
    # Calculate clip cap
    clip_caps = [
        self_tune_cfg.get("max_clip_z"),
        cfg.get("model_quality", {}).get("max_clip_z"),
        50.0,
    ]
    clip_cap = max(
        (float(c) for c in clip_caps if c is not None),
        default=max(current_clip_z, 20.0)
    )
    
    proposed_clip = round(current_clip_z * 1.2, 2)
    if proposed_clip <= current_clip_z + 0.05:
        proposed_clip = current_clip_z + 2.0
    
    new_clip_z = min(proposed_clip, clip_cap)
    
    if new_clip_z > current_clip_z + 0.05:
        return [f"thresholds.self_tune.clip_z: {current_clip_z:.2f}->{new_clip_z:.2f}"]
    elif current_clip_z >= clip_cap - 0.05:
        Console.warn(
            f"[AUTO-TUNE] Clip_z already at ceiling {clip_cap:.2f} while saturation is {detector_quality.get('max_saturation_pct'):.1f}%"
        )
    else:
        Console.info("Clip limit already near target, no change applied", component="AUTO-TUNE")
    
    return []


def _tune_k_sigma(anomaly_metrics: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    """Tune k_sigma parameter based on anomaly rate."""
    from core.observability import Console
    
    if anomaly_metrics.get("anomaly_rate", 0) <= 0.10:
        return []
    
    try:
        current_k = float(cfg.get("episodes", {}).get("cpd", {}).get("k_sigma", 2.0))
    except (TypeError, ValueError):
        current_k = 2.0
    
    new_k = min(round(current_k * 1.1, 3), 4.0)
    
    if new_k > current_k + 0.05:
        return [f"episodes.cpd.k_sigma: {current_k:.3f}->{new_k:.3f}"]
    
    Console.info("k_sigma already increased recently, skipping change", component="AUTO-TUNE")
    return []


def _tune_k_max(regime_metrics: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    """Tune k_max parameter based on regime quality."""
    from core.observability import Console
    
    if regime_metrics.get("silhouette", 1.0) >= 0.15:
        return []
    
    auto_k_cfg = cfg.get("regimes", {}).get("auto_k", {})
    try:
        current_k_max = int(auto_k_cfg.get("k_max", cfg.get("regimes", {}).get("k_max", 8)))
    except (TypeError, ValueError):
        current_k_max = 8
    
    new_k_max = min(current_k_max + 2, 12)
    
    if new_k_max > current_k_max:
        return [f"regimes.auto_k.k_max: {current_k_max}->{int(new_k_max)}"]
    
    Console.info("k_max already at configured ceiling, no change applied", component="AUTO-TUNE")
    return []


def _persist_refit_request(
    reasons: List[str],
    anomaly_rate_trigger: bool,
    current_anomaly_rate: float,
    drift_score_trigger: bool,
    drift_score: float,
    regime_metrics: Dict[str, Any],
    refit_flag_path: Path,
    sql_client: Optional["SQLClient"],
    equip_id: int,
    equip: str,
    SQL_MODE: bool,
) -> None:
    """Persist refit request to file or SQL."""
    from core.observability import Console
    
    try:
        if not SQL_MODE:
            # File mode: atomic write
            tmp_path = refit_flag_path.with_suffix(".pending")
            with tmp_path.open("w", encoding="utf-8") as rf:
                rf.write(f"requested_at={pd.Timestamp.now().isoformat()}\\n")
                rf.write(f"reasons={'; '.join(reasons)}\\n")
                if anomaly_rate_trigger:
                    rf.write(f"anomaly_rate={current_anomaly_rate:.2%}\\n")
                if drift_score_trigger:
                    rf.write(f"drift_score={drift_score:.2f}\\n")
            try:
                os.replace(tmp_path, refit_flag_path)
            except Exception:
                tmp_path.rename(refit_flag_path)
            Console.info(f"Refit flag written atomically -> {refit_flag_path}", component="MODEL")
        else:
            # SQL mode
            if not sql_client:
                Console.warn("SQL client unavailable; cannot write refit request", component="MODEL", equip=equip)
                return
            
            try:
                with sql_client.cursor() as cur:
                    cur.execute(
                        """
                        IF NOT EXISTS (SELECT 1 FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[ACM_RefitRequests]') AND type in (N'U'))
                        BEGIN
                            CREATE TABLE [dbo].[ACM_RefitRequests] (
                                [RequestID] INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
                                [EquipID] INT NOT NULL,
                                [RequestedAt] DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
                                [Reason] NVARCHAR(MAX) NULL,
                                [AnomalyRate] FLOAT NULL,
                                [DriftScore] FLOAT NULL,
                                [ModelAgeHours] FLOAT NULL,
                                [RegimeQuality] FLOAT NULL,
                                [Acknowledged] BIT NOT NULL DEFAULT 0,
                                [AcknowledgedAt] DATETIME2 NULL
                            );
                            CREATE INDEX [IX_RefitRequests_EquipID_Ack] ON [dbo].[ACM_RefitRequests]([EquipID], [Acknowledged]);
                        END
                        """
                    )
                    cur.execute(
                        """
                        INSERT INTO [dbo].[ACM_RefitRequests]
                            (EquipID, Reason, AnomalyRate, DriftScore, RegimeQuality)
                        VALUES
                            (?, ?, ?, ?, ?)
                        """,
                        (
                            int(equip_id),
                            "; ".join(reasons),
                            float(current_anomaly_rate) if anomaly_rate_trigger else None,
                            float(drift_score) if drift_score_trigger else None,
                            float(regime_metrics.get("silhouette", 0.0)),
                        ),
                    )
                Console.info("SQL refit request recorded in ACM_RefitRequests", component="MODEL")
            except Exception as sql_re:
                Console.warn(
                    f"Failed to write SQL refit request: {sql_re}",
                    component="MODEL",
                    equip=equip,
                    error=str(sql_re)[:200]
                )
    except Exception as re:
        Console.warn(
            f"Failed to write refit flag: {re}",
            component="MODEL",
            equip=equip,
            error=str(re)[:200]
        )

'''

def main():
    filepath = "core/acm_main.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the insertion point - after _write_culprits_jsonl function
    pattern = r'(def _write_culprits_jsonl\([\s\S]*?Console\.warn\(f"Failed to write culprits\.jsonl: \{ce\}".*?\n)'
    match = re.search(pattern, content)
    
    if not match:
        print("ERROR: Could not find _write_culprits_jsonl function")
        return
    
    insert_pos = match.end()
    
    # Check if helper already exists
    if "def _run_autonomous_tuning" in content:
        print("WARNING: _run_autonomous_tuning already exists")
        return
    
    # Insert the helper
    new_content = content[:insert_pos] + HELPER_CODE + content[insert_pos:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("SUCCESS: Added _run_autonomous_tuning and related helper functions")

if __name__ == "__main__":
    main()
