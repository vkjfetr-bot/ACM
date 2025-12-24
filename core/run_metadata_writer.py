"""
ACM Run Metadata Writer

Writes comprehensive run-level metadata to ACM_Runs table for:
- Run tracking and auditing
- Performance monitoring
- Quality assessment
- Refit scheduling

Called at the end of every ACM run (success or failure).
"""

from datetime import datetime, timezone
from typing import Optional
import pandas as pd
from core.observability import Console


def write_run_metadata(
    sql_client,
    run_id: str,
    equip_id: int,
    equip_name: str,
    started_at: datetime,
    completed_at: datetime,
    config_signature: str,
    train_row_count: int,
    score_row_count: int,
    episode_count: int,
    health_status: str,
    avg_health_index: float,
    min_health_index: float,
    max_fused_z: float,
    data_quality_score: float,
    refit_requested: bool,
    kept_columns: str,
    error_message: Optional[str] = None
) -> bool:
    """
    Write run metadata to ACM_Runs table.
    
    Args:
        sql_client: SQL connection client
        run_id: Unique run identifier (UUID)
        equip_id: Equipment ID
        equip_name: Equipment name
        started_at: Run start timestamp (UTC)
        completed_at: Run completion timestamp (UTC)
        config_signature: MD5 hash of config for change detection
        train_row_count: Number of training rows processed
        score_row_count: Number of scoring rows processed
        episode_count: Number of anomaly episodes detected
        health_status: Overall health status (HEALTHY, CAUTION, ALERT)
        avg_health_index: Average health index (0-100)
        min_health_index: Minimum health index (0-100)
        max_fused_z: Maximum fused z-score
        data_quality_score: Data quality metric (0-100)
        refit_requested: Whether model refit was requested
        kept_columns: Comma-separated list of sensor columns used
        error_message: Error message if run failed (optional)
    
    Returns:
        bool: True if write succeeded, False otherwise
    """
    
    if sql_client is None:
        Console.warn("No SQL client provided, skipping ACM_Runs write", component="RUN_META", run_id=run_id, equip_id=equip_id)
        return False
    
    try:
        # Calculate duration
        duration_seconds = int((completed_at - started_at).total_seconds())
        
        # Ensure timestamps are UTC naive (SQL datetime2 requirement)
        if started_at.tzinfo is not None:
            started_at = started_at.replace(tzinfo=None)
        if completed_at.tzinfo is not None:
            completed_at = completed_at.replace(tzinfo=None)
        
        # Build UPDATE statement (row already exists from _sql_start_run)
        update_sql = """
        UPDATE dbo.ACM_Runs
        SET EquipName = ?,
            CompletedAt = ?,
            DurationSeconds = ?,
            TrainRowCount = ?,
            ScoreRowCount = ?,
            EpisodeCount = ?,
            HealthStatus = ?,
            AvgHealthIndex = ?,
            MinHealthIndex = ?,
            MaxFusedZ = ?,
            DataQualityScore = ?,
            RefitRequested = ?,
            KeptColumns = ?,
            ErrorMessage = ?
        WHERE RunID = ?
        """
        
        # Prepare record (note: RunID is last for WHERE clause)
        record = (
            equip_name,
            completed_at,
            duration_seconds,
            train_row_count,
            score_row_count,
            episode_count,
            health_status,
            float(avg_health_index) if avg_health_index is not None else None,
            float(min_health_index) if min_health_index is not None else None,
            float(max_fused_z) if max_fused_z is not None else None,
            float(data_quality_score) if data_quality_score is not None else None,
            refit_requested,
            kept_columns,
            error_message,
            run_id
        )
        
        # Execute update
        with sql_client.cursor() as cur:
            cur.execute(update_sql, record)
        
        # Commit
        sql_client.conn.commit()
        
        Console.info(f"Wrote run metadata to ACM_Runs: {run_id}", component="RUN_META")
        return True
        
    except Exception as e:
        Console.error(f"Failed to write ACM_Runs: {e}", component="RUN_META", run_id=run_id, equip_id=equip_id, equip_name=equip_name, error_type=type(e).__name__, error=str(e)[:200])
        try:
            sql_client.conn.rollback()
        except:
            pass
        return False


def compute_run_health_status(avg_health: float, min_health: float) -> str:
    """
    Determine overall run health status based on health metrics.
    
    Args:
        avg_health: Average health index (0-100)
        min_health: Minimum health index (0-100)
    
    Returns:
        str: "HEALTHY", "CAUTION", or "ALERT"
    """
    # Alert if minimum health is critically low
    if min_health < 50:
        return "ALERT"
    
    # Alert if average health is low
    if avg_health < 70:
        return "ALERT"
    
    # Caution if minimum health is borderline
    if min_health < 70:
        return "CAUTION"
    
    # Caution if average health is moderate
    if avg_health < 90:
        return "CAUTION"
    
    # Healthy
    return "HEALTHY"


def extract_run_metadata_from_scores(scores: pd.DataFrame, per_regime_enabled: bool = False, regime_count: int = 0) -> dict:
    """
    Extract health and quality metrics from scores dataframe.
    
    Args:
        scores: Scores dataframe with fused z-scores and precomputed health
        per_regime_enabled: Whether per-regime calibration was enabled (DET-07)
        regime_count: Number of regimes detected
    
    Returns:
        dict: Metadata including health metrics and calibration info
    """
    import numpy as np
    metadata = {}
    
    try:
        # Use precomputed health if available
        if "__health" in scores.columns:
            health = scores["__health"]
        else:
            # v10.1.0: Fallback uses softer sigmoid formula
            # OLD: 100/(1+Z^2) was too aggressive
            z_threshold = 5.0
            steepness = 1.5
            abs_z = np.abs(scores["fused"])
            normalized = (abs_z - z_threshold / 2) / (z_threshold / 4)
            sigmoid = 1 / (1 + np.exp(-normalized * steepness))
            health = np.clip(100.0 * (1 - sigmoid), 0.0, 100.0)
        
        metadata["avg_health_index"] = float(health.mean())
        metadata["min_health_index"] = float(health.min())
        metadata["max_fused_z"] = float(scores["fused"].abs().max())
        
        # Health status
        metadata["health_status"] = compute_run_health_status(
            metadata["avg_health_index"],
            metadata["min_health_index"]
        )
        
        # DET-07: Add per-regime calibration info
        metadata["per_regime_enabled"] = per_regime_enabled
        metadata["regime_count"] = regime_count
        
    except Exception as e:
        Console.warn(f"Failed to extract health metrics: {e}", component="RUN_META", error_type=type(e).__name__, error=str(e)[:200])
        metadata["avg_health_index"] = None
        metadata["min_health_index"] = None
        metadata["max_fused_z"] = None
        metadata["health_status"] = "UNKNOWN"
        metadata["per_regime_enabled"] = False
        metadata["regime_count"] = 0
    
    return metadata


def extract_data_quality_score(data_quality_path=None, sql_client=None, run_id=None, equip_id=None) -> float:
    """
    Extract overall data quality score from data_quality.csv or ACM_DataQuality SQL table.
    
    OM-CSV-03: Updated to support SQL mode - queries ACM_DataQuality if sql_client provided,
    falls back to CSV if data_quality_path exists.
    
    RM-COR-01: Validates expected schema columns and logs missing fields
    to help diagnose schema drift or incomplete data quality metrics.
    
    Args:
        data_quality_path: Path to data_quality.csv file (file mode)
        sql_client: SQL client for database queries (SQL mode)
        run_id: RunID to query from SQL (SQL mode)
        equip_id: EquipID to query from SQL (SQL mode)
    
    Returns:
        float: Quality score (0-100), or 100.0 if no data found
    """
    try:
        import pandas as pd
        from pathlib import Path
        
        # OM-CSV-03: SQL mode - query ACM_DataQuality
        if sql_client and run_id:
            try:
                query = """
                    SELECT sensor, train_null_pct, score_null_pct
                    FROM dbo.ACM_DataQuality
                    WHERE RunID = ? AND EquipID = ? AND CheckName = 'data_quality'
                """
                with sql_client.cursor() as cur:
                    cur.execute(query, (run_id, int(equip_id or 0)))
                    rows = cur.fetchall()
                
                if rows:
                    # Fix: Convert pyodbc.Row objects to tuples to ensure pandas infers shape correctly
                    rows = [tuple(r) for r in rows]
                    df = pd.DataFrame(rows, columns=["sensor", "train_null_pct", "score_null_pct"])
                    # Average null rate across train and score
                    avg_null_pct = (df["train_null_pct"].mean() + df["score_null_pct"].mean()) / 2.0
                    quality_score = 100.0 * (1.0 - avg_null_pct / 100.0)
                    Console.debug(f"Data quality from SQL: avg_null={avg_null_pct:.2f}%, score={quality_score:.1f}", component="RUN_META")
                    return float(quality_score)
                else:
                    Console.debug("No data quality records found in SQL, defaulting to 100.0", component="RUN_META")
                    return 100.0
            except Exception as e:
                Console.warn(f"Failed to query ACM_DataQuality: {e}, falling back to CSV", component="RUN_META", run_id=run_id, equip_id=equip_id, error_type=type(e).__name__)
        
        # File mode: read from CSV
        if data_quality_path and Path(data_quality_path).exists():
            df = pd.read_csv(data_quality_path)
            
            # RM-COR-01: Schema validation - check for expected columns
            expected_columns = {"sensor_name", "null_rate", "constant_rate", "outlier_rate"}
            actual_columns = set(df.columns)
            missing_columns = expected_columns - actual_columns
            
            if missing_columns:
                Console.warn(
                    f"[RUN_META] Data quality schema incomplete: missing columns {sorted(missing_columns)}. "
                    f"Quality score coverage may be reduced."
                )
            
            # Calculate quality score based on null rates
            if "null_rate" in df.columns:
                # 100 - (average null rate across sensors)
                avg_null_rate = df["null_rate"].mean()
                quality_score = 100.0 * (1.0 - avg_null_rate / 100.0)
                
                # Log additional quality metrics if available
                if "constant_rate" in df.columns and "outlier_rate" in df.columns:
                    avg_constant = df["constant_rate"].mean()
                    avg_outlier = df["outlier_rate"].mean()
                    Console.debug(
                        f"[RUN_META] Data quality metrics: null={avg_null_rate:.2f}%, "
                        f"constant={avg_constant:.2f}%, outlier={avg_outlier:.2f}%"
                    )
                
                return float(quality_score)
            else:
                Console.warn(
                    "[RUN_META] Missing 'null_rate' column in data_quality.csv. "
                    "Defaulting to quality score 100.0 (optimistic fallback)."
                )
                return 100.0
        
    except Exception as e:
        Console.warn(f"Failed to extract data quality score: {e}", component="RUN_META", error_type=type(e).__name__, error=str(e)[:200])
        return 100.0


def write_retrain_metadata(
    sql_client,
    run_id: str,
    equip_id: int,
    equip_name: str,
    retrain_decision: bool,
    retrain_reason: str,
    forecast_state_version: int,
    model_age_batches: Optional[int] = None,
    forecast_rmse: Optional[float] = None,
    forecast_mae: Optional[float] = None,
    forecast_mape: Optional[float] = None,
) -> bool:
    """
    Write forecasting retrain decision + model age + quality metrics to ACM_RunMetadata.

    Args:
        sql_client: Active SQL client (must expose cursor()/conn)
        run_id: Current run unique identifier (UUID string)
        equip_id: Equipment numeric ID
        equip_name: Equipment code/name
        retrain_decision: Whether retraining occurred/is requested this batch
        retrain_reason: Reason string from should_retrain()
        forecast_state_version: Incrementing state version after merge
        model_age_batches: Batches since last retrain (optional if not tracked yet)
        forecast_rmse: Backtest RMSE (optional placeholder)
        forecast_mae: Backtest MAE (optional placeholder)
        forecast_mape: Backtest MAPE (optional placeholder)

    Returns:
        bool: True if insert succeeded.
    """
    if sql_client is None:
        Console.warn("No SQL client; skipping ACM_RunMetadata write", component="RUN_META", run_id=run_id, equip_id=equip_id)
        return False

    try:
        insert_sql = """
        INSERT INTO dbo.ACM_RunMetadata (
            RunID, EquipID, EquipName, CreatedAt,
            RetrainDecision, RetrainReason, ForecastStateVersion,
            ModelAgeBatches, ForecastRMSE, ForecastMAE, ForecastMAPE
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        created_at = datetime.utcnow().replace(tzinfo=None)
        record = (
            run_id,
            int(equip_id),
            equip_name,
            created_at,
            bool(retrain_decision),
            retrain_reason[:250] if retrain_reason else None,
            int(forecast_state_version) if forecast_state_version is not None else None,
            int(model_age_batches) if model_age_batches is not None else None,
            float(forecast_rmse) if forecast_rmse is not None else None,
            float(forecast_mae) if forecast_mae is not None else None,
            float(forecast_mape) if forecast_mape is not None else None,
        )

        with sql_client.cursor() as cur:
            cur.execute(insert_sql, record)
        sql_client.conn.commit()
        Console.info(f"Wrote retrain metadata RunID={run_id} StateV={forecast_state_version}", component="RUN_META")
        return True
    except Exception as e:
        Console.error(f"Failed to write ACM_RunMetadata: {e}", component="RUN_META", run_id=run_id, equip_id=equip_id, equip_name=equip_name, error_type=type(e).__name__, error=str(e)[:200])
        try:
            sql_client.conn.rollback()
        except Exception:
            pass
        return False


def write_timer_stats(
    sql_client,
    run_id: str,
    equip_id: int,
    batch_num: int,
    timings: dict
) -> bool:
    """
    Write detailed timer stats to ACM_RunTimers table.
    
    Args:
        sql_client: Active SQL client
        run_id: Current RunID
        equip_id: Equipment ID
        batch_num: Current batch number
        timings: Dictionary of {section_name: duration_seconds}
        
    Returns:
        bool: True if successful
    """
    if not sql_client or not timings:
        return False
        
    try:
        # Create table if not exists (Lazy migration)
        # In prod, this should be a migration script, but for dev speed we do it here
        create_table_sql = """
        IF OBJECT_ID('dbo.ACM_RunTimers', 'U') IS NULL
        BEGIN
            CREATE TABLE dbo.ACM_RunTimers (
                TimerID INT IDENTITY(1,1) PRIMARY KEY,
                RunID VARCHAR(50) NOT NULL,
                EquipID INT NOT NULL,
                BatchNum INT DEFAULT 0,
                Section VARCHAR(100) NOT NULL,
                DurationSeconds FLOAT NOT NULL,
                CreatedAt DATETIME DEFAULT GETUTCDATE(),
                INDEX IX_ACM_RunTimers_RunID (RunID),
                INDEX IX_ACM_RunTimers_EquipID (EquipID)
            )
        END
        """
        
        insert_sql = """
        INSERT INTO dbo.ACM_RunTimers (RunID, EquipID, BatchNum, Section, DurationSeconds)
        VALUES (?, ?, ?, ?, ?)
        """
        
        # Flatten timings to list of tuples
        # T.timings values might be floats or tuples, usually floats in this system
        rows = []
        for name, duration in timings.items():
            # Handle if duration is complex object (unlikely in simple Timer)
            try:
                val = float(duration)
                rows.append((run_id, int(equip_id), int(batch_num), str(name), val))
            except (ValueError, TypeError):
                continue
                
        if not rows:
            return False
            
        with sql_client.cursor() as cur:
            # check/create table first
            cur.execute(create_table_sql)
            
            # fast insert
            cur.fast_executemany = True
            cur.executemany(insert_sql, rows)
            
        sql_client.conn.commit()
        Console.debug(f"Wrote {len(rows)} timer records to ACM_RunTimers", component="PERF")
        return True
        
    except Exception as e:
        Console.warn(f"Failed to write timer stats: {e}", component="PERF", run_id=run_id, equip_id=equip_id, batch_num=batch_num, timer_count=len(timings), error_type=type(e).__name__, error=str(e)[:200])
        return False
