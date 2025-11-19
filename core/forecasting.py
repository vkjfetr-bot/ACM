"""
Unified Forecasting Module
===========================

Single source of truth for all ACM forecasting capabilities:
- AR(1) baseline detector for per-sensor residual analysis
- Enhanced multi-model forecasting with SQL integration
- RUL estimation and failure probability calculation

Replaces legacy modules:
- forecast.py (AR1Detector)
- enhanced_forecasting.py (file-based engine)
- enhanced_forecasting_sql.py (SQL wrapper)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd

from utils.logger import Console  # type: ignore
from core import rul_estimator  # type: ignore

# Temporary import for file-based enhanced forecasting until fully migrated
try:
    from core import enhanced_forecasting_deprecated as _enhanced_forecasting
    # Alias for backward compatibility
    EnhancedForecastingEngine = _enhanced_forecasting.EnhancedForecastingEngine
except ImportError:
    Console.warn("[FORECASTING] enhanced_forecasting_deprecated not found; file-mode forecasting unavailable")
    EnhancedForecastingEngine = None  # type: ignore


# ============================================================================
# AR(1) Baseline Detector
# ============================================================================

class AR1Detector:
    """
    Per-sensor AR(1) baseline model for residual scoring.
    
    Calculates AR(1) coefficients (phi) and mean (mu) for each sensor.
    Scores new data by calculating the absolute z-score of the residuals, 
    normalized by the TRAIN-time residual standard deviation.
    
    Usage:
        detector = AR1Detector(ar1_cfg={})
        detector.fit(train_df)
        scores = detector.score(test_df)
    """
    
    def __init__(self, ar1_cfg: Dict[str, Any] | None = None):
        """
        Initialize the AR(1) detector.
        
        Args:
            ar1_cfg: Configuration dict with optional keys:
                - eps (float): Numeric stability epsilon (default: 1e-9)
                - phi_cap (float): Max absolute phi value (default: 0.999)
                - sd_floor (float): Min std dev (default: 1e-6)
                - fuse (str): Fusion strategy "mean"|"median"|"p95" (default: "mean")
        """
        self.cfg = ar1_cfg or {}
        self._eps: float = float(self.cfg.get("eps", 1e-9))
        self._phi_cap: float = float(self.cfg.get("phi_cap", 0.999))
        self._sd_floor: float = float(self.cfg.get("sd_floor", 1e-6))
        self._fuse: Literal["mean", "median", "p95"] = self.cfg.get("fuse", "mean")
        
        # Trained parameters per column: (phi, mu)
        self.phimap: Dict[str, Tuple[float, float]] = {}
        # TRAIN residual std per column for normalization
        self.sdmap: Dict[str, float] = {}
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> "AR1Detector":
        """
        Fit the AR(1) model for each column in the training data.
        
        Args:
            X: Training feature matrix
            
        Returns:
            self for chaining
        """
        self.phimap = {}
        self.sdmap = {}
        
        if not isinstance(X, pd.DataFrame) or X.shape[0] == 0:
            self._is_fitted = True
            return self
        
        for c in X.columns:
            col = X[c].to_numpy(copy=False, dtype=np.float32)
            finite = np.isfinite(col)
            x = col[finite]
            
            if x.size < 3:
                mu = float(np.nanmean(col)) if x.size else 0.0
                if not np.isfinite(mu):
                    mu = 0.0
                phi = 0.0
                self.phimap[c] = (phi, mu)
                resid = (x - mu) if x.size else np.array([0.0], dtype=np.float32)
                sd = float(np.std(resid)) if resid.size else self._sd_floor
                self.sdmap[c] = max(sd, self._sd_floor)
                continue
            
            mu = float(np.nanmean(x))
            if not np.isfinite(mu):
                mu = 0.0
            xc = x - mu
            var_xc = float(np.var(xc)) if xc.size else 0.0
            phi = 0.0
            
            if np.isfinite(var_xc) and var_xc >= 1e-8:
                num = float(np.dot(xc[1:], xc[:-1]))
                den = float(np.dot(xc[:-1], xc[:-1]))
                if abs(den) >= 1e-9:
                    phi = num / den
            else:
                Console.warn(f"[AR1] Column '{c}': near-constant signal; using phi=0")
            
            if abs(phi) > self._phi_cap:
                original_phi = phi
                phi = float(np.sign(phi) * self._phi_cap)
                Console.warn(f"[AR1] Column '{c}': phi={original_phi:.3f} clamped to {phi:.3f}")
            
            if len(x) < 20:
                Console.warn(f"[AR1] Column '{c}': only {len(x)} samples; coefficients may be unstable")
            
            self.phimap[c] = (phi, mu)
            
            # Compute TRAIN residuals & std for normalization during score()
            x_shift = np.empty_like(x, dtype=np.float32)
            x_shift[0] = mu
            x_shift[1:] = x[:-1]
            pred = (x_shift - mu) * phi + mu
            resid = x - pred
            resid_for_sd = resid[1:] if resid.size > 1 else resid
            sd = float(np.std(resid_for_sd))
            self.sdmap[c] = max(sd, self._sd_floor)
        
        self._is_fitted = True
        return self
    
    def score(self, X: pd.DataFrame, return_per_sensor: bool = False) -> np.ndarray | Tuple[np.ndarray, pd.DataFrame]:
        """
        Calculate absolute z-scores of residuals using TRAIN-time residual std.
        
        Args:
            X: Scoring feature matrix
            return_per_sensor: If True, also return DataFrame of per-sensor |z|
            
        Returns:
            Fused absolute z-scores (len == len(X))
            Optionally: (fused_scores, per_sensor_df) when return_per_sensor=True
        """
        if not self._is_fitted:
            return np.zeros(len(X), dtype=np.float32)
        
        per_cols: Dict[str, np.ndarray] = {}
        n = len(X)
        
        if n == 0 or X.shape[1] == 0:
            return (np.zeros(0, dtype=np.float32), pd.DataFrame(index=X.index)) if return_per_sensor else np.zeros(0, dtype=np.float32)
        
        for c in X.columns:
            series = X[c].to_numpy(copy=False, dtype=np.float32)
            ph, mu = self.phimap.get(c, (0.0, float(np.nanmean(series))))
            if not np.isfinite(mu):
                mu = 0.0
            
            sd_train = self.sdmap.get(c, self._sd_floor)
            if not np.isfinite(sd_train) or sd_train <= self._sd_floor:
                sd_train = self._sd_floor
            
            # Impute NaNs to mu for prediction path
            series_finite = series.copy()
            if np.isnan(series_finite).any():
                series_finite = np.where(np.isfinite(series_finite), series_finite, mu).astype(np.float32, copy=False)
            
            # One-step AR(1) prediction
            pred = np.empty_like(series_finite, dtype=np.float32)
            first_obs = series_finite[0] if series_finite.size else mu
            pred[0] = first_obs if np.isfinite(first_obs) else mu
            if n > 1:
                pred[1:] = (series_finite[:-1] - mu) * ph + mu
            
            resid = series - pred  # Keep NaNs where original series had NaNs
            z = np.abs(resid) / sd_train
            per_cols[c] = z.astype(np.float32, copy=False)
        
        if not per_cols:
            return (np.zeros(n, dtype=np.float32), pd.DataFrame(index=X.index)) if return_per_sensor else np.zeros(n, dtype=np.float32)
        
        col_names = list(per_cols.keys())
        matrix = np.column_stack([per_cols[name] for name in col_names]) if col_names else np.zeros((n, 0), dtype=np.float32)
        
        with np.errstate(all="ignore"):
            if self._fuse == "median":
                fused = np.nanmedian(matrix, axis=1).astype(np.float32)
            elif self._fuse == "p95":
                fused = np.nanpercentile(matrix, 95, axis=1).astype(np.float32)
            else:
                fused = np.nanmean(matrix, axis=1).astype(np.float32)
        
        if return_per_sensor:
            Z = pd.DataFrame({name: matrix[:, i] for i, name in enumerate(col_names)}, index=X.index)
            return fused, Z
        return fused
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize detector state for persistence."""
        return {"phimap": self.phimap, "sdmap": self.sdmap, "cfg": self.cfg}
    
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AR1Detector":
        """Deserialize detector state from dict."""
        inst = cls(payload.get("cfg"))
        inst.phimap = dict(payload.get("phimap", {}))
        inst.sdmap = dict(payload.get("sdmap", {}))
        inst._is_fitted = True
        return inst


# ============================================================================
# Enhanced Forecasting (SQL-backed)
# ============================================================================


def run_enhanced_forecasting_sql(
    sql_client: Any,
    equip_id: Optional[int],
    run_id: Optional[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    SQL-only entrypoint for enhanced forecasting.

    This mirrors EnhancedForecastingEngine.run(...) but:
    - Loads inputs from SQL (ACM_HealthTimeline, ACM_Scores_Wide)
    - Does NOT write any files or create directories
    - Returns in-memory tables/metrics for the caller to persist to SQL

    Returns:
        {
          "tables": {
             "failure_probability_ts": DataFrame,
             "failure_causation": DataFrame,
             "enhanced_maintenance_recommendation": DataFrame,
             "recommended_actions": DataFrame,
          },
          "metrics": {...}
        }
    """
    if sql_client is None:
        Console.warn("[ENHANCED_FORECAST] SQL client not provided; skipping enhanced forecasting")
        return {"tables": {}, "metrics": {}}

    if equip_id is None or not run_id:
        Console.warn("[ENHANCED_FORECAST] Missing EquipID/RunID; skipping enhanced forecasting")
        return {"tables": {}, "metrics": {}}

    engine = enhanced_forecasting.EnhancedForecastingEngine(config)
    if not engine.forecast_config.enabled:
        Console.info("[ENHANCED_FORECAST] Module disabled via config.forecasting.enabled")
        return {"tables": {}, "metrics": {}}

    # --- Cleanup old forecast data to prevent RunID overlap in charts ---
    try:
        import os
        try:
            keep_runs = int(os.getenv("ACM_FORECAST_RUNS_RETAIN", "2"))
        except Exception:
            keep_runs = 2
        keep_runs = max(1, min(int(keep_runs), 50))
        cur = sql_client.cursor()
        # Keep only the 2 most recent RunIDs to preserve some history while reducing clutter
        cur.execute("""
            WITH RankedRuns AS (
                SELECT DISTINCT RunID, 
                       ROW_NUMBER() OVER (ORDER BY MAX(CreatedAt) DESC) AS rn
                FROM dbo.ACM_HealthForecast_TS
                WHERE EquipID = ?
                GROUP BY RunID
            )
            DELETE FROM dbo.ACM_HealthForecast_TS
            WHERE EquipID = ? 
              AND RunID IN (SELECT RunID FROM RankedRuns WHERE rn > ?)
        """, (equip_id, equip_id, keep_runs))
        
        cur.execute("""
            WITH RankedRuns AS (
                SELECT DISTINCT RunID, 
                       ROW_NUMBER() OVER (ORDER BY MAX(CreatedAt) DESC) AS rn
                FROM dbo.ACM_FailureForecast_TS
                WHERE EquipID = ?
                GROUP BY RunID
            )
            DELETE FROM dbo.ACM_FailureForecast_TS
            WHERE EquipID = ? 
              AND RunID IN (SELECT RunID FROM RankedRuns WHERE rn > ?)
        """, (equip_id, equip_id, keep_runs))
        
        if not sql_client.conn.autocommit:
            sql_client.conn.commit()
        Console.info(f"[ENHANCED_FORECAST] Cleaned old forecast data for EquipID={equip_id} (kept {keep_runs} RunIDs)")
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Failed to cleanup old forecasts: {e}")
        # Non-fatal, continue with forecasting

    # --- Load health timeline from SQL (reuse rul_estimator helper) ---
    try:
        df_health = rul_estimator._load_health_timeline(  # type: ignore[attr-defined]
            Path("."),
            sql_client=sql_client,
            equip_id=equip_id,
            run_id=str(run_id),
        )
    except Exception as e:  # pragma: no cover - defensive
        Console.warn(f"[ENHANCED_FORECAST] Failed to load health timeline via rul_estimator: {e}")
        df_health = None

    if df_health is None:
        Console.warn("[ENHANCED_FORECAST] No health timeline available from SQL; skipping")
        return {"tables": {}, "metrics": {}}

    if "HealthIndex" not in df_health.columns or "Timestamp" not in df_health.columns:
        Console.warn("[ENHANCED_FORECAST] Health timeline missing required columns; skipping")
        return {"tables": {}, "metrics": {}}

    try:
        ts = pd.to_datetime(df_health["Timestamp"], utc=True)
        hi = pd.Series(df_health["HealthIndex"].astype(float).to_numpy(), index=ts)
        hi = hi.sort_index()
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Failed to prepare health series: {e}")
        return {"tables": {}, "metrics": {}}

    if hi.size < 20:
        Console.warn(f"[ENHANCED_FORECAST] Insufficient health history ({hi.size} points); skipping")
        return {"tables": {}, "metrics": {}}

    # --- Load detector scores from ACM_Scores_Wide ---
    try:
        cur = sql_client.cursor()
        cur.execute(
            """
            SELECT Timestamp,
                   ar1_z, pca_spe_z, pca_t2_z, mhal_z,
                   iforest_z, gmm_z, cusum_z, drift_z,
                   hst_z, river_hst_z, fused
            FROM dbo.ACM_Scores_Wide
            WHERE EquipID = ? AND RunID = ?
            ORDER BY Timestamp
            """,
            (equip_id, run_id),
        )
        rows = cur.fetchall() or []
        cur.close()
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Failed to load detector scores from SQL: {e}")
        rows = []

    if not rows:
        Console.warn("[ENHANCED_FORECAST] No detector scores in ACM_Scores_Wide; skipping enhanced analysis")
        return {"tables": {}, "metrics": {}}

    cols = [
        "Timestamp",
        "ar1_z",
        "pca_spe_z",
        "pca_t2_z",
        "mhal_z",
        "iforest_z",
        "gmm_z",
        "cusum_z",
        "drift_z",
        "hst_z",
        "river_hst_z",
        "fused",
    ]
    df_scores = pd.DataFrame.from_records(rows, columns=cols)
    df_scores["Timestamp"] = pd.to_datetime(df_scores["Timestamp"], errors="coerce", utc=True)
    df_scores = df_scores.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    df_scores = df_scores.set_index("Timestamp")

    if df_scores.empty:
        Console.warn("[ENHANCED_FORECAST] Detector scores dataframe empty after cleaning; skipping")
        return {"tables": {}, "metrics": {}}

    # --- Core enhanced forecasting logic (mirrors EnhancedForecastingEngine.run) ---
    try:
        import traceback
        forecast_result = engine.forecaster.forecast(
            health_history=hi,
            horizons=engine.forecast_config.forecast_horizons,
        )
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Forecasting failed: {e}")
        Console.debug(f"[ENHANCED_FORECAST] Traceback: {traceback.format_exc()}")
        return {"tables": {}, "metrics": {}}

    try:
        failure_probs_df = engine.prob_calculator.compute_probabilities(
            forecast_result,
            engine.forecast_config.failure_threshold,
        )
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Failure probability computation failed: {e}")
        return {"tables": {}, "metrics": {}}

    try:
        rul_hours = engine._estimate_rul(  # type: ignore[attr-defined]
            forecast_result,
            engine.forecast_config.failure_threshold,
        )
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] RUL estimation failed: {e}")
        return {"tables": {}, "metrics": {}}

    try:
        predicted_failure_time = engine._get_failure_time(  # type: ignore[attr-defined]
            hi.index[-1],
            rul_hours,
        )
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Failed to derive predicted failure time: {e}")
        return {"tables": {}, "metrics": {}}

    try:
        causation_df, failure_patterns = engine.causation_analyzer.analyze_causation(
            df_scores,
            predicted_failure_time,
        )
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Causation analysis failed: {e}")
        causation_df, failure_patterns = pd.DataFrame(), []

    try:
        maintenance_rec = engine.maintenance_recommender.generate_recommendation(
            failure_probs_df,
            causation_df,
            failure_patterns,
            rul_hours,
        )
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Maintenance recommendation failed: {e}")
        maintenance_rec = None

    tables: Dict[str, pd.DataFrame] = {}

    now_ts = hi.index[-1]

    if not failure_probs_df.empty:
        fp_df = failure_probs_df.copy()
        fp_df.insert(0, "Timestamp", now_ts)
        fp_df.insert(0, "EquipID", int(equip_id))
        fp_df.insert(0, "RunID", str(run_id))
        tables["failure_probability_ts"] = fp_df

    if causation_df is not None and not causation_df.empty and maintenance_rec is not None:
        fc_df = causation_df.copy()
        fc_df.insert(0, "PredictedFailureTime", predicted_failure_time)
        fc_df.insert(1, "FailurePattern", ",".join(maintenance_rec.get("failure_patterns", [])))
        fc_df.insert(0, "EquipID", int(equip_id))
        fc_df.insert(0, "RunID", str(run_id))
        tables["failure_causation"] = fc_df

    if maintenance_rec is not None:
        maint_df = pd.DataFrame(
            [
                {
                    "UrgencyScore": maintenance_rec.get("urgency_score"),
                    "MaintenanceRequired": maintenance_rec.get("maintenance_required"),
                    "EarliestMaintenance": maintenance_rec.get("window", {}).get("earliest_maintenance"),
                    "PreferredWindowStart": maintenance_rec.get("window", {}).get("preferred_window_start"),
                    "PreferredWindowEnd": maintenance_rec.get("window", {}).get("preferred_window_end"),
                    "LatestSafeTime": maintenance_rec.get("window", {}).get("latest_safe_time"),
                    "FailureProbAtLatest": maintenance_rec.get("window", {}).get("failure_prob_at_latest"),
                    "FailurePattern": ",".join(maintenance_rec.get("failure_patterns", [])),
                    "Confidence": maintenance_rec.get("confidence"),
                    "EstimatedDuration_Hours": sum(
                        a.get("estimated_duration_hours", 0.0)
                        for a in maintenance_rec.get("recommended_actions", [])
                    ),
                }
            ]
        )
        maint_df.insert(0, "EquipID", int(equip_id))
        maint_df.insert(0, "RunID", str(run_id))
        tables["enhanced_maintenance_recommendation"] = maint_df

        actions = maintenance_rec.get("recommended_actions", [])
        if actions:
            actions_df = pd.DataFrame(actions)
            actions_df.insert(0, "EquipID", int(equip_id))
            actions_df.insert(0, "RunID", str(run_id))
            tables["recommended_actions"] = actions_df

        metrics = {
            "rul_hours": float(rul_hours),
            "max_failure_probability": float(
                failure_probs_df["FailureProbability"].max()
            )
            if not failure_probs_df.empty
            else 0.0,
            "maintenance_required": bool(maintenance_rec.get("maintenance_required", False)),
            "urgency_score": float(maintenance_rec.get("urgency_score", 0.0)),
            "confidence": float(maintenance_rec.get("confidence", 0.0)),
        }
    else:
        metrics = {
            "rul_hours": float(rul_hours),
            "max_failure_probability": float(
                failure_probs_df["FailureProbability"].max()
            )
            if not failure_probs_df.empty
            else 0.0,
            "maintenance_required": False,
            "urgency_score": 0.0,
            "confidence": 0.0,
        }

    Console.info(
        f"[ENHANCED_FORECAST] RUL={metrics['rul_hours']:.1f}h, "
        f"MaxFailProb={metrics['max_failure_probability']*100:.1f}%, "
        f"MaintenanceRequired={metrics['maintenance_required']}, "
        f"Urgency={metrics['urgency_score']:.0f}/100"
    )

    return {"tables": tables, "metrics": metrics}

