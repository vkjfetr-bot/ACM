# core/model_lifecycle.py
"""
Model Lifecycle Management for V11
===================================
Tracks model maturity states and promotion criteria.

MaturityState Lifecycle:
    COLDSTART -> LEARNING -> CONVERGED -> DEPRECATED

Promotion Criteria (LEARNING -> CONVERGED):
    - Minimum 7 days of training data
    - Regime silhouette score >= 0.15
    - Stability ratio >= 0.8 (no regime thrashing)
    - At least 3 consecutive successful runs

This module enforces V11 Rule #12: All model changes must be versioned.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List
import json

from core.observability import Console


class MaturityState(str, Enum):
    """Model maturity states for lifecycle tracking."""
    COLDSTART = "COLDSTART"    # Initial state - insufficient data
    LEARNING = "LEARNING"      # Training in progress, not yet reliable
    CONVERGED = "CONVERGED"    # Quality criteria passed, production-ready
    DEPRECATED = "DEPRECATED"  # Superseded by newer version
    
    def __str__(self) -> str:
        return self.value


@dataclass
class PromotionCriteria:
    """Criteria for promoting model from LEARNING to CONVERGED.
    
    P0-FIX (v11.2.2): ANALYTICAL AUDIT FLAW #10 - Tightened promotion criteria
    Previous thresholds were too permissive, allowing unreliable models to reach
    CONVERGED state. New thresholds ensure production-grade reliability.
    
    CHANGES:
    - min_silhouette_score: 0.15 → 0.40 (require decent cluster separation)
    - min_stability_ratio: 0.6 → 0.75 (reduce regime thrashing from 40% to 25%)
    - min_training_rows: 200 → 400 (better statistical significance)
    - min_consecutive_runs: 3 → 5 (more validation before promotion)
    - max_forecast_mape: 50.0 → 35.0 (tighter forecasting accuracy)
    - max_forecast_rmse: 15.0 → 12.0 (tighter error bounds)
    
    FLAW FIX #3: Added forecast quality thresholds. Poor forecasting models
    should not reach CONVERGED state even with good clustering metrics.
    
    These defaults can be overridden via config_table.csv under 'lifecycle' category:
    - lifecycle.promotion.min_training_days
    - lifecycle.promotion.min_silhouette_score
    - lifecycle.promotion.min_stability_ratio
    - lifecycle.promotion.min_consecutive_runs
    - lifecycle.promotion.min_training_rows
    - lifecycle.promotion.max_forecast_mape (NEW)
    - lifecycle.promotion.max_forecast_rmse (NEW)
    
    v11.2.2: Strengthened defaults for production reliability based on analytical audit
    v11.2.1: Added forecast quality criteria
    v11.0.1: Relaxed defaults for faster promotion in industrial settings (now reverted)
    
    Reference:
        MAPE < 35% is good for industrial forecasting (Hyndman 2018, adjusted)
        Silhouette > 0.4 indicates reasonable cluster separation (Rousseeuw 1987)
        RMSE < 12 on 0-100 health scale = good prediction accuracy
    """
    min_training_days: int = 7  # Keep at 7 days (sufficient for weekly patterns)
    min_silhouette_score: float = 0.40  # CHANGED from 0.15 (P0 FIX)
    min_stability_ratio: float = 0.75  # CHANGED from 0.6 (P0 FIX)
    min_consecutive_runs: int = 5  # CHANGED from 3 (P0 FIX)
    min_training_rows: int = 400  # CHANGED from 200 (P0 FIX)
    max_forecast_mape: float = 35.0  # CHANGED from 50.0 (P0 FIX)
    max_forecast_rmse: float = 12.0  # CHANGED from 15.0 (P0 FIX)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "PromotionCriteria":
        """
        Create PromotionCriteria from config dictionary.
        
        Looks for values in cfg['lifecycle']['promotion'] with fallback to defaults.
        
        Args:
            cfg: Configuration dictionary (from ConfigDict)
            
        Returns:
            PromotionCriteria with values from config or defaults
        """
        lifecycle = cfg.get("lifecycle", {}) or {}
        promotion = lifecycle.get("promotion", {}) or {}
        
        return cls(
            min_training_days=int(promotion.get("min_training_days", 7)),
            min_silhouette_score=float(promotion.get("min_silhouette_score", 0.40)),  # v11.2.2: Changed default
            min_stability_ratio=float(promotion.get("min_stability_ratio", 0.75)),  # v11.2.2: Changed default
            min_consecutive_runs=int(promotion.get("min_consecutive_runs", 5)),  # v11.2.2: Changed default
            min_training_rows=int(promotion.get("min_training_rows", 400)),  # v11.2.2: Changed default
            max_forecast_mape=float(promotion.get("max_forecast_mape", 35.0)),  # v11.2.2: Changed default
            max_forecast_rmse=float(promotion.get("max_forecast_rmse", 12.0)),  # v11.2.2: Changed default
        )


@dataclass
class ModelState:
    """Current state of a model version for an equipment."""
    equip_id: int
    version: int
    maturity: MaturityState
    created_at: datetime
    promoted_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    
    # Quality metrics
    silhouette_score: Optional[float] = None
    stability_ratio: Optional[float] = None
    training_rows: int = 0
    training_days: float = 0.0
    consecutive_runs: int = 0
    
    # FLAW FIX #3: Forecast quality metrics
    forecast_mape: Optional[float] = None  # v11.2.1: Mean Absolute Percentage Error
    forecast_rmse: Optional[float] = None  # v11.2.1: Root Mean Square Error
    
    # Run tracking
    last_run_id: Optional[str] = None
    last_run_at: Optional[datetime] = None
    total_runs: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for SQL persistence."""
        return {
            'EquipID': self.equip_id,
            'Version': self.version,
            'MaturityState': str(self.maturity),
            'CreatedAt': self.created_at,
            'PromotedAt': self.promoted_at,
            'DeprecatedAt': self.deprecated_at,
            'SilhouetteScore': self.silhouette_score,
            'StabilityRatio': self.stability_ratio,
            'TrainingRows': self.training_rows,
            'TrainingDays': self.training_days,
            'ConsecutiveRuns': self.consecutive_runs,
            'LastRunID': self.last_run_id,
            'LastRunAt': self.last_run_at,
            'TotalRuns': self.total_runs,
        }


def check_promotion_eligibility(
    state: ModelState,
    criteria: Optional[PromotionCriteria] = None
) -> tuple[bool, List[str]]:
    """
    Check if a model in LEARNING state can be promoted to CONVERGED.
    
    FLAW FIX #3: Added forecast quality checks (MAPE, RMSE).
    Models with poor forecasting cannot be promoted even with good clustering.
    
    Args:
        state: Current model state
        criteria: Promotion criteria (uses defaults if not provided)
        
    Returns:
        Tuple of (eligible: bool, reasons: list of unmet criteria)
    """
    if criteria is None:
        criteria = PromotionCriteria()
    
    unmet: List[str] = []
    
    if state.maturity != MaturityState.LEARNING:
        return False, [f"Not in LEARNING state (current: {state.maturity})"]
    
    if state.training_days < criteria.min_training_days:
        unmet.append(f"training_days={state.training_days:.1f} < {criteria.min_training_days}")
    
    if state.silhouette_score is None or state.silhouette_score < criteria.min_silhouette_score:
        score = state.silhouette_score or 0.0
        unmet.append(f"silhouette={score:.3f} < {criteria.min_silhouette_score}")
    
    if state.stability_ratio is None or state.stability_ratio < criteria.min_stability_ratio:
        ratio = state.stability_ratio or 0.0
        unmet.append(f"stability={ratio:.2f} < {criteria.min_stability_ratio}")
    
    if state.consecutive_runs < criteria.min_consecutive_runs:
        unmet.append(f"consecutive_runs={state.consecutive_runs} < {criteria.min_consecutive_runs}")
    
    if state.training_rows < criteria.min_training_rows:
        unmet.append(f"training_rows={state.training_rows} < {criteria.min_training_rows}")
    
    # FIXED: Check forecast quality metrics (v11.2.1)
    if state.forecast_mape is not None and state.forecast_mape > criteria.max_forecast_mape:
        unmet.append(f"forecast_mape={state.forecast_mape:.1f}% > {criteria.max_forecast_mape}%")
    
    if state.forecast_rmse is not None and state.forecast_rmse > criteria.max_forecast_rmse:
        unmet.append(f"forecast_rmse={state.forecast_rmse:.2f} > {criteria.max_forecast_rmse}")
    
    eligible = len(unmet) == 0
    return eligible, unmet


def promote_model(state: ModelState) -> ModelState:
    """
    Promote a model from LEARNING to CONVERGED.
    
    Args:
        state: Current model state (must be LEARNING)
        
    Returns:
        Updated model state with CONVERGED maturity
        
    Raises:
        ValueError: If model is not in LEARNING state
    """
    if state.maturity != MaturityState.LEARNING:
        raise ValueError(f"Cannot promote model in {state.maturity} state")
    
    state.maturity = MaturityState.CONVERGED
    state.promoted_at = datetime.now()
    
    Console.info(
        f"Model v{state.version} promoted to CONVERGED",
        component="LIFECYCLE",
        equip_id=state.equip_id,
        version=state.version,
        silhouette=state.silhouette_score,
        stability=state.stability_ratio,
    )
    
    return state


def deprecate_model(state: ModelState, reason: str = "") -> ModelState:
    """
    Deprecate a model (superseded by newer version).
    
    Args:
        state: Current model state
        reason: Reason for deprecation
        
    Returns:
        Updated model state with DEPRECATED maturity
    """
    old_maturity = state.maturity
    state.maturity = MaturityState.DEPRECATED
    state.deprecated_at = datetime.now()
    
    Console.info(
        f"Model v{state.version} deprecated ({old_maturity} -> DEPRECATED)",
        component="LIFECYCLE",
        equip_id=state.equip_id,
        version=state.version,
        reason=reason,
    )
    
    return state


def create_new_model_state(
    equip_id: int,
    version: int,
    training_rows: int,
    training_start: datetime,
    training_end: datetime,
    silhouette_score: Optional[float] = None,
    run_id: Optional[str] = None,
) -> ModelState:
    """
    Create a new model state in LEARNING maturity.
    
    Args:
        equip_id: Equipment ID
        version: Model version number
        training_rows: Number of rows in training data
        training_start: Start of training window
        training_end: End of training window
        silhouette_score: Initial regime silhouette score
        run_id: Run ID that created this model
        
    Returns:
        New ModelState in LEARNING maturity
    """
    training_days = (training_end - training_start).total_seconds() / 86400.0
    
    state = ModelState(
        equip_id=equip_id,
        version=version,
        maturity=MaturityState.LEARNING,
        created_at=datetime.now(),
        silhouette_score=silhouette_score,
        training_rows=training_rows,
        training_days=training_days,
        consecutive_runs=1,
        last_run_id=run_id,
        last_run_at=datetime.now(),
        total_runs=1,
    )
    
    Console.info(
        f"Created model v{version} in LEARNING state",
        component="LIFECYCLE",
        equip_id=equip_id,
        version=version,
        training_rows=training_rows,
        training_days=f"{training_days:.1f}",
    )
    
    return state


def update_model_state_from_run(
    state: ModelState,
    run_id: str,
    run_success: bool,
    silhouette_score: Optional[float] = None,
    stability_ratio: Optional[float] = None,
    additional_rows: int = 0,
    additional_days: float = 0.0,
    forecast_mape: Optional[float] = None,
    forecast_rmse: Optional[float] = None,
) -> ModelState:
    """
    Update model state after a run completes.
    
    FLAW FIX #3: Added forecast_mape and forecast_rmse parameters.
    
    Args:
        state: Current model state
        run_id: Run ID
        run_success: Whether run completed successfully
        silhouette_score: Updated silhouette score (if available)
        stability_ratio: Updated stability ratio (if available)
        additional_rows: Rows processed in this run
        additional_days: Days of data processed in this run
        forecast_mape: Forecast MAPE from this run (if available)
        forecast_rmse: Forecast RMSE from this run (if available)
        
    Returns:
        Updated model state
    """
    state.last_run_id = run_id
    state.last_run_at = datetime.now()
    state.total_runs += 1
    
    if run_success:
        state.consecutive_runs += 1
    else:
        state.consecutive_runs = 0  # Reset on failure
    
    if silhouette_score is not None:
        state.silhouette_score = silhouette_score
    
    if stability_ratio is not None:
        state.stability_ratio = stability_ratio
    
    # FIXED: Update forecast quality metrics (v11.2.1)
    if forecast_mape is not None:
        state.forecast_mape = forecast_mape
    
    if forecast_rmse is not None:
        state.forecast_rmse = forecast_rmse
    
    state.training_rows += additional_rows
    state.training_days += additional_days
    
    return state


def get_active_model_dict(
    state: ModelState,
    regime_version: Optional[int] = None,
    threshold_version: Optional[int] = None,
    forecast_version: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get dict suitable for write_active_models in output_manager.
    
    Args:
        state: Current model state
        regime_version: Regime state version (if different from model version)
        threshold_version: Active threshold version (default: use regime version)
        forecast_version: Active forecast model version (default: use regime version)
        
    Returns:
        Dict for ACM_ActiveModels table
        
    Note: Only includes columns that exist in the ACM_ActiveModels table schema.
          Additional metrics are logged but not persisted to avoid schema changes.
    """
    # Log the full state for observability
    Console.info(
        f"Model state v{state.version}: {state.maturity.value}",
        component="LIFECYCLE",
        silhouette=state.silhouette_score,
        stability=state.stability_ratio,
        training_rows=state.training_rows,
        training_days=f"{state.training_days:.1f}",
        consecutive_runs=state.consecutive_runs,
    )
    
    # Use regime version as default for threshold/forecast if not specified
    effective_regime_version = regime_version or state.version
    
    # Return only columns that exist in ACM_ActiveModels table
    return {
        'ActiveRegimeVersion': effective_regime_version,
        'RegimeMaturityState': str(state.maturity),
        'RegimePromotedAt': state.promoted_at,
        'ActiveThresholdVersion': threshold_version or effective_regime_version,
        'ActiveForecastVersion': forecast_version or effective_regime_version,
    }


def load_model_state_from_sql(
    sql_client,
    equip_id: int,
) -> Optional[ModelState]:
    """
    Load current model state from ACM_ActiveModels.
    
    Args:
        sql_client: SQL client
        equip_id: Equipment ID
        
    Returns:
        ModelState if found, None otherwise
        
    Note: This function only queries columns that exist in the current schema.
          Training metrics are not persisted to ACM_ActiveModels yet - they are
          reconstructed from run history or defaults.
    """
    try:
        with sql_client.cursor() as cur:
            cur.execute("""
                SELECT 
                    ActiveRegimeVersion,
                    RegimeMaturityState,
                    RegimePromotedAt,
                    LastUpdatedAt,
                    LastUpdatedBy
                FROM dbo.[ACM_ActiveModels]
                WHERE EquipID = ?
            """, (equip_id,))
            row = cur.fetchone()
            
            if row is None:
                return None
            
            version = row[0] or 1
            maturity_str = row[1] or "LEARNING"
            
            try:
                maturity = MaturityState(maturity_str)
            except ValueError:
                maturity = MaturityState.LEARNING
            
            return ModelState(
                equip_id=equip_id,
                version=version,
                maturity=maturity,
                created_at=row[3] or datetime.now(),  # LastUpdatedAt as proxy
                promoted_at=row[2],
                # These are not persisted yet - use defaults
                silhouette_score=None,
                stability_ratio=None,
                training_rows=0,
                training_days=0.0,
                consecutive_runs=0,
                last_run_id=row[4],
                last_run_at=row[3],
            )
    except Exception as e:
        Console.warn(f"Failed to load model state: {e}", component="LIFECYCLE", error=str(e)[:200])
        return None
