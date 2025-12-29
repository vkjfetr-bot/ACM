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
    """Criteria for promoting model from LEARNING to CONVERGED."""
    min_training_days: int = 7
    min_silhouette_score: float = 0.15
    min_stability_ratio: float = 0.8
    min_consecutive_runs: int = 3
    min_training_rows: int = 1000


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
) -> ModelState:
    """
    Update model state after a run completes.
    
    Args:
        state: Current model state
        run_id: Run ID
        run_success: Whether run completed successfully
        silhouette_score: Updated silhouette score (if available)
        stability_ratio: Updated stability ratio (if available)
        additional_rows: Rows processed in this run
        additional_days: Days of data processed in this run
        
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
    
    state.training_rows += additional_rows
    state.training_days += additional_days
    
    return state


def get_active_model_dict(
    state: ModelState,
    regime_version: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get dict suitable for write_active_models in output_manager.
    
    Args:
        state: Current model state
        regime_version: Regime state version (if different from model version)
        
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
    
    # Return only columns that exist in ACM_ActiveModels table
    return {
        'ActiveRegimeVersion': regime_version or state.version,
        'RegimeMaturityState': str(state.maturity),
        'RegimePromotedAt': state.promoted_at,
        'ActiveThresholdVersion': None,  # Future: threshold versioning
        'ActiveForecastVersion': None,   # Future: forecast model versioning
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
