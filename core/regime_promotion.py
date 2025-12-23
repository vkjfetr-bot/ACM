"""
Regime Promotion for ACM v11.0.0

Handles the promotion of regime models from LEARNING to CONVERGED state.
Includes evaluation, logging, and audit trail.

Phase 2.10 Implementation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from core.observability import Console
from core.regime_manager import MaturityState, ActiveModelsManager
from core.regime_definitions import RegimeDefinition, RegimeDefinitionStore
from core.regime_evaluation import (
    RegimeMetrics,
    RegimeEvaluator,
    PromotionCriteria,
)

if TYPE_CHECKING:
    from core.sql_client import SQLClient


# =============================================================================
# Promotion Log Entry
# =============================================================================

@dataclass
class PromotionLogEntry:
    """
    Record of a promotion attempt (successful or not).
    """
    equip_id: int
    regime_version: int
    attempt_time: datetime
    success: bool
    from_state: MaturityState
    to_state: MaturityState
    
    # Metrics at evaluation time
    stability: float = 0.0
    novelty_rate: float = 0.0
    coverage: float = 0.0
    balance: float = 0.0
    sample_count: int = 0
    overall_score: float = 0.0
    
    # Failure reasons if not successful
    failure_reasons: List[str] = field(default_factory=list)
    
    # Audit
    triggered_by: str = "SYSTEM"
    notes: Optional[str] = None


# =============================================================================
# Regime Promoter
# =============================================================================

class RegimePromoter:
    """
    Handles regime model promotion lifecycle.
    
    Evaluates models, checks criteria, and manages state transitions
    from LEARNING to CONVERGED.
    """
    
    def __init__(
        self,
        sql_client: "SQLClient",
        criteria: Optional[PromotionCriteria] = None,
    ):
        """
        Initialize promoter.
        
        Args:
            sql_client: SQL client instance
            criteria: Promotion criteria (uses defaults if None)
        """
        self.sql = sql_client
        self.criteria = criteria or PromotionCriteria()
        self.active_models = ActiveModelsManager(sql_client)
        self.definitions = RegimeDefinitionStore(sql_client)
        self.evaluator = RegimeEvaluator()
    
    def evaluate_for_promotion(
        self,
        equip_id: int,
        version: Optional[int] = None,
    ) -> Tuple[bool, RegimeMetrics, List[str]]:
        """
        Evaluate whether a regime model is ready for promotion.
        
        Args:
            equip_id: Equipment ID
            version: Specific version to evaluate (or active version)
            
        Returns:
            (can_promote, metrics, failure_reasons)
        """
        # Get version to evaluate
        if version is None:
            active = self.active_models.get_active(equip_id)
            version = active.regime_version
            
            if version is None:
                return False, self._empty_metrics(), ["No active regime version"]
        
        # Load definition
        definition = self.definitions.load(equip_id, version)
        if definition is None:
            return False, self._empty_metrics(), [f"Definition v{version} not found"]
        
        # Get assignment history from ACM_RegimeTimeline
        labels, confidences = self._load_assignment_history(equip_id, version)
        
        if len(labels) == 0:
            return False, self._empty_metrics(), ["No assignment history found"]
        
        # Evaluate metrics
        metrics = self.evaluator.evaluate(
            labels,
            confidences=confidences,
            centroids=definition.centroid_array,
        )
        metrics.regime_version = version
        
        # Calculate days in LEARNING state
        days_in_learning = self._get_days_in_learning(equip_id)
        
        # Check promotion criteria
        can_promote, failures = self.criteria.evaluate(metrics, days_in_learning)
        
        Console.info(
            f"Promotion evaluation: version={version}, score={metrics.overall_score:.3f}, "
            f"can_promote={can_promote}",
            component="PROMOTION",
            equip_id=equip_id,
        )
        
        if failures:
            Console.info(f"Failures: {failures}", component="PROMOTION")
        
        return can_promote, metrics, failures
    
    def promote(
        self,
        equip_id: int,
        version: Optional[int] = None,
        force: bool = False,
        triggered_by: str = "SYSTEM",
        notes: Optional[str] = None,
    ) -> Tuple[bool, PromotionLogEntry]:
        """
        Attempt to promote a regime model to CONVERGED.
        
        Args:
            equip_id: Equipment ID
            version: Version to promote (or active version)
            force: Skip criteria check if True
            triggered_by: User/system that triggered promotion
            notes: Optional notes for audit log
            
        Returns:
            (success, log_entry)
        """
        # Get current state
        active = self.active_models.get_active(equip_id)
        current_state = active.regime_maturity
        target_version = version or active.regime_version
        
        if target_version is None:
            log = PromotionLogEntry(
                equip_id=equip_id,
                regime_version=0,
                attempt_time=datetime.now(),
                success=False,
                from_state=current_state,
                to_state=current_state,
                failure_reasons=["No regime version to promote"],
                triggered_by=triggered_by,
                notes=notes,
            )
            self._save_log(log)
            return False, log
        
        # Evaluate
        can_promote, metrics, failures = self.evaluate_for_promotion(equip_id, target_version)
        
        # Create log entry
        log = PromotionLogEntry(
            equip_id=equip_id,
            regime_version=target_version,
            attempt_time=datetime.now(),
            success=False,
            from_state=current_state,
            to_state=MaturityState.CONVERGED,
            stability=metrics.stability,
            novelty_rate=metrics.novelty_rate,
            coverage=metrics.coverage,
            balance=metrics.balance,
            sample_count=metrics.sample_count,
            overall_score=metrics.overall_score,
            failure_reasons=failures,
            triggered_by=triggered_by,
            notes=notes,
        )
        
        # Check if promotion allowed
        if not can_promote and not force:
            log.success = False
            log.to_state = current_state
            self._save_log(log)
            Console.warn(
                f"Promotion denied for v{target_version}: {failures}",
                component="PROMOTION",
                equip_id=equip_id,
            )
            return False, log
        
        if force and not can_promote:
            Console.warn(
                f"Force-promoting v{target_version} despite failures: {failures}",
                component="PROMOTION",
                equip_id=equip_id,
            )
            log.notes = (log.notes or "") + f" [FORCED: {failures}]"
        
        # Perform promotion
        try:
            self.active_models.promote_regime(
                equip_id,
                target_version,
                MaturityState.CONVERGED,
                triggered_by,
            )
            
            log.success = True
            self._save_log(log)
            
            Console.ok(
                f"Promoted regime v{target_version} to CONVERGED",
                component="PROMOTION",
                equip_id=equip_id,
            )
            
            return True, log
            
        except Exception as e:
            log.success = False
            log.failure_reasons.append(str(e))
            self._save_log(log)
            Console.error(f"Promotion failed: {e}", component="PROMOTION")
            return False, log
    
    def deprecate(
        self,
        equip_id: int,
        version: int,
        triggered_by: str = "SYSTEM",
        reason: Optional[str] = None,
    ) -> bool:
        """
        Deprecate a regime version.
        
        Args:
            equip_id: Equipment ID
            version: Version to deprecate
            triggered_by: User/system that triggered deprecation
            reason: Reason for deprecation
            
        Returns:
            Success boolean
        """
        try:
            active = self.active_models.get_active(equip_id)
            
            # If this is the active version, we can't deprecate it
            if active.regime_version == version:
                Console.warn(
                    f"Cannot deprecate active version {version}",
                    component="PROMOTION",
                    equip_id=equip_id,
                )
                return False
            
            # Log the deprecation
            log = PromotionLogEntry(
                equip_id=equip_id,
                regime_version=version,
                attempt_time=datetime.now(),
                success=True,
                from_state=MaturityState.LEARNING,  # Assumed
                to_state=MaturityState.DEPRECATED,
                triggered_by=triggered_by,
                notes=reason or "Manual deprecation",
            )
            self._save_log(log)
            
            Console.ok(
                f"Deprecated regime v{version}",
                component="PROMOTION",
                equip_id=equip_id,
            )
            
            return True
            
        except Exception as e:
            Console.error(f"Deprecation failed: {e}", component="PROMOTION")
            return False
    
    def get_promotion_history(
        self,
        equip_id: int,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get promotion history for equipment.
        
        Args:
            equip_id: Equipment ID
            limit: Maximum entries to return
            
        Returns:
            List of promotion log entries as dicts
        """
        cur = self.sql.cursor()
        try:
            cur.execute("""
                SELECT TOP (?)
                    RegimeVersion, AttemptTime, Success,
                    FromState, ToState, Stability, NoveltyRate,
                    Coverage, Balance, SampleCount, OverallScore,
                    FailureReasons, TriggeredBy, Notes
                FROM ACM_RegimePromotionLog
                WHERE EquipID = ?
                ORDER BY AttemptTime DESC
            """, (limit, equip_id))
            
            history = []
            for row in cur.fetchall():
                history.append({
                    "version": row[0],
                    "attempt_time": row[1],
                    "success": row[2],
                    "from_state": row[3],
                    "to_state": row[4],
                    "stability": row[5],
                    "novelty_rate": row[6],
                    "coverage": row[7],
                    "balance": row[8],
                    "sample_count": row[9],
                    "overall_score": row[10],
                    "failure_reasons": row[11],
                    "triggered_by": row[12],
                    "notes": row[13],
                })
            
            return history
            
        except Exception as e:
            Console.warn(f"Could not load promotion history: {e}", component="PROMOTION")
            return []
        finally:
            cur.close()
    
    def _load_assignment_history(
        self,
        equip_id: int,
        version: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load regime assignment history from SQL."""
        cur = self.sql.cursor()
        try:
            cur.execute("""
                SELECT RegimeLabel, AssignmentConfidence
                FROM ACM_RegimeTimeline
                WHERE EquipID = ? 
                  AND (RegimeVersion = ? OR RegimeVersion IS NULL)
                ORDER BY Timestamp ASC
            """, (equip_id, version))
            
            rows = cur.fetchall()
            if not rows:
                return np.array([]), np.array([])
            
            labels = np.array([r[0] for r in rows])
            confidences = np.array([r[1] if r[1] is not None else 0.5 for r in rows])
            
            return labels, confidences
            
        except Exception as e:
            Console.warn(f"Could not load assignment history: {e}", component="PROMOTION")
            return np.array([]), np.array([])
        finally:
            cur.close()
    
    def _get_days_in_learning(self, equip_id: int) -> int:
        """Get days since regime entered LEARNING state."""
        active = self.active_models.get_active(equip_id)
        
        if active.regime_promoted_at is None:
            return 0
        
        delta = datetime.now() - pd.Timestamp(active.regime_promoted_at).to_pydatetime()
        return delta.days
    
    def _save_log(self, log: PromotionLogEntry) -> None:
        """Save promotion log to SQL."""
        cur = self.sql.cursor()
        try:
            cur.execute("""
                INSERT INTO ACM_RegimePromotionLog (
                    EquipID, RegimeVersion, AttemptTime, Success,
                    FromState, ToState, Stability, NoveltyRate,
                    Coverage, Balance, SampleCount, OverallScore,
                    FailureReasons, TriggeredBy, Notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log.equip_id, log.regime_version, log.attempt_time, log.success,
                log.from_state.value, log.to_state.value,
                log.stability, log.novelty_rate, log.coverage, log.balance,
                log.sample_count, log.overall_score,
                "; ".join(log.failure_reasons) if log.failure_reasons else None,
                log.triggered_by, log.notes,
            ))
            
            if not self.sql.conn.autocommit:
                self.sql.conn.commit()
                
        except Exception as e:
            Console.warn(f"Could not save promotion log: {e}", component="PROMOTION")
        finally:
            cur.close()
    
    def _empty_metrics(self) -> RegimeMetrics:
        """Return empty metrics."""
        return RegimeMetrics(
            stability=0.0,
            novelty_rate=1.0,
            coverage=0.0,
            balance=0.0,
            transition_entropy=0.0,
            self_transition_rate=0.0,
            avg_silhouette=0.0,
            separation=0.0,
            sample_count=0,
        )


# =============================================================================
# Auto-Promotion Check
# =============================================================================

def check_auto_promotion(
    sql_client: "SQLClient",
    equip_id: int,
) -> Optional[int]:
    """
    Check if equipment is ready for automatic promotion.
    
    Called at the end of each ACM run to see if LEARNING
    models are ready to be promoted to CONVERGED.
    
    Args:
        sql_client: SQL client
        equip_id: Equipment ID
        
    Returns:
        Promoted version number, or None if not promoted
    """
    promoter = RegimePromoter(sql_client)
    active = promoter.active_models.get_active(equip_id)
    
    # Only auto-promote LEARNING models
    if active.regime_maturity != MaturityState.LEARNING:
        return None
    
    # Check if ready
    can_promote, metrics, _ = promoter.evaluate_for_promotion(equip_id)
    
    if can_promote:
        success, _ = promoter.promote(
            equip_id,
            triggered_by="AUTO_PROMOTION",
            notes="Automatic promotion after meeting criteria",
        )
        
        if success:
            return active.regime_version
    
    return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PromotionLogEntry",
    "RegimePromoter",
    "check_auto_promotion",
]
