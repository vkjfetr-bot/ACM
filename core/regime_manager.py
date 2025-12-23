# core/regime_manager.py
"""
ACM Regime Management - v11.0.0

Centralized management of regime models, versioning, and maturity states.

Phase 2 Implementation Items:
- P2.1: ActiveModelsManager (ACM_ActiveModels pointer)
- P2.2: Cold start handling
- P2.3: UNKNOWN/EMERGING regime labels
- P2.4: Clean regime discovery inputs (validation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import json
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from core.sql_client import SQLClient


# =============================================================================
# Regime Label Constants
# =============================================================================

REGIME_UNKNOWN = -1    # Point doesn't match any known regime
REGIME_EMERGING = -2   # Potential new regime forming (borderline)


# =============================================================================
# P2.1 - Maturity States
# =============================================================================

class MaturityState(Enum):
    """
    Maturity states for regime models.
    
    Lifecycle: INITIALIZING -> LEARNING -> CONVERGED -> DEPRECATED
    """
    INITIALIZING = "INITIALIZING"  # No regimes discovered yet (cold start)
    LEARNING = "LEARNING"          # Regimes discovered but not validated
    CONVERGED = "CONVERGED"        # Regimes validated and stable
    DEPRECATED = "DEPRECATED"      # Replaced by newer version
    
    @property
    def allows_threshold_conditioning(self) -> bool:
        """Whether this state allows regime-conditioned thresholds."""
        return self == MaturityState.CONVERGED
    
    @property
    def allows_regime_forecasting(self) -> bool:
        """Whether this state allows regime-based forecasting."""
        return self == MaturityState.CONVERGED
    
    @property
    def is_operational(self) -> bool:
        """Whether this state is operational (not cold-start)."""
        return self in (MaturityState.LEARNING, MaturityState.CONVERGED)


# =============================================================================
# P2.1 - Active Models Pointer
# =============================================================================

@dataclass
class ActiveModels:
    """
    Current active model versions for an equipment.
    
    This is the single source of truth for which model versions
    are currently in use for production scoring.
    """
    equip_id: int
    
    # Regime model
    regime_version: Optional[int] = None
    regime_maturity: MaturityState = MaturityState.INITIALIZING
    regime_promoted_at: Optional[pd.Timestamp] = None
    
    # Threshold model
    threshold_version: Optional[int] = None
    threshold_promoted_at: Optional[pd.Timestamp] = None
    
    # Forecasting model
    forecast_version: Optional[int] = None
    forecast_promoted_at: Optional[pd.Timestamp] = None
    
    # Audit
    last_updated_at: Optional[pd.Timestamp] = None
    last_updated_by: str = "SYSTEM"
    
    @property
    def is_cold_start(self) -> bool:
        """Check if in cold-start state (no active regime model)."""
        return self.regime_version is None
    
    @property
    def allows_regime_conditioning(self) -> bool:
        """Check if regime-conditioned operations are allowed."""
        return (
            not self.is_cold_start and 
            self.regime_maturity.allows_threshold_conditioning
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "equip_id": self.equip_id,
            "regime_version": self.regime_version,
            "regime_maturity": self.regime_maturity.value,
            "regime_promoted_at": self.regime_promoted_at.isoformat() if self.regime_promoted_at else None,
            "threshold_version": self.threshold_version,
            "forecast_version": self.forecast_version,
            "is_cold_start": self.is_cold_start,
        }


class ActiveModelsManager:
    """
    Manages ACM_ActiveModels table reads/writes.
    
    All model access should go through this manager to ensure
    consistent version tracking.
    """
    
    # SQL statements
    _GET_ACTIVE_SQL = """
        SELECT EquipID, ActiveRegimeVersion, RegimeMaturityState,
               RegimePromotedAt, ActiveThresholdVersion, ThresholdPromotedAt,
               ActiveForecastVersion, ForecastPromotedAt,
               LastUpdatedAt, LastUpdatedBy
        FROM ACM_ActiveModels 
        WHERE EquipID = ?
    """
    
    _UPSERT_SQL = """
        MERGE INTO ACM_ActiveModels AS target
        USING (SELECT ? AS EquipID) AS source
        ON target.EquipID = source.EquipID
        WHEN MATCHED THEN
            UPDATE SET 
                ActiveRegimeVersion = ?,
                RegimeMaturityState = ?,
                RegimePromotedAt = ?,
                ActiveThresholdVersion = ?,
                ThresholdPromotedAt = ?,
                ActiveForecastVersion = ?,
                ForecastPromotedAt = ?,
                LastUpdatedAt = GETDATE(),
                LastUpdatedBy = ?
        WHEN NOT MATCHED THEN
            INSERT (EquipID, ActiveRegimeVersion, RegimeMaturityState, 
                    RegimePromotedAt, ActiveThresholdVersion, ThresholdPromotedAt,
                    ActiveForecastVersion, ForecastPromotedAt, 
                    LastUpdatedAt, LastUpdatedBy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, GETDATE(), ?);
    """
    
    _PROMOTE_REGIME_SQL = """
        MERGE INTO ACM_ActiveModels AS target
        USING (SELECT ? AS EquipID) AS source
        ON target.EquipID = source.EquipID
        WHEN MATCHED THEN
            UPDATE SET 
                ActiveRegimeVersion = ?,
                RegimeMaturityState = ?,
                RegimePromotedAt = GETDATE(),
                LastUpdatedAt = GETDATE(),
                LastUpdatedBy = ?
        WHEN NOT MATCHED THEN
            INSERT (EquipID, ActiveRegimeVersion, RegimeMaturityState, 
                    RegimePromotedAt, LastUpdatedAt, LastUpdatedBy)
            VALUES (?, ?, ?, GETDATE(), GETDATE(), ?);
    """
    
    def __init__(self, sql_client: "SQLClient"):
        """
        Initialize with SQL client.
        
        Args:
            sql_client: Connected SQLClient instance
        """
        self.sql = sql_client
        self._cache: Dict[int, ActiveModels] = {}
    
    def get_active(self, equip_id: int, use_cache: bool = True) -> ActiveModels:
        """
        Get current active models for equipment.
        
        Args:
            equip_id: Equipment ID
            use_cache: Whether to use cached value if available
            
        Returns:
            ActiveModels instance (cold-start if not found)
        """
        if use_cache and equip_id in self._cache:
            return self._cache[equip_id]
        
        try:
            cur = self.sql.cursor()
            cur.execute(self._GET_ACTIVE_SQL, (equip_id,))
            row = cur.fetchone()
            
            if row is None:
                # Cold start - no active models registered
                result = ActiveModels(
                    equip_id=equip_id,
                    regime_version=None,
                    regime_maturity=MaturityState.INITIALIZING,
                    threshold_version=None,
                    forecast_version=None
                )
            else:
                result = ActiveModels(
                    equip_id=row[0],  # EquipID
                    regime_version=row[1],  # ActiveRegimeVersion
                    regime_maturity=MaturityState(row[2]) if row[2] else MaturityState.INITIALIZING,
                    regime_promoted_at=pd.Timestamp(row[3]) if row[3] else None,
                    threshold_version=row[4],  # ActiveThresholdVersion
                    threshold_promoted_at=pd.Timestamp(row[5]) if row[5] else None,
                    forecast_version=row[6],  # ActiveForecastVersion
                    forecast_promoted_at=pd.Timestamp(row[7]) if row[7] else None,
                    last_updated_at=pd.Timestamp(row[8]) if row[8] else None,
                    last_updated_by=row[9] or "SYSTEM"
                )
            
            self._cache[equip_id] = result
            return result
            
        except Exception as e:
            # If table doesn't exist or query fails, return cold-start state
            return ActiveModels(
                equip_id=equip_id,
                regime_version=None,
                regime_maturity=MaturityState.INITIALIZING
            )
    
    def promote_regime(self, 
                       equip_id: int, 
                       version: int,
                       new_state: MaturityState,
                       updated_by: str = "SYSTEM") -> None:
        """
        Promote a regime version to active.
        
        Args:
            equip_id: Equipment ID
            version: Regime version to promote
            new_state: New maturity state
            updated_by: User/system that triggered promotion
        """
        try:
            cur = self.sql.cursor()
            cur.execute(
                self._PROMOTE_REGIME_SQL,
                (equip_id, version, new_state.value, updated_by,
                 equip_id, version, new_state.value, updated_by)
            )
            self.sql.commit()
            
            # Invalidate cache
            self._cache.pop(equip_id, None)
            
        except Exception as e:
            raise RuntimeError(f"Failed to promote regime: {e}") from e
    
    def promote_threshold(self,
                          equip_id: int,
                          version: int,
                          updated_by: str = "SYSTEM") -> None:
        """Promote a threshold version to active."""
        try:
            cur = self.sql.cursor()
            cur.execute("""
                MERGE INTO ACM_ActiveModels AS target
                USING (SELECT ? AS EquipID) AS source
                ON target.EquipID = source.EquipID
                WHEN MATCHED THEN
                    UPDATE SET 
                        ActiveThresholdVersion = ?,
                        ThresholdPromotedAt = GETDATE(),
                        LastUpdatedAt = GETDATE(),
                        LastUpdatedBy = ?
                WHEN NOT MATCHED THEN
                    INSERT (EquipID, ActiveThresholdVersion, ThresholdPromotedAt,
                            LastUpdatedAt, LastUpdatedBy)
                    VALUES (?, ?, GETDATE(), GETDATE(), ?);
            """, (equip_id, version, updated_by, equip_id, version, updated_by))
            self.sql.commit()
            self._cache.pop(equip_id, None)
        except Exception as e:
            raise RuntimeError(f"Failed to promote threshold: {e}") from e
    
    def check_cold_start(self, equip_id: int) -> bool:
        """
        Check if equipment is in cold-start state.
        
        Args:
            equip_id: Equipment ID
            
        Returns:
            True if no active regime model exists
        """
        return self.get_active(equip_id).is_cold_start
    
    def clear_cache(self, equip_id: Optional[int] = None) -> None:
        """Clear cached active models."""
        if equip_id:
            self._cache.pop(equip_id, None)
        else:
            self._cache.clear()


# =============================================================================
# P2.3 - Regime Assignment with UNKNOWN/EMERGING
# =============================================================================

@dataclass
class RegimeAssignment:
    """Result of regime assignment."""
    label: int  # 0, 1, 2, ... or REGIME_UNKNOWN (-1) or REGIME_EMERGING (-2)
    confidence: float  # 0.0 to 1.0
    nearest_regime: int  # The nearest known regime (for UNKNOWN cases)
    distance: float  # Distance to nearest centroid
    
    @property
    def is_known(self) -> bool:
        """Check if assigned to a known regime."""
        return self.label >= 0
    
    @property
    def is_unknown(self) -> bool:
        """Check if assigned as UNKNOWN."""
        return self.label == REGIME_UNKNOWN
    
    @property
    def is_emerging(self) -> bool:
        """Check if assigned as EMERGING."""
        return self.label == REGIME_EMERGING
    
    @property
    def label_str(self) -> str:
        """Human-readable label."""
        if self.label == REGIME_UNKNOWN:
            return "UNKNOWN"
        elif self.label == REGIME_EMERGING:
            return "EMERGING"
        else:
            return f"R{self.label}"


class RegimeAssigner:
    """
    Assigns data points to regimes with confidence scoring.
    
    Key difference from v10: Can return UNKNOWN or EMERGING
    instead of forcing nearest regime assignment.
    """
    
    def __init__(self, 
                 centroids: np.ndarray,
                 scaler_mean: np.ndarray,
                 scaler_scale: np.ndarray,
                 avg_centroid_distance: float,
                 unknown_threshold: float = 2.0,
                 emerging_threshold: float = 1.5):
        """
        Initialize regime assigner.
        
        Args:
            centroids: Cluster centroids (n_clusters x n_features)
            scaler_mean: StandardScaler mean
            scaler_scale: StandardScaler scale
            avg_centroid_distance: Average distance to centroid from training
            unknown_threshold: Distance ratio for UNKNOWN classification
            emerging_threshold: Distance ratio for EMERGING classification
        """
        self.centroids = np.array(centroids)
        self.scaler_mean = np.array(scaler_mean)
        self.scaler_scale = np.array(scaler_scale)
        self.avg_centroid_distance = avg_centroid_distance
        self.unknown_threshold = unknown_threshold
        self.emerging_threshold = emerging_threshold
        self.n_regimes = len(centroids)
    
    def assign(self, X: np.ndarray) -> RegimeAssignment:
        """
        Assign a single point to a regime.
        
        Args:
            X: Feature vector (1D array)
            
        Returns:
            RegimeAssignment with label, confidence, and metadata
        """
        # Scale input
        X_scaled = (X - self.scaler_mean) / (self.scaler_scale + 1e-10)
        
        # Calculate distances to all centroids
        distances = np.linalg.norm(self.centroids - X_scaled, axis=1)
        nearest_idx = int(np.argmin(distances))
        nearest_dist = distances[nearest_idx]
        
        # Calculate confidence based on distance ratio
        dist_ratio = nearest_dist / (self.avg_centroid_distance + 1e-10)
        confidence = max(0.0, min(1.0, 1.0 - (dist_ratio - 1.0)))
        
        # Determine label based on distance
        if dist_ratio > self.unknown_threshold:
            # Too far from any known regime
            label = REGIME_UNKNOWN
        elif dist_ratio > self.emerging_threshold:
            # Borderline - could be new regime forming
            label = REGIME_EMERGING
        else:
            # Assign to nearest regime
            label = nearest_idx
        
        return RegimeAssignment(
            label=label,
            confidence=confidence,
            nearest_regime=nearest_idx,
            distance=float(nearest_dist)
        )
    
    def assign_batch(self, X: np.ndarray) -> List[RegimeAssignment]:
        """
        Assign multiple points to regimes.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            
        Returns:
            List of RegimeAssignment for each point
        """
        return [self.assign(x) for x in X]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for SQL storage)."""
        return {
            "centroids": self.centroids.tolist(),
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "avg_centroid_distance": self.avg_centroid_distance,
            "unknown_threshold": self.unknown_threshold,
            "emerging_threshold": self.emerging_threshold,
            "n_regimes": self.n_regimes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeAssigner":
        """Deserialize from dictionary."""
        return cls(
            centroids=np.array(data["centroids"]),
            scaler_mean=np.array(data["scaler_mean"]),
            scaler_scale=np.array(data["scaler_scale"]),
            avg_centroid_distance=data["avg_centroid_distance"],
            unknown_threshold=data.get("unknown_threshold", 2.0),
            emerging_threshold=data.get("emerging_threshold", 1.5),
        )


# =============================================================================
# P2.4 - Clean Regime Input Validation
# =============================================================================

# Forbidden patterns for regime discovery inputs
FORBIDDEN_REGIME_INPUT_PATTERNS = [
    "_z",       # Detector z-scores
    "pca_",     # PCA outputs
    "iforest",  # IForest outputs
    "gmm_",     # GMM outputs
    "omr_",     # OMR outputs
    "ar1_",     # AR1 outputs
    "fused",    # Fused scores
    "health",   # Health indices
    "resid",    # Residuals
    "regime",   # Previously computed regimes
    "_score",   # Any score columns
]


class RegimeInputValidator:
    """
    Validates inputs for regime discovery.
    
    Prevents data leakage by ensuring no detector outputs
    or derived metrics are used for regime clustering.
    """
    
    def __init__(self, 
                 forbidden_patterns: Optional[List[str]] = None,
                 strict: bool = True):
        """
        Initialize validator.
        
        Args:
            forbidden_patterns: List of column name patterns to reject
            strict: If True, raise exception on violation. If False, just warn.
        """
        self.forbidden_patterns = forbidden_patterns or FORBIDDEN_REGIME_INPUT_PATTERNS
        self.strict = strict
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """
        Validate DataFrame for regime discovery.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            (is_valid, valid_columns, rejected_columns)
        """
        valid_cols = []
        rejected_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            is_forbidden = False
            
            for pattern in self.forbidden_patterns:
                if pattern in col_lower:
                    is_forbidden = True
                    break
            
            if is_forbidden:
                rejected_cols.append(col)
            else:
                valid_cols.append(col)
        
        is_valid = len(rejected_cols) == 0 or not self.strict
        
        return is_valid, valid_cols, rejected_cols
    
    def filter_clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return DataFrame with only clean columns for regime discovery.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with forbidden columns removed
        """
        _, valid_cols, rejected_cols = self.validate(df)
        
        if rejected_cols:
            from core.observability import Console
            Console.warn(
                f"Filtered {len(rejected_cols)} forbidden columns from regime inputs",
                component="REGIME",
                rejected_cols=rejected_cols[:5]  # Show first 5
            )
        
        return df[valid_cols].copy()
    
    def assert_clean(self, df: pd.DataFrame) -> None:
        """
        Assert that DataFrame has no forbidden columns.
        
        Raises:
            ValueError: If any forbidden columns are found
        """
        is_valid, _, rejected_cols = self.validate(df)
        
        if not is_valid:
            raise ValueError(
                f"Data leakage detected: regime inputs contain forbidden columns: {rejected_cols}"
            )


# =============================================================================
# P2.5/P2.6 - Integration with RegimeDefinitionStore
# =============================================================================

from core.regime_definitions import (
    RegimeDefinition,
    RegimeDefinitionStore,
    RegimeCentroid,
)


@dataclass
class RegimeContext:
    """
    Full regime context for pipeline execution.
    
    Combines active model info with loaded definition.
    """
    equip_id: int
    active_models: ActiveModels
    definition: Optional[RegimeDefinition] = None
    assigner: Optional[RegimeAssigner] = None
    
    @property
    def is_ready(self) -> bool:
        """Check if regime model is ready for production use."""
        return (
            self.definition is not None and
            self.active_models.regime_maturity == MaturityState.CONVERGED
        )
    
    @property
    def should_use_global(self) -> bool:
        """Check if should use global (non-regime-conditioned) processing."""
        return not self.is_ready
    
    @property
    def regime_version(self) -> Optional[int]:
        """Get active regime version."""
        return self.active_models.regime_version


class RegimeManager:
    """
    Unified facade for regime management.
    
    Combines ActiveModelsManager and RegimeDefinitionStore for
    convenient access during pipeline execution.
    """
    
    def __init__(self, sql_client: "SQLClient"):
        """
        Initialize regime manager.
        
        Args:
            sql_client: SQL client instance
        """
        self.sql = sql_client
        self.active_models = ActiveModelsManager(sql_client)
        self.definitions = RegimeDefinitionStore(sql_client)
        self.input_validator = RegimeInputValidator()
    
    def get_context(self, equip_id: int) -> RegimeContext:
        """
        Get full regime context for equipment.
        
        Args:
            equip_id: Equipment ID
            
        Returns:
            RegimeContext with active models and loaded definition
        """
        active = self.active_models.get_active(equip_id)
        
        definition = None
        assigner = None
        
        if active.regime_version is not None:
            definition = self.definitions.load(equip_id, active.regime_version)
            
            if definition is not None:
                # Create assigner from definition
                avg_dist = float(np.mean([c.radius for c in definition.centroids]) or 1.0)
                assigner = RegimeAssigner(
                    centroids=definition.centroid_array,
                    scaler_mean=definition.scaler_mean,
                    scaler_scale=definition.scaler_scale,
                    avg_centroid_distance=avg_dist,
                )
        
        return RegimeContext(
            equip_id=equip_id,
            active_models=active,
            definition=definition,
            assigner=assigner,
        )
    
    def is_cold_start(self, equip_id: int) -> bool:
        """Check if equipment is in cold-start state."""
        return self.active_models.check_cold_start(equip_id)
    
    def save_and_activate(self, 
                          definition: RegimeDefinition,
                          maturity: MaturityState = MaturityState.LEARNING,
                          updated_by: str = "SYSTEM") -> int:
        """
        Save new regime definition and activate it.
        
        Args:
            definition: RegimeDefinition to save
            maturity: Initial maturity state
            updated_by: Audit user
            
        Returns:
            New version number
        """
        version = self.definitions.save(definition)
        self.active_models.promote_regime(
            definition.equip_id, version, maturity, updated_by
        )
        return version
    
    def validate_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and filter DataFrame for regime discovery.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        return self.input_validator.filter_clean_columns(df)
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.active_models.clear_cache()
        self.definitions.clear_cache()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "REGIME_UNKNOWN",
    "REGIME_EMERGING",
    "FORBIDDEN_REGIME_INPUT_PATTERNS",
    
    # Enums
    "MaturityState",
    
    # Dataclasses
    "ActiveModels",
    "RegimeAssignment",
    "RegimeContext",
    
    # Managers
    "ActiveModelsManager",
    "RegimeAssigner",
    "RegimeInputValidator",
    "RegimeManager",
    
    # Re-exports from regime_definitions
    "RegimeDefinition",
    "RegimeDefinitionStore",
    "RegimeCentroid",
]
