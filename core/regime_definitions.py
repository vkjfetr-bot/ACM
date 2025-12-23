"""
Regime Definitions for ACM v11.0.0

Provides immutable, versioned storage of regime models.
Write-once semantics ensure model reproducibility.

Phase 2.5 Implementation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import numpy as np
import pandas as pd

from core.observability import Console


# =============================================================================
# Constants
# =============================================================================

# Special regime labels
REGIME_UNKNOWN = -1   # Point doesn't match any known regime
REGIME_EMERGING = -2  # Potential new regime forming
REGIME_GLOBAL = 0     # Default/global regime (cold-start fallback)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RegimeCentroid:
    """
    Single regime centroid definition.
    
    Attributes:
        label: Regime label (0, 1, 2, ...)
        centroid: Feature values at centroid
        radius: Average distance from centroid to assigned points
        n_points: Number of training points assigned to this regime
    """
    label: int
    centroid: np.ndarray
    radius: float = 0.0
    n_points: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "centroid": self.centroid.tolist(),
            "radius": float(self.radius),
            "n_points": self.n_points,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeCentroid":
        """Create from dictionary."""
        return cls(
            label=data["label"],
            centroid=np.array(data["centroid"]),
            radius=data.get("radius", 0.0),
            n_points=data.get("n_points", 0),
        )


@dataclass
class RegimeDefinition:
    """
    Immutable regime model definition.
    
    Stored in ACM_RegimeDefinitions table with versioning.
    """
    equip_id: int
    version: int
    num_regimes: int
    
    # Model components
    centroids: List[RegimeCentroid]
    feature_columns: List[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    
    # Optional transition matrix
    transition_matrix: Optional[np.ndarray] = None
    
    # Training metadata
    training_row_count: int = 0
    training_start_time: Optional[datetime] = None
    training_end_time: Optional[datetime] = None
    discovery_params: Dict[str, Any] = field(default_factory=dict)
    
    # Audit
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    
    @property
    def centroid_array(self) -> np.ndarray:
        """Get centroids as numpy array (n_regimes, n_features)."""
        return np.vstack([c.centroid for c in self.centroids])
    
    def get_regime_labels(self) -> List[int]:
        """Get list of regime labels."""
        return [c.label for c in self.centroids]
    
    def assign_regime(self, X: np.ndarray, 
                      unknown_threshold: float = 2.0) -> Tuple[int, float]:
        """
        Assign regime label to a single observation.
        
        Args:
            X: Feature vector (1D array)
            unknown_threshold: Distance multiplier for UNKNOWN classification
            
        Returns:
            (label, confidence) tuple
        """
        # Scale features
        X_scaled = (X - self.scaler_mean) / self.scaler_scale
        
        # Compute distances to all centroids
        centroid_matrix = self.centroid_array
        centroid_scaled = (centroid_matrix - self.scaler_mean) / self.scaler_scale
        
        distances = np.linalg.norm(centroid_scaled - X_scaled, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]
        
        # Get average radius for confidence calculation
        avg_radius = np.mean([c.radius for c in self.centroids]) or 1.0
        
        # Compute confidence (higher is better)
        confidence = max(0.0, 1.0 - nearest_dist / (unknown_threshold * avg_radius))
        
        # Check if too far from all centroids
        if nearest_dist > unknown_threshold * avg_radius:
            if nearest_dist > 1.5 * unknown_threshold * avg_radius:
                return REGIME_UNKNOWN, confidence
            else:
                return REGIME_EMERGING, confidence
        
        return self.centroids[nearest_idx].label, confidence
    
    def assign_regimes_batch(self, X: np.ndarray,
                             unknown_threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign regime labels to multiple observations.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            unknown_threshold: Distance multiplier for UNKNOWN classification
            
        Returns:
            (labels, confidences) arrays
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        confidences = np.zeros(n_samples)
        
        for i in range(n_samples):
            labels[i], confidences[i] = self.assign_regime(X[i], unknown_threshold)
        
        return labels, confidences
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "equip_id": self.equip_id,
            "version": self.version,
            "num_regimes": self.num_regimes,
            "centroids": [c.to_dict() for c in self.centroids],
            "feature_columns": self.feature_columns,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "transition_matrix": self.transition_matrix.tolist() if self.transition_matrix is not None else None,
            "training_row_count": self.training_row_count,
            "training_start_time": self.training_start_time.isoformat() if self.training_start_time else None,
            "training_end_time": self.training_end_time.isoformat() if self.training_end_time else None,
            "discovery_params": self.discovery_params,
        }
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "RegimeDefinition":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        
        return cls(
            equip_id=data["equip_id"],
            version=data["version"],
            num_regimes=data["num_regimes"],
            centroids=[RegimeCentroid.from_dict(c) for c in data["centroids"]],
            feature_columns=data["feature_columns"],
            scaler_mean=np.array(data["scaler_mean"]),
            scaler_scale=np.array(data["scaler_scale"]),
            transition_matrix=np.array(data["transition_matrix"]) if data.get("transition_matrix") else None,
            training_row_count=data.get("training_row_count", 0),
            training_start_time=datetime.fromisoformat(data["training_start_time"]) if data.get("training_start_time") else None,
            training_end_time=datetime.fromisoformat(data["training_end_time"]) if data.get("training_end_time") else None,
            discovery_params=data.get("discovery_params", {}),
        )


@dataclass
class RegimeAssignment:
    """
    Result of regime assignment for a single timestamp.
    """
    timestamp: datetime
    regime_label: int
    regime_version: int
    confidence: float
    
    @property
    def is_unknown(self) -> bool:
        return self.regime_label == REGIME_UNKNOWN
    
    @property
    def is_emerging(self) -> bool:
        return self.regime_label == REGIME_EMERGING
    
    @property
    def is_known(self) -> bool:
        return self.regime_label >= 0


# =============================================================================
# Regime Definition Store
# =============================================================================

class RegimeDefinitionStore:
    """
    Manages regime definitions in SQL.
    
    Provides:
    - Write-once semantics (immutable after creation)
    - Version auto-increment per equipment
    - JSON serialization for model components
    """
    
    def __init__(self, sql_client):
        """
        Initialize store.
        
        Args:
            sql_client: SQL client instance
        """
        self.sql = sql_client
        self._cache: Dict[Tuple[int, int], RegimeDefinition] = {}
    
    def save(self, definition: RegimeDefinition) -> int:
        """
        Save a new regime definition (immutable - creates new version).
        
        Args:
            definition: RegimeDefinition to save
            
        Returns:
            Assigned version number
        """
        # Get next version for this equipment
        cur = self.sql.cursor()
        try:
            cur.execute(
                "SELECT COALESCE(MAX(RegimeVersion), 0) + 1 FROM ACM_RegimeDefinitions WHERE EquipID = ?",
                (definition.equip_id,)
            )
            next_version = cur.fetchone()[0]
            
            # Serialize components to JSON
            centroids_json = json.dumps([c.to_dict() for c in definition.centroids])
            feature_cols_json = json.dumps(definition.feature_columns)
            scaler_json = json.dumps({
                "mean": definition.scaler_mean.tolist(),
                "scale": definition.scaler_scale.tolist(),
            })
            transition_json = json.dumps(definition.transition_matrix.tolist()) if definition.transition_matrix is not None else None
            params_json = json.dumps(definition.discovery_params) if definition.discovery_params else None
            
            # Insert new definition
            cur.execute("""
                INSERT INTO ACM_RegimeDefinitions (
                    EquipID, RegimeVersion, NumRegimes,
                    Centroids, FeatureColumns, ScalerParams, TransitionMatrix,
                    DiscoveryParams, TrainingRowCount, TrainingStartTime, TrainingEndTime,
                    CreatedAt, CreatedBy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE(), ?)
            """, (
                definition.equip_id, next_version, definition.num_regimes,
                centroids_json, feature_cols_json, scaler_json, transition_json,
                params_json, definition.training_row_count,
                definition.training_start_time, definition.training_end_time,
                definition.created_by,
            ))
            
            if not self.sql.conn.autocommit:
                self.sql.conn.commit()
            
            Console.info(f"Saved regime definition v{next_version}",
                        component="REGIME", equip_id=definition.equip_id,
                        version=next_version, num_regimes=definition.num_regimes)
            
            return next_version
            
        finally:
            cur.close()
    
    def load(self, equip_id: int, version: int) -> Optional[RegimeDefinition]:
        """
        Load a regime definition by equipment and version.
        
        Args:
            equip_id: Equipment ID
            version: Regime version
            
        Returns:
            RegimeDefinition or None if not found
        """
        # Check cache
        cache_key = (equip_id, version)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        cur = self.sql.cursor()
        try:
            cur.execute("""
                SELECT EquipID, RegimeVersion, NumRegimes,
                       Centroids, FeatureColumns, ScalerParams, TransitionMatrix,
                       DiscoveryParams, TrainingRowCount, TrainingStartTime, TrainingEndTime,
                       CreatedAt, CreatedBy
                FROM ACM_RegimeDefinitions
                WHERE EquipID = ? AND RegimeVersion = ?
            """, (equip_id, version))
            
            row = cur.fetchone()
            if row is None:
                return None
            
            # Parse JSON fields
            centroids_data = json.loads(row[3])
            feature_cols = json.loads(row[4])
            scaler_data = json.loads(row[5])
            transition_data = json.loads(row[6]) if row[6] else None
            params_data = json.loads(row[7]) if row[7] else {}
            
            definition = RegimeDefinition(
                equip_id=row[0],
                version=row[1],
                num_regimes=row[2],
                centroids=[RegimeCentroid.from_dict(c) for c in centroids_data],
                feature_columns=feature_cols,
                scaler_mean=np.array(scaler_data["mean"]),
                scaler_scale=np.array(scaler_data["scale"]),
                transition_matrix=np.array(transition_data) if transition_data else None,
                discovery_params=params_data,
                training_row_count=row[8] or 0,
                training_start_time=row[9],
                training_end_time=row[10],
                created_at=row[11],
                created_by=row[12],
            )
            
            # Cache result
            self._cache[cache_key] = definition
            return definition
            
        finally:
            cur.close()
    
    def load_latest(self, equip_id: int) -> Optional[RegimeDefinition]:
        """
        Load the latest regime definition for an equipment.
        
        Args:
            equip_id: Equipment ID
            
        Returns:
            Latest RegimeDefinition or None if none exist
        """
        cur = self.sql.cursor()
        try:
            cur.execute(
                "SELECT MAX(RegimeVersion) FROM ACM_RegimeDefinitions WHERE EquipID = ?",
                (equip_id,)
            )
            result = cur.fetchone()
            if result is None or result[0] is None:
                return None
            
            return self.load(equip_id, result[0])
            
        finally:
            cur.close()
    
    def list_versions(self, equip_id: int) -> List[Dict[str, Any]]:
        """
        List all regime versions for an equipment.
        
        Args:
            equip_id: Equipment ID
            
        Returns:
            List of version metadata dicts
        """
        cur = self.sql.cursor()
        try:
            cur.execute("""
                SELECT RegimeVersion, NumRegimes, TrainingRowCount,
                       TrainingStartTime, TrainingEndTime, CreatedAt
                FROM ACM_RegimeDefinitions
                WHERE EquipID = ?
                ORDER BY RegimeVersion DESC
            """, (equip_id,))
            
            versions = []
            for row in cur.fetchall():
                versions.append({
                    "version": row[0],
                    "num_regimes": row[1],
                    "training_row_count": row[2],
                    "training_start": row[3],
                    "training_end": row[4],
                    "created_at": row[5],
                })
            
            return versions
            
        finally:
            cur.close()
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()


# =============================================================================
# Helper Functions
# =============================================================================

def create_regime_definition(
    equip_id: int,
    centroids: np.ndarray,
    feature_columns: List[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    training_df: Optional[pd.DataFrame] = None,
    discovery_params: Optional[Dict[str, Any]] = None,
) -> RegimeDefinition:
    """
    Factory function to create a RegimeDefinition.
    
    Args:
        equip_id: Equipment ID
        centroids: Centroid matrix (n_regimes, n_features)
        feature_columns: List of feature column names
        scaler_mean: Scaler mean values
        scaler_scale: Scaler scale values
        training_df: Optional training DataFrame for metadata
        discovery_params: Optional discovery parameters
        
    Returns:
        New RegimeDefinition instance
    """
    n_regimes = centroids.shape[0]
    
    # Create centroid objects
    centroid_list = []
    for i in range(n_regimes):
        centroid_list.append(RegimeCentroid(
            label=i,
            centroid=centroids[i],
            radius=1.0,  # Default, should be computed from training data
            n_points=0,
        ))
    
    # Extract training metadata if DataFrame provided
    training_row_count = 0
    training_start = None
    training_end = None
    
    if training_df is not None and len(training_df) > 0:
        training_row_count = len(training_df)
        if "Timestamp" in training_df.columns:
            training_start = training_df["Timestamp"].min()
            training_end = training_df["Timestamp"].max()
    
    return RegimeDefinition(
        equip_id=equip_id,
        version=0,  # Will be assigned on save
        num_regimes=n_regimes,
        centroids=centroid_list,
        feature_columns=feature_columns,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        training_row_count=training_row_count,
        training_start_time=training_start,
        training_end_time=training_end,
        discovery_params=discovery_params or {},
    )


def get_regime_label_name(label: int) -> str:
    """
    Get human-readable name for a regime label.
    
    Args:
        label: Regime label
        
    Returns:
        Label name string
    """
    if label == REGIME_UNKNOWN:
        return "UNKNOWN"
    elif label == REGIME_EMERGING:
        return "EMERGING"
    elif label == REGIME_GLOBAL:
        return "GLOBAL"
    else:
        return f"REGIME_{label}"
