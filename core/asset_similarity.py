"""
Asset Similarity for ACM v11.0.0 (P5.10)

Compute asset similarity for cold-start transfer learning.

Key Features:
- AssetProfile: Fingerprint of an asset's characteristics
- SimilarityScore: Quantified similarity between assets
- AssetSimilarity: Find similar assets, transfer baselines

Use Cases:
- Cold-start: Bootstrap new assets from similar existing assets
- Anomaly validation: Compare behavior across similar assets
- Fleet analysis: Group assets by operational similarity

Usage:
    similarity = AssetSimilarity(min_similarity=0.7)
    
    # Build profiles for existing assets
    for equip_id, equip_type, data, regimes in assets:
        similarity.build_profile(equip_id, equip_type, data, regimes)
    
    # Find similar assets for a new asset
    new_profile = similarity.build_profile(new_id, "FD_FAN", new_data, None)
    matches = similarity.find_similar(new_profile)
    
    # Transfer baseline from most similar asset
    if matches and matches[0].transferable:
        transferred, confidence = similarity.transfer_baseline(
            source_id=matches[0].source_equip_id,
            target_id=new_id,
            source_baseline=existing_baseline
        )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Protocol
import numpy as np
import pandas as pd


class BaselineNormalizerProtocol(Protocol):
    """Protocol for baseline normalizer compatibility."""
    
    def get_sensor_means(self) -> Dict[str, float]:
        """Get sensor means from baseline."""
        ...
    
    def get_sensor_stds(self) -> Dict[str, float]:
        """Get sensor standard deviations from baseline."""
        ...


@dataclass
class AssetProfile:
    """
    Profile of an asset for similarity computation.
    
    Captures the operational fingerprint of an asset including
    sensor configuration, statistics, and behavioral patterns.
    
    Attributes:
        equip_id: Equipment ID
        equip_type: Equipment type (e.g., "FD_FAN", "GAS_TURBINE")
        sensor_names: List of sensor column names
        sensor_means: Mean value for each sensor
        sensor_stds: Standard deviation for each sensor
        regime_count: Number of distinct operating regimes
        typical_health: Typical health score (0-100)
        data_hours: Total hours of historical data
    """
    equip_id: int
    equip_type: str
    sensor_names: List[str] = field(default_factory=list)
    sensor_means: Dict[str, float] = field(default_factory=dict)
    sensor_stds: Dict[str, float] = field(default_factory=dict)
    regime_count: int = 0
    typical_health: float = 85.0
    data_hours: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "equip_id": self.equip_id,
            "equip_type": self.equip_type,
            "sensor_names": self.sensor_names,
            "sensor_means": {k: round(v, 4) for k, v in self.sensor_means.items()},
            "sensor_stds": {k: round(v, 4) for k, v in self.sensor_stds.items()},
            "regime_count": self.regime_count,
            "typical_health": round(self.typical_health, 2),
            "data_hours": round(self.data_hours, 1)
        }
    
    def sensor_count(self) -> int:
        """Get number of sensors."""
        return len(self.sensor_names)
    
    def has_sensor(self, sensor: str) -> bool:
        """Check if profile includes a sensor."""
        return sensor in self.sensor_names


@dataclass
class SimilarityScore:
    """
    Similarity between two assets.
    
    Provides detailed breakdown of similarity factors
    for auditability and decision support.
    
    Attributes:
        source_equip_id: ID of the source (reference) asset
        target_equip_id: ID of the target (new) asset
        overall_similarity: Combined similarity score (0-1)
        sensor_similarity: Based on sensor overlap and statistics (0-1)
        behavior_similarity: Based on regime patterns and health (0-1)
        transferable: Whether models can be transferred
        transfer_confidence: Confidence in transfer success (0-1)
    """
    source_equip_id: int
    target_equip_id: int
    overall_similarity: float
    sensor_similarity: float
    behavior_similarity: float
    transferable: bool = False
    transfer_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_equip_id": self.source_equip_id,
            "target_equip_id": self.target_equip_id,
            "overall_similarity": round(self.overall_similarity, 4),
            "sensor_similarity": round(self.sensor_similarity, 4),
            "behavior_similarity": round(self.behavior_similarity, 4),
            "transferable": self.transferable,
            "transfer_confidence": round(self.transfer_confidence, 4)
        }
    
    def to_sql_row(self) -> Dict[str, Any]:
        """Convert to SQL insert row."""
        return {
            "SourceEquipID": self.source_equip_id,
            "TargetEquipID": self.target_equip_id,
            "OverallSimilarity": round(self.overall_similarity, 4),
            "SensorSimilarity": round(self.sensor_similarity, 4),
            "BehaviorSimilarity": round(self.behavior_similarity, 4),
            "Transferable": 1 if self.transferable else 0,
            "TransferConfidence": round(self.transfer_confidence, 4)
        }


@dataclass
class TransferResult:
    """
    Result of a baseline transfer operation.
    
    Attributes:
        source_equip_id: ID of source asset
        target_equip_id: ID of target asset
        confidence: Confidence in the transfer (0-1)
        sensors_transferred: List of sensors included in transfer
        scaling_factors: Per-sensor scaling applied
        notes: Any warnings or notes about the transfer
    """
    source_equip_id: int
    target_equip_id: int
    confidence: float
    sensors_transferred: List[str] = field(default_factory=list)
    scaling_factors: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_equip_id": self.source_equip_id,
            "target_equip_id": self.target_equip_id,
            "confidence": round(self.confidence, 4),
            "sensors_transferred": self.sensors_transferred,
            "scaling_factors": {k: round(v, 4) for k, v in self.scaling_factors.items()},
            "notes": self.notes
        }


class AssetSimilarity:
    """
    Compute asset similarity for cold-start transfer learning.
    
    Enables bootstrapping new assets from similar existing assets
    with full auditability.
    
    Similarity is computed based on:
    1. Same equipment type (required)
    2. Sensor overlap (how many sensors in common)
    3. Statistical similarity (means and stds similar)
    4. Behavioral similarity (regime count, health patterns)
    
    Attributes:
        min_similarity: Minimum overall similarity for transfer (default 0.7)
        profiles: Dictionary of asset profiles by equip_id
    """
    
    def __init__(self, min_similarity: float = 0.7):
        """
        Initialize AssetSimilarity.
        
        Args:
            min_similarity: Minimum similarity threshold (0-1) for transfer
        """
        self.min_similarity = min_similarity
        self.profiles: Dict[int, AssetProfile] = {}
    
    def build_profile(
        self,
        equip_id: int,
        equip_type: str,
        data: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        typical_health: float = 85.0
    ) -> AssetProfile:
        """
        Build profile from historical data.
        
        Args:
            equip_id: Equipment ID
            equip_type: Equipment type (e.g., "FD_FAN")
            data: DataFrame with sensor columns and Timestamp
            regime_labels: Series of regime labels (optional)
            typical_health: Typical health score (default 85.0)
        
        Returns:
            AssetProfile for the asset
        """
        # Identify sensor columns (exclude metadata)
        excluded = {"Timestamp", "EquipID", "EntryDateTime", "RunID", "RowNumber"}
        sensor_cols = [c for c in data.columns if c not in excluded]
        
        # Compute sensor statistics
        sensor_means = {}
        sensor_stds = {}
        
        for col in sensor_cols:
            if col in data.columns and data[col].notna().any():
                sensor_means[col] = float(data[col].mean())
                sensor_stds[col] = float(data[col].std())
        
        # Compute regime count
        regime_count = 0
        if regime_labels is not None and len(regime_labels) > 0:
            regime_count = len(regime_labels.unique())
        
        # Compute data span
        data_hours = 0.0
        if "Timestamp" in data.columns and len(data) > 0:
            try:
                ts = pd.to_datetime(data["Timestamp"])
                data_hours = (ts.max() - ts.min()).total_seconds() / 3600
            except Exception:
                pass
        
        profile = AssetProfile(
            equip_id=equip_id,
            equip_type=equip_type,
            sensor_names=sensor_cols,
            sensor_means=sensor_means,
            sensor_stds=sensor_stds,
            regime_count=regime_count,
            typical_health=typical_health,
            data_hours=data_hours
        )
        
        self.profiles[equip_id] = profile
        return profile
    
    def find_similar(
        self,
        target_profile: AssetProfile,
        candidates: Optional[List[int]] = None
    ) -> List[SimilarityScore]:
        """
        Find assets similar to target.
        
        Matching criteria:
        1. Same equipment type (required)
        2. Overlapping sensors
        3. Similar sensor statistics
        4. Similar regime structure
        
        Args:
            target_profile: Profile of asset to find matches for
            candidates: Optional list of candidate equip_ids to consider
        
        Returns:
            List of SimilarityScore, sorted by overall_similarity descending
        """
        results = []
        
        candidate_ids = candidates or list(self.profiles.keys())
        
        for source_id in candidate_ids:
            if source_id == target_profile.equip_id:
                continue
            
            source = self.profiles.get(source_id)
            if source is None:
                continue
            
            # Must be same type
            if source.equip_type != target_profile.equip_type:
                continue
            
            # Compute sensor similarity
            sensor_sim = self._sensor_similarity(source, target_profile)
            
            # Compute behavior similarity
            behavior_sim = self._behavior_similarity(source, target_profile)
            
            # Overall weighted (sensors more important)
            overall = 0.6 * sensor_sim + 0.4 * behavior_sim
            
            results.append(SimilarityScore(
                source_equip_id=source_id,
                target_equip_id=target_profile.equip_id,
                overall_similarity=overall,
                sensor_similarity=sensor_sim,
                behavior_similarity=behavior_sim,
                transferable=overall >= self.min_similarity,
                transfer_confidence=overall if overall >= self.min_similarity else 0.0
            ))
        
        return sorted(results, key=lambda x: x.overall_similarity, reverse=True)
    
    def _sensor_similarity(
        self,
        source: AssetProfile,
        target: AssetProfile
    ) -> float:
        """
        Compute sensor overlap and statistical similarity.
        
        Args:
            source: Source asset profile
            target: Target asset profile
        
        Returns:
            Similarity score (0-1)
        """
        # Sensor name overlap
        common = set(source.sensor_names) & set(target.sensor_names)
        if not common:
            return 0.0
        
        max_sensors = max(len(source.sensor_names), len(target.sensor_names))
        overlap = len(common) / max_sensors if max_sensors > 0 else 0.0
        
        # Statistical similarity for common sensors
        stat_diffs = []
        for sensor in common:
            mean_src = source.sensor_means.get(sensor, 0)
            mean_tgt = target.sensor_means.get(sensor, 0)
            mean_diff = abs(mean_src - mean_tgt)
            
            std_src = source.sensor_stds.get(sensor, 1)
            std_tgt = target.sensor_stds.get(sensor, 1)
            
            # Normalized difference (how many stds apart)
            max_std = max(std_src, std_tgt, 1e-10)
            normalized_diff = mean_diff / max_std
            stat_diffs.append(normalized_diff)
        
        # Convert to similarity (exponential decay)
        mean_diff = np.mean(stat_diffs) if stat_diffs else 0
        stat_similarity = float(np.exp(-mean_diff))
        
        return 0.5 * overlap + 0.5 * stat_similarity
    
    def _behavior_similarity(
        self,
        source: AssetProfile,
        target: AssetProfile
    ) -> float:
        """
        Compute behavioral similarity based on regimes and health.
        
        Args:
            source: Source asset profile
            target: Target asset profile
        
        Returns:
            Similarity score (0-1)
        """
        # Regime count similarity
        regime_diff = abs(source.regime_count - target.regime_count)
        regime_sim = 1 / (1 + regime_diff)
        
        # Health similarity
        health_diff = abs(source.typical_health - target.typical_health)
        health_sim = max(0.0, 1 - (health_diff / 100))
        
        return 0.5 * regime_sim + 0.5 * health_sim
    
    def transfer_baseline(
        self,
        source_id: int,
        target_id: int,
        source_baseline: Any
    ) -> TransferResult:
        """
        Transfer baseline from source to target asset.
        
        Args:
            source_id: Source equipment ID
            target_id: Target equipment ID
            source_baseline: Baseline normalizer from source asset
        
        Returns:
            TransferResult with confidence and details
        
        Raises:
            ValueError: If profiles not found or not similar enough
        """
        source_profile = self.profiles.get(source_id)
        target_profile = self.profiles.get(target_id)
        
        if not source_profile:
            raise ValueError(f"Source profile not found: {source_id}")
        if not target_profile:
            raise ValueError(f"Target profile not found: {target_id}")
        
        # Find similarity score
        scores = self.find_similar(target_profile, [source_id])
        
        if not scores:
            raise ValueError(f"Could not compute similarity between {source_id} and {target_id}")
        
        score = scores[0]
        
        if not score.transferable:
            raise ValueError(
                f"Assets not similar enough for transfer: "
                f"{score.overall_similarity:.3f} < {self.min_similarity}"
            )
        
        # Compute scaling factors for sensors
        common_sensors = set(source_profile.sensor_names) & set(target_profile.sensor_names)
        scaling_factors = {}
        
        for sensor in common_sensors:
            src_mean = source_profile.sensor_means.get(sensor, 0)
            tgt_mean = target_profile.sensor_means.get(sensor, 0)
            
            src_std = source_profile.sensor_stds.get(sensor, 1)
            tgt_std = target_profile.sensor_stds.get(sensor, 1)
            
            # Scale factor to adjust source baseline to target scale
            if abs(src_std) > 1e-10:
                scale = tgt_std / src_std
            else:
                scale = 1.0
            
            scaling_factors[sensor] = scale
        
        return TransferResult(
            source_equip_id=source_id,
            target_equip_id=target_id,
            confidence=score.transfer_confidence,
            sensors_transferred=list(common_sensors),
            scaling_factors=scaling_factors,
            notes=f"Transferred from {source_id} with similarity {score.overall_similarity:.3f}"
        )
    
    def get_profile(self, equip_id: int) -> Optional[AssetProfile]:
        """
        Get profile for an equipment.
        
        Args:
            equip_id: Equipment ID
        
        Returns:
            AssetProfile or None if not found
        """
        return self.profiles.get(equip_id)
    
    def list_profiles(self, equip_type: Optional[str] = None) -> List[AssetProfile]:
        """
        List all profiles, optionally filtered by type.
        
        Args:
            equip_type: Optional equipment type filter
        
        Returns:
            List of matching profiles
        """
        profiles = list(self.profiles.values())
        
        if equip_type:
            profiles = [p for p in profiles if p.equip_type == equip_type]
        
        return profiles
    
    def clear_profiles(self) -> None:
        """Clear all profiles."""
        self.profiles = {}
    
    def profile_count(self) -> int:
        """Get count of stored profiles."""
        return len(self.profiles)
