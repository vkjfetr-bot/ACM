"""
ACM v11.0.0 - Episode Manager Module.

Implements episode-only alerting where episodes are the SOLE alerting primitive.
Point-level anomalies are never surfaced to operators.

Key features:
- Episode detection from contiguous anomaly regions
- Severity classification (LOW, MEDIUM, HIGH, CRITICAL)
- Lifecycle management (ACTIVE, RESOLVED, SUPPRESSED, ESCALATED)
- Cooldown to prevent re-alerting
- Sensor attribution for root cause analysis
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import uuid
import numpy as np
import pandas as pd


class EpisodeSeverity(Enum):
    """
    Episode severity levels.
    
    Severity determines alert routing and response priority.
    """
    LOW = 1       # Anomaly detected, likely noise - monitor
    MEDIUM = 2    # Persistent anomaly, monitor closely
    HIGH = 3      # Significant deviation, investigate
    CRITICAL = 4  # Imminent failure risk, immediate action
    
    @property
    def color(self) -> str:
        """Color for visualization."""
        return {
            EpisodeSeverity.LOW: "yellow",
            EpisodeSeverity.MEDIUM: "orange",
            EpisodeSeverity.HIGH: "red",
            EpisodeSeverity.CRITICAL: "darkred",
        }[self]
    
    @property
    def requires_action(self) -> bool:
        """Whether this severity requires operator action."""
        return self in (EpisodeSeverity.HIGH, EpisodeSeverity.CRITICAL)


class EpisodeStatus(Enum):
    """
    Episode lifecycle status.
    
    Tracks the current state of an episode from detection to resolution.
    """
    ACTIVE = "ACTIVE"           # Currently ongoing
    RESOLVED = "RESOLVED"       # Returned to normal
    SUPPRESSED = "SUPPRESSED"   # Operator dismissed
    ESCALATED = "ESCALATED"     # Escalated to higher tier
    ACKNOWLEDGED = "ACKNOWLEDGED"  # Operator acknowledged but not resolved


@dataclass
class SensorAttribution:
    """Attribution of episode to specific sensors."""
    sensor_name: str
    contribution_pct: float  # 0-100
    peak_z_score: float
    mean_z_score: float
    detector: str  # Which detector flagged this sensor


@dataclass
class Episode:
    """
    The SOLE alerting primitive in ACM v11.
    
    All alerts are episode-based. Point-level anomalies are never
    surfaced to operators - they are only used internally to build episodes.
    """
    id: str
    equip_id: int
    start_time: pd.Timestamp
    end_time: Optional[pd.Timestamp]
    severity: EpisodeSeverity
    status: EpisodeStatus
    
    # Score metrics
    peak_z_score: float
    mean_z_score: float
    duration_hours: float
    sample_count: int
    
    # Attribution
    top_contributors: List[SensorAttribution] = field(default_factory=list)
    affected_detectors: List[str] = field(default_factory=list)
    
    # Context
    regime_at_onset: int = -1
    regime_at_resolution: Optional[int] = None
    
    # Quality metrics
    detector_agreement: float = 0.0  # How much detectors agreed
    confidence: float = 0.0          # How confident episode is real
    false_positive_probability: float = 0.0
    
    # Operator interaction
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[pd.Timestamp] = None
    suppression_reason: Optional[str] = None
    
    # Metadata
    created_at: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    updated_at: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    
    @property
    def is_active(self) -> bool:
        """Check if episode is currently active."""
        return self.status == EpisodeStatus.ACTIVE
    
    @property
    def is_resolved(self) -> bool:
        """Check if episode is resolved."""
        return self.status in (EpisodeStatus.RESOLVED, EpisodeStatus.SUPPRESSED)
    
    @property
    def duration_minutes(self) -> float:
        """Duration in minutes."""
        return self.duration_hours * 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for SQL persistence."""
        return {
            "EpisodeID": self.id,
            "EquipID": self.equip_id,
            "StartTime": self.start_time,
            "EndTime": self.end_time,
            "Severity": self.severity.name,
            "SeverityLevel": self.severity.value,
            "Status": self.status.value,
            "PeakZScore": round(self.peak_z_score, 4),
            "MeanZScore": round(self.mean_z_score, 4),
            "DurationHours": round(self.duration_hours, 4),
            "SampleCount": self.sample_count,
            "TopContributors": ",".join(a.sensor_name for a in self.top_contributors[:5]),
            "AffectedDetectors": ",".join(self.affected_detectors),
            "RegimeAtOnset": self.regime_at_onset,
            "DetectorAgreement": round(self.detector_agreement, 4),
            "Confidence": round(self.confidence, 4),
            "FalsePositiveProbability": round(self.false_positive_probability, 4),
            "CreatedAt": self.created_at,
            "UpdatedAt": self.updated_at,
        }
    
    def update_end_time(self, end_time: pd.Timestamp, regime: Optional[int] = None) -> None:
        """Update episode end time when resolved."""
        self.end_time = end_time
        self.status = EpisodeStatus.RESOLVED
        self.regime_at_resolution = regime
        self.updated_at = pd.Timestamp.now()
        if self.start_time is not None:
            self.duration_hours = (end_time - self.start_time).total_seconds() / 3600


@dataclass
class EpisodeConfig:
    """Configuration for episode detection."""
    # Detection thresholds
    min_duration_minutes: float = 30.0
    threshold_low: float = 3.0
    threshold_medium: float = 4.0
    threshold_high: float = 5.5
    threshold_critical: float = 7.0
    
    # Cooldown
    cooldown_hours: float = 6.0
    
    # Minimum samples
    min_samples: int = 3
    
    # Attribution
    top_n_contributors: int = 5
    
    # Confidence
    min_confidence: float = 0.3
    
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "EpisodeConfig":
        """Create from config dict."""
        episode_cfg = cfg.get("episode", {})
        return cls(
            min_duration_minutes=float(episode_cfg.get("min_duration_minutes", 30.0)),
            threshold_low=float(episode_cfg.get("threshold_low", 3.0)),
            threshold_medium=float(episode_cfg.get("threshold_medium", 4.0)),
            threshold_high=float(episode_cfg.get("threshold_high", 5.5)),
            threshold_critical=float(episode_cfg.get("threshold_critical", 7.0)),
            cooldown_hours=float(episode_cfg.get("cooldown_hours", 6.0)),
            min_samples=int(episode_cfg.get("min_samples", 3)),
            top_n_contributors=int(episode_cfg.get("top_contributors", 5)),
            min_confidence=float(episode_cfg.get("min_confidence", 0.3)),
        )


class EpisodeManager:
    """
    Central episode lifecycle manager.
    
    CRITICAL: This is the ONLY way alerts are generated in ACM v11.
    Point-level anomalies are NEVER surfaced to operators.
    
    Responsibilities:
    1. Detect episodes from fused anomaly scores
    2. Track episode lifecycle (active → resolved)
    3. Apply cooldown to prevent alert fatigue
    4. Extract sensor attribution for root cause
    5. Compute episode confidence
    
    Usage:
        manager = EpisodeManager(config)
        
        # Detect episodes from fused scores
        episodes = manager.detect_episodes(
            fused_z_df=df,
            detector_outputs=detector_outputs,
            regime_labels=regime_labels,
        )
        
        # Check for active episodes
        active = manager.get_active_episodes(equip_id=1)
    """
    
    def __init__(self, config: Optional[EpisodeConfig] = None, equip_id: int = 0):
        """
        Initialize episode manager.
        
        Args:
            config: Episode detection configuration
            equip_id: Default equipment ID
        """
        self.config = config or EpisodeConfig()
        self.equip_id = equip_id
        
        # Track active episodes per equipment
        self.active_episodes: Dict[int, List[Episode]] = {}
        
        # Cooldown tracking (equip_id -> last episode end time)
        self._cooldown_until: Dict[int, pd.Timestamp] = {}
        
        # Statistics
        self._episodes_detected = 0
        self._episodes_suppressed = 0
    
    def detect_episodes(
        self,
        fused_z_df: pd.DataFrame,
        detector_outputs: Optional[Dict[str, pd.DataFrame]] = None,
        regime_labels: Optional[pd.Series] = None,
        timestamp_col: str = "Timestamp",
        fused_col: str = "fused_z",
        confidence_col: str = "fusion_confidence",
        agreement_col: str = "detector_agreement",
    ) -> List[Episode]:
        """
        Detect episodes from fused anomaly scores.
        
        Steps:
        1. Find contiguous regions where fused_z > threshold
        2. Filter by minimum duration and samples
        3. Compute severity based on peak score
        4. Extract sensor attribution from detector outputs
        5. Compute confidence based on detector agreement
        
        Args:
            fused_z_df: DataFrame with fused z-scores and timestamps
            detector_outputs: Optional dict of detector_name -> DataFrame
            regime_labels: Optional series of regime labels
            timestamp_col: Name of timestamp column
            fused_col: Name of fused z-score column
            confidence_col: Name of confidence column
            agreement_col: Name of detector agreement column
            
        Returns:
            List of detected Episode objects
        """
        if fused_z_df.empty:
            return []
        
        # Validate required columns
        if fused_col not in fused_z_df.columns:
            return []
        
        episodes: List[Episode] = []
        threshold = self.config.threshold_low
        
        # Get timestamps
        if timestamp_col in fused_z_df.columns:
            timestamps = pd.to_datetime(fused_z_df[timestamp_col])
        else:
            timestamps = fused_z_df.index
        
        fused_z = fused_z_df[fused_col].values
        
        # Get confidence and agreement if available
        confidence = fused_z_df[confidence_col].values if confidence_col in fused_z_df.columns else np.ones(len(fused_z))
        agreement = fused_z_df[agreement_col].values if agreement_col in fused_z_df.columns else np.ones(len(fused_z))
        
        # Find anomaly regions
        is_anomaly = fused_z > threshold
        
        # Label contiguous regions
        region_changes = is_anomaly != np.roll(is_anomaly, 1)
        region_changes[0] = True
        region_id = np.cumsum(region_changes)
        region_id[~is_anomaly] = 0
        
        unique_regions = np.unique(region_id)
        unique_regions = unique_regions[unique_regions > 0]
        
        for rid in unique_regions:
            mask = region_id == rid
            indices = np.where(mask)[0]
            
            if len(indices) < self.config.min_samples:
                continue
            
            region_timestamps = timestamps.iloc[indices]
            region_fused = fused_z[mask]
            region_confidence = confidence[mask]
            region_agreement = agreement[mask]
            
            # Compute duration
            start_time = region_timestamps.min()
            end_time = region_timestamps.max()
            duration_hours = (end_time - start_time).total_seconds() / 3600
            duration_minutes = duration_hours * 60
            
            if duration_minutes < self.config.min_duration_minutes:
                continue
            
            # Check cooldown
            if self._in_cooldown(self.equip_id, start_time):
                self._episodes_suppressed += 1
                continue
            
            # Compute severity
            peak_z = float(np.nanmax(region_fused))
            mean_z = float(np.nanmean(region_fused))
            severity = self._compute_severity(peak_z)
            
            # Compute confidence
            mean_agreement = float(np.nanmean(region_agreement))
            mean_confidence = float(np.nanmean(region_confidence))
            episode_confidence = mean_confidence * mean_agreement
            
            # Get regime at onset
            regime_at_onset = -1
            if regime_labels is not None and len(regime_labels) > 0:
                try:
                    regime_at_onset = int(regime_labels.iloc[indices[0]])
                except (IndexError, TypeError):
                    pass
            
            # Determine if still active
            is_last_point_anomaly = is_anomaly[-1] if len(is_anomaly) > 0 else False
            status = EpisodeStatus.ACTIVE if is_last_point_anomaly else EpisodeStatus.RESOLVED
            
            # Get affected detectors
            affected_detectors = []
            if detector_outputs:
                for det_name, det_df in detector_outputs.items():
                    z_col = f"{det_name}_z" if f"{det_name}_z" in det_df.columns else det_name
                    if z_col in det_df.columns:
                        det_z = det_df[z_col].iloc[indices]
                        if det_z.max() > threshold:
                            affected_detectors.append(det_name)
            
            # Generate episode ID
            episode_id = f"EP_{self.equip_id}_{start_time.strftime('%Y%m%d%H%M%S')}_{rid}"
            
            episode = Episode(
                id=episode_id,
                equip_id=self.equip_id,
                start_time=start_time,
                end_time=end_time if status == EpisodeStatus.RESOLVED else None,
                severity=severity,
                status=status,
                peak_z_score=peak_z,
                mean_z_score=mean_z,
                duration_hours=duration_hours,
                sample_count=len(indices),
                affected_detectors=affected_detectors,
                regime_at_onset=regime_at_onset,
                detector_agreement=mean_agreement,
                confidence=episode_confidence,
                false_positive_probability=1 - episode_confidence,
            )
            
            episodes.append(episode)
            self._episodes_detected += 1
            
            # Track active episodes
            if status == EpisodeStatus.ACTIVE:
                if self.equip_id not in self.active_episodes:
                    self.active_episodes[self.equip_id] = []
                self.active_episodes[self.equip_id].append(episode)
            else:
                # Set cooldown
                self._cooldown_until[self.equip_id] = end_time + pd.Timedelta(hours=self.config.cooldown_hours)
        
        return episodes
    
    def _compute_severity(self, peak_z: float) -> EpisodeSeverity:
        """Determine severity from peak z-score."""
        if peak_z >= self.config.threshold_critical:
            return EpisodeSeverity.CRITICAL
        elif peak_z >= self.config.threshold_high:
            return EpisodeSeverity.HIGH
        elif peak_z >= self.config.threshold_medium:
            return EpisodeSeverity.MEDIUM
        return EpisodeSeverity.LOW
    
    def _in_cooldown(self, equip_id: int, timestamp: pd.Timestamp) -> bool:
        """Check if equipment is in cooldown period."""
        cooldown_until = self._cooldown_until.get(equip_id)
        if cooldown_until is None:
            return False
        return timestamp < cooldown_until
    
    def get_active_episodes(self, equip_id: Optional[int] = None) -> List[Episode]:
        """Get all currently active episodes."""
        if equip_id is not None:
            return self.active_episodes.get(equip_id, [])
        
        all_active = []
        for eps in self.active_episodes.values():
            all_active.extend(eps)
        return all_active
    
    def resolve_episode(
        self,
        episode_id: str,
        end_time: pd.Timestamp,
        regime: Optional[int] = None,
    ) -> bool:
        """
        Resolve an active episode.
        
        Returns True if episode was found and resolved.
        """
        for equip_id, episodes in self.active_episodes.items():
            for i, ep in enumerate(episodes):
                if ep.id == episode_id:
                    ep.update_end_time(end_time, regime)
                    self.active_episodes[equip_id].pop(i)
                    
                    # Set cooldown
                    self._cooldown_until[equip_id] = end_time + pd.Timedelta(hours=self.config.cooldown_hours)
                    return True
        return False
    
    def suppress_episode(self, episode_id: str, reason: str, user: str) -> bool:
        """
        Suppress an episode (operator dismissal).
        
        Returns True if episode was found and suppressed.
        """
        for equip_id, episodes in self.active_episodes.items():
            for i, ep in enumerate(episodes):
                if ep.id == episode_id:
                    ep.status = EpisodeStatus.SUPPRESSED
                    ep.suppression_reason = reason
                    ep.acknowledged_by = user
                    ep.acknowledged_at = pd.Timestamp.now()
                    ep.updated_at = pd.Timestamp.now()
                    self.active_episodes[equip_id].pop(i)
                    return True
        return False
    
    def acknowledge_episode(self, episode_id: str, user: str) -> bool:
        """
        Acknowledge an episode without resolving it.
        
        Returns True if episode was found and acknowledged.
        """
        for episodes in self.active_episodes.values():
            for ep in episodes:
                if ep.id == episode_id:
                    ep.status = EpisodeStatus.ACKNOWLEDGED
                    ep.acknowledged_by = user
                    ep.acknowledged_at = pd.Timestamp.now()
                    ep.updated_at = pd.Timestamp.now()
                    return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get episode manager statistics."""
        active_count = sum(len(eps) for eps in self.active_episodes.values())
        return {
            "episodes_detected": self._episodes_detected,
            "episodes_suppressed": self._episodes_suppressed,
            "active_episodes": active_count,
            "equipment_with_active": len(self.active_episodes),
        }
    
    def clear_cooldowns(self) -> None:
        """Clear all cooldowns (for testing or reset)."""
        self._cooldown_until.clear()
    
    def reset(self) -> None:
        """Reset manager state."""
        self.active_episodes.clear()
        self._cooldown_until.clear()
        self._episodes_detected = 0
        self._episodes_suppressed = 0


# ============================================================================
# Convenience Functions
# ============================================================================

def merge_overlapping_episodes(episodes: List[Episode], gap_hours: float = 1.0) -> List[Episode]:
    """
    Merge episodes that are close together in time.
    
    Args:
        episodes: List of episodes to merge
        gap_hours: Maximum gap between episodes to merge
        
    Returns:
        Merged list of episodes
    """
    if len(episodes) <= 1:
        return episodes
    
    # Sort by start time
    sorted_eps = sorted(episodes, key=lambda e: e.start_time)
    
    merged = [sorted_eps[0]]
    
    for ep in sorted_eps[1:]:
        prev = merged[-1]
        
        # Check if should merge
        prev_end = prev.end_time or prev.start_time
        gap = (ep.start_time - prev_end).total_seconds() / 3600
        
        if gap <= gap_hours and prev.equip_id == ep.equip_id:
            # Merge episodes
            prev.end_time = ep.end_time
            prev.peak_z_score = max(prev.peak_z_score, ep.peak_z_score)
            prev.mean_z_score = (prev.mean_z_score + ep.mean_z_score) / 2
            prev.sample_count += ep.sample_count
            prev.duration_hours = (
                (prev.end_time or prev.start_time) - prev.start_time
            ).total_seconds() / 3600
            prev.severity = max(prev.severity, ep.severity, key=lambda s: s.value)
            prev.updated_at = pd.Timestamp.now()
        else:
            merged.append(ep)
    
    return merged


def episodes_to_dataframe(episodes: List[Episode]) -> pd.DataFrame:
    """Convert list of episodes to DataFrame."""
    if not episodes:
        return pd.DataFrame()
    
    return pd.DataFrame([ep.to_dict() for ep in episodes])
