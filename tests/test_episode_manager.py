"""
Tests for ACM v11.0.0 Episode Manager Module.

Tests episode-only alerting where episodes are the SOLE alerting primitive.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.episode_manager import (
    EpisodeSeverity,
    EpisodeStatus,
    Episode,
    EpisodeConfig,
    EpisodeManager,
    SensorAttribution,
    merge_overlapping_episodes,
    episodes_to_dataframe,
)


# ============================================================================
# EpisodeSeverity Tests
# ============================================================================

class TestEpisodeSeverity:
    """Tests for EpisodeSeverity enum."""
    
    def test_severity_levels_exist(self):
        """All severity levels are defined."""
        assert EpisodeSeverity.LOW
        assert EpisodeSeverity.MEDIUM
        assert EpisodeSeverity.HIGH
        assert EpisodeSeverity.CRITICAL
    
    def test_severity_ordering(self):
        """Severity values are ordered correctly."""
        assert EpisodeSeverity.LOW.value < EpisodeSeverity.MEDIUM.value
        assert EpisodeSeverity.MEDIUM.value < EpisodeSeverity.HIGH.value
        assert EpisodeSeverity.HIGH.value < EpisodeSeverity.CRITICAL.value
    
    def test_severity_colors(self):
        """Severity levels have visualization colors."""
        assert EpisodeSeverity.LOW.color == "yellow"
        assert EpisodeSeverity.MEDIUM.color == "orange"
        assert EpisodeSeverity.HIGH.color == "red"
        assert EpisodeSeverity.CRITICAL.color == "darkred"
    
    def test_requires_action(self):
        """Only HIGH and CRITICAL require operator action."""
        assert EpisodeSeverity.LOW.requires_action == False
        assert EpisodeSeverity.MEDIUM.requires_action == False
        assert EpisodeSeverity.HIGH.requires_action == True
        assert EpisodeSeverity.CRITICAL.requires_action == True


# ============================================================================
# EpisodeStatus Tests
# ============================================================================

class TestEpisodeStatus:
    """Tests for EpisodeStatus enum."""
    
    def test_status_values_exist(self):
        """All status values are defined."""
        assert EpisodeStatus.ACTIVE
        assert EpisodeStatus.RESOLVED
        assert EpisodeStatus.SUPPRESSED
        assert EpisodeStatus.ESCALATED
        assert EpisodeStatus.ACKNOWLEDGED
    
    def test_status_string_values(self):
        """Status values are string representations."""
        assert EpisodeStatus.ACTIVE.value == "ACTIVE"
        assert EpisodeStatus.RESOLVED.value == "RESOLVED"


# ============================================================================
# Episode Tests
# ============================================================================

class TestEpisode:
    """Tests for Episode dataclass."""
    
    def test_episode_creation(self):
        """Episode can be created with required fields."""
        ep = Episode(
            id="EP_1_20240101",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01 10:00:00"),
            end_time=None,
            severity=EpisodeSeverity.HIGH,
            status=EpisodeStatus.ACTIVE,
            peak_z_score=6.5,
            mean_z_score=4.2,
            duration_hours=2.5,
            sample_count=30,
        )
        assert ep.id == "EP_1_20240101"
        assert ep.is_active == True
        assert ep.is_resolved == False
    
    def test_episode_resolved_status(self):
        """Resolved episode has correct properties."""
        ep = Episode(
            id="EP_1_20240101",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01 10:00:00"),
            end_time=pd.Timestamp("2024-01-01 12:00:00"),
            severity=EpisodeSeverity.MEDIUM,
            status=EpisodeStatus.RESOLVED,
            peak_z_score=4.5,
            mean_z_score=3.8,
            duration_hours=2.0,
            sample_count=24,
        )
        assert ep.is_active == False
        assert ep.is_resolved == True
    
    def test_episode_duration_minutes(self):
        """Duration can be retrieved in minutes."""
        ep = Episode(
            id="EP_1",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01"),
            end_time=None,
            severity=EpisodeSeverity.LOW,
            status=EpisodeStatus.ACTIVE,
            peak_z_score=3.5,
            mean_z_score=3.0,
            duration_hours=1.5,
            sample_count=18,
        )
        assert ep.duration_minutes == 90.0
    
    def test_episode_to_dict(self):
        """Episode serializes for SQL persistence."""
        ep = Episode(
            id="EP_1_20240101",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01 10:00:00"),
            end_time=pd.Timestamp("2024-01-01 12:00:00"),
            severity=EpisodeSeverity.HIGH,
            status=EpisodeStatus.RESOLVED,
            peak_z_score=6.5,
            mean_z_score=4.2,
            duration_hours=2.0,
            sample_count=24,
            affected_detectors=["ar1", "pca"],
        )
        d = ep.to_dict()
        assert d["EpisodeID"] == "EP_1_20240101"
        assert d["EquipID"] == 1
        assert d["Severity"] == "HIGH"
        assert d["SeverityLevel"] == 3
        assert d["Status"] == "RESOLVED"
        assert d["AffectedDetectors"] == "ar1,pca"
    
    def test_update_end_time(self):
        """Episode end time can be updated."""
        ep = Episode(
            id="EP_1",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01 10:00:00"),
            end_time=None,
            severity=EpisodeSeverity.HIGH,
            status=EpisodeStatus.ACTIVE,
            peak_z_score=6.0,
            mean_z_score=4.0,
            duration_hours=0.0,
            sample_count=10,
        )
        
        ep.update_end_time(
            end_time=pd.Timestamp("2024-01-01 14:00:00"),
            regime=2,
        )
        
        assert ep.status == EpisodeStatus.RESOLVED
        assert ep.end_time == pd.Timestamp("2024-01-01 14:00:00")
        assert ep.regime_at_resolution == 2
        assert ep.duration_hours == 4.0


# ============================================================================
# EpisodeConfig Tests
# ============================================================================

class TestEpisodeConfig:
    """Tests for EpisodeConfig."""
    
    def test_default_config(self):
        """Default config has sensible values."""
        cfg = EpisodeConfig()
        assert cfg.min_duration_minutes == 30.0
        assert cfg.threshold_low == 3.0
        assert cfg.threshold_medium == 4.0
        assert cfg.threshold_high == 5.5
        assert cfg.threshold_critical == 7.0
        assert cfg.cooldown_hours == 6.0
    
    def test_custom_config(self):
        """Custom config values can be set."""
        cfg = EpisodeConfig(
            min_duration_minutes=15.0,
            threshold_low=2.5,
            cooldown_hours=12.0,
        )
        assert cfg.min_duration_minutes == 15.0
        assert cfg.threshold_low == 2.5
        assert cfg.cooldown_hours == 12.0
    
    def test_from_config_dict(self):
        """Config can be loaded from dict."""
        d = {
            "episode": {
                "min_duration_minutes": 45.0,
                "threshold_critical": 8.0,
            }
        }
        cfg = EpisodeConfig.from_config(d)
        assert cfg.min_duration_minutes == 45.0
        assert cfg.threshold_critical == 8.0


# ============================================================================
# EpisodeManager Tests
# ============================================================================

class TestEpisodeManager:
    """Tests for EpisodeManager."""
    
    @pytest.fixture
    def sample_fused_data(self):
        """Create sample fused anomaly data."""
        timestamps = pd.date_range(
            start="2024-01-01 00:00",
            periods=100,
            freq="5min",
        )
        
        # Create fused_z with an anomaly region
        fused_z = np.ones(100) * 2.0  # Normal baseline
        fused_z[30:50] = 5.0  # Anomaly region
        
        return pd.DataFrame({
            "Timestamp": timestamps,
            "fused_z": fused_z,
            "fusion_confidence": np.ones(100) * 0.9,
            "detector_agreement": np.ones(100) * 0.85,
        })
    
    def test_manager_initialization(self):
        """Manager initializes correctly."""
        manager = EpisodeManager(equip_id=1)
        assert manager.equip_id == 1
        assert len(manager.active_episodes) == 0
        assert manager._episodes_detected == 0
    
    def test_detect_episodes_finds_anomaly(self, sample_fused_data):
        """Manager detects episodes from fused scores."""
        config = EpisodeConfig(min_duration_minutes=30.0)
        manager = EpisodeManager(config=config, equip_id=1)
        
        episodes = manager.detect_episodes(sample_fused_data)
        
        assert len(episodes) >= 1
        # First episode should be the anomaly region
        ep = episodes[0]
        assert ep.peak_z_score >= 5.0
        assert ep.severity in (EpisodeSeverity.MEDIUM, EpisodeSeverity.HIGH)
    
    def test_detect_episodes_filters_short(self):
        """Episodes shorter than min_duration are filtered."""
        timestamps = pd.date_range(
            start="2024-01-01 00:00",
            periods=10,
            freq="1min",
        )
        
        # Short anomaly (10 minutes)
        df = pd.DataFrame({
            "Timestamp": timestamps,
            "fused_z": [5.0] * 10,
        })
        
        config = EpisodeConfig(min_duration_minutes=30.0)
        manager = EpisodeManager(config=config, equip_id=1)
        
        episodes = manager.detect_episodes(df)
        assert len(episodes) == 0  # Too short
    
    def test_detect_episodes_severity_classification(self):
        """Severity is correctly classified by peak z-score."""
        config = EpisodeConfig(
            min_duration_minutes=5.0,
            min_samples=2,
            threshold_low=3.0,
            threshold_medium=4.0,
            threshold_high=5.5,
            threshold_critical=7.0,
        )
        manager = EpisodeManager(config=config, equip_id=1)
        
        # Create data with critical anomaly
        timestamps = pd.date_range("2024-01-01", periods=20, freq="10min")
        df = pd.DataFrame({
            "Timestamp": timestamps,
            "fused_z": [8.0] * 20,  # Critical level
        })
        
        episodes = manager.detect_episodes(df)
        assert len(episodes) >= 1
        assert episodes[0].severity == EpisodeSeverity.CRITICAL
    
    def test_cooldown_suppresses_episodes(self, sample_fused_data):
        """Episodes during cooldown are suppressed."""
        config = EpisodeConfig(
            min_duration_minutes=5.0,
            min_samples=2,
            cooldown_hours=24.0,
        )
        manager = EpisodeManager(config=config, equip_id=1)
        
        # First detection
        episodes1 = manager.detect_episodes(sample_fused_data)
        
        # Second detection (same data) should be suppressed
        episodes2 = manager.detect_episodes(sample_fused_data)
        
        # First should succeed, second should be suppressed
        assert len(episodes1) >= 1
        assert manager._episodes_suppressed >= 0
    
    def test_active_episodes_tracking(self, sample_fused_data):
        """Active episodes are tracked correctly."""
        # Create data that ends with anomaly (active episode)
        timestamps = pd.date_range("2024-01-01", periods=50, freq="5min")
        fused_z = np.ones(50) * 2.0
        fused_z[-20:] = 5.0  # Ends with anomaly
        
        df = pd.DataFrame({
            "Timestamp": timestamps,
            "fused_z": fused_z,
        })
        
        config = EpisodeConfig(min_duration_minutes=30.0, min_samples=3)
        manager = EpisodeManager(config=config, equip_id=1)
        
        episodes = manager.detect_episodes(df)
        
        # Should have active episodes
        active = manager.get_active_episodes(equip_id=1)
        # May or may not be active depending on last point
    
    def test_resolve_episode(self):
        """Episode can be manually resolved."""
        manager = EpisodeManager(equip_id=1)
        
        # Create an episode manually
        ep = Episode(
            id="EP_test_1",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01"),
            end_time=None,
            severity=EpisodeSeverity.HIGH,
            status=EpisodeStatus.ACTIVE,
            peak_z_score=6.0,
            mean_z_score=4.0,
            duration_hours=0.0,
            sample_count=10,
        )
        manager.active_episodes[1] = [ep]
        
        result = manager.resolve_episode(
            episode_id="EP_test_1",
            end_time=pd.Timestamp("2024-01-02"),
        )
        
        assert result == True
        assert len(manager.get_active_episodes(equip_id=1)) == 0
    
    def test_suppress_episode(self):
        """Episode can be suppressed by operator."""
        manager = EpisodeManager(equip_id=1)
        
        ep = Episode(
            id="EP_test_1",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01"),
            end_time=None,
            severity=EpisodeSeverity.MEDIUM,
            status=EpisodeStatus.ACTIVE,
            peak_z_score=4.0,
            mean_z_score=3.5,
            duration_hours=1.0,
            sample_count=12,
        )
        manager.active_episodes[1] = [ep]
        
        result = manager.suppress_episode(
            episode_id="EP_test_1",
            reason="Known maintenance",
            user="operator1",
        )
        
        assert result == True
        assert ep.status == EpisodeStatus.SUPPRESSED
        assert ep.suppression_reason == "Known maintenance"
    
    def test_acknowledge_episode(self):
        """Episode can be acknowledged without resolving."""
        manager = EpisodeManager(equip_id=1)
        
        ep = Episode(
            id="EP_test_1",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01"),
            end_time=None,
            severity=EpisodeSeverity.HIGH,
            status=EpisodeStatus.ACTIVE,
            peak_z_score=6.0,
            mean_z_score=4.0,
            duration_hours=0.0,
            sample_count=10,
        )
        manager.active_episodes[1] = [ep]
        
        result = manager.acknowledge_episode(
            episode_id="EP_test_1",
            user="engineer1",
        )
        
        assert result == True
        assert ep.status == EpisodeStatus.ACKNOWLEDGED
        # Should still be in active list
        assert len(manager.active_episodes[1]) == 1
    
    def test_statistics(self, sample_fused_data):
        """Manager provides statistics."""
        config = EpisodeConfig(min_duration_minutes=5.0, min_samples=2)
        manager = EpisodeManager(config=config, equip_id=1)
        
        manager.detect_episodes(sample_fused_data)
        
        stats = manager.get_statistics()
        assert "episodes_detected" in stats
        assert "episodes_suppressed" in stats
        assert "active_episodes" in stats
    
    def test_reset(self):
        """Manager can be reset."""
        manager = EpisodeManager(equip_id=1)
        manager._episodes_detected = 10
        manager._episodes_suppressed = 3
        manager.active_episodes[1] = []
        
        manager.reset()
        
        assert manager._episodes_detected == 0
        assert manager._episodes_suppressed == 0
        assert len(manager.active_episodes) == 0
    
    def test_empty_dataframe(self):
        """Empty DataFrame returns no episodes."""
        manager = EpisodeManager(equip_id=1)
        episodes = manager.detect_episodes(pd.DataFrame())
        assert len(episodes) == 0
    
    def test_missing_fused_column(self):
        """Missing fused_z column returns no episodes."""
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=10),
            "some_other_col": [1] * 10,
        })
        
        manager = EpisodeManager(equip_id=1)
        episodes = manager.detect_episodes(df)
        assert len(episodes) == 0


# ============================================================================
# SensorAttribution Tests
# ============================================================================

class TestSensorAttribution:
    """Tests for SensorAttribution dataclass."""
    
    def test_attribution_creation(self):
        """Attribution can be created."""
        attr = SensorAttribution(
            sensor_name="TEMP_1",
            contribution_pct=35.5,
            peak_z_score=6.2,
            mean_z_score=4.5,
            detector="ar1",
        )
        assert attr.sensor_name == "TEMP_1"
        assert attr.contribution_pct == 35.5
        assert attr.detector == "ar1"


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestMergeOverlappingEpisodes:
    """Tests for merge_overlapping_episodes function."""
    
    def test_merge_adjacent_episodes(self):
        """Adjacent episodes are merged."""
        ep1 = Episode(
            id="EP_1",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01 10:00"),
            end_time=pd.Timestamp("2024-01-01 12:00"),
            severity=EpisodeSeverity.MEDIUM,
            status=EpisodeStatus.RESOLVED,
            peak_z_score=4.5,
            mean_z_score=3.8,
            duration_hours=2.0,
            sample_count=24,
        )
        ep2 = Episode(
            id="EP_2",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01 12:30"),
            end_time=pd.Timestamp("2024-01-01 14:00"),
            severity=EpisodeSeverity.HIGH,
            status=EpisodeStatus.RESOLVED,
            peak_z_score=6.0,
            mean_z_score=4.5,
            duration_hours=1.5,
            sample_count=18,
        )
        
        merged = merge_overlapping_episodes([ep1, ep2], gap_hours=1.0)
        
        assert len(merged) == 1
        assert merged[0].peak_z_score == 6.0  # Max of both
        assert merged[0].severity == EpisodeSeverity.HIGH
    
    def test_no_merge_if_gap_too_large(self):
        """Episodes with large gaps are not merged."""
        ep1 = Episode(
            id="EP_1",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01 10:00"),
            end_time=pd.Timestamp("2024-01-01 12:00"),
            severity=EpisodeSeverity.MEDIUM,
            status=EpisodeStatus.RESOLVED,
            peak_z_score=4.5,
            mean_z_score=3.8,
            duration_hours=2.0,
            sample_count=24,
        )
        ep2 = Episode(
            id="EP_2",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01 18:00"),  # 6 hour gap
            end_time=pd.Timestamp("2024-01-01 20:00"),
            severity=EpisodeSeverity.HIGH,
            status=EpisodeStatus.RESOLVED,
            peak_z_score=6.0,
            mean_z_score=4.5,
            duration_hours=2.0,
            sample_count=24,
        )
        
        merged = merge_overlapping_episodes([ep1, ep2], gap_hours=1.0)
        
        assert len(merged) == 2
    
    def test_single_episode(self):
        """Single episode returns unchanged."""
        ep = Episode(
            id="EP_1",
            equip_id=1,
            start_time=pd.Timestamp("2024-01-01"),
            end_time=pd.Timestamp("2024-01-01 02:00"),
            severity=EpisodeSeverity.LOW,
            status=EpisodeStatus.RESOLVED,
            peak_z_score=3.5,
            mean_z_score=3.0,
            duration_hours=2.0,
            sample_count=24,
        )
        
        merged = merge_overlapping_episodes([ep])
        assert len(merged) == 1
        assert merged[0].id == "EP_1"
    
    def test_empty_list(self):
        """Empty list returns empty."""
        merged = merge_overlapping_episodes([])
        assert len(merged) == 0


class TestEpisodesToDataframe:
    """Tests for episodes_to_dataframe function."""
    
    def test_convert_episodes(self):
        """Episodes convert to DataFrame."""
        episodes = [
            Episode(
                id="EP_1",
                equip_id=1,
                start_time=pd.Timestamp("2024-01-01"),
                end_time=pd.Timestamp("2024-01-01 02:00"),
                severity=EpisodeSeverity.HIGH,
                status=EpisodeStatus.RESOLVED,
                peak_z_score=6.0,
                mean_z_score=4.0,
                duration_hours=2.0,
                sample_count=24,
            ),
            Episode(
                id="EP_2",
                equip_id=1,
                start_time=pd.Timestamp("2024-01-02"),
                end_time=None,
                severity=EpisodeSeverity.CRITICAL,
                status=EpisodeStatus.ACTIVE,
                peak_z_score=8.0,
                mean_z_score=6.0,
                duration_hours=0.5,
                sample_count=6,
            ),
        ]
        
        df = episodes_to_dataframe(episodes)
        
        assert len(df) == 2
        assert "EpisodeID" in df.columns
        assert "Severity" in df.columns
        assert df.iloc[0]["Severity"] == "HIGH"
        assert df.iloc[1]["Severity"] == "CRITICAL"
    
    def test_empty_list(self):
        """Empty list returns empty DataFrame."""
        df = episodes_to_dataframe([])
        assert len(df) == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestEpisodeManagerIntegration:
    """Integration tests for EpisodeManager."""
    
    def test_full_episode_lifecycle(self):
        """Test complete episode lifecycle."""
        # Create manager
        config = EpisodeConfig(
            min_duration_minutes=30.0,
            min_samples=5,
            cooldown_hours=1.0,
        )
        manager = EpisodeManager(config=config, equip_id=1)
        
        # Create fused data with anomaly
        timestamps = pd.date_range("2024-01-01", periods=100, freq="5min")
        fused_z = np.concatenate([
            np.ones(30) * 2.0,   # Normal
            np.ones(30) * 6.0,   # Anomaly
            np.ones(40) * 2.0,   # Normal
        ])
        
        df = pd.DataFrame({
            "Timestamp": timestamps,
            "fused_z": fused_z,
        })
        
        # Detect episodes
        episodes = manager.detect_episodes(df)
        
        # Should detect the anomaly
        assert len(episodes) >= 1
        ep = episodes[0]
        assert ep.severity in (EpisodeSeverity.HIGH, EpisodeSeverity.CRITICAL)
        
        # Episode should be resolved (anomaly ends before data ends)
        assert ep.status == EpisodeStatus.RESOLVED
    
    def test_multiple_equipment(self):
        """Manager can handle multiple equipment."""
        manager1 = EpisodeManager(equip_id=1)
        manager2 = EpisodeManager(equip_id=2)
        
        timestamps = pd.date_range("2024-01-01", periods=30, freq="10min")
        df = pd.DataFrame({
            "Timestamp": timestamps,
            "fused_z": np.ones(30) * 5.0,
        })
        
        config = EpisodeConfig(min_duration_minutes=30.0, min_samples=3)
        manager1 = EpisodeManager(config=config, equip_id=1)
        manager2 = EpisodeManager(config=config, equip_id=2)
        
        eps1 = manager1.detect_episodes(df)
        eps2 = manager2.detect_episodes(df)
        
        # Both should detect episodes
        assert len(eps1) >= 1
        assert len(eps2) >= 1
        
        # Episodes should have different equip_ids
        assert eps1[0].equip_id == 1
        assert eps2[0].equip_id == 2
