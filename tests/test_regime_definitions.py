"""
Tests for core/regime_definitions.py

Phase 2.5: ACM_RegimeDefinitions with versioning
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

from core.regime_definitions import (
    RegimeCentroid,
    RegimeDefinition,
    RegimeAssignment,
    RegimeDefinitionStore,
    create_regime_definition,
    get_regime_label_name,
    REGIME_UNKNOWN,
    REGIME_EMERGING,
    REGIME_GLOBAL,
)


# =============================================================================
# Test Constants
# =============================================================================

class TestConstants:
    """Test regime label constants."""
    
    def test_unknown_label(self):
        assert REGIME_UNKNOWN == -1
    
    def test_emerging_label(self):
        assert REGIME_EMERGING == -2
    
    def test_global_label(self):
        assert REGIME_GLOBAL == 0


# =============================================================================
# Test RegimeCentroid
# =============================================================================

class TestRegimeCentroid:
    """Test RegimeCentroid dataclass."""
    
    def test_create_centroid(self):
        centroid = RegimeCentroid(
            label=0,
            centroid=np.array([1.0, 2.0, 3.0]),
            radius=0.5,
            n_points=100,
        )
        
        assert centroid.label == 0
        np.testing.assert_array_equal(centroid.centroid, [1.0, 2.0, 3.0])
        assert centroid.radius == 0.5
        assert centroid.n_points == 100
    
    def test_to_dict(self):
        centroid = RegimeCentroid(
            label=1,
            centroid=np.array([1.0, 2.0]),
            radius=0.3,
            n_points=50,
        )
        
        d = centroid.to_dict()
        
        assert d["label"] == 1
        assert d["centroid"] == [1.0, 2.0]
        assert d["radius"] == 0.3
        assert d["n_points"] == 50
    
    def test_from_dict(self):
        data = {
            "label": 2,
            "centroid": [3.0, 4.0, 5.0],
            "radius": 0.7,
            "n_points": 200,
        }
        
        centroid = RegimeCentroid.from_dict(data)
        
        assert centroid.label == 2
        np.testing.assert_array_equal(centroid.centroid, [3.0, 4.0, 5.0])
        assert centroid.radius == 0.7
        assert centroid.n_points == 200
    
    def test_from_dict_defaults(self):
        data = {
            "label": 0,
            "centroid": [1.0],
        }
        
        centroid = RegimeCentroid.from_dict(data)
        
        assert centroid.radius == 0.0
        assert centroid.n_points == 0


# =============================================================================
# Test RegimeDefinition
# =============================================================================

class TestRegimeDefinition:
    """Test RegimeDefinition dataclass."""
    
    @pytest.fixture
    def sample_definition(self):
        """Create a sample regime definition."""
        centroids = [
            RegimeCentroid(0, np.array([0.0, 0.0]), 1.0, 100),
            RegimeCentroid(1, np.array([5.0, 5.0]), 1.0, 100),
        ]
        
        return RegimeDefinition(
            equip_id=1,
            version=1,
            num_regimes=2,
            centroids=centroids,
            feature_columns=["feat_a", "feat_b"],
            scaler_mean=np.array([0.0, 0.0]),
            scaler_scale=np.array([1.0, 1.0]),
            training_row_count=200,
        )
    
    def test_create_definition(self, sample_definition):
        assert sample_definition.equip_id == 1
        assert sample_definition.version == 1
        assert sample_definition.num_regimes == 2
        assert len(sample_definition.centroids) == 2
    
    def test_centroid_array(self, sample_definition):
        arr = sample_definition.centroid_array
        
        assert arr.shape == (2, 2)
        np.testing.assert_array_equal(arr[0], [0.0, 0.0])
        np.testing.assert_array_equal(arr[1], [5.0, 5.0])
    
    def test_get_regime_labels(self, sample_definition):
        labels = sample_definition.get_regime_labels()
        
        assert labels == [0, 1]
    
    def test_assign_regime_near_centroid(self, sample_definition):
        # Point near centroid 0
        label, conf = sample_definition.assign_regime(np.array([0.1, 0.1]))
        
        assert label == 0
        assert conf > 0.5
    
    def test_assign_regime_far_from_all(self, sample_definition):
        # Point very far from both centroids
        label, conf = sample_definition.assign_regime(np.array([100.0, 100.0]))
        
        assert label == REGIME_UNKNOWN
    
    def test_assign_regimes_batch(self, sample_definition):
        X = np.array([
            [0.0, 0.0],   # Near centroid 0
            [5.0, 5.0],   # Near centroid 1
            [0.5, 0.5],   # Still near centroid 0
        ])
        
        labels, confidences = sample_definition.assign_regimes_batch(X)
        
        assert len(labels) == 3
        assert labels[0] == 0
        assert labels[1] == 1
        assert labels[2] == 0
    
    def test_to_json_and_back(self, sample_definition):
        json_str = sample_definition.to_json()
        
        restored = RegimeDefinition.from_json(json_str)
        
        assert restored.equip_id == sample_definition.equip_id
        assert restored.version == sample_definition.version
        assert restored.num_regimes == sample_definition.num_regimes
        assert len(restored.centroids) == len(sample_definition.centroids)
        assert restored.feature_columns == sample_definition.feature_columns
        np.testing.assert_array_equal(restored.scaler_mean, sample_definition.scaler_mean)
    
    def test_json_with_transition_matrix(self, sample_definition):
        sample_definition.transition_matrix = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
        ])
        
        json_str = sample_definition.to_json()
        restored = RegimeDefinition.from_json(json_str)
        
        np.testing.assert_array_equal(
            restored.transition_matrix,
            sample_definition.transition_matrix,
        )


# =============================================================================
# Test RegimeAssignment
# =============================================================================

class TestRegimeAssignment:
    """Test RegimeAssignment dataclass."""
    
    def test_is_unknown(self):
        assignment = RegimeAssignment(
            timestamp=datetime.now(),
            regime_label=REGIME_UNKNOWN,
            regime_version=1,
            confidence=0.1,
        )
        
        assert assignment.is_unknown is True
        assert assignment.is_emerging is False
        assert assignment.is_known is False
    
    def test_is_emerging(self):
        assignment = RegimeAssignment(
            timestamp=datetime.now(),
            regime_label=REGIME_EMERGING,
            regime_version=1,
            confidence=0.3,
        )
        
        assert assignment.is_unknown is False
        assert assignment.is_emerging is True
        assert assignment.is_known is False
    
    def test_is_known(self):
        assignment = RegimeAssignment(
            timestamp=datetime.now(),
            regime_label=0,
            regime_version=1,
            confidence=0.9,
        )
        
        assert assignment.is_unknown is False
        assert assignment.is_emerging is False
        assert assignment.is_known is True


# =============================================================================
# Test Factory Function
# =============================================================================

class TestCreateRegimeDefinition:
    """Test create_regime_definition factory."""
    
    def test_basic_creation(self):
        centroids = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ])
        
        definition = create_regime_definition(
            equip_id=1,
            centroids=centroids,
            feature_columns=["f1", "f2"],
            scaler_mean=np.array([0.0, 0.0]),
            scaler_scale=np.array([1.0, 1.0]),
        )
        
        assert definition.equip_id == 1
        assert definition.num_regimes == 3
        assert len(definition.centroids) == 3
        assert definition.centroids[0].label == 0
        assert definition.centroids[1].label == 1
        assert definition.centroids[2].label == 2
    
    def test_with_training_df(self):
        import pandas as pd
        
        training_df = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
            "value": np.random.randn(100),
        })
        
        definition = create_regime_definition(
            equip_id=1,
            centroids=np.array([[0.0]]),
            feature_columns=["value"],
            scaler_mean=np.array([0.0]),
            scaler_scale=np.array([1.0]),
            training_df=training_df,
        )
        
        assert definition.training_row_count == 100
        assert definition.training_start_time == training_df["Timestamp"].min()
        assert definition.training_end_time == training_df["Timestamp"].max()


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestGetRegimeLabelName:
    """Test get_regime_label_name helper."""
    
    def test_unknown(self):
        assert get_regime_label_name(REGIME_UNKNOWN) == "UNKNOWN"
    
    def test_emerging(self):
        assert get_regime_label_name(REGIME_EMERGING) == "EMERGING"
    
    def test_global(self):
        assert get_regime_label_name(REGIME_GLOBAL) == "GLOBAL"
    
    def test_numbered_regime(self):
        assert get_regime_label_name(1) == "REGIME_1"
        assert get_regime_label_name(5) == "REGIME_5"


# =============================================================================
# Test RegimeDefinitionStore (with mocked SQL)
# =============================================================================

class TestRegimeDefinitionStore:
    """Test RegimeDefinitionStore with mocked SQL client."""
    
    @pytest.fixture
    def mock_sql_client(self):
        """Create a mock SQL client."""
        mock = MagicMock()
        mock.conn.autocommit = True
        return mock
    
    @pytest.fixture
    def store(self, mock_sql_client):
        """Create a store with mock SQL client."""
        return RegimeDefinitionStore(mock_sql_client)
    
    def test_save_gets_next_version(self, store, mock_sql_client):
        # Setup mock cursor
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            (1,),  # Next version
        ]
        mock_sql_client.cursor.return_value = mock_cursor
        
        definition = RegimeDefinition(
            equip_id=1,
            version=0,
            num_regimes=2,
            centroids=[
                RegimeCentroid(0, np.array([0.0, 0.0]), 1.0, 50),
                RegimeCentroid(1, np.array([1.0, 1.0]), 1.0, 50),
            ],
            feature_columns=["f1", "f2"],
            scaler_mean=np.array([0.0, 0.0]),
            scaler_scale=np.array([1.0, 1.0]),
        )
        
        version = store.save(definition)
        
        assert version == 1
        assert mock_cursor.execute.call_count == 2  # SELECT + INSERT
    
    def test_list_versions(self, store, mock_sql_client):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (2, 3, 1000, datetime(2024, 1, 2), datetime(2024, 1, 1), datetime(2024, 1, 3)),
            (1, 2, 500, datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 2)),
        ]
        mock_sql_client.cursor.return_value = mock_cursor
        
        versions = store.list_versions(equip_id=1)
        
        assert len(versions) == 2
        assert versions[0]["version"] == 2
        assert versions[0]["num_regimes"] == 3
        assert versions[1]["version"] == 1
    
    def test_cache_management(self, store):
        # Cache should start empty
        assert len(store._cache) == 0
        
        # Clear shouldn't fail on empty cache
        store.clear_cache()
        
        assert len(store._cache) == 0
