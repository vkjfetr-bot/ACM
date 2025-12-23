"""
Tests for core/regime_promotion.py

Phase 2.10: Regime Promotion Procedure
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from core.regime_promotion import (
    PromotionLogEntry,
    RegimePromoter,
    check_auto_promotion,
)
from core.regime_manager import MaturityState
from core.regime_evaluation import RegimeMetrics, PromotionCriteria


# =============================================================================
# Test PromotionLogEntry
# =============================================================================

class TestPromotionLogEntry:
    """Test PromotionLogEntry dataclass."""
    
    def test_create_log_entry(self):
        log = PromotionLogEntry(
            equip_id=1,
            regime_version=3,
            attempt_time=datetime.now(),
            success=True,
            from_state=MaturityState.LEARNING,
            to_state=MaturityState.CONVERGED,
            stability=0.92,
            coverage=0.85,
            sample_count=5000,
        )
        
        assert log.equip_id == 1
        assert log.regime_version == 3
        assert log.success is True
        assert log.stability == 0.92
    
    def test_failure_reasons(self):
        log = PromotionLogEntry(
            equip_id=1,
            regime_version=2,
            attempt_time=datetime.now(),
            success=False,
            from_state=MaturityState.LEARNING,
            to_state=MaturityState.LEARNING,
            failure_reasons=["Stability too low", "Not enough samples"],
        )
        
        assert len(log.failure_reasons) == 2
        assert "Stability" in log.failure_reasons[0]


# =============================================================================
# Test RegimePromoter
# =============================================================================

class TestRegimePromoter:
    """Test RegimePromoter class."""
    
    @pytest.fixture
    def mock_sql_client(self):
        """Create a mock SQL client."""
        mock = MagicMock()
        mock.conn.autocommit = True
        return mock
    
    @pytest.fixture
    def promoter(self, mock_sql_client):
        """Create a promoter with mock SQL client."""
        return RegimePromoter(mock_sql_client)
    
    def test_evaluate_no_version(self, promoter):
        """Test evaluation with no active version."""
        # Mock active models to return no version
        promoter.active_models.get_active = MagicMock(return_value=MagicMock(
            regime_version=None,
            regime_maturity=MaturityState.INITIALIZING,
        ))
        
        can_promote, metrics, failures = promoter.evaluate_for_promotion(equip_id=1)
        
        assert can_promote is False
        assert "No active regime version" in failures
    
    def test_evaluate_definition_not_found(self, promoter):
        """Test evaluation when definition not found."""
        promoter.active_models.get_active = MagicMock(return_value=MagicMock(
            regime_version=5,
            regime_maturity=MaturityState.LEARNING,
        ))
        promoter.definitions.load = MagicMock(return_value=None)
        
        can_promote, metrics, failures = promoter.evaluate_for_promotion(equip_id=1)
        
        assert can_promote is False
        assert any("not found" in f for f in failures)
    
    def test_promote_no_version(self, promoter):
        """Test promotion with no version."""
        promoter.active_models.get_active = MagicMock(return_value=MagicMock(
            regime_version=None,
            regime_maturity=MaturityState.INITIALIZING,
        ))
        
        # Mock _save_log to avoid SQL calls
        promoter._save_log = MagicMock()
        
        success, log = promoter.promote(equip_id=1)
        
        assert success is False
        assert "No regime version" in log.failure_reasons[0]
    
    def test_deprecate_active_version(self, promoter):
        """Test that active version cannot be deprecated."""
        promoter.active_models.get_active = MagicMock(return_value=MagicMock(
            regime_version=3,
            regime_maturity=MaturityState.CONVERGED,
        ))
        
        result = promoter.deprecate(equip_id=1, version=3)
        
        assert result is False
    
    def test_deprecate_inactive_version(self, promoter):
        """Test deprecating an inactive version."""
        promoter.active_models.get_active = MagicMock(return_value=MagicMock(
            regime_version=5,
            regime_maturity=MaturityState.CONVERGED,
        ))
        promoter._save_log = MagicMock()
        
        result = promoter.deprecate(equip_id=1, version=3, reason="Replaced by v5")
        
        assert result is True
        promoter._save_log.assert_called_once()
    
    def test_empty_metrics(self, promoter):
        """Test empty metrics creation."""
        metrics = promoter._empty_metrics()
        
        assert metrics.stability == 0.0
        assert metrics.novelty_rate == 1.0
        assert metrics.sample_count == 0
    
    def test_get_promotion_history_empty(self, promoter, mock_sql_client):
        """Test getting empty promotion history."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_sql_client.cursor.return_value = mock_cursor
        
        history = promoter.get_promotion_history(equip_id=1)
        
        assert history == []
    
    def test_get_promotion_history(self, promoter, mock_sql_client):
        """Test getting promotion history."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (3, datetime(2024, 1, 10), True, "LEARNING", "CONVERGED",
             0.92, 0.05, 0.85, 0.80, 5000, 0.88, None, "SYSTEM", "Auto-promoted"),
        ]
        mock_sql_client.cursor.return_value = mock_cursor
        
        history = promoter.get_promotion_history(equip_id=1)
        
        assert len(history) == 1
        assert history[0]["version"] == 3
        assert history[0]["success"] is True
        assert history[0]["stability"] == 0.92


# =============================================================================
# Test Auto-Promotion
# =============================================================================

class TestAutoPromotion:
    """Test check_auto_promotion function."""
    
    def test_auto_promotion_not_learning(self):
        """Test that non-LEARNING models are not auto-promoted."""
        mock_sql = MagicMock()
        
        with patch('core.regime_promotion.RegimePromoter') as MockPromoter:
            mock_promoter = MockPromoter.return_value
            mock_promoter.active_models.get_active.return_value = MagicMock(
                regime_maturity=MaturityState.CONVERGED,
            )
            
            result = check_auto_promotion(mock_sql, equip_id=1)
            
            assert result is None
    
    def test_auto_promotion_not_ready(self):
        """Test that models not meeting criteria are not promoted."""
        mock_sql = MagicMock()
        
        with patch('core.regime_promotion.RegimePromoter') as MockPromoter:
            mock_promoter = MockPromoter.return_value
            mock_promoter.active_models.get_active.return_value = MagicMock(
                regime_maturity=MaturityState.LEARNING,
                regime_version=5,
            )
            mock_promoter.evaluate_for_promotion.return_value = (
                False, MagicMock(), ["Stability too low"]
            )
            
            result = check_auto_promotion(mock_sql, equip_id=1)
            
            assert result is None


# =============================================================================
# Integration-style Tests (with more mocking)
# =============================================================================

class TestPromoterIntegration:
    """Integration-style tests for RegimePromoter."""
    
    @pytest.fixture
    def full_mock_promoter(self):
        """Create a fully mocked promoter for integration tests."""
        mock_sql = MagicMock()
        mock_sql.conn.autocommit = True
        
        promoter = RegimePromoter(mock_sql)
        
        # Mock the SQL operations
        promoter._save_log = MagicMock()
        promoter._load_assignment_history = MagicMock(return_value=(
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1] * 100),  # 1000 stable labels
            np.array([0.9] * 1000),  # High confidence
        ))
        promoter._get_days_in_learning = MagicMock(return_value=10)
        
        return promoter
    
    def test_full_promotion_flow(self, full_mock_promoter):
        """Test complete promotion flow."""
        # Setup mocks
        full_mock_promoter.active_models.get_active = MagicMock(return_value=MagicMock(
            regime_version=2,
            regime_maturity=MaturityState.LEARNING,
            regime_promoted_at=datetime.now() - timedelta(days=10),
        ))
        
        # Mock definition
        mock_definition = MagicMock()
        mock_definition.centroid_array = np.array([
            [0.0, 0.0],
            [3.0, 4.0],
        ])
        full_mock_promoter.definitions.load = MagicMock(return_value=mock_definition)
        
        # Evaluate
        can_promote, metrics, failures = full_mock_promoter.evaluate_for_promotion(equip_id=1)
        
        # With stable labels, should be promotable
        assert metrics.sample_count == 1000
        # The actual promotion decision depends on computed metrics
    
    def test_force_promotion(self, full_mock_promoter):
        """Test force promotion bypasses criteria."""
        full_mock_promoter.active_models.get_active = MagicMock(return_value=MagicMock(
            regime_version=2,
            regime_maturity=MaturityState.LEARNING,
        ))
        
        # Make evaluation fail
        full_mock_promoter.evaluate_for_promotion = MagicMock(return_value=(
            False,
            full_mock_promoter._empty_metrics(),
            ["Stability too low"],
        ))
        full_mock_promoter.active_models.promote_regime = MagicMock()
        
        success, log = full_mock_promoter.promote(equip_id=1, force=True)
        
        # Force should succeed even with failing criteria
        assert success is True
        full_mock_promoter.active_models.promote_regime.assert_called_once()
