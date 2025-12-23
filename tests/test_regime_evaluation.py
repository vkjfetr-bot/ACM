"""
Tests for core/regime_evaluation.py

Phase 2.9: Regime Evaluation Metrics
"""

import pytest
import numpy as np
from datetime import datetime

from core.regime_evaluation import (
    RegimeMetrics,
    RegimeEvaluator,
    PromotionCriteria,
    evaluate_regime_model,
)
from core.regime_manager import REGIME_UNKNOWN, REGIME_EMERGING


# =============================================================================
# Test RegimeMetrics
# =============================================================================

class TestRegimeMetrics:
    """Test RegimeMetrics dataclass."""
    
    def test_overall_score_perfect(self):
        """Perfect metrics should give high score."""
        metrics = RegimeMetrics(
            stability=1.0,
            novelty_rate=0.0,
            coverage=1.0,
            balance=1.0,
            transition_entropy=0.0,
            self_transition_rate=1.0,
            avg_silhouette=1.0,
            separation=5.0,
            sample_count=1000,
        )
        
        assert metrics.overall_score > 0.9
    
    def test_overall_score_poor(self):
        """Poor metrics should give low score."""
        metrics = RegimeMetrics(
            stability=0.3,
            novelty_rate=0.5,
            coverage=0.2,
            balance=0.2,
            transition_entropy=2.0,
            self_transition_rate=0.2,
            avg_silhouette=-0.5,
            separation=0.5,
            sample_count=100,
        )
        
        assert metrics.overall_score < 0.5
    
    def test_is_acceptable_good(self):
        """Good metrics should be acceptable."""
        metrics = RegimeMetrics(
            stability=0.90,
            novelty_rate=0.05,
            coverage=0.80,
            balance=0.80,
            transition_entropy=0.5,
            self_transition_rate=0.8,
            avg_silhouette=0.5,
            separation=3.0,
            sample_count=2000,
        )
        
        assert metrics.is_acceptable is True
    
    def test_is_acceptable_fails_stability(self):
        """Low stability should fail acceptance."""
        metrics = RegimeMetrics(
            stability=0.70,  # Below 0.80 threshold
            novelty_rate=0.05,
            coverage=0.80,
            balance=0.80,
            transition_entropy=0.5,
            self_transition_rate=0.8,
            avg_silhouette=0.5,
            separation=3.0,
            sample_count=2000,
        )
        
        assert metrics.is_acceptable is False
    
    def test_to_dict(self):
        """Test serialization to dict."""
        metrics = RegimeMetrics(
            stability=0.9,
            novelty_rate=0.1,
            coverage=0.8,
            balance=0.7,
            transition_entropy=0.5,
            self_transition_rate=0.6,
            avg_silhouette=0.4,
            separation=2.0,
            sample_count=500,
        )
        
        d = metrics.to_dict()
        
        assert d["Stability"] == 0.9
        assert d["NoveltyRate"] == 0.1
        assert d["Coverage"] == 0.8
        assert d["SampleCount"] == 500
        assert "OverallScore" in d


# =============================================================================
# Test RegimeEvaluator
# =============================================================================

class TestRegimeEvaluator:
    """Test RegimeEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        return RegimeEvaluator(stability_window=3)
    
    def test_evaluate_stable_labels(self, evaluator):
        """Stable labels should give high stability score."""
        # All same regime
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        metrics = evaluator.evaluate(labels)
        
        assert metrics.stability == 1.0
        assert metrics.novelty_rate == 0.0
    
    def test_evaluate_unstable_labels(self, evaluator):
        """Alternating labels should give low stability."""
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        metrics = evaluator.evaluate(labels)
        
        assert metrics.stability < 0.5
    
    def test_evaluate_novelty_rate(self, evaluator):
        """UNKNOWN labels should increase novelty rate."""
        labels = np.array([0, 0, REGIME_UNKNOWN, 0, 0, REGIME_UNKNOWN, 0, 0, 0, 0])
        
        metrics = evaluator.evaluate(labels)
        
        assert metrics.novelty_rate == 0.2  # 2/10
    
    def test_evaluate_coverage(self, evaluator):
        """Coverage should reflect regime usage."""
        # Only uses 2 of 3 possible regimes (0, 1, 2)
        labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        
        metrics = evaluator.evaluate(labels)
        
        # Coverage is 2/2 = 1.0 (uses all discovered regimes)
        assert metrics.coverage == 1.0
    
    def test_evaluate_balance_perfect(self, evaluator):
        """Perfectly balanced labels should give balance=1.0."""
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        
        metrics = evaluator.evaluate(labels)
        
        assert metrics.balance > 0.9
    
    def test_evaluate_balance_unbalanced(self, evaluator):
        """Unbalanced labels should give lower balance."""
        # 7:1 ratio is very unbalanced
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        
        metrics = evaluator.evaluate(labels)
        
        # Balance should be lower than perfectly balanced (which would be ~1.0)
        # 7:1 ratio gives entropy about 0.54
        assert metrics.balance < 0.7
    
    def test_evaluate_self_transition_high(self, evaluator):
        """Stable sequences should have high self-transition rate."""
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        metrics = evaluator.evaluate(labels)
        
        # 8 self-transitions, 1 regime change, so rate = 8/9
        assert metrics.self_transition_rate > 0.8
    
    def test_evaluate_empty_labels(self, evaluator):
        """Empty labels should return empty metrics."""
        labels = np.array([])
        
        metrics = evaluator.evaluate(labels)
        
        assert metrics.sample_count == 0
        assert metrics.novelty_rate == 1.0  # Default for empty
    
    def test_evaluate_with_centroids(self, evaluator):
        """Centroids should enable separation metric."""
        labels = np.array([0, 0, 1, 1])
        centroids = np.array([
            [0.0, 0.0],
            [3.0, 4.0],  # Distance = 5.0 from origin
        ])
        
        metrics = evaluator.evaluate(labels, centroids=centroids)
        
        assert metrics.separation == 5.0
    
    def test_evaluate_with_features(self, evaluator):
        """Features should enable silhouette calculation."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        features = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.9, 5.1],
        ])
        
        metrics = evaluator.evaluate(labels, features=features)
        
        # Well-separated clusters should have positive silhouette
        assert metrics.avg_silhouette > 0.5


# =============================================================================
# Test PromotionCriteria
# =============================================================================

class TestPromotionCriteria:
    """Test PromotionCriteria dataclass."""
    
    @pytest.fixture
    def criteria(self):
        return PromotionCriteria()
    
    def test_evaluate_passes(self, criteria):
        """Good metrics should pass promotion."""
        metrics = RegimeMetrics(
            stability=0.90,
            novelty_rate=0.05,
            coverage=0.80,
            balance=0.80,
            transition_entropy=0.5,
            self_transition_rate=0.8,
            avg_silhouette=0.5,
            separation=3.0,
            sample_count=2000,
        )
        
        can_promote, failures = criteria.evaluate(metrics, days_in_learning=10)
        
        assert can_promote is True
        assert len(failures) == 0
    
    def test_evaluate_fails_stability(self, criteria):
        """Low stability should fail promotion."""
        metrics = RegimeMetrics(
            stability=0.70,  # Below 0.85 threshold
            novelty_rate=0.05,
            coverage=0.80,
            balance=0.80,
            transition_entropy=0.5,
            self_transition_rate=0.8,
            avg_silhouette=0.5,
            separation=3.0,
            sample_count=2000,
        )
        
        can_promote, failures = criteria.evaluate(metrics, days_in_learning=10)
        
        assert can_promote is False
        assert any("Stability" in f for f in failures)
    
    def test_evaluate_fails_novelty(self, criteria):
        """High novelty rate should fail promotion."""
        metrics = RegimeMetrics(
            stability=0.90,
            novelty_rate=0.20,  # Above 0.10 threshold
            coverage=0.80,
            balance=0.80,
            transition_entropy=0.5,
            self_transition_rate=0.8,
            avg_silhouette=0.5,
            separation=3.0,
            sample_count=2000,
        )
        
        can_promote, failures = criteria.evaluate(metrics, days_in_learning=10)
        
        assert can_promote is False
        assert any("Novelty" in f for f in failures)
    
    def test_evaluate_fails_sample_count(self, criteria):
        """Low sample count should fail promotion."""
        metrics = RegimeMetrics(
            stability=0.90,
            novelty_rate=0.05,
            coverage=0.80,
            balance=0.80,
            transition_entropy=0.5,
            self_transition_rate=0.8,
            avg_silhouette=0.5,
            separation=3.0,
            sample_count=500,  # Below 1000 threshold
        )
        
        can_promote, failures = criteria.evaluate(metrics, days_in_learning=10)
        
        assert can_promote is False
        assert any("Sample count" in f for f in failures)
    
    def test_evaluate_fails_days(self, criteria):
        """Few days in learning should fail promotion."""
        metrics = RegimeMetrics(
            stability=0.90,
            novelty_rate=0.05,
            coverage=0.80,
            balance=0.80,
            transition_entropy=0.5,
            self_transition_rate=0.8,
            avg_silhouette=0.5,
            separation=3.0,
            sample_count=2000,
        )
        
        can_promote, failures = criteria.evaluate(metrics, days_in_learning=3)  # Below 7
        
        assert can_promote is False
        assert any("Days" in f for f in failures)


# =============================================================================
# Test Helper Function
# =============================================================================

class TestEvaluateRegimeModel:
    """Test evaluate_regime_model convenience function."""
    
    def test_basic_evaluation(self):
        """Test basic usage of helper function."""
        labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        
        metrics = evaluate_regime_model(labels)
        
        assert metrics.sample_count == 10
        assert 0 <= metrics.stability <= 1
        assert 0 <= metrics.novelty_rate <= 1
    
    def test_with_centroids(self):
        """Test with centroids for separation."""
        labels = np.array([0, 0, 1, 1])
        centroids = np.array([[0, 0], [2, 0]])
        
        metrics = evaluate_regime_model(labels, centroids=centroids)
        
        assert metrics.separation == 2.0
