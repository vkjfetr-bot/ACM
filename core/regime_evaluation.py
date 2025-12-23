"""
Regime Evaluation Metrics for ACM v11.0.0

Provides quality metrics for regime models to support
promotion decisions and model comparison.

Phase 2.9 Implementation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

from core.observability import Console
from core.regime_manager import REGIME_UNKNOWN, REGIME_EMERGING


# =============================================================================
# Evaluation Metrics Dataclass
# =============================================================================

@dataclass
class RegimeMetrics:
    """
    Quality metrics for a regime model.
    
    These metrics are computed from assignment results and used
    for promotion decisions and model comparison.
    """
    
    # Core metrics (0-1, higher is better except where noted)
    stability: float           # % of points that don't change regime within window
    novelty_rate: float        # % of points assigned UNKNOWN/EMERGING (lower is better)
    coverage: float            # % of regimes used (ideal = 1.0)
    balance: float             # Entropy-based balance of regime usage
    
    # Transition metrics
    transition_entropy: float  # Entropy of transitions (lower = more predictable)
    self_transition_rate: float  # % of transitions that stay in same regime
    
    # Quality scores (computed from centroids)
    avg_silhouette: float      # Cluster quality (-1 to 1, higher is better)
    separation: float          # Inter-centroid distance (higher is better)
    
    # Metadata
    evaluated_at: Optional[datetime] = None
    sample_count: int = 0
    regime_version: Optional[int] = None
    
    @property
    def overall_score(self) -> float:
        """
        Compute overall quality score (0-1).
        
        Weighted combination of metrics favoring stability and coverage.
        """
        weights = {
            "stability": 0.25,
            "coverage": 0.20,
            "balance": 0.15,
            "self_transition_rate": 0.15,
            "avg_silhouette": 0.15,
            "separation": 0.10,
        }
        
        # Normalize separation to 0-1 range (assuming max 5.0)
        norm_separation = min(1.0, self.separation / 5.0)
        
        # Normalize silhouette from [-1, 1] to [0, 1]
        norm_silhouette = (self.avg_silhouette + 1.0) / 2.0
        
        score = (
            weights["stability"] * self.stability +
            weights["coverage"] * self.coverage +
            weights["balance"] * self.balance +
            weights["self_transition_rate"] * self.self_transition_rate +
            weights["avg_silhouette"] * norm_silhouette +
            weights["separation"] * norm_separation
        )
        
        # Penalize high novelty rate
        novelty_penalty = max(0.0, (self.novelty_rate - 0.05) * 2.0)
        score = max(0.0, score - novelty_penalty)
        
        return min(1.0, score)
    
    @property
    def is_acceptable(self) -> bool:
        """Check if metrics meet minimum thresholds for promotion."""
        return (
            self.stability >= 0.80 and
            self.novelty_rate <= 0.15 and
            self.coverage >= 0.50 and
            self.avg_silhouette >= 0.0
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for SQL storage."""
        return {
            "Stability": self.stability,
            "NoveltyRate": self.novelty_rate,
            "Coverage": self.coverage,
            "Balance": self.balance,
            "TransitionEntropy": self.transition_entropy,
            "SelfTransitionRate": self.self_transition_rate,
            "AvgSilhouette": self.avg_silhouette,
            "Separation": self.separation,
            "OverallScore": self.overall_score,
            "SampleCount": self.sample_count,
        }


# =============================================================================
# Regime Evaluator
# =============================================================================

class RegimeEvaluator:
    """
    Evaluates regime model quality from assignment results.
    
    Computes various metrics to assess:
    - Stability: How consistent are regime assignments over time?
    - Coverage: Are all discovered regimes being used?
    - Balance: Are regimes reasonably balanced in usage?
    - Novelty: How many points can't be assigned (UNKNOWN/EMERGING)?
    - Transitions: Are regime transitions predictable?
    """
    
    def __init__(self, stability_window: int = 6):
        """
        Initialize evaluator.
        
        Args:
            stability_window: Number of consecutive points to check for stability
        """
        self.stability_window = stability_window
    
    def evaluate(self,
                 labels: np.ndarray,
                 confidences: Optional[np.ndarray] = None,
                 centroids: Optional[np.ndarray] = None,
                 features: Optional[np.ndarray] = None) -> RegimeMetrics:
        """
        Evaluate regime model quality.
        
        Args:
            labels: Array of regime labels (including UNKNOWN/EMERGING)
            confidences: Optional array of assignment confidences
            centroids: Optional centroid matrix for separation metrics
            features: Optional feature matrix for silhouette computation
            
        Returns:
            RegimeMetrics with computed values
        """
        n_samples = len(labels)
        
        if n_samples == 0:
            return self._empty_metrics()
        
        # Known regimes (exclude UNKNOWN and EMERGING)
        known_mask = labels >= 0
        known_labels = labels[known_mask]
        unique_regimes = np.unique(known_labels) if len(known_labels) > 0 else np.array([])
        n_regimes = len(unique_regimes)
        
        # Core metrics
        stability = self._compute_stability(labels)
        novelty_rate = self._compute_novelty_rate(labels)
        coverage = self._compute_coverage(labels, n_regimes)
        balance = self._compute_balance(labels, n_regimes)
        
        # Transition metrics
        transition_entropy, self_transition_rate = self._compute_transition_metrics(labels)
        
        # Quality metrics (if centroids provided)
        avg_silhouette = 0.0
        separation = 0.0
        
        if centroids is not None and len(centroids) > 1:
            separation = self._compute_separation(centroids)
        
        if features is not None and len(known_labels) > 1 and n_regimes > 1:
            avg_silhouette = self._compute_silhouette(features[known_mask], known_labels)
        
        return RegimeMetrics(
            stability=stability,
            novelty_rate=novelty_rate,
            coverage=coverage,
            balance=balance,
            transition_entropy=transition_entropy,
            self_transition_rate=self_transition_rate,
            avg_silhouette=avg_silhouette,
            separation=separation,
            evaluated_at=datetime.now(),
            sample_count=n_samples,
        )
    
    def _compute_stability(self, labels: np.ndarray) -> float:
        """
        Compute stability as fraction of points in stable windows.
        
        A window is stable if all points have the same regime.
        """
        if len(labels) < self.stability_window:
            return 1.0 if len(np.unique(labels)) == 1 else 0.0
        
        stable_count = 0
        total_windows = len(labels) - self.stability_window + 1
        
        for i in range(total_windows):
            window = labels[i:i + self.stability_window]
            if len(np.unique(window)) == 1:
                stable_count += 1
        
        return stable_count / total_windows if total_windows > 0 else 0.0
    
    def _compute_novelty_rate(self, labels: np.ndarray) -> float:
        """Compute fraction of UNKNOWN/EMERGING assignments."""
        if len(labels) == 0:
            return 0.0
        
        novelty_count = np.sum((labels == REGIME_UNKNOWN) | (labels == REGIME_EMERGING))
        return novelty_count / len(labels)
    
    def _compute_coverage(self, labels: np.ndarray, expected_regimes: int) -> float:
        """Compute fraction of expected regimes that are actually used."""
        if expected_regimes == 0:
            return 1.0
        
        known_labels = labels[labels >= 0]
        if len(known_labels) == 0:
            return 0.0
        
        used_regimes = len(np.unique(known_labels))
        return used_regimes / expected_regimes
    
    def _compute_balance(self, labels: np.ndarray, n_regimes: int) -> float:
        """
        Compute balance as normalized entropy of regime distribution.
        
        1.0 = perfectly balanced, 0.0 = all in one regime.
        """
        known_labels = labels[labels >= 0]
        if len(known_labels) == 0 or n_regimes <= 1:
            return 1.0
        
        # Count occurrences
        unique, counts = np.unique(known_labels, return_counts=True)
        probs = counts / counts.sum()
        
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(n_regimes)
        
        return entropy / max_entropy if max_entropy > 0 else 1.0
    
    def _compute_transition_metrics(self, labels: np.ndarray) -> Tuple[float, float]:
        """
        Compute transition-related metrics.
        
        Returns:
            (transition_entropy, self_transition_rate)
        """
        if len(labels) < 2:
            return 0.0, 1.0
        
        # Build transition matrix
        known_labels = labels[labels >= 0]
        if len(known_labels) < 2:
            return 0.0, 1.0
        
        unique_labels = np.unique(known_labels)
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        n = len(unique_labels)
        
        if n <= 1:
            return 0.0, 1.0
        
        trans_matrix = np.zeros((n, n))
        
        prev_label = known_labels[0]
        for label in known_labels[1:]:
            if prev_label in label_to_idx and label in label_to_idx:
                trans_matrix[label_to_idx[prev_label], label_to_idx[label]] += 1
            prev_label = label
        
        # Normalize rows
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_probs = trans_matrix / row_sums
        
        # Transition entropy (average row entropy)
        row_entropies = []
        for row in trans_probs:
            if row.sum() > 0:
                entropy = -np.sum(row * np.log(row + 1e-10))
                row_entropies.append(entropy)
        
        avg_entropy = np.mean(row_entropies) if row_entropies else 0.0
        
        # Self-transition rate (diagonal)
        self_trans = np.trace(trans_matrix)
        total_trans = trans_matrix.sum()
        self_rate = self_trans / total_trans if total_trans > 0 else 0.0
        
        return float(avg_entropy), float(self_rate)
    
    def _compute_separation(self, centroids: np.ndarray) -> float:
        """Compute average inter-centroid distance."""
        n = len(centroids)
        if n < 2:
            return 0.0
        
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                distances.append(dist)
        
        return float(np.mean(distances))
    
    def _compute_silhouette(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute average silhouette score.
        
        Uses simplified computation without sklearn dependency.
        """
        n = len(labels)
        unique_labels = np.unique(labels)
        
        if n < 2 or len(unique_labels) < 2:
            return 0.0
        
        silhouettes = []
        
        for i in range(n):
            label_i = labels[i]
            
            # a(i) = mean distance to same-cluster points
            same_mask = labels == label_i
            same_mask[i] = False
            if same_mask.sum() > 0:
                a_i = np.mean(np.linalg.norm(features[same_mask] - features[i], axis=1))
            else:
                a_i = 0.0
            
            # b(i) = min mean distance to other clusters
            b_i = float('inf')
            for other_label in unique_labels:
                if other_label != label_i:
                    other_mask = labels == other_label
                    if other_mask.sum() > 0:
                        mean_dist = np.mean(np.linalg.norm(features[other_mask] - features[i], axis=1))
                        b_i = min(b_i, mean_dist)
            
            if b_i == float('inf'):
                b_i = 0.0
            
            # Silhouette for point i
            if max(a_i, b_i) > 0:
                s_i = (b_i - a_i) / max(a_i, b_i)
            else:
                s_i = 0.0
            
            silhouettes.append(s_i)
        
        return float(np.mean(silhouettes))
    
    def _empty_metrics(self) -> RegimeMetrics:
        """Return empty metrics for edge cases."""
        return RegimeMetrics(
            stability=0.0,
            novelty_rate=1.0,
            coverage=0.0,
            balance=0.0,
            transition_entropy=0.0,
            self_transition_rate=0.0,
            avg_silhouette=0.0,
            separation=0.0,
            evaluated_at=datetime.now(),
            sample_count=0,
        )


# =============================================================================
# Promotion Criteria
# =============================================================================

@dataclass
class PromotionCriteria:
    """
    Criteria for promoting regime model from LEARNING to CONVERGED.
    """
    min_stability: float = 0.85
    max_novelty_rate: float = 0.10
    min_coverage: float = 0.50
    min_silhouette: float = 0.0
    min_sample_count: int = 1000
    min_days_in_learning: int = 7
    
    def evaluate(self, metrics: RegimeMetrics, 
                 days_in_learning: int = 0) -> Tuple[bool, List[str]]:
        """
        Evaluate if metrics meet promotion criteria.
        
        Returns:
            (can_promote, list_of_failures)
        """
        failures = []
        
        if metrics.stability < self.min_stability:
            failures.append(
                f"Stability {metrics.stability:.2f} < {self.min_stability}"
            )
        
        if metrics.novelty_rate > self.max_novelty_rate:
            failures.append(
                f"Novelty rate {metrics.novelty_rate:.2f} > {self.max_novelty_rate}"
            )
        
        if metrics.coverage < self.min_coverage:
            failures.append(
                f"Coverage {metrics.coverage:.2f} < {self.min_coverage}"
            )
        
        if metrics.avg_silhouette < self.min_silhouette:
            failures.append(
                f"Silhouette {metrics.avg_silhouette:.2f} < {self.min_silhouette}"
            )
        
        if metrics.sample_count < self.min_sample_count:
            failures.append(
                f"Sample count {metrics.sample_count} < {self.min_sample_count}"
            )
        
        if days_in_learning < self.min_days_in_learning:
            failures.append(
                f"Days in learning {days_in_learning} < {self.min_days_in_learning}"
            )
        
        return len(failures) == 0, failures


# =============================================================================
# Helper Functions
# =============================================================================

def evaluate_regime_model(labels: np.ndarray,
                          centroids: Optional[np.ndarray] = None,
                          features: Optional[np.ndarray] = None) -> RegimeMetrics:
    """
    Convenience function to evaluate a regime model.
    
    Args:
        labels: Regime assignment labels
        centroids: Optional centroid matrix
        features: Optional feature matrix
        
    Returns:
        RegimeMetrics
    """
    evaluator = RegimeEvaluator()
    return evaluator.evaluate(labels, centroids=centroids, features=features)
