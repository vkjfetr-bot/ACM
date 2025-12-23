"""
ACM v11.0.0 Calibrated Fusion Module.

Implements calibrated evidence combination for multi-detector fusion with:
- Explicit NaN handling with confidence dampening
- Detector disagreement penalty
- Calibrated weights based on historical performance
- Per-run fusion quality tracking

This module is part of Phase 3 (P3.4) of the v11 refactor.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import numpy as np
import pandas as pd

from core.detector_protocol import DetectorOutput


# =============================================================================
# FUSION DATA STRUCTURES
# =============================================================================

@dataclass
class FusionWeights:
    """
    Calibrated weights for detector fusion.
    
    Weights determine how much each detector contributes to the final
    fused anomaly score. Higher weights mean more influence.
    """
    detector_weights: Dict[str, float]
    calibration_method: str  # "equal", "performance", "inverse_correlation", "episode_separability"
    calibrated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Normalize weights to sum to 1
        if self.detector_weights:
            total = sum(self.detector_weights.values())
            if total > 0:
                self.detector_weights = {
                    k: v / total for k, v in self.detector_weights.items()
                }
    
    def get_weight(self, detector_name: str, default: float = 0.1) -> float:
        """Get weight for a detector, with fallback default."""
        return self.detector_weights.get(detector_name, default)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "detector_weights": self.detector_weights,
            "calibration_method": self.calibration_method,
            "calibrated_at": self.calibrated_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FusionWeights":
        """Deserialize from dictionary."""
        return cls(
            detector_weights=data["detector_weights"],
            calibration_method=data["calibration_method"],
            calibrated_at=datetime.fromisoformat(data["calibrated_at"]),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def equal_weights(cls, detector_names: List[str]) -> "FusionWeights":
        """Create equal weights for all detectors."""
        n = len(detector_names)
        weights = {name: 1.0 / n for name in detector_names} if n > 0 else {}
        return cls(
            detector_weights=weights,
            calibration_method="equal",
            calibrated_at=datetime.now(),
        )


@dataclass
class FusionResult:
    """
    Result of multi-detector fusion.
    
    Contains the fused anomaly score along with quality metrics
    that indicate how reliable the fusion is.
    """
    timestamp: pd.Series
    fused_z: pd.Series              # Combined anomaly score
    confidence: pd.Series           # Fusion confidence (0-1)
    detector_agreement: pd.Series   # How much detectors agree (0-1)
    n_contributing: pd.Series       # Per-row count of non-NaN detectors
    contributing_detectors: List[str]
    missing_detectors: List[str] = field(default_factory=list)
    weights_used: Optional[FusionWeights] = None
    
    def __post_init__(self):
        # Validate lengths match
        n = len(self.timestamp)
        for name, series in [
            ("fused_z", self.fused_z),
            ("confidence", self.confidence),
            ("detector_agreement", self.detector_agreement),
            ("n_contributing", self.n_contributing),
        ]:
            if len(series) != n:
                raise ValueError(
                    f"{name} length {len(series)} != timestamp length {n}"
                )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for output."""
        df = pd.DataFrame({
            "timestamp": self.timestamp,
            "fused_z": self.fused_z,
            "fusion_confidence": self.confidence,
            "detector_agreement": self.detector_agreement,
            "n_contributing": self.n_contributing,
        })
        df["n_missing"] = len(self.missing_detectors)
        return df
    
    @property
    def mean_confidence(self) -> float:
        """Average fusion confidence."""
        return float(self.confidence.mean())
    
    @property
    def mean_agreement(self) -> float:
        """Average detector agreement."""
        return float(self.detector_agreement.mean())
    
    @property
    def disagreement_rate(self) -> float:
        """Fraction of rows with significant disagreement."""
        return float((self.detector_agreement < 0.5).mean())


@dataclass
class FusionQualityMetrics:
    """
    Quality metrics for a fusion run.
    
    Used to track and persist fusion quality to ACM_FusionQuality table.
    """
    run_id: str
    equip_id: int
    detectors_used: List[str]
    missing_detectors: List[str]
    agreement_mean: float
    agreement_std: float
    confidence_mean: float
    disagreement_rate: float
    weights_used: Optional[Dict[str, float]] = None
    calibration_method: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Serialize for SQL persistence."""
        return {
            "RunID": self.run_id,
            "EquipID": self.equip_id,
            "DetectorsUsed": ",".join(self.detectors_used),
            "DetectorCount": len(self.detectors_used),
            "MissingDetectors": ",".join(self.missing_detectors) if self.missing_detectors else None,
            "AgreementMean": self.agreement_mean,
            "AgreementStd": self.agreement_std,
            "ConfidenceMean": self.confidence_mean,
            "DisagreementRate": self.disagreement_rate,
            "WeightsUsed": json.dumps(self.weights_used) if self.weights_used else None,
            "CalibrationMethod": self.calibration_method,
            "CreatedAt": self.created_at,
        }


# =============================================================================
# CALIBRATED FUSION
# =============================================================================

class CalibratedFusion:
    """
    Calibrated evidence combiner for multi-detector fusion.
    
    Improvements over v10:
    1. Explicit NaN handling with confidence dampening
    2. Detector disagreement penalty
    3. Calibrated weights based on historical performance
    4. Per-run fusion quality tracking
    
    Usage:
        fusion = CalibratedFusion()
        
        # Option 1: Use default equal weights
        result = fusion.fuse(detector_outputs)
        
        # Option 2: Provide calibrated weights
        weights = fusion.calibrate_from_performance(historical_data)
        result = fusion.fuse(detector_outputs, weights=weights)
        
        # Get quality metrics
        metrics = fusion.get_quality_metrics(result, run_id, equip_id)
    """
    
    # Default detector weights (from v10 configuration)
    DEFAULT_WEIGHTS = {
        "ar1": 0.20,
        "pca_spe": 0.30,
        "pca_t2": 0.20,
        "iforest": 0.15,
        "gmm": 0.05,
        "omr": 0.10,
    }
    
    def __init__(
        self,
        nan_penalty: float = 0.2,
        disagreement_penalty: float = 0.1,
        disagreement_threshold: float = 2.0,
        min_detectors: int = 2,
    ):
        """
        Initialize fusion combiner.
        
        Args:
            nan_penalty: Confidence reduction per missing detector (0-1)
            disagreement_penalty: Confidence reduction when detectors disagree (0-1)
            disagreement_threshold: Std threshold for "significant disagreement"
            min_detectors: Minimum detectors required for valid fusion
        """
        self.nan_penalty = nan_penalty
        self.disagreement_penalty = disagreement_penalty
        self.disagreement_threshold = disagreement_threshold
        self.min_detectors = min_detectors
        
        # Track expected detectors for missing detection
        self._expected_detectors = set(self.DEFAULT_WEIGHTS.keys())
    
    def set_expected_detectors(self, detector_names: List[str]) -> None:
        """Set which detectors are expected to be present."""
        self._expected_detectors = set(detector_names)
    
    def fuse(
        self,
        detector_outputs: Dict[str, DetectorOutput],
        weights: Optional[FusionWeights] = None,
    ) -> FusionResult:
        """
        Fuse multiple detector outputs into single anomaly score.
        
        Args:
            detector_outputs: Dictionary of detector_name -> DetectorOutput
            weights: Optional calibrated weights (default: equal weights)
            
        Returns:
            FusionResult with fused scores and quality metrics
        """
        if not detector_outputs:
            raise ValueError("No detector outputs to fuse")
        
        if len(detector_outputs) < self.min_detectors:
            raise ValueError(
                f"Need at least {self.min_detectors} detectors, got {len(detector_outputs)}"
            )
        
        # Use provided weights or create equal weights
        if weights is None:
            weights = FusionWeights.equal_weights(list(detector_outputs.keys()))
        
        # Build z-score DataFrame
        z_scores = {}
        timestamps = None
        for name, output in detector_outputs.items():
            z_scores[name] = output.z_score
            if timestamps is None:
                timestamps = output.timestamp
        
        z_df = pd.DataFrame(z_scores)
        
        # Identify missing detectors
        contributing = list(detector_outputs.keys())
        missing = list(self._expected_detectors - set(contributing))
        
        # Compute weighted fusion
        fused_z, agreement = self._compute_weighted_fusion(z_df, weights)
        
        # Compute confidence with penalties
        confidence = self._compute_confidence(z_df, agreement, missing)
        
        # Compute per-row count of non-NaN detectors
        n_contributing = z_df.notna().sum(axis=1)
        
        return FusionResult(
            timestamp=timestamps,
            fused_z=fused_z,
            confidence=confidence,
            detector_agreement=agreement,
            n_contributing=n_contributing,
            contributing_detectors=contributing,
            missing_detectors=missing,
            weights_used=weights,
        )
    
    def fuse_from_dataframe(
        self,
        z_df: pd.DataFrame,
        timestamp_col: str = "Timestamp",
        weights: Optional[FusionWeights] = None,
    ) -> FusionResult:
        """
        Fuse from a DataFrame with z-score columns.
        
        Convenience method for when detector outputs are already in DataFrame form.
        Expects columns ending in '_z' to be z-scores.
        
        Args:
            z_df: DataFrame with timestamp and z-score columns
            timestamp_col: Name of timestamp column
            weights: Optional calibrated weights
            
        Returns:
            FusionResult
        """
        # Extract z-score columns
        z_cols = [c for c in z_df.columns if c.endswith("_z")]
        
        if not z_cols:
            raise ValueError("No z-score columns found (expected *_z pattern)")
        
        # Build detector outputs
        detector_outputs = {}
        for col in z_cols:
            # Extract detector name (remove _z suffix)
            det_name = col[:-2] if col.endswith("_z") else col
            detector_outputs[det_name] = DetectorOutput(
                timestamp=z_df[timestamp_col] if timestamp_col in z_df.columns else pd.Series(range(len(z_df))),
                z_score=z_df[col],
                raw_score=z_df[col],  # Use z as raw for compatibility
                is_anomaly=z_df[col].abs() > 3.0,
                confidence=pd.Series(1.0, index=z_df.index),
                detector_name=det_name,
            )
        
        return self.fuse(detector_outputs, weights=weights)
    
    def _compute_weighted_fusion(
        self,
        z_df: pd.DataFrame,
        weights: FusionWeights,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Compute weighted fusion of z-scores.
        
        Returns:
            Tuple of (fused_z, agreement)
        """
        # Get weights for each column
        weight_values = np.array([
            weights.get_weight(col, 1.0 / len(z_df.columns))
            for col in z_df.columns
        ])
        
        # Weighted mean ignoring NaN
        # For each row: sum(weight * value) / sum(weights where value is not NaN)
        weighted_sum = (z_df * weight_values).sum(axis=1, skipna=True)
        valid_weight_sum = (~z_df.isna() * weight_values).sum(axis=1)
        
        fused_z = weighted_sum / (valid_weight_sum + 1e-10)
        
        # Compute agreement as inverse of normalized std
        z_std = z_df.std(axis=1, skipna=True)
        z_mean = z_df.mean(axis=1, skipna=True).abs()
        
        # Normalize std by mean to get coefficient of variation
        cv = z_std / (z_mean + 1e-10)
        
        # Convert to agreement score (0-1)
        # Low CV = high agreement
        agreement = 1 / (1 + cv)
        agreement = agreement.clip(0, 1)
        
        return fused_z, agreement
    
    def _compute_confidence(
        self,
        z_df: pd.DataFrame,
        agreement: pd.Series,
        missing_detectors: List[str],
    ) -> pd.Series:
        """
        Compute fusion confidence with penalties.
        
        Confidence is reduced by:
        1. Missing detectors (each missing detector reduces confidence)
        2. Disagreement between detectors
        """
        # Start with agreement as base confidence
        confidence = agreement.copy()
        
        # Penalty for missing detectors
        n_missing = len(missing_detectors)
        missing_penalty = n_missing * self.nan_penalty
        confidence -= missing_penalty
        
        # Penalty for per-row NaN values
        row_missing = z_df.isna().sum(axis=1)
        row_penalty = row_missing * (self.nan_penalty / 2)  # Half penalty for row-level
        confidence -= row_penalty
        
        # Penalty for significant disagreement
        z_std = z_df.std(axis=1, skipna=True)
        significant_disagreement = z_std > self.disagreement_threshold
        confidence -= significant_disagreement.astype(float) * self.disagreement_penalty
        
        return confidence.clip(0, 1)
    
    def get_quality_metrics(
        self,
        result: FusionResult,
        run_id: str,
        equip_id: int,
    ) -> FusionQualityMetrics:
        """
        Extract quality metrics from a fusion result.
        
        Args:
            result: FusionResult from fuse()
            run_id: ACM run ID
            equip_id: Equipment ID
            
        Returns:
            FusionQualityMetrics for persistence
        """
        return FusionQualityMetrics(
            run_id=run_id,
            equip_id=equip_id,
            detectors_used=result.contributing_detectors,
            missing_detectors=result.missing_detectors,
            agreement_mean=float(result.detector_agreement.mean()),
            agreement_std=float(result.detector_agreement.std()),
            confidence_mean=float(result.confidence.mean()),
            disagreement_rate=result.disagreement_rate,
            weights_used=result.weights_used.detector_weights if result.weights_used else None,
            calibration_method=result.weights_used.calibration_method if result.weights_used else None,
        )
    
    def calibrate_from_performance(
        self,
        historical_performance: pd.DataFrame,
        method: str = "performance",
    ) -> FusionWeights:
        """
        Calibrate weights based on historical detector performance.
        
        Args:
            historical_performance: DataFrame with columns:
                - detector: Detector name
                - true_positive_rate: TPR for this detector
                - false_positive_rate: FPR for this detector
            method: Calibration method name
            
        Returns:
            Calibrated FusionWeights
        """
        if historical_performance.empty:
            return FusionWeights.equal_weights(list(self.DEFAULT_WEIGHTS.keys()))
        
        required_cols = {"detector", "true_positive_rate", "false_positive_rate"}
        if not required_cols.issubset(historical_performance.columns):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        weights = {}
        for detector in historical_performance["detector"].unique():
            detector_data = historical_performance[
                historical_performance["detector"] == detector
            ]
            
            # Mean performance metrics
            tpr = detector_data["true_positive_rate"].mean()
            fpr = detector_data["false_positive_rate"].mean()
            
            # Weight = TPR * (1 - FPR)
            # High TPR and low FPR = high weight
            weight = tpr * (1 - fpr)
            weights[detector] = max(0.05, weight)  # Minimum weight of 5%
        
        return FusionWeights(
            detector_weights=weights,
            calibration_method=method,
            calibrated_at=datetime.now(),
            metadata={"source": "historical_performance"},
        )
    
    def calibrate_from_correlation(
        self,
        detector_correlations: Dict[str, Dict[str, float]],
    ) -> FusionWeights:
        """
        Calibrate weights to reduce impact of correlated detectors.
        
        Highly correlated detectors get reduced weights to avoid
        double-counting evidence.
        
        Args:
            detector_correlations: Nested dict of pairwise correlations
                {det1: {det2: corr, det3: corr}, ...}
                
        Returns:
            Calibrated FusionWeights
        """
        detectors = list(detector_correlations.keys())
        n = len(detectors)
        
        if n == 0:
            return FusionWeights.equal_weights(list(self.DEFAULT_WEIGHTS.keys()))
        
        # Compute average correlation for each detector
        avg_corr = {}
        for det in detectors:
            correlations = list(detector_correlations.get(det, {}).values())
            avg_corr[det] = np.mean(np.abs(correlations)) if correlations else 0
        
        # Weight is inverse of average correlation
        # High correlation = low weight (to avoid double-counting)
        weights = {}
        for det in detectors:
            # Weight = 1 / (1 + avg_correlation)
            weight = 1.0 / (1.0 + avg_corr[det])
            weights[det] = max(0.05, weight)
        
        return FusionWeights(
            detector_weights=weights,
            calibration_method="inverse_correlation",
            calibrated_at=datetime.now(),
            metadata={"avg_correlations": avg_corr},
        )


# =============================================================================
# DETECTOR CORRELATION TRACKING
# =============================================================================

@dataclass
class CorrelationResult:
    """Pairwise correlation between detectors."""
    detector1: str
    detector2: str
    correlation: float
    is_redundant: bool  # correlation > threshold
    
    def to_dict(self) -> dict:
        """Serialize for SQL persistence."""
        return {
            "Detector1": self.detector1,
            "Detector2": self.detector2,
            "Correlation": self.correlation,
            "IsRedundant": self.is_redundant,
        }


class DetectorCorrelationTracker:
    """
    Track and flag redundant/unstable detectors.
    
    High correlation between detectors suggests redundancy -
    they're detecting the same things. This can lead to:
    1. Double-counting evidence in fusion
    2. Wasted computation
    3. False confidence in results
    """
    
    REDUNDANCY_THRESHOLD = 0.95
    INSTABILITY_THRESHOLD = 0.5  # Correlation variance over time
    
    def __init__(
        self,
        redundancy_threshold: float = 0.95,
    ):
        self.redundancy_threshold = redundancy_threshold
    
    def compute_correlations(
        self,
        detector_outputs: Dict[str, DetectorOutput],
    ) -> List[CorrelationResult]:
        """
        Compute pairwise correlations between all detectors.
        
        Args:
            detector_outputs: Dictionary of detector_name -> DetectorOutput
            
        Returns:
            List of CorrelationResult for each detector pair
        """
        # Build z-score DataFrame
        z_df = pd.DataFrame({
            name: output.z_score
            for name, output in detector_outputs.items()
        })
        
        return self.compute_correlations_from_df(z_df)
    
    def compute_correlations_from_df(
        self,
        z_df: pd.DataFrame,
    ) -> List[CorrelationResult]:
        """
        Compute correlations from DataFrame of z-scores.
        
        Args:
            z_df: DataFrame with detector z-scores as columns
            
        Returns:
            List of CorrelationResult
        """
        results = []
        detectors = list(z_df.columns)
        
        for i, d1 in enumerate(detectors):
            for d2 in detectors[i+1:]:
                # Handle NaN values
                valid_mask = ~(z_df[d1].isna() | z_df[d2].isna())
                if valid_mask.sum() < 10:  # Need minimum samples
                    corr = np.nan
                else:
                    corr = z_df.loc[valid_mask, d1].corr(z_df.loc[valid_mask, d2])
                
                results.append(CorrelationResult(
                    detector1=d1,
                    detector2=d2,
                    correlation=corr if not np.isnan(corr) else 0.0,
                    is_redundant=abs(corr) > self.redundancy_threshold if not np.isnan(corr) else False,
                ))
        
        return results
    
    def identify_redundant_detectors(
        self,
        results: List[CorrelationResult],
    ) -> List[str]:
        """
        Identify detectors that should be considered for removal.
        
        When two detectors are highly correlated, flags the one that
        appears second alphabetically (heuristic).
        
        Args:
            results: List of CorrelationResult from compute_correlations
            
        Returns:
            List of detector names that are redundant
        """
        redundant = set()
        for r in results:
            if r.is_redundant:
                # Flag the "later" detector alphabetically
                redundant.add(max(r.detector1, r.detector2))
        return sorted(list(redundant))
    
    def get_correlation_matrix(
        self,
        results: List[CorrelationResult],
    ) -> pd.DataFrame:
        """
        Convert correlation results to a matrix form.
        
        Args:
            results: List of CorrelationResult
            
        Returns:
            DataFrame correlation matrix
        """
        # Collect all detector names
        detectors = set()
        for r in results:
            detectors.add(r.detector1)
            detectors.add(r.detector2)
        detectors = sorted(list(detectors))
        
        # Build matrix
        matrix = pd.DataFrame(
            np.eye(len(detectors)),
            index=detectors,
            columns=detectors,
        )
        
        for r in results:
            matrix.loc[r.detector1, r.detector2] = r.correlation
            matrix.loc[r.detector2, r.detector1] = r.correlation
        
        return matrix


if __name__ == "__main__":
    # Quick self-test
    import numpy as np
    
    # Create mock detector outputs
    n = 100
    np.random.seed(42)
    
    outputs = {
        "ar1": DetectorOutput(
            timestamp=pd.Series(range(n)),
            z_score=pd.Series(np.random.randn(n)),
            raw_score=pd.Series(np.random.randn(n)),
            is_anomaly=pd.Series([False] * n),
            confidence=pd.Series([1.0] * n),
            detector_name="ar1",
        ),
        "pca_spe": DetectorOutput(
            timestamp=pd.Series(range(n)),
            z_score=pd.Series(np.random.randn(n) * 1.5),
            raw_score=pd.Series(np.random.randn(n)),
            is_anomaly=pd.Series([False] * n),
            confidence=pd.Series([1.0] * n),
            detector_name="pca_spe",
        ),
    }
    
    fusion = CalibratedFusion()
    result = fusion.fuse(outputs)
    
    print(f"Fused z-score mean: {result.fused_z.mean():.3f}")
    print(f"Mean confidence: {result.mean_confidence:.3f}")
    print(f"Mean agreement: {result.mean_agreement:.3f}")
    print("Self-test passed!")
