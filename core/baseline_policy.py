"""
Baseline Policy Module for ACM v11.

P5.5 - Define and enforce per-equipment baseline window requirements.

This module provides:
- BaselineRequirements: Per-equipment baseline data requirements
- BaselineAssessment: Assessment result of baseline data quality
- BaselinePolicy: Policy enforcement for baseline window validation

Usage:
    from core.baseline_policy import BaselinePolicy, BaselineQuality

    policy = BaselinePolicy()
    assessment = policy.assess(data_df, "GAS_TURBINE", regime_labels)

    if assessment.can_proceed:
        print(f"Quality: {assessment.quality.value}")
    else:
        print(f"Violations: {assessment.violations}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import pandas as pd
import numpy as np

__all__ = [
    "BaselineQuality",
    "BaselineRequirements",
    "BaselineAssessment",
    "BaselinePolicy",
]


class BaselineQuality(Enum):
    """Quality assessment of baseline data."""
    EXCELLENT = "EXCELLENT"      # Meets all requirements
    ADEQUATE = "ADEQUATE"        # Meets minimum requirements
    MARGINAL = "MARGINAL"        # Below minimum, proceed with caution
    INSUFFICIENT = "INSUFFICIENT"  # Cannot proceed


@dataclass
class BaselineRequirements:
    """
    Per-equipment baseline requirements.

    Attributes:
        min_rows: Minimum number of rows required
        min_hours: Minimum time span in hours
        max_gap_hours: Maximum allowed gap between consecutive rows
        min_sensor_coverage: Minimum fraction of non-null sensor values
        require_regime_diversity: Whether multiple regimes are required
        min_regimes: Minimum number of distinct regimes
    """
    min_rows: int = 500
    min_hours: float = 168.0        # 7 days
    max_gap_hours: float = 24.0
    min_sensor_coverage: float = 0.9  # 90% non-null
    require_regime_diversity: bool = True
    min_regimes: int = 1


@dataclass
class BaselineAssessment:
    """
    Assessment of baseline data quality.

    Attributes:
        quality: Overall quality rating
        actual_rows: Number of rows in baseline data
        actual_hours: Time span in hours
        max_gap_hours: Largest gap found in data
        sensor_coverage: Fraction of non-null sensor values
        regime_count: Number of distinct regimes found
        violations: List of specific requirement violations
        can_proceed: Whether baseline is sufficient to proceed
    """
    quality: BaselineQuality
    actual_rows: int
    actual_hours: float
    max_gap_hours: float
    sensor_coverage: float
    regime_count: int
    violations: List[str]
    can_proceed: bool

    def to_sql_row(self, run_id: int, equip_id: int, equipment_type: str) -> Dict[str, Any]:
        """Convert to dict for SQL persistence."""
        return {
            "RunID": run_id,
            "EquipID": equip_id,
            "EquipmentType": equipment_type,
            "Quality": self.quality.value,
            "ActualRows": self.actual_rows,
            "ActualHours": self.actual_hours,
            "MaxGapHours": self.max_gap_hours,
            "SensorCoverage": self.sensor_coverage,
            "RegimeCount": self.regime_count,
            "Violations": "; ".join(self.violations) if self.violations else None,
            "CanProceed": self.can_proceed
        }


class BaselinePolicy:
    """
    Define and enforce per-equipment baseline window requirements.

    Validates baseline data quality and determines if analysis can proceed.

    Parameters:
        equipment_requirements: Optional dict mapping equipment types to requirements

    Example:
        >>> policy = BaselinePolicy()
        >>> assessment = policy.assess(data_df, "GAS_TURBINE", regime_labels)
        >>> print(f"Quality: {assessment.quality.value}, Can proceed: {assessment.can_proceed}")
    """

    # Default requirements by equipment type
    DEFAULT_REQUIREMENTS = {
        "*": BaselineRequirements(),  # Global default (7 days, 500 rows)
        "GAS_TURBINE": BaselineRequirements(min_rows=1000, min_hours=336),  # 14 days
        "FD_FAN": BaselineRequirements(min_rows=500, min_hours=168),  # 7 days
        "COMPRESSOR": BaselineRequirements(min_rows=800, min_hours=240),  # 10 days
        "PUMP": BaselineRequirements(min_rows=400, min_hours=120),  # 5 days
    }

    def __init__(
        self,
        equipment_requirements: Optional[Dict[str, BaselineRequirements]] = None
    ):
        """
        Initialize baseline policy.

        Parameters:
            equipment_requirements: Optional dict overriding default requirements
        """
        self.requirements = equipment_requirements or self.DEFAULT_REQUIREMENTS.copy()

    def get_requirements(self, equipment_type: str) -> BaselineRequirements:
        """
        Get requirements for equipment type.

        Parameters:
            equipment_type: Equipment type name

        Returns:
            BaselineRequirements for the equipment type, or global default
        """
        return self.requirements.get(
            equipment_type,
            self.requirements.get("*", BaselineRequirements())
        )

    def set_requirements(
        self,
        equipment_type: str,
        requirements: BaselineRequirements
    ) -> None:
        """
        Set requirements for equipment type.

        Parameters:
            equipment_type: Equipment type name
            requirements: Requirements to set
        """
        self.requirements[equipment_type] = requirements

    def assess(
        self,
        data: pd.DataFrame,
        equipment_type: str,
        regime_labels: Optional[pd.Series] = None,
        timestamp_col: str = "Timestamp"
    ) -> BaselineAssessment:
        """
        Assess baseline data against requirements.

        Parameters:
            data: Baseline data DataFrame
            equipment_type: Equipment type for requirement lookup
            regime_labels: Optional series of regime labels
            timestamp_col: Name of timestamp column

        Returns:
            BaselineAssessment with quality rating and violations
        """
        reqs = self.get_requirements(equipment_type)
        violations = []

        # Check row count
        actual_rows = len(data)
        if actual_rows < reqs.min_rows:
            violations.append(f"Only {actual_rows} rows (need {reqs.min_rows})")

        # Check time span
        if timestamp_col in data.columns and len(data) > 0:
            timestamps = pd.to_datetime(data[timestamp_col])
            time_span = timestamps.max() - timestamps.min()
            actual_hours = time_span.total_seconds() / 3600
        else:
            actual_hours = 0.0

        if actual_hours < reqs.min_hours:
            violations.append(f"Only {actual_hours:.1f}h of data (need {reqs.min_hours}h)")

        # Check gaps
        if timestamp_col in data.columns and len(data) > 1:
            timestamps = pd.to_datetime(data[timestamp_col]).sort_values()
            gaps = timestamps.diff().dt.total_seconds() / 3600
            max_gap = float(gaps.max()) if not gaps.empty and not gaps.isna().all() else 0.0
        else:
            max_gap = 0.0

        if max_gap > reqs.max_gap_hours:
            violations.append(f"Gap of {max_gap:.1f}h exceeds {reqs.max_gap_hours}h")

        # Check sensor coverage
        exclude_cols = {timestamp_col, "EquipID", "EntryDateTime", "Timestamp"}
        sensor_cols = [c for c in data.columns if c not in exclude_cols]

        if sensor_cols and len(data) > 0:
            coverage = float(data[sensor_cols].notna().mean().mean())
        else:
            coverage = 0.0

        if coverage < reqs.min_sensor_coverage:
            violations.append(f"Sensor coverage {coverage:.1%} below {reqs.min_sensor_coverage:.1%}")

        # Check regime diversity
        if regime_labels is not None:
            # Exclude unknown (-1) and emerging (-2) regimes
            valid_regimes = regime_labels[regime_labels >= 0]
            regime_count = len(valid_regimes.unique())
        else:
            regime_count = 0

        if reqs.require_regime_diversity and regime_count < reqs.min_regimes:
            violations.append(f"Only {regime_count} regimes (need {reqs.min_regimes})")

        # Determine quality
        quality = self._compute_quality(actual_rows, reqs.min_rows, violations)

        return BaselineAssessment(
            quality=quality,
            actual_rows=actual_rows,
            actual_hours=actual_hours,
            max_gap_hours=max_gap,
            sensor_coverage=coverage,
            regime_count=regime_count,
            violations=violations,
            can_proceed=quality != BaselineQuality.INSUFFICIENT
        )

    def _compute_quality(
        self,
        actual_rows: int,
        min_rows: int,
        violations: List[str]
    ) -> BaselineQuality:
        """
        Compute quality rating based on violations and row count.

        Parameters:
            actual_rows: Actual row count
            min_rows: Required minimum rows
            violations: List of violations found

        Returns:
            BaselineQuality rating
        """
        if not violations:
            return BaselineQuality.EXCELLENT

        # If only one violation and we have at least 70% of required rows
        if len(violations) == 1 and actual_rows >= min_rows * 0.7:
            return BaselineQuality.ADEQUATE

        # If we have at least 50% of required rows
        if actual_rows >= min_rows * 0.5:
            return BaselineQuality.MARGINAL

        return BaselineQuality.INSUFFICIENT

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all configured requirements."""
        summary = {}
        for equip_type, reqs in self.requirements.items():
            summary[equip_type] = {
                "min_rows": reqs.min_rows,
                "min_hours": reqs.min_hours,
                "max_gap_hours": reqs.max_gap_hours,
                "min_sensor_coverage": reqs.min_sensor_coverage,
                "require_regime_diversity": reqs.require_regime_diversity,
                "min_regimes": reqs.min_regimes,
            }
        return summary
