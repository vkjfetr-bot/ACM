"""
Sensor Attribution via Counterfactual Analysis (v10.0.0)

Identifies which sensors contribute most to failure risk via counterfactual analysis.
Replaces logic from rul_engine.py (lines 489-611).

Key Features:
- Counterfactual analysis: RUL impact when zeroing each sensor
- Rank sensors by failure contribution
- Top-N sensor identification for maintenance prioritization
- Integration with ACM_SensorHotspots table

References:
- Pearl (2009): "Causality" - Counterfactual analysis framework
- Molnar (2020): "Interpretable Machine Learning" - Feature importance via perturbation

Future R&D Entrypoints (M15):
---------------------------------
TODO: CAUSAL_COUNTERFACTUAL - Full Pearl-style counterfactual attribution
      - Method: compute_counterfactual(sensor_data, rul_estimator, baseline_rul)
      - For each sensor: zero out contribution, recompute RUL, measure delta
      - Requires RULEstimator integration for each perturbation
      - Output: Causal importance scores (larger delta = higher contribution)
      
TODO: FAULT_SIGNATURE_DETECTION - Identify recurring sensor patterns
      - Method: detect_fault_signatures(sensor_history, known_faults)
      - Match current sensor profile against historical fault patterns
      - Output: Fault signature match scores for ACM_FaultFamilies
      
TODO: SENSOR_INTERACTION_GRAPH - Build sensor dependency network
      - Method: build_interaction_graph(correlation_matrix)
      - Identify sensor clusters and propagation paths
      - Output: NetworkX graph for root cause analysis
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from core.observability import Console, Heartbeat


@dataclass
class SensorAttribution:
    """Sensor attribution result with failure contribution metrics"""
    sensor_name: str
    failure_contribution: float  # 0-1 scale (proportion of total risk)
    z_score_at_failure: float    # Z-score when health crosses threshold
    alert_count: int              # Number of threshold crossings
    rank: int                     # Rank by failure contribution (1=highest)


class SensorAttributor:
    """
    Sensor attribution via counterfactual analysis and hotspot ranking.
    
    Attribution Methods:
    1. Counterfactual Analysis: Compute RUL with and without each sensor
       - Larger RUL delta = higher sensor contribution to failure
    2. SQL Hotspot Loading: Use pre-computed ACM_SensorHotspots contributions
    
    Usage:
        attributor = SensorAttributor(sql_client=sql_client)
        
        # Method 1: Load from ACM_SensorHotspots
        attributions = attributor.load_from_sql(equip_id=1, run_id="run_123")
        top3 = attributor.get_top_n(attributions, n=3)
        
        # Method 2: Counterfactual analysis (future implementation)
        # attributions = attributor.compute_counterfactual(
        #     sensor_data=df,
        #     rul_estimator=rul_est,
        #     baseline_rul=100.0
        # )
    """
    
    def __init__(self, sql_client: Optional[Any] = None):
        """
        Initialize sensor attributor.
        
        Args:
            sql_client: Database connection (pyodbc) for loading ACM_SensorHotspots
        """
        self.sql_client = sql_client
    
    def load_from_sql(self, equip_id: int, run_id: str) -> List[SensorAttribution]:
        """
        Load sensor attributions from ACM_SensorHotspots table.
        
        Query Strategy:
        1. Try full query with FailureContribution, ZScoreAtFailure, AlertCount
        2. If schema missing columns, fallback to MaxAbsZ-based derivation
        3. Derive missing columns via normalization
        
        Args:
            equip_id: Equipment ID
            run_id: ACM run identifier
        
        Returns:
            List of SensorAttribution objects, sorted by failure contribution (descending)
        """
        if self.sql_client is None:
            Console.warn("No SQL client provided; cannot load attributions", component="SENSOR_ATTR", equip_id=equip_id, run_id=run_id)
            return []
        
        try:
            df = self._load_hotspots_dataframe(equip_id, run_id)
            
            if df.empty:
                Console.warn("No sensor hotspots found", component="SENSOR_ATTR", equip_id=equip_id, run_id=run_id)
                return []
            
            # Convert DataFrame to SensorAttribution objects
            attributions = []
            for idx, row in df.iterrows():
                attributions.append(SensorAttribution(
                    sensor_name=str(row["SensorName"]),
                    failure_contribution=float(row["FailureContribution"]),
                    z_score_at_failure=float(row["ZScoreAtFailure"]),
                    alert_count=int(row["AlertCount"]),
                    rank=idx + 1  # 1-indexed rank
                ))
            
            Console.info(f"Loaded {len(attributions)} sensor attributions from SQL", component="SENSOR_ATTR")
            return attributions
            
        except Exception as e:
            Console.warn(f"Failed to load sensor attributions: {e}", component="SENSOR_ATTR", equip_id=equip_id, run_id=run_id, error_type=type(e).__name__, error=str(e)[:200])
            return []
    
    def _load_hotspots_dataframe(self, equip_id: int, run_id: str) -> pd.DataFrame:
        """
        Load ACM_SensorHotspots with fallback for legacy schemas.
        
        Handles two schema versions:
        - v10: FailureContribution, ZScoreAtFailure, AlertCount columns
        - v9: MaxAbsZ, MaxSignedZ, LatestSignedZ, AboveAlertCount (derive contributions)
        
        Args:
            equip_id: Equipment ID
            run_id: ACM run identifier
        
        Returns:
            DataFrame with columns: SensorName, FailureContribution, ZScoreAtFailure, AlertCount
        """
        def _run_query(sql: str) -> pd.DataFrame:
            """Execute query and return DataFrame"""
            cur = self.sql_client.cursor()
            try:
                cur.execute(sql, (equip_id, run_id))
                rows = cur.fetchall()
                columns = [col[0] for col in (cur.description or [])]
                return pd.DataFrame.from_records(rows, columns=columns)
            finally:
                cur.close()
        
        # Try full schema query first
        base_query = """
            SELECT SensorName,
                   FailureContribution,
                   ZScoreAtFailure,
                   AlertCount,
                   MaxAbsZ,
                   MaxSignedZ,
                   LatestSignedZ,
                   AboveAlertCount
            FROM dbo.ACM_SensorHotspots
            WHERE EquipID = ? AND RunID = ?
            ORDER BY FailureContribution DESC
        """
        
        # Fallback query for legacy schema
        fallback_query = """
            SELECT SensorName,
                   MaxAbsZ,
                   MaxSignedZ,
                   LatestSignedZ,
                   AboveAlertCount
            FROM dbo.ACM_SensorHotspots
            WHERE EquipID = ? AND RunID = ?
            ORDER BY MaxAbsZ DESC
        """
        
        try:
            df = _run_query(base_query)
        except Exception as primary_err:
            Console.warn(
                "SensorHotspots schema missing attribution columns; deriving contributions instead",
                component="SENSOR_ATTR", equip_id=equip_id, run_id=run_id, error_type=type(primary_err).__name__, error=str(primary_err)[:200]
            )
            try:
                df = _run_query(fallback_query)
            except Exception as fallback_err:
                Console.warn(f"Failed to load sensor hotspots: {fallback_err}", component="SENSOR_ATTR", equip_id=equip_id, run_id=run_id, error_type=type(fallback_err).__name__, error=str(fallback_err)[:200])
                return pd.DataFrame()
        
        if df.empty:
            return df
        
        # Derive missing columns (handle legacy schema OR NULL values)
        df = self._derive_missing_columns(df)
        
        # Return sorted by FailureContribution
        result = df[["SensorName", "FailureContribution", "ZScoreAtFailure", "AlertCount"]].copy()
        result = result.sort_values("FailureContribution", ascending=False).reset_index(drop=True)
        
        return result
    
    def _derive_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive missing attribution columns from MaxAbsZ-based data.
        
        Derivation Rules:
        - FailureContribution: Normalize MaxAbsZ to sum to 1.0
        - ZScoreAtFailure: Use MaxSignedZ or LatestSignedZ
        - AlertCount: Use AboveAlertCount
        
        Args:
            df: DataFrame with at least MaxAbsZ column
        
        Returns:
            DataFrame with all attribution columns
        """
        # Derive FailureContribution
        if "FailureContribution" not in df.columns or df["FailureContribution"].isna().any():
            abs_vals = pd.to_numeric(df.get("MaxAbsZ"), errors="coerce").abs().fillna(0.0)
            total = abs_vals.sum()
            
            if total > 0:
                df["FailureContribution"] = (abs_vals / total).clip(lower=0.0)
            elif abs_vals.max() > 0:
                # Normalize by max if sum is zero
                df["FailureContribution"] = (abs_vals / abs_vals.max()).clip(lower=0.0)
            else:
                df["FailureContribution"] = 0.0
        
        # Derive ZScoreAtFailure
        if "ZScoreAtFailure" not in df.columns or df["ZScoreAtFailure"].isna().any():
            z_source = df.get("MaxSignedZ")
            if z_source is None or (hasattr(z_source, "isna") and z_source.isna().all()):
                z_source = df.get("LatestSignedZ")
            if z_source is None:
                z_source = pd.Series([0.0] * len(df))
            df["ZScoreAtFailure"] = pd.to_numeric(z_source, errors="coerce").fillna(0.0)
        
        # Derive AlertCount
        if "AlertCount" not in df.columns or df["AlertCount"].isna().any():
            alerts = df.get("AboveAlertCount")
            if alerts is None:
                alerts = pd.Series([0] * len(df))
            df["AlertCount"] = pd.to_numeric(alerts, errors="coerce").fillna(0).astype(int)
        
        return df
    
    def get_top_n(self, attributions: List[SensorAttribution], n: int = 3) -> List[SensorAttribution]:
        """
        Get top N sensors by failure contribution.
        
        Args:
            attributions: List of SensorAttribution objects
            n: Number of top sensors to return (default 3)
        
        Returns:
            List of top N SensorAttribution objects
        """
        # Sort by failure contribution (descending)
        sorted_attrs = sorted(attributions, key=lambda x: x.failure_contribution, reverse=True)
        
        # Return top N
        return sorted_attrs[:n]
    
    def format_top_n(self, attributions: List[SensorAttribution], n: int = 3) -> str:
        """
        Format top N sensors as human-readable string.
        
        Args:
            attributions: List of SensorAttribution objects
            n: Number of top sensors to format (default 3)
        
        Returns:
            Formatted string like "Sensor1 (42.5%), Sensor2 (28.3%), Sensor3 (15.2%)"
        """
        top_n = self.get_top_n(attributions, n)
        
        if not top_n:
            return "No sensor attributions available"
        
        parts = []
        for attr in top_n:
            parts.append(f"{attr.sensor_name} ({attr.failure_contribution * 100:.1f}%)")
        
        return ", ".join(parts)
    
    def compute_attribution_scores(
        self,
        sensor_data: pd.DataFrame,
        contribution_column: str = "MaxAbsZ"
    ) -> List[SensorAttribution]:
        """
        Compute attribution scores directly from sensor data DataFrame.
        
        Useful when ACM_SensorHotspots is not available or for ad-hoc analysis.
        
        Args:
            sensor_data: DataFrame with sensor names and contribution scores
            contribution_column: Column name to use for scoring (default "MaxAbsZ")
        
        Returns:
            List of SensorAttribution objects
        """
        if sensor_data.empty:
            return []
        
        if contribution_column not in sensor_data.columns:
            Console.warn("Contribution column not found in sensor data", component="SENSOR_ATTR", column=contribution_column, available_columns=list(sensor_data.columns)[:10])
            return []
        
        # Extract scores
        scores = pd.to_numeric(sensor_data[contribution_column], errors="coerce").abs().fillna(0.0)
        total = scores.sum()
        
        if total == 0:
            Console.warn("All contribution scores are zero", component="SENSOR_ATTR", column=contribution_column, n_sensors=len(sensor_data))
            return []
        
        # Normalize to proportions
        proportions = scores / total
        
        # Create SensorAttribution objects
        attributions = []
        for idx, (sensor_name, proportion) in enumerate(zip(sensor_data.index, proportions)):
            attributions.append(SensorAttribution(
                sensor_name=str(sensor_name),
                failure_contribution=float(proportion),
                z_score_at_failure=0.0,  # Not available in ad-hoc mode
                alert_count=0,           # Not available in ad-hoc mode
                rank=idx + 1
            ))
        
        # Sort by contribution
        attributions.sort(key=lambda x: x.failure_contribution, reverse=True)
        
        # Update ranks
        for idx, attr in enumerate(attributions):
            attr.rank = idx + 1
        
        return attributions


def rank_sensors_by_contribution(
    attributions: List[SensorAttribution]
) -> pd.DataFrame:
    """
    Convert sensor attributions to ranked DataFrame.
    
    Useful for saving to ACM_RUL_Attribution table or generating reports.
    
    Args:
        attributions: List of SensorAttribution objects
    
    Returns:
        DataFrame with columns: Rank, SensorName, FailureContribution, ZScoreAtFailure, AlertCount
    """
    if not attributions:
        return pd.DataFrame(columns=["Rank", "SensorName", "FailureContribution", "ZScoreAtFailure", "AlertCount"])
    
    records = []
    for attr in attributions:
        records.append({
            "Rank": attr.rank,
            "SensorName": attr.sensor_name,
            "FailureContribution": attr.failure_contribution,
            "ZScoreAtFailure": attr.z_score_at_failure,
            "AlertCount": attr.alert_count
        })
    
    df = pd.DataFrame(records)
    return df
