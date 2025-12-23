#!/usr/bin/env python
"""
Offline Historical Replay for Regime Discovery

Phase 2.8 Implementation

This script runs regime discovery on accumulated historical data
without affecting production ACM runs. Used for:
1. Initial regime discovery for new equipment
2. Re-training regimes with more history
3. Testing new discovery parameters

Usage:
    python scripts/offline_replay.py --equip FD_FAN --days 90
    python scripts/offline_replay.py --equip GAS_TURBINE --start 2024-01-01 --end 2024-06-01
    python scripts/offline_replay.py --equip FD_FAN --days 180 --promote
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.observability import Console
from core.sql_client import SQLClient
from core.regime_manager import (
    MaturityState,
    RegimeManager,
    RegimeInputValidator,
    REGIME_UNKNOWN,
    REGIME_EMERGING,
)
from core.regime_definitions import (
    RegimeDefinition,
    RegimeCentroid,
    create_regime_definition,
)
from core.regime_evaluation import (
    RegimeEvaluator,
    PromotionCriteria,
    RegimeMetrics,
)
from utils.config_dict import ConfigDict


# =============================================================================
# Constants
# =============================================================================

DEFAULT_DAYS = 90
MIN_ROWS_FOR_DISCOVERY = 500
MAX_ROWS_FOR_DISCOVERY = 100000  # Limit for performance


# =============================================================================
# Data Loading
# =============================================================================

def load_historian_data(
    sql: SQLClient,
    equip_id: int,
    equip_name: str,
    start_time: datetime,
    end_time: datetime,
) -> Optional[pd.DataFrame]:
    """
    Load historian data for regime discovery.
    
    Args:
        sql: SQL client
        equip_id: Equipment ID
        equip_name: Equipment name
        start_time: Start of data range
        end_time: End of data range
        
    Returns:
        DataFrame with historian data or None if no data
    """
    Console.info(f"Loading historian data: {start_time} to {end_time}",
                 component="REPLAY", equip_id=equip_id)
    
    cur = sql.cursor()
    try:
        # Try stored procedure first
        try:
            cur.execute(
                "EXEC dbo.usp_ACM_GetHistorianData_TEMP @StartTime=?, @EndTime=?, @EquipmentName=?",
                (start_time, end_time, equip_name)
            )
        except Exception:
            # Fallback to direct table query
            table_name = f"{equip_name}_Data"
            cur.execute(f"""
                SELECT * FROM [{table_name}]
                WHERE EntryDateTime BETWEEN ? AND ?
                ORDER BY EntryDateTime ASC
            """, (start_time, end_time))
        
        rows = cur.fetchall()
        if not rows:
            Console.warn(f"No data found for {equip_name} in range",
                        component="REPLAY")
            return None
        
        # Get column names
        columns = [desc[0] for desc in cur.description]
        df = pd.DataFrame.from_records(rows, columns=columns)
        
        # Standardize timestamp column
        time_cols = [c for c in df.columns if c.lower() in ("entrydatetime", "timestamp")]
        if time_cols:
            df = df.rename(columns={time_cols[0]: "Timestamp"})
        
        Console.ok(f"Loaded {len(df):,} rows for regime discovery",
                  component="REPLAY", equip_id=equip_id)
        
        return df
        
    finally:
        cur.close()


def prepare_features(
    df: pd.DataFrame,
    config: ConfigDict,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for regime discovery.
    
    Args:
        df: Raw historian data
        config: Equipment config
        
    Returns:
        (feature_df, feature_columns)
    """
    # Get input validation
    validator = RegimeInputValidator()
    
    # Get regime feature columns from config
    regime_cols = config.get("regime.feature_columns", None)
    
    if regime_cols is None:
        # Auto-select numeric columns, excluding forbidden patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        clean_df = validator.filter_clean_columns(df[numeric_cols])
        feature_cols = clean_df.columns.tolist()
    else:
        # Use configured columns
        feature_cols = [c for c in regime_cols if c in df.columns]
    
    # Remove any timestamp-like columns
    feature_cols = [c for c in feature_cols if c.lower() not in ("timestamp", "entrydatetime")]
    
    if not feature_cols:
        raise ValueError("No valid feature columns for regime discovery")
    
    Console.info(f"Using {len(feature_cols)} features for discovery",
                component="REPLAY", features=feature_cols[:10])
    
    # Extract features and handle missing values
    feature_df = df[feature_cols].copy()
    feature_df = feature_df.dropna()
    
    return feature_df, feature_cols


# =============================================================================
# Regime Discovery
# =============================================================================

def discover_regimes(
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    config: ConfigDict,
    equip_id: int,
) -> Tuple[RegimeDefinition, np.ndarray, np.ndarray]:
    """
    Run regime discovery algorithm.
    
    Args:
        feature_df: Features for clustering
        feature_cols: Feature column names
        config: Equipment config
        equip_id: Equipment ID
        
    Returns:
        (RegimeDefinition, labels, confidences)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    Console.section("Running Regime Discovery")
    
    # Get parameters from config
    n_regimes = config.get("regime.n_clusters", 3)
    max_regimes = config.get("regime.max_clusters", 8)
    auto_n = config.get("regime.auto_n_clusters", True)
    
    # Prepare feature matrix
    X = feature_df.values
    
    # Subsample if too large
    if len(X) > MAX_ROWS_FOR_DISCOVERY:
        Console.warn(f"Subsampling from {len(X):,} to {MAX_ROWS_FOR_DISCOVERY:,} rows",
                    component="REPLAY")
        indices = np.random.choice(len(X), MAX_ROWS_FOR_DISCOVERY, replace=False)
        X = X[indices]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Auto-detect n_clusters using elbow method
    if auto_n:
        n_regimes = _auto_select_n_clusters(X_scaled, max_regimes)
        Console.info(f"Auto-selected n_clusters = {n_regimes}", component="REPLAY")
    
    # Run K-Means
    Console.info(f"Running KMeans with n_clusters={n_regimes}", component="REPLAY")
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Compute cluster statistics
    centroids_raw = scaler.inverse_transform(kmeans.cluster_centers_)
    centroid_list = []
    
    for i in range(n_regimes):
        mask = labels == i
        n_points = int(mask.sum())
        
        # Compute radius as mean distance to centroid
        if n_points > 0:
            cluster_points = X_scaled[mask]
            distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[i], axis=1)
            radius = float(np.mean(distances))
        else:
            radius = 1.0
        
        centroid_list.append(RegimeCentroid(
            label=i,
            centroid=centroids_raw[i],
            radius=radius,
            n_points=n_points,
        ))
        
        Console.info(f"  Regime {i}: {n_points:,} points, radius={radius:.3f}",
                    component="REPLAY")
    
    # Build transition matrix
    transition_matrix = _compute_transition_matrix(labels, n_regimes)
    
    # Create definition
    definition = RegimeDefinition(
        equip_id=equip_id,
        version=0,  # Will be assigned on save
        num_regimes=n_regimes,
        centroids=centroid_list,
        feature_columns=feature_cols,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        transition_matrix=transition_matrix,
        training_row_count=len(X),
        discovery_params={
            "method": "kmeans",
            "n_clusters": n_regimes,
            "auto_n": auto_n,
        },
    )
    
    # Compute confidences
    confidences = _compute_assignment_confidences(X_scaled, kmeans.cluster_centers_, labels)
    
    return definition, labels, confidences


def _auto_select_n_clusters(X_scaled: np.ndarray, max_k: int) -> int:
    """Auto-select number of clusters using elbow method."""
    from sklearn.cluster import KMeans
    
    inertias = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    
    # Find elbow using second derivative
    if len(inertias) < 3:
        return 3
    
    # Compute rate of change
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    
    # Elbow is where second derivative is maximum
    elbow_idx = np.argmax(diffs2) + 2  # +2 for offset from diff operations
    
    return list(k_range)[min(elbow_idx, len(k_range) - 1)]


def _compute_transition_matrix(labels: np.ndarray, n_regimes: int) -> np.ndarray:
    """Compute normalized transition matrix from labels."""
    trans = np.zeros((n_regimes, n_regimes))
    
    for i in range(1, len(labels)):
        trans[labels[i-1], labels[i]] += 1
    
    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    
    return trans / row_sums


def _compute_assignment_confidences(
    X_scaled: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute confidence scores for assignments."""
    n = len(labels)
    confidences = np.zeros(n)
    
    for i in range(n):
        # Distance to assigned centroid
        dist_assigned = np.linalg.norm(X_scaled[i] - centroids[labels[i]])
        
        # Distance to nearest other centroid
        dist_others = []
        for j, c in enumerate(centroids):
            if j != labels[i]:
                dist_others.append(np.linalg.norm(X_scaled[i] - c))
        
        dist_nearest_other = min(dist_others) if dist_others else dist_assigned + 1
        
        # Confidence = ratio of distances
        if dist_nearest_other > 0:
            # Higher when far from other centroids relative to assigned
            confidences[i] = min(1.0, dist_nearest_other / (dist_assigned + dist_nearest_other))
        else:
            confidences[i] = 0.5
    
    return confidences


# =============================================================================
# Main Entry Point
# =============================================================================

def run_offline_replay(
    equip_name: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    days: int = DEFAULT_DAYS,
    promote: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run offline regime discovery replay.
    
    Args:
        equip_name: Equipment name
        start_time: Optional start time
        end_time: Optional end time  
        days: Days of history if start/end not specified
        promote: Whether to promote to LEARNING state
        dry_run: If True, don't save to SQL
        
    Returns:
        Results dictionary
    """
    Console.header(f"Offline Regime Replay: {equip_name}")
    
    # Calculate time range
    if end_time is None:
        end_time = datetime.now()
    if start_time is None:
        start_time = end_time - timedelta(days=days)
    
    Console.info(f"Time range: {start_time} to {end_time}",
                component="REPLAY", days=days)
    
    # Connect to SQL
    sql = SQLClient()
    
    # Get equipment ID
    cur = sql.cursor()
    cur.execute("SELECT EquipID FROM Equipment WHERE EquipCode = ?", (equip_name,))
    row = cur.fetchone()
    cur.close()
    
    if not row:
        Console.error(f"Equipment not found: {equip_name}", component="REPLAY")
        return {"success": False, "error": "Equipment not found"}
    
    equip_id = row[0]
    Console.info(f"Equipment ID: {equip_id}", component="REPLAY")
    
    # Load config
    config = ConfigDict()
    
    # Load historian data
    df = load_historian_data(sql, equip_id, equip_name, start_time, end_time)
    
    if df is None or len(df) < MIN_ROWS_FOR_DISCOVERY:
        Console.error(f"Insufficient data: need at least {MIN_ROWS_FOR_DISCOVERY} rows",
                     component="REPLAY")
        return {"success": False, "error": "Insufficient data"}
    
    # Prepare features
    feature_df, feature_cols = prepare_features(df, config)
    
    Console.info(f"Prepared {len(feature_df):,} rows with {len(feature_cols)} features",
                component="REPLAY")
    
    # Run discovery
    definition, labels, confidences = discover_regimes(
        feature_df, feature_cols, config, equip_id
    )
    
    # Evaluate the discovered regimes
    Console.section("Evaluating Regime Quality")
    evaluator = RegimeEvaluator()
    metrics = evaluator.evaluate(
        labels,
        confidences=confidences,
        centroids=definition.centroid_array,
    )
    
    Console.info(f"Stability: {metrics.stability:.3f}", component="REPLAY")
    Console.info(f"Coverage: {metrics.coverage:.3f}", component="REPLAY")
    Console.info(f"Balance: {metrics.balance:.3f}", component="REPLAY")
    Console.info(f"Novelty Rate: {metrics.novelty_rate:.3f}", component="REPLAY")
    Console.info(f"Overall Score: {metrics.overall_score:.3f}", component="REPLAY")
    
    # Check promotion criteria
    criteria = PromotionCriteria()
    can_promote, failures = criteria.evaluate(metrics, days_in_learning=0)
    
    if can_promote:
        Console.ok("Regime model meets promotion criteria", component="REPLAY")
    else:
        Console.warn(f"Regime model fails criteria: {failures}", component="REPLAY")
    
    # Save if not dry run
    version = None
    if not dry_run:
        Console.section("Saving Regime Definition")
        
        regime_manager = RegimeManager(sql)
        
        if promote:
            version = regime_manager.save_and_activate(
                definition,
                maturity=MaturityState.LEARNING,
                updated_by="offline_replay",
            )
            Console.ok(f"Saved and activated as version {version}", component="REPLAY")
        else:
            # Just save without activating
            version = regime_manager.definitions.save(definition)
            Console.ok(f"Saved as version {version} (not activated)", component="REPLAY")
    else:
        Console.info("Dry run - not saving to SQL", component="REPLAY")
    
    sql.close()
    
    Console.header("Replay Complete")
    
    return {
        "success": True,
        "equip_id": equip_id,
        "equip_name": equip_name,
        "n_regimes": definition.num_regimes,
        "training_rows": len(feature_df),
        "version": version,
        "metrics": metrics.to_dict(),
        "promoted": promote and not dry_run,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Offline Historical Replay for Regime Discovery"
    )
    
    parser.add_argument(
        "--equip", "-e",
        required=True,
        help="Equipment name (e.g., FD_FAN, GAS_TURBINE)"
    )
    
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Days of history to use (default: {DEFAULT_DAYS})"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote to LEARNING state after discovery"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run discovery without saving to SQL"
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_time = None
    end_time = None
    
    if args.start:
        start_time = datetime.strptime(args.start, "%Y-%m-%d")
    if args.end:
        end_time = datetime.strptime(args.end, "%Y-%m-%d")
    
    # Run replay
    result = run_offline_replay(
        equip_name=args.equip,
        start_time=start_time,
        end_time=end_time,
        days=args.days,
        promote=args.promote,
        dry_run=args.dry_run,
    )
    
    if result["success"]:
        Console.ok("Replay completed successfully")
        return 0
    else:
        Console.error(f"Replay failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
