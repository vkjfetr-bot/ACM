"""Tests for P2.12: Replay Reproducibility.

This module tests that regime discovery produces identical results when
given identical inputs and parameters. This is critical for debugging,
auditing, and ensuring consistent behavior across replays.
"""

import hashlib
import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field


# =============================================================================
# Hash Utilities for Reproducibility
# =============================================================================

def compute_dataframe_hash(df: pd.DataFrame, precision: int = 6) -> str:
    """
    Compute a deterministic hash of a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to hash.
    precision : int
        Decimal precision for floating point values (default 6).
    
    Returns
    -------
    str
        16-character hex hash.
    
    Notes
    -----
    - Rounds floats to specified precision for stability
    - Sorts columns alphabetically for order independence
    - Uses index values in hash computation
    """
    if df.empty:
        return "0" * 16
    
    # Sort columns for consistency
    df_sorted = df[sorted(df.columns)].copy()
    
    # Round numeric columns for float precision stability
    for col in df_sorted.select_dtypes(include=[np.number]).columns:
        df_sorted[col] = df_sorted[col].round(precision)
    
    # Include index in hash
    content = f"{df_sorted.index.tolist()}{df_sorted.values.tobytes()}{list(df_sorted.columns)}"
    
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compute_array_hash(arr: np.ndarray, precision: int = 6) -> str:
    """
    Compute a deterministic hash of a numpy array.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to hash.
    precision : int
        Decimal precision for floating point values.
    
    Returns
    -------
    str
        16-character hex hash.
    """
    if arr.size == 0:
        return "0" * 16
    
    # Round for float precision stability
    arr_rounded = np.round(arr.astype(np.float64), precision)
    buf = np.ascontiguousarray(arr_rounded).tobytes()
    
    return hashlib.sha256(buf).hexdigest()[:16]


def compute_params_hash(params: Dict[str, Any]) -> str:
    """
    Compute a deterministic hash of parameters dictionary.
    
    Parameters
    ----------
    params : dict
        Parameters to hash.
    
    Returns
    -------
    str
        16-character hex hash.
    """
    import json
    # Sort keys for consistency
    sorted_params = dict(sorted(params.items()))
    content = json.dumps(sorted_params, sort_keys=True, default=str)
    
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class ReproducibilityValidator:
    """
    Validates that operations produce identical results across runs.
    
    Usage
    -----
    >>> validator = ReproducibilityValidator()
    >>> validator.record_run("run1", input_hash="abc123", output_hash="def456", params={"k": 3})
    >>> validator.record_run("run2", input_hash="abc123", output_hash="def456", params={"k": 3})
    >>> validator.validate()  # Returns True if all runs with same inputs match
    """
    
    def __init__(self):
        self._runs: Dict[str, Dict[str, Any]] = {}
    
    def record_run(
        self,
        run_id: str,
        input_hash: str,
        output_hash: str,
        params: Dict[str, Any],
        labels: np.ndarray = None
    ):
        """Record a run for later validation."""
        self._runs[run_id] = {
            "input_hash": input_hash,
            "output_hash": output_hash,
            "params_hash": compute_params_hash(params),
            "params": params,
            "labels": labels.copy() if labels is not None else None
        }
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate that runs with identical inputs produce identical outputs.
        
        Returns
        -------
        Tuple[bool, str]
            (is_valid, message)
        """
        # Group runs by (input_hash, params_hash)
        groups: Dict[Tuple[str, str], list] = {}
        for run_id, run_data in self._runs.items():
            key = (run_data["input_hash"], run_data["params_hash"])
            if key not in groups:
                groups[key] = []
            groups[key].append((run_id, run_data))
        
        errors = []
        for (input_h, params_h), runs in groups.items():
            if len(runs) < 2:
                continue  # Need at least 2 runs to compare
            
            output_hashes = set(r[1]["output_hash"] for r in runs)
            if len(output_hashes) > 1:
                run_ids = [r[0] for r in runs]
                errors.append(
                    f"Reproducibility violation: runs {run_ids} have same inputs "
                    f"(input={input_h}, params={params_h}) but different outputs: {output_hashes}"
                )
        
        if errors:
            return False, "\n".join(errors)
        return True, "All runs with identical inputs produced identical outputs"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of recorded runs."""
        return {
            "total_runs": len(self._runs),
            "run_ids": list(self._runs.keys()),
            "unique_inputs": len(set(r["input_hash"] for r in self._runs.values())),
            "unique_params": len(set(r["params_hash"] for r in self._runs.values())),
        }


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sensor_data():
    """Generate reproducible sensor data with known characteristics."""
    np.random.seed(42)
    n_samples = 500
    
    # Create data with clear regime structure
    regime_0_size = n_samples // 2
    regime_1_size = n_samples - regime_0_size
    
    data = pd.DataFrame({
        'temp': np.concatenate([
            np.random.normal(100, 5, regime_0_size),
            np.random.normal(150, 8, regime_1_size)
        ]),
        'pressure': np.concatenate([
            np.random.normal(50, 2, regime_0_size),
            np.random.normal(80, 4, regime_1_size)
        ]),
        'vibration': np.concatenate([
            np.random.normal(10, 1, regime_0_size),
            np.random.normal(20, 2, regime_1_size)
        ]),
    })
    
    return data


@pytest.fixture
def regime_params():
    """Standard regime discovery parameters."""
    return {
        "k_min": 2,
        "k_max": 5,
        "n_init": 10,
        "random_state": 42,
        "max_iter": 300,
    }


# =============================================================================
# Hash Function Tests
# =============================================================================

class TestHashFunctions:
    """Tests for hash utility functions."""
    
    def test_dataframe_hash_deterministic(self, sensor_data):
        """Same DataFrame produces same hash."""
        hash1 = compute_dataframe_hash(sensor_data)
        hash2 = compute_dataframe_hash(sensor_data)
        
        assert hash1 == hash2
    
    def test_dataframe_hash_different_data(self, sensor_data):
        """Different data produces different hash."""
        modified = sensor_data.copy()
        modified.iloc[0, 0] = 999.0
        
        hash1 = compute_dataframe_hash(sensor_data)
        hash2 = compute_dataframe_hash(modified)
        
        assert hash1 != hash2
    
    def test_dataframe_hash_column_order_independent(self):
        """Column order should not affect hash."""
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'b': [3, 4], 'a': [1, 2]})
        
        hash1 = compute_dataframe_hash(df1)
        hash2 = compute_dataframe_hash(df2)
        
        assert hash1 == hash2
    
    def test_dataframe_hash_precision_tolerance(self):
        """Small floating point differences within precision are ignored."""
        df1 = pd.DataFrame({'a': [1.0000001, 2.0]})
        df2 = pd.DataFrame({'a': [1.0000002, 2.0]})
        
        # With precision=6, these should hash the same
        hash1 = compute_dataframe_hash(df1, precision=6)
        hash2 = compute_dataframe_hash(df2, precision=6)
        
        assert hash1 == hash2
    
    def test_dataframe_hash_empty(self):
        """Empty DataFrame produces consistent hash."""
        df = pd.DataFrame()
        
        hash1 = compute_dataframe_hash(df)
        hash2 = compute_dataframe_hash(df)
        
        assert hash1 == hash2
        assert hash1 == "0" * 16
    
    def test_array_hash_deterministic(self):
        """Same array produces same hash."""
        arr = np.array([1.0, 2.0, 3.0])
        
        hash1 = compute_array_hash(arr)
        hash2 = compute_array_hash(arr)
        
        assert hash1 == hash2
    
    def test_array_hash_different_data(self):
        """Different arrays produce different hashes."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 4.0])
        
        hash1 = compute_array_hash(arr1)
        hash2 = compute_array_hash(arr2)
        
        assert hash1 != hash2
    
    def test_params_hash_deterministic(self, regime_params):
        """Same params produce same hash."""
        hash1 = compute_params_hash(regime_params)
        hash2 = compute_params_hash(regime_params)
        
        assert hash1 == hash2
    
    def test_params_hash_key_order_independent(self):
        """Parameter key order should not affect hash."""
        params1 = {"a": 1, "b": 2, "c": 3}
        params2 = {"c": 3, "a": 1, "b": 2}
        
        hash1 = compute_params_hash(params1)
        hash2 = compute_params_hash(params2)
        
        assert hash1 == hash2


# =============================================================================
# ReproducibilityValidator Tests
# =============================================================================

class TestReproducibilityValidator:
    """Tests for ReproducibilityValidator class."""
    
    def test_empty_validator_passes(self):
        """Empty validator should pass validation."""
        validator = ReproducibilityValidator()
        is_valid, msg = validator.validate()
        
        assert is_valid
    
    def test_single_run_passes(self):
        """Single run should pass validation."""
        validator = ReproducibilityValidator()
        validator.record_run("run1", "input123", "output456", {"k": 3})
        
        is_valid, msg = validator.validate()
        
        assert is_valid
    
    def test_identical_runs_pass(self):
        """Identical runs should pass validation."""
        validator = ReproducibilityValidator()
        validator.record_run("run1", "input123", "output456", {"k": 3})
        validator.record_run("run2", "input123", "output456", {"k": 3})
        validator.record_run("run3", "input123", "output456", {"k": 3})
        
        is_valid, msg = validator.validate()
        
        assert is_valid
        assert "identical" in msg.lower()
    
    def test_different_inputs_different_outputs_pass(self):
        """Different inputs with different outputs should pass."""
        validator = ReproducibilityValidator()
        validator.record_run("run1", "input123", "output456", {"k": 3})
        validator.record_run("run2", "input789", "output999", {"k": 3})
        
        is_valid, msg = validator.validate()
        
        assert is_valid
    
    def test_same_inputs_different_outputs_fail(self):
        """Same inputs with different outputs should fail."""
        validator = ReproducibilityValidator()
        validator.record_run("run1", "input123", "output456", {"k": 3})
        validator.record_run("run2", "input123", "output999", {"k": 3})  # Different output!
        
        is_valid, msg = validator.validate()
        
        assert not is_valid
        assert "violation" in msg.lower()
    
    def test_different_params_different_outputs_pass(self):
        """Different params with different outputs should pass."""
        validator = ReproducibilityValidator()
        validator.record_run("run1", "input123", "output456", {"k": 3})
        validator.record_run("run2", "input123", "output999", {"k": 5})  # Different k
        
        is_valid, msg = validator.validate()
        
        assert is_valid
    
    def test_get_summary(self):
        """Test summary generation."""
        validator = ReproducibilityValidator()
        validator.record_run("run1", "input1", "out1", {"k": 3})
        validator.record_run("run2", "input1", "out1", {"k": 3})
        validator.record_run("run3", "input2", "out2", {"k": 5})
        
        summary = validator.get_summary()
        
        assert summary["total_runs"] == 3
        assert summary["unique_inputs"] == 2
        assert len(summary["run_ids"]) == 3


# =============================================================================
# Regime Discovery Reproducibility Tests
# =============================================================================

class TestRegimeDiscoveryReproducibility:
    """Tests that regime discovery is reproducible."""
    
    def test_kmeans_with_fixed_seed_reproducible(self, sensor_data):
        """KMeans with fixed seed produces identical results."""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
        
        np.random.seed(42)
        X = sensor_data.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run 1
        kmeans1 = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=10)
        labels1 = kmeans1.fit_predict(X_scaled)
        
        # Run 2 (fresh model)
        kmeans2 = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=10)
        labels2 = kmeans2.fit_predict(X_scaled)
        
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_label_alignment_reproducible(self, sensor_data):
        """Label alignment should be reproducible."""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
        
        X = sensor_data.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run clustering twice with same seed
        kmeans1 = MiniBatchKMeans(n_clusters=3, random_state=123, n_init=10)
        labels1 = kmeans1.fit_predict(X_scaled)
        
        kmeans2 = MiniBatchKMeans(n_clusters=3, random_state=123, n_init=10)
        labels2 = kmeans2.fit_predict(X_scaled)
        
        # Verify labels are identical
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_input_hash_detects_changes(self, sensor_data):
        """Input hash should detect data changes."""
        original_hash = compute_dataframe_hash(sensor_data)
        
        # Modify single value
        modified = sensor_data.copy()
        modified.iloc[100, 0] = 9999.0
        modified_hash = compute_dataframe_hash(modified)
        
        assert original_hash != modified_hash
    
    def test_full_reproducibility_workflow(self, sensor_data, regime_params):
        """Full workflow should be reproducible."""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
        
        validator = ReproducibilityValidator()
        input_hash = compute_dataframe_hash(sensor_data)
        
        def run_regime_discovery(run_id: str):
            """Run regime discovery and record for validation."""
            X = sensor_data.values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = MiniBatchKMeans(
                n_clusters=regime_params["k_min"],
                random_state=regime_params["random_state"],
                n_init=regime_params["n_init"],
                max_iter=regime_params["max_iter"]
            )
            labels = kmeans.fit_predict(X_scaled)
            
            output_hash = compute_array_hash(labels)
            validator.record_run(run_id, input_hash, output_hash, regime_params, labels)
            
            return labels
        
        # Run 3 times
        labels1 = run_regime_discovery("run1")
        labels2 = run_regime_discovery("run2")
        labels3 = run_regime_discovery("run3")
        
        # All should match
        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_array_equal(labels2, labels3)
        
        # Validator should pass
        is_valid, msg = validator.validate()
        assert is_valid, msg


# =============================================================================
# Integration Tests
# =============================================================================

class TestReproducibilityIntegration:
    """Integration tests for reproducibility across full pipeline."""
    
    def test_multiple_sessions_same_results(self, sensor_data):
        """Simulate multiple independent sessions producing same results."""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
        
        def simulate_session(session_id: int):
            """Simulate an independent session."""
            # Fresh random state
            np.random.seed(42)  # Reset to consistent state
            
            X = sensor_data.values.copy()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            return labels
        
        # Simulate 3 sessions
        session1_labels = simulate_session(1)
        session2_labels = simulate_session(2)
        session3_labels = simulate_session(3)
        
        # All sessions should produce identical results
        np.testing.assert_array_equal(session1_labels, session2_labels)
        np.testing.assert_array_equal(session2_labels, session3_labels)
    
    def test_hash_chain_validation(self, sensor_data, regime_params):
        """Test that we can validate a chain of operations."""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
        
        # Step 1: Hash inputs
        data_hash = compute_dataframe_hash(sensor_data)
        params_hash = compute_params_hash(regime_params)
        
        # Step 2: Run operation
        X = sensor_data.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = MiniBatchKMeans(
            n_clusters=2,
            random_state=regime_params["random_state"],
            n_init=regime_params["n_init"]
        )
        labels = kmeans.fit_predict(X_scaled)
        
        # Step 3: Hash outputs
        labels_hash = compute_array_hash(labels)
        
        # Step 4: Verify chain
        chain = {
            "input_hash": data_hash,
            "params_hash": params_hash,
            "output_hash": labels_hash,
        }
        
        # Re-run with same inputs
        kmeans2 = MiniBatchKMeans(
            n_clusters=2,
            random_state=regime_params["random_state"],
            n_init=regime_params["n_init"]
        )
        labels2 = kmeans2.fit_predict(X_scaled)
        labels2_hash = compute_array_hash(labels2)
        
        # Hashes should match
        assert chain["output_hash"] == labels2_hash
    
    def test_scaler_reproducibility(self, sensor_data):
        """Test that StandardScaler produces identical results."""
        from sklearn.preprocessing import StandardScaler
        
        X = sensor_data.values
        
        # Run 1
        scaler1 = StandardScaler()
        X_scaled1 = scaler1.fit_transform(X)
        
        # Run 2
        scaler2 = StandardScaler()
        X_scaled2 = scaler2.fit_transform(X)
        
        np.testing.assert_array_almost_equal(X_scaled1, X_scaled2)
        np.testing.assert_array_almost_equal(scaler1.mean_, scaler2.mean_)
        np.testing.assert_array_almost_equal(scaler1.scale_, scaler2.scale_)
    
    def test_centroid_stability(self, sensor_data):
        """Test that cluster centroids are stable across runs."""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
        
        X = sensor_data.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run 1
        kmeans1 = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans1.fit(X_scaled)
        centroids1 = np.sort(kmeans1.cluster_centers_, axis=0)
        
        # Run 2
        kmeans2 = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans2.fit(X_scaled)
        centroids2 = np.sort(kmeans2.cluster_centers_, axis=0)
        
        np.testing.assert_array_almost_equal(centroids1, centroids2)
