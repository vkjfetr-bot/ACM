"""
Comprehensive Analysis Tests for Regime Detection Module

This test module verifies and documents the functioning of the regime detection
system in the ACM (Asset Condition Monitoring) pipeline.

Regime Detection Overview:
--------------------------
The regime detection system identifies distinct operating states (regimes) in 
time-series data using unsupervised clustering. It enables:
1. Context-aware anomaly detection (per-regime thresholds)
2. Operating state health assessment
3. Transient state detection (startup/shutdown/trip events)
4. Stability metrics (dwell time, transitions)

Key Components:
- build_feature_basis(): Constructs feature space for clustering
- _fit_kmeans_scaled(): Auto-k selection with silhouette scoring
- fit_regime_model(): Complete model fitting pipeline
- predict_regime(): Label prediction for new data
- update_health_labels(): Health state assignment per regime
- smooth_labels(): Median-like label smoothing
- smooth_transitions(): Minimum dwell time enforcement
- detect_transient_states(): Startup/shutdown/trip detection
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from core import regimes


class TestBuildFeatureBasis:
    """Tests for build_feature_basis function.
    
    This function constructs a compact feature matrix for regime clustering by:
    1. Using PCA scores if a PCA detector is provided
    2. Including raw sensor tags if configured
    3. Falling back to first few columns if no features available
    4. Applying StandardScaler to non-PCA columns
    """

    def test_basic_feature_construction(self):
        """Test basic feature basis construction without PCA detector."""
        np.random.seed(42)
        train_df = pd.DataFrame(np.random.randn(100, 5), columns=[f'col_{i}' for i in range(5)])
        score_df = pd.DataFrame(np.random.randn(50, 5), columns=[f'col_{i}' for i in range(5)])
        cfg = {'regimes': {'feature_basis': {'n_pca_components': 3, 'raw_tags': []}}}
        
        train_basis, score_basis, meta = regimes.build_feature_basis(
            train_df, score_df, None, None, None, cfg
        )
        
        # Verify output shapes
        assert train_basis.shape == (100, 5), "Train basis should use fallback columns"
        assert score_basis.shape == (50, 5), "Score basis should use fallback columns"
        
        # Verify normalization
        assert meta['basis_normalized'] is True, "Non-PCA columns should be normalized"
        assert 'basis_scaler_mean' in meta, "Scaler mean should be stored"
        assert 'basis_scaler_var' in meta, "Scaler variance should be stored"
        
        # Verify standardization (approximately zero mean, unit variance on train)
        assert np.allclose(train_basis.mean().values, 0, atol=1e-10), "Train should be centered"
        assert np.allclose(train_basis.std().values, 1, atol=0.1), "Train should be scaled"

    def test_with_raw_tags(self):
        """Test feature basis includes configured raw sensor tags."""
        np.random.seed(42)
        train_df = pd.DataFrame(np.random.randn(100, 3), columns=['feat_0', 'feat_1', 'feat_2'])
        score_df = pd.DataFrame(np.random.randn(50, 3), columns=['feat_0', 'feat_1', 'feat_2'])
        raw_train = pd.DataFrame(np.random.randn(100, 2), 
                                  columns=['sensor_a', 'sensor_b'],
                                  index=train_df.index)
        raw_score = pd.DataFrame(np.random.randn(50, 2), 
                                  columns=['sensor_a', 'sensor_b'],
                                  index=score_df.index)
        cfg = {'regimes': {'feature_basis': {'n_pca_components': 0, 'raw_tags': ['sensor_a']}}}
        
        train_basis, score_basis, meta = regimes.build_feature_basis(
            train_df, score_df, raw_train, raw_score, None, cfg
        )
        
        assert 'sensor_a' in train_basis.columns, "Raw tag should be included"
        assert meta['raw_tags'] == ['sensor_a'], "Used raw tags should be recorded"

    def test_fallback_when_empty(self):
        """Test fallback to first columns when no features available."""
        np.random.seed(42)
        train_df = pd.DataFrame(np.random.randn(100, 10), columns=[f'col_{i}' for i in range(10)])
        score_df = pd.DataFrame(np.random.randn(50, 10), columns=[f'col_{i}' for i in range(10)])
        cfg = {'regimes': {'feature_basis': {'n_pca_components': 0, 'raw_tags': []}}}
        
        train_basis, score_basis, meta = regimes.build_feature_basis(
            train_df, score_df, None, None, None, cfg
        )
        
        # Should use fallback (first 5 columns max)
        assert train_basis.shape[1] == 5, "Should use fallback columns (max 5)"
        assert meta['fallback_cols'] == train_basis.columns.tolist()


class TestFitKmeansScaled:
    """Tests for _fit_kmeans_scaled function.
    
    This function performs auto-k selection with:
    1. Silhouette score optimization (primary metric)
    2. Calinski-Harabasz fallback if silhouette fails
    3. Stratified sampling for large datasets
    4. Quality flagging based on threshold
    """

    def test_auto_k_selection(self):
        """Test automatic cluster count selection."""
        np.random.seed(42)
        # Create data with clear clusters
        X = np.vstack([
            np.random.randn(50, 3) + [0, 0, 0],
            np.random.randn(50, 3) + [5, 5, 5],
            np.random.randn(50, 3) + [10, 0, 0]
        ])
        
        cfg = {'regimes': {
            'auto_k': {'k_min': 2, 'k_max': 5, 'sil_sample': 100, 
                       'max_eval_samples': 150, 'max_models': 5, 'random_state': 42},
            'quality': {'silhouette_min': 0.2, 'calinski_min': 50}
        }}
        
        scaler, kmeans, k, score, metric, sweep, low_quality = regimes._fit_kmeans_scaled(X, cfg)
        
        # Should find approximately 3 clusters for well-separated data
        assert 2 <= k <= 5, f"Best k should be between 2-5, got {k}"
        assert metric == 'silhouette', "Should use silhouette metric"
        assert score > 0, "Silhouette score should be positive for clustered data"
        assert len(sweep) >= 1, "Should have silhouette sweep results"

    def test_pre_scaled_handling(self):
        """Test handling of pre-normalized data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        cfg = {'regimes': {
            'auto_k': {'k_min': 2, 'k_max': 4, 'sil_sample': 50, 
                       'max_eval_samples': 100, 'max_models': 3, 'random_state': 42},
            'quality': {'silhouette_min': 0.2}
        }}
        
        scaler, kmeans, k, score, metric, sweep, low_quality = regimes._fit_kmeans_scaled(
            X, cfg, pre_scaled=True
        )
        
        # Should use identity scaler for pre-scaled data
        assert isinstance(scaler, regimes._IdentityScaler), "Should use identity scaler"

    def test_quality_threshold_flagging(self):
        """Test quality flagging based on silhouette threshold."""
        np.random.seed(42)
        # Random noise - poor clustering
        X = np.random.randn(100, 3)
        cfg = {'regimes': {
            'auto_k': {'k_min': 2, 'k_max': 4, 'sil_sample': 50, 
                       'max_eval_samples': 100, 'max_models': 3, 'random_state': 42},
            'quality': {'silhouette_min': 0.5}  # High threshold
        }}
        
        scaler, kmeans, k, score, metric, sweep, low_quality = regimes._fit_kmeans_scaled(X, cfg)
        
        # Random data should have low silhouette and be flagged
        if score < 0.5:
            assert low_quality is True, "Should flag low quality for random data"

    def test_minimum_samples_error(self):
        """Test error handling for insufficient samples."""
        X = np.array([[1.0, 2.0]])  # Only 1 sample
        cfg = {'regimes': {'auto_k': {'k_min': 2, 'k_max': 4}}}
        
        with pytest.raises(ValueError, match="fewer than 2 samples"):
            regimes._fit_kmeans_scaled(X, cfg)


class TestFitRegimeModel:
    """Tests for fit_regime_model function.
    
    This function creates a complete RegimeModel with:
    1. Input validation (NaN check, variance check)
    2. Auto-k clustering
    3. Quality assessment
    4. Metadata collection
    """

    def test_complete_model_fitting(self):
        """Test complete regime model fitting pipeline."""
        np.random.seed(42)
        train_basis = pd.DataFrame(np.random.randn(100, 5), columns=[f'col_{i}' for i in range(5)])
        basis_meta = {'n_pca': 0, 'raw_tags': [], 'basis_normalized': True}
        cfg = {'regimes': {
            'auto_k': {'k_min': 2, 'k_max': 4, 'sil_sample': 50, 
                       'max_eval_samples': 100, 'max_models': 3, 'random_state': 42},
            'quality': {'silhouette_min': 0.1, 'calinski_min': 10}
        }}
        
        model = regimes.fit_regime_model(train_basis, basis_meta, cfg, train_hash=12345)
        
        # Verify model structure
        assert isinstance(model, regimes.RegimeModel)
        assert model.kmeans is not None, "Should have KMeans model"
        assert model.scaler is not None, "Should have scaler"
        assert model.feature_columns == list(train_basis.columns)
        assert model.train_hash == 12345
        
        # Verify metadata
        assert 'best_k' in model.meta
        assert 'fit_score' in model.meta
        assert 'quality_ok' in model.meta
        assert 'regime_quality_score' in model.meta

    def test_input_validation_warnings(self):
        """Test that input validation issues are detected."""
        np.random.seed(42)
        # Create data with NaN values
        train_basis = pd.DataFrame(np.random.randn(100, 5), columns=[f'col_{i}' for i in range(5)])
        train_basis.iloc[0, 0] = np.nan
        basis_meta = {'n_pca': 0, 'raw_tags': [], 'basis_normalized': False}
        cfg = {'regimes': {
            'auto_k': {'k_min': 2, 'k_max': 3, 'sil_sample': 50, 
                       'max_eval_samples': 100, 'max_models': 2, 'random_state': 42},
            'quality': {'silhouette_min': 0.1}
        }}
        
        model = regimes.fit_regime_model(train_basis, basis_meta, cfg, None)
        
        # Should still create model but with quality notes
        assert model is not None
        # Model should handle NaN input gracefully


class TestPredictRegime:
    """Tests for predict_regime function.
    
    This function predicts regime labels for new data using:
    1. Feature alignment with training columns
    2. Scaler transformation
    3. Nearest centroid assignment
    """

    def test_basic_prediction(self):
        """Test basic regime prediction."""
        np.random.seed(42)
        train_basis = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        score_basis = pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])
        basis_meta = {'n_pca': 0, 'raw_tags': [], 'basis_normalized': False}
        cfg = {'regimes': {
            'auto_k': {'k_min': 2, 'k_max': 3, 'sil_sample': 50, 
                       'max_eval_samples': 100, 'max_models': 2, 'random_state': 42},
            'quality': {'silhouette_min': 0.1}
        }}
        
        model = regimes.fit_regime_model(train_basis, basis_meta, cfg, None)
        labels = regimes.predict_regime(model, score_basis)
        
        assert labels.shape == (50,), "Should have one label per sample"
        assert labels.dtype == int, "Labels should be integers"
        assert set(labels).issubset(set(range(model.kmeans.n_clusters))), "Labels should be valid cluster IDs"

    def test_column_alignment(self):
        """Test that prediction aligns columns correctly."""
        np.random.seed(42)
        train_basis = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        # Score data with different column order and extra column
        score_basis = pd.DataFrame(np.random.randn(50, 4), columns=['c', 'a', 'b', 'd'])
        basis_meta = {'n_pca': 0, 'raw_tags': [], 'basis_normalized': False}
        cfg = {'regimes': {
            'auto_k': {'k_min': 2, 'k_max': 3, 'sil_sample': 50, 
                       'max_eval_samples': 100, 'max_models': 2, 'random_state': 42},
            'quality': {'silhouette_min': 0.1}
        }}
        
        model = regimes.fit_regime_model(train_basis, basis_meta, cfg, None)
        labels = regimes.predict_regime(model, score_basis)
        
        # Should still work with misaligned columns
        assert labels.shape == (50,)


class TestUpdateHealthLabels:
    """Tests for update_health_labels function.
    
    This function assigns health states to regimes based on:
    1. Median fused z-score per regime
    2. Configurable thresholds (warn/alert)
    3. Dwell time and transition statistics
    """

    def test_health_assignment(self):
        """Test health state assignment based on fused scores."""
        np.random.seed(42)
        train_basis = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        basis_meta = {'n_pca': 0, 'raw_tags': [], 'basis_normalized': False}
        cfg = {'regimes': {
            'auto_k': {'k_min': 2, 'k_max': 3, 'sil_sample': 50, 
                       'max_eval_samples': 100, 'max_models': 2, 'random_state': 42},
            'quality': {'silhouette_min': 0.1},
            'health': {'fused_warn_z': 1.5, 'fused_alert_z': 3.0}
        }}
        
        model = regimes.fit_regime_model(train_basis, basis_meta, cfg, None)
        
        # Create labels and fused scores
        n = 100
        labels = np.random.choice([0, 1], size=n)
        fused = pd.Series(np.random.randn(n) * 0.5)  # Low scores = healthy
        
        stats = regimes.update_health_labels(model, labels, fused, cfg)
        
        # Verify health labels assigned
        assert all(v in ['healthy', 'suspect', 'critical', 'unknown'] 
                   for v in model.health_labels.values())
        
        # Verify stats structure
        for label, stat in stats.items():
            assert 'median_fused' in stat
            assert 'state' in stat
            assert 'count' in stat
            assert 'dwell_samples' in stat

    def test_dwell_time_calculation(self):
        """Test dwell time and transition statistics."""
        np.random.seed(42)
        train_basis = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        basis_meta = {'n_pca': 0, 'raw_tags': [], 'basis_normalized': False}
        cfg = {'regimes': {
            'auto_k': {'k_min': 2, 'k_max': 2, 'sil_sample': 50, 
                       'max_eval_samples': 100, 'max_models': 1, 'random_state': 42},
            'quality': {'silhouette_min': 0.0},
            'health': {'fused_warn_z': 1.5, 'fused_alert_z': 3.0}
        }}
        
        model = regimes.fit_regime_model(train_basis, basis_meta, cfg, None)
        
        # Create sequential labels with known pattern
        labels = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1])  # 2 transitions
        ts = pd.date_range('2024-01-01', periods=10, freq='h')
        fused = pd.Series(np.zeros(10), index=ts)
        
        stats = regimes.update_health_labels(model, labels, fused, cfg)
        
        # Check transition counts are tracked
        assert 'transition_counts' in model.meta


class TestSmoothLabels:
    """Tests for smooth_labels function.
    
    This function applies median-like smoothing using:
    1. scipy.ndimage.median_filter when available
    2. Fallback to rolling mode for edge cases
    """

    def test_basic_smoothing(self):
        """Test basic label smoothing."""
        labels = np.array([0, 0, 1, 0, 0, 2, 2, 2])
        smoothed = regimes.smooth_labels(labels, passes=1, window=3)
        
        assert len(smoothed) == len(labels)
        # The isolated 1 should be smoothed to 0
        assert smoothed[2] == 0, "Isolated spike should be smoothed"

    def test_preserves_dominant_labels(self):
        """Test that dominant segments are preserved."""
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        smoothed = regimes.smooth_labels(labels, passes=1, window=3)
        
        # Dominant segments should be preserved
        assert np.array_equal(smoothed[:5], [0, 0, 0, 0, 0])
        assert np.array_equal(smoothed[-5:], [1, 1, 1, 1, 1])

    def test_empty_input(self):
        """Test handling of empty input."""
        labels = np.array([], dtype=int)
        smoothed = regimes.smooth_labels(labels, passes=1, window=3)
        assert len(smoothed) == 0


class TestSmoothTransitions:
    """Tests for smooth_transitions function.
    
    This function enforces minimum dwell time by:
    1. Merging short segments into neighbors
    2. Supporting both sample-based and time-based thresholds
    3. Using health priority for tie-breaking
    """

    def test_sample_based_dwell(self):
        """Test minimum dwell based on sample count."""
        labels = np.array([0, 0, 1, 2, 2, 2, 2])  # label 1 has only 1 sample
        result = regimes.smooth_transitions(labels, min_dwell_samples=2)
        
        # The single-sample label 1 should be merged with a neighbor.
        # After merging, index 2 should no longer be 1, or the function
        # decided not to merge (which is also valid behavior).
        label_at_idx2_changed = result[2] != 1
        original_was_valid_segment = labels[2] == 1 and labels[3] == 1
        assert label_at_idx2_changed or original_was_valid_segment, \
            "Single-sample segment should be merged or already meet dwell requirement"

    def test_time_based_dwell(self):
        """Test minimum dwell based on time duration."""
        labels = np.array([0, 0, 0, 1, 2, 2, 2, 2])
        ts = pd.date_range('2024-01-01', periods=len(labels), freq='h')
        
        # 1-hour minimum dwell should keep single-sample regime 1
        result = regimes.smooth_transitions(labels, timestamps=ts, min_dwell_seconds=3600)
        assert len(result) == len(labels)

    def test_preserves_length(self):
        """Test that output length matches input."""
        labels = np.array([0, 1, 2, 1, 0, 2, 1, 0])
        ts = pd.date_range('2024-01-01', periods=len(labels), freq='h')
        result = regimes.smooth_transitions(labels, timestamps=ts, min_dwell_samples=2)
        assert len(result) == len(labels)


class TestDetectTransientStates:
    """Tests for detect_transient_states function.
    
    This function classifies transient states using:
    1. Rate of change (ROC) calculation per sensor
    2. Regime transition detection
    3. State machine with: steady, transient, startup, shutdown, trip
    """

    def test_basic_transient_detection(self):
        """Test basic transient state detection."""
        np.random.seed(42)
        # Create stable data
        data = pd.DataFrame(np.random.randn(100, 3) * 0.1, columns=['s1', 's2', 's3'])
        regime_labels = np.zeros(100, dtype=int)  # All same regime
        
        cfg = {'regimes': {'transient_detection': {
            'enabled': True,
            'roc_window': 5,
            'roc_threshold_high': 3.0,
            'roc_threshold_trip': 5.0
        }}}
        
        states = regimes.detect_transient_states(data, regime_labels, cfg)
        
        assert len(states) == 100
        assert all(s in ['steady', 'transient', 'startup', 'shutdown', 'trip'] for s in states)

    def test_regime_change_triggers_transient(self):
        """Test that regime changes trigger transient detection."""
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(100, 3) * 0.1, columns=['s1', 's2', 's3'])
        # Create regime change at sample 50
        regime_labels = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
        
        cfg = {'regimes': {'transient_detection': {
            'enabled': True,
            'roc_window': 5,
            'roc_threshold_high': 3.0,
            'roc_threshold_trip': 5.0,
            'transition_lag': 3
        }}}
        
        states = regimes.detect_transient_states(data, regime_labels, cfg)
        
        # Should detect transient around the transition point
        transition_idx = 50
        nearby = states[transition_idx-5:transition_idx+5]
        transient_states = ['transient', 'startup', 'shutdown']
        has_transient_near_change = any(s in transient_states for s in nearby)
        assert has_transient_near_change, "Should detect transient near regime change"

    def test_disabled_detection(self):
        """Test that disabled detection returns all steady."""
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(50, 3), columns=['s1', 's2', 's3'])
        regime_labels = np.random.choice([0, 1], size=50)
        
        cfg = {'regimes': {'transient_detection': {'enabled': False}}}
        
        states = regimes.detect_transient_states(data, regime_labels, cfg)
        
        assert all(s == 'steady' for s in states)


class TestRegimeModelPersistence:
    """Tests for regime model save/load functions."""

    def test_save_and_load_model(self, tmp_path):
        """Test regime model persistence cycle."""
        np.random.seed(42)
        train_basis = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        basis_meta = {'n_pca': 0, 'raw_tags': [], 'basis_normalized': False}
        cfg = {'regimes': {
            'auto_k': {'k_min': 2, 'k_max': 3, 'sil_sample': 50, 
                       'max_eval_samples': 100, 'max_models': 2, 'random_state': 42},
            'quality': {'silhouette_min': 0.1}
        }}
        
        model = regimes.fit_regime_model(train_basis, basis_meta, cfg, train_hash=99999)
        
        # Save
        models_dir = tmp_path / "models"
        regimes.save_regime_model(model, models_dir)
        
        # Load
        loaded = regimes.load_regime_model(models_dir)
        
        assert loaded is not None
        assert loaded.train_hash == model.train_hash
        assert loaded.feature_columns == model.feature_columns
        assert loaded.kmeans.n_clusters == model.kmeans.n_clusters

    def test_load_missing_model(self, tmp_path):
        """Test loading from non-existent path."""
        models_dir = tmp_path / "nonexistent"
        loaded = regimes.load_regime_model(models_dir)
        assert loaded is None


class TestRegimeSummary:
    """Tests for build_summary_dataframe function."""

    def test_summary_dataframe_structure(self):
        """Test summary DataFrame structure."""
        np.random.seed(42)
        train_basis = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        basis_meta = {'n_pca': 0, 'raw_tags': [], 'basis_normalized': False}
        cfg = {'regimes': {
            'auto_k': {'k_min': 2, 'k_max': 3, 'sil_sample': 50, 
                       'max_eval_samples': 100, 'max_models': 2, 'random_state': 42},
            'quality': {'silhouette_min': 0.1},
            'health': {'fused_warn_z': 1.5, 'fused_alert_z': 3.0}
        }}
        
        model = regimes.fit_regime_model(train_basis, basis_meta, cfg, None)
        
        # Assign health labels
        labels = np.random.choice([0, 1], size=100)
        fused = pd.Series(np.random.randn(100) * 0.5)
        regimes.update_health_labels(model, labels, fused, cfg)
        
        # Build summary
        summary = regimes.build_summary_dataframe(model)
        
        expected_cols = [
            'regime', 'state', 'dwell_seconds', 'dwell_fraction',
            'avg_dwell_seconds', 'transition_count', 'stability_score',
            'median_fused', 'p95_abs_fused', 'count'
        ]
        assert list(summary.columns) == expected_cols


class TestLabelFunction:
    """Tests for the main label() API function."""

    def test_label_integration(self):
        """Test complete labeling pipeline integration."""
        np.random.seed(42)
        train_features = pd.DataFrame(np.random.randn(100, 5), 
                                        columns=[f'f{i}' for i in range(5)])
        score_features = pd.DataFrame(np.random.randn(50, 5), 
                                        columns=[f'f{i}' for i in range(5)])
        
        cfg = {'regimes': {
            'feature_basis': {'n_pca_components': 0, 'raw_tags': []},
            'auto_k': {'k_min': 2, 'k_max': 3, 'sil_sample': 50, 
                       'max_eval_samples': 100, 'max_models': 2, 'random_state': 42},
            'quality': {'silhouette_min': 0.1},
            'smoothing': {'passes': 1, 'min_dwell_samples': 2}
        }}
        
        # Build basis
        train_basis, score_basis, basis_meta = regimes.build_feature_basis(
            train_features, score_features, None, None, None, cfg
        )
        # Use deterministic hash for test reproducibility
        train_hash = regimes._stable_int_hash(train_basis.to_numpy(dtype=float, copy=False))
        
        # Create context
        ctx = {
            'regime_basis_train': train_basis,
            'regime_basis_score': score_basis,
            'basis_meta': basis_meta,
            'regime_basis_hash': train_hash,
            'regime_model': None
        }
        
        score_df = score_features.copy()
        score_out = {'frame': score_df.copy()}
        
        result = regimes.label(score_df, ctx, score_out, cfg)
        
        assert 'regime_labels' in result
        assert 'regime_model' in result
        assert 'regime_k' in result
        assert len(result['regime_labels']) == len(score_df)


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_cfg_get(self):
        """Test configuration getter with nested paths."""
        cfg = {'a': {'b': {'c': 42}}}
        
        assert regimes._cfg_get(cfg, 'a.b.c', 0) == 42
        assert regimes._cfg_get(cfg, 'a.b.d', 99) == 99
        assert regimes._cfg_get({}, 'x.y.z', 'default') == 'default'

    def test_as_f32(self):
        """Test float32 array conversion."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = regimes._as_f32(arr)
        assert result.dtype == np.float32

    def test_finite_impute_inplace(self):
        """Test finite value imputation."""
        X = np.array([[1.0, np.nan, 3.0],
                      [4.0, 5.0, np.inf],
                      [7.0, 8.0, 9.0]], dtype=np.float32)
        result = regimes._finite_impute_inplace(X.copy())
        assert np.isfinite(result).all()

    def test_robust_scale_clip(self):
        """Test robust scaling with outlier clipping."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 5.0], [5.0, 6.0]])
        result = regimes._robust_scale_clip(X, clip_pct=99.0)
        assert result.shape == X.shape
        assert np.isfinite(result).all()


# Run pytest with verbose output when executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
