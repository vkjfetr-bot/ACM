# Ideal Unsupervised Equipment Condition Monitoring System Design

**Purpose**: Design a completely functional system that achieves true unsupervised fault diagnosis  
**Scope**: From cold-start through continuous operation, identifying operating conditions, detecting faults, classifying fault types, and predicting failures—all without human labeling  
**Philosophy**: Physics-informed, semantically meaningful, production-ready

---

## Executive Summary

This document outlines the design of a **next-generation unsupervised equipment condition monitoring system** that addresses the analytical flaws identified in V11 while maintaining its strengths.

### Core Capabilities

1. **Semantic Operating Mode Discovery**: Discover and identify operational states (idle, startup, full-load) using physics-informed clustering
2. **Multi-Level Fault Detection**: Detect anomalies at sensor, subsystem, and equipment levels with adaptive thresholds
3. **Unsupervised Fault Classification**: Build a learned taxonomy of fault signatures without human labels
4. **Failure Mode Prediction**: Predict not just when equipment will fail, but what will fail (bearing, motor, sensor, control)
5. **Fleet-Wide Learning**: Transfer knowledge across similar equipment for fast cold-start and consistent regime identification

### Key Design Principles

1. **Physics-Informed**: Use domain knowledge to guide learning (causality, temporal dynamics, physical constraints)
2. **Semantic Correctness**: Ensure discovered patterns have operational meaning, not just statistical validity
3. **Transparency**: Every prediction must include confidence, contributing factors, and uncertainty quantification
4. **Adaptability**: System must handle maintenance events, seasonal patterns, and equipment aging
5. **Scalability**: Design for 100s of equipment with 1000s of sensors

---

## System Architecture: 7-Phase Pipeline

### Phase 1: Data Ingestion & Validation ✅ (Keep from V11)
**Document**: `DESIGN_01_DATA_INGESTION.md`

- Multi-source data collection (SQL, OPC-UA, CSV)
- Schema validation with physics constraints
- Data quality assessment and repair
- Temporal alignment and resampling

### Phase 2: Physics-Informed Feature Engineering 🔄 (Enhanced)
**Document**: `DESIGN_02_FEATURE_ENGINEERING.md`

- Raw sensor features (current implementation)
- **NEW**: Causal lag features (temp follows power with 5-min lag)
- **NEW**: Thermodynamic features (efficiency ratios, energy balance)
- **NEW**: Frequency domain features (vibration harmonics, FFT)
- **NEW**: Process relationship features (flow/pressure ratios)

### Phase 3: Semantic Operating Mode Discovery 🆕 (Redesign)
**Document**: `DESIGN_03_OPERATING_MODES.md`

- **NEW**: Hybrid clustering (K-Means + physics constraints)
- **NEW**: State transition validation (startup must precede full-load)
- **NEW**: Temporal structure validation (minimum dwell times)
- **NEW**: Fleet-wide regime alignment (consistent labels across equipment)
- **NEW**: Semantic naming (auto-labeling based on sensor signatures)

### Phase 4: Multi-Level Anomaly Detection 🔄 (Enhanced)
**Document**: `DESIGN_04_ANOMALY_DETECTION.md`

- Keep: 6 detector architecture (AR1, PCA, IForest, GMM, OMR)
- **NEW**: Covariance-aware fusion (Mahalanobis distance)
- **NEW**: Adaptive thresholds (regime-specific, seasonal-adjusted, age-adjusted)
- **NEW**: Hierarchical detection (sensor → subsystem → equipment)
- **NEW**: Maintenance event detection (health jump recognition)

### Phase 5: Fault Signature Learning & Classification 🆕 (New Capability)
**Document**: `DESIGN_05_FAULT_CLASSIFICATION.md`

- **NEW**: Fault signature extraction (detector responses + sensor patterns + context)
- **NEW**: Unsupervised fault clustering (build learned taxonomy)
- **NEW**: Signature matching (assign new faults to historical clusters)
- **NEW**: Fleet-wide fault library (aggregate patterns across equipment)
- **NEW**: Semantic fault naming (auto-generate labels from signatures)

### Phase 6: Failure Mode Prediction 🔄 (Enhanced)
**Document**: `DESIGN_06_FAILURE_PREDICTION.md`

- Keep: RUL estimation with confidence bounds
- **NEW**: Failure mode taxonomy (bearing, motor, sensor, control)
- **NEW**: Multi-path degradation models (different modes have different trajectories)
- **NEW**: Maintenance-aware forecasting (detect and handle health resets)
- **NEW**: Failure signature prediction (what sensors will fail, in what pattern)

### Phase 7: Fleet Learning & Transfer 🆕 (Activate)
**Document**: `DESIGN_07_FLEET_LEARNING.md`

- **NEW**: Cold-start transfer learning (bootstrap from similar equipment)
- **NEW**: Fleet-wide regime alignment (consistent cluster IDs)
- **NEW**: Fault pattern aggregation (learn from all equipment)
- **NEW**: Similarity-based model sharing (transfer successful models)

---

## Detailed Design Documents

Each phase has a dedicated design document with:

1. **Purpose & Goals**: What this phase achieves
2. **Current State Analysis**: What V11 does (good and bad)
3. **Proposed Solution**: Detailed algorithm design
4. **Mathematical Foundation**: Equations, proofs, validations
5. **Implementation Plan**: Pseudocode, data structures, APIs
6. **Testing Strategy**: Unit tests, integration tests, validation metrics
7. **Migration Path**: How to evolve from V11

### Design Document Index

1. **DESIGN_01_DATA_INGESTION.md**: Data collection, validation, quality control
2. **DESIGN_02_FEATURE_ENGINEERING.md**: Physics-informed feature extraction
3. **DESIGN_03_OPERATING_MODES.md**: Semantic regime discovery and identification
4. **DESIGN_04_ANOMALY_DETECTION.md**: Multi-level fault detection with adaptive thresholds
5. **DESIGN_05_FAULT_CLASSIFICATION.md**: Unsupervised fault signature learning and taxonomy
6. **DESIGN_06_FAILURE_PREDICTION.md**: Failure mode prediction and RUL forecasting
7. **DESIGN_07_FLEET_LEARNING.md**: Transfer learning and fleet-wide knowledge sharing

---

## Key Innovations Over V11

### 1. Physics-Informed Clustering (Phase 3)

**Problem**: K-Means finds density clusters, not operational modes

**Solution**: Hybrid approach combining statistical clustering with physics constraints

```python
class PhysicsInformedClustering:
    def fit(self, sensor_data, physics_constraints):
        # Step 1: Standard K-Means initialization
        kmeans = MiniBatchKMeans(n_clusters=k)
        initial_labels = kmeans.fit_predict(sensor_data)
        
        # Step 2: Physics validation
        valid_transitions = validate_state_transitions(
            labels=initial_labels,
            timestamps=sensor_data.index,
            constraints=physics_constraints  # startup → full-load, not reverse
        )
        
        # Step 3: Temporal validation
        valid_dwells = validate_dwell_times(
            labels=initial_labels,
            min_dwell={'startup': 5_min, 'full_load': 30_min}
        )
        
        # Step 4: Causal validation
        valid_causality = validate_causal_structure(
            labels=initial_labels,
            sensor_data=sensor_data,
            causality={'power→temp': 5_min_lag}
        )
        
        # Step 5: Refine clusters that fail validation
        refined_labels = refine_clusters(
            initial_labels,
            validation_failures=[~valid_transitions, ~valid_dwells, ~valid_causality]
        )
        
        return refined_labels, semantic_labels
```

**Impact**: Clusters correspond to operational modes, not statistical artifacts

---

### 2. Covariance-Aware Detector Fusion (Phase 4)

**Problem**: Weighted sum assumes detector independence (PCA-SPE and PCA-T² are 80% correlated)

**Solution**: Mahalanobis fusion using detector covariance matrix

```python
class CovarianceAwareFusion:
    def __init__(self):
        self.detector_cov = None  # Learned during training
        
    def fit(self, training_detectors):
        # Compute detector covariance matrix
        detector_matrix = np.column_stack([
            training_detectors['ar1_z'],
            training_detectors['pca_spe_z'],
            training_detectors['pca_t2_z'],
            training_detectors['iforest_z'],
            training_detectors['gmm_z'],
            training_detectors['omr_z'],
        ])
        self.detector_cov = np.cov(detector_matrix.T)
        self.detector_mean = detector_matrix.mean(axis=0)
        
    def fuse(self, detector_scores):
        # Mahalanobis distance instead of weighted sum
        z_vector = np.array([
            detector_scores['ar1_z'],
            detector_scores['pca_spe_z'],
            detector_scores['pca_t2_z'],
            detector_scores['iforest_z'],
            detector_scores['gmm_z'],
            detector_scores['omr_z'],
        ])
        
        diff = z_vector - self.detector_mean
        fused_z = np.sqrt(diff @ np.linalg.inv(self.detector_cov) @ diff.T)
        
        return fused_z
```

**Impact**: Correct fusion variance, lower false positive rate

---

### 3. Fault Signature Clustering (Phase 5)

**Problem**: Detector names ≠ fault types (can't classify faults without labels)

**Solution**: Learn fault taxonomy from historical fault signatures

```python
class FaultSignatureLearning:
    def __init__(self):
        self.fault_clusters = None
        self.cluster_signatures = {}
        
    def extract_signature(self, episode):
        # Multi-dimensional signature
        signature = {
            'detector_response': {
                'ar1': episode['ar1_z'].max(),
                'pca_spe': episode['pca_spe_z'].max(),
                'pca_t2': episode['pca_t2_z'].max(),
                'iforest': episode['iforest_z'].max(),
                'gmm': episode['gmm_z'].max(),
                'omr': episode['omr_z'].max(),
            },
            'sensor_pattern': {
                'top_sensor': episode['sensors'].idxmax(),
                'sensor_variance': episode['sensors'].var(axis=0).values,
                'temporal_pattern': extract_temporal_pattern(episode),
            },
            'operational_context': {
                'regime': episode['regime_label'].mode(),
                'health_before': episode['health_before'],
                'duration_hours': episode['duration'],
            },
            'frequency_signature': {
                'vibration_harmonics': fft_harmonics(episode['vibration']),
                'dominant_frequency': dominant_freq(episode),
            }
        }
        return signature
    
    def build_fault_taxonomy(self, historical_episodes):
        # Extract signatures from all episodes
        signatures = [self.extract_signature(ep) for ep in historical_episodes]
        
        # Flatten signatures to feature vectors
        signature_vectors = flatten_signatures(signatures)
        
        # Cluster fault signatures (unsupervised)
        clustering = HDBSCAN(min_cluster_size=5)
        fault_labels = clustering.fit_predict(signature_vectors)
        
        # Build cluster prototypes
        for cluster_id in np.unique(fault_labels):
            cluster_mask = fault_labels == cluster_id
            cluster_signatures = [sig for sig, mask in zip(signatures, cluster_mask) if mask]
            
            # Compute prototype (median signature)
            prototype = compute_prototype(cluster_signatures)
            
            # Auto-generate semantic label
            semantic_label = generate_fault_label(prototype)
            # e.g., "High-Vibration-Bearing-Like" or "Current-Spike-Motor-Like"
            
            self.cluster_signatures[cluster_id] = {
                'prototype': prototype,
                'semantic_label': semantic_label,
                'count': cluster_mask.sum(),
                'examples': cluster_signatures[:5],  # Keep examples
            }
    
    def classify_fault(self, new_episode):
        # Extract signature
        new_signature = self.extract_signature(new_episode)
        new_vector = flatten_signature(new_signature)
        
        # Find closest cluster
        distances = {}
        for cluster_id, cluster_data in self.cluster_signatures.items():
            prototype_vector = flatten_signature(cluster_data['prototype'])
            distances[cluster_id] = euclidean_distance(new_vector, prototype_vector)
        
        best_cluster = min(distances, key=distances.get)
        confidence = 1 / (1 + distances[best_cluster])
        
        return {
            'fault_type': self.cluster_signatures[best_cluster]['semantic_label'],
            'cluster_id': best_cluster,
            'confidence': confidence,
            'similar_examples': self.cluster_signatures[best_cluster]['examples'],
        }
```

**Impact**: True unsupervised fault classification with learned taxonomy

---

### 4. Maintenance-Aware Forecasting (Phase 6)

**Problem**: Assumes monotonic degradation (false alarms after maintenance)

**Solution**: Detect health jumps and reset forecasts

```python
class MaintenanceAwareForecasting:
    def __init__(self):
        self.maintenance_threshold = 10  # Health jump > 10 points
        self.forecasts = {}
        
    def detect_maintenance(self, health_timeline):
        # Detect significant health jumps (maintenance events)
        health_diff = health_timeline.diff()
        maintenance_events = health_diff > self.maintenance_threshold
        
        if maintenance_events.any():
            event_times = health_timeline.index[maintenance_events]
            return event_times
        return []
    
    def forecast_with_maintenance_awareness(self, health_timeline):
        # Check for recent maintenance
        maintenance_times = self.detect_maintenance(health_timeline)
        
        if maintenance_times:
            # Reset forecast after last maintenance
            last_maintenance = maintenance_times[-1]
            relevant_timeline = health_timeline[health_timeline.index > last_maintenance]
            
            # Log maintenance event
            log_maintenance_event(last_maintenance, health_jump=health_timeline.loc[last_maintenance])
        else:
            relevant_timeline = health_timeline
        
        # Forecast on post-maintenance data only
        forecast = exponential_smoothing(relevant_timeline, alpha=0.3, horizon=168)
        
        # Compute RUL
        rul = time_to_threshold(forecast, threshold=50, current_health=relevant_timeline.iloc[-1])
        
        return {
            'rul_hours': rul,
            'forecast': forecast,
            'maintenance_detected': len(maintenance_times) > 0,
            'last_maintenance': maintenance_times[-1] if maintenance_times else None,
        }
```

**Impact**: No false alarms after maintenance, accurate post-maintenance RUL

---

### 5. Fleet-Wide Transfer Learning (Phase 7)

**Problem**: New equipment starts from scratch (7-day cold-start)

**Solution**: Bootstrap from similar equipment

```python
class FleetTransferLearning:
    def __init__(self):
        self.equipment_profiles = {}
        self.regime_mappings = {}
        
    def bootstrap_new_equipment(self, target_id, target_type, target_data):
        # Find similar equipment
        similar_equipment = self.find_similar(target_type, min_similarity=0.7)
        
        if not similar_equipment:
            # No transfer possible - use traditional cold-start
            return None
        
        # Select best match
        best_match = similar_equipment[0]
        
        # Transfer regime model
        source_regime_model = load_regime_model(best_match.equip_id)
        
        # Adapt to target equipment scale
        target_profile = build_quick_profile(target_data)  # 30 samples enough
        adaptation_factors = compute_adaptation_factors(
            source_profile=self.equipment_profiles[best_match.equip_id],
            target_profile=target_profile
        )
        
        # Scale centroids
        adapted_centroids = source_regime_model.centroids * adaptation_factors
        
        # Create adapted model
        adapted_model = RegimeModel(
            centroids=adapted_centroids,
            feature_columns=source_regime_model.feature_columns,
            scaler=adapt_scaler(source_regime_model.scaler, adaptation_factors),
            transferred_from=best_match.equip_id,
            adaptation_confidence=best_match.similarity,
        )
        
        # Transfer detector models
        adapted_detectors = {}
        for detector_name in ['ar1', 'pca', 'iforest', 'gmm', 'omr']:
            source_detector = load_detector(best_match.equip_id, detector_name)
            adapted_detectors[detector_name] = adapt_detector(
                source_detector,
                adaptation_factors,
                target_data
            )
        
        return {
            'regime_model': adapted_model,
            'detectors': adapted_detectors,
            'cold_start_days': 1,  # vs 7 without transfer
            'transfer_source': best_match.equip_id,
            'transfer_confidence': best_match.similarity,
        }
    
    def align_regimes_fleet_wide(self, equipment_type):
        # Get all equipment of this type
        equipment_list = get_equipment_by_type(equipment_type)
        
        # Build consensus regime definitions
        all_centroids = []
        for equip_id in equipment_list:
            regime_model = load_regime_model(equip_id)
            if regime_model:
                all_centroids.append(regime_model.centroids)
        
        # Cluster centroids to find consensus regimes
        flattened_centroids = np.vstack(all_centroids)
        consensus_clustering = KMeans(n_clusters=4)  # Assume 4 common modes
        consensus_labels = consensus_clustering.fit_predict(flattened_centroids)
        consensus_centroids = consensus_clustering.cluster_centers_
        
        # Assign semantic labels to consensus regimes
        semantic_labels = auto_label_regimes(consensus_centroids, equipment_type)
        # e.g., {0: "idle", 1: "startup", 2: "full_load", 3: "shutdown"}
        
        # Map each equipment's local regimes to consensus regimes
        for equip_id in equipment_list:
            local_model = load_regime_model(equip_id)
            mapping = map_local_to_consensus(
                local_centroids=local_model.centroids,
                consensus_centroids=consensus_centroids
            )
            self.regime_mappings[equip_id] = {
                'local_to_consensus': mapping,
                'semantic_labels': semantic_labels,
            }
        
        return consensus_centroids, semantic_labels
```

**Impact**: 1-day cold-start, consistent regime IDs across fleet

---

## System Performance Targets

### Cold-Start Performance
- **Target**: Operational predictions within 24 hours (vs 7 days in V11)
- **Method**: Transfer learning from similar equipment
- **Fallback**: If no similar equipment, 3-day minimum (relaxed promotion criteria)

### Regime Discovery Accuracy
- **Target**: >90% semantic correctness (validated against physics)
- **Metric**: Cluster-to-operational-mode alignment score
- **Validation**: State transition validity, dwell time conformance

### Fault Detection Performance
- **Target**: <1% false positive rate (vs ~3% in V11)
- **Method**: Covariance-aware fusion + adaptive thresholds
- **Metric**: Precision/recall on held-out anomaly data

### Fault Classification Accuracy
- **Target**: >80% match to human expert classification (validation only)
- **Method**: Learned fault taxonomy with signature matching
- **Metric**: Cluster purity when compared to expert labels (validation)

### RUL Prediction Accuracy
- **Target**: ±20% error at 7-day horizon (vs ±40% in V11)
- **Method**: Maintenance-aware forecasting + failure mode modeling
- **Metric**: Mean absolute percentage error (MAPE)

---

## Migration Strategy from V11

### Phase A: Quick Wins (1-2 weeks)
1. Fix V11 critical bugs (variable scoping, race conditions)
2. Relax promotion criteria to realistic values
3. Activate transfer learning (infrastructure exists)
4. Add maintenance event detection

### Phase B: Core Enhancements (4-6 weeks)
1. Implement physics-informed feature engineering
2. Add covariance-aware detector fusion
3. Implement adaptive threshold system
4. Build fault signature extraction

### Phase C: New Capabilities (8-12 weeks)
1. Implement physics-informed clustering
2. Build fault signature clustering and taxonomy
3. Implement failure mode prediction
4. Deploy fleet-wide regime alignment

### Phase D: Production Hardening (4-6 weeks)
1. Comprehensive testing (unit, integration, end-to-end)
2. Performance optimization (batch processing, caching)
3. Observability enhancement (dashboards, alerting)
4. Documentation and training

**Total Timeline**: 16-24 weeks to full deployment

---

## Technology Stack Recommendations

### Core ML/Statistical Libraries
- **sklearn**: Keep for basic clustering, PCA, isolation forest
- **hdbscan**: Add for density-based clustering with variable densities
- **statsmodels**: Add for ARIMA, state-space models (better than exponential smoothing)
- **scipy**: Keep for statistical tests, optimization

### Time Series & Forecasting
- **prophet**: Add for seasonal-aware forecasting (Facebook's library)
- **pmdarima**: Add for auto-ARIMA (better than manual fitting)
- **tslearn**: Add for time series clustering and DTW

### Physics-Informed ML
- **sympy**: Add for symbolic physics equations
- **control**: Add for control systems analysis (transfer functions)
- **thermopy**: Add for thermodynamic calculations

### Scalability
- **dask**: Add for parallel processing (100s of equipment)
- **ray**: Add for distributed training (fleet-wide learning)
- **joblib**: Keep for model persistence

### Validation & Explainability
- **shap**: Add for feature importance (explain predictions)
- **eli5**: Add for model interpretation
- **dtreeviz**: Add for decision tree visualization

---

## Success Metrics

### Technical Metrics
1. **Cold-start time**: <24 hours (vs 7 days)
2. **False positive rate**: <1% (vs ~3%)
3. **RUL accuracy**: MAPE <20% at 7-day horizon
4. **Regime discovery**: >90% physics-validated
5. **Fault classification**: >80% cluster purity

### Business Metrics
1. **Equipment downtime**: Reduce by 30%
2. **Maintenance efficiency**: Reduce unnecessary maintenance by 50%
3. **Spare parts inventory**: Reduce by 20% (predict what will fail)
4. **Fleet coverage**: 100s of equipment (vs 10s in V11)
5. **Time to value**: 24 hours (vs 7 days)

### Operational Metrics
1. **Pipeline reliability**: >99.9% uptime
2. **Processing latency**: <5 minutes for batch, <100ms for online
3. **Storage efficiency**: <1GB per equipment-year
4. **Compute cost**: <$1 per equipment-month
5. **Model quality**: >95% passing maturity criteria within 3 days

---

## Next Steps

**Immediate**: Read the 7 detailed design documents for each phase

**Priority Order**:
1. **DESIGN_03_OPERATING_MODES.md**: Biggest analytical improvement (semantic correctness)
2. **DESIGN_05_FAULT_CLASSIFICATION.md**: Enables true unsupervised diagnosis
3. **DESIGN_07_FLEET_LEARNING.md**: Solves cold-start problem
4. **DESIGN_04_ANOMALY_DETECTION.md**: Improves detection accuracy
5. **DESIGN_06_FAILURE_PREDICTION.md**: Better RUL predictions
6. **DESIGN_02_FEATURE_ENGINEERING.md**: Foundation for everything
7. **DESIGN_01_DATA_INGESTION.md**: V11 already good, minor enhancements

Each document contains:
- Detailed algorithms with pseudocode
- Mathematical foundations with proofs
- Implementation specifications
- Testing strategies
- API designs
- Migration paths

---

**End of Master Design Document**

**Next**: Proceed to individual design documents for implementation details
