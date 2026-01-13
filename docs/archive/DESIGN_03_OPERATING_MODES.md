# DESIGN-03: Semantic Operating Mode Discovery

**Phase**: 3 of 7  
**Priority**: CRITICAL (Biggest analytical improvement)  
**Complexity**: High  
**Dependencies**: Phase 2 (Feature Engineering)

---

## Purpose & Goals

**Transform regime discovery from statistical clustering to semantic operational mode identification.**

### Current Problems (V11)
1. K-Means finds density clusters, not operational modes
2. Silhouette score favors separation over semantic correctness
3. No physics constraints (clusters ignore causality)
4. No fleet consistency (cluster 0 ≠ cluster 0 across equipment)
5. Can't name discovered modes ("cluster 0" vs "full-load")

### Target Capabilities
1. ✅ Discover operational modes that match physics (idle, startup, full-load, shutdown)
2. ✅ Validate clusters against temporal structure (state transitions, dwell times)
3. ✅ Align regimes across fleet (consistent cluster IDs for same equipment type)
4. ✅ Auto-generate semantic labels ("startup" not "cluster 1")
5. ✅ Confidence scoring for regime assignments

---

## Architectural Overview

### Three-Layer Approach

```
┌─────────────────────────────────────────────┐
│  Layer 1: Statistical Clustering            │
│  (Find candidate groupings in data)         │
│  Algorithm: MiniBatchKMeans + HDBSCAN       │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Layer 2: Physics Validation                │
│  (Validate against domain constraints)      │
│  • State transition validity                │
│  • Temporal structure (dwell times)         │
│  • Causal relationships (power→temp)        │
│  • Energy balance constraints               │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Layer 3: Semantic Identification           │
│  (Name clusters based on signatures)        │
│  • Sensor signature matching                │
│  • Fleet-wide alignment                     │
│  • Auto-labeling                            │
└─────────────────────────────────────────────┘
```

---

## Layer 1: Statistical Clustering

### Algorithm: Hybrid K-Means + HDBSCAN

**Why Hybrid?**
- K-Means: Good for globular clusters (most operational modes)
- HDBSCAN: Handles variable density (transient states, rare modes)

### Implementation

```python
class HybridRegimeClustering:
    """
    Combines K-Means (for main operational modes) with HDBSCAN (for transients).
    """
    
    def __init__(self, config):
        self.k_min = config.get('k_min', 3)  # idle, run, shutdown minimum
        self.k_max = config.get('k_max', 6)
        self.hdbscan_min_cluster_size = config.get('hdbscan_min_size', 50)
        
    def fit(self, X, timestamps, feature_names):
        """
        Fit hybrid clustering model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            timestamps: DatetimeIndex for temporal validation
            feature_names: List of feature names for signature extraction
            
        Returns:
            labels: Cluster assignments (-1 = transient/rare state)
            model: Fitted clustering model
            metadata: Quality metrics and validation results
        """
        # Step 1: K-Means sweep for main modes
        kmeans_results = self._fit_kmeans_sweep(X, timestamps)
        
        # Step 2: HDBSCAN for outliers/transients
        hdbscan_results = self._fit_hdbscan(X, timestamps)
        
        # Step 3: Combine results
        combined_labels = self._combine_clusterings(
            kmeans_labels=kmeans_results['labels'],
            hdbscan_labels=hdbscan_results['labels'],
            X=X,
            timestamps=timestamps
        )
        
        return combined_labels, kmeans_results['model'], {
            'kmeans_k': kmeans_results['k'],
            'kmeans_silhouette': kmeans_results['silhouette'],
            'hdbscan_clusters': len(np.unique(hdbscan_results['labels'])),
            'final_k': len(np.unique(combined_labels)),
        }
    
    def _fit_kmeans_sweep(self, X, timestamps):
        """K-Means with physics-informed k-selection."""
        best_score = -np.inf
        best_k = self.k_min
        best_model = None
        best_labels = None
        
        for k in range(self.k_min, self.k_max + 1):
            # Fit K-Means
            kmeans = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Compute composite score (not just silhouette)
            score = self._compute_composite_score(
                X=X,
                labels=labels,
                timestamps=timestamps,
                centroids=kmeans.cluster_centers_
            )
            
            if score > best_score:
                best_score = score
                best_k = k
                best_model = kmeans
                best_labels = labels
        
        return {
            'k': best_k,
            'model': best_model,
            'labels': best_labels,
            'silhouette': silhouette_score(X, best_labels),
            'composite_score': best_score,
        }
    
    def _compute_composite_score(self, X, labels, timestamps, centroids):
        """
        Composite score combining statistical and physics criteria.
        
        Score = 0.3*silhouette + 0.3*temporal_validity + 0.2*transition_validity + 0.2*stability
        """
        # Statistical: Silhouette score
        sil_score = silhouette_score(X, labels, sample_size=min(4000, len(X)))
        
        # Temporal: Dwell time validity
        temporal_score = self._temporal_validity_score(labels, timestamps)
        
        # Physics: State transition validity
        transition_score = self._transition_validity_score(labels, timestamps)
        
        # Stability: Reproducibility across bootstrap samples
        stability_score = self._stability_score(X, labels, centroids)
        
        composite = (
            0.3 * sil_score +
            0.3 * temporal_score +
            0.2 * transition_score +
            0.2 * stability_score
        )
        
        return composite
    
    def _temporal_validity_score(self, labels, timestamps):
        """
        Validate that clusters have reasonable dwell times.
        
        Operating modes should persist for minutes to hours, not seconds.
        """
        # Compute run lengths for each label
        runs = self._compute_run_lengths(labels, timestamps)
        
        # Expected dwell times (domain knowledge)
        expected_dwells = {
            'min_dwell_minutes': 2,   # At least 2 minutes per mode
            'max_flicker_rate': 0.1,  # <10% rapid transitions
        }
        
        # Check minimum dwell
        median_dwells = {label: np.median(run_lengths) 
                        for label, run_lengths in runs.items()}
        
        min_dwell_met = sum(
            dwell >= expected_dwells['min_dwell_minutes'] 
            for dwell in median_dwells.values()
        ) / len(median_dwells)
        
        # Check flicker rate (too many rapid transitions = bad clustering)
        transition_count = np.sum(np.diff(labels) != 0)
        flicker_rate = transition_count / len(labels)
        flicker_penalty = max(0, 1 - flicker_rate / expected_dwells['max_flicker_rate'])
        
        temporal_score = 0.7 * min_dwell_met + 0.3 * flicker_penalty
        return temporal_score
    
    def _transition_validity_score(self, labels, timestamps):
        """
        Validate state transitions against physics.
        
        Examples of invalid transitions:
        - full_load → idle (should pass through shutdown)
        - shutdown → startup (equipment must stop first)
        """
        # Build transition matrix
        transition_matrix = self._build_transition_matrix(labels)
        
        # Define allowed transitions (this is domain knowledge, not labels!)
        # We infer likely roles by power signature
        power_signatures = self._infer_power_signatures(labels)
        
        # Sort clusters by average power (proxy for operational mode)
        sorted_clusters = sorted(power_signatures.items(), key=lambda x: x[1])
        low_power, med_power, high_power = [c[0] for c in sorted_clusters[:3]]
        
        # Forbidden transitions
        forbidden = [
            (high_power, low_power),  # full_load → idle (must shutdown first)
        ]
        
        # Count forbidden transitions
        forbidden_count = sum(
            transition_matrix[i, j] for i, j in forbidden if i < len(transition_matrix) and j < len(transition_matrix[0])
        )
        
        total_transitions = np.sum(transition_matrix) - np.trace(transition_matrix)
        
        if total_transitions == 0:
            return 0.0
        
        transition_score = 1 - (forbidden_count / total_transitions)
        return max(0.0, transition_score)
    
    def _stability_score(self, X, labels, centroids):
        """
        Stability: How reproducible are clusters across bootstrap samples?
        
        Good clustering: Same k and similar centroids across subsamples
        Bad clustering: Different k or very different centroids
        """
        n_bootstrap = 5
        n_samples = len(X)
        sample_size = int(0.8 * n_samples)
        
        bootstrap_ks = []
        bootstrap_centroids = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            X_boot = X[indices]
            
            # Re-cluster
            kmeans_boot = MiniBatchKMeans(n_clusters=len(np.unique(labels)), random_state=None)
            kmeans_boot.fit(X_boot)
            
            bootstrap_ks.append(kmeans_boot.n_clusters)
            bootstrap_centroids.append(kmeans_boot.cluster_centers_)
        
        # Check k stability
        k_std = np.std(bootstrap_ks)
        k_stability = np.exp(-k_std)  # Penalize variance in k
        
        # Check centroid stability (average pairwise distance)
        centroid_distances = []
        for i in range(n_bootstrap):
            for j in range(i+1, n_bootstrap):
                # Match centroids (Hungarian algorithm)
                matched_dist = self._match_centroids(
                    bootstrap_centroids[i],
                    bootstrap_centroids[j]
                )
                centroid_distances.append(matched_dist)
        
        avg_centroid_dist = np.mean(centroid_distances) if centroid_distances else 0
        centroid_stability = np.exp(-avg_centroid_dist)
        
        stability = 0.5 * k_stability + 0.5 * centroid_stability
        return stability
```

---

## Layer 2: Physics Validation

### Causal Relationship Validation

```python
class PhysicsValidator:
    """
    Validate clusters against physical laws and domain knowledge.
    """
    
    def __init__(self, equipment_type):
        self.equipment_type = equipment_type
        self.causal_relationships = self._load_causal_model(equipment_type)
        
    def _load_causal_model(self, equipment_type):
        """
        Define causal relationships for equipment type.
        
        For example, for a motor-driven fan:
        - Power consumption → Temperature (5-min lag)
        - Speed → Vibration (instant)
        - Flow → Pressure (2-min lag)
        """
        causal_models = {
            'FD_FAN': {
                'power→temperature': {'lag_minutes': 5, 'correlation_min': 0.6},
                'speed→vibration': {'lag_minutes': 0, 'correlation_min': 0.7},
                'flow→pressure': {'lag_minutes': 2, 'correlation_min': 0.5},
            },
            'GAS_TURBINE': {
                'fuel_flow→exhaust_temp': {'lag_minutes': 1, 'correlation_min': 0.8},
                'power→vibration': {'lag_minutes': 0, 'correlation_min': 0.6},
            },
        }
        return causal_models.get(equipment_type, {})
    
    def validate_clusters(self, X, labels, timestamps, sensor_data):
        """
        Validate that clusters respect causal relationships.
        
        Returns:
            validation_results: Dict with scores per cluster
        """
        results = {}
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = sensor_data[cluster_mask]
            cluster_times = timestamps[cluster_mask]
            
            # Validate each causal relationship
            causal_scores = {}
            for relationship, params in self.causal_relationships.items():
                cause, effect = relationship.split('→')
                
                if cause in cluster_data.columns and effect in cluster_data.columns:
                    score = self._validate_causal_relationship(
                        cause_data=cluster_data[cause],
                        effect_data=cluster_data[effect],
                        lag_minutes=params['lag_minutes'],
                        min_correlation=params['correlation_min'],
                        timestamps=cluster_times
                    )
                    causal_scores[relationship] = score
            
            # Overall cluster validity
            if causal_scores:
                cluster_validity = np.mean(list(causal_scores.values()))
            else:
                cluster_validity = 0.5  # Neutral if no causal relationships defined
            
            results[cluster_id] = {
                'causal_scores': causal_scores,
                'overall_validity': cluster_validity,
                'passes_validation': cluster_validity > 0.6,
            }
        
        return results
    
    def _validate_causal_relationship(self, cause_data, effect_data, lag_minutes, min_correlation, timestamps):
        """
        Validate that effect follows cause with expected lag and correlation.
        """
        # Compute lag in samples
        if len(timestamps) > 1:
            sample_rate = (timestamps[1] - timestamps[0]).total_seconds() / 60  # minutes per sample
            lag_samples = int(lag_minutes / sample_rate)
        else:
            lag_samples = 0
        
        # Shift effect by lag
        if lag_samples > 0:
            cause_aligned = cause_data[:-lag_samples]
            effect_aligned = effect_data[lag_samples:]
        else:
            cause_aligned = cause_data
            effect_aligned = effect_data
        
        # Compute correlation
        if len(cause_aligned) > 2:
            correlation, p_value = pearsonr(cause_aligned, effect_aligned)
        else:
            correlation = 0
            p_value = 1.0
        
        # Score based on correlation strength and statistical significance
        if correlation >= min_correlation and p_value < 0.05:
            score = 1.0
        elif correlation >= min_correlation * 0.8:
            score = 0.7
        elif correlation >= min_correlation * 0.6:
            score = 0.4
        else:
            score = 0.0
        
        return score
```

### Energy Balance Validation

```python
class EnergyBalanceValidator:
    """
    Validate clusters against thermodynamic constraints.
    
    Example: Power in ≈ Heat out + Useful work (efficiency losses)
    """
    
    def validate_energy_balance(self, cluster_data, equipment_type):
        """
        Check if cluster respects energy conservation.
        
        For example, for a motor:
        Electrical Power In = Mechanical Power Out + Heat Losses
        """
        if equipment_type == 'FD_FAN':
            return self._validate_fan_energy_balance(cluster_data)
        elif equipment_type == 'GAS_TURBINE':
            return self._validate_turbine_energy_balance(cluster_data)
        else:
            return 0.5  # Neutral if no model defined
    
    def _validate_fan_energy_balance(self, data):
        """
        Fan energy balance: Power ≈ (Flow × Pressure) / Efficiency + Losses
        """
        required_cols = ['Power', 'Flow', 'Pressure', 'Temperature']
        if not all(col in data.columns for col in required_cols):
            return 0.5
        
        # Compute hydraulic power
        hydraulic_power = data['Flow'] * data['Pressure'] / 1000  # kW
        
        # Expected efficiency range (60-85% for fans)
        electrical_power = data['Power']
        implied_efficiency = hydraulic_power / (electrical_power + 1e-6)
        
        # Check if efficiency is physical
        efficiency_valid = ((implied_efficiency >= 0.4) & (implied_efficiency <= 0.9)).mean()
        
        # Check if losses correlate with temperature
        losses = electrical_power - hydraulic_power
        temp_correlation = pearsonr(losses, data['Temperature'])[0]
        
        score = 0.7 * efficiency_valid + 0.3 * max(0, temp_correlation)
        return score
```

---

## Layer 3: Semantic Identification

### Auto-Labeling Based on Signatures

```python
class SemanticRegimeLabeler:
    """
    Auto-generate semantic labels for clusters based on sensor signatures.
    """
    
    def __init__(self, equipment_type):
        self.equipment_type = equipment_type
        self.signature_templates = self._load_signature_templates(equipment_type)
        
    def _load_signature_templates(self, equipment_type):
        """
        Define expected sensor signatures for known operational modes.
        
        These are NOT labels - they're physics-based expectations.
        """
        templates = {
            'FD_FAN': {
                'idle': {
                    'power': (0, 0.1),      # 0-10% of rated
                    'speed': (0, 0.05),     # 0-5% of max
                    'vibration': (0, 0.3),  # Low vibration
                    'temperature': (-0.5, 0.1),  # Near ambient (normalized)
                },
                'startup': {
                    'power': (0.1, 0.5),    # Rising power
                    'speed': (0.1, 0.7),    # Rising speed
                    'vibration': (0.3, 0.8), # High transient vibration
                    'temperature_trend': 'rising',  # Temperature increasing
                },
                'full_load': {
                    'power': (0.7, 1.0),    # High power
                    'speed': (0.8, 1.0),    # High speed
                    'vibration': (0.2, 0.6), # Moderate vibration
                    'temperature': (0.6, 1.0),  # High temperature
                },
                'shutdown': {
                    'power': (0.1, 0.4),    # Decreasing power
                    'speed': (0.1, 0.5),    # Decreasing speed
                    'vibration': (0.1, 0.5), # Decreasing vibration
                    'temperature_trend': 'falling',  # Temperature decreasing
                },
            }
        }
        return templates.get(equipment_type, {})
    
    def label_clusters(self, cluster_centroids, cluster_stats, feature_names):
        """
        Assign semantic labels to clusters based on sensor signatures.
        
        Args:
            cluster_centroids: Cluster centers in feature space
            cluster_stats: Additional statistics per cluster
            feature_names: Names of features
            
        Returns:
            labels: Dict mapping cluster_id → semantic_label
        """
        labels = {}
        
        for cluster_id, centroid in enumerate(cluster_centroids):
            # Extract sensor values from centroid
            sensor_signature = self._extract_sensor_signature(
                centroid,
                feature_names
            )
            
            # Match to templates
            matches = {}
            for template_name, template_signature in self.signature_templates.items():
                match_score = self._match_signature(
                    sensor_signature,
                    template_signature,
                    cluster_stats[cluster_id]
                )
                matches[template_name] = match_score
            
            # Best match
            if matches:
                best_label = max(matches, key=matches.get)
                confidence = matches[best_label]
                
                if confidence > 0.6:
                    labels[cluster_id] = best_label
                else:
                    # Low confidence - use descriptive label
                    labels[cluster_id] = self._generate_descriptive_label(
                        sensor_signature
                    )
            else:
                labels[cluster_id] = f"mode_{cluster_id}"
        
        return labels
    
    def _match_signature(self, observed, template, cluster_stats):
        """
        Compute match score between observed signature and template.
        """
        matches = []
        
        for sensor, expected_range in template.items():
            if sensor.endswith('_trend'):
                # Trend validation
                base_sensor = sensor.replace('_trend', '')
                if base_sensor in cluster_stats:
                    actual_trend = cluster_stats[base_sensor + '_trend']
                    matches.append(1.0 if actual_trend == expected_range else 0.0)
            else:
                # Range validation
                if sensor in observed:
                    value = observed[sensor]
                    if isinstance(expected_range, tuple):
                        low, high = expected_range
                        if low <= value <= high:
                            matches.append(1.0)
                        else:
                            # Partial credit for close matches
                            distance = min(abs(value - low), abs(value - high))
                            matches.append(max(0, 1 - distance))
                    else:
                        matches.append(0.5)  # Unknown format
        
        if matches:
            return np.mean(matches)
        return 0.0
    
    def _generate_descriptive_label(self, signature):
        """
        Generate descriptive label when no template matches.
        
        Example: "high_power_moderate_vibration"
        """
        descriptors = []
        
        if 'power' in signature:
            if signature['power'] > 0.7:
                descriptors.append('high_power')
            elif signature['power'] < 0.2:
                descriptors.append('low_power')
        
        if 'vibration' in signature:
            if signature['vibration'] > 0.6:
                descriptors.append('high_vibration')
        
        if 'temperature' in signature:
            if signature['temperature'] > 0.7:
                descriptors.append('high_temp')
        
        if descriptors:
            return '_'.join(descriptors)
        else:
            return 'unclassified_mode'
```

---

## Fleet-Wide Regime Alignment

### Consensus Regime Discovery

```python
class FleetRegimeAlignment:
    """
    Align regimes across equipment fleet for consistent identification.
    """
    
    def align_fleet_regimes(self, equipment_ids, equipment_type):
        """
        Build consensus regime definitions for equipment type.
        
        Args:
            equipment_ids: List of equipment IDs of same type
            equipment_type: Equipment type (e.g., 'FD_FAN')
            
        Returns:
            consensus_regimes: Cluster centers representing consensus modes
            mappings: Dict mapping each equipment's local regimes to consensus
        """
        # Collect all centroids from fleet
        all_centroids = []
        all_labels = []
        equipment_centroid_map = {}
        
        for equip_id in equipment_ids:
            regime_model = self._load_regime_model(equip_id)
            if regime_model and regime_model.centroids is not None:
                centroids = regime_model.centroids
                all_centroids.append(centroids)
                equipment_centroid_map[equip_id] = centroids
                all_labels.append(regime_model.semantic_labels)
        
        if not all_centroids:
            return None, {}
        
        # Stack all centroids
        stacked_centroids = np.vstack(all_centroids)
        
        # Cluster centroids to find consensus regimes
        # Use median number of clusters across equipment
        n_clusters_per_equipment = [len(c) for c in all_centroids]
        consensus_k = int(np.median(n_clusters_per_equipment))
        
        consensus_clustering = KMeans(n_clusters=consensus_k, random_state=42)
        consensus_labels = consensus_clustering.fit_predict(stacked_centroids)
        consensus_centroids = consensus_clustering.cluster_centers_
        
        # Assign semantic labels to consensus regimes
        labeler = SemanticRegimeLabeler(equipment_type)
        
        # Compute stats for consensus clusters
        consensus_stats = {}
        for i in range(consensus_k):
            mask = consensus_labels == i
            cluster_centroids = stacked_centroids[mask]
            consensus_stats[i] = {
                'count': mask.sum(),
                'std': np.std(cluster_centroids, axis=0).mean(),
            }
        
        consensus_semantic_labels = labeler.label_clusters(
            consensus_centroids,
            consensus_stats,
            feature_names=None  # Would need to pass from model
        )
        
        # Map each equipment's local regimes to consensus
        mappings = {}
        for equip_id, local_centroids in equipment_centroid_map.items():
            # Find closest consensus regime for each local regime
            local_to_consensus = {}
            for local_id, local_centroid in enumerate(local_centroids):
                distances = [
                    np.linalg.norm(local_centroid - consensus_centroid)
                    for consensus_centroid in consensus_centroids
                ]
                closest_consensus = np.argmin(distances)
                local_to_consensus[local_id] = {
                    'consensus_id': closest_consensus,
                    'consensus_label': consensus_semantic_labels.get(closest_consensus, f"mode_{closest_consensus}"),
                    'distance': distances[closest_consensus],
                }
            
            mappings[equip_id] = local_to_consensus
        
        return {
            'consensus_centroids': consensus_centroids,
            'consensus_labels': consensus_semantic_labels,
            'consensus_k': consensus_k,
        }, mappings
```

---

## Confidence Scoring for Regime Assignments

### Multi-Factor Confidence

```python
class RegimeAssignmentConfidence:
    """
    Compute confidence for regime assignments using multiple factors.
    """
    
    def compute_confidence(self, observation, assigned_regime, regime_model, context):
        """
        Confidence = f(distance, stability, physics_validity, historical_prevalence)
        
        Args:
            observation: Feature vector for current observation
            assigned_regime: Assigned cluster ID
            regime_model: Fitted regime model
            context: Additional context (previous regimes, timestamps, etc.)
            
        Returns:
            confidence: Float in [0, 1]
            factors: Dict with individual factor scores
        """
        factors = {}
        
        # Factor 1: Distance to centroid (closer = higher confidence)
        centroid = regime_model.centroids[assigned_regime]
        distance = np.linalg.norm(observation - centroid)
        
        # Normalize by typical cluster radius
        cluster_radius = self._estimate_cluster_radius(regime_model, assigned_regime)
        normalized_distance = distance / (cluster_radius + 1e-6)
        
        distance_confidence = np.exp(-normalized_distance)
        factors['distance'] = distance_confidence
        
        # Factor 2: Temporal stability (did we just switch? rapid switching = low confidence)
        if context.get('previous_regime') == assigned_regime:
            stability_confidence = 1.0
        else:
            # Just switched - check if transition is valid
            transition_valid = self._validate_transition(
                context.get('previous_regime'),
                assigned_regime,
                regime_model
            )
            stability_confidence = 0.7 if transition_valid else 0.3
        factors['stability'] = stability_confidence
        
        # Factor 3: Physics validity (does assignment respect causal relationships?)
        if 'sensor_data' in context:
            physics_confidence = self._validate_physics(
                context['sensor_data'],
                assigned_regime,
                regime_model
            )
        else:
            physics_confidence = 0.5
        factors['physics'] = physics_confidence
        
        # Factor 4: Historical prevalence (how common is this regime?)
        prevalence = regime_model.cluster_counts[assigned_regime] / regime_model.total_samples
        # Rare regimes get slightly lower confidence (could be noise)
        prevalence_confidence = min(1.0, prevalence * 10)  # 10% prevalence = 1.0 confidence
        factors['prevalence'] = prevalence_confidence
        
        # Combined confidence (weighted geometric mean)
        weights = {'distance': 0.4, 'stability': 0.3, 'physics': 0.2, 'prevalence': 0.1}
        confidence = np.prod([
            factors[k] ** weights[k]
            for k in weights.keys()
        ])
        
        return confidence, factors
```

---

## Testing & Validation

### Unit Tests

```python
def test_physics_validation():
    """Test that clusters respect causal relationships."""
    # Create synthetic data with known causal structure
    power = np.random.randn(1000)
    temperature = np.roll(power, 5) + np.random.randn(1000) * 0.1  # 5-step lag
    
    # Cluster into 2 modes (high power, low power)
    labels = (power > 0).astype(int)
    
    validator = PhysicsValidator('FD_FAN')
    results = validator.validate_clusters(
        X=None,
        labels=labels,
        timestamps=pd.date_range('2024-01-01', periods=1000, freq='1min'),
        sensor_data=pd.DataFrame({'power': power, 'temperature': temperature})
    )
    
    # Should detect high correlation with lag
    assert all(r['overall_validity'] > 0.6 for r in results.values())

def test_semantic_labeling():
    """Test auto-labeling of clusters."""
    # Create synthetic centroids
    centroids = np.array([
        [0.05, 0.02, 0.1],  # idle (low power, low speed, low vibration)
        [0.9, 0.95, 0.4],   # full_load (high power, high speed, moderate vibration)
    ])
    
    labeler = SemanticRegimeLabeler('FD_FAN')
    labels = labeler.label_clusters(
        cluster_centroids=centroids,
        cluster_stats={},
        feature_names=['power', 'speed', 'vibration']
    )
    
    assert 'idle' in labels.values()
    assert 'full_load' in labels.values()
```

### Integration Tests

```python
def test_end_to_end_regime_discovery():
    """Test complete regime discovery pipeline."""
    # Load real equipment data
    data = load_equipment_data('FD_FAN_001', days=7)
    
    # Extract features
    features = extract_regime_features(data)
    
    # Discover regimes
    clustering = HybridRegimeClustering(config={})
    labels, model, metadata = clustering.fit(
        X=features.values,
        timestamps=features.index,
        feature_names=features.columns
    )
    
    # Validate physics
    validator = PhysicsValidator('FD_FAN')
    validation_results = validator.validate_clusters(
        X=features.values,
        labels=labels,
        timestamps=features.index,
        sensor_data=data
    )
    
    # Auto-label
    labeler = SemanticRegimeLabeler('FD_FAN')
    semantic_labels = labeler.label_clusters(
        cluster_centroids=model.cluster_centers_,
        cluster_stats={},
        feature_names=features.columns
    )
    
    # Assertions
    assert metadata['final_k'] >= 3  # At least idle, run, shutdown
    assert all(r['passes_validation'] for r in validation_results.values())
    assert 'idle' in semantic_labels.values() or 'low_power' in semantic_labels.values()
```

---

## API Design

### Public Interface

```python
class SemanticRegimeDiscovery:
    """
    Public API for semantic operating mode discovery.
    
    Usage:
        discovery = SemanticRegimeDiscovery(equipment_type='FD_FAN')
        regime_model = discovery.fit(sensor_data)
        labels, confidence = discovery.predict(new_data)
        semantic_labels = discovery.get_semantic_labels()
    """
    
    def __init__(self, equipment_type, config=None):
        self.equipment_type = equipment_type
        self.config = config or {}
        
        self.clustering = HybridRegimeClustering(self.config)
        self.validator = PhysicsValidator(equipment_type)
        self.labeler = SemanticRegimeLabeler(equipment_type)
        self.confidence_scorer = RegimeAssignmentConfidence()
        
        self.regime_model = None
        self.semantic_labels = None
        
    def fit(self, sensor_data, timestamps=None):
        """
        Discover operating modes from historical data.
        
        Args:
            sensor_data: DataFrame with sensor columns
            timestamps: Optional DatetimeIndex (uses sensor_data.index if None)
            
        Returns:
            self: For method chaining
        """
        if timestamps is None:
            timestamps = sensor_data.index
        
        # Extract regime features
        features = self._extract_regime_features(sensor_data)
        
        # Cluster
        labels, model, metadata = self.clustering.fit(
            X=features.values,
            timestamps=timestamps,
            feature_names=features.columns
        )
        
        # Validate
        validation_results = self.validator.validate_clusters(
            X=features.values,
            labels=labels,
            timestamps=timestamps,
            sensor_data=sensor_data
        )
        
        # Auto-label
        self.semantic_labels = self.labeler.label_clusters(
            cluster_centroids=model.cluster_centers_,
            cluster_stats={},
            feature_names=features.columns
        )
        
        # Store
        self.regime_model = {
            'clustering_model': model,
            'centroids': model.cluster_centers_,
            'feature_columns': features.columns.tolist(),
            'semantic_labels': self.semantic_labels,
            'validation_results': validation_results,
            'metadata': metadata,
        }
        
        return self
    
    def predict(self, sensor_data, context=None):
        """
        Assign new observations to discovered regimes.
        
        Args:
            sensor_data: DataFrame with sensor columns
            context: Optional context for confidence scoring
            
        Returns:
            labels: Array of cluster IDs
            confidence: Array of confidence scores
        """
        if self.regime_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        features = self._extract_regime_features(sensor_data)
        model = self.regime_model['clustering_model']
        
        # Predict
        labels = model.predict(features.values)
        
        # Compute confidence
        confidence = np.zeros(len(labels))
        for i, (label, obs) in enumerate(zip(labels, features.values)):
            conf, _ = self.confidence_scorer.compute_confidence(
                observation=obs,
                assigned_regime=label,
                regime_model=self.regime_model,
                context=context or {}
            )
            confidence[i] = conf
        
        return labels, confidence
    
    def get_semantic_labels(self):
        """Get semantic labels for each cluster."""
        if self.semantic_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.semantic_labels
```

---

## Migration Path from V11

### Phase 1: Drop-In Replacement (2 weeks)
1. Implement `SemanticRegimeDiscovery` class
2. Add physics validation as optional step
3. Keep existing K-Means for comparison
4. A/B test: V11 K-Means vs Physics-Informed clustering

### Phase 2: Enable Physics Features (2 weeks)
1. Add causal lag features to feature engineering
2. Enable physics validation in clustering
3. Monitor improvement in cluster quality

### Phase 3: Enable Auto-Labeling (1 week)
1. Define signature templates for equipment types
2. Enable semantic labeling
3. Validate labels match operational expectations

### Phase 4: Fleet Alignment (2 weeks)
1. Implement fleet-wide consensus regimes
2. Deploy across all equipment of same type
3. Validate consistency across fleet

**Total Migration**: 7 weeks

---

## Success Metrics

### Technical Metrics
1. **Physics Validation Pass Rate**: >90% of clusters pass validation
2. **Semantic Correctness**: >85% of auto-labels match expected modes
3. **Fleet Consistency**: >80% regime alignment across equipment
4. **Confidence Calibration**: Confidence scores correlate with prediction accuracy

### Business Metrics
1. **Operator Trust**: Operators understand regime labels (vs "cluster 0")
2. **False Positive Reduction**: 30% reduction in per-regime anomaly detection FP
3. **Cross-Equipment Analysis**: Enable fleet-wide fault pattern analysis

---

**End of Design Document 03**

**Next**: Proceed to DESIGN_04_ANOMALY_DETECTION.md or DESIGN_05_FAULT_CLASSIFICATION.md
