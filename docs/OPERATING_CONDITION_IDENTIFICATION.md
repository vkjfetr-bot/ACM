# Operating Condition Identification in ACM: Self-Learning & Gradual Evolution

**Purpose**: Comprehensive guide to identifying equipment operating conditions without human labeling, with continuous improvement over time  
**Scope**: From cold-start through mature operation, handling new states, fleet learning, and operational mode evolution  
**Philosophy**: Start simple, learn continuously, improve gradually

---

## Executive Summary

Equipment operating condition identification must:
1. **Start automatically** with minimal data (cold-start)
2. **Learn incrementally** as more data arrives
3. **Evolve gracefully** as equipment behavior changes
4. **Transfer knowledge** across similar equipment
5. **Maintain consistency** while adapting to new patterns

**Key Insight**: Operating conditions emerge from patterns in sensor data, validated by physics, and stabilized through consensus.

---

## Part 1: What Are Operating Conditions?

### Definition

**Operating Condition** = A persistent, physically meaningful state characterized by:
- Stable sensor values (power, temperature, vibration, flow, pressure)
- Predictable state transitions (startup → full-load, not reverse)
- Minimum dwell time (seconds for transients, minutes for stable modes)
- Consistent physics relationships (power → temperature with lag)

### Examples for Different Equipment

**Motor-Driven Fan**:
- **Idle**: Motor off, zero flow, ambient temperature
- **Startup**: Motor accelerating, rising flow, increasing vibration
- **Full-Load**: Rated speed, maximum flow, stable operation
- **Part-Load**: Reduced speed, partial flow, efficient operation
- **Shutdown**: Deceleration, falling flow, coast-down

**Gas Turbine**:
- **Offline**: Zero fuel, ambient temperature, no rotation
- **Startup**: Fuel ignition, temperature rise, acceleration
- **Base-Load**: Design point operation, maximum efficiency
- **Peak-Load**: Maximum output, reduced efficiency
- **Hot-Standby**: Idle rotation, ready for load

**Pump System**:
- **Stopped**: Zero flow, zero discharge pressure
- **Priming**: Initial flow establishment, pressure buildup
- **Normal-Operation**: Design flow and pressure
- **Cavitation**: Erratic flow, pressure fluctuations, vibration
- **Runout**: Excessive flow, low discharge pressure

### Why Not Just Use Labels?

**Human labels are**:
- Expensive (requires domain expert time)
- Inconsistent (different operators label differently)
- Incomplete (rare states not labeled)
- Static (don't adapt as equipment ages)
- Not scalable (100s of equipment, 1000s of sensors)

**Self-learning is**:
- Automatic (no human intervention)
- Consistent (same algorithm for all equipment)
- Complete (discovers all states, even rare ones)
- Adaptive (evolves as behavior changes)
- Scalable (works for entire fleet)

---

## Part 2: Cold-Start Strategy (Day 0-3)

### Objective

Establish initial operating condition taxonomy from minimal data.

### Minimum Data Requirements

**Absolute Minimum**: 200 observations (~3 days at 30-min cadence)
- Too few: Unstable clustering
- Too many: Delays time-to-value

**Ideal**: 500-1000 observations (~7-14 days)
- Better cluster stability
- More confident physics validation

### Algorithm: Bootstrap Clustering

```python
class ColdStartRegimeDiscovery:
    """
    Bootstrap operating mode discovery from minimal data.
    
    Strategy:
    1. Wait for minimum data threshold
    2. Extract regime features (power, temperature, vibration, flow)
    3. Apply conservative clustering (fewer clusters, high confidence)
    4. Validate against physics
    5. Mark as LEARNING state (low confidence)
    """
    
    def __init__(self, equipment_type):
        self.equipment_type = equipment_type
        self.min_observations = 200
        self.bootstrap_k_min = 2  # Conservative: at least off/on
        self.bootstrap_k_max = 4  # Conservative: don't over-cluster
        
    def can_bootstrap(self, data_count):
        """Check if enough data for bootstrap."""
        return data_count >= self.min_observations
    
    def bootstrap(self, sensor_data, timestamps):
        """
        Perform initial bootstrap clustering.
        
        Args:
            sensor_data: DataFrame with sensor columns
            timestamps: DatetimeIndex
            
        Returns:
            initial_model: RegimeModel with LEARNING maturity
        """
        print(f"[COLD-START] Starting with {len(sensor_data)} observations")
        
        # Step 1: Extract regime features
        features = self._extract_regime_features(sensor_data)
        
        # Step 2: Conservative K-Means
        # Use fewer clusters initially (avoid over-fitting sparse data)
        best_k = self._find_initial_k(features, timestamps)
        
        kmeans = MiniBatchKMeans(
            n_clusters=best_k,
            n_init=20,  # More initializations for stability
            random_state=42
        )
        labels = kmeans.fit_predict(features.values)
        
        # Step 3: Physics validation
        physics_valid = self._validate_physics_bootstrap(
            labels,
            sensor_data,
            timestamps
        )
        
        # Step 4: Generate provisional labels
        provisional_labels = self._generate_provisional_labels(
            kmeans.cluster_centers_,
            features.columns,
            sensor_data
        )
        
        # Step 5: Create LEARNING state model
        model = RegimeModel(
            kmeans=kmeans,
            feature_columns=features.columns.tolist(),
            scaler=StandardScaler().fit(features.values),
            semantic_labels=provisional_labels,
            maturity='LEARNING',  # Not yet reliable
            confidence=0.3,  # Low confidence during bootstrap
            training_observations=len(sensor_data),
            physics_validation=physics_valid,
            version=1,
        )
        
        print(f"[COLD-START] Discovered {best_k} initial operating modes:")
        for cluster_id, label in provisional_labels.items():
            count = np.sum(labels == cluster_id)
            print(f"  Mode {cluster_id}: {label} ({count} observations, {100*count/len(labels):.1f}%)")
        
        return model
    
    def _find_initial_k(self, features, timestamps):
        """
        Find initial number of clusters conservatively.
        
        Bootstrap strategy: Start with fewer clusters, expand later.
        """
        best_k = self.bootstrap_k_min
        best_score = -np.inf
        
        X = features.values
        
        for k in range(self.bootstrap_k_min, self.bootstrap_k_max + 1):
            # Fit K-Means
            kmeans = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Simple validation: silhouette + temporal stability
            if len(np.unique(labels)) < k:
                continue  # Empty cluster
            
            silhouette = silhouette_score(X, labels, sample_size=min(1000, len(X)))
            temporal_stability = self._temporal_stability(labels, timestamps)
            
            # Composite score (favor simpler models during bootstrap)
            score = 0.6 * silhouette + 0.4 * temporal_stability
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k
    
    def _temporal_stability(self, labels, timestamps):
        """
        Check temporal stability (good clusters = stable over time).
        
        Measure: Autocorrelation of cluster assignments.
        High autocorrelation = stable modes (good)
        Low autocorrelation = rapid switching (bad clustering)
        """
        # Compute autocorrelation at lag=1
        labels_shifted = np.roll(labels, 1)
        labels_shifted[0] = labels[0]  # Don't wrap around
        
        # Fraction of consecutive samples in same cluster
        stability = np.mean(labels == labels_shifted)
        
        return stability
    
    def _validate_physics_bootstrap(self, labels, sensor_data, timestamps):
        """
        Validate clusters against physics (simplified for bootstrap).
        
        Bootstrap validation:
        - Check power ordering (high power modes vs low power modes)
        - Check temperature correlation with power
        - Check that transitions make sense
        """
        validation_results = {}
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = sensor_data[cluster_mask]
            
            # Compute average power for this cluster
            if 'Power' in cluster_data.columns:
                avg_power = cluster_data['Power'].mean()
            else:
                avg_power = 0
            
            validation_results[cluster_id] = {
                'avg_power': avg_power,
                'sample_count': cluster_mask.sum(),
            }
        
        # Check power ordering makes sense (clusters should separate by load)
        power_values = [v['avg_power'] for v in validation_results.values()]
        power_std = np.std(power_values)
        
        # If clusters have different power levels, that's good
        physics_score = min(1.0, power_std / 10.0)  # Normalize
        
        return {
            'overall_score': physics_score,
            'clusters': validation_results,
        }
    
    def _generate_provisional_labels(self, centroids, feature_names, sensor_data):
        """
        Generate provisional semantic labels for bootstrap clusters.
        
        Strategy: Use simple heuristics based on power level.
        These are NOT final labels - will be refined later.
        """
        labels = {}
        
        # Sort clusters by power (if available)
        power_idx = None
        for i, name in enumerate(feature_names):
            if 'power' in name.lower():
                power_idx = i
                break
        
        if power_idx is not None:
            # Sort by power level
            power_levels = [(i, centroids[i, power_idx]) for i in range(len(centroids))]
            power_levels.sort(key=lambda x: x[1])
            
            # Assign provisional labels
            if len(power_levels) == 2:
                labels[power_levels[0][0]] = "low-power"
                labels[power_levels[1][0]] = "high-power"
            elif len(power_levels) == 3:
                labels[power_levels[0][0]] = "idle"
                labels[power_levels[1][0]] = "partial-load"
                labels[power_levels[2][0]] = "full-load"
            elif len(power_levels) == 4:
                labels[power_levels[0][0]] = "idle"
                labels[power_levels[1][0]] = "startup"
                labels[power_levels[2][0]] = "partial-load"
                labels[power_levels[3][0]] = "full-load"
        else:
            # Fallback: Just number them
            for i in range(len(centroids)):
                labels[i] = f"mode-{i}"
        
        return labels
```

### Transfer Learning (If Similar Equipment Exists)

```python
class TransferLearningBootstrap:
    """
    Bootstrap new equipment from similar equipment (1-day cold-start).
    """
    
    def find_similar_equipment(self, target_type, target_data_sample):
        """
        Find most similar equipment with mature regime model.
        
        Similarity based on:
        - Equipment type match
        - Sensor overlap (common sensors)
        - Statistical similarity (similar mean/std values)
        """
        candidates = []
        
        for equip in get_equipment_by_type(target_type):
            if not has_mature_regime_model(equip):
                continue
            
            # Compute similarity
            source_profile = load_equipment_profile(equip)
            
            # Sensor overlap
            common_sensors = set(target_data_sample.columns) & set(source_profile.sensors)
            sensor_overlap = len(common_sensors) / max(len(target_data_sample.columns), len(source_profile.sensors))
            
            # Statistical similarity (for common sensors)
            stat_similarity = 0
            if common_sensors:
                stat_diffs = []
                for sensor in common_sensors:
                    target_mean = target_data_sample[sensor].mean()
                    source_mean = source_profile.sensor_stats[sensor]['mean']
                    target_std = target_data_sample[sensor].std()
                    source_std = source_profile.sensor_stats[sensor]['std']
                    
                    # Normalized difference
                    mean_diff = abs(target_mean - source_mean) / (source_std + 1e-6)
                    stat_diffs.append(mean_diff)
                
                stat_similarity = np.exp(-np.mean(stat_diffs))
            
            # Overall similarity
            similarity = 0.6 * sensor_overlap + 0.4 * stat_similarity
            
            if similarity > 0.5:  # Minimum threshold
                candidates.append({
                    'equip_id': equip,
                    'similarity': similarity,
                    'common_sensors': common_sensors,
                })
        
        if candidates:
            # Return best match
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            return candidates[0]
        return None
    
    def transfer_regime_model(self, source_model, target_data_sample):
        """
        Adapt source regime model to target equipment.
        
        Steps:
        1. Extract feature scaling factors (target vs source)
        2. Scale centroids
        3. Adapt detector models
        4. Mark as TRANSFERRED (needs validation)
        """
        # Compute scaling factors
        scaling_factors = {}
        for sensor in source_model.feature_columns:
            if sensor in target_data_sample.columns:
                source_scale = source_model.sensor_scales.get(sensor, 1.0)
                target_scale = target_data_sample[sensor].std()
                scaling_factors[sensor] = target_scale / (source_scale + 1e-6)
            else:
                scaling_factors[sensor] = 1.0
        
        # Scale centroids
        adapted_centroids = source_model.kmeans.cluster_centers_.copy()
        for i, sensor in enumerate(source_model.feature_columns):
            adapted_centroids[:, i] *= scaling_factors[sensor]
        
        # Create adapted model
        adapted_kmeans = MiniBatchKMeans(n_clusters=len(adapted_centroids))
        adapted_kmeans.cluster_centers_ = adapted_centroids
        adapted_kmeans.n_clusters = len(adapted_centroids)
        
        transferred_model = RegimeModel(
            kmeans=adapted_kmeans,
            feature_columns=source_model.feature_columns,
            scaler=source_model.scaler,  # Use source scaler initially
            semantic_labels=source_model.semantic_labels.copy(),
            maturity='TRANSFERRED',  # Needs validation on target data
            confidence=0.5,  # Medium confidence (better than bootstrap)
            transferred_from=source_model.equip_id,
            transfer_similarity=source_model.similarity,
            version=1,
        )
        
        print(f"[TRANSFER] Transferred regime model from Equipment {source_model.equip_id}")
        print(f"  Similarity: {source_model.similarity:.2f}")
        print(f"  Modes: {list(transferred_model.semantic_labels.values())}")
        
        return transferred_model
```

---

## Part 3: Learning Phase (Day 3-14)

### Objective

Refine initial clusters, increase confidence, prepare for promotion to CONVERGED.

### Continuous Refinement Strategy

```python
class LearningPhaseRefinement:
    """
    Gradually refine regime model during LEARNING phase.
    
    Strategy:
    1. Accumulate more data (increase from 200 to 1000+ observations)
    2. Re-cluster periodically (every 3-7 days)
    3. Validate improvements (silhouette, physics, temporal)
    4. Update semantic labels based on richer data
    5. Track stability (are clusters converging?)
    """
    
    def __init__(self):
        self.refinement_interval_days = 3
        self.min_improvement_threshold = 0.05  # 5% improvement to accept
        
    def should_refine(self, model, current_data_count):
        """
        Decide if model should be refined.
        
        Criteria:
        - At least 3 days since last refinement
        - Data count increased significantly (>20%)
        - Model still in LEARNING state
        """
        if model.maturity != 'LEARNING':
            return False
        
        days_since_training = (datetime.now() - model.last_trained).days
        data_growth = current_data_count / model.training_observations
        
        return (days_since_training >= self.refinement_interval_days and
                data_growth >= 1.2)
    
    def refine_model(self, current_model, new_data, timestamps):
        """
        Refine regime model with additional data.
        
        Strategy: Re-cluster with more data, compare to current model.
        Accept refinement only if significant improvement.
        """
        print(f"[LEARNING] Refining model (current: {current_model.n_clusters} clusters, "
              f"{current_model.training_observations} obs)")
        
        # Extract features from new data
        features = extract_regime_features(new_data)
        
        # Try different k values around current k
        current_k = current_model.n_clusters
        k_candidates = range(max(2, current_k - 1), current_k + 2)
        
        best_model = None
        best_score = current_model.quality_score
        
        for k in k_candidates:
            # Fit new model
            kmeans = MiniBatchKMeans(n_clusters=k, n_init=20, random_state=None)
            labels = kmeans.fit_predict(features.values)
            
            # Evaluate quality
            quality = self._evaluate_quality(
                labels,
                features,
                timestamps,
                new_data
            )
            
            if quality > best_score + self.min_improvement_threshold:
                best_score = quality
                best_model = {
                    'kmeans': kmeans,
                    'labels': labels,
                    'quality': quality,
                    'k': k,
                }
        
        if best_model is not None:
            # Accept refinement
            print(f"[LEARNING] Accepting refinement: k={current_k}→{best_model['k']}, "
                  f"quality={current_model.quality_score:.3f}→{best_model['quality']:.3f}")
            
            # Update model
            refined = self._create_refined_model(
                best_model,
                features.columns,
                new_data,
                current_model
            )
            
            return refined
        else:
            # No improvement - keep current model
            print(f"[LEARNING] No improvement found, keeping current model")
            return current_model
    
    def _evaluate_quality(self, labels, features, timestamps, sensor_data):
        """
        Composite quality score for regime model.
        
        Components:
        - Silhouette score (cluster separation)
        - Temporal stability (autocorrelation)
        - Physics validation (causal relationships)
        - Coverage (all modes represented)
        """
        X = features.values
        
        # Silhouette
        silhouette = silhouette_score(X, labels, sample_size=min(2000, len(X)))
        
        # Temporal stability
        temporal = self._temporal_stability(labels)
        
        # Physics validation
        physics = self._physics_validation_score(labels, sensor_data)
        
        # Coverage (no mode < 2% of data)
        unique, counts = np.unique(labels, return_counts=True)
        min_representation = counts.min() / len(labels)
        coverage = min(1.0, min_representation / 0.02)  # Penalty if any mode <2%
        
        # Composite
        quality = (
            0.35 * silhouette +
            0.25 * temporal +
            0.25 * physics +
            0.15 * coverage
        )
        
        return quality
```

### Gradual Semantic Label Improvement

```python
class SemanticLabelEvolution:
    """
    Improve semantic labels as more data becomes available.
    
    Evolution:
    Day 0-3: Generic labels ("low-power", "high-power")
    Day 3-7: Refined labels ("idle", "partial-load", "full-load")
    Day 7-14: Precise labels ("startup", "steady-state", "shutdown")
    Day 14+: Fleet-aligned labels (consistent across equipment)
    """
    
    def evolve_labels(self, model, data_days, equipment_type):
        """
        Evolve labels based on maturity and data richness.
        """
        if data_days < 3:
            # Bootstrap labels (simple)
            return self._bootstrap_labels(model)
        elif data_days < 7:
            # Refined labels (physics-based)
            return self._refined_labels(model, equipment_type)
        elif data_days < 14:
            # Precise labels (temporal patterns)
            return self._precise_labels(model, equipment_type)
        else:
            # Fleet-aligned labels
            return self._fleet_aligned_labels(model, equipment_type)
    
    def _refined_labels(self, model, equipment_type):
        """
        Physics-based label refinement.
        
        Use sensor signatures + temporal patterns to name clusters.
        """
        labels = {}
        centroids = model.kmeans.cluster_centers_
        
        # Load equipment-specific signature templates
        templates = load_signature_templates(equipment_type)
        
        for cluster_id in range(len(centroids)):
            centroid = centroids[cluster_id]
            
            # Extract sensor signature from centroid
            signature = extract_signature_from_centroid(
                centroid,
                model.feature_columns
            )
            
            # Match to templates
            best_match = None
            best_score = 0
            
            for template_name, template_sig in templates.items():
                score = match_signature(signature, template_sig)
                if score > best_score:
                    best_score = score
                    best_match = template_name
            
            if best_match and best_score > 0.6:
                labels[cluster_id] = best_match
            else:
                # Fallback to descriptive label
                labels[cluster_id] = generate_descriptive_label(signature)
        
        return labels
```

---

## Part 4: Converged State (Day 14+)

### Promotion Criteria

```python
class PromotionValidator:
    """
    Validate regime model ready for CONVERGED state.
    
    Criteria (all must pass):
    1. Sufficient training data (>=1000 observations, >=7 days)
    2. High quality score (silhouette >=0.15, physics >=0.6)
    3. Temporal stability (autocorrelation >=0.8)
    4. Consecutive successful runs (>=3 runs without degradation)
    5. Fleet validation (if fleet consensus available)
    """
    
    def check_promotion(self, model, run_history):
        """
        Check if model can be promoted to CONVERGED.
        
        Returns:
            eligible: bool
            unmet_criteria: List[str]
        """
        unmet = []
        
        # Criterion 1: Data volume
        if model.training_observations < 1000:
            unmet.append(f"Insufficient observations: {model.training_observations} < 1000")
        if model.training_days < 7:
            unmet.append(f"Insufficient days: {model.training_days} < 7")
        
        # Criterion 2: Quality
        if model.silhouette_score < 0.15:
            unmet.append(f"Low silhouette: {model.silhouette_score:.3f} < 0.15")
        if model.physics_score < 0.6:
            unmet.append(f"Low physics validation: {model.physics_score:.3f} < 0.6")
        
        # Criterion 3: Stability
        if model.temporal_stability < 0.8:
            unmet.append(f"Low temporal stability: {model.temporal_stability:.3f} < 0.8")
        
        # Criterion 4: Consecutive runs
        recent_runs = run_history[-3:]
        if len(recent_runs) < 3:
            unmet.append(f"Insufficient run history: {len(recent_runs)} < 3")
        elif not all(r['success'] and r['quality_ok'] for r in recent_runs):
            unmet.append("Not all recent runs successful")
        
        # Criterion 5: Fleet validation (if applicable)
        if has_fleet_consensus(model.equipment_type):
            fleet_alignment = check_fleet_alignment(model)
            if fleet_alignment < 0.7:
                unmet.append(f"Poor fleet alignment: {fleet_alignment:.3f} < 0.7")
        
        eligible = len(unmet) == 0
        return eligible, unmet
```

---

## Part 5: Continuous Evolution (Mature Operation)

### Handling New Operating Conditions

```python
class NewModeDetector:
    """
    Detect when equipment enters previously unseen operating condition.
    
    Strategy:
    - Monitor assignment confidence
    - Detect patterns of low-confidence UNKNOWN assignments
    - Propose new cluster if pattern persists
    - Validate before adding to taxonomy
    """
    
    def detect_new_mode(self, recent_assignments, confidence_history):
        """
        Detect if equipment consistently operating in UNKNOWN state.
        
        Criteria for new mode:
        - >5% of recent assignments are UNKNOWN
        - UNKNOWN assignments cluster together (not random)
        - Pattern persists for >3 days
        - Physics validates this is a real mode (not noise)
        """
        unknown_rate = np.mean(recent_assignments == UNKNOWN_REGIME_LABEL)
        
        if unknown_rate > 0.05:  # >5% unknown
            # Check if UNKNOWNs cluster together
            unknown_indices = np.where(recent_assignments == UNKNOWN_REGIME_LABEL)[0]
            
            # Are they temporally clustered? (consecutive UNKNOWNs)
            consecutive_runs = self._find_consecutive_runs(unknown_indices)
            long_runs = [r for r in consecutive_runs if len(r) > 10]  # >10 consecutive
            
            if long_runs:
                print(f"[EVOLUTION] Detected potential new operating mode!")
                print(f"  UNKNOWN rate: {unknown_rate:.1%}")
                print(f"  Long UNKNOWN runs: {len(long_runs)}")
                
                return True, unknown_indices
        
        return False, None
    
    def propose_new_cluster(self, model, unknown_data, unknown_timestamps):
        """
        Propose adding a new cluster for persistent UNKNOWN pattern.
        
        Steps:
        1. Extract features from UNKNOWN observations
        2. Check if they form coherent cluster (high density)
        3. Validate against physics
        4. Propose new cluster centroid
        5. Generate semantic label
        6. Return augmented model (requires approval)
        """
        # Extract features
        unknown_features = extract_regime_features(unknown_data)
        
        # Check coherence (do they cluster?)
        dbscan = DBSCAN(eps=0.5, min_samples=20)
        unknown_labels = dbscan.fit_predict(unknown_features.values)
        
        coherent = np.sum(unknown_labels != -1) / len(unknown_labels) > 0.8
        
        if not coherent:
            print("[EVOLUTION] UNKNOWN observations too scattered - not a coherent mode")
            return None
        
        # Compute new centroid
        new_centroid = unknown_features.values[unknown_labels != -1].mean(axis=0)
        
        # Validate physics
        physics_valid = validate_new_mode_physics(
            unknown_data,
            unknown_timestamps,
            model.equipment_type
        )
        
        if not physics_valid:
            print("[EVOLUTION] New mode fails physics validation - likely noise")
            return None
        
        # Generate semantic label
        new_label = generate_label_for_new_mode(
            new_centroid,
            unknown_features.columns,
            model.semantic_labels
        )
        
        print(f"[EVOLUTION] Proposing new mode: '{new_label}'")
        print(f"  Observations: {np.sum(unknown_labels != -1)}")
        print(f"  Coherence: {coherent}")
        
        # Create augmented model
        augmented_model = add_cluster_to_model(
            model,
            new_centroid,
            new_label,
            maturity='LEARNING'  # New cluster needs validation
        )
        
        return augmented_model
```

### Adapting to Equipment Aging

```python
class AgingAdaptation:
    """
    Adapt regime model as equipment ages and behavior changes.
    
    Aging effects:
    - Baseline shifts (higher vibration, temperature)
    - Regime boundaries shift (full-load achieved at lower power)
    - New transient states emerge (degraded startup)
    """
    
    def detect_aging_drift(self, model, recent_data, historical_baseline):
        """
        Detect if regime boundaries have shifted due to aging.
        
        Method: Compare recent centroid positions to historical baseline.
        """
        recent_features = extract_regime_features(recent_data)
        recent_centroids = compute_cluster_centroids(recent_features, model)
        
        # Compare to historical centroids
        centroid_drift = []
        for i in range(len(model.kmeans.cluster_centers_)):
            historical = model.kmeans.cluster_centers_[i]
            recent = recent_centroids[i]
            
            drift = np.linalg.norm(historical - recent)
            centroid_drift.append(drift)
        
        max_drift = max(centroid_drift)
        
        if max_drift > 0.5:  # Significant drift
            print(f"[AGING] Detected regime drift: max={max_drift:.3f}")
            return True, centroid_drift
        
        return False, centroid_drift
    
    def adapt_centroids(self, model, recent_data, drift_amount):
        """
        Gradually adapt centroids to account for aging.
        
        Strategy: Exponential moving average
        - New centroids = 0.9 * old + 0.1 * recent
        - Gradual adaptation prevents sudden changes
        """
        recent_features = extract_regime_features(recent_data)
        recent_centroids = compute_cluster_centroids(recent_features, model)
        
        adapted_centroids = (
            0.9 * model.kmeans.cluster_centers_ +
            0.1 * recent_centroids
        )
        
        # Update model
        model.kmeans.cluster_centers_ = adapted_centroids
        model.version += 1
        model.last_adapted = datetime.now()
        
        print(f"[AGING] Adapted regime centroids (version {model.version})")
        
        return model
```

---

## Part 6: Fleet Learning & Consistency

### Fleet-Wide Regime Alignment

```python
class FleetRegimeAligner:
    """
    Align regimes across equipment fleet for consistency.
    
    Goal: Cluster 0 on Equipment A = Cluster 0 on Equipment B (same semantic meaning)
    """
    
    def build_fleet_consensus(self, equipment_type):
        """
        Build consensus regime definitions for equipment type.
        
        Steps:
        1. Collect all regime models for equipment type
        2. Extract centroids from each model
        3. Cluster centroids to find consensus modes
        4. Map each equipment's local regimes to consensus
        5. Update semantic labels fleet-wide
        """
        # Get all equipment of this type
        equipment_list = get_equipment_by_type(equipment_type)
        
        # Collect centroids
        all_centroids = []
        equipment_centroid_map = {}
        
        for equip_id in equipment_list:
            model = load_regime_model(equip_id)
            if model and model.maturity in ['CONVERGED', 'LEARNING']:
                all_centroids.append(model.kmeans.cluster_centers_)
                equipment_centroid_map[equip_id] = model
        
        if len(all_centroids) < 3:
            print(f"[FLEET] Insufficient equipment for consensus ({len(all_centroids)} < 3)")
            return None
        
        # Stack all centroids
        stacked = np.vstack(all_centroids)
        
        # Cluster to find consensus regimes
        n_consensus = int(np.median([len(c) for c in all_centroids]))
        
        consensus_kmeans = KMeans(n_clusters=n_consensus, random_state=42)
        consensus_kmeans.fit(stacked)
        
        # Generate semantic labels for consensus regimes
        consensus_labels = generate_consensus_labels(
            consensus_kmeans.cluster_centers_,
            equipment_type
        )
        
        print(f"[FLEET] Built consensus with {n_consensus} modes:")
        for i, label in consensus_labels.items():
            print(f"  Consensus Mode {i}: {label}")
        
        # Map each equipment to consensus
        mappings = {}
        for equip_id, model in equipment_centroid_map.items():
            mapping = self._map_to_consensus(
                model.kmeans.cluster_centers_,
                consensus_kmeans.cluster_centers_,
                consensus_labels
            )
            mappings[equip_id] = mapping
        
        return {
            'consensus_centroids': consensus_kmeans.cluster_centers_,
            'consensus_labels': consensus_labels,
            'equipment_mappings': mappings,
        }
    
    def _map_to_consensus(self, local_centroids, consensus_centroids, consensus_labels):
        """
        Map equipment's local regime IDs to consensus IDs.
        
        Returns: {local_id: {'consensus_id': int, 'consensus_label': str}}
        """
        mapping = {}
        
        for local_id in range(len(local_centroids)):
            local_centroid = local_centroids[local_id]
            
            # Find closest consensus centroid
            distances = [
                np.linalg.norm(local_centroid - consensus_centroid)
                for consensus_centroid in consensus_centroids
            ]
            closest_consensus_id = np.argmin(distances)
            
            mapping[local_id] = {
                'consensus_id': closest_consensus_id,
                'consensus_label': consensus_labels[closest_consensus_id],
                'distance': distances[closest_consensus_id],
            }
        
        return mapping
```

---

## Part 7: Complete Workflow Example

### Day-by-Day Evolution

```python
# Day 0: Equipment installed, no data
equipment = Equipment(id=123, type='FD_FAN')
regime_tracker = RegimeEvolutionTracker(equipment)

# Day 1-2: Collecting data (waiting for minimum)
for batch in data_stream:
    regime_tracker.ingest_batch(batch)
    if regime_tracker.data_count < 200:
        print(f"[DAY {regime_tracker.days}] Collecting data... ({regime_tracker.data_count}/200)")

# Day 3: Bootstrap clustering
if regime_tracker.can_bootstrap():
    print(f"[DAY 3] Starting bootstrap clustering")
    
    # Check for similar equipment (transfer learning)
    transfer_finder = TransferLearningBootstrap()
    similar = transfer_finder.find_similar_equipment(
        equipment.type,
        regime_tracker.get_data_sample()
    )
    
    if similar:
        # Transfer learning (1-day cold-start)
        print(f"[DAY 3] Found similar equipment {similar['equip_id']} (similarity: {similar['similarity']:.2f})")
        regime_model = transfer_finder.transfer_regime_model(
            load_regime_model(similar['equip_id']),
            regime_tracker.get_all_data()
        )
    else:
        # Bootstrap from scratch
        print(f"[DAY 3] No similar equipment - bootstrapping from scratch")
        bootstrap = ColdStartRegimeDiscovery(equipment.type)
        regime_model = bootstrap.bootstrap(
            regime_tracker.get_all_data(),
            regime_tracker.get_timestamps()
        )
    
    regime_tracker.set_model(regime_model)

# Day 4-14: Learning phase (refinement)
learning_refiner = LearningPhaseRefinement()

for day in range(4, 15):
    # Ingest daily data
    daily_batch = get_daily_data(equipment, day)
    regime_tracker.ingest_batch(daily_batch)
    
    # Check if refinement needed
    if learning_refiner.should_refine(regime_model, regime_tracker.data_count):
        print(f"[DAY {day}] Refining regime model")
        regime_model = learning_refiner.refine_model(
            regime_model,
            regime_tracker.get_all_data(),
            regime_tracker.get_timestamps()
        )
        regime_tracker.set_model(regime_model)
    
    # Evolve semantic labels
    label_evolver = SemanticLabelEvolution()
    updated_labels = label_evolver.evolve_labels(
        regime_model,
        regime_tracker.days,
        equipment.type
    )
    regime_model.semantic_labels = updated_labels

# Day 14: Check promotion to CONVERGED
print(f"[DAY 14] Checking promotion eligibility")
promotion_validator = PromotionValidator()

eligible, unmet = promotion_validator.check_promotion(
    regime_model,
    regime_tracker.run_history
)

if eligible:
    print("[DAY 14] ✅ Model promoted to CONVERGED state")
    regime_model.maturity = 'CONVERGED'
    regime_model.confidence = 0.95
else:
    print(f"[DAY 14] ⚠️ Not yet eligible for promotion:")
    for criterion in unmet:
        print(f"  - {criterion}")

# Day 15+: Mature operation
new_mode_detector = NewModeDetector()
aging_adapter = AgingAdaptation()

for day in range(15, 100):
    daily_batch = get_daily_data(equipment, day)
    regime_tracker.ingest_batch(daily_batch)
    
    # Assign regimes
    assignments, confidences = regime_model.predict(daily_batch)
    
    # Check for new modes
    has_new_mode, unknown_indices = new_mode_detector.detect_new_mode(
        assignments,
        confidences
    )
    
    if has_new_mode:
        print(f"[DAY {day}] Detected potential new operating mode")
        unknown_data = daily_batch.iloc[unknown_indices]
        unknown_times = regime_tracker.get_timestamps()[unknown_indices]
        
        augmented_model = new_mode_detector.propose_new_cluster(
            regime_model,
            unknown_data,
            unknown_times
        )
        
        if augmented_model:
            print(f"[DAY {day}] Added new mode: {augmented_model.new_mode_label}")
            regime_model = augmented_model
    
    # Check for aging drift (monthly)
    if day % 30 == 0:
        has_drift, drift_amounts = aging_adapter.detect_aging_drift(
            regime_model,
            regime_tracker.get_recent_data(days=30),
            regime_tracker.get_historical_baseline()
        )
        
        if has_drift:
            print(f"[DAY {day}] Adapting to equipment aging")
            regime_model = aging_adapter.adapt_centroids(
                regime_model,
                regime_tracker.get_recent_data(days=30),
                drift_amounts
            )

# Day 100: Fleet alignment
if regime_tracker.days >= 100:
    print("[DAY 100] Participating in fleet-wide regime alignment")
    fleet_aligner = FleetRegimeAligner()
    
    fleet_consensus = fleet_aligner.build_fleet_consensus(equipment.type)
    
    if fleet_consensus:
        # Update local model to use consensus labels
        local_mapping = fleet_consensus['equipment_mappings'][equipment.id]
        
        print("[DAY 100] Aligned to fleet consensus:")
        for local_id, mapping in local_mapping.items():
            local_label = regime_model.semantic_labels[local_id]
            consensus_label = mapping['consensus_label']
            print(f"  Local Mode {local_id} ('{local_label}') → Consensus '{consensus_label}'")
        
        # Update semantic labels
        regime_model.semantic_labels = {
            local_id: mapping['consensus_label']
            for local_id, mapping in local_mapping.items()
        }
```

---

## Summary

Operating condition identification in ACM follows a **maturity progression**:

1. **Day 0-3 (Cold-Start)**: Bootstrap with minimal data or transfer from similar equipment
2. **Day 3-14 (Learning)**: Refine clusters, improve labels, validate physics
3. **Day 14+ (Converged)**: Stable operation with high confidence
4. **Day 30+ (Evolution)**: Detect new modes, adapt to aging, align with fleet

**Key Principles**:
- ✅ **Self-learning**: No human labels required
- ✅ **Gradual evolution**: Improves continuously as data accumulates
- ✅ **Physics-informed**: Validates against domain knowledge
- ✅ **Fleet consistency**: Aligns across similar equipment
- ✅ **Adaptive**: Handles new modes and equipment aging

**Result**: Equipment operating conditions emerge naturally from data, validated by physics, and stabilized through consensus.

---

**End of Document**
