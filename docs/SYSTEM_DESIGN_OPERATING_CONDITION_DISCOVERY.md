# System Design: Operating Condition Discovery Through Self-Supervised Learning

## Executive Summary

This document defines the complete technical architecture for discovering and identifying equipment operating conditions through self-supervised learning—without human labeling. The system must learn operational modes (idle, startup, full-load, etc.) directly from sensor data patterns, validate discoveries against physics constraints, and continuously evolve as equipment behavior changes.

**Core Innovation**: Physics-informed unsupervised clustering that transforms statistical density clusters into semantically meaningful operational modes with fleet-wide consistency.

---

## 1. Analytical Backbone

### 1.1 Problem Formulation

**Input**: Time-series sensor data `S(t) = [s₁(t), s₂(t), ..., sₙ(t)]` where:
- `sᵢ(t)`: Sensor i reading at time t
- No labels, no supervision, no prior knowledge of operational modes

**Output**: Operating regime assignment `R(t) ∈ {r₁, r₂, ..., rₖ}` where:
- `rᵢ`: Semantic regime label (e.g., "idle", "startup", "full-load")
- `k`: Number of regimes (discovered, not pre-defined)
- Each regime has physical meaning and operational interpretation

**Constraints**:
1. **Semantic correctness**: Regimes must correspond to actual operational states
2. **Physics consistency**: Regime transitions must obey causal relationships
3. **Fleet alignment**: Same regime label means same operational state across equipment
4. **Temporal stability**: Regimes persist over time, not transient noise
5. **Continuous evolution**: Adapt to new modes and equipment aging

### 1.2 Multi-Layer Discovery Architecture

The system operates in three cascading layers:

```
Layer 1: Statistical Clustering
    ↓ (Candidate groupings from density structure)
Layer 2: Physics Validation  
    ↓ (Validated clusters that obey domain constraints)
Layer 3: Semantic Identification
    ↓ (Meaningful operational mode labels)
```

Each layer filters and enriches the output of the previous layer, progressively transforming statistical artifacts into operational knowledge.

---

## 2. Core Algorithms

### 2.1 Hybrid Clustering Engine

**Problem**: K-Means assumes uniform density and globular clusters. HDBSCAN handles variable density but produces unstable cluster counts.

**Solution**: Hybrid approach that combines both strengths.

#### Algorithm: HybridRegimeClustering

```python
class HybridRegimeClustering:
    """
    Combines K-Means (for stable k-selection) with HDBSCAN (for variable density).
    
    Mathematical Foundation:
    - K-Means minimizes within-cluster sum of squares: Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
    - HDBSCAN builds density-based hierarchy: clusters = f(min_samples, min_cluster_size)
    - Hybrid selects k from K-Means, refines boundaries with HDBSCAN
    """
    
    def __init__(self, k_range=(2, 8), hdbscan_min_samples=10):
        self.k_range = k_range
        self.hdbscan_min_samples = hdbscan_min_samples
        
    def discover_regimes(self, features: np.ndarray, 
                        temporal_index: pd.DatetimeIndex) -> ClusteringResult:
        """
        Discover operating regimes through hybrid clustering.
        
        Args:
            features: (N, D) array of engineered features
            temporal_index: (N,) timestamps for temporal validation
            
        Returns:
            ClusteringResult with labels, centroids, quality metrics
        """
        # Step 1: K-Means candidate exploration
        kmeans_results = []
        for k in range(self.k_range[0], self.k_range[1] + 1):
            km = KMeans(n_clusters=k, n_init=20, random_state=42)
            labels = km.fit_predict(features)
            
            # Compute composite quality score
            quality = self._compute_quality(features, labels, temporal_index)
            kmeans_results.append({
                'k': k,
                'labels': labels,
                'centroids': km.cluster_centers_,
                'quality': quality
            })
        
        # Step 2: Select best k based on composite score
        best_kmeans = max(kmeans_results, key=lambda x: x['quality']['composite'])
        k_optimal = best_kmeans['k']
        
        # Step 3: HDBSCAN refinement (variable density boundaries)
        hdb = HDBSCAN(min_samples=self.hdbscan_min_samples,
                     min_cluster_size=len(features) // (k_optimal * 3))
        hdb_labels = hdb.fit_predict(features)
        
        # Step 4: Merge HDBSCAN noise (-1) into nearest K-Means cluster
        refined_labels = self._merge_noise(
            kmeans_labels=best_kmeans['labels'],
            hdbscan_labels=hdb_labels,
            features=features,
            centroids=best_kmeans['centroids']
        )
        
        # Step 5: Recompute centroids after refinement
        final_centroids = self._compute_centroids(features, refined_labels)
        
        return ClusteringResult(
            labels=refined_labels,
            centroids=final_centroids,
            k=k_optimal,
            quality_scores=best_kmeans['quality']
        )
    
    def _compute_quality(self, features, labels, temporal_index) -> dict:
        """
        Composite quality score combining multiple criteria.
        
        Quality = 0.3*silhouette + 0.3*temporal + 0.2*transition + 0.2*stability
        
        This weighting balances:
        - Cluster separation (silhouette)
        - Temporal coherence (autocorrelation)
        - State machine validity (transition rate)
        - Long-term stability (persistence)
        """
        # Silhouette coefficient [-1, 1], higher is better
        sil_score = silhouette_score(features, labels)
        
        # Temporal autocorrelation [0, 1], measures regime persistence
        temporal_score = self._temporal_autocorrelation(labels, temporal_index)
        
        # Transition score [0, 1], penalizes excessive transitions
        transition_score = self._transition_quality(labels, temporal_index)
        
        # Stability score [0, 1], measures cluster tightness over time
        stability_score = self._temporal_stability(features, labels, temporal_index)
        
        composite = (0.3 * sil_score + 
                    0.3 * temporal_score + 
                    0.2 * transition_score + 
                    0.2 * stability_score)
        
        return {
            'silhouette': sil_score,
            'temporal': temporal_score,
            'transition': transition_score,
            'stability': stability_score,
            'composite': composite
        }
    
    def _temporal_autocorrelation(self, labels, temporal_index) -> float:
        """
        Measure regime persistence using lag-1 autocorrelation.
        
        High autocorrelation = regimes persist over time (good)
        Low autocorrelation = regimes flip rapidly (bad)
        """
        # Convert labels to time series
        ts = pd.Series(labels, index=temporal_index)
        
        # Compute autocorrelation at lag=1
        acf_values = acf(ts, nlags=1, fft=True)
        return max(0.0, acf_values[1])  # Clamp to [0, 1]
    
    def _transition_quality(self, labels, temporal_index) -> float:
        """
        Penalize excessive state transitions.
        
        Operational modes should be stable states, not rapid oscillations.
        Typical equipment: 2-5 transitions per day is reasonable.
        """
        # Count transitions
        transitions = np.sum(labels[1:] != labels[:-1])
        
        # Compute transition rate per day
        time_span_days = (temporal_index[-1] - temporal_index[0]).days + 1
        transitions_per_day = transitions / time_span_days
        
        # Ideal range: 2-5 transitions/day
        # Penalty for too few (k=1) or too many (noise)
        if transitions_per_day < 2:
            score = transitions_per_day / 2.0
        elif transitions_per_day <= 5:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (transitions_per_day - 5) / 20)
        
        return score
    
    def _temporal_stability(self, features, labels, temporal_index) -> float:
        """
        Measure cluster tightness over time.
        
        Stable regimes: centroids don't drift significantly.
        """
        # Split data into temporal chunks
        chunks = self._split_temporal_chunks(features, labels, temporal_index, n_chunks=5)
        
        # Compute centroids in each chunk
        chunk_centroids = []
        for chunk_features, chunk_labels in chunks:
            centroids = self._compute_centroids(chunk_features, chunk_labels)
            chunk_centroids.append(centroids)
        
        # Measure centroid drift (variance across chunks)
        max_drift = 0.0
        for k in range(len(chunk_centroids[0])):
            centroid_sequence = [c[k] for c in chunk_centroids]
            drift = np.std(centroid_sequence, axis=0).mean()
            max_drift = max(max_drift, drift)
        
        # Normalize drift to [0, 1] score (lower drift = higher score)
        # Typical acceptable drift: < 0.5 standard deviations
        stability = max(0.0, 1.0 - max_drift / 0.5)
        return stability
    
    def _merge_noise(self, kmeans_labels, hdbscan_labels, features, centroids):
        """
        Merge HDBSCAN noise points into nearest K-Means cluster.
        """
        merged_labels = kmeans_labels.copy()
        
        # Find noise points (HDBSCAN label = -1)
        noise_mask = (hdbscan_labels == -1)
        
        # Assign each noise point to nearest centroid
        for idx in np.where(noise_mask)[0]:
            distances = np.linalg.norm(centroids - features[idx], axis=1)
            merged_labels[idx] = np.argmin(distances)
        
        return merged_labels
    
    def _compute_centroids(self, features, labels):
        """Compute cluster centroids."""
        k = len(np.unique(labels))
        centroids = np.zeros((k, features.shape[1]))
        for i in range(k):
            mask = (labels == i)
            if mask.sum() > 0:
                centroids[i] = features[mask].mean(axis=0)
        return centroids
```

---

### 2.2 Physics Validation Layer

**Problem**: Statistical clusters may violate physical laws (e.g., high power with low temperature).

**Solution**: Multi-criteria physics validation that rejects non-physical clusters.

#### Algorithm: PhysicsValidator

```python
class PhysicsValidator:
    """
    Validates clustering results against physics constraints.
    
    Validation Criteria:
    1. Causal ordering (power → temperature with lag)
    2. Energy balance (input power ≈ output heat + work)
    3. State transition validity (finite state machine)
    4. Temporal dynamics (startup → run → shutdown sequence)
    """
    
    def __init__(self, sensor_metadata: dict):
        """
        Args:
            sensor_metadata: Dict mapping sensor names to types:
                {'Power': 'power', 'Temp': 'temperature', 'Vibration': 'vibration'}
        """
        self.sensor_metadata = sensor_metadata
        self.power_sensors = [k for k, v in sensor_metadata.items() if v == 'power']
        self.temp_sensors = [k for k, v in sensor_metadata.items() if v == 'temperature']
        
    def validate_clustering(self, 
                           data: pd.DataFrame,
                           labels: np.ndarray,
                           centroids: np.ndarray) -> ValidationResult:
        """
        Comprehensive physics validation.
        
        Returns:
            ValidationResult with pass/fail and detailed scores
        """
        results = {}
        
        # Test 1: Power ordering (idle < partial < full)
        results['power_ordering'] = self._validate_power_ordering(
            data, labels, centroids
        )
        
        # Test 2: Temperature correlation (high power → high temp)
        results['temp_correlation'] = self._validate_temperature_correlation(
            data, labels
        )
        
        # Test 3: State transition validity
        results['state_transitions'] = self._validate_state_transitions(
            labels, data.index
        )
        
        # Test 4: Energy balance
        results['energy_balance'] = self._validate_energy_balance(
            data, labels, centroids
        )
        
        # Test 5: Temporal dynamics (startup duration, shutdown duration)
        results['temporal_dynamics'] = self._validate_temporal_dynamics(
            labels, data.index
        )
        
        # Compute overall physics score (weighted average)
        overall_score = (
            0.25 * results['power_ordering']['score'] +
            0.25 * results['temp_correlation']['score'] +
            0.20 * results['state_transitions']['score'] +
            0.15 * results['energy_balance']['score'] +
            0.15 * results['temporal_dynamics']['score']
        )
        
        # Pass threshold: >= 0.6
        passed = (overall_score >= 0.6)
        
        return ValidationResult(
            passed=passed,
            overall_score=overall_score,
            criteria_scores=results
        )
    
    def _validate_power_ordering(self, data, labels, centroids) -> dict:
        """
        Validate that regimes can be ordered by power level.
        
        Operational modes should have clear power hierarchy:
        idle < startup < partial-load < full-load
        """
        if not self.power_sensors:
            return {'score': 1.0, 'reason': 'No power sensors available'}
        
        # Compute average power per cluster
        k = len(np.unique(labels))
        power_levels = np.zeros(k)
        
        for i in range(k):
            mask = (labels == i)
            cluster_data = data[mask]
            # Average across all power sensors
            power_levels[i] = cluster_data[self.power_sensors].mean().mean()
        
        # Check if power levels are distinct (not overlapping)
        power_sorted = np.sort(power_levels)
        gaps = power_sorted[1:] - power_sorted[:-1]
        min_gap = gaps.min()
        power_range = power_levels.max() - power_levels.min()
        
        # Score: gap size relative to range
        # Good: gaps > 10% of range
        # Bad: gaps < 5% of range
        if power_range > 0:
            gap_ratio = min_gap / power_range
            score = np.clip(gap_ratio / 0.1, 0.0, 1.0)
        else:
            score = 0.0
        
        return {
            'score': score,
            'power_levels': power_levels.tolist(),
            'min_gap_ratio': gap_ratio if power_range > 0 else 0.0
        }
    
    def _validate_temperature_correlation(self, data, labels) -> dict:
        """
        Validate power-temperature correlation.
        
        Physical law: Higher power → higher temperature (with lag).
        """
        if not self.power_sensors or not self.temp_sensors:
            return {'score': 1.0, 'reason': 'Missing sensors'}
        
        # Compute cross-correlation between power and temp
        power_signal = data[self.power_sensors].mean(axis=1)
        temp_signal = data[self.temp_sensors].mean(axis=1)
        
        # Test lags from 0 to 30 minutes (assuming ~1 min sampling)
        max_lag = 30
        correlations = []
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(power_signal, temp_signal)[0, 1]
            else:
                corr = np.corrcoef(power_signal[:-lag], temp_signal[lag:])[0, 1]
            correlations.append(corr)
        
        # Best correlation across lags
        max_corr = max(correlations)
        best_lag = np.argmax(correlations)
        
        # Score: correlation strength (expect > 0.5)
        score = np.clip(max_corr / 0.5, 0.0, 1.0)
        
        return {
            'score': score,
            'max_correlation': max_corr,
            'optimal_lag_minutes': best_lag
        }
    
    def _validate_state_transitions(self, labels, temporal_index) -> dict:
        """
        Validate state transition logic.
        
        Physical constraint: Some transitions are impossible.
        Example: idle → full-load (must go through startup)
        """
        # Build transition matrix
        k = len(np.unique(labels))
        transition_matrix = np.zeros((k, k))
        
        for i in range(len(labels) - 1):
            from_state = labels[i]
            to_state = labels[i + 1]
            transition_matrix[from_state, to_state] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transition_matrix, row_sums, 
                                    where=row_sums > 0)
        
        # Check for forbidden transitions (hardcoded heuristic)
        # If we have 3+ states, extreme transitions are suspicious
        # (e.g., lowest power → highest power without intermediate)
        if k >= 3:
            # Find lowest and highest power states (proxy: diagonal dominance)
            self_transition_rates = np.diag(transition_probs)
            # States with high self-transition are "stable" (idle, full-load)
            # States with low self-transition are "transient" (startup, shutdown)
            
            # Heuristic: At least 50% of transition probability should be
            # to adjacent states (not jumping across)
            score = self_transition_rates.mean()
        else:
            # With only 2 states, any transition is valid
            score = 1.0
        
        return {
            'score': score,
            'transition_matrix': transition_matrix.tolist(),
            'self_transition_rate': self_transition_rates.mean() if k >= 3 else 1.0
        }
    
    def _validate_energy_balance(self, data, labels, centroids) -> dict:
        """
        Validate energy balance: Input power ≈ Output (heat + work).
        
        Simplified: Check that temperature rise is proportional to power.
        """
        if not self.power_sensors or not self.temp_sensors:
            return {'score': 1.0, 'reason': 'Missing sensors'}
        
        # For each cluster, compute power input vs temperature rise
        k = len(np.unique(labels))
        energy_ratios = []
        
        for i in range(k):
            mask = (labels == i)
            cluster_data = data[mask]
            
            avg_power = cluster_data[self.power_sensors].mean().mean()
            avg_temp = cluster_data[self.temp_sensors].mean().mean()
            
            energy_ratios.append(avg_temp / (avg_power + 1e-6))
        
        # Energy ratios should be consistent across clusters
        # (same equipment, same thermal dynamics)
        energy_ratios = np.array(energy_ratios)
        cv = np.std(energy_ratios) / (np.mean(energy_ratios) + 1e-6)
        
        # Score: low coefficient of variation (< 0.3 is good)
        score = np.clip(1.0 - cv / 0.3, 0.0, 1.0)
        
        return {
            'score': score,
            'energy_ratio_cv': cv,
            'energy_ratios': energy_ratios.tolist()
        }
    
    def _validate_temporal_dynamics(self, labels, temporal_index) -> dict:
        """
        Validate temporal dynamics (startup/shutdown durations).
        
        Operational modes should have characteristic dwell times:
        - Idle: minutes to hours
        - Startup: seconds to minutes
        - Full-load: minutes to hours
        - Shutdown: seconds to minutes
        """
        # Find mode durations (consecutive same-label runs)
        run_lengths = []
        current_label = labels[0]
        current_length = 1
        
        for i in range(1, len(labels)):
            if labels[i] == current_label:
                current_length += 1
            else:
                run_lengths.append((current_label, current_length))
                current_label = labels[i]
                current_length = 1
        run_lengths.append((current_label, current_length))
        
        # Compute median dwell time per cluster
        k = len(np.unique(labels))
        median_dwell_times = np.zeros(k)
        
        for i in range(k):
            cluster_runs = [length for label, length in run_lengths if label == i]
            if cluster_runs:
                median_dwell_times[i] = np.median(cluster_runs)
        
        # Expect dwell times > 5 samples (avoid rapid oscillation)
        min_dwell = median_dwell_times.min()
        score = np.clip(min_dwell / 5.0, 0.0, 1.0)
        
        return {
            'score': score,
            'median_dwell_times': median_dwell_times.tolist(),
            'min_dwell_time': min_dwell
        }
```

---

### 2.3 Semantic Regime Labeling

**Problem**: Clusters are numbered (0, 1, 2) with no operational meaning.

**Solution**: Auto-generate semantic labels from cluster sensor signatures.

#### Algorithm: SemanticRegimeLabeler

```python
class SemanticRegimeLabeler:
    """
    Automatically generate meaningful labels for discovered regimes.
    
    Labeling Strategy:
    1. Power-based primary label (idle/low/medium/high)
    2. Dynamic signature (stable/transient)
    3. Temporal context (startup/running/shutdown)
    4. Fleet consensus (align with similar equipment)
    """
    
    def __init__(self, sensor_metadata: dict, equipment_type: str):
        self.sensor_metadata = sensor_metadata
        self.equipment_type = equipment_type
        self.power_sensors = [k for k, v in sensor_metadata.items() if v == 'power']
        
    def generate_labels(self,
                       data: pd.DataFrame,
                       labels: np.ndarray,
                       centroids: np.ndarray,
                       transition_matrix: np.ndarray) -> dict:
        """
        Generate semantic labels for each cluster.
        
        Returns:
            Dict mapping cluster_id → semantic_label
        """
        k = len(np.unique(labels))
        semantic_labels = {}
        
        # Step 1: Power-based ordering
        power_levels = self._compute_power_levels(data, labels, centroids)
        power_order = np.argsort(power_levels)  # Low to high
        
        # Step 2: Identify regime types
        for rank, cluster_id in enumerate(power_order):
            # Get regime characteristics
            is_transient = self._is_transient(cluster_id, transition_matrix)
            power_level = power_levels[cluster_id]
            power_percentile = rank / (k - 1) if k > 1 else 0.5
            
            # Generate label
            if is_transient:
                # Transient states: startup or shutdown
                if power_percentile < 0.5:
                    label = "shutdown"
                else:
                    label = "startup"
            else:
                # Stable states: idle or load levels
                if power_percentile < 0.2:
                    label = "idle"
                elif power_percentile < 0.4:
                    label = "low-load"
                elif power_percentile < 0.7:
                    label = "medium-load"
                else:
                    label = "full-load"
            
            semantic_labels[cluster_id] = label
        
        return semantic_labels
    
    def _compute_power_levels(self, data, labels, centroids):
        """Compute average power for each cluster."""
        k = len(np.unique(labels))
        power_levels = np.zeros(k)
        
        for i in range(k):
            mask = (labels == i)
            cluster_data = data[mask]
            power_levels[i] = cluster_data[self.power_sensors].mean().mean()
        
        return power_levels
    
    def _is_transient(self, cluster_id, transition_matrix):
        """
        Determine if cluster represents transient state.
        
        Transient states: Low self-transition probability.
        Stable states: High self-transition probability.
        """
        # Normalize transition matrix
        row_sum = transition_matrix[cluster_id].sum()
        if row_sum == 0:
            return False
        
        self_transition_prob = transition_matrix[cluster_id, cluster_id] / row_sum
        
        # Threshold: < 0.6 is transient
        return self_transition_prob < 0.6
```

---

### 2.4 Fleet Regime Alignment

**Problem**: Each equipment independently discovers regimes. "Cluster 0" on Equipment A ≠ "Cluster 0" on Equipment B.

**Solution**: Fleet-wide consensus building and label alignment.

#### Algorithm: FleetRegimeAligner

```python
class FleetRegimeAligner:
    """
    Align regime labels across equipment fleet for consistency.
    
    Goal: Same semantic label = same operational mode across all equipment.
    
    Method:
    1. Collect centroids from all equipment
    2. Cluster centroids to find consensus modes
    3. Map each equipment's local regimes to consensus
    4. Propagate consistent labels
    """
    
    def __init__(self, equipment_type: str):
        self.equipment_type = equipment_type
        
    def build_consensus(self, 
                       equipment_models: List[EquipmentModel]) -> ConsensusRegimes:
        """
        Build consensus regimes from fleet of equipment models.
        
        Args:
            equipment_models: List of trained models from similar equipment
            
        Returns:
            ConsensusRegimes with canonical centroids and labels
        """
        # Step 1: Collect all centroids and normalize
        all_centroids = []
        equipment_ids = []
        local_to_equipment = []
        
        for eq_model in equipment_models:
            # Normalize centroids to [0, 1] range (z-score then sigmoid)
            normalized = self._normalize_centroids(eq_model.centroids)
            
            for local_id, centroid in enumerate(normalized):
                all_centroids.append(centroid)
                equipment_ids.append(eq_model.equipment_id)
                local_to_equipment.append((eq_model.equipment_id, local_id))
        
        all_centroids = np.array(all_centroids)
        
        # Step 2: Cluster centroids to find consensus modes
        # Use hierarchical clustering for interpretability
        linkage_matrix = linkage(all_centroids, method='ward')
        
        # Determine optimal number of consensus clusters
        # Try different cuts and evaluate consistency
        best_k = self._find_optimal_consensus_k(
            linkage_matrix, all_centroids, equipment_ids
        )
        
        # Cut dendrogram at optimal k
        consensus_labels = fcluster(linkage_matrix, best_k, criterion='maxclust')
        
        # Step 3: Compute consensus centroids
        consensus_centroids = np.zeros((best_k, all_centroids.shape[1]))
        for i in range(1, best_k + 1):  # fcluster labels start at 1
            mask = (consensus_labels == i)
            consensus_centroids[i - 1] = all_centroids[mask].mean(axis=0)
        
        # Step 4: Generate semantic labels for consensus regimes
        consensus_semantic_labels = self._generate_consensus_labels(
            consensus_centroids, best_k
        )
        
        # Step 5: Build mapping: (equipment_id, local_id) → consensus_id
        mapping = {}
        for idx, (eq_id, local_id) in enumerate(local_to_equipment):
            consensus_id = consensus_labels[idx] - 1  # Convert to 0-indexed
            mapping[(eq_id, local_id)] = {
                'consensus_id': consensus_id,
                'consensus_label': consensus_semantic_labels[consensus_id]
            }
        
        return ConsensusRegimes(
            centroids=consensus_centroids,
            labels=consensus_semantic_labels,
            equipment_mapping=mapping,
            n_equipment=len(equipment_models)
        )
    
    def _normalize_centroids(self, centroids: np.ndarray) -> np.ndarray:
        """
        Normalize centroids to comparable scale using z-score + sigmoid.
        
        This handles different equipment having different sensor ranges.
        """
        # Z-score normalization
        mean = centroids.mean(axis=0)
        std = centroids.std(axis=0) + 1e-6
        z_scored = (centroids - mean) / std
        
        # Sigmoid to [0, 1]
        normalized = 1 / (1 + np.exp(-z_scored))
        
        return normalized
    
    def _find_optimal_consensus_k(self, linkage_matrix, centroids, equipment_ids):
        """
        Find optimal number of consensus clusters.
        
        Criteria:
        - High silhouette score
        - Each consensus cluster contains centroids from multiple equipment
        - Reasonable range (2-6 consensus modes)
        """
        k_range = range(2, min(7, len(np.unique(equipment_ids)) + 1))
        scores = []
        
        for k in k_range:
            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            
            # Silhouette score
            sil = silhouette_score(centroids, labels)
            
            # Equipment diversity: each cluster should have multiple equipment
            diversity = self._compute_cluster_diversity(labels, equipment_ids)
            
            # Composite score
            score = 0.6 * sil + 0.4 * diversity
            scores.append(score)
        
        best_k = k_range[np.argmax(scores)]
        return best_k
    
    def _compute_cluster_diversity(self, labels, equipment_ids):
        """
        Measure how many different equipment contribute to each cluster.
        
        Good: Each cluster has centroids from multiple equipment (consensus)
        Bad: Each cluster has centroids from only one equipment (no consensus)
        """
        k = len(np.unique(labels))
        equipment_ids = np.array(equipment_ids)
        
        diversities = []
        for cluster_id in range(1, k + 1):
            mask = (labels == cluster_id)
            unique_equipment = len(np.unique(equipment_ids[mask]))
            total_equipment = len(np.unique(equipment_ids))
            diversity = unique_equipment / total_equipment
            diversities.append(diversity)
        
        return np.mean(diversities)
    
    def _generate_consensus_labels(self, consensus_centroids, k):
        """
        Generate semantic labels for consensus regimes.
        
        Use power-based heuristic (assuming power is first feature).
        """
        # Assume first feature is power (or dominant feature)
        power_levels = consensus_centroids[:, 0]
        power_order = np.argsort(power_levels)
        
        labels = {}
        for rank, cluster_id in enumerate(power_order):
            power_percentile = rank / (k - 1) if k > 1 else 0.5
            
            if power_percentile < 0.2:
                labels[cluster_id] = "idle"
            elif power_percentile < 0.4:
                labels[cluster_id] = "low-load"
            elif power_percentile < 0.7:
                labels[cluster_id] = "medium-load"
            else:
                labels[cluster_id] = "full-load"
        
        return labels
```

---

## 3. Code Structure & Architecture

### 3.1 Module Organization

```
core/
├── regime_discovery/
│   ├── __init__.py
│   ├── clustering.py          # HybridRegimeClustering
│   ├── physics_validation.py  # PhysicsValidator
│   ├── semantic_labeling.py   # SemanticRegimeLabeler
│   ├── fleet_alignment.py     # FleetRegimeAligner
│   └── regime_model.py        # RegimeModel, ClusteringResult classes
│
├── feature_engineering/
│   ├── __init__.py
│   ├── physics_features.py    # Causal lags, efficiency ratios
│   ├── temporal_features.py   # Temporal patterns, FFT
│   └── feature_selector.py    # Dimensionality reduction
│
├── model_lifecycle/
│   ├── __init__.py
│   ├── maturity_tracker.py    # Track model maturity progression
│   ├── quality_monitor.py     # Monitor regime quality over time
│   └── evolution_manager.py   # Handle regime evolution (new modes, aging)
│
└── persistence/
    ├── __init__.py
    ├── regime_store.py         # Save/load regime models from SQL
    └── consensus_store.py      # Save/load fleet consensus
```

### 3.2 Core Data Structures

```python
@dataclass
class ClusteringResult:
    """Result from hybrid clustering algorithm."""
    labels: np.ndarray              # (N,) cluster assignments
    centroids: np.ndarray           # (k, D) cluster centers
    k: int                          # Number of clusters
    quality_scores: dict            # Quality metrics
    timestamp: datetime             # When clustering was performed

@dataclass
class ValidationResult:
    """Result from physics validation."""
    passed: bool                    # Overall pass/fail
    overall_score: float            # [0, 1] composite score
    criteria_scores: dict           # Individual criterion scores
    violations: List[str]           # List of failed criteria

@dataclass
class RegimeModel:
    """Complete regime discovery model for one equipment."""
    equipment_id: int
    equipment_type: str
    centroids: np.ndarray           # (k, D) normalized centroids
    labels: dict                    # cluster_id → semantic_label
    quality_scores: dict            # Quality metrics
    physics_validation: ValidationResult
    feature_names: List[str]        # Feature column names
    maturity_state: str             # LEARNING, CONVERGED, DEPRECATED
    version: int                    # Model version number
    created_at: datetime
    updated_at: datetime

@dataclass
class ConsensusRegimes:
    """Fleet-wide consensus regimes."""
    centroids: np.ndarray           # (k_consensus, D) consensus centroids
    labels: dict                    # consensus_id → semantic_label
    equipment_mapping: dict         # (eq_id, local_id) → consensus info
    n_equipment: int                # Number of equipment in consensus
    created_at: datetime
```

### 3.3 Integration with ACM Main Pipeline

```python
# In core/acm_main.py

def run_acm_pipeline(equipment_id, mode, start_time, end_time):
    """
    Main ACM pipeline with integrated regime discovery.
    """
    # ... (existing data loading) ...
    
    # Regime Discovery Module
    if mode in ['offline', 'auto']:
        # OFFLINE mode: Discover or refine regimes
        regime_model = discover_or_refine_regimes(
            equipment_id=equipment_id,
            data=sensor_data,
            timestamps=timestamps,
            mode=mode
        )
    else:
        # ONLINE mode: Load frozen regime model
        regime_model = load_regime_model(equipment_id)
        if regime_model is None:
            raise RuntimeError(f"ONLINE mode requires existing regime model for equipment {equipment_id}")
    
    # Assign regime labels to all observations
    regime_labels, regime_confidence = assign_regimes(
        data=sensor_data,
        regime_model=regime_model,
        return_confidence=True
    )
    
    # ... (continue with anomaly detection, forecasting) ...


def discover_or_refine_regimes(equipment_id, data, timestamps, mode):
    """
    Discover new regime model or refine existing one.
    """
    from core.regime_discovery import (
        HybridRegimeClustering,
        PhysicsValidator,
        SemanticRegimeLabeler,
        FleetRegimeAligner
    )
    from core.feature_engineering import PhysicsFeatureEngineer
    
    # Load existing model if available
    existing_model = load_regime_model(equipment_id)
    
    # Should we refine?
    should_refine = (
        existing_model is not None and
        existing_model.maturity_state == 'LEARNING' and
        should_refine_model(existing_model, len(data))
    )
    
    if not should_refine and existing_model is not None:
        # Use existing model without refinement
        return existing_model
    
    # Feature engineering
    feature_engineer = PhysicsFeatureEngineer(
        sensor_metadata=get_sensor_metadata(equipment_id)
    )
    features = feature_engineer.transform(data)
    
    # Hybrid clustering
    clusterer = HybridRegimeClustering(k_range=(2, 6))
    clustering_result = clusterer.discover_regimes(
        features=features,
        temporal_index=timestamps
    )
    
    # Physics validation
    validator = PhysicsValidator(
        sensor_metadata=get_sensor_metadata(equipment_id)
    )
    validation_result = validator.validate_clustering(
        data=data,
        labels=clustering_result.labels,
        centroids=clustering_result.centroids
    )
    
    if not validation_result.passed:
        Console.warn(f"Physics validation failed (score={validation_result.overall_score:.2f})")
        # Fall back to simpler model or keep existing
        if existing_model is not None:
            return existing_model
        # Otherwise accept with warning
    
    # Semantic labeling
    labeler = SemanticRegimeLabeler(
        sensor_metadata=get_sensor_metadata(equipment_id),
        equipment_type=get_equipment_type(equipment_id)
    )
    
    # Build transition matrix for labeling
    transition_matrix = build_transition_matrix(clustering_result.labels)
    
    semantic_labels = labeler.generate_labels(
        data=data,
        labels=clustering_result.labels,
        centroids=clustering_result.centroids,
        transition_matrix=transition_matrix
    )
    
    # Try fleet alignment (if enough similar equipment)
    consensus = try_fleet_alignment(equipment_id, clustering_result.centroids)
    if consensus is not None:
        # Override local labels with consensus labels
        semantic_labels = apply_consensus_labels(
            equipment_id=equipment_id,
            local_labels=semantic_labels,
            consensus=consensus
        )
    
    # Build regime model
    regime_model = RegimeModel(
        equipment_id=equipment_id,
        equipment_type=get_equipment_type(equipment_id),
        centroids=clustering_result.centroids,
        labels=semantic_labels,
        quality_scores=clustering_result.quality_scores,
        physics_validation=validation_result,
        feature_names=feature_engineer.feature_names,
        maturity_state=determine_maturity_state(clustering_result, validation_result),
        version=(existing_model.version + 1 if existing_model else 1),
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Save to SQL
    save_regime_model(regime_model)
    
    return regime_model
```

---

## 4. Continuous Evolution Mechanisms

### 4.1 New Mode Detection

```python
class NewModeDetector:
    """
    Detect emergence of new operating modes not in current model.
    
    Indicators:
    - Persistent UNKNOWN label assignments (>5% for >3 days)
    - Spatially coherent UNKNOWN cluster (DBSCAN finds pattern)
    - Physics validation passes for UNKNOWN subset
    """
    
    def detect_new_mode(self, 
                       recent_assignments: np.ndarray,
                       recent_confidence: np.ndarray,
                       recent_data: pd.DataFrame,
                       current_model: RegimeModel) -> Optional[NewModeProposal]:
        """
        Analyze recent assignments for new mode emergence.
        
        Returns:
            NewModeProposal if new mode detected, None otherwise
        """
        # Check UNKNOWN rate
        unknown_rate = (recent_assignments == UNKNOWN_REGIME_LABEL).mean()
        if unknown_rate < 0.05:
            return None  # Not enough UNKNOWNs
        
        # Extract UNKNOWN subset
        unknown_mask = (recent_assignments == UNKNOWN_REGIME_LABEL)
        unknown_data = recent_data[unknown_mask]
        
        # Check temporal persistence (consecutive runs)
        consecutive_unknowns = find_consecutive_runs(
            recent_assignments, target_label=UNKNOWN_REGIME_LABEL
        )
        max_run_length = max([len(run) for run in consecutive_unknowns])
        
        if max_run_length < 10:
            return None  # No persistent pattern
        
        # Check spatial coherence (DBSCAN)
        features = extract_features(unknown_data)
        db = DBSCAN(eps=0.3, min_samples=5)
        db_labels = db.fit_predict(features)
        
        # If DBSCAN finds cluster (not all noise), new mode likely exists
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        if n_clusters == 0:
            return None  # All noise, no coherent pattern
        
        # Propose new mode
        new_centroid = features[db_labels == 0].mean(axis=0)  # Use largest cluster
        
        return NewModeProposal(
            centroid=new_centroid,
            support_count=unknown_mask.sum(),
            confidence=0.7,  # Initial confidence
            temporal_span_days=(recent_data.index[-1] - recent_data.index[0]).days
        )
```

### 4.2 Aging Adaptation

```python
class AgingAdaptationManager:
    """
    Adapt regime centroids as equipment ages.
    
    Method: Exponential moving average of centroids.
    - Fast adaptation: α=0.2 (20% new, 80% old)
    - Slow adaptation: α=0.05 (5% new, 95% old)
    """
    
    def adapt_model(self,
                   current_model: RegimeModel,
                   recent_data: pd.DataFrame,
                   recent_labels: np.ndarray) -> RegimeModel:
        """
        Adapt model centroids using recent data.
        """
        # Compute recent centroids
        features = extract_features(recent_data)
        recent_centroids = compute_centroids(features, recent_labels)
        
        # Detect drift magnitude
        max_drift = np.max(np.linalg.norm(
            recent_centroids - current_model.centroids, axis=1
        ))
        
        # Choose adaptation rate based on drift
        if max_drift > 0.5:
            alpha = 0.2  # Fast adaptation (significant drift)
        elif max_drift > 0.2:
            alpha = 0.1  # Moderate adaptation
        else:
            alpha = 0.05  # Slow adaptation (minor drift)
        
        # Exponential moving average
        adapted_centroids = (
            alpha * recent_centroids +
            (1 - alpha) * current_model.centroids
        )
        
        # Create updated model
        updated_model = dataclasses.replace(
            current_model,
            centroids=adapted_centroids,
            version=current_model.version + 1,
            updated_at=datetime.now()
        )
        
        return updated_model
```

---

## 5. Performance Characteristics

### 5.1 Computational Complexity

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Hybrid Clustering | O(N·D·k·i) | O(N·D) | N=samples, D=features, k=clusters, i=iterations |
| Physics Validation | O(N·D + k²) | O(N) | Transition matrix is k² |
| Semantic Labeling | O(k·D) | O(k) | Linear in clusters |
| Fleet Alignment | O(M·k·D·log(M·k)) | O(M·k·D) | M=equipment, hierarchical clustering |
| New Mode Detection | O(N·D·log N) | O(N) | DBSCAN dominates |

**Typical Scale**:
- N = 1,000 - 10,000 observations per run
- D = 10 - 50 features
- k = 2 - 6 regimes
- M = 5 - 100 equipment in fleet

**Total Runtime**: 1-5 seconds per equipment per run (acceptable for batch processing)

### 5.2 Quality Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Silhouette Score | >0.40 | sklearn.metrics.silhouette_score |
| Physics Validation | >0.60 | PhysicsValidator.overall_score |
| Temporal Autocorrelation | >0.80 | statsmodels.tsa.stattools.acf |
| Fleet Alignment | >0.70 | Consensus cluster diversity |
| Label Stability | <5% changes/day | Track semantic label transitions |

---

## 6. Testing Strategy

### 6.1 Unit Tests

```python
# tests/test_hybrid_clustering.py
def test_hybrid_clustering_k_selection():
    """Test that hybrid clustering selects reasonable k."""
    # Synthetic data with 3 clear clusters
    X = generate_3_cluster_data(n_per_cluster=100)
    
    clusterer = HybridRegimeClustering(k_range=(2, 5))
    result = clusterer.discover_regimes(X, temporal_index)
    
    assert result.k == 3  # Should detect 3 clusters
    assert result.quality_scores['silhouette'] > 0.5

def test_physics_validator_power_ordering():
    """Test power ordering validation."""
    # Create data with clear power hierarchy
    data = create_power_hierarchy_data()
    
    validator = PhysicsValidator(sensor_metadata={'Power': 'power'})
    validation = validator._validate_power_ordering(data, labels, centroids)
    
    assert validation['score'] > 0.8  # Should pass
```

### 6.2 Integration Tests

```python
# tests/test_regime_discovery_integration.py
def test_end_to_end_regime_discovery():
    """Test complete regime discovery pipeline."""
    # Load real equipment data
    data = load_test_equipment_data(equipment_id=1, days=14)
    
    # Run discovery
    regime_model = discover_or_refine_regimes(
        equipment_id=1,
        data=data,
        timestamps=data.index,
        mode='offline'
    )
    
    # Validate results
    assert regime_model is not None
    assert 2 <= regime_model.k <= 6
    assert regime_model.physics_validation.passed
    assert all(label in ['idle', 'low-load', 'medium-load', 'full-load', 'startup', 'shutdown']
              for label in regime_model.labels.values())
```

### 6.3 Physics Validation Tests

```python
# tests/test_physics_validation.py
def test_reject_non_physical_clustering():
    """Test that physics validation rejects invalid clusters."""
    # Create data that violates physics (high power, low temp)
    data = create_non_physical_data()
    
    validator = PhysicsValidator(sensor_metadata={
        'Power': 'power',
        'Temperature': 'temperature'
    })
    
    result = validator.validate_clustering(data, labels, centroids)
    
    assert not result.passed  # Should fail validation
    assert result.criteria_scores['temp_correlation']['score'] < 0.5
```

---

## 7. V11 Compatibility Layer

V11 uses `core/regimes.py` with K-Means only. New system is backward compatible:

```python
# core/regime_discovery/compatibility.py

def discover_regimes_v11_compatible(data, timestamps, **kwargs):
    """
    V11-compatible wrapper for new regime discovery.
    
    Returns labels and centroids in V11 format.
    """
    from core.regime_discovery import HybridRegimeClustering
    
    clusterer = HybridRegimeClustering()
    result = clusterer.discover_regimes(
        features=data,  # V11 passes pre-engineered features
        temporal_index=timestamps
    )
    
    # Return in V11 format (labels, centroids, quality)
    return result.labels, result.centroids, result.quality_scores['silhouette']
```

---

## 8. Performance Targets

### 8.1 Quality Metrics

- **Regime Correctness**: >90% of regimes pass physics validation
- **Label Consistency**: >80% agreement between human expert and auto-generated labels (on sample)
- **Fleet Alignment**: >70% of equipment in same type assigned same consensus labels
- **Temporal Stability**: <5% regime transitions per day (avoids rapid oscillation)

### 8.2 Efficiency Metrics

- **Computational Time**: <5 seconds per equipment for regime discovery
- **Memory Footprint**: <500 MB for typical equipment (10K observations, 50 features)
- **False Positive Rate**: <1% (3× improvement over V11's ~3%)
- **Cold-start Capability**: Operational with 200+ observations (vs 1000+ in V11)

---

## Summary

This system design transforms statistical clustering into physics-informed, semantically meaningful operating mode discovery through:

1. **Hybrid clustering** - Combines K-Means stability with HDBSCAN flexibility
2. **Physics validation** - Ensures discovered regimes obey domain constraints (power ordering, causality, energy balance)
3. **Semantic labeling** - Auto-generates human-interpretable regime names from sensor signatures
4. **Fleet alignment** - Hierarchical clustering of centroids ensures consistency across equipment
5. **Continuous evolution** - Detects new modes and adapts to equipment aging

The architecture provides complete pseudocode, data structures, and integration patterns ready for implementation.
