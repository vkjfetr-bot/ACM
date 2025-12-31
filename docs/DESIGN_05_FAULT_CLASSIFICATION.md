# DESIGN-05: Unsupervised Fault Classification

**Phase**: 5 of 7  
**Priority**: CRITICAL (Enables true unsupervised fault diagnosis)  
**Complexity**: High  
**Dependencies**: Phase 3 (Operating Modes), Phase 4 (Anomaly Detection)

---

## Purpose & Goals

**Build a learned taxonomy of fault types without human labels, enabling true unsupervised fault classification.**

### Current Problem (V11)

```
Episode Detected:
  Top Culprit: "Multivariate Outlier (PCA-T²)"  ← Detector name, NOT fault type
  Top Sensors: BearingTemp, Vibration
  
Operator asks: "What's wrong?"
System answer: "PCA-T² detected an outlier"  ← Meaningless to operator
```

**Reality**: Many different faults trigger PCA-T²:
- Bearing failure
- Misalignment
- Lubrication loss
- Control system fault
- Sensor calibration drift

**V11 cannot distinguish these without human labels!**

### Target Capability

```
Episode Detected:
  Fault Type: "Bearing-Degradation-Like (Cluster 3)"
  Confidence: 0.85
  Similar Historical Episodes: 12 previous cases
  Typical Progression: Gradual over 2-4 weeks
  Leading Indicators: Rising vibration + rising temperature
  Recommended Action: Schedule bearing inspection within 2 weeks
```

---

## Architecture: 4-Stage Pipeline

```
Stage 1: Fault Signature Extraction
  ↓ Multi-dimensional signature per episode
Stage 2: Signature Clustering (Unsupervised)
  ↓ Learned fault taxonomy
Stage 3: Signature Matching & Classification
  ↓ New episodes assigned to learned clusters
Stage 4: Semantic Naming & Interpretation
  ↓ Auto-generated fault type names
```

---

## Stage 1: Fault Signature Extraction

### Multi-Dimensional Signature

A fault signature captures **what makes this fault unique**:

```python
@dataclass
class FaultSignature:
    """
    Multi-dimensional representation of a fault episode.
    """
    # Detector Response Pattern (which detectors fired, how strongly)
    detector_profile: Dict[str, float]  # {'ar1': 2.3, 'pca_spe': 5.7, ...}
    
    # Sensor Pattern (which sensors deviated, in what direction)
    sensor_deviations: Dict[str, float]  # {'BearingTemp': +15°C, 'Vibration': +0.3g}
    sensor_variance_profile: Dict[str, float]  # Which sensors varied most
    
    # Temporal Pattern (how did fault evolve over time)
    onset_speed: str  # 'sudden' (<1 hour), 'gradual' (days), 'intermittent'
    duration_hours: float
    peak_time_relative: float  # When did fault peak (0=start, 1=end)
    
    # Operational Context (what was equipment doing)
    regime_at_onset: int  # Operating mode when fault started
    regime_stability: float  # Did regime change during episode?
    health_before: float  # Equipment health before fault
    
    # Frequency Domain (vibration/current harmonics)
    vibration_harmonics: np.ndarray  # FFT of vibration during episode
    dominant_frequency_hz: float
    frequency_pattern: str  # '1X', '2X', 'broadband', etc.
    
    # Physical Relationships (which physics laws violated)
    energy_balance_violation: float  # Power in vs work out discrepancy
    causal_relationship_breaks: List[str]  # ['power→temp correlation lost']
    
    # Progression Pattern (how fault developed)
    trend_direction: str  # 'worsening', 'stable', 'recovering'
    acceleration: float  # Rate of change in health decline
```

### Implementation

```python
class FaultSignatureExtractor:
    """
    Extract multi-dimensional signatures from fault episodes.
    """
    
    def extract_signature(
        self,
        episode_data: pd.DataFrame,
        episode_metadata: Dict,
        sensor_data: pd.DataFrame,
        detector_scores: pd.DataFrame,
    ) -> FaultSignature:
        """
        Extract complete fault signature from episode.
        
        Args:
            episode_data: Time series data during episode
            episode_metadata: Episode start, end, duration, etc.
            sensor_data: Raw sensor values
            detector_scores: Detector z-scores
            
        Returns:
            FaultSignature object
        """
        signature = FaultSignature()
        
        # Detector Response Pattern
        signature.detector_profile = self._extract_detector_profile(detector_scores)
        
        # Sensor Pattern
        signature.sensor_deviations = self._extract_sensor_deviations(
            sensor_data,
            episode_metadata['start_time'],
            episode_metadata['end_time']
        )
        signature.sensor_variance_profile = self._compute_sensor_variance_profile(episode_data)
        
        # Temporal Pattern
        signature.onset_speed = self._classify_onset_speed(
            detector_scores,
            episode_metadata['start_time']
        )
        signature.duration_hours = episode_metadata['duration_hours']
        signature.peak_time_relative = self._find_peak_time(detector_scores)
        
        # Operational Context
        signature.regime_at_onset = episode_metadata.get('regime', -1)
        signature.regime_stability = self._compute_regime_stability(episode_data)
        signature.health_before = episode_metadata.get('health_before', 85.0)
        
        # Frequency Domain
        if 'vibration' in sensor_data.columns:
            signature.vibration_harmonics = self._compute_fft(sensor_data['vibration'])
            signature.dominant_frequency_hz = self._find_dominant_frequency(
                signature.vibration_harmonics
            )
            signature.frequency_pattern = self._classify_frequency_pattern(
                signature.vibration_harmonics
            )
        
        # Physical Relationships
        signature.energy_balance_violation = self._check_energy_balance(sensor_data)
        signature.causal_relationship_breaks = self._check_causal_relationships(sensor_data)
        
        # Progression Pattern
        signature.trend_direction = self._classify_trend(detector_scores)
        signature.acceleration = self._compute_acceleration(detector_scores)
        
        return signature
    
    def _extract_detector_profile(self, detector_scores: pd.DataFrame) -> Dict[str, float]:
        """
        Characterize which detectors fired and how strongly.
        
        Returns normalized profile (vector sums to 1).
        """
        detector_cols = ['ar1_z', 'pca_spe_z', 'pca_t2_z', 'iforest_z', 'gmm_z', 'omr_z']
        
        # Use max z-score for each detector as signature
        max_scores = {}
        for col in detector_cols:
            if col in detector_scores.columns:
                max_scores[col] = detector_scores[col].max()
        
        # Normalize to sum to 1 (like a probability distribution)
        total = sum(max_scores.values())
        if total > 0:
            profile = {k: v / total for k, v in max_scores.items()}
        else:
            profile = {k: 1.0 / len(max_scores) for k in max_scores.keys()}
        
        return profile
    
    def _extract_sensor_deviations(
        self,
        sensor_data: pd.DataFrame,
        start_time,
        end_time
    ) -> Dict[str, float]:
        """
        Compute how each sensor deviated from baseline during episode.
        
        Returns deviation in physical units (not z-scores).
        """
        # Baseline: 1 hour before episode
        baseline_end = start_time
        baseline_start = start_time - pd.Timedelta(hours=1)
        baseline_data = sensor_data[
            (sensor_data.index >= baseline_start) &
            (sensor_data.index < baseline_end)
        ]
        
        # Episode data
        episode_sensor_data = sensor_data[
            (sensor_data.index >= start_time) &
            (sensor_data.index <= end_time)
        ]
        
        # Compute deviations
        deviations = {}
        for col in sensor_data.columns:
            if col in baseline_data.columns and col in episode_sensor_data.columns:
                baseline_mean = baseline_data[col].mean()
                episode_mean = episode_sensor_data[col].mean()
                deviation = episode_mean - baseline_mean
                deviations[col] = deviation
        
        return deviations
    
    def _classify_onset_speed(self, detector_scores, start_time) -> str:
        """
        Classify how quickly fault developed.
        
        sudden: <1 hour from normal to anomaly
        rapid: 1-6 hours
        gradual: >6 hours
        intermittent: Multiple on/off cycles
        """
        # Look at 6 hours before episode
        lookback = start_time - pd.Timedelta(hours=6)
        pre_episode = detector_scores[
            (detector_scores.index >= lookback) &
            (detector_scores.index < start_time)
        ]
        
        if len(pre_episode) == 0:
            return 'unknown'
        
        # Compute fused score over time
        fused = pre_episode.mean(axis=1)  # Simple average across detectors
        
        # Find when fused score crossed threshold (z>3)
        threshold = 3.0
        crossings = fused > threshold
        
        if crossings.sum() == 0:
            # Sudden onset (went from normal to anomaly instantly)
            return 'sudden'
        
        # Time from first crossing to episode start
        first_crossing = crossings.idxmax()
        buildup_time = (start_time - first_crossing).total_seconds() / 3600
        
        if buildup_time < 1:
            return 'sudden'
        elif buildup_time < 6:
            return 'rapid'
        else:
            return 'gradual'
    
    def _compute_fft(self, vibration_data: pd.Series) -> np.ndarray:
        """
        Compute FFT of vibration signal.
        
        Returns magnitude spectrum up to Nyquist frequency.
        """
        # Detrend
        from scipy.signal import detrend
        vibration_detrended = detrend(vibration_data.values)
        
        # FFT
        fft_result = np.fft.rfft(vibration_detrended)
        magnitude = np.abs(fft_result)
        
        # Normalize
        magnitude_normalized = magnitude / magnitude.sum()
        
        return magnitude_normalized
    
    def _classify_frequency_pattern(self, harmonics: np.ndarray) -> str:
        """
        Classify vibration frequency pattern.
        
        Patterns:
        - 1X: Dominant peak at running speed (imbalance, misalignment)
        - 2X: Peak at 2x running speed (looseness, coupling issues)
        - Harmonics: Multiple peaks at integer multiples (gear problems)
        - Broadband: Energy spread across frequencies (bearing degradation)
        """
        # Find peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(harmonics, height=0.1)
        
        if len(peaks) == 0:
            return 'no_pattern'
        
        # Check if single dominant peak
        peak_heights = properties['peak_heights']
        if len(peaks) == 1:
            return '1X'
        
        # Check if peaks are at integer multiples (harmonics)
        peak_freqs = peaks
        fundamental = peak_freqs[0]
        is_harmonic = all(
            abs(freq - fundamental * (i + 1)) < 2
            for i, freq in enumerate(peak_freqs[:5])
        )
        
        if is_harmonic:
            return 'harmonics'
        
        # Check for broadband (many small peaks)
        if len(peaks) > 10 and peak_heights.std() < 0.05:
            return 'broadband'
        
        return 'multi_peak'
```

---

## Stage 2: Signature Clustering

### Unsupervised Fault Taxonomy Learning

```python
class FaultTaxonomyLearner:
    """
    Build learned fault taxonomy from historical signatures.
    """
    
    def __init__(self, min_cluster_size=5, min_samples=3):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.fault_clusters = None
        self.cluster_prototypes = {}
        
    def build_taxonomy(self, fault_signatures: List[FaultSignature]):
        """
        Cluster fault signatures to discover fault types.
        
        Args:
            fault_signatures: List of FaultSignature objects
            
        Returns:
            fault_taxonomy: Dict mapping cluster_id → cluster_info
        """
        # Convert signatures to feature vectors
        signature_vectors = self._flatten_signatures(fault_signatures)
        
        # Cluster using HDBSCAN (handles variable density)
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',  # Excess of mass
        )
        
        cluster_labels = clusterer.fit_predict(signature_vectors)
        
        # Build cluster prototypes
        fault_taxonomy = {}
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:
                continue  # Noise cluster
            
            # Get all signatures in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_signatures = [
                sig for sig, mask in zip(fault_signatures, cluster_mask)
                if mask
            ]
            
            # Compute prototype (median signature)
            prototype = self._compute_prototype(cluster_signatures)
            
            # Auto-generate semantic name
            semantic_name = self._generate_fault_name(prototype, cluster_signatures)
            
            # Compute cluster statistics
            stats = self._compute_cluster_stats(cluster_signatures)
            
            fault_taxonomy[cluster_id] = {
                'semantic_name': semantic_name,
                'prototype': prototype,
                'count': len(cluster_signatures),
                'examples': cluster_signatures[:5],  # Keep first 5 for reference
                'stats': stats,
            }
        
        self.fault_clusters = fault_taxonomy
        return fault_taxonomy
    
    def _flatten_signatures(self, signatures: List[FaultSignature]) -> np.ndarray:
        """
        Convert FaultSignature objects to feature vectors for clustering.
        
        Features (total ~30-40 dimensions):
        - Detector profile (6 values)
        - Top 10 sensor deviations (10 values)
        - Temporal features (3 values: onset_speed_encoded, duration, peak_time)
        - Context features (3 values: regime, regime_stability, health_before)
        - Frequency features (5 values: dominant_freq, pattern_encoded, ...)
        - Physics features (2 values: energy_violation, causal_breaks_count)
        - Progression features (2 values: trend_encoded, acceleration)
        """
        vectors = []
        
        for sig in signatures:
            vector = []
            
            # Detector profile (6 values, sum to 1)
            for det in ['ar1_z', 'pca_spe_z', 'pca_t2_z', 'iforest_z', 'gmm_z', 'omr_z']:
                vector.append(sig.detector_profile.get(det, 0))
            
            # Top 10 sensor deviations (normalized by std)
            sensor_devs = sorted(
                sig.sensor_deviations.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]
            for _, dev in sensor_devs:
                vector.append(dev)
            # Pad if < 10 sensors
            while len(vector) < 16:  # 6 detectors + 10 sensors
                vector.append(0)
            
            # Temporal features
            onset_encoding = {'sudden': 0, 'rapid': 1, 'gradual': 2, 'intermittent': 3, 'unknown': 1.5}
            vector.append(onset_encoding.get(sig.onset_speed, 1.5))
            vector.append(np.log1p(sig.duration_hours))  # Log-transform duration
            vector.append(sig.peak_time_relative)
            
            # Context features
            vector.append(sig.regime_at_onset / 10.0)  # Normalize regime ID
            vector.append(sig.regime_stability)
            vector.append(sig.health_before / 100.0)  # Normalize health
            
            # Frequency features
            vector.append(sig.dominant_frequency_hz / 100.0 if sig.dominant_frequency_hz else 0)
            pattern_encoding = {
                '1X': 0, '2X': 1, 'harmonics': 2, 'broadband': 3, 'multi_peak': 4, 'no_pattern': 2.5
            }
            vector.append(pattern_encoding.get(sig.frequency_pattern, 2.5))
            
            # Physics features
            vector.append(sig.energy_balance_violation)
            vector.append(len(sig.causal_relationship_breaks))
            
            # Progression features
            trend_encoding = {'worsening': 0, 'stable': 1, 'recovering': 2}
            vector.append(trend_encoding.get(sig.trend_direction, 1))
            vector.append(sig.acceleration)
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    def _generate_fault_name(
        self,
        prototype: FaultSignature,
        cluster_examples: List[FaultSignature]
    ) -> str:
        """
        Auto-generate semantic fault name from cluster characteristics.
        
        Examples:
        - "Bearing-Degradation-Like" (high vibration broadband, gradual onset)
        - "Motor-Current-Spike-Like" (current spike, sudden onset)
        - "Control-Oscillation-Like" (intermittent, 1X frequency)
        - "Sensor-Drift-Like" (AR1 dominant, gradual, single sensor)
        """
        descriptors = []
        
        # Dominant detector
        dominant_det = max(prototype.detector_profile.items(), key=lambda x: x[1])[0]
        detector_hints = {
            'ar1_z': 'Drift',
            'pca_spe_z': 'Correlation-Loss',
            'pca_t2_z': 'Multivariate-Outlier',
            'iforest_z': 'Rare-State',
            'gmm_z': 'Distribution-Shift',
            'omr_z': 'Model-Residual',
        }
        
        # Dominant sensor pattern
        top_sensor = max(prototype.sensor_deviations.items(), key=lambda x: abs(x[1]))[0]
        
        # Infer subsystem from sensor name
        subsystem_hints = {
            'bearing': 'Bearing',
            'vibration': 'Vibration',
            'temp': 'Thermal',
            'current': 'Motor',
            'pressure': 'Hydraulic',
            'flow': 'Flow',
        }
        
        subsystem = 'Unknown'
        for keyword, name in subsystem_hints.items():
            if keyword.lower() in top_sensor.lower():
                subsystem = name
                break
        
        # Temporal pattern
        onset_descriptors = {
            'sudden': 'Acute',
            'gradual': 'Degradation',
            'rapid': 'Rapid',
            'intermittent': 'Intermittent',
        }
        temporal = onset_descriptors.get(prototype.onset_speed, '')
        
        # Frequency pattern (for vibration faults)
        freq_descriptors = {
            'broadband': 'Broadband',
            '1X': '1X-Harmonic',
            'harmonics': 'Harmonic',
        }
        freq = freq_descriptors.get(prototype.frequency_pattern, '')
        
        # Assemble name
        if subsystem == 'Bearing' and freq == 'Broadband' and temporal == 'Degradation':
            name = "Bearing-Degradation-Like"
        elif subsystem == 'Motor' and temporal == 'Acute':
            name = "Motor-Fault-Like"
        elif 'Drift' in detector_hints.get(dominant_det, ''):
            name = f"{subsystem}-Drift-Like"
        else:
            # Generic name
            parts = [subsystem, temporal, freq]
            parts = [p for p in parts if p]
            name = '-'.join(parts) + '-Like'
        
        return name
```

---

## Stage 3: Signature Matching & Classification

```python
class FaultClassifier:
    """
    Classify new fault episodes using learned taxonomy.
    """
    
    def __init__(self, fault_taxonomy: Dict):
        self.fault_taxonomy = fault_taxonomy
        self.extractor = FaultSignatureExtractor()
        
    def classify_fault(
        self,
        episode_data: pd.DataFrame,
        episode_metadata: Dict,
        sensor_data: pd.DataFrame,
        detector_scores: pd.DataFrame,
    ) -> Dict:
        """
        Classify a new fault episode.
        
        Returns:
            classification: Dict with fault_type, confidence, similar_examples
        """
        # Extract signature
        signature = self.extractor.extract_signature(
            episode_data,
            episode_metadata,
            sensor_data,
            detector_scores
        )
        
        # Match to clusters
        matches = {}
        for cluster_id, cluster_info in self.fault_taxonomy.items():
            similarity = self._compute_signature_similarity(
                signature,
                cluster_info['prototype']
            )
            matches[cluster_id] = similarity
        
        # Best match
        if matches:
            best_cluster = max(matches, key=matches.get)
            confidence = matches[best_cluster]
            
            if confidence > 0.6:
                # High confidence match
                result = {
                    'fault_type': self.fault_taxonomy[best_cluster]['semantic_name'],
                    'cluster_id': best_cluster,
                    'confidence': confidence,
                    'similar_examples': self.fault_taxonomy[best_cluster]['examples'],
                    'cluster_stats': self.fault_taxonomy[best_cluster]['stats'],
                    'is_novel': False,
                }
            else:
                # Low confidence - possibly novel fault type
                result = {
                    'fault_type': 'Novel-Fault (no close match)',
                    'cluster_id': -1,
                    'confidence': confidence,
                    'similar_examples': [],
                    'is_novel': True,
                }
        else:
            result = {
                'fault_type': 'Unclassified',
                'cluster_id': -1,
                'confidence': 0.0,
                'similar_examples': [],
                'is_novel': True,
            }
        
        return result
    
    def _compute_signature_similarity(
        self,
        sig1: FaultSignature,
        sig2: FaultSignature
    ) -> float:
        """
        Compute similarity between two fault signatures.
        
        Uses weighted cosine similarity across signature components.
        """
        # Flatten both signatures
        vec1 = self._signature_to_vector(sig1)
        vec2 = self._signature_to_vector(sig2)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
        
        return similarity
```

---

## Stage 4: Semantic Interpretation & Recommendations

```python
class FaultInterpreter:
    """
    Generate human-readable interpretation and recommendations.
    """
    
    def interpret_fault(self, classification: Dict, signature: FaultSignature) -> Dict:
        """
        Generate interpretation and recommendations for a classified fault.
        
        Returns:
            interpretation: Dict with description, leading_indicators, recommendations
        """
        fault_type = classification['fault_type']
        cluster_stats = classification.get('cluster_stats', {})
        
        # Generate description
        description = self._generate_description(fault_type, signature)
        
        # Identify leading indicators
        leading_indicators = self._identify_leading_indicators(signature)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            fault_type,
            signature,
            cluster_stats
        )
        
        return {
            'description': description,
            'leading_indicators': leading_indicators,
            'recommendations': recommendations,
            'typical_progression': cluster_stats.get('typical_progression', 'Unknown'),
            'average_duration': cluster_stats.get('average_duration_hours', 0),
        }
    
    def _generate_description(self, fault_type: str, signature: FaultSignature) -> str:
        """Generate human-readable fault description."""
        onset_desc = {
            'sudden': 'developed suddenly',
            'gradual': 'developed gradually over time',
            'rapid': 'developed rapidly',
            'intermittent': 'appeared intermittently',
        }
        
        desc = f"Fault classified as '{fault_type}', which {onset_desc.get(signature.onset_speed, 'occurred')}. "
        
        # Top affected sensors
        top_sensors = sorted(
            signature.sensor_deviations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        sensor_names = ', '.join([s[0] for s in top_sensors])
        desc += f"Primary affected sensors: {sensor_names}. "
        
        return desc
    
    def _generate_recommendations(
        self,
        fault_type: str,
        signature: FaultSignature,
        cluster_stats: Dict
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Bearing faults
        if 'bearing' in fault_type.lower():
            recommendations.append("Schedule bearing inspection within 2 weeks")
            recommendations.append("Check lubrication levels and quality")
            if signature.onset_speed == 'gradual':
                recommendations.append("Monitor vibration trends closely")
            else:
                recommendations.append("Consider immediate shutdown if vibration exceeds safety limits")
        
        # Motor faults
        elif 'motor' in fault_type.lower():
            recommendations.append("Inspect motor windings and connections")
            recommendations.append("Check motor current draw and phase balance")
            recommendations.append("Verify motor cooling system operation")
        
        # Drift faults
        elif 'drift' in fault_type.lower():
            recommendations.append("Verify sensor calibration")
            recommendations.append("Check for environmental factors affecting sensors")
            recommendations.append("Consider sensor replacement if drift persists")
        
        # Generic recommendations
        else:
            recommendations.append("Review historical similar faults for guidance")
            recommendations.append("Monitor fault progression over next 24-48 hours")
            recommendations.append("Consider expert consultation if fault persists")
        
        return recommendations
```

---

## Complete Workflow Example

```python
# Historical learning phase
taxonomy_learner = FaultTaxonomyLearner()

# Collect historical fault episodes
historical_episodes = load_historical_episodes(days=365)

# Extract signatures
signatures = []
for episode in historical_episodes:
    sig = extractor.extract_signature(
        episode.data,
        episode.metadata,
        episode.sensor_data,
        episode.detector_scores
    )
    signatures.append(sig)

# Build taxonomy (unsupervised clustering)
fault_taxonomy = taxonomy_learner.build_taxonomy(signatures)

print(f"Discovered {len(fault_taxonomy)} fault types:")
for cluster_id, info in fault_taxonomy.items():
    print(f"  {cluster_id}: {info['semantic_name']} ({info['count']} examples)")

# Online classification phase
classifier = FaultClassifier(fault_taxonomy)
interpreter = FaultInterpreter()

# New fault detected
new_episode = detect_new_episode(current_data)

# Classify
classification = classifier.classify_fault(
    new_episode.data,
    new_episode.metadata,
    new_episode.sensor_data,
    new_episode.detector_scores
)

# Interpret
interpretation = interpreter.interpret_fault(
    classification,
    new_episode.signature
)

# Report to operator
print(f"""
Fault Detected: {classification['fault_type']}
Confidence: {classification['confidence']:.2f}

Description: {interpretation['description']}

Leading Indicators:
{chr(10).join('  - ' + ind for ind in interpretation['leading_indicators'])}

Recommendations:
{chr(10).join('  - ' + rec for rec in interpretation['recommendations'])}

Similar Historical Cases: {len(classification['similar_examples'])}
""")
```

---

## Migration Path from V11

### Phase 1: Signature Extraction (2 weeks)
- Implement FaultSignatureExtractor
- Extract signatures from V11 episodes retroactively
- Validate signature quality

### Phase 2: Taxonomy Learning (2 weeks)
- Implement clustering on signatures
- Build initial taxonomy from 3-6 months of data
- Validate clusters make sense

### Phase 3: Classification (1 week)
- Implement real-time classification
- A/B test: V11 detector names vs learned fault types
- Collect operator feedback

### Phase 4: Interpretation (1 week)
- Add semantic interpretation
- Generate recommendations
- Integrate with operator interface

**Total: 6 weeks**

---

## Success Metrics

### Technical
1. **Cluster Purity**: >80% when validated against expert labels
2. **Classification Confidence**: Median confidence >0.75
3. **Novel Fault Detection**: >90% of truly novel faults flagged as low-confidence

### Business
1. **Operator Understanding**: >90% of operators understand fault type labels
2. **Actionable Recommendations**: >70% of recommendations deemed useful by operators
3. **Reduced Investigation Time**: 50% reduction in time to diagnose faults

---

**End of Design Document 05**

**Conclusion**: This design enables TRUE unsupervised fault classification by learning fault taxonomy from data, not relying on detector names. Operators get semantic fault types ("Bearing-Degradation-Like") instead of technical jargon ("PCA-T²").
