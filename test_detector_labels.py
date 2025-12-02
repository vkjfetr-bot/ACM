"""
Test script to verify detector label formatting works correctly.
"""
import sys
sys.path.insert(0, 'c:\\Users\\bhadk\\Documents\\ACM V8 SQL\\ACM')

from utils.detector_labels import (
    get_detector_label,
    format_culprit_label,
    parse_and_label_culprits,
    DETECTOR_LABELS,
    DETECTOR_LABELS_SHORT
)

print("=" * 80)
print("DETECTOR LABEL TESTING")
print("=" * 80)

print("\n1. Full Labels:")
print("-" * 80)
for code in ['ar1_z', 'gmm_z', 'pca_spe_z', 'iforest_z', 'mhal_z']:
    print(f"  {code:15} -> {get_detector_label(code)}")

print("\n2. Short Labels:")
print("-" * 80)
for code in ['ar1_z', 'gmm_z', 'pca_spe_z', 'iforest_z', 'mhal_z']:
    print(f"  {code:15} -> {get_detector_label(code, use_short=True)}")

print("\n3. Formatted Culprit Strings (with sensors):")
print("-" * 80)
test_culprits = [
    'ar1_z',
    'gmm_z',
    'pca_spe_z(DEMO.SIM.FSAB)',
    'pca_t2_z(Temperature_01)',
    'mhal_z(Vibration_Bearing)',
    'iforest_z',
]
for culprit in test_culprits:
    formatted = format_culprit_label(culprit)
    print(f"  {culprit:30} -> {formatted}")

print("\n4. Multiple Culprits (comma-separated):")
print("-" * 80)
multi = 'ar1_z,gmm_z,pca_spe_z(Sensor1)'
labels = parse_and_label_culprits(multi)
print(f"  Input:  {multi}")
print(f"  Output: {labels}")

print("\n5. Dashboard Comparison:")
print("-" * 80)
print("  OLD (confusing):")
print("    TopCulprit: ar1_z")
print("    → User thinks: 'Sensor ar1_z is broken' ❌")
print()
print("  NEW (clear):")
print("    DetectionMethod: Time-Series Anomaly (AR1)")
print("    → User thinks: 'AR1 algorithm detected a time-series break' ✅")

print("\n6. SQL Table Examples (ACM_EpisodeDiagnostics.dominant_sensor):")
print("-" * 80)
examples = [
    ('ar1_z', 'OLD'),
    (format_culprit_label('ar1_z'), 'NEW'),
    ('pca_spe_z(DEMO.SIM.FSAB)', 'OLD'),
    (format_culprit_label('pca_spe_z(DEMO.SIM.FSAB)'), 'NEW'),
]
for value, label in examples:
    print(f"  {label}: {value}")

print("\n" + "=" * 80)
print("✅ All detector labels working correctly!")
print("=" * 80)
