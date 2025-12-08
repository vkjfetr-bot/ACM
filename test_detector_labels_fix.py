"""
Quick validation test for detector label fixes
"""
from utils.detector_labels import get_detector_label, DETECTOR_LABELS_SQL

# Test SQL-safe labels
test_cases = [
    ('ar1_z', 'Time-Series Anomaly (AR1)'),
    ('pca_spe_z', 'Correlation Break (PCA-SPE)'),
    ('pca_t2_z', 'Multivariate Outlier (PCA-T2)'),  # Note: T2 not T²
    ('omr_z', 'Baseline Consistency (OMR)'),
    ('gmm_z', 'Density Anomaly (GMM)'),
    ('iforest_z', 'Rare State (IsolationForest)'),
    ('mhal_z', 'Multivariate Distance (Mahalanobis)'),
    ('fused_z', 'Fused Multi-Detector'),
]

print("Testing SQL-safe detector labels:")
print("=" * 60)
all_pass = True
for code, expected in test_cases:
    result = get_detector_label(code, sql_safe=True)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_pass = False
    print(f"{status} {code:15s} -> {result}")
    if result != expected:
        print(f"  Expected: {expected}")

print("=" * 60)
if all_pass:
    print("✓ All tests passed!")
else:
    print("✗ Some tests failed!")

# Verify no Unicode characters in SQL-safe labels
print("\nChecking for non-ASCII characters in SQL labels:")
print("=" * 60)
has_unicode = False
for code, label in DETECTOR_LABELS_SQL.items():
    try:
        label.encode('ascii')
        print(f"✓ {code:15s} -> ASCII-safe")
    except UnicodeEncodeError as e:
        has_unicode = True
        print(f"✗ {code:15s} -> Contains non-ASCII: {e}")

print("=" * 60)
if not has_unicode:
    print("✓ All SQL labels are ASCII-safe!")
else:
    print("✗ Some labels contain non-ASCII characters!")
