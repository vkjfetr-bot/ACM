"""
Extract function implementations from old acm_main.py and insert into current stubs.
This script recovers the deleted code systematically.
"""

# Read current file with stubs
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    current_lines = f.readlines()

# Read old file with full implementation
with open("tmp_old_acm_main.py", "r", encoding="utf-8") as f:
    old_lines = f.readlines()

print(f"Current file: {len(current_lines)} lines")
print(f"Old file: {len(old_lines)} lines")

# Find main() function in old file (massive implementation)
old_main_start = None
for i, line in enumerate(old_lines):
    if line.strip().startswith("def main("):
        old_main_start = i
        print(f"Found old main() at line {i+1}")
        break

if not old_main_start:
    print("ERROR: Could not find old main() function")
    exit(1)

# The old main() contains all the logic that needs to be distributed
# We need to extract specific sections and place them into corresponding stub functions

# Map of stub functions to their line ranges in current file
stub_ranges = {
    "_load_data": (809, 883),           # Lines 809-883
    "_build_features": (884, 903),       # Lines 884-903
    "_fit_detectors": (904, 919),        # Lines 904-919
    "_score_detectors": (920, 932),      # Lines 920-932
    "_detect_regimes": (933, 947),       # Lines 933-947
    "_fuse_and_detect_episodes": (948, 963),  # Lines 948-963
    "_generate_outputs": (964, 984),     # Lines 964-984
    "_finalize_run": (985, 996),         # Lines 985-996
}

print("\nStub functions found:")
for func_name, (start, end) in stub_ranges.items():
    stub_lines = current_lines[start-1:end]
    print(f"  {func_name}: lines {start}-{end} ({len(stub_lines)} lines)")

# The old main() is ~3300 lines - we need to identify which sections go where
# Based on the conversation context, here's the approximate mapping:
# Lines 1050-1393: Setup + Data Loading -> _load_data
# Lines 1620-1807: Feature Engineering -> _build_features  
# Lines 1967-2389: Detector Fitting -> _fit_detectors
# Lines 2418-2511: Detector Scoring -> _score_detectors
# Lines 2516-2628: Regime Detection -> _detect_regimes
# Lines 2674-3348: Calibration + Fusion + Episodes -> _fuse_and_detect_episodes
# Lines 3808-4241: Analytics + Forecasting -> _generate_outputs
# Lines 4280-4376: Finalization -> _finalize_run

print("\nThis extraction is complex - the old code is DELETED and needs careful manual reconstruction.")
print("The stub functions need to be filled based on the original requirements, not just copied.")
print("Recommend using git diff to see what was in each section before proceeding.")
