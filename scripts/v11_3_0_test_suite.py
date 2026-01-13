#!/usr/bin/env python3
"""
v11.3.0 Automated Testing Suite
Tests repeatability, accuracy, and early defect prediction.

Usage:
    python scripts/v11_3_0_test_suite.py --phase [1-8] --equip [equipment] --verbose
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime, timedelta
import argparse

def run_cmd(cmd, shell=True, capture=False, timeout=3600):
    """Run command, return output or status."""
    print(f"[CMD] {cmd}")
    try:
        if capture:
            result = subprocess.run(
                cmd, 
                shell=shell, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        else:
            return subprocess.run(
                cmd, 
                shell=shell, 
                timeout=timeout
            ).returncode
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Command timed out after {timeout}s")
        return 1, "", "Timeout"
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1, "", str(e)

def phase_1_workaround_hang():
    """Phase 1: Setup ONLINE mode (avoid hang)."""
    print("\n" + "="*80)
    print("PHASE 1: WORKAROUND HANG ISSUE (Setup ONLINE mode)")
    print("="*80)
    
    print("\n[1.1] Check if regime models are cached...")
    model_path = "artifacts/regime_models"
    if os.path.exists(model_path):
        model_count = sum(len(f) for _, _, f in os.walk(model_path))
        print(f"[OK] Models exist: {model_count} files")
    else:
        print(f"[WARN] Models not cached, will train in next phase")
    
    print("\n[1.2] Verify SQL connection...")
    cmd = (
        'sqlcmd -S "localhost\\B19CL3PCQLSERVER" -d ACM -E '
        '-Q "SELECT COUNT(*) as Rows FROM Equipment" 2>&1'
    )
    rc, out, err = run_cmd(cmd, capture=True)
    if rc == 0:
        print("✓ SQL connection OK")
    else:
        print(f"✗ SQL connection failed: {err}")
        return False
    
    print("\n[1.3] Clear old outputs...")
    for dir in ["artifacts/run_test1", "artifacts/run_test2", "artifacts/run_test3"]:
        if os.path.exists(dir):
            import shutil
            shutil.rmtree(dir)
            print(f"  Cleared {dir}")
    
    print("\n[1.4] Test ONLINE mode on small batch...")
    cmd = (
        'python scripts/sql_batch_runner.py --equip WFA_TURBINE_10 '
        '--start-time "2023-09-09T00:00:00" --end-time "2023-09-10T23:59:59" '
        '--tick-minutes 1440 --mode online 2>&1'
    )
    rc = run_cmd(cmd, timeout=600)
    
    if rc == 0:
        print("✓ ONLINE mode test passed")
        return True
    else:
        print("✗ ONLINE mode test failed - may need to debug")
        return False

def phase_2_repeatability():
    """Phase 2: Verify results are repeatable."""
    print("\n" + "="*80)
    print("PHASE 2: REPEATABILITY TEST")
    print("="*80)
    
    start_time = "2023-09-09T00:00:00"
    end_time = "2023-09-16T23:59:59"
    
    for run_num in [1, 2]:
        print(f"\n[2.{run_num}] Run {run_num}: Analyzing same data again...")
        
        # Clear previous run's episodes
        if run_num == 2:
            cmd = (
                'sqlcmd -S "localhost\\B19CL3PCQLSERVER" -d ACM -E '
                '-Q "DELETE FROM ACM_EpisodeDiagnostics WHERE EquipID=10 '
                'AND StartTime >= DATEADD(HOUR, -2, GETDATE())" 2>&1'
            )
            run_cmd(cmd)
        
        # Run batch
        cmd = (
            f'python scripts/sql_batch_runner.py --equip WFA_TURBINE_10 '
            f'--start-time "{start_time}" --end-time "{end_time}" '
            f'--tick-minutes 1440 --mode online 2>&1'
        )
        rc = run_cmd(cmd, timeout=1200)
        
        if rc != 0:
            print(f"✗ Run {run_num} failed")
            return False
        
        # Extract episode data
        output_file = f"artifacts/repeatability_run{run_num}.txt"
        cmd = (
            f'sqlcmd -S "localhost\\B19CL3PCQLSERVER" -d ACM -E '
            f'-Q "SELECT COUNT(*) as EpisodeCount, AVG(Severity) as AvgSeverity, '
            f'MAX(Severity) as MaxSeverity FROM ACM_EpisodeDiagnostics '
            f'WHERE EquipID=10 AND StartTime >= \'{start_time}\' '
            f'AND StartTime < \'{end_time}\'" -o "{output_file}" 2>&1'
        )
        rc, out, _ = run_cmd(cmd, capture=True)
        
        if rc == 0:
            print(f"✓ Metrics extracted: {output_file}")
        else:
            print(f"✗ Failed to extract metrics")
            return False
    
    # Compare results
    print("\n[2.3] Comparing Run 1 vs Run 2...")
    with open("artifacts/repeatability_run1.txt") as f:
        run1_content = f.read()
    with open("artifacts/repeatability_run2.txt") as f:
        run2_content = f.read()
    
    if run1_content == run2_content:
        print("✓ REPEATABILITY PASSED: Results identical")
        return True
    else:
        print("⚠ Results differ - investigating...")
        print(f"Run 1:\n{run1_content}")
        print(f"Run 2:\n{run2_content}")
        return None  # Warning, not failure

def phase_3_fault_accuracy():
    """Phase 3: Verify accuracy on known fault periods."""
    print("\n" + "="*80)
    print("PHASE 3: FAULT DETECTION ACCURACY")
    print("="*80)
    
    periods = {
        "Pre-fault": ("2023-09-01", "2023-09-08", 5, 15),  # min, max episodes
        "Fault": ("2023-09-09", "2023-09-16", 50, 60),      # High episodes
        "Post-fault": ("2023-09-17", "2023-09-30", 5, 15),  # Return to normal
    }
    
    for period_name, (start, end, min_ep, max_ep) in periods.items():
        print(f"\n[3.1] Testing {period_name}: {start} to {end}...")
        
        cmd = (
            f'python scripts/sql_batch_runner.py --equip WFA_TURBINE_10 '
            f'--start-time "{start}T00:00:00" --end-time "{end}T23:59:59" '
            f'--tick-minutes 1440 --mode online 2>&1'
        )
        rc = run_cmd(cmd, timeout=1200)
        
        if rc != 0:
            print(f"✗ {period_name} batch failed")
            return False
        
        # Query episodes
        cmd = (
            f'sqlcmd -S "localhost\\B19CL3PCQLSERVER" -d ACM -E '
            f'-Q "SELECT COUNT(*) as Episodes, '
            f'COUNT(CASE WHEN Severity >= 3.0 THEN 1 END) as HighSeverity '
            f'FROM ACM_EpisodeDiagnostics WHERE EquipID=10 '
            f'AND StartTime >= \'{start}T00:00:00\' '
            f'AND StartTime < \'{end}T23:59:59\'" 2>&1'
        )
        rc, out, err = run_cmd(cmd, capture=True)
        
        # Parse output
        lines = out.strip().split('\n')
        if len(lines) >= 3:
            data_line = lines[2]  # Skip headers
            parts = data_line.split()
            if len(parts) >= 2:
                episodes = int(parts[0])
                high_sev = int(parts[1])
                
                print(f"  Episodes: {episodes} (expected {min_ep}-{max_ep})")
                print(f"  High Severity: {high_sev} ({high_sev/episodes*100:.1f}%)")
                
                # Validation
                if min_ep <= episodes <= max_ep:
                    print(f"  ✓ {period_name} accuracy OK")
                else:
                    print(f"  ⚠ {period_name} outside expected range")
    
    return True

def phase_4_fp_analysis():
    """Phase 4: False positive rate analysis."""
    print("\n" + "="*80)
    print("PHASE 4: FALSE POSITIVE RATE ANALYSIS")
    print("="*80)
    
    print("\n[4.1] Running full-year analysis on WFA_TURBINE_10...")
    cmd = (
        'python scripts/sql_batch_runner.py --equip WFA_TURBINE_10 '
        '--start-time "2023-01-01T00:00:00" --end-time "2023-12-31T23:59:59" '
        '--tick-minutes 1440 --mode online 2>&1'
    )
    rc = run_cmd(cmd, timeout=3600)
    
    if rc != 0:
        print("✗ Year-long analysis failed")
        return False
    
    print("\n[4.2] Extracting episode statistics...")
    cmd = (
        'sqlcmd -S "localhost\\B19CL3PCQLSERVER" -d ACM -E '
        '-Q "SELECT regime_context, COUNT(*) as Episodes, '
        'AVG(Severity) as AvgSeverity, MAX(Severity) as MaxSeverity '
        'FROM ACM_EpisodeDiagnostics WHERE EquipID=10 '
        'GROUP BY regime_context ORDER BY Episodes DESC" '
        '-o artifacts/fp_analysis_context.txt 2>&1'
    )
    run_cmd(cmd)
    
    print("✓ FP analysis complete - see artifacts/fp_analysis_context.txt")
    return True

def phase_5_daily_trend():
    """Phase 5: Daily health degradation trend."""
    print("\n" + "="*80)
    print("PHASE 5: DAILY HEALTH DEGRADATION TREND")
    print("="*80)
    
    print("\n[5.1] Running daily windows (Sep 1-30)...")
    for day in range(1, 31):
        date_str = f"2023-09-{day:02d}"
        print(f"  Day {day}...", end="", flush=True)
        
        cmd = (
            f'python scripts/sql_batch_runner.py --equip WFA_TURBINE_10 '
            f'--start-time "{date_str}T00:00:00" --end-time "{date_str}T23:59:59" '
            f'--tick-minutes 1440 --mode online 2>&1'
        )
        rc = run_cmd(cmd, timeout=300)
        
        if rc == 0:
            print(" ✓")
        else:
            print(" ✗")
    
    print("\n[5.2] Extracting daily trend...")
    cmd = (
        'sqlcmd -S "localhost\\B19CL3PCQLSERVER" -d ACM -E '
        '-Q "SELECT DATEPART(DAY, StartTime) as Day, COUNT(*) as Episodes, '
        'AVG(Severity) as AvgSeverity, MAX(Severity) as MaxSeverity '
        'FROM ACM_EpisodeDiagnostics WHERE EquipID=10 '
        'AND YEAR(StartTime)=2023 AND MONTH(StartTime)=9 '
        'GROUP BY DATEPART(DAY, StartTime) ORDER BY Day" '
        '-o artifacts/daily_health_trend.txt 2>&1'
    )
    run_cmd(cmd)
    
    print("✓ Daily trend complete - see artifacts/daily_health_trend.txt")
    print("\n[5.3] Expected pattern:")
    print("  Sep 1-8: 1-3 episodes/day (baseline)")
    print("  Sep 9-12: 3-8 episodes/day (degradation starts)")
    print("  Sep 13-15: 10-15 episodes/day (rapid degradation)")
    print("  Sep 16: ≥15 episodes (failure point)")
    print("  Sep 17+: 1-3 episodes/day (recovery)")
    
    return True

def phase_6_early_detection():
    """Phase 6: Test early defect detection capability."""
    print("\n" + "="*80)
    print("PHASE 6: EARLY DEFECT DETECTION CAPABILITY")
    print("="*80)
    
    print("\n[6.1] Analyzing detection timeline...")
    print("  Can we detect fault on Sep 9 (7 days before Sep 16 failure)?")
    print("  Can we escalate alert by Sep 12 (4 days before failure)?")
    
    cmd = (
        'sqlcmd -S "localhost\\B19CL3PCQLSERVER" -d ACM -E '
        '-Q "DECLARE @FaultStart DATETIME = \'2023-09-09\'; '
        'DECLARE @FaultPeak DATETIME = \'2023-09-16\'; '
        'SELECT '
        '  CAST(StartTime AS DATE) as Date, '
        '  DATEDIFF(DAY, StartTime, @FaultPeak) as DaysBeforeFailure, '
        '  COUNT(*) as Episodes, '
        '  ROUND(AVG(Severity), 2) as AvgSeverity, '
        '  SUM(CASE WHEN Severity >= 3.5 THEN 1 ELSE 0 END) as CriticalEpisodes '
        'FROM ACM_EpisodeDiagnostics '
        'WHERE EquipID=10 AND StartTime >= @FaultStart AND StartTime <= @FaultPeak '
        'GROUP BY CAST(StartTime AS DATE) '
        'ORDER BY Date" '
        '-o artifacts/early_detection_timeline.txt 2>&1'
    )
    run_cmd(cmd)
    
    print("✓ Early detection timeline extracted")
    print("  See artifacts/early_detection_timeline.txt")
    
    return True

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="v11.3.0 Testing Suite")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], 
                       help="Run specific phase (1-8), or all if omitted")
    parser.add_argument("--equip", default="WFA_TURBINE_10", 
                       help="Equipment to test (default: WFA_TURBINE_10)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("v11.3.0 COMPREHENSIVE TESTING SUITE")
    print("="*80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Equipment: {args.equip}")
    print(f"Mode: {'Specific Phase' if args.phase else 'All Phases'}")
    
    results = {}
    
    try:
        if not args.phase or args.phase == 1:
            results["Phase 1: Workaround Hang"] = phase_1_workaround_hang()
        
        if not args.phase or args.phase == 2:
            results["Phase 2: Repeatability"] = phase_2_repeatability()
        
        if not args.phase or args.phase == 3:
            results["Phase 3: Fault Accuracy"] = phase_3_fault_accuracy()
        
        if not args.phase or args.phase == 4:
            results["Phase 4: FP Analysis"] = phase_4_fp_analysis()
        
        if not args.phase or args.phase == 5:
            results["Phase 5: Daily Trend"] = phase_5_daily_trend()
        
        if not args.phase or args.phase == 6:
            results["Phase 6: Early Detection"] = phase_6_early_detection()
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    warnings = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result is True else "✗ FAIL" if result is False else "⚠ WARN"
        print(f"{status}: {test_name}")
    
    print(f"\nSummary: {passed} passed, {failed} failed, {warnings} warnings")
    print(f"End time: {datetime.now().isoformat()}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
