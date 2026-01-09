#!/usr/bin/env python3
"""
Helper script to run ACM Distilled for multiple equipment.

This script automates running acm_distilled.py for:
- 2 Wind Turbines (WIND_TURBINE_01, WIND_TURBINE_02)
- FD Fan (FD_FAN)

Usage:
    # Run for all equipment with default time range (last 30 days)
    python run_distilled_batch.py
    
    # Run for specific time range
    python run_distilled_batch.py --start-time "2024-01-01T00:00:00" --end-time "2024-01-31T23:59:59"
    
    # Run for specific equipment only
    python run_distilled_batch.py --equip WIND_TURBINE_01 FD_FAN
    
    # Save reports to files
    python run_distilled_batch.py --output-dir reports/
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


# Default equipment list
DEFAULT_EQUIPMENT = [
    "WIND_TURBINE_01",
    "WIND_TURBINE_02", 
    "FD_FAN"
]


def run_acm_distilled(
    equipment: str,
    start_time: str,
    end_time: str,
    output_file: Optional[Path] = None
) -> bool:
    """
    Run acm_distilled.py for a single equipment.
    
    Args:
        equipment: Equipment code
        start_time: Analysis start time (ISO format)
        end_time: Analysis end time (ISO format)
        output_file: Optional output file path
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        sys.executable,
        "acm_distilled.py",
        "--equip", equipment,
        "--start-time", start_time,
        "--end-time", end_time
    ]
    
    if output_file:
        cmd.extend(["--output", str(output_file)])
    
    print(f"\n{'='*80}")
    print(f"Running ACM Distilled: {equipment}")
    print(f"Time range: {start_time} to {end_time}")
    if output_file:
        print(f"Output: {output_file}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n✓ SUCCESS: {equipment}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ FAILED: {equipment}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {equipment}")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run ACM Distilled for multiple equipment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for all equipment (last 30 days)
  python run_distilled_batch.py
  
  # Run for specific time range
  python run_distilled_batch.py --start-time "2024-01-01T00:00:00" --end-time "2024-01-31T23:59:59"
  
  # Run for specific equipment
  python run_distilled_batch.py --equip WIND_TURBINE_01 FD_FAN
  
  # Save reports to directory
  python run_distilled_batch.py --output-dir reports/
        """
    )
    
    parser.add_argument(
        '--equip',
        nargs='+',
        default=DEFAULT_EQUIPMENT,
        help=f'Equipment codes to analyze (default: {", ".join(DEFAULT_EQUIPMENT)})'
    )
    
    parser.add_argument(
        '--start-time',
        help='Analysis start time (ISO format: YYYY-MM-DDTHH:MM:SS). Default: 30 days ago'
    )
    
    parser.add_argument(
        '--end-time',
        help='Analysis end time (ISO format: YYYY-MM-DDTHH:MM:SS). Default: now'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Directory to save reports (default: print to console)'
    )
    
    args = parser.parse_args()
    
    # Set default time range if not provided (last 30 days)
    if args.end_time:
        end_time = args.end_time
    else:
        end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    if args.start_time:
        start_time = args.start_time
    else:
        start_dt = datetime.now() - timedelta(days=30)
        start_time = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Create output directory if specified
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    
    # Run for each equipment
    results = {}
    for equip in args.equip:
        output_file = None
        if args.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = args.output_dir / f"{equip}_{timestamp}.txt"
        
        success = run_acm_distilled(equip, start_time, end_time, output_file)
        results[equip] = success
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH RUN SUMMARY")
    print(f"{'='*80}")
    print(f"Time Range: {start_time} to {end_time}")
    print(f"Equipment Analyzed: {len(args.equip)}")
    print(f"\nResults:")
    
    success_count = 0
    for equip, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {equip:20s} - {status}")
        if success:
            success_count += 1
    
    print(f"\nTotal: {success_count}/{len(args.equip)} successful")
    print(f"{'='*80}\n")
    
    # Exit with appropriate code
    if success_count == len(args.equip):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
