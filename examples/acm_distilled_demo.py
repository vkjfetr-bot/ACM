#!/usr/bin/env python3
"""
ACM Distilled - Example Usage Demo

This script shows example usage patterns and expected output format.
Run with: python examples/acm_distilled_demo.py
"""

def show_examples():
    print("=" * 80)
    print("ACM DISTILLED - EXAMPLE USAGE")
    print("=" * 80)
    print()
    
    print("1. BASIC USAGE:")
    print("-" * 80)
    print("python acm_distilled.py --equip FD_FAN \\")
    print("    --start-time '2024-01-01T00:00:00' \\")
    print("    --end-time '2024-01-31T23:59:59'")
    print()
    
    print("2. SAVE TO FILE:")
    print("-" * 80)
    print("python acm_distilled.py --equip GAS_TURBINE \\")
    print("    --start-time '2024-10-01T00:00:00' \\")
    print("    --end-time '2024-10-31T23:59:59' \\")
    print("    --output /tmp/report.txt")
    print()

if __name__ == "__main__":
    show_examples()
