#!/usr/bin/env python3
"""
Comprehensive Chart Analysis and Optimization Plan
Run: 20251105_010417 (15 charts generated, 3.316s, 10.6% of runtime)
"""

<<<<<<< HEAD
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
=======
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
>>>>>>> 3d95a39f2dd1a1333531c7363d383cea730a3a74

from utils.logger import Console

chart_dir = Path("artifacts/FD_FAN/run_20251105_010417/charts")

Console.info("="*70)
Console.info("CHART ANALYSIS & OPTIMIZATION PLAN")
Console.info("="*70)
Console.info(f"\n📊 Run: 20251105_010417")
Console.info(f"📈 Charts generated: 15")
<<<<<<< HEAD
Console.info(f"⏱️  Time: 3.316s (10.6% of total runtime)")
Console.info(f"📁 Location: {chart_dir}")
=======
Console.info(f"⏱️  Time: 3.316s (10.6% of runtime)")
Console.info(f"📁 Location: {chart_dir}", chart_dir=str(chart_dir))
>>>>>>> 3d95a39f2dd1a1333531c7363d383cea730a3a74

charts = [
    "contribution_bars.png",
    "defect_dashboard.png", 
    "defect_severity.png",
    "detector_comparison.png",
    "episodes_timeline.png",
    "health_distribution_over_time.png",
    "health_timeline.png",
    "regime_distribution.png",
    "regime_scatter.png",
    "sensor_anomaly_heatmap.png",
    "sensor_daily_profile.png",
    "sensor_defect_heatmap.png",
    "sensor_hotspots.png",
    "sensor_sparklines.png",
    "sensor_timeseries_events.png"
]

Console.info("\n" + "="*70)
Console.info("CHART CATEGORIZATION & ASSESSMENT")
Console.info("="*70)

high_value = {
    "defect_dashboard.png": "Quick KPI overview - very useful for triage",
    "detector_comparison.png": "Shows detector behavior over time - essential for debugging",
    "episodes_timeline.png": "Visual episode summary - high value",
    "health_timeline.png": "Health trend over time - critical for monitoring",
    "sensor_anomaly_heatmap.png": "Identifies problematic sensors - high diagnostic value"
}

medium_value = {
    "contribution_bars.png": "Snapshot detector contribution - useful but static",
    "defect_severity.png": "Episode severity distribution - moderate value",
    "regime_distribution.png": "Regime balance - useful for understanding states",
    "regime_scatter.png": "Regime vs anomaly score - good visualization",
    "sensor_daily_profile.png": "Diurnal pattern detection - moderate value"
}

low_value = {
    "health_distribution_over_time.png": "Heatmap of health by date/hour - niche use",
    "sensor_defect_heatmap.png": "Sensor vs severity matrix - only useful with many episodes",
    "sensor_hotspots.png": "Top anomalous sensors - likely redundant with heatmap",
    "sensor_sparklines.png": "Mini sensor timeseries - hard to read, low utility",
    "sensor_timeseries_events.png": "Full sensor plots - very slow to generate, often unclear"
}

Console.info("\n🟢 HIGH VALUE CHARTS (5) - Keep and potentially improve:")
for chart, desc in high_value.items():
    status = "✓" if (chart_dir / chart).exists() else "✗"
    Console.info(f"  {status} {chart:40s} | {desc}")

Console.info("\n🟡 MEDIUM VALUE CHARTS (5) - Keep but consider simplifying:")
for chart, desc in medium_value.items():
    status = "✓" if (chart_dir / chart).exists() else "✗"
    Console.info(f"  {status} {chart:40s} | {desc}")

Console.info("\n🔴 LOW VALUE CHARTS (5) - Consider removing:")
for chart, desc in low_value.items():
    status = "✓" if (chart_dir / chart).exists() else "✗"
    Console.info(f"  {status} {chart:40s} | {desc}")

Console.info("\n" + "="*70)
Console.info("OPTIMIZATION RECOMMENDATIONS")
Console.info("="*70)

Console.info("\n📌 TIER 1 - Remove Low-Value Charts (saves ~1.0-1.5s)")
Console.info("   - Remove: sensor_timeseries_events.png (slowest, least clear)")
Console.info("   - Remove: sensor_sparklines.png (too small to read)")
Console.info("   - Remove: sensor_hotspots.png (redundant with heatmap)")
Console.info("   - Keep as optional: health_distribution_over_time.png, sensor_defect_heatmap.png")

Console.info("\n📌 TIER 2 - Improve High-Value Charts")
Console.info("   ✓ defect_dashboard.png: Add regime distribution stats")
Console.info("   ✓ detector_comparison.png: Add episode markers/shading")
Console.info("   ✓ health_timeline.png: Add regime overlay")
Console.info("   ✓ episodes_timeline.png: Add severity color legend")

Console.info("\n📌 TIER 3 - Consolidation Opportunities")
Console.info("   • Merge contribution_bars.png + detector_comparison.png into single figure")
Console.info("   • Merge regime_distribution.png + regime_scatter.png into single 2-panel figure")
Console.info("   • Make chart generation more selective based on data characteristics:")
Console.info("     - Skip sensor_defect_heatmap if <5 episodes")
Console.info("     - Skip regime charts if k=1 (homogeneous data)")

Console.info("\n📌 TIER 4 - Performance Optimizations")
Console.info("   • Use lower sampling rates for timeseries (current: step=max(1, len/2000))")
Console.info("   • Reduce DPI from 150 to 100 for non-critical charts")
Console.info("   • Skip heavy computations (rolling windows) on large datasets")

Console.info("\n" + "="*70)
Console.info("EXPECTED IMPROVEMENTS")
Console.info("="*70)
Console.info("\n📊 Current: 15 charts, 3.316s (10.6% of runtime)")
Console.info("🎯 Target: 8-10 charts, ~1.5s (5% of runtime)")
Console.info("💡 Approach:")
Console.info("   1. Remove 3 low-value charts immediately: -1.0s")
<<<<<<< HEAD
Console.info("   2. Consolidate 2 pairs into combined figures: -0.4s")
=======
Console.info("   2. Consolidate 2 pairs into combined figures: -0.4s")  
>>>>>>> 3d95a39f2dd1a1333531c7363d383cea730a3a74
Console.info("   3. Optimize sampling/DPI: -0.4s")
Console.info("   4. Total savings: ~1.8s (55% reduction)")

Console.info("\n" + "="*70)
Console.info("IMPLEMENTATION PRIORITY")
Console.info("="*70)
Console.info("\n1️⃣ Quick Win: Remove sensor_timeseries_events.png (saves ~0.5s)")
Console.info("2️⃣ Quick Win: Remove sensor_sparklines.png (saves ~0.2s)")
Console.info("3️⃣ Quick Win: Remove sensor_hotspots.png (saves ~0.2s)")
Console.info("4️⃣ Enhancement: Add episode markers to detector_comparison.png")
Console.info("5️⃣ Enhancement: Add regime overlay to health_timeline.png")
Console.info("6️⃣ Consolidation: Merge regime charts into single figure")

Console.info("\n" + "="*70)
