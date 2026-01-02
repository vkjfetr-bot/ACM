"""
Extract memory and CPU usage from Tempo traces and generate a CSV report.

This script reads trace data from Grafana Tempo and extracts resource metrics
(memory, CPU, duration) from span attributes set by observability.Span.

Usage:
    python scripts/extract_memory_report.py --trace-id TRACE_ID
    python scripts/extract_memory_report.py --run-id RUN_ID
    python scripts/extract_memory_report.py --last N  # Last N hours

Output:
    artifacts/memory_report_{equipment}_{timestamp}.csv
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

# Tempo API endpoint
TEMPO_BASE = "http://localhost:3200"

def get_trace(trace_id: str) -> Optional[dict]:
    """Fetch trace data from Tempo by trace ID."""
    url = f"{TEMPO_BASE}/api/traces/{trace_id}"
    try:
        with urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())
    except URLError as e:
        print(f"ERROR: Failed to fetch trace {trace_id}: {e}")
        return None

def search_traces(service: str = "acm-pipeline", limit: int = 100, 
                  start_ns: Optional[int] = None, end_ns: Optional[int] = None) -> list:
    """Search for traces in Tempo matching criteria."""
    params = [f"limit={limit}"]
    if start_ns:
        params.append(f"start={start_ns}")
    if end_ns:
        params.append(f"end={end_ns}")
    
    # Use TraceQL to find ACM traces
    query = '{resource.service.name=~"acm-.*"}'
    params.append(f"q={query}")
    
    url = f"{TEMPO_BASE}/api/search?{'&'.join(params)}"
    try:
        with urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("traces", [])
    except URLError as e:
        print(f"ERROR: Failed to search traces: {e}")
        return []

def extract_span_data(trace_data: dict) -> list[dict]:
    """Extract resource metrics from spans in trace data."""
    spans = []
    
    for batch in trace_data.get("batches", []):
        for scope_span in batch.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                attrs = {}
                for attr in span.get("attributes", []):
                    key = attr.get("key", "")
                    value = attr.get("value", {})
                    # Extract value from OTLP format
                    if "stringValue" in value:
                        attrs[key] = value["stringValue"]
                    elif "intValue" in value:
                        attrs[key] = int(value["intValue"])
                    elif "doubleValue" in value:
                        attrs[key] = float(value["doubleValue"])
                    elif "boolValue" in value:
                        attrs[key] = value["boolValue"]
                
                # Only include spans with resource tracking data
                if "acm.mem_mb" in attrs or "acm.duration_s" in attrs:
                    # Parse span name (format: "section:equipment")
                    span_name = span.get("name", "")
                    if ":" in span_name:
                        section, equipment = span_name.rsplit(":", 1)
                    else:
                        section = span_name
                        equipment = attrs.get("acm.equipment", "")
                    
                    # Convert nanoseconds to datetime
                    start_ns = int(span.get("startTimeUnixNano", 0))
                    end_ns = int(span.get("endTimeUnixNano", 0))
                    start_dt = datetime.fromtimestamp(start_ns / 1e9)
                    end_dt = datetime.fromtimestamp(end_ns / 1e9)
                    
                    spans.append({
                        "trace_id": attrs.get("acm.run_id", ""),
                        "equipment": equipment,
                        "section": section,
                        "phase": attrs.get("acm.phase", ""),
                        "category": attrs.get("acm.category", ""),
                        "start_time": start_dt.isoformat(),
                        "end_time": end_dt.isoformat(),
                        "duration_s": attrs.get("acm.duration_s", 0),
                        "mem_mb": attrs.get("acm.mem_mb", 0),
                        "mem_delta_mb": attrs.get("acm.mem_delta_mb", 0),
                        "cpu_pct": attrs.get("acm.cpu_pct", 0),
                        "batch_num": attrs.get("acm.batch_num", 0),
                        "batch_total": attrs.get("acm.batch_total", 0),
                    })
    
    return spans

def generate_report(spans: list[dict], output_path: Path) -> None:
    """Generate CSV report from span data."""
    if not spans:
        print("No spans found with resource data.")
        return
    
    # Sort by start time
    spans.sort(key=lambda x: x["start_time"])
    
    # Write CSV
    fieldnames = [
        "trace_id", "equipment", "section", "phase", "category",
        "start_time", "end_time", "duration_s", 
        "mem_mb", "mem_delta_mb", "cpu_pct",
        "batch_num", "batch_total"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(spans)
    
    print(f"\nWrote {len(spans)} spans to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("MEMORY USAGE REPORT")
    print("=" * 80)
    
    # Group by phase and show memory
    phase_stats: dict[str, list] = {}
    for span in spans:
        phase = span["phase"] or span["category"]
        if phase not in phase_stats:
            phase_stats[phase] = []
        phase_stats[phase].append(span)
    
    print(f"\n{'Phase':<20} {'Section':<35} {'Duration':>10} {'Mem MB':>10} {'Delta MB':>10} {'CPU %':>8}")
    print("-" * 95)
    
    # Sort by peak memory within each phase
    all_by_mem = sorted(spans, key=lambda x: x["mem_mb"], reverse=True)
    
    # Show top 20 by peak memory
    print("\nTOP 20 PHASES BY PEAK MEMORY:")
    print("-" * 95)
    for span in all_by_mem[:20]:
        print(f"{span['phase']:<20} {span['section']:<35} {span['duration_s']:>10.2f} {span['mem_mb']:>10.1f} {span['mem_delta_mb']:>10.1f} {span['cpu_pct']:>8.1f}")
    
    # Show top 20 by memory delta (growth)
    all_by_delta = sorted(spans, key=lambda x: x["mem_delta_mb"], reverse=True)
    print("\nTOP 20 PHASES BY MEMORY GROWTH:")
    print("-" * 95)
    for span in all_by_delta[:20]:
        print(f"{span['phase']:<20} {span['section']:<35} {span['duration_s']:>10.2f} {span['mem_mb']:>10.1f} {span['mem_delta_mb']:>10.1f} {span['cpu_pct']:>8.1f}")


def main():
    parser = argparse.ArgumentParser(description="Extract memory report from Tempo traces")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trace-id", help="Specific trace ID to analyze")
    group.add_argument("--run-id", help="ACM run ID to find and analyze")
    group.add_argument("--last", type=int, help="Analyze traces from last N hours")
    parser.add_argument("--output", help="Output CSV path (default: artifacts/memory_report_*.csv)")
    
    args = parser.parse_args()
    
    # Collect spans from traces
    all_spans = []
    
    if args.trace_id:
        print(f"Fetching trace: {args.trace_id}")
        trace_data = get_trace(args.trace_id)
        if trace_data:
            all_spans = extract_span_data(trace_data)
    
    elif args.run_id:
        print(f"Searching for run: {args.run_id}")
        # Search for traces with this run ID
        traces = search_traces(limit=1000)
        for trace_info in traces:
            trace_id = trace_info.get("traceID")
            if trace_id:
                trace_data = get_trace(trace_id)
                if trace_data:
                    spans = extract_span_data(trace_data)
                    # Filter for matching run_id
                    matching = [s for s in spans if s["trace_id"] == args.run_id]
                    all_spans.extend(matching)
                    if matching:
                        break  # Found the run
    
    elif args.last:
        print(f"Searching traces from last {args.last} hours...")
        now = datetime.now()
        start = now - timedelta(hours=args.last)
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(now.timestamp() * 1e9)
        
        traces = search_traces(limit=100, start_ns=start_ns, end_ns=end_ns)
        print(f"Found {len(traces)} traces")
        
        for trace_info in traces:
            trace_id = trace_info.get("traceID")
            if trace_id:
                trace_data = get_trace(trace_id)
                if trace_data:
                    spans = extract_span_data(trace_data)
                    all_spans.extend(spans)
    
    if not all_spans:
        print("No trace data found with resource metrics.")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Get equipment from first span
        equipment = all_spans[0].get("equipment", "unknown") if all_spans else "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"artifacts/memory_report_{equipment}_{timestamp}.csv")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(all_spans, output_path)


if __name__ == "__main__":
    main()
