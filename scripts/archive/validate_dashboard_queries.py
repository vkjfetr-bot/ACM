#!/usr/bin/env python3
"""
Validate all dashboard queries against V10 schema.
Tests: table existence, column names, query syntax, expected data.
"""

import json
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from core.sql_client import SQLClient
from utils.config_dict import ConfigDict

# Dashboard path
DASHBOARD_PATH = REPO_ROOT / "grafana_dashboards" / "asset_health_dashboard_v3_fixed.json"

# Load dashboard JSON
with open(DASHBOARD_PATH) as f:
    dashboard = json.load(f)

print("=" * 80)
print("DASHBOARD QUERY VALIDATION REPORT")
print("=" * 80)
print(f"Dashboard: {DASHBOARD_PATH.name}")
print(f"Total Panels: {len(dashboard.get('panels', []))}")
print()

# Extract all queries
queries = []
for panel in dashboard.get("panels", []):
    if panel.get("type") in ["stat", "timeseries", "table"]:
        title = panel.get("title", "Unknown")
        for target in panel.get("targets", []):
            raw_sql = target.get("rawSql", "")
            if raw_sql:
                queries.append({
                    "panel": title,
                    "query": raw_sql,
                    "type": panel.get("type")
                })

print(f"Total SQL Queries Found: {len(queries)}\n")

# Try to connect to SQL
print("-" * 80)
print("CONNECTING TO SQL DATABASE...")
print("-" * 80)

try:
    cfg = ConfigDict.from_csv("configs/config_table.csv")
    client = SQLClient.from_ini(db_section="acm")
    print("[OK] SQL connection successful")
    print(f"  Server: {client.server}")
    print(f"  Database: {client.database}\n")
except Exception as e:
    print(f"[FAIL] SQL connection failed: {e}\n")
    print("Cannot validate queries without database connection.")
    sys.exit(1)

# Validate each query
print("-" * 80)
print("QUERY VALIDATION RESULTS")
print("-" * 80)

issues = []
valid_count = 0

for i, q in enumerate(queries, 1):
    panel_name = q["panel"]
    query = q["query"]
    query_short = query[:80].replace("\n", " ") + ("..." if len(query) > 80 else "")
    
    print(f"\n[{i}] Panel: {panel_name}")
    print(f"    Query: {query_short}")
    print(f"    Type: {q['type']}")
    
    try:
        # Replace $equipment with a test value
        test_query = query.replace("$equipment", "1")
        
        # Try to execute with LIMIT
        if not test_query.upper().endswith(")"):
            test_query = f"SELECT * FROM ({test_query}) AS t OFFSET 0 ROWS FETCH NEXT 0 ROWS ONLY"
        
        cursor = client.conn.cursor()
        cursor.execute(test_query)
        cols = [desc[0] for desc in cursor.description] if cursor.description else []
        cursor.close()
        
        print(f"    ✓ VALID")
        if cols:
            print(f"      Columns: {', '.join(cols[:5])}" + ("..." if len(cols) > 5 else ""))
        valid_count += 1
        
    except Exception as e:
        error_msg = str(e)
        print(f"    ✗ INVALID")
        print(f"      Error: {error_msg}")
        issues.append({
            "panel": panel_name,
            "error": error_msg,
            "query": query[:200]
        })

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Valid Queries: {valid_count}/{len(queries)}")
print(f"Invalid Queries: {len(issues)}/{len(queries)}")

if issues:
    print("\nISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. Panel: {issue['panel']}")
        print(f"   Error: {issue['error']}")
        print(f"   Query: {issue['query'][:200]}")

print("\n" + "=" * 80)
print("MISSING/UNDERUSED FEATURES CHECK")
print("=" * 80)

# Check which ACM features are covered
features = {
    "Health Monitoring": False,
    "OMR Detector": False,
    "Anomaly Detection": False,
    "Operating Regimes": False,
    "Episodes & Drift": False,
    "Forecasting": False,
    "RUL": False,
    "Data Quality": False,
    "Detector Fusion": False,
    "Maintenance Recommendations": False,
    "Defect Analysis": False,
    "Sensor Ranking": False,
}

query_text = " ".join([q["query"].upper() for q in queries])

if "ACMHEALTHTIMELINE" in query_text or "HEALTHINDEX" in query_text:
    features["Health Monitoring"] = True
if "OMRCONTRIBUTION" in query_text or "OMR_Z" in query_text:
    features["OMR Detector"] = True
if "SENSORHOTSPOT" in query_text:
    features["Anomaly Detection"] = True
if "REGIMETIMELINE" in query_text or "REGIME" in query_text:
    features["Operating Regimes"] = True
if "EPISODE" in query_text or "DRIFT" in query_text:
    features["Episodes & Drift"] = True
if "FORECAST" in query_text:
    features["Forecasting"] = True
if "RUL" in query_text:
    features["RUL"] = True
if "DATAQUALITY" in query_text:
    features["Data Quality"] = True
if "FUSION" in query_text or "CONTRIBUTION" in query_text:
    features["Detector Fusion"] = True
if "MAINTENANCE" in query_text:
    features["Maintenance Recommendations"] = True
if "DEFECT" in query_text:
    features["Defect Analysis"] = True
if "SENSORRANKING" in query_text:
    features["Sensor Ranking"] = True

print("\nFeature Coverage:")
for feature, covered in features.items():
    status = "✓" if covered else "✗"
    print(f"  {status} {feature}")

uncovered = [f for f, c in features.items() if not c]
if uncovered:
    print(f"\nMissing Features ({len(uncovered)}):")
    for f in uncovered:
        print(f"  - {f}")

print("\n" + "=" * 80)
