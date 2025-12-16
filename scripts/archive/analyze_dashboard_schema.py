#!/usr/bin/env python3
"""
Analyze dashboard queries for schema compliance.
Manual check without needing live SQL connection.
"""

import json
from pathlib import Path

DASHBOARD_PATH = Path("grafana_dashboards/asset_health_dashboard_v3_fixed.json")

# Load dashboard
with open(DASHBOARD_PATH) as f:
    dashboard = json.load(f)

print("=" * 100)
print("DASHBOARD QUERY ANALYSIS - V10 SCHEMA VALIDATION")
print("=" * 100)
print(f"\nDashboard: {DASHBOARD_PATH.name}")
print(f"Total Panels: {len(dashboard.get('panels', []))}\n")

# Known V10 tables from schema
V10_TABLES = {
    "ACM_HealthTimeline": ["Timestamp", "HealthIndex", "HealthZone", "FusedZ", "RunID", "EquipID"],
    "ACM_OMRTimeline": ["Timestamp", "OMR_Z", "RunID", "EquipID"],
    "ACM_RegimeTimeline": ["Timestamp", "RegimeLabel", "RegimeState", "RunID", "EquipID"],
    "ACM_SensorHotspots": ["SensorName", "MaxTimestamp", "LatestTimestamp", "MaxAbsZ", "MaxSignedZ", 
                           "LatestAbsZ", "LatestSignedZ", "ValueAtPeak", "LatestValue", "TrainMean", 
                           "TrainStd", "AboveWarnCount", "AboveAlertCount", "RunID", "EquipID"],
    "ACM_OMRContributionsLong": ["Timestamp", "SensorName", "ContributionPct", "RunID", "EquipID"],
    "ACM_RUL_Summary": ["RunID", "EquipID", "RUL_Hours", "LowerBound", "UpperBound", "Confidence", "Method", "LastUpdate"],
    "ACM_RUL_TS": ["RunID", "EquipID", "Timestamp", "RUL_Hours", "LowerBound", "UpperBound", "Confidence", "Method"],
    "ACM_HealthForecast_TS": ["RunID", "EquipID", "Timestamp", "ForecastHealth", "CiLower", "CiUpper", "ForecastStd", "Method"],
    "ACM_FailureForecast_TS": ["RunID", "EquipID", "Timestamp", "FailureProb", "ThresholdUsed", "Method"],
    "ACM_EpisodeMetrics": ["TotalEpisodes", "TotalDurationHours", "AvgDurationHours", "MedianDurationHours", 
                          "MaxDurationHours", "MinDurationHours", "RatePerDay", "MeanInterarrivalHours", "RunID", "EquipID"],
    "ACM_DriftSeries": ["Timestamp", "DriftValue", "RunID", "EquipID"],
    "ACM_SensorHotspotTimeline": ["Timestamp", "SensorName", "Rank", "AbsZ", "RunID", "EquipID"],
    "ACM_DataQuality": ["sensor", "train_count", "train_nulls", "train_null_pct", "train_std", "train_longest_gap",
                       "train_flatline_span", "train_min_ts", "train_max_ts", "score_count", "score_nulls", 
                       "score_null_pct", "score_std", "score_longest_gap", "score_flatline_span", "score_min_ts",
                       "score_max_ts", "interp_method", "sampling_secs", "notes", "RunID", "EquipID", "CheckName", "CheckResult"],
    "ACM_ContributionTimeline": ["Timestamp", "DetectorType", "ContributionPct", "RunID", "EquipID"],
    "ACM_MaintenanceRecommendation": ["RunID", "EquipID", "EarliestMaintenance", "PreferredWindowStart", 
                                       "PreferredWindowEnd", "FailureProbAtWindowEnd", "Comment"],
    "ACM_Runs": ["RunID", "EquipID", "EquipName", "StartedAt", "CompletedAt", "DurationSeconds", "ConfigSignature",
                "TrainRowCount", "ScoreRowCount", "EpisodeCount", "HealthStatus", "AvgHealthIndex", "MinHealthIndex",
                "MaxFusedZ", "DataQualityScore", "RefitRequested", "ErrorMessage", "KeptColumns", "CreatedAt"],
    "ACM_RegimeOccupancy": ["RegimeLabel", "RecordCount", "Percentage", "RunID", "EquipID"],
    "Equipment": ["EquipID", "EquipName", "EquipCode", "EquipType"],
    "ACM_Scores_Wide": ["Timestamp", "ar1_z", "pca_spe_z", "pca_t2_z", "mhal_z", "iforest_z", "gmm_z", 
                       "cusum_z", "drift_z", "hst_z", "river_hst_z", "fused", "regime_label", "RunID", "EquipID"],
    "ACM_Anomaly_Events": ["EventID", "EventTime", "DetectorType", "SensorName", "ZScore", "RunID", "EquipID"],
    "ACM_FusionQualityReport": ["Detector", "Weight", "MeanZ", "MaxZ", "RunID", "EquipID"],
    "ACM_OMRContributions": ["Timestamp", "SensorName", "ContributionPct", "RunID", "EquipID"],
}

# Extract queries
queries = []
for panel_idx, panel in enumerate(dashboard.get("panels", [])):
    if panel.get("type") in ["stat", "timeseries", "table", "piechart"]:
        title = panel.get("title", f"Panel {panel_idx}")
        for target in panel.get("targets", []):
            raw_sql = target.get("rawSql", "")
            if raw_sql:
                queries.append({
                    "idx": len(queries) + 1,
                    "panel": title,
                    "panel_type": panel.get("type"),
                    "query": raw_sql,
                    "grid_pos": panel.get("gridPos")
                })

print(f"Total SQL Queries: {len(queries)}\n")
print("-" * 100)
print("QUERY VALIDATION RESULTS")
print("-" * 100)

issues_found = []

for q in queries:
    panel_name = q["panel"]
    query = q["query"]
    query_short = query[:100].replace("\n", " ")
    
    print(f"\n[{q['idx']:2d}] {panel_name}")
    print(f"      Type: {q['panel_type']}")
    print(f"      Query: {query_short}...")
    
    # Check for known problematic tables
    query_upper = query.upper()
    
    # Check for non-existent tables
    problems = []
    
    if "ACM_OMRCONTRIBUTIONSLONG" in query_upper:
        # Should this be ACM_OMRContributions or similar?
        if "SELECT TOP 15" in query and "SensorName" in query and "ContributionPct" in query:
            # This is trying to get contributions
            problems.append("Table 'ACM_OMRContributionsLong' does not exist in V10 schema. Use 'ACM_ContributionTimeline' or 'ACM_Scores_Wide'")
    
    if "ACM_HEALTHFORECAST" in query_upper and "ACM_HEALTHFORECAST_TS" not in query_upper:
        problems.append("Table 'ACM_HealthForecast' does not exist. Use 'ACM_HealthForecast_TS' with correct columns: ForecastHealth (not ForecastValue)")
    
    if "ACM_ANOMALY_EVENTS" in query_upper:
        if "EventTime" not in query and "EventType" not in query:
            problems.append("Table 'ACM_Anomaly_Events' may not exist. Consider using 'ACM_SensorHotspotTimeline' or 'ACM_DefectTimeline'")
    
    # Check for column mismatches
    if "LATESTHEALTHSCORE" in query_upper:
        problems.append("Column 'LatestHealthScore' does not exist in ACM_HealthTimeline. Use 'HealthIndex' instead")
    
    if "FORECASTVALUE" in query_upper and "ACM_HEALTHFORECAST_TS" in query_upper:
        problems.append("Column 'ForecastValue' does not exist in ACM_HealthForecast_TS. Use 'ForecastHealth' instead")
    
    if "RUL_CONFIDENCE" in query_upper:
        problems.append("Column 'RUL_Confidence' does not exist in ACM_RUL_Summary. Use 'Confidence' instead")
    
    if "FAILUREPROBABILITY" in query_upper and "ACM_FAILUREFORECAST" in query_upper:
        problems.append("Column 'FailureProbability' does not exist. Use 'FailureProb' from ACM_FailureForecast_TS")
    
    if "DRIFTSCORE" in query_upper and "ACM_DRIFTSERIES" in query_upper:
        problems.append("Column 'DriftScore' does not exist in ACM_DriftSeries. Use 'DriftValue' instead")
    
    if problems:
        for p in problems:
            print(f"      [ISSUE] {p}")
            issues_found.append({"panel": panel_name, "query": query[:100], "issue": p})
    else:
        print(f"      [OK] Query appears valid")

print("\n" + "=" * 100)
print("FEATURE COVERAGE ANALYSIS")
print("=" * 100)

features_checked = {
    "Health Score Monitoring": ("HealthIndex" in " ".join([q["query"].upper() for q in queries]), "ACM_HealthTimeline"),
    "OMR Detector": ("OMR_Z" in " ".join([q["query"].upper() for q in queries]), "ACM_OMRTimeline"),
    "Anomaly Detection (Hotspots)": ("SENSORHOTSPOT" in " ".join([q["query"].upper() for q in queries]), "ACM_SensorHotspots"),
    "Operating Regimes": ("REGIMETIMELINE" in " ".join([q["query"].upper() for q in queries]), "ACM_RegimeTimeline"),
    "Episodes & Metrics": ("EPISODEMETRICS" in " ".join([q["query"].upper() for q in queries]), "ACM_EpisodeMetrics"),
    "Drift Detection": ("DRIFTSERIES" in " ".join([q["query"].upper() for q in queries]), "ACM_DriftSeries"),
    "Health Forecasting": ("HEALTHFORECAST" in " ".join([q["query"].upper() for q in queries]), "ACM_HealthForecast_TS"),
    "Failure Probability": ("FAILUREFORECAST" in " ".join([q["query"].upper() for q in queries]), "ACM_FailureForecast_TS"),
    "RUL Estimation": ("RUL" in " ".join([q["query"].upper() for q in queries]), "ACM_RUL_Summary/ACM_RUL_TS"),
    "Data Quality": ("DATAQUALITY" in " ".join([q["query"].upper() for q in queries]), "ACM_DataQuality"),
    "Detector Fusion": ("CONTRIBUT" in " ".join([q["query"].upper() for q in queries]), "ACM_ContributionTimeline"),
    "Maintenance Recommendations": ("MAINTENANCE" in " ".join([q["query"].upper() for q in queries]), "ACM_MaintenanceRecommendation"),
}

print("\nFeature Coverage Summary:")
print(f"{'Feature':<40} {'Status':<8} {'Table Used'}")
print("-" * 100)

for feature, (found, table) in features_checked.items():
    status = "[COVERED]" if found else "[MISSING]"
    print(f"{feature:<40} {status:<8} {table}")

missing_features = [f for f, (found, _) in features_checked.items() if not found]

print("\n" + "=" * 100)
print(f"ISSUES SUMMARY: {len(issues_found)} issues found")
print("=" * 100)

if issues_found:
    print("\nDetailed Issues:")
    for i, issue in enumerate(issues_found, 1):
        print(f"\n{i}. Panel: {issue['panel']}")
        print(f"   Problem: {issue['issue']}")
        print(f"   Query: {issue['query']}")

if missing_features:
    print(f"\nMissing ACM Features ({len(missing_features)}):")
    for f in missing_features:
        print(f"  - {f}")

print("\n" + "=" * 100)
print("RECOMMENDATIONS")
print("=" * 100)
print("""
1. Replace ACM_OMRContributionsLong with actual V10 table (likely ACM_OMRContributions or derive from ACM_Scores_Wide)
2. Fix column name references: LatestHealthScore -> HealthIndex, ForecastValue -> ForecastHealth, etc.
3. Add missing features: Maintenance Recommendations, Defect Analysis
4. Verify Anomaly_Events table exists or switch to alternative approach
5. Add sensor-level forecasting from ACM_SensorForecast_TS if not already present
""")
