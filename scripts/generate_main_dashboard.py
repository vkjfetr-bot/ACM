"""
Generate ACM Main Dashboard JSON
Creates a comprehensive equipment monitoring dashboard for Grafana
"""
import json
from pathlib import Path

def create_dashboard():
    """Generate the complete ACM main dashboard structure."""
    
    dashboard = {
        "annotations": {
            "list": [{
                "builtIn": 1,
                "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                "enable": True,
                "hide": True,
                "iconColor": "rgba(0, 211, 255, 1)",
                "name": "Annotations & Alerts",
                "type": "dashboard"
            }]
        },
        "description": "Comprehensive ACM Equipment Health Monitoring Dashboard",
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "panels": [],
        "schemaVersion": 39,
        "tags": ["acm", "equipment", "health", "predictive-maintenance"],
        "templating": {
            "list": [
                {
                    "current": {"selected": False, "text": "", "value": ""},
                    "hide": 0,
                    "includeAll": False,
                    "label": "Data Source",
                    "multi": False,
                    "name": "datasource",
                    "options": [],
                    "query": "mssql",
                    "refresh": 1,
                    "regex": "",
                    "skipUrlSync": False,
                    "type": "datasource"
                },
                {
                    "current": {"selected": False, "text": "", "value": ""},
                    "datasource": {"type": "mssql", "uid": "${datasource}"},
                    "definition": "SELECT EquipID AS __value, EquipName AS __text FROM Equipment WHERE EquipID > 0 ORDER BY EquipName",
                    "hide": 0,
                    "includeAll": False,
                    "label": "Equipment",
                    "multi": False,
                    "name": "equipment",
                    "options": [],
                    "query": "SELECT EquipID AS __value, EquipName AS __text FROM Equipment WHERE EquipID > 0 ORDER BY EquipName",
                    "refresh": 1,
                    "regex": "",
                    "skipUrlSync": False,
                    "sort": 1,
                    "type": "query"
                }
            ]
        },
        "time": {"from": "now-5y", "to": "now"},
        "timepicker": {"refresh_intervals": ["30s", "1m", "5m", "15m", "30m", "1h"]},
        "timezone": "browser",
        "title": "ACM Equipment Dashboard",
        "uid": "acm-main-dashboard",
        "version": 1,
        "weekStart": ""
    }
    
    panels = []
    panel_id = 1
    y_pos = 0
    
    # ========== SECTION 1: EXECUTIVE SUMMARY ==========
    panels.append({
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
        "id": panel_id,
        "panels": [],
        "title": "EXECUTIVE SUMMARY",
        "type": "row"
    })
    panel_id += 1
    y_pos += 1
    
    # Health Index Gauge
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Current equipment health index (0-100%)",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "max": 100, "min": 0,
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "red", "value": None},
                    {"color": "orange", "value": 50},
                    {"color": "yellow", "value": 70},
                    {"color": "green", "value": 85}
                ]},
                "unit": "percent"
            },
            "overrides": []
        },
        "gridPos": {"h": 5, "w": 4, "x": 0, "y": y_pos},
        "id": panel_id,
        "options": {
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "showThresholdLabels": True,
            "showThresholdMarkers": True
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "table",
            "rawQuery": True,
            "rawSql": "SELECT TOP 1 HealthIndex\nFROM ACM_HealthTimeline\nWHERE EquipID = $equipment\nORDER BY Timestamp DESC",
            "refId": "A"
        }],
        "title": "Health Index",
        "type": "gauge"
    })
    panel_id += 1
    
    # Health Zone Stat
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Current health zone classification",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "mappings": [
                    {"options": {"HEALTHY": {"color": "green", "index": 0, "text": "HEALTHY"}}, "type": "value"},
                    {"options": {"WARNING": {"color": "yellow", "index": 1, "text": "WARNING"}}, "type": "value"},
                    {"options": {"ALERT": {"color": "orange", "index": 2, "text": "ALERT"}}, "type": "value"},
                    {"options": {"CRITICAL": {"color": "red", "index": 3, "text": "CRITICAL"}}, "type": "value"}
                ],
                "thresholds": {"mode": "absolute", "steps": [{"color": "text", "value": None}]}
            },
            "overrides": []
        },
        "gridPos": {"h": 5, "w": 4, "x": 4, "y": y_pos},
        "id": panel_id,
        "options": {
            "colorMode": "background",
            "graphMode": "none",
            "justifyMode": "center",
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "auto"
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "table",
            "rawQuery": True,
            "rawSql": "SELECT TOP 1 HealthZone\nFROM ACM_HealthTimeline\nWHERE EquipID = $equipment\nORDER BY Timestamp DESC",
            "refId": "A"
        }],
        "title": "Health Zone",
        "type": "stat"
    })
    panel_id += 1
    
    # RUL Stat
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Remaining Useful Life prediction (hours)",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "red", "value": None},
                    {"color": "orange", "value": 72},
                    {"color": "yellow", "value": 168},
                    {"color": "green", "value": 720}
                ]},
                "unit": "h"
            },
            "overrides": []
        },
        "gridPos": {"h": 5, "w": 4, "x": 8, "y": y_pos},
        "id": panel_id,
        "options": {
            "colorMode": "background",
            "graphMode": "none",
            "justifyMode": "center",
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "auto"
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "table",
            "rawQuery": True,
            "rawSql": "SELECT TOP 1 RUL_Hours\nFROM ACM_RUL\nWHERE EquipID = $equipment\n  AND (P10_LowerBound IS NOT NULL OR P50_Median IS NOT NULL)\nORDER BY CreatedAt DESC",
            "refId": "A"
        }],
        "title": "RUL (Hours)",
        "type": "stat"
    })
    panel_id += 1
    
    # RUL Confidence Gauge
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Confidence in RUL prediction",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "max": 1, "min": 0,
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "red", "value": None},
                    {"color": "orange", "value": 0.3},
                    {"color": "yellow", "value": 0.6},
                    {"color": "green", "value": 0.8}
                ]},
                "unit": "percentunit"
            },
            "overrides": []
        },
        "gridPos": {"h": 5, "w": 4, "x": 12, "y": y_pos},
        "id": panel_id,
        "options": {
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "showThresholdLabels": False,
            "showThresholdMarkers": True
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "table",
            "rawQuery": True,
            "rawSql": "SELECT TOP 1 Confidence\nFROM ACM_RUL\nWHERE EquipID = $equipment\nORDER BY CreatedAt DESC",
            "refId": "A"
        }],
        "title": "RUL Confidence",
        "type": "gauge"
    })
    panel_id += 1
    
    # Fused Z-Score Stat
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Current fused anomaly score",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "decimals": 2,
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 2},
                    {"color": "orange", "value": 3},
                    {"color": "red", "value": 4}
                ]}
            },
            "overrides": []
        },
        "gridPos": {"h": 5, "w": 4, "x": 16, "y": y_pos},
        "id": panel_id,
        "options": {
            "colorMode": "background",
            "graphMode": "area",
            "justifyMode": "center",
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "value_and_name"
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "table",
            "rawQuery": True,
            "rawSql": "SELECT TOP 1 fused AS FusedZ\nFROM ACM_Scores_Wide\nWHERE EquipID = $equipment\nORDER BY Timestamp DESC",
            "refId": "A"
        }],
        "title": "Fused Z-Score",
        "type": "stat"
    })
    panel_id += 1
    
    # Episodes Count Stat
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Total anomaly episodes detected",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 10},
                    {"color": "orange", "value": 25},
                    {"color": "red", "value": 50}
                ]}
            },
            "overrides": []
        },
        "gridPos": {"h": 5, "w": 4, "x": 20, "y": y_pos},
        "id": panel_id,
        "options": {
            "colorMode": "background",
            "graphMode": "none",
            "justifyMode": "center",
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "auto"
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "table",
            "rawQuery": True,
            "rawSql": "SELECT TOP 1 EpisodeCount\nFROM ACM_Episodes\nWHERE EquipID = $equipment\nORDER BY RunID DESC",
            "refId": "A"
        }],
        "title": "Episodes",
        "type": "stat"
    })
    panel_id += 1
    y_pos += 5
    
    # Latest Run Table
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Latest ACM run information",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "custom": {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False},
                "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}
            },
            "overrides": [{
                "matcher": {"id": "byName", "options": "Status"},
                "properties": [
                    {"id": "mappings", "value": [
                        {"options": {"COMPLETED": {"color": "green", "index": 0, "text": "OK"}}, "type": "value"},
                        {"options": {"RUNNING": {"color": "blue", "index": 1, "text": "RUNNING"}}, "type": "value"},
                        {"options": {"FAILED": {"color": "red", "index": 2, "text": "FAILED"}}, "type": "value"}
                    ]},
                    {"id": "custom.cellOptions", "value": {"type": "color-background"}}
                ]
            }]
        },
        "gridPos": {"h": 5, "w": 12, "x": 0, "y": y_pos},
        "id": panel_id,
        "options": {
            "cellHeight": "sm",
            "footer": {"countRows": False, "fields": "", "reducer": ["sum"], "show": False},
            "showHeader": True
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "table",
            "rawQuery": True,
            "rawSql": "SELECT TOP 1\n    EquipName AS Equipment,\n    FORMAT(StartedAt, 'yyyy-MM-dd HH:mm') AS StartedAt,\n    COALESCE(DurationSeconds, 0) AS DurationSec,\n    COALESCE(ScoreRowCount, 0) AS RowsProcessed,\n    HealthStatus AS Status\nFROM ACM_Runs\nWHERE EquipID = $equipment\nORDER BY StartedAt DESC",
            "refId": "A"
        }],
        "title": "Latest Run",
        "type": "table"
    })
    panel_id += 1
    
    # Detector Status Table
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Current detector status and severity",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "custom": {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False},
                "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}
            },
            "overrides": [
                {
                    "matcher": {"id": "byName", "options": "CurrentZ"},
                    "properties": [
                        {"id": "custom.cellOptions", "value": {"mode": "gradient", "type": "gauge"}},
                        {"id": "min", "value": 0},
                        {"id": "max", "value": 5},
                        {"id": "thresholds", "value": {"mode": "absolute", "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 2},
                            {"color": "orange", "value": 3},
                            {"color": "red", "value": 4}
                        ]}}
                    ]
                },
                {
                    "matcher": {"id": "byName", "options": "ActiveDefect"},
                    "properties": [
                        {"id": "mappings", "value": [{"options": {"0": {"color": "green", "index": 0, "text": "OK"}, "1": {"color": "red", "index": 1, "text": "DEFECT"}}, "type": "value"}]},
                        {"id": "custom.cellOptions", "value": {"type": "color-background"}}
                    ]
                }
            ]
        },
        "gridPos": {"h": 5, "w": 12, "x": 12, "y": y_pos},
        "id": panel_id,
        "options": {
            "cellHeight": "sm",
            "footer": {"countRows": False, "fields": "", "reducer": ["sum"], "show": False},
            "showHeader": True,
            "sortBy": [{"desc": True, "displayName": "CurrentZ"}]
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "table",
            "rawQuery": True,
            "rawSql": "SELECT \n    DetectorType AS Detector,\n    Severity,\n    ROUND(CurrentZ, 2) AS CurrentZ,\n    ActiveDefect\nFROM ACM_SensorDefects\nWHERE EquipID = $equipment\nORDER BY CurrentZ DESC",
            "refId": "A"
        }],
        "title": "Detector Status",
        "type": "table"
    })
    panel_id += 1
    y_pos += 5
    
    # ========== SECTION 2: HEALTH TRENDS ==========
    panels.append({
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
        "id": panel_id,
        "panels": [],
        "title": "HEALTH TRENDS",
        "type": "row"
    })
    panel_id += 1
    y_pos += 1
    
    # Health Timeline with Forecast
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Health index over time with confidence band forecast",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisBorderShow": False,
                    "axisCenteredZero": False,
                    "axisColorMode": "text",
                    "axisLabel": "Health %",
                    "axisPlacement": "auto",
                    "barAlignment": 0,
                    "drawStyle": "line",
                    "fillOpacity": 10,
                    "gradientMode": "none",
                    "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                    "insertNulls": False,
                    "lineInterpolation": "smooth",
                    "lineWidth": 2,
                    "pointSize": 5,
                    "scaleDistribution": {"type": "linear"},
                    "showPoints": "never",
                    "spanNulls": 3600000,
                    "stacking": {"group": "A", "mode": "none"},
                    "thresholdsStyle": {"mode": "dashed"}
                },
                "max": 100, "min": 0,
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "red", "value": None},
                    {"color": "orange", "value": 50},
                    {"color": "yellow", "value": 70},
                    {"color": "green", "value": 85}
                ]},
                "unit": "percent"
            },
            "overrides": [
                {
                    "matcher": {"id": "byName", "options": "Forecast"},
                    "properties": [
                        {"id": "custom.lineStyle", "value": {"dash": [10, 10], "fill": "dash"}},
                        {"id": "color", "value": {"fixedColor": "orange", "mode": "fixed"}}
                    ]
                },
                {
                    "matcher": {"id": "byName", "options": "CILower"},
                    "properties": [
                        {"id": "custom.lineWidth", "value": 0},
                        {"id": "custom.fillOpacity", "value": 0},
                        {"id": "color", "value": {"fixedColor": "transparent", "mode": "fixed"}}
                    ]
                },
                {
                    "matcher": {"id": "byName", "options": "CIUpper"},
                    "properties": [
                        {"id": "custom.fillBelowTo", "value": "CILower"},
                        {"id": "custom.lineWidth", "value": 0},
                        {"id": "custom.fillOpacity", "value": 20},
                        {"id": "color", "value": {"fixedColor": "orange", "mode": "fixed"}}
                    ]
                }
            ]
        },
        "gridPos": {"h": 9, "w": 24, "x": 0, "y": y_pos},
        "id": panel_id,
        "options": {
            "legend": {"calcs": ["lastNotNull", "min", "max", "mean"], "displayMode": "table", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "none"}
        },
        "targets": [
            {
                "datasource": {"type": "mssql", "uid": "${datasource}"},
                "format": "time_series",
                "rawQuery": True,
                "rawSql": "SELECT Timestamp AS time, HealthIndex AS ActualHealth\nFROM ACM_HealthTimeline\nWHERE EquipID = $equipment\n  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()\nORDER BY time ASC",
                "refId": "A"
            },
            {
                "datasource": {"type": "mssql", "uid": "${datasource}"},
                "format": "time_series",
                "rawQuery": True,
                "rawSql": "SELECT Timestamp AS time, ForecastHealth AS Forecast\nFROM ACM_HealthForecast\nWHERE EquipID = $equipment\n  AND Timestamp BETWEEN $__timeFrom() AND DATEADD(DAY, 7, $__timeTo())\nORDER BY time ASC",
                "refId": "B"
            },
            {
                "datasource": {"type": "mssql", "uid": "${datasource}"},
                "format": "time_series",
                "rawQuery": True,
                "rawSql": "SELECT Timestamp AS time, CiLower AS CILower\nFROM ACM_HealthForecast\nWHERE EquipID = $equipment\n  AND Timestamp BETWEEN $__timeFrom() AND DATEADD(DAY, 7, $__timeTo())\nORDER BY time ASC",
                "refId": "C"
            },
            {
                "datasource": {"type": "mssql", "uid": "${datasource}"},
                "format": "time_series",
                "rawQuery": True,
                "rawSql": "SELECT Timestamp AS time, CiUpper AS CIUpper\nFROM ACM_HealthForecast\nWHERE EquipID = $equipment\n  AND Timestamp BETWEEN $__timeFrom() AND DATEADD(DAY, 7, $__timeTo())\nORDER BY time ASC",
                "refId": "D"
            }
        ],
        "title": "Health Timeline with Forecast",
        "type": "timeseries"
    })
    panel_id += 1
    y_pos += 9
    
    # Health Zone Timeline
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Health zone distribution over time",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "custom": {"fillOpacity": 70, "insertNulls": False, "lineWidth": 0, "spanNulls": False},
                "mappings": [
                    {"options": {"HEALTHY": {"color": "green", "index": 0}}, "type": "value"},
                    {"options": {"WARNING": {"color": "yellow", "index": 1}}, "type": "value"},
                    {"options": {"ALERT": {"color": "orange", "index": 2}}, "type": "value"},
                    {"options": {"CRITICAL": {"color": "red", "index": 3}}, "type": "value"}
                ],
                "thresholds": {"mode": "absolute", "steps": [{"color": "blue", "value": None}]}
            },
            "overrides": []
        },
        "gridPos": {"h": 4, "w": 24, "x": 0, "y": y_pos},
        "id": panel_id,
        "options": {
            "alignValue": "center",
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "mergeValues": True,
            "rowHeight": 0.9,
            "showValue": "auto",
            "tooltip": {"mode": "single", "sort": "none"}
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "time_series",
            "rawQuery": True,
            "rawSql": "SELECT \n    Timestamp AS time,\n    HealthZone AS Zone\nFROM ACM_HealthTimeline\nWHERE EquipID = $equipment\n  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()\nORDER BY time ASC",
            "refId": "A"
        }],
        "title": "Health Zone Timeline",
        "type": "state-timeline"
    })
    panel_id += 1
    y_pos += 4
    
    # ========== SECTION 3: DETECTOR SIGNALS ==========
    panels.append({
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
        "id": panel_id,
        "panels": [],
        "title": "DETECTOR SIGNALS",
        "type": "row"
    })
    panel_id += 1
    y_pos += 1
    
    # Detector Signals Timeline
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Individual detector Z-scores over time",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisBorderShow": False,
                    "axisCenteredZero": False,
                    "axisColorMode": "text",
                    "axisLabel": "Z-Score",
                    "axisPlacement": "auto",
                    "barAlignment": 0,
                    "drawStyle": "line",
                    "fillOpacity": 0,
                    "gradientMode": "none",
                    "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                    "insertNulls": False,
                    "lineInterpolation": "smooth",
                    "lineWidth": 1,
                    "pointSize": 5,
                    "scaleDistribution": {"type": "linear"},
                    "showPoints": "never",
                    "spanNulls": 3600000,
                    "stacking": {"group": "A", "mode": "none"},
                    "thresholdsStyle": {"mode": "line"}
                },
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 2},
                    {"color": "orange", "value": 3},
                    {"color": "red", "value": 4}
                ]}
            },
            "overrides": [{
                "matcher": {"id": "byName", "options": "Fused"},
                "properties": [
                    {"id": "custom.lineWidth", "value": 3},
                    {"id": "color", "value": {"fixedColor": "purple", "mode": "fixed"}}
                ]
            }]
        },
        "gridPos": {"h": 10, "w": 24, "x": 0, "y": y_pos},
        "id": panel_id,
        "options": {
            "legend": {"calcs": ["lastNotNull", "max"], "displayMode": "table", "placement": "right", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"}
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "time_series",
            "rawQuery": True,
            "rawSql": "SELECT \n    Timestamp AS time,\n    ar1_z AS AR1,\n    pca_spe_z AS PCASPE,\n    pca_t2_z AS PCAT2,\n    iforest_z AS IForest,\n    gmm_z AS GMM,\n    cusum_z AS CUSUM,\n    fused AS Fused\nFROM ACM_Scores_Wide\nWHERE EquipID = $equipment\n  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()\nORDER BY time ASC",
            "refId": "A"
        }],
        "title": "Detector Signals Timeline",
        "type": "timeseries"
    })
    panel_id += 1
    y_pos += 10
    
    # ========== SECTION 4: SENSOR DIAGNOSTICS ==========
    panels.append({
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
        "id": panel_id,
        "panels": [],
        "title": "SENSOR DIAGNOSTICS",
        "type": "row"
    })
    panel_id += 1
    y_pos += 1
    
    # Sensor Hotspots Table
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Top contributing sensors ranked by maximum Z-score",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "custom": {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False},
                "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}
            },
            "overrides": [
                {
                    "matcher": {"id": "byName", "options": "MaxAbsZ"},
                    "properties": [
                        {"id": "custom.cellOptions", "value": {"mode": "gradient", "type": "gauge"}},
                        {"id": "min", "value": 0},
                        {"id": "max", "value": 6},
                        {"id": "thresholds", "value": {"mode": "absolute", "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 2},
                            {"color": "orange", "value": 3},
                            {"color": "red", "value": 4}
                        ]}}
                    ]
                },
                {
                    "matcher": {"id": "byName", "options": "LatestZ"},
                    "properties": [
                        {"id": "custom.cellOptions", "value": {"mode": "gradient", "type": "gauge"}},
                        {"id": "min", "value": 0},
                        {"id": "max", "value": 5},
                        {"id": "thresholds", "value": {"mode": "absolute", "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 2},
                            {"color": "orange", "value": 3},
                            {"color": "red", "value": 4}
                        ]}}
                    ]
                }
            ]
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": y_pos},
        "id": panel_id,
        "options": {
            "cellHeight": "sm",
            "footer": {"countRows": False, "fields": "", "reducer": ["sum"], "show": False},
            "showHeader": True,
            "sortBy": [{"desc": True, "displayName": "MaxAbsZ"}]
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "table",
            "rawQuery": True,
            "rawSql": "SELECT TOP 10\n    SensorName,\n    ROUND(MaxAbsZ, 2) AS MaxAbsZ,\n    ROUND(LatestAbsZ, 2) AS LatestZ,\n    AboveAlertCount AS Alerts\nFROM ACM_SensorHotspots\nWHERE EquipID = $equipment\nORDER BY MaxAbsZ DESC",
            "refId": "A"
        }],
        "title": "Sensor Hotspots",
        "type": "table"
    })
    panel_id += 1
    
    # Episode Culprits Table
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Episode culprit sensors by contribution percentage",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "custom": {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False},
                "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}
            },
            "overrides": [{
                "matcher": {"id": "byName", "options": "ContributionPct"},
                "properties": [
                    {"id": "unit", "value": "percent"},
                    {"id": "custom.cellOptions", "value": {"mode": "gradient", "type": "gauge"}},
                    {"id": "min", "value": 0},
                    {"id": "max", "value": 100}
                ]
            }]
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": y_pos},
        "id": panel_id,
        "options": {
            "cellHeight": "sm",
            "footer": {"countRows": False, "fields": "", "reducer": ["sum"], "show": False},
            "showHeader": True,
            "sortBy": [{"desc": True, "displayName": "ContributionPct"}]
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "table",
            "rawQuery": True,
            "rawSql": "SELECT TOP 10\n    DetectorType,\n    ROUND(ContributionPct, 1) AS ContributionPct,\n    Rank\nFROM ACM_EpisodeCulprits\nWHERE EquipID = $equipment\nORDER BY CreatedAt DESC, Rank ASC",
            "refId": "A"
        }],
        "title": "Episode Culprits",
        "type": "table"
    })
    panel_id += 1
    y_pos += 8
    
    # ========== SECTION 5: OPERATING REGIMES ==========
    panels.append({
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
        "id": panel_id,
        "panels": [],
        "title": "OPERATING REGIMES",
        "type": "row"
    })
    panel_id += 1
    y_pos += 1
    
    # Operating Regime Timeline
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Operating regime classification over time",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "custom": {"fillOpacity": 70, "insertNulls": False, "lineWidth": 0, "spanNulls": False},
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "blue", "value": None},
                    {"color": "green", "value": 1},
                    {"color": "yellow", "value": 2},
                    {"color": "orange", "value": 3},
                    {"color": "red", "value": 4}
                ]}
            },
            "overrides": []
        },
        "gridPos": {"h": 5, "w": 24, "x": 0, "y": y_pos},
        "id": panel_id,
        "options": {
            "alignValue": "center",
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "mergeValues": True,
            "rowHeight": 0.9,
            "showValue": "auto",
            "tooltip": {"mode": "single", "sort": "none"}
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "time_series",
            "rawQuery": True,
            "rawSql": "SELECT \n    Timestamp AS time,\n    RegimeLabel AS Regime\nFROM ACM_RegimeTimeline\nWHERE EquipID = $equipment\n  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()\nORDER BY time ASC",
            "refId": "A"
        }],
        "title": "Operating Regime Timeline",
        "type": "state-timeline"
    })
    panel_id += 1
    y_pos += 5
    
    # ========== SECTION 6: RUL AND FORECASTING ==========
    panels.append({
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
        "id": panel_id,
        "panels": [],
        "title": "RUL AND FORECASTING",
        "type": "row"
    })
    panel_id += 1
    y_pos += 1
    
    # RUL Prediction Details Table
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "RUL prediction details with confidence intervals",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "custom": {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False},
                "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}
            },
            "overrides": [
                {
                    "matcher": {"id": "byName", "options": "RUL_Hours"},
                    "properties": [
                        {"id": "unit", "value": "h"},
                        {"id": "thresholds", "value": {"mode": "absolute", "steps": [
                            {"color": "red", "value": None},
                            {"color": "orange", "value": 72},
                            {"color": "yellow", "value": 168},
                            {"color": "green", "value": 720}
                        ]}},
                        {"id": "custom.cellOptions", "value": {"type": "color-background"}}
                    ]
                },
                {
                    "matcher": {"id": "byName", "options": "Confidence"},
                    "properties": [
                        {"id": "unit", "value": "percentunit"},
                        {"id": "custom.cellOptions", "value": {"mode": "gradient", "type": "gauge"}},
                        {"id": "min", "value": 0},
                        {"id": "max", "value": 1}
                    ]
                }
            ]
        },
        "gridPos": {"h": 7, "w": 12, "x": 0, "y": y_pos},
        "id": panel_id,
        "options": {
            "cellHeight": "sm",
            "footer": {"countRows": False, "fields": "", "reducer": ["sum"], "show": False},
            "showHeader": True
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "table",
            "rawQuery": True,
            "rawSql": "SELECT TOP 1\n    ROUND(RUL_Hours, 1) AS RUL_Hours,\n    ROUND(P10_LowerBound, 1) AS P10,\n    ROUND(P50_Median, 1) AS P50,\n    ROUND(P90_UpperBound, 1) AS P90,\n    ROUND(Confidence, 3) AS Confidence,\n    Method,\n    TopSensor1,\n    TopSensor2,\n    TopSensor3\nFROM ACM_RUL\nWHERE EquipID = $equipment\n  AND (P10_LowerBound IS NOT NULL OR P50_Median IS NOT NULL)\nORDER BY CreatedAt DESC",
            "refId": "A"
        }],
        "title": "RUL Prediction Details",
        "type": "table"
    })
    panel_id += 1
    
    # Failure Probability Trend
    panels.append({
        "datasource": {"type": "mssql", "uid": "${datasource}"},
        "description": "Failure probability over time",
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisBorderShow": False,
                    "axisCenteredZero": False,
                    "axisColorMode": "text",
                    "axisLabel": "Probability",
                    "axisPlacement": "auto",
                    "barAlignment": 0,
                    "drawStyle": "line",
                    "fillOpacity": 30,
                    "gradientMode": "opacity",
                    "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                    "insertNulls": False,
                    "lineInterpolation": "smooth",
                    "lineWidth": 2,
                    "pointSize": 5,
                    "scaleDistribution": {"type": "linear"},
                    "showPoints": "never",
                    "spanNulls": 3600000,
                    "stacking": {"group": "A", "mode": "none"},
                    "thresholdsStyle": {"mode": "line"}
                },
                "max": 1, "min": 0,
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 0.3},
                    {"color": "orange", "value": 0.5},
                    {"color": "red", "value": 0.7}
                ]},
                "unit": "percentunit"
            },
            "overrides": []
        },
        "gridPos": {"h": 7, "w": 12, "x": 12, "y": y_pos},
        "id": panel_id,
        "options": {
            "legend": {"calcs": ["lastNotNull", "max"], "displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "single", "sort": "none"}
        },
        "targets": [{
            "datasource": {"type": "mssql", "uid": "${datasource}"},
            "format": "time_series",
            "rawQuery": True,
            "rawSql": "SELECT \n    Timestamp AS time,\n    FailureProb AS FailureProbability\nFROM ACM_FailureForecast\nWHERE EquipID = $equipment\n  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()\nORDER BY time ASC",
            "refId": "A"
        }],
        "title": "Failure Probability Trend",
        "type": "timeseries"
    })
    panel_id += 1
    y_pos += 7
    
    # ========== SECTION 7: OPERATIONS (COLLAPSED) ==========
    panels.append({
        "collapsed": True,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
        "id": panel_id,
        "panels": [
            {
                "datasource": {"type": "mssql", "uid": "${datasource}"},
                "description": "Recent ACM runs for this equipment",
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "custom": {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False},
                        "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}
                    },
                    "overrides": [{"matcher": {"id": "byName", "options": "Duration"}, "properties": [{"id": "unit", "value": "s"}]}]
                },
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": y_pos + 1},
                "id": panel_id + 1,
                "options": {
                    "cellHeight": "sm",
                    "footer": {"countRows": False, "fields": "", "reducer": ["sum"], "show": False},
                    "showHeader": True
                },
                "targets": [{
                    "datasource": {"type": "mssql", "uid": "${datasource}"},
                    "format": "table",
                    "rawQuery": True,
                    "rawSql": "SELECT TOP 20\n    FORMAT(StartedAt, 'yyyy-MM-dd HH:mm') AS Started,\n    DurationSeconds AS Duration,\n    TrainRowCount AS TrainRows,\n    ScoreRowCount AS ScoreRows,\n    EpisodeCount AS Episodes,\n    HealthStatus AS Status\nFROM ACM_Runs\nWHERE EquipID = $equipment\nORDER BY StartedAt DESC",
                    "refId": "A"
                }],
                "title": "Run History",
                "type": "table"
            },
            {
                "datasource": {"type": "mssql", "uid": "${datasource}"},
                "description": "Pipeline stage execution times",
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisBorderShow": False,
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "drawStyle": "bars",
                            "fillOpacity": 70,
                            "gradientMode": "none",
                            "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                            "insertNulls": False,
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {"type": "linear"},
                            "showPoints": "never",
                            "spanNulls": False,
                            "stacking": {"group": "A", "mode": "none"},
                            "thresholdsStyle": {"mode": "off"}
                        },
                        "mappings": [],
                        "unit": "s"
                    },
                    "overrides": []
                },
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": y_pos + 1},
                "id": panel_id + 2,
                "options": {
                    "legend": {"calcs": ["mean", "max"], "displayMode": "table", "placement": "right", "showLegend": True},
                    "tooltip": {"mode": "single", "sort": "none"}
                },
                "targets": [{
                    "datasource": {"type": "mssql", "uid": "${datasource}"},
                    "format": "table",
                    "rawQuery": True,
                    "rawSql": "SELECT \n    Section AS Stage,\n    AVG(DurationSeconds) AS AvgDuration\nFROM ACM_RunTimers\nWHERE EquipID = $equipment\nGROUP BY Section\nORDER BY AvgDuration DESC",
                    "refId": "A"
                }],
                "title": "Pipeline Stage Timings",
                "type": "barchart"
            }
        ],
        "title": "OPERATIONS AND DIAGNOSTICS",
        "type": "row"
    })
    
    dashboard["panels"] = panels
    return dashboard


def main():
    """Generate and save the dashboard JSON."""
    dashboard = create_dashboard()
    
    output_path = Path(__file__).parent.parent / "grafana_dashboards" / "acm_main_dashboard.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)
    
    # Count panels
    rows = [p for p in dashboard["panels"] if p["type"] == "row"]
    panels = len(dashboard["panels"]) - len(rows)
    
    print(f"Dashboard generated: {output_path}")
    print(f"  Sections: {len(rows)}")
    print(f"  Panels: {panels}")


if __name__ == "__main__":
    main()
