"""Check forecast panels in Grafana dashboard."""
import json
from pathlib import Path

dashboard_path = Path("grafana_dashboards/asset_health_dashboard.json")
with open(dashboard_path) as f:
    dashboard = json.load(f)

forecast_panels = [
    p for p in dashboard['panels'] 
    if any(keyword in p.get('title', '') for keyword in ['Forecast', 'RUL', 'Failure Prob'])
]

print(f"Found {len(forecast_panels)} forecast-related panels:\n")
for panel in forecast_panels:
    title = panel.get('title', 'Untitled')
    targets = panel.get('targets', [])
    if targets:
        sql = targets[0].get('rawSql', 'N/A')
        print(f"Panel: {title}")
        print(f"SQL: {sql[:300]}...")
        print()
