"""Extract and analyze all panels from asset_health_dashboard.json."""
import json
import sys
sys.path.insert(0, '.')

from pathlib import Path
from core.sql_client import SQLClient
from utils.config_dict import ConfigDict

# Load dashboard JSON
dashboard_path = Path("grafana_dashboards/asset_health_dashboard.json")
with open(dashboard_path, 'r', encoding='utf-8') as f:
    dashboard = json.load(f)

# Extract all panels
panels = []
for panel in dashboard.get('panels', []):
    panel_info = {
        'id': panel.get('id'),
        'title': panel.get('title', 'Untitled'),
        'type': panel.get('type'),
        'targets': []
    }
    
    for target in panel.get('targets', []):
        if 'rawSql' in target:
            panel_info['targets'].append(target['rawSql'])
    
    if panel_info['targets']:  # Only include panels with SQL queries
        panels.append(panel_info)

print(f"\n{'='*80}")
print(f"DASHBOARD PANEL ANALYSIS: {len(panels)} panels with SQL queries")
print(f"{'='*80}\n")

# Connect to SQL and check each panel's data
cfg = ConfigDict({
    'sql_connection': {
        'server': 'localhost\\B19CL3PCQLSERVER',
        'database': 'ACM',
        'trusted_connection': True
    }
})

sql = SQLClient(cfg)
sql.connect()
cur = sql.cursor()

for idx, panel in enumerate(panels, 1):
    print(f"\n{'─'*80}")
    print(f"Panel {idx}/{len(panels)}: {panel['title']} (ID: {panel['id']}, Type: {panel['type']})")
    print(f"{'─'*80}")
    
    for query_idx, query in enumerate(panel['targets'], 1):
        print(f"\n  Query {query_idx}:")
        print(f"  {query[:200]}..." if len(query) > 200 else f"  {query}")
        
        # Extract table names from query
        import re
        table_pattern = r'FROM\s+(\w+)|JOIN\s+(\w+)'
        tables = re.findall(table_pattern, query, re.IGNORECASE)
        tables = [t[0] or t[1] for t in tables if t[0] or t[1]]
        
        if tables:
            print(f"\n  Tables used: {', '.join(set(tables))}")
            
            # Check if tables exist and have data
            for table in set(tables):
                try:
                    cur.execute(f"SELECT COUNT(*) FROM dbo.[{table}]")
                    count = cur.fetchone()[0]
                    
                    if count == 0:
                        print(f"    ❌ {table}: 0 rows (NO DATA)")
                    else:
                        # Get date range if table has Timestamp column
                        try:
                            cur.execute(f"SELECT MIN(Timestamp), MAX(Timestamp) FROM dbo.[{table}] WHERE Timestamp IS NOT NULL")
                            result = cur.fetchone()
                            if result and result[0]:
                                print(f"    ✓ {table}: {count:,} rows | {result[0]} → {result[1]}")
                            else:
                                print(f"    ✓ {table}: {count:,} rows (no timestamp data)")
                        except:
                            print(f"    ✓ {table}: {count:,} rows")
                            
                except Exception as e:
                    print(f"    ❌ {table}: ERROR - {str(e)[:100]}")

sql.close()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}\n")
