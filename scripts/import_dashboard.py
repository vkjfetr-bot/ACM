#!/usr/bin/env python
"""Import ACM dashboards to Grafana via API."""
import json
import urllib.request
import os
import sys

GRAFANA_URL = "http://localhost:3000"
AUTH_HEADER = "Basic YWRtaW46YWRtaW4="  # admin:admin base64


def import_dashboard(json_path: str, folder_uid: str = "") -> bool:
    """Import a single dashboard from JSON file."""
    print(f"Importing: {os.path.basename(json_path)}")
    
    with open(json_path, encoding="utf-8") as f:
        dash = json.load(f)
    
    dash["id"] = None
    
    payload = {"dashboard": dash, "folderUid": folder_uid, "overwrite": True}
    data = json.dumps(payload).encode("utf-8")
    
    req = urllib.request.Request(
        f"{GRAFANA_URL}/api/dashboards/db",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": AUTH_HEADER},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            print(f"  OK: uid={result.get('uid')}, url={result.get('url')}")
            return True
    except urllib.error.HTTPError as e:
        print(f"  ERROR {e.code}: {e.read().decode('utf-8')}")
        return False


def main():
    dashboard_dir = os.path.join(os.path.dirname(__file__), "..", "grafana_dashboards")
    dashboard_dir = os.path.abspath(dashboard_dir)
    
    json_files = []
    for f in os.listdir(dashboard_dir):
        if f.endswith(".json") and not f.startswith("import_"):
            json_files.append(os.path.join(dashboard_dir, f))
    
    if not json_files:
        print("No dashboard JSON files found!")
        sys.exit(1)
    
    print(f"Found {len(json_files)} dashboards to import\n")
    
    success = 0
    failed = 0
    for json_path in sorted(json_files):
        if import_dashboard(json_path):
            success += 1
        else:
            failed += 1
    
    print(f"\nDone: {success} imported, {failed} failed")


if __name__ == "__main__":
    main()
