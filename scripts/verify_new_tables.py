"""Verify newly created OMR tables."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.sql_client import SQLClient
from utils.config_dict import ConfigDict

cfg = ConfigDict.from_csv("configs/config_table.csv", equip_id=1)
client = SQLClient(cfg)
client.connect()
cursor = client.cursor()

tables = ['ACM_OMR_Metrics', 'ACM_OMR_TopContributors', 'ACM_DetectorContributions', 'ACM_OMR_SensorContributions']
print("\nNewly created tables:")
print("-" * 60)
for t in tables:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {t}")
        count = cursor.fetchone()[0]
        status = "✓ READY" if count == 0 else f"✓ HAS {count:,} ROWS"
        print(f"{t:40s} {status}")
    except Exception as e:
        print(f"{t:40s} ✗ ERROR: {e}")

cursor.close()
client.close()
print("\nDone!")
