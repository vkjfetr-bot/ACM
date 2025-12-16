import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.sql_client import SQLClient
from utils.config_dict import ConfigDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def apply_sql_patch(client, patch_file):
    """Reads and executes a SQL patch file."""
    if not os.path.exists(patch_file):
        logger.error(f"Patch file not found: {patch_file}")
        return False

    logger.info(f"Applying patch: {patch_file}")
    try:
        with open(patch_file, 'r') as f:
            sql_content = f.read()
        
        # Split by GO if present, or execute as one block if simple
        # Simple split by GO on a line by itself
        batches = [b.strip() for b in sql_content.split('\nGO')]
        
        cursor = client.cursor()
        for batch in batches:
            if batch:
                cursor.execute(batch)
                client.conn.commit()
        
        logger.info(f"Successfully applied patch: {patch_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to apply patch {patch_file}: {e}")
        return False

def main():
    # Load config to get SQL connection details
    # Use from_csv to load base config which contains SQL connection info
    cfg = ConfigDict.from_csv("configs/config_table.csv", equip_id=0)
    sql_cfg = cfg.get("sql", {}) or {}
    
    # Initialize SQL Client
    try:
        client = SQLClient(sql_cfg)
        client.connect()
        logger.info("Connected to SQL Server.")
    except Exception as e:
        logger.error(f"Failed to connect to SQL Server: {e}")
        return

    # List of patches to apply in order
    patches = [
        "scripts/sql/patches/2025-11-19_migrate_all_runid_to_uniqueidentifier.sql",
        "scripts/sql/patches/2025-11-19_add_metric_type_to_pca_metrics.sql"
    ]

    for patch in patches:
        apply_sql_patch(client, patch)

    logger.info("All patches processed.")

if __name__ == "__main__":
    main()
