"""
Test script for SQL-44: Verify SQL historian data loading
"""
import sys
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from core.sql_client import SQLClient
from core.output_manager import OutputManager
from utils.logger import Console


_LOG_PREFIX_HANDLERS = (
    ("[ERROR]", Console.error),
    ("[ERR]", Console.error),
    ("[WARN]", Console.warn),
    ("[WARNING]", Console.warn),
    ("[OK]", Console.ok),
    ("[DEBUG]", Console.debug),
)


def _log(*args: Any, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False) -> None:
    """Route _log("") calls through the structured Console logger."""
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message = f"{message}{end}"

    if not message:
        return

    if file is sys.stderr:
        Console.error(message)
        return

    trimmed = message.lstrip()
    for prefix, handler in _LOG_PREFIX_HANDLERS:
        if trimmed.startswith(prefix):
            handler(message)
            return

    Console.info(message)


print = _log

# Configuration
EQUIPMENT = "FD_FAN"
START_TIME = "2012-01-06 00:00:00"  # Use full date range from SQL table
END_TIME = "2012-03-01 00:00:00"    # ~2 months should give us plenty of data

_log("="*70)
_log("SQL-44: Testing SQL Historian Data Loading")
_log("="*70)
_log("")

# Connect to SQL
_log("[SQL] Connecting to ACM database...")
try:
    sql_client = SQLClient.from_ini('acm')
    sql_client.connect()
    _log("  [OK] Connected successfully")
except Exception as e:
    _log(f"  [ERROR] Failed to connect: {e}")
    sys.exit(1)

# Create OutputManager
_log("")
_log("[INIT] Creating OutputManager...")
output_mgr = OutputManager(
    sql_client=sql_client,
    run_id="test-sql-44",
    equip_id=1  # FD_FAN
)
_log("  [OK] OutputManager created")

# Minimal config for data loading
cfg = {
    "data": {
        "timestamp_col": "EntryDateTime",
        "sampling_secs": None,  # Auto-detect
        "allow_resample": True,
        "resample_strict": False,
        "interp_method": "linear",
        "max_fill_ratio": 0.20,
        "cold_start_split_ratio": 0.6,  # 60% train, 40% score
        "min_train_samples": 100,
    },
    "runtime": {
        "max_fill_ratio": 0.20,
    }
}

# Test SQL historian data loading
_log("")
_log(f"[LOAD] Loading data for {EQUIPMENT} from {START_TIME} to {END_TIME}...")
try:
    start_utc = pd.Timestamp(START_TIME)
    end_utc = pd.Timestamp(END_TIME)
    
    train, score, meta = output_mgr.load_data(
        cfg=cfg,
        start_utc=start_utc,
        end_utc=end_utc,
        equipment_name=EQUIPMENT,
        sql_mode=True
    )
    
    _log("")
    _log("="*70)
    _log("SUCCESS! Data loaded from SQL historian")
    _log("="*70)
    _log("")
    _log(f"[RESULT] Train shape: {train.shape}")
    _log(f"[RESULT] Score shape: {score.shape}")
    _log(f"[RESULT] Total rows: {len(train) + len(score)}")
    _log("")
    _log(f"[META] Timestamp column: {meta.timestamp_col}")
    _log(f"[META] Cadence OK: {meta.cadence_ok}")
    _log(f"[META] Kept columns: {len(meta.kept_cols)}")
    _log(f"[META] Dropped columns: {len(meta.dropped_cols)}")
    _log(f"[META] Sampling seconds: {meta.sampling_seconds}")
    _log("")
    _log("Train columns:", list(train.columns[:5]), "..." if len(train.columns) > 5 else "")
    _log("Score columns:", list(score.columns[:5]), "..." if len(score.columns) > 5 else "")
    _log("")
    _log("Train index range:", train.index.min(), "to", train.index.max())
    _log("Score index range:", score.index.min(), "to", score.index.max())
    _log("")
    _log("="*70)
    _log("✓ SQL-44 implementation validated!")
    _log("="*70)
    
except Exception as e:
    _log("")
    _log("="*70)
    _log(f"[ERROR] Failed to load data: {e}")
    _log("="*70)
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    sql_client.close()
