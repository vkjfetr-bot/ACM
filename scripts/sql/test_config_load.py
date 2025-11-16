# scripts/sql/test_config_load.py
"""Test loading config from SQL database."""

import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.sql_config import get_equipment_config
import json
from utils.logger import Console


_LOG_PREFIX_HANDLERS = (
    ("✗", Console.error),
    ("[ERROR]", Console.error),
    ("[ERR]", Console.error),
    ("[WARN]", Console.warn),
    ("[WARNING]", Console.warn),
    ("✓", Console.ok),
)


def _log(*args: Any, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False) -> None:
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

_log("="*60)
_log("Testing SQL Config Loading")
_log("="*60)

# Test 1: Load global defaults
_log("\n1. Loading global defaults (EquipID=0)...")
try:
    cfg = get_equipment_config(equipment_code=None, use_sql=True, fallback_to_yaml=False)
    _log(f"✓ Loaded {len(cfg)} top-level config categories")
    _log(f"  Categories: {list(cfg.keys())}")
    
    # Show some sample values
    _log(f"\n  Sample values:")
    _log(f"    features.window = {cfg['features']['window']}")
    _log(f"    fusion.weights.ar1_z = {cfg['fusion']['weights']['ar1_z']}")
    _log(f"    thresholds.q = {cfg['thresholds']['q']}")
    _log(f"    models.pca.n_components = {cfg['models']['pca']['n_components']}")
    
except Exception as e:
    _log(f"✗ Failed: {e}")

# Test 2: Try loading for non-existent equipment (should use global defaults)
_log("\n2. Loading for non-existent equipment (should use global)...")
try:
    cfg = get_equipment_config(equipment_code='TEST_EQUIPMENT_999', use_sql=True, fallback_to_yaml=False)
    _log(f"✓ Loaded config (using global defaults)")
    
except Exception as e:
    _log(f"✗ Failed: {e}")

# Test 3: Show full fusion config
_log("\n3. Full fusion configuration:")
try:
    cfg = get_equipment_config(use_sql=True, fallback_to_yaml=False)
    _log(json.dumps(cfg['fusion'], indent=2))
except Exception as e:
    _log(f"✗ Failed: {e}")

_log("\n" + "="*60)
_log("Test Complete")
_log("="*60)
