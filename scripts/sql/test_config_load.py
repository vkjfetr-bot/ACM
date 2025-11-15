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

print("="*60)
print("Testing SQL Config Loading")
print("="*60)

# Test 1: Load global defaults
print("\n1. Loading global defaults (EquipID=0)...")
try:
    cfg = get_equipment_config(equipment_code=None, use_sql=True, fallback_to_yaml=False)
    print(f"✓ Loaded {len(cfg)} top-level config categories")
    print(f"  Categories: {list(cfg.keys())}")
    
    # Show some sample values
    print(f"\n  Sample values:")
    print(f"    features.window = {cfg['features']['window']}")
    print(f"    fusion.weights.ar1_z = {cfg['fusion']['weights']['ar1_z']}")
    print(f"    thresholds.q = {cfg['thresholds']['q']}")
    print(f"    models.pca.n_components = {cfg['models']['pca']['n_components']}")
    
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Try loading for non-existent equipment (should use global defaults)
print("\n2. Loading for non-existent equipment (should use global)...")
try:
    cfg = get_equipment_config(equipment_code='TEST_EQUIPMENT_999', use_sql=True, fallback_to_yaml=False)
    print(f"✓ Loaded config (using global defaults)")
    
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: Show full fusion config
print("\n3. Full fusion configuration:")
try:
    cfg = get_equipment_config(use_sql=True, fallback_to_yaml=False)
    print(json.dumps(cfg['fusion'], indent=2))
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "="*60)
print("Test Complete")
print("="*60)
