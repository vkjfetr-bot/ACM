#!/usr/bin/env python3
"""
Quick test: Load 1 day of data, train ensemble, verify NO UNKNOWN regimes.
"""
import sys
sys.path.insert(0, str(__file__).replace(r"\scripts\test_ensemble_quick.py", ""))

from core.sql_client import SQLClient
from core.data_loader import DataLoader
from utils.config_dict import ConfigDict
import pandas as pd
import numpy as np

def main():
    # Load config
    cfg = ConfigDict.from_csv("configs/config_table.csv")
    
    # Connect to SQL
    sql_client = SQLClient.from_ini("acm")
    sql_client.connect()
    
    # Load 1 day of data for WFA_TURBINE_10
    loader = DataLoader(sql_client=sql_client)
    
    print("\n=== TEST 1: Load 1 day of data ===")
    train, score, meta = loader.load_from_sql(
        equipment_name="WFA_TURBINE_10",
        start_utc=pd.Timestamp("2022-10-09 08:40:00"),
        end_utc=pd.Timestamp("2022-10-10 08:40:00"),
        is_coldstart=True,
        cfg=dict(cfg)
    )
    
    print(f"Train shape: {train.shape}, Score shape: {score.shape}")
    print(f"Train data loaded: {len(train)} rows")
    print(f"Score data loaded: {len(score)} rows")
    
    if len(train) == 0:
        print("ERROR: No training data!")
        return False
    
    print("\n=== TEST 2: Train ensemble regimes ===")
    from core.regimes import RegimeDetector
    
    detector = RegimeDetector(cfg=cfg.get("regimes", {}))
    
    # Fit on train data
    print(f"Fitting ensemble on {len(train)} rows...")
    detector.fit(train)
    print(f"✓ Ensemble fit complete")
    print(f"  - Primary method: {detector.method_}")
    print(f"  - Has fallback model: {hasattr(detector, 'fallback_model_') and detector.fallback_model_ is not None}")
    
    # Predict on score data
    print(f"\nPredicting regimes on {len(score)} rows...")
    labels = detector.predict(score)
    print(f"✓ Regimes predicted")
    
    print("\n=== TEST 3: Check regime distribution ===")
    unique_labels = np.unique(labels)
    print(f"Unique regime labels: {sorted(unique_labels)}")
    
    # Check for UNKNOWN regimes (-1)
    unknown_count = np.sum(labels == -1)
    unknown_pct = 100 * unknown_count / len(labels) if len(labels) > 0 else 0
    
    print(f"UNKNOWN regimes (-1): {unknown_count} / {len(labels)} ({unknown_pct:.1f}%)")
    
    # Distribution
    for lbl in sorted(unique_labels):
        count = np.sum(labels == lbl)
        pct = 100 * count / len(labels)
        label_name = "UNKNOWN" if lbl == -1 else f"regime_{lbl}"
        print(f"  {label_name}: {count} ({pct:.1f}%)")
    
    if unknown_count == 0:
        print("\n✓ SUCCESS: No UNKNOWN regimes found!")
        return True
    else:
        print(f"\n✗ FAILURE: Found {unknown_count} UNKNOWN regimes!")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
