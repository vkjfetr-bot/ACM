"""
Test script for SQL ModelRegistry end-to-end validation (SQL-23)

This script validates the complete model persistence cycle:
1. Save models to SQL ModelRegistry
2. Load models from SQL ModelRegistry
3. Verify model integrity and predictions match

Usage:
    python scripts/test_model_registry.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from core.model_persistence import ModelVersionManager
from core.sql_client import SQLClient
from utils.config_dict import ConfigDict
from core.observability import Console
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture


def create_test_models():
    """Create simple test models for validation."""
    np.random.seed(42)
    
    # Create synthetic training data
    n_samples = 100
    n_features = 5
    X_train = np.random.randn(n_samples, n_features)
    
    # Train simple models
    pca_model = PCA(n_components=3)
    pca_model.fit(X_train)
    
    iforest_model = IsolationForest(contamination=0.01, random_state=42)
    iforest_model.fit(X_train)
    
    gmm_model = GaussianMixture(n_components=2, random_state=42)
    gmm_model.fit(X_train)
    
    # AR1 parameters (synthetic)
    ar1_params = {
        "phimap": {f"sensor_{i}": 0.5 + i*0.1 for i in range(n_features)},
        "sdmap": {f"sensor_{i}": 1.0 + i*0.05 for i in range(n_features)}
    }
    
    # NOTE: mhal_params removed v9.1.0 - MHAL detector deprecated (redundant with PCA-T2)
    
    models = {
        "pca_model": pca_model,
        "iforest_model": iforest_model,
        "gmm_model": gmm_model,
        "ar1_params": ar1_params,
        "feature_medians": pd.Series(np.random.randn(n_features), index=[f"feature_{i}" for i in range(n_features)])
    }
    
    return models, X_train


def test_model_save_load(sql_client, equip_id):
    """Test save and load cycle."""
    Console.info("\n" + "="*80)
    Console.info("TEST 1: Model Save/Load Cycle")
    Console.info("="*80 + "\n")
    
    # Create test models
    Console.info("Creating synthetic test models...", component="TEST")
    models, X_train = create_test_models()
    
    # Create metadata
    metadata = {
        "config_signature": "test_signature_12345",
        "train_rows": len(X_train),
        "train_sensors": [f"sensor_{i}" for i in range(X_train.shape[1])],
        "created_at": pd.Timestamp.now().isoformat(),
        "models": {
            "pca": {"n_components": 3, "variance_ratio_sum": 0.85},
            "iforest": {"n_estimators": 100, "contamination": 0.01},
            "gmm": {"n_components": 2, "covariance_type": "full"}
        }
    }
    
    # Save models
    Console.info("Saving models to SQL ModelRegistry...", component="TEST")
    model_manager = ModelVersionManager(
        equip="TEST_EQUIP",
        artifact_root=Path("artifacts/TEST_EQUIP"),
        sql_client=sql_client,
        equip_id=equip_id,
        sql_only_mode=True
    )
    
    try:
        version = model_manager.save_models(models=models, metadata=metadata, version=999)
        Console.info(f"✓ Saved models to version {version}", component="TEST")
    except Exception as e:
        Console.error(f"✗ Failed to save models: {e}", component="TEST")
        import traceback
        traceback.print_exc()
        return False
    
    # Load models
    Console.info("\n[TEST] Loading models from SQL ModelRegistry...")
    try:
        loaded_models, loaded_manifest = model_manager.load_models(version=999)
        
        if not loaded_models:
            Console.error("✗ Failed to load models - got None", component="TEST")
            return False
        
        Console.info(f"✓ Loaded {len(loaded_models)} models", component="TEST")
        Console.info(f"Loaded model types: {list(loaded_models.keys())}", component="TEST")
        
    except Exception as e:
        Console.error(f"✗ Failed to load models: {e}", component="TEST")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify models
    Console.info("\n[TEST] Verifying model integrity...")
    success = True
    
    # Test PCA predictions
    if "pca_model" in loaded_models:
        try:
            X_test = np.random.randn(10, X_train.shape[1])
            original_transform = models["pca_model"].transform(X_test)
            loaded_transform = loaded_models["pca_model"].transform(X_test)
            
            if np.allclose(original_transform, loaded_transform):
                Console.info("✓ PCA predictions match", component="TEST")
            else:
                Console.error("✗ PCA predictions differ", component="TEST")
                success = False
        except Exception as e:
            Console.error(f"✗ PCA test failed: {e}", component="TEST")
            success = False
    
    # Test IForest predictions
    if "iforest_model" in loaded_models:
        try:
            X_test = np.random.randn(10, X_train.shape[1])
            original_scores = models["iforest_model"].decision_function(X_test)
            loaded_scores = loaded_models["iforest_model"].decision_function(X_test)
            
            if np.allclose(original_scores, loaded_scores):
                Console.info("✓ IForest predictions match", component="TEST")
            else:
                Console.error("✗ IForest predictions differ", component="TEST")
                success = False
        except Exception as e:
            Console.error(f"✗ IForest test failed: {e}", component="TEST")
            success = False
    
    # Test GMM predictions
    if "gmm_model" in loaded_models:
        try:
            X_test = np.random.randn(10, X_train.shape[1])
            original_probs = models["gmm_model"].predict_proba(X_test)
            loaded_probs = loaded_models["gmm_model"].predict_proba(X_test)
            
            if np.allclose(original_probs, loaded_probs):
                Console.info("✓ GMM predictions match", component="TEST")
            else:
                Console.error("✗ GMM predictions differ", component="TEST")
                success = False
        except Exception as e:
            Console.error(f"✗ GMM test failed: {e}", component="TEST")
            success = False
    
    # Verify AR1 params
    if "ar1_params" in loaded_models:
        try:
            orig_phi = models["ar1_params"]["phimap"]
            loaded_phi = loaded_models["ar1_params"]["phimap"]
            if orig_phi == loaded_phi:
                Console.info("✓ AR1 parameters match", component="TEST")
            else:
                Console.error("✗ AR1 parameters differ", component="TEST")
                success = False
        except Exception as e:
            Console.error(f"✗ AR1 test failed: {e}", component="TEST")
            success = False
    
    return success


def test_manifest_reconstruction(sql_client, equip_id):
    """Test manifest reconstruction from SQL metadata."""
    Console.info("\n" + "="*80)
    Console.info("TEST 2: Manifest Reconstruction")
    Console.info("="*80 + "\n")
    
    model_manager = ModelVersionManager(
        equip="TEST_EQUIP",
        artifact_root=Path("artifacts/TEST_EQUIP"),
        sql_client=sql_client,
        equip_id=equip_id,
        sql_only_mode=True
    )
    
    try:
        loaded_models, loaded_manifest = model_manager.load_models(version=999)
        
        if not loaded_manifest:
            Console.error("✗ Failed to reconstruct manifest", component="TEST")
            return False
        
        Console.info("[TEST] Manifest keys: " + ", ".join(loaded_manifest.keys()))
        
        # Verify expected keys
        expected_keys = ["version", "source", "equip", "config_signature", "train_rows"]
        missing_keys = [k for k in expected_keys if k not in loaded_manifest]
        
        if missing_keys:
            Console.warn(f"Missing expected keys: {missing_keys}", component="TEST")
        
        Console.info(f"✓ Manifest reconstructed with {len(loaded_manifest)} keys", component="TEST")
        Console.info(f"- Version: {loaded_manifest.get('version')}", component="TEST")
        Console.info(f"- Source: {loaded_manifest.get('source')}", component="TEST")
        Console.info(f"- Train rows: {loaded_manifest.get('train_rows')}", component="TEST")
        Console.info(f"- Config signature: {loaded_manifest.get('config_signature')}", component="TEST")
        
        return True
        
    except Exception as e:
        Console.error(f"✗ Manifest reconstruction failed: {e}", component="TEST")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_data(sql_client, equip_id):
    """Clean up test data from ModelRegistry."""
    Console.info("\n[TEST] Cleaning up test data...")
    try:
        cursor = sql_client.conn.cursor()
        cursor.execute("DELETE FROM ModelRegistry WHERE EquipID = ? AND Version = 999", (equip_id,))
        sql_client.conn.commit()
        deleted = cursor.rowcount
        Console.info(f"✓ Deleted {deleted} test model records", component="TEST")
    except Exception as e:
        Console.warn(f"Failed to clean up: {e}", component="TEST")


def main():
    """Run all tests."""
    Console.info("="*80)
    Console.info("SQL ModelRegistry End-to-End Test (SQL-23)")
    Console.info("="*80 + "\n")
    
    # Load config
    try:
        cfg = ConfigDict.from_csv("configs/config_table.csv", equip_id=1)  # Use FD_FAN's equip_id
        Console.info("✓ Loaded configuration", component="TEST")
    except Exception as e:
        Console.error(f"✗ Failed to load config: {e}", component="TEST")
        import traceback
        traceback.print_exc()
        return 1
    
    # Connect to SQL
    try:
        sql_client = SQLClient(cfg, db_section="acm")
        sql_client.connect()
        Console.info("✓ Connected to SQL Server", component="TEST")
    except Exception as e:
        Console.error(f"✗ Failed to connect to SQL: {e}", component="TEST")
        import traceback
        traceback.print_exc()
        return 1
    
    # Get equipment ID for TEST_EQUIP (use FD_FAN's ID for testing)
    try:
        cursor = sql_client.conn.cursor()
        cursor.execute("SELECT EquipID FROM Equipment WHERE EquipCode = 'FD_FAN'")
        row = cursor.fetchone()
        if row:
            equip_id = row[0]
            Console.info(f"✓ Using EquipID={equip_id} for testing", component="TEST")
        else:
            Console.error("✗ Failed to get EquipID for FD_FAN", component="TEST")
            return 1
    except Exception as e:
        Console.error(f"✗ Failed to query EquipID: {e}", component="TEST")
        return 1
    
    # Run tests
    all_success = True
    
    try:
        # Test 1: Save/Load cycle
        if not test_model_save_load(sql_client, equip_id):
            all_success = False
        
        # Test 2: Manifest reconstruction
        if not test_manifest_reconstruction(sql_client, equip_id):
            all_success = False
        
    finally:
        # Cleanup
        cleanup_test_data(sql_client, equip_id)
        if sql_client and sql_client.conn:
            sql_client.conn.close()
    
    # Summary
    Console.info("\n" + "="*80)
    if all_success:
        Console.info("✓ ALL TESTS PASSED")
        Console.info("="*80 + "\n")
        return 0
    else:
        Console.error("✗ SOME TESTS FAILED")
        Console.info("="*80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
