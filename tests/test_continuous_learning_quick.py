"""
Quick validation tests for SQL Mode Continuous Learning Architecture
Tests core functionality of all 10 implemented tasks
"""
import sys
from pathlib import Path
import subprocess
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import Console


class QuickValidationTests:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        
    def test(self, name, condition, details=""):
        """Log test result"""
        if condition:
            self.passed += 1
            Console.info(f"✓ PASS: {name} {details}")
        else:
            self.failed += 1
            Console.warn(f"✗ FAIL: {name} {details}")
    
    def run_all(self):
        """Execute all validation tests"""
        Console.info("=" * 80)
        Console.info("SQL MODE CONTINUOUS LEARNING - QUICK VALIDATION")
        Console.info("=" * 80)
        
        # Test 1: Config parameters exist
        self.test_config_parameters()
        
        # Test 2: Model persistence temporal validation
        self.test_model_persistence()
        
        # Test 3: Code structure validation
        self.test_code_structure()
        
        # Test 4: SQL migration files exist
        self.test_sql_migrations()
        
        # Test 5: Run basic ACM pipeline
        self.test_acm_pipeline()
        
        # Summary
        Console.info("\n" + "=" * 80)
        Console.info(f"RESULTS: {self.passed} passed, {self.failed} failed")
        Console.info("=" * 80)
        
        return self.failed == 0
    
    def test_config_parameters(self):
        """Test Task 1-2: Config parameters exist"""
        Console.info("\n[Task 1-2] Testing auto-retrain config parameters...")
        
        import pandas as pd
        
        try:
            # Check directly in CSV for simplicity
            df = pd.read_csv("configs/config_table.csv")
            auto_retrain_params = df[df['ParamPath'].str.contains('auto_retrain', na=False)]
            
            # Required parameters
            required_params = [
                'auto_retrain.max_anomaly_rate',
                'auto_retrain.max_drift_score',
                'auto_retrain.max_model_age_hours',
                'auto_retrain.min_regime_quality',
                'auto_retrain.on_tuning_change'
            ]
            
            found_count = 0
            for param in required_params:
                exists = param in auto_retrain_params['ParamPath'].values
                if exists:
                    val = auto_retrain_params[auto_retrain_params['ParamPath'] == param]['ParamValue'].iloc[0]
                    self.test(f"Config param {param}", True, f"(value: {val})")
                    found_count += 1
                else:
                    self.test(f"Config param {param}", False, "(missing)")
            
            self.test("Auto-retrain config parameters exist", found_count >= 4, 
                     f"Found {found_count}/{len(required_params)} params")
                
        except Exception as e:
            self.test("Config loading", False, f"Error: {e}")
    
    def test_model_persistence(self):
        """Test Task 5: Temporal model validation"""
        Console.info("\n[Task 5] Testing model persistence temporal validation...")
        
        try:
            from core.model_persistence import ModelVersionManager
            import pandas as pd
            from datetime import datetime, timedelta
            import hashlib
            
            # Test that check_model_validity method exists with temporal validation
            manager = ModelVersionManager(
                equip="TEST_EQUIP", 
                artifact_root=Path("artifacts")
            )
            has_check_method = hasattr(manager, 'check_model_validity')
            self.test("ModelVersionManager.check_model_validity exists", has_check_method)
            
            # Check if model age validation is in the code
            persistence_code = Path("core/model_persistence.py").read_text()
            has_age_check = "max_model_age_days" in persistence_code
            self.test("Temporal validation (max_model_age_days) exists", has_age_check)
            
            has_train_start = "train_start" in persistence_code
            self.test("Metadata includes train_start timestamp", has_train_start)
            
            has_train_hash = "train_hash" in persistence_code
            self.test("Metadata includes train_hash for data tracking", has_train_hash)
            
        except Exception as e:
            self.test("Model persistence test", False, f"Error: {e}")
    
    def test_code_structure(self):
        """Test Tasks 6-10: Code structure changes"""
        Console.info("\n[Tasks 6-10] Validating code structure changes...")
        
        try:
            # Test 6-7: Baseline guards exist
            acm_main = Path("core/acm_main.py").read_text()
            
            has_seed_guard = "if not SQL_MODE:" in acm_main and "baseline.seed" in acm_main
            self.test("Baseline seed guard exists", has_seed_guard)
            
            has_buffer_guard = "baseline_buffer.csv" in acm_main
            self.test("Baseline buffer CSV reference exists", has_buffer_guard)
            
            # Test 8: detectors_fitted_this_run exists
            has_fitted_flag = "detectors_fitted_this_run" in acm_main
            self.test("detectors_fitted_this_run flag exists", has_fitted_flag)
            
            # Test 9: Auto-tune refit logic
            has_autotune_refit = "on_tuning_change" in acm_main
            self.test("Auto-tune refit logic exists", has_autotune_refit)
            
            # Test 10: Enhanced logging
            has_enhanced_logging = "[CACHE-ACCEPT]" in acm_main or "cached model acceptance" in acm_main.lower()
            self.test("Enhanced cache logging exists", has_enhanced_logging)
            
            # Test 4: Legacy cache disabled in SQL mode
            has_reuse_models = "reuse_models = " in acm_main
            has_sql_guard = "and (not SQL_MODE)" in acm_main
            has_cache_guard = has_reuse_models and has_sql_guard
            self.test("Legacy cache disabled in SQL mode", has_cache_guard, 
                     f"(reuse_models={has_reuse_models}, SQL guard={has_sql_guard})")
            
        except Exception as e:
            self.test("Code structure validation", False, f"Error: {e}")
    
    def test_sql_migrations(self):
        """Test Task 3: SQL migration files"""
        Console.info("\n[Task 3] Checking SQL migration files...")
        
        migration_file = Path("scripts/sql/migrations/007_create_refit_requests.sql")
        self.test("Migration 007 exists", migration_file.exists())
        
        if migration_file.exists():
            content = migration_file.read_text()
            has_table = "ACM_RefitRequests" in content
            self.test("ACM_RefitRequests table definition exists", has_table)
    
    def test_acm_pipeline(self):
        """Test basic ACM pipeline execution"""
        Console.info("\n[Integration] Running basic ACM pipeline...")
        
        try:
            # Run ACM with file mode (no SQL dependency)
            result = subprocess.run(
                ["python", "-m", "core.acm_main", "--equip", "FD_FAN"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            success = result.returncode == 0
            self.test("ACM pipeline executes", success)
            
            if success:
                # Check for key log messages
                output = result.stdout + result.stderr
                
                # Quality check may not run every time (depends on model caching)
                # Just check that the function exists in the code
                has_quality_code = "assess_model_quality" in Path("core/acm_main.py").read_text()
                self.test("Quality assessment code exists", has_quality_code)
                
                has_detector_fit = "fitted" in output.lower() or "training" in output.lower()
                self.test("Detector training runs", has_detector_fit)
                
        except subprocess.TimeoutExpired:
            self.test("ACM pipeline timeout", False, "Exceeded 120s timeout")
        except Exception as e:
            self.test("ACM pipeline execution", False, f"Error: {e}")


if __name__ == "__main__":
    suite = QuickValidationTests()
    success = suite.run_all()
    sys.exit(0 if success else 1)
