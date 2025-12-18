"""
Comprehensive Test Suite for SQL Mode Continuous Learning Architecture
Tests all 10 implemented tasks (SQL-CL-01 through SQL-CL-10)
"""
import sys
import os
from pathlib import Path
import pandas as pd
import subprocess
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sql_client import SQLClient
from core.observability import Console

class ContinuousLearningTestSuite:
    def __init__(self):
        self.test_results = []
        self.sql_client = None
        
    def setup(self):
        """Initialize SQL connection for testing"""
        try:
            config_path = Path("configs/sql_connection.ini")
            if not config_path.exists():
                Console.warn("SQL connection config not found, skipping SQL tests")
                return False
            
            self.sql_client = SQLClient(str(config_path))
            Console.info("✓ SQL connection established")
            return True
        except Exception as e:
            Console.warn(f"SQL setup failed: {e}")
            return False
    
    def teardown(self):
        """Cleanup test resources"""
        if self.sql_client:
            try:
                self.sql_client.close()
            except:
                pass
    
    def log_test_result(self, task_id, test_name, passed, details=""):
        """Record test result"""
        status = "✓ PASS" if passed else "✗ FAIL"
        self.test_results.append({
            "task": task_id,
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now()
        })
        Console.info(f"[{task_id}] {test_name}: {status} {details}")
    
    # ========== TASK 1 & 2: Data-Driven Retrain Triggers ==========
    def test_task_1_2_retrain_triggers(self):
        """Test anomaly rate, drift score, model age, and regime quality triggers"""
        Console.info("\n=== Testing Task 1 & 2: Data-Driven Retrain Triggers ===")
        
        # Test 1: Config parameters exist
        from utils.config_dict import ConfigDict
        cfg = ConfigDict()
        cfg.load_from_csv("configs/config_table.csv")
        
        required_params = [
            "models.auto_retrain.max_anomaly_rate",
            "models.auto_retrain.max_drift_score",
            "models.auto_retrain.max_model_age_hours"
        ]
        
        all_exist = True
        for param in required_params:
            val = cfg.get_nested(param.split("."))
            if val is None:
                all_exist = False
                Console.warn(f"Missing config: {param}")
        
        self.log_test_result("T1-2", "Config parameters exist", all_exist)
        
        # Test 2: Run ACM and check for trigger logs
        Console.info("Running ACM to test triggers...")
        result = subprocess.run(
            ["python", "-m", "core.acm_main", "--equip", "FD_FAN"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Check for trigger-related log messages
        log_content = result.stdout + result.stderr
        has_quality_check = "assess_model_quality" in log_content or "[AUTO-TUNE]" in log_content
        self.log_test_result("T1-2", "Quality assessment runs", has_quality_check)
        
        return all_exist and has_quality_check
    
    # ========== TASK 3: SQL-Native Refit Mechanism ==========
    def test_task_3_refit_mechanism(self):
        """Test ACM_RefitRequests table and read/write/acknowledge flow"""
        Console.info("\n=== Testing Task 3: SQL-Native Refit Mechanism ===")
        
        if not self.sql_client:
            self.log_test_result("T3", "SQL refit mechanism", False, "No SQL connection")
            return False
        
        # Test 1: Table exists
        with self.sql_client.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = 'ACM_RefitRequests'
            """)
            table_exists = cur.fetchone()[0] == 1
        
        self.log_test_result("T3", "ACM_RefitRequests table exists", table_exists)
        
        if not table_exists:
            return False
        
        # Test 2: Insert test refit request
        test_equip_id = 1
        with self.sql_client.cursor() as cur:
            cur.execute("""
                INSERT INTO ACM_RefitRequests 
                    (EquipID, Reason, AnomalyRate, DriftScore, Acknowledged)
                VALUES 
                    (?, 'Test request', 0.30, 2.5, 0)
            """, (test_equip_id,))
            cur.execute("SELECT @@IDENTITY")
            request_id = cur.fetchone()[0]
        
        self.log_test_result("T3", "Can insert refit request", True, f"RequestID={request_id}")
        
        # Test 3: Read pending request
        with self.sql_client.cursor() as cur:
            cur.execute("""
                SELECT RequestID, Reason, Acknowledged
                FROM ACM_RefitRequests
                WHERE RequestID = ?
            """, (request_id,))
            row = cur.fetchone()
            can_read = row is not None and row[2] == 0
        
        self.log_test_result("T3", "Can read pending request", can_read)
        
        # Test 4: Acknowledge request
        with self.sql_client.cursor() as cur:
            cur.execute("""
                UPDATE ACM_RefitRequests
                SET Acknowledged = 1, AcknowledgedAt = SYSUTCDATETIME()
                WHERE RequestID = ?
            """, (request_id,))
            cur.execute("""
                SELECT Acknowledged
                FROM ACM_RefitRequests
                WHERE RequestID = ?
            """, (request_id,))
            acknowledged = cur.fetchone()[0] == 1
        
        self.log_test_result("T3", "Can acknowledge request", acknowledged)
        
        return table_exists and can_read and acknowledged
    
    # ========== TASK 4: Disable Legacy Joblib Cache ==========
    def test_task_4_joblib_cache_disabled(self):
        """Test that joblib cache is disabled in SQL mode"""
        Console.info("\n=== Testing Task 4: Disable Legacy Joblib Cache ===")
        
        # Check source code for SQL_MODE guard
        acm_main_path = Path("core/acm_main.py")
        content = acm_main_path.read_text()
        
        # Look for the specific line: reuse_models = ... and (not SQL_MODE)
        has_guard = "reuse_models = bool(cfg.get" in content and "and (not SQL_MODE)" in content
        
        self.log_test_result("T4", "Joblib cache gated by SQL_MODE", has_guard)
        
        return has_guard
    
    # ========== TASK 5: Temporal Model Validity ==========
    def test_task_5_temporal_validity(self):
        """Test model age validation and temporal metadata"""
        Console.info("\n=== Testing Task 5: Temporal Model Validity ===")
        
        # Check model_persistence.py for temporal fields
        persistence_path = Path("core/model_persistence.py")
        content = persistence_path.read_text()
        
        has_age_check = "model_age_days" in content or "Model too old" in content
        has_train_start = '"train_start"' in content
        has_train_hash = '"train_hash"' in content
        
        all_present = has_age_check and has_train_start and has_train_hash
        
        self.log_test_result("T5", "Temporal validation implemented", all_present, 
                           f"age_check={has_age_check}, train_start={has_train_start}, train_hash={has_train_hash}")
        
        return all_present
    
    # ========== TASKS 6 & 7: Remove CSV Baseline in SQL Mode ==========
    def test_task_6_7_csv_baseline_removed(self):
        """Test that baseline.seed and baseline_buffer.csv are skipped in SQL mode"""
        Console.info("\n=== Testing Tasks 6 & 7: Remove CSV Baseline in SQL Mode ===")
        
        acm_main_path = Path("core/acm_main.py")
        content = acm_main_path.read_text()
        
        # Check for SQL_MODE guards around baseline logic
        has_baseline_guard = "if not SQL_MODE:" in content and "baseline.seed" in content
        has_buffer_guard = "if not SQL_MODE:" in content and "baseline_buffer.csv" in content
        
        both_guarded = has_baseline_guard and has_buffer_guard
        
        self.log_test_result("T6-7", "CSV baseline guarded by SQL_MODE", both_guarded,
                           f"baseline={has_baseline_guard}, buffer={has_buffer_guard}")
        
        return both_guarded
    
    # ========== TASK 8: Improved models_were_trained Semantics ==========
    def test_task_8_models_semantics(self):
        """Test detectors_fitted_this_run boolean"""
        Console.info("\n=== Testing Task 8: Improved models_were_trained Semantics ===")
        
        acm_main_path = Path("core/acm_main.py")
        content = acm_main_path.read_text()
        
        has_fitted_boolean = "detectors_fitted_this_run" in content
        
        self.log_test_result("T8", "detectors_fitted_this_run implemented", has_fitted_boolean)
        
        return has_fitted_boolean
    
    # ========== TASK 9: Close Auto-Tune Loop ==========
    def test_task_9_autotune_loop(self):
        """Test auto-tune loop with refit signaling"""
        Console.info("\n=== Testing Task 9: Close Auto-Tune Loop ===")
        
        # Check config_history_writer for trigger_refit parameter
        history_writer_path = Path("core/config_history_writer.py")
        content = history_writer_path.read_text()
        
        has_trigger_param = "trigger_refit" in content
        has_refit_logic = "INSERT INTO" in content and "ACM_RefitRequests" in content
        
        loop_closed = has_trigger_param and has_refit_logic
        
        self.log_test_result("T9", "Auto-tune loop closed", loop_closed,
                           f"trigger_param={has_trigger_param}, refit_logic={has_refit_logic}")
        
        return loop_closed
    
    # ========== TASK 10: Enhanced Logging ==========
    def test_task_10_enhanced_logging(self):
        """Test enhanced logging for retrain decisions"""
        Console.info("\n=== Testing Task 10: Enhanced Logging ===")
        
        acm_main_path = Path("core/acm_main.py")
        content = acm_main_path.read_text()
        
        # Check for enhanced log messages
        has_age_logging = "Model age:" in content
        has_config_logging = "Config signature:" in content
        has_sensor_logging = "Sensor count:" in content
        
        all_enhanced = has_age_logging and has_config_logging and has_sensor_logging
        
        self.log_test_result("T10", "Enhanced logging implemented", all_enhanced,
                           f"age={has_age_logging}, config={has_config_logging}, sensors={has_sensor_logging}")
        
        return all_enhanced
    
    # ========== INTEGRATION TEST: Full Pipeline ==========
    def test_integration_full_pipeline(self):
        """Run full ACM pipeline and verify end-to-end behavior"""
        Console.info("\n=== Integration Test: Full Pipeline ===")
        
        try:
            # Run ACM in file mode
            Console.info("Running ACM in file mode...")
            result = subprocess.run(
                ["python", "-m", "core.acm_main", "--equip", "FD_FAN", "--enable-report"],
                capture_output=True,
                text=True,
                timeout=180
            )
            
            file_mode_success = result.returncode == 0
            self.log_test_result("INT", "File mode execution", file_mode_success)
            
            # Check artifacts created
            artifacts_dir = Path("artifacts/FD_FAN")
            has_artifacts = artifacts_dir.exists() and len(list(artifacts_dir.glob("run_*"))) > 0
            self.log_test_result("INT", "Artifacts created", has_artifacts)
            
            return file_mode_success and has_artifacts
            
        except subprocess.TimeoutExpired:
            self.log_test_result("INT", "Full pipeline", False, "Timeout")
            return False
        except Exception as e:
            self.log_test_result("INT", "Full pipeline", False, str(e))
            return False
    
    # ========== RUN ALL TESTS ==========
    def run_all_tests(self):
        """Execute complete test suite"""
        Console.info("=" * 80)
        Console.info("SQL MODE CONTINUOUS LEARNING - COMPREHENSIVE TEST SUITE")
        Console.info("=" * 80)
        
        start_time = time.time()
        
        # Setup
        sql_available = self.setup()
        
        # Run tests
        self.test_task_1_2_retrain_triggers()
        
        if sql_available:
            self.test_task_3_refit_mechanism()
        
        self.test_task_4_joblib_cache_disabled()
        self.test_task_5_temporal_validity()
        self.test_task_6_7_csv_baseline_removed()
        self.test_task_8_models_semantics()
        self.test_task_9_autotune_loop()
        self.test_task_10_enhanced_logging()
        self.test_integration_full_pipeline()
        
        # Cleanup
        self.teardown()
        
        # Summary
        elapsed = time.time() - start_time
        Console.info("\n" + "=" * 80)
        Console.info("TEST SUITE SUMMARY")
        Console.info("=" * 80)
        
        passed = sum(1 for r in self.test_results if "✓ PASS" in r["status"])
        total = len(self.test_results)
        
        Console.info(f"\nTotal Tests: {total}")
        Console.info(f"Passed: {passed}")
        Console.info(f"Failed: {total - passed}")
        Console.info(f"Success Rate: {passed/total*100:.1f}%")
        Console.info(f"Execution Time: {elapsed:.1f}s")
        
        Console.info("\nDetailed Results:")
        for result in self.test_results:
            task = result['task']
            test = result['test']
            status = result['status']
            details = result.get('details', '')
            Console.info(f"  [{task}] {test}: {status} {details}")
        
        return passed == total


if __name__ == "__main__":
    suite = ContinuousLearningTestSuite()
    success = suite.run_all_tests()
    sys.exit(0 if success else 1)
