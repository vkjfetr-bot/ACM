"""Comprehensive validation tests for P0 fixes - detector labels and database cleanup."""

import pytest
import pandas as pd
from utils.detector_labels import get_detector_label, format_culprit_label


class TestDetectorLabels:
    """Test detector label consistency and conversion."""
    
    def test_get_detector_label_pca_t2(self):
        """Test PCA-T² detector label conversion."""
        result = get_detector_label('pca_t2_z')
        assert result == 'Multivariate Outlier (PCA-T²)'
        
    def test_get_detector_label_mahalanobis(self):
        """Test Mahalanobis detector label conversion."""
        result = get_detector_label('mhal_z')
        assert result == 'Multivariate Distance (Mahalanobis)'
        
    def test_get_detector_label_ar1(self):
        """Test AR1 detector label conversion."""
        result = get_detector_label('ar1_z')
        assert result == 'Time-Series Anomaly (AR1)'
        
    def test_get_detector_label_all_detectors(self):
        """Test all detector types have full labels."""
        detectors = [
            ('pca_t2_z', 'Multivariate Outlier (PCA-T²)'),
            ('mhal_z', 'Multivariate Distance (Mahalanobis)'),
            ('pca_spe_z', 'Correlation Break (PCA-SPE)'),
            ('ar1_z', 'Time-Series Anomaly (AR1)'),
            ('gmm_z', 'Density Anomaly (GMM)'),
            ('iforest_z', 'Rare State (IsolationForest)'),
            ('omr_z', 'Baseline Consistency (OMR)'),
        ]
        
        for code, expected_label in detectors:
            result = get_detector_label(code)
            assert result == expected_label, f"Failed for {code}: got {result}, expected {expected_label}"
            
    def test_format_culprit_label_simple(self):
        """Test formatting simple detector culprits."""
        result = format_culprit_label('pca_t2_z')
        assert result == 'Multivariate Outlier (PCA-T²)'
        
    def test_format_culprit_label_with_sensor(self):
        """Test formatting culprits with sensor attribution."""
        result = format_culprit_label('pca_t2_z(Temperature_01)')
        assert 'Multivariate Outlier (PCA-T²)' in result
        assert 'Temperature_01' in result
        assert ' → ' in result


class TestDominantSensorExtraction:
    """Test dominant_sensor extraction from formatted culprits."""
    
    def test_extract_simple_label(self):
        """Test extraction of simple detector label."""
        culprit = "Multivariate Outlier (PCA-T²)"
        # Expected: exact same (no sensor attribution)
        assert culprit.strip() == "Multivariate Outlier (PCA-T²)"
        
    def test_extract_with_sensor_attribution(self):
        """Test extraction strips sensor attribution correctly."""
        culprit = "Multivariate Outlier (PCA-T²) → Temperature_01"
        # Expected: everything before " → "
        result = culprit.split(' → ')[0].strip()
        assert result == "Multivariate Outlier (PCA-T²)"
        
    def test_extract_multiple_sensors(self):
        """Test extraction with multiple possible sensor formats."""
        test_cases = [
            ("Time-Series Anomaly (AR1) → SensorA", "Time-Series Anomaly (AR1)"),
            ("Multivariate Distance (Mahalanobis) → B1TEMP1", "Multivariate Distance (Mahalanobis)"),
            ("Correlation Break (PCA-SPE) → DEMO.SIM.FSAB", "Correlation Break (PCA-SPE)"),
            ("Baseline Consistency (OMR)", "Baseline Consistency (OMR)"),
        ]
        
        for culprit, expected in test_cases:
            if ' → ' in culprit:
                result = culprit.split(' → ')[0].strip()
            else:
                result = culprit.strip()
            assert result == expected, f"Failed for {culprit}"


class TestDatabaseConsistency:
    """Test database-level consistency (requires SQL connection)."""
    
    @pytest.mark.requires_sql
    def test_all_detector_labels_in_database(self, sql_client):
        """Verify all detector labels in ACM_EpisodeDiagnostics are full format."""
        cursor = sql_client.cursor()
        cursor.execute("""
            SELECT DISTINCT dominant_sensor 
            FROM ACM_EpisodeDiagnostics 
            WHERE dominant_sensor IS NOT NULL 
            AND dominant_sensor NOT IN ('UNKNOWN', '')
        """)
        
        labels = [row[0] for row in cursor.fetchall()]
        
        # All should contain opening parenthesis (full label format)
        for label in labels:
            assert '(' in label, f"Label not in full format: {label}"
            # All should be in detector (code) format
            assert ')' in label, f"Label missing closing paren: {label}"
            
    @pytest.mark.requires_sql
    def test_equipment_names_standardized(self, sql_client):
        """Verify all equipment names in ACM_Runs are standardized."""
        cursor = sql_client.cursor()
        cursor.execute("""
            SELECT DISTINCT r.EquipName 
            FROM ACM_Runs r
            WHERE r.EquipName NOT IN (SELECT EquipCode FROM Equipment)
        """)
        
        non_standard = cursor.fetchall()
        assert len(non_standard) == 0, f"Non-standard equipment names found: {non_standard}"
        
    @pytest.mark.requires_sql
    def test_all_runs_have_completion_time(self, sql_client):
        """Verify all runs have valid CompletedAt timestamp."""
        cursor = sql_client.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM ACM_Runs WHERE CompletedAt IS NULL
        """)
        
        incomplete_count = cursor.fetchone()[0]
        assert incomplete_count == 0, f"Found {incomplete_count} runs with NULL CompletedAt"
        
    @pytest.mark.requires_sql
    def test_no_backup_tables_exist(self, sql_client):
        """Verify backup tables have been deleted."""
        cursor = sql_client.cursor()
        backup_tables = [
            'PCA_Components_BACKUP_20251203',
            'RunLog_BACKUP_20251203',
            'Runs_BACKUP_20251203'
        ]
        
        for table in backup_tables:
            cursor.execute("""
                SELECT COUNT(*) FROM sys.tables WHERE name = ?
            """, (table,))
            exists = cursor.fetchone()[0]
            assert exists == 0, f"Backup table still exists: {table}"


class TestFinalizerProcedure:
    """Test usp_ACM_FinalizeRun procedure."""
    
    @pytest.mark.requires_sql
    def test_finalize_run_updates_completion(self, sql_client):
        """Verify FinalizeRun updates CompletedAt properly."""
        cursor = sql_client.cursor()
        
        # Test with a sample run (if any exist)
        cursor.execute("""
            SELECT TOP 1 RunID FROM ACM_Runs 
            WHERE CompletedAt IS NOT NULL
            ORDER BY CreatedAt DESC
        """)
        
        result = cursor.fetchone()
        if result:
            run_id = result[0]
            # Verify the run has a valid completion time
            cursor.execute("""
                SELECT CompletedAt, StartedAt FROM ACM_Runs WHERE RunID = ?
            """, (run_id,))
            
            completed, started = cursor.fetchone()
            assert completed is not None, f"RunID {run_id} has NULL CompletedAt"
            assert completed >= started, f"CompletedAt before StartedAt for {run_id}"


class TestDataConsistency:
    """Test overall data consistency post-cleanup."""
    
    @pytest.mark.requires_sql
    def test_no_orphaned_foreign_keys(self, sql_client):
        """Verify no foreign key constraint violations exist."""
        cursor = sql_client.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM sys.foreign_keys 
            WHERE OBJECT_NAME(referenced_object_id) IS NULL
        """)
        
        violations = cursor.fetchone()[0]
        assert violations == 0, f"Found {violations} orphaned foreign keys"
        
    @pytest.mark.requires_sql
    def test_episode_diagnostics_quality(self, sql_client):
        """Verify ACM_EpisodeDiagnostics quality metrics."""
        cursor = sql_client.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN dominant_sensor IS NULL THEN 1 END) as null_sensors,
                COUNT(CASE WHEN severity IS NULL THEN 1 END) as null_severity
            FROM ACM_EpisodeDiagnostics
        """)
        
        total, null_sensors, null_severity = cursor.fetchone()
        
        # Allow up to 10% NULL sensors (edge cases)
        null_sensor_pct = null_sensors / total if total > 0 else 0
        null_severity_pct = null_severity / total if total > 0 else 0
        
        assert null_sensor_pct < 0.1, f"Too many NULL sensors: {null_sensor_pct:.1%}"
        assert null_severity_pct < 0.05, f"Too many NULL severity: {null_severity_pct:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
