"""
Tests for core/pipeline_instrumentation.py - Pipeline Stage Instrumentation

v11.0.0 Phase 1.5 Tests
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from core.pipeline_instrumentation import (
    StageMetrics,
    PipelineRun,
    PipelineTracker,
    PipelineStages,
    stage_timer,
    create_tracker,
    time_operation,
)


# =============================================================================
# StageMetrics Tests
# =============================================================================

class TestStageMetrics:
    """Tests for StageMetrics dataclass."""
    
    def test_basic_creation(self):
        """Create basic stage metrics."""
        metrics = StageMetrics(
            stage="data_load",
            duration_ms=150.5,
            rows_in=1000,
            rows_out=950,
        )
        
        assert metrics.stage == "data_load"
        assert metrics.duration_ms == 150.5
        assert metrics.rows_in == 1000
        assert metrics.rows_out == 950
    
    def test_to_dict(self):
        """Convert to dictionary."""
        metrics = StageMetrics(
            stage="feature_compute",
            duration_ms=200.0,
            rows_out=500,
            features_count=25,
            metadata={"model": "pca"}
        )
        
        d = metrics.to_dict()
        
        assert d["Stage"] == "feature_compute"
        assert d["DurationMs"] == 200.0
        assert d["FeaturesCount"] == 25
    
    def test_default_values(self):
        """Default values are set."""
        metrics = StageMetrics(stage="test", duration_ms=100.0)
        
        assert metrics.rows_in == 0
        assert metrics.rows_out == 0
        assert metrics.features_count == 0
        assert metrics.metadata == {}


# =============================================================================
# PipelineRun Tests
# =============================================================================

class TestPipelineRun:
    """Tests for PipelineRun dataclass."""
    
    def test_basic_creation(self):
        """Create pipeline run."""
        run = PipelineRun(
            run_id=123,
            equip_id=1,
            equipment_name="FD_FAN"
        )
        
        assert run.run_id == 123
        assert run.equipment_name == "FD_FAN"
        assert run.status == "IN_PROGRESS"
    
    def test_add_stage(self):
        """Add stage to run."""
        run = PipelineRun()
        
        run.add_stage(StageMetrics(stage="load", duration_ms=100))
        run.add_stage(StageMetrics(stage="compute", duration_ms=200))
        
        assert len(run.stages) == 2
    
    def test_total_duration(self):
        """Calculate total duration."""
        run = PipelineRun()
        run.add_stage(StageMetrics(stage="load", duration_ms=100))
        run.add_stage(StageMetrics(stage="compute", duration_ms=200))
        run.add_stage(StageMetrics(stage="write", duration_ms=50))
        
        assert run.total_duration_ms == 350.0
    
    def test_total_rows(self):
        """Calculate total rows processed."""
        run = PipelineRun()
        run.add_stage(StageMetrics(stage="load", duration_ms=100, rows_out=1000))
        run.add_stage(StageMetrics(stage="filter", duration_ms=50, rows_out=800))
        
        assert run.total_rows_processed == 1000
    
    def test_mark_complete(self):
        """Mark run as complete."""
        run = PipelineRun()
        run.mark_complete("SUCCESS")
        
        assert run.status == "SUCCESS"
        assert run.end_time is not None
    
    def test_mark_failed(self):
        """Mark run as failed."""
        run = PipelineRun()
        run.mark_failed("Connection error")
        
        assert run.status == "FAILED"
        assert run.error_message == "Connection error"
        assert run.end_time is not None
    
    def test_summary(self):
        """Generate summary."""
        run = PipelineRun(run_id=1, equipment_name="TEST")
        run.add_stage(StageMetrics(stage="load", duration_ms=100))
        run.mark_complete()
        
        summary = run.summary()
        
        assert summary["run_id"] == 1
        assert summary["equipment_name"] == "TEST"
        assert summary["n_stages"] == 1
        assert "stage_breakdown" in summary


# =============================================================================
# stage_timer Tests
# =============================================================================

class TestStageTimer:
    """Tests for stage_timer context manager."""
    
    @patch("core.pipeline_instrumentation.Span")
    @patch("core.pipeline_instrumentation.Metrics")
    def test_basic_timing(self, mock_metrics, mock_span):
        """Basic stage timing."""
        mock_span.return_value.__enter__ = MagicMock()
        mock_span.return_value.__exit__ = MagicMock()
        
        with stage_timer("test_stage", emit_metrics=False, emit_traces=False) as m:
            time.sleep(0.01)  # 10ms
            m["rows_out"] = 500
        
        assert "duration_ms" in m
        assert m["duration_ms"] >= 10  # At least 10ms
        assert m["rows_out"] == 500
    
    @patch("core.pipeline_instrumentation.Span")
    @patch("core.pipeline_instrumentation.Metrics")
    def test_collect_metrics(self, mock_metrics, mock_span):
        """Collect metrics during stage."""
        mock_span.return_value.__enter__ = MagicMock()
        mock_span.return_value.__exit__ = MagicMock()
        
        with stage_timer("data_load", emit_traces=False) as m:
            m["rows_in"] = 1000
            m["rows_out"] = 950
            m["features_count"] = 20
        
        assert m["rows_in"] == 1000
        assert m["rows_out"] == 950
        assert m["features_count"] == 20
    
    @patch("core.pipeline_instrumentation.Span")
    @patch("core.pipeline_instrumentation.Metrics")
    def test_error_handling(self, mock_metrics, mock_span):
        """Handle errors during stage."""
        mock_span.return_value.__enter__ = MagicMock()
        mock_span.return_value.__exit__ = MagicMock()
        
        with pytest.raises(ValueError):
            with stage_timer("error_stage", emit_metrics=False, emit_traces=False) as m:
                raise ValueError("Test error")


# =============================================================================
# PipelineTracker Tests
# =============================================================================

class TestPipelineTracker:
    """Tests for PipelineTracker class."""
    
    @patch("core.pipeline_instrumentation.Span")
    @patch("core.pipeline_instrumentation.Metrics")
    def test_basic_tracking(self, mock_metrics, mock_span):
        """Basic pipeline tracking."""
        mock_span.return_value.__enter__ = MagicMock()
        mock_span.return_value.__exit__ = MagicMock()
        
        tracker = PipelineTracker(
            equipment_name="FD_FAN",
            emit_metrics=False,
            emit_traces=False
        )
        
        with tracker.stage("load") as m:
            m["rows_out"] = 1000
        
        with tracker.stage("compute") as m:
            m["rows_out"] = 1000
            m["features_count"] = 50
        
        run = tracker.finish()
        
        assert len(run.stages) == 2
        assert run.status == "SUCCESS"
    
    @patch("core.pipeline_instrumentation.Span")
    @patch("core.pipeline_instrumentation.Metrics")
    def test_stage_metrics_recorded(self, mock_metrics, mock_span):
        """Stage metrics are recorded."""
        mock_span.return_value.__enter__ = MagicMock()
        mock_span.return_value.__exit__ = MagicMock()
        
        tracker = PipelineTracker(emit_metrics=False, emit_traces=False)
        
        with tracker.stage("data_load") as m:
            m["rows_out"] = 500
        
        stages = tracker.get_stages()
        
        assert len(stages) == 1
        assert stages[0].stage == "data_load"
        assert stages[0].rows_out == 500
    
    @patch("core.pipeline_instrumentation.Span")
    @patch("core.pipeline_instrumentation.Metrics")
    def test_fail_pipeline(self, mock_metrics, mock_span):
        """Mark pipeline as failed."""
        mock_span.return_value.__enter__ = MagicMock()
        mock_span.return_value.__exit__ = MagicMock()
        
        tracker = PipelineTracker(emit_metrics=False, emit_traces=False)
        run = tracker.fail("Database connection failed")
        
        assert run.status == "FAILED"
        assert run.error_message == "Database connection failed"
    
    @patch("core.pipeline_instrumentation.Span")
    @patch("core.pipeline_instrumentation.Metrics")
    def test_get_summary(self, mock_metrics, mock_span):
        """Get run summary."""
        mock_span.return_value.__enter__ = MagicMock()
        mock_span.return_value.__exit__ = MagicMock()
        
        tracker = PipelineTracker(
            equipment_name="TEST",
            run_id=42,
            emit_metrics=False,
            emit_traces=False
        )
        
        with tracker.stage("load") as m:
            m["rows_out"] = 100
        
        tracker.finish()
        summary = tracker.get_summary()
        
        assert summary["run_id"] == 42
        assert summary["equipment_name"] == "TEST"
        assert summary["n_stages"] == 1


# =============================================================================
# PipelineStages Tests
# =============================================================================

class TestPipelineStages:
    """Tests for PipelineStages constants."""
    
    def test_stage_names(self):
        """Stage names are strings."""
        assert PipelineStages.DATA_LOAD == "data_load"
        assert PipelineStages.FEATURE_COMPUTE == "feature_compute"
        assert PipelineStages.REGIME_DETECT == "regime_detect"
    
    def test_detector_stages(self):
        """Detector stage names."""
        assert PipelineStages.DETECTOR_AR1 == "detector_ar1"
        assert PipelineStages.DETECTOR_PCA == "detector_pca"
        assert PipelineStages.DETECTOR_IFOREST == "detector_iforest"
    
    def test_all_stages(self):
        """Get all stage names."""
        all_stages = PipelineStages.all_stages()
        
        assert "data_load" in all_stages
        assert "feature_compute" in all_stages
        assert "sql_write" in all_stages
        assert len(all_stages) > 10


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_tracker(self):
        """Create tracker with factory."""
        tracker = create_tracker(
            equipment_name="FD_FAN",
            equip_id=1,
            run_id=123
        )
        
        assert tracker.equipment_name == "FD_FAN"
        assert tracker.equip_id == 1
        assert tracker.run_id == 123
    
    @patch("core.pipeline_instrumentation.Span")
    @patch("core.pipeline_instrumentation.Metrics")
    def test_time_operation(self, mock_metrics, mock_span):
        """Time individual operation."""
        mock_span.return_value.__enter__ = MagicMock()
        mock_span.return_value.__exit__ = MagicMock()
        
        with time_operation("custom_op", "FD_FAN") as m:
            time.sleep(0.005)
        
        assert "duration_ms" in m


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for pipeline instrumentation."""
    
    @patch("core.pipeline_instrumentation.Span")
    @patch("core.pipeline_instrumentation.Metrics")
    def test_full_pipeline_flow(self, mock_metrics, mock_span):
        """Test complete pipeline flow."""
        mock_span.return_value.__enter__ = MagicMock()
        mock_span.return_value.__exit__ = MagicMock()
        
        tracker = create_tracker(
            equipment_name="GAS_TURBINE",
            equip_id=5,
            run_id=999
        )
        # Disable metrics/traces for testing
        tracker.emit_metrics = False
        tracker.emit_traces = False
        
        # Simulate pipeline stages
        with tracker.stage(PipelineStages.DATA_LOAD) as m:
            m["rows_out"] = 10000
        
        with tracker.stage(PipelineStages.FEATURE_COMPUTE) as m:
            m["rows_in"] = 10000
            m["rows_out"] = 9500
            m["features_count"] = 30
        
        with tracker.stage(PipelineStages.REGIME_DETECT) as m:
            m["rows_in"] = 9500
            m["n_regimes"] = 3
        
        with tracker.stage(PipelineStages.FUSION) as m:
            m["rows_out"] = 9500
        
        with tracker.stage(PipelineStages.SQL_WRITE) as m:
            m["rows_written"] = 9500
        
        run = tracker.finish()
        
        assert run.status == "SUCCESS"
        assert len(run.stages) == 5
        assert run.total_duration_ms > 0
        
        # Check stage breakdown
        summary = tracker.get_summary()
        assert "data_load" in summary["stage_breakdown"]
        assert "sql_write" in summary["stage_breakdown"]
