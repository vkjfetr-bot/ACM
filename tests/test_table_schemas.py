"""
Tests for core/table_schemas.py - Table Schema Validation

v11.0.0 Phase 1.8 Tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from core.table_schemas import (
    ColumnSpec,
    TableSchema,
    TABLE_SCHEMAS,
    validate_dataframe,
    apply_schema_defaults,
    get_key_columns,
    has_schema,
    list_validated_tables,
    register_schema,
)


# =============================================================================
# ColumnSpec Tests
# =============================================================================

class TestColumnSpec:
    """Tests for ColumnSpec dataclass."""
    
    def test_basic_creation(self):
        """Create basic column spec."""
        spec = ColumnSpec(int, nullable=False)
        
        assert spec.python_type == int
        assert spec.nullable is False
        assert spec.default is None
    
    def test_with_default(self):
        """Column spec with default value."""
        spec = ColumnSpec(str, nullable=True, default="UNKNOWN")
        
        assert spec.default == "UNKNOWN"
    
    def test_nullable_default(self):
        """Nullable defaults to True."""
        spec = ColumnSpec(float)
        
        assert spec.nullable is True


# =============================================================================
# TableSchema Tests
# =============================================================================

class TestTableSchema:
    """Tests for TableSchema dataclass."""
    
    def test_basic_creation(self):
        """Create basic table schema."""
        schema = TableSchema(
            required_columns={
                "ID": ColumnSpec(int, nullable=False),
                "Name": ColumnSpec(str, nullable=False),
            },
            key_columns=["ID"],
        )
        
        assert len(schema.required_columns) == 2
        assert schema.key_columns == ["ID"]
    
    def test_validate_valid_df(self):
        """Validate a valid DataFrame."""
        schema = TableSchema(
            required_columns={
                "RunID": ColumnSpec(int, nullable=False),
                "Value": ColumnSpec(float, nullable=True),
            },
        )
        
        df = pd.DataFrame({
            "RunID": [1, 2, 3],
            "Value": [1.0, 2.0, None],
        })
        
        errors = schema.validate(df, "TestTable")
        
        assert len(errors) == 0
    
    def test_validate_missing_column(self):
        """Detect missing required column."""
        schema = TableSchema(
            required_columns={
                "RunID": ColumnSpec(int, nullable=False),
                "EquipID": ColumnSpec(int, nullable=False),
            },
        )
        
        df = pd.DataFrame({
            "RunID": [1, 2, 3],
            # Missing EquipID
        })
        
        errors = schema.validate(df, "TestTable")
        
        assert len(errors) == 1
        assert "EquipID" in errors[0]
    
    def test_validate_null_in_not_null(self):
        """Detect NULL in NOT NULL column."""
        schema = TableSchema(
            required_columns={
                "RunID": ColumnSpec(int, nullable=False),
            },
        )
        
        df = pd.DataFrame({
            "RunID": [1, None, 3],
        })
        
        errors = schema.validate(df, "TestTable")
        
        assert len(errors) == 1
        assert "NULL" in errors[0]
    
    def test_apply_defaults(self):
        """Apply default values."""
        schema = TableSchema(
            required_columns={
                "RunID": ColumnSpec(int, nullable=False),
            },
            auto_add_columns={
                "ACMVersion": "11.0.0",
                "Source": "ACM",
            },
        )
        
        df = pd.DataFrame({"RunID": [1, 2, 3]})
        
        result = schema.apply_defaults(df)
        
        assert "ACMVersion" in result.columns
        assert result["ACMVersion"].iloc[0] == "11.0.0"
        assert "Source" in result.columns


# =============================================================================
# Validation Function Tests
# =============================================================================

class TestValidateDataframe:
    """Tests for validate_dataframe function."""
    
    def test_validate_known_table(self):
        """Validate against known table schema."""
        df = pd.DataFrame({
            "RunID": [1],
            "EquipID": [1],
            "Timestamp": [datetime.now()],
            "Health": [85.0],
        })
        
        errors = validate_dataframe(df, "ACM_HealthTimeline", raise_on_error=False)
        
        assert len(errors) == 0
    
    def test_validate_unknown_table(self):
        """Unknown table returns empty errors with warning."""
        df = pd.DataFrame({"X": [1]})
        
        errors = validate_dataframe(df, "NonExistentTable", raise_on_error=False)
        
        assert len(errors) == 0  # No schema = no validation
    
    def test_validate_raises_on_error(self):
        """Raise exception when validation fails."""
        df = pd.DataFrame({
            "RunID": [1],
            # Missing EquipID and Timestamp
        })
        
        with pytest.raises(ValueError) as exc_info:
            validate_dataframe(df, "ACM_HealthTimeline", raise_on_error=True)
        
        assert "Schema validation failed" in str(exc_info.value)


class TestApplySchemaDefaults:
    """Tests for apply_schema_defaults function."""
    
    def test_apply_version_default(self):
        """Apply ACMVersion default."""
        df = pd.DataFrame({
            "RunID": [1],
            "EquipID": [1],
            "Timestamp": [datetime.now()],
            "Health": [85.0],
        })
        
        result = apply_schema_defaults(df, "ACM_HealthTimeline")
        
        assert "ACMVersion" in result.columns
    
    def test_unknown_table_unchanged(self):
        """Unknown table returns DataFrame unchanged."""
        df = pd.DataFrame({"X": [1]})
        
        result = apply_schema_defaults(df, "UnknownTable")
        
        assert list(result.columns) == ["X"]


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_key_columns(self):
        """Get key columns for table."""
        keys = get_key_columns("ACM_Scores_Wide")
        
        assert "RunID" in keys
        assert "EquipID" in keys
        assert "Timestamp" in keys
    
    def test_get_key_columns_unknown(self):
        """Unknown table returns empty list."""
        keys = get_key_columns("UnknownTable")
        
        assert keys == []
    
    def test_has_schema(self):
        """Check if schema exists."""
        assert has_schema("ACM_Scores_Wide") is True
        assert has_schema("UnknownTable") is False
    
    def test_list_validated_tables(self):
        """List tables with schemas."""
        tables = list_validated_tables()
        
        assert "ACM_Scores_Wide" in tables
        assert "ACM_HealthTimeline" in tables
        assert len(tables) >= 5


class TestSchemaRegistration:
    """Tests for dynamic schema registration."""
    
    def test_register_new_schema(self):
        """Register a new schema."""
        schema = TableSchema(
            required_columns={
                "ID": ColumnSpec(int, nullable=False),
            },
            key_columns=["ID"],
        )
        
        register_schema("ACM_TestTable", schema)
        
        assert has_schema("ACM_TestTable")
        
        # Cleanup
        del TABLE_SCHEMAS["ACM_TestTable"]


# =============================================================================
# Predefined Schema Tests
# =============================================================================

class TestPredefinedSchemas:
    """Tests for predefined table schemas."""
    
    def test_scores_wide_schema(self):
        """ACM_Scores_Wide schema is valid."""
        schema = TABLE_SCHEMAS["ACM_Scores_Wide"]
        
        assert "RunID" in schema.required_columns
        assert "EquipID" in schema.required_columns
        assert "Timestamp" in schema.required_columns
        assert "ACMVersion" in schema.auto_add_columns
    
    def test_health_timeline_schema(self):
        """ACM_HealthTimeline schema is valid."""
        schema = TABLE_SCHEMAS["ACM_HealthTimeline"]
        
        assert "Health" in schema.required_columns
        assert schema.required_columns["Health"].nullable is False
    
    def test_rul_schema(self):
        """ACM_RUL schema is valid."""
        schema = TABLE_SCHEMAS["ACM_RUL"]
        
        assert "RunID" in schema.required_columns
        assert "RUL_Hours" in schema.optional_columns
        assert "P10_LowerBound" in schema.optional_columns
    
    def test_episodes_schema(self):
        """ACM_Episodes schema is valid."""
        schema = TABLE_SCHEMAS["ACM_Episodes"]
        
        assert "EpisodeID" in schema.required_columns
        assert "StartTime" in schema.required_columns
