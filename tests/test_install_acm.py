"""
Tests for install/install_acm.py SQL rewrite behavior.
"""

from pathlib import Path
import sys

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from install import install_acm


def test_rewrite_sql_for_database_replaces_use_and_dbid():
    sql = """
    IF DB_ID(N'ACM') IS NULL
    BEGIN
        CREATE DATABASE [ACM];
    END
    GO
    USE [ACM];
    GO
    """
    rewritten = install_acm.rewrite_sql_for_database(sql, "ACM_TEST")

    assert "DB_ID(N'ACM_TEST')" in rewritten
    assert "CREATE DATABASE [ACM_TEST]" in rewritten
    assert "USE [ACM_TEST]" in rewritten


def test_rewrite_sql_for_database_does_not_touch_table_prefixes():
    sql = """
    USE [ACM];
    GO
    CREATE TABLE dbo.ACM_Scores_Wide (
        Id INT NOT NULL
    );
    """
    rewritten = install_acm.rewrite_sql_for_database(sql, "ACM_TEST")

    assert "USE [ACM_TEST]" in rewritten
    assert "dbo.ACM_Scores_Wide" in rewritten
    assert "dbo.ACM_TEST_Scores_Wide" not in rewritten


def test_rewrite_sql_for_database_handles_empty_database():
    sql = "USE [ACM];"
    rewritten = install_acm.rewrite_sql_for_database(sql, "")
    assert rewritten == sql
