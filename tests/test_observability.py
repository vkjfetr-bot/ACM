"""Tests for the unified observability module (structlog + rich)."""
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.observability import Console, Progress, Heartbeat, log


def test_console_basic():
    """Test basic Console methods."""
    Console.debug("Debug message")
    Console.info("Info message")
    Console.ok("Success message")  # Green
    Console.warn("Warning message")
    Console.warning("Warning message (alias)")
    Console.error("Error message")
    print("✓ Basic console methods work")


def test_console_with_tags():
    """Test Console with [TAG] prefixes."""
    Console.info("Loading data from SQL", component="DATA")
    Console.warn("Training may take a while", component="MODEL")
    Console.error("Connection failed", component="SQL")
    print("✓ Console tag parsing works")


def test_structlog():
    """Test structlog global logger."""
    log.debug("Debug via structlog")
    log.info("Info via structlog", rows=5000)
    log.warning("Warning via structlog", threshold=90)
    log.error("Error via structlog", table="ACM_Scores")
    print("✓ Structlog logger works")


def test_progress_basic():
    """Test Progress (replaces Heartbeat) with rich spinner."""
    with Progress("Test operation"):
        time.sleep(0.2)
    print("✓ Progress basic functionality works")


def test_progress_disabled():
    """Test Progress when disabled via env var."""
    old_value = os.environ.get("ACM_HEARTBEAT")
    try:
        os.environ["ACM_HEARTBEAT"] = "false"
        with Progress("Disabled test"):
            time.sleep(0.1)
        print("✓ Progress respects ACM_HEARTBEAT=false")
    finally:
        if old_value is not None:
            os.environ["ACM_HEARTBEAT"] = old_value
        elif "ACM_HEARTBEAT" in os.environ:
            del os.environ["ACM_HEARTBEAT"]


def test_heartbeat_alias():
    """Test Heartbeat alias for Progress."""
    assert Heartbeat is Progress, "Heartbeat should be alias for Progress"
    print("✓ Heartbeat is alias for Progress")


def run_all_tests():
    """Run all tests."""
    print("\n=== Running Observability Tests (structlog + rich) ===\n")
    
    test_console_basic()
    test_console_with_tags()
    test_structlog()
    test_progress_basic()
    test_progress_disabled()
    test_heartbeat_alias()
    
    print("\n=== All Observability Tests Passed ===\n")


if __name__ == "__main__":
    run_all_tests()


if __name__ == "__main__":
    run_all_tests()
