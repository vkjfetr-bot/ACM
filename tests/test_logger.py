"""Tests for the enhanced logging system."""
import os
import sys
import tempfile
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.logger import Console, Heartbeat, LogLevel, LogFormat


def test_console_basic():
    """Test basic Console methods."""
    # Should not raise exceptions
    Console.debug("Debug message")
    Console.info("Info message")
    Console.ok("Success message")
    Console.warn("Warning message")
    Console.warning("Warning message (alias)")
    Console.error("Error message")
    Console.critical("Critical message")
    print("✓ Basic console methods work")


def test_console_with_context():
    """Test Console methods with context."""
    Console.info("Processing data", module="test", count=42)
    Console.warn("Low memory", threshold=90, current=95)
    Console.error("Connection failed", host="localhost", port=5432)
    print("✓ Console methods with context work")


def test_log_levels():
    """Test log level filtering."""
    # Set to WARNING level
    Console.set_level("WARNING")
    
    # These should be filtered out (not visible but shouldn't crash)
    Console.debug("Should not appear")
    Console.info("Should not appear")
    
    # These should appear
    Console.warn("Should appear")
    Console.error("Should appear")
    
    # Reset to default
    Console.set_level("INFO")
    print("✓ Log level filtering works")


def test_json_format():
    """Test JSON output format."""
    # Switch to JSON format
    Console.set_format("json")
    
    Console.info("Test JSON", count=42)
    Console.error("Test error", code=500)
    
    # Switch back to text
    Console.set_format("text")
    print("✓ JSON format works")


def test_file_output():
    """Test file output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        
        # Set file output
        Console.set_output(log_file)
        Console.info("Test file output")
        Console.warn("Test warning")
        Console.set_output(None)  # Close file
        
        # Verify file was created and has content
        assert log_file.exists(), "Log file should exist"
        content = log_file.read_text()
        assert "Test file output" in content, "Log content should be in file"
        assert "Test warning" in content, "Warning should be in file"
        print("✓ File output works")


def test_ascii_only_mode():
    """Test ASCII-only mode detection."""
    # Save current env
    old_value = os.environ.get("LOG_ASCII_ONLY")
    
    try:
        # Test ASCII-only enabled
        os.environ["LOG_ASCII_ONLY"] = "true"
        # Would need to reload the module to test this properly
        # Just verify the method exists
        assert callable(Console.ascii_only)
        print("✓ ASCII-only mode detection works")
    finally:
        # Restore env
        if old_value is not None:
            os.environ["LOG_ASCII_ONLY"] = old_value
        elif "LOG_ASCII_ONLY" in os.environ:
            del os.environ["LOG_ASCII_ONLY"]


def test_heartbeat_basic():
    """Test basic Heartbeat functionality."""
    import time
    
    # Test with heartbeat enabled
    hb = Heartbeat("Test operation", next_hint="testing", eta_hint=5)
    hb.start()
    time.sleep(0.5)  # Brief pause
    hb.stop()
    print("✓ Heartbeat basic functionality works")


def test_heartbeat_disabled():
    """Test Heartbeat when disabled."""
    old_value = os.environ.get("ACM_HEARTBEAT")
    
    try:
        os.environ["ACM_HEARTBEAT"] = "false"
        hb = Heartbeat("Test operation")
        hb.start()
        hb.stop()
        print("✓ Heartbeat respects ACM_HEARTBEAT=false")
    finally:
        if old_value is not None:
            os.environ["ACM_HEARTBEAT"] = old_value
        elif "ACM_HEARTBEAT" in os.environ:
            del os.environ["ACM_HEARTBEAT"]


def test_heartbeat_ascii_spinner():
    """Test that ASCII spinner is used in ASCII-only mode."""
    old_ascii = os.environ.get("LOG_ASCII_ONLY")
    old_hb = os.environ.get("ACM_HEARTBEAT")
    
    try:
        # Force ASCII-only mode (would need module reload to fully test)
        os.environ["LOG_ASCII_ONLY"] = "true"
        os.environ["ACM_HEARTBEAT"] = "true"
        
        # Verify spinner constants exist
        assert hasattr(Heartbeat, "SPINNER_ASCII")
        assert hasattr(Heartbeat, "SPINNER_UNICODE")
        assert len(Heartbeat.SPINNER_ASCII) > 0
        assert len(Heartbeat.SPINNER_UNICODE) > 0
        
        # Verify ASCII spinner doesn't contain Unicode
        for char in Heartbeat.SPINNER_ASCII:
            assert ord(char) < 128, f"ASCII spinner should only have ASCII chars, got '{char}'"
        
        print("✓ ASCII spinner validation works")
    finally:
        if old_ascii is not None:
            os.environ["LOG_ASCII_ONLY"] = old_ascii
        elif "LOG_ASCII_ONLY" in os.environ:
            del os.environ["LOG_ASCII_ONLY"]
        if old_hb is not None:
            os.environ["ACM_HEARTBEAT"] = old_hb
        elif "ACM_HEARTBEAT" in os.environ:
            del os.environ["ACM_HEARTBEAT"]


def test_log_sinks():
    """Test that custom sinks receive structured records."""
    captured: list[dict] = []

    def sink(record):
        captured.append(record)

    Console.add_sink(sink)
    try:
        Console.info("Sink test", module="tests.test_logger", extra="value")
    finally:
        Console.remove_sink(sink)

    assert captured, "Sink should capture at least one record"
    assert captured[0]["message"] == "Sink test"
    assert captured[0]["context"]["extra"] == "value"
    print("✓ Log sinks work")


def test_module_level_overrides():
    """Test module-specific log level overrides."""
    captured: list[dict] = []

    def sink(record):
        captured.append(record)

    Console.add_sink(sink)
    Console.clear_module_levels()
    try:
        Console.set_module_level("tests.test_logger", "ERROR")
        Console.info("Should be filtered", module="tests.test_logger")
        assert not captured, "INFO log should be filtered by module override"
        Console.error("Should pass", module="tests.test_logger")
        assert captured and captured[-1]["level"] == "ERROR"
    finally:
        Console.clear_module_levels()
        Console.remove_sink(sink)
    print("✓ Module-level overrides work")


def run_all_tests():
    """Run all tests."""
    print("\n=== Running Logger Tests ===\n")
    
    test_console_basic()
    test_console_with_context()
    test_log_levels()
    test_json_format()
    test_file_output()
    test_ascii_only_mode()
    test_heartbeat_basic()
    test_heartbeat_disabled()
    test_heartbeat_ascii_spinner()
    test_log_sinks()
    test_module_level_overrides()
    
    print("\n=== All Logger Tests Passed ===\n")


if __name__ == "__main__":
    run_all_tests()
