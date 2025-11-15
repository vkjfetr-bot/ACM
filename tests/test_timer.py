"""Tests for the Timer utility."""
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.timer import Timer


def test_timer_basic():
    """Test basic Timer functionality."""
    T = Timer(enable=True)
    
    with T.section("test_section"):
        time.sleep(0.1)
    
    # Timer should have logged the section
    assert "test_section" in T.totals
    assert T.totals["test_section"] >= 0.1
    print("✓ Basic timer functionality works")


def test_timer_disabled():
    """Test Timer when disabled."""
    T = Timer(enable=False)
    
    with T.section("test_section"):
        time.sleep(0.1)
    
    # Should not track when disabled
    assert len(T.totals) == 0
    print("✓ Timer respects enable=False")


def test_timer_env_var():
    """Test Timer respects ACM_TIMINGS environment variable."""
    old_value = os.environ.get("ACM_TIMINGS")
    
    try:
        # Test with timings disabled
        os.environ["ACM_TIMINGS"] = "0"
        T = Timer()  # Should pick up env var
        
        with T.section("test_section"):
            time.sleep(0.05)
        
        # Should not track when disabled via env
        assert len(T.totals) == 0
        print("✓ Timer respects ACM_TIMINGS environment variable")
    finally:
        if old_value is not None:
            os.environ["ACM_TIMINGS"] = old_value
        elif "ACM_TIMINGS" in os.environ:
            del os.environ["ACM_TIMINGS"]


def test_timer_log():
    """Test Timer log method."""
    T = Timer(enable=True)
    
    # Should not raise exceptions
    T.log("custom_event", count=42, status="ok")
    print("✓ Timer log method works")


def test_timer_wrap_decorator():
    """Test Timer wrap decorator."""
    T = Timer(enable=True)
    
    @T.wrap("decorated_func")
    def test_func():
        time.sleep(0.05)
        return "done"
    
    result = test_func()
    
    assert result == "done"
    assert "decorated_func" in T.totals
    print("✓ Timer wrap decorator works")


def test_timer_json_mode():
    """Test Timer in JSON mode."""
    old_value = os.environ.get("LOG_FORMAT")
    
    try:
        os.environ["LOG_FORMAT"] = "json"
        T = Timer(enable=True)
        
        with T.section("json_section"):
            time.sleep(0.05)
        
        T.log("json_event", status="ok")
        
        # Should have tracked the section
        assert "json_section" in T.totals
        print("✓ Timer JSON mode works")
    finally:
        if old_value is not None:
            os.environ["LOG_FORMAT"] = old_value
        elif "LOG_FORMAT" in os.environ:
            del os.environ["LOG_FORMAT"]


def run_all_tests():
    """Run all tests."""
    print("\n=== Running Timer Tests ===\n")
    
    test_timer_basic()
    test_timer_disabled()
    test_timer_env_var()
    test_timer_log()
    test_timer_wrap_decorator()
    test_timer_json_mode()
    
    print("\n=== All Timer Tests Passed ===\n")


if __name__ == "__main__":
    run_all_tests()
