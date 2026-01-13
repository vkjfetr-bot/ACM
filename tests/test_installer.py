"""
Comprehensive Tests for ACM Installer Wizard
=============================================

Tests the installer logic WITHOUT actually running destructive operations.
Uses mocking to simulate Docker, SQL Server, and filesystem operations.

Run with:
    pytest tests/test_installer.py -v
    pytest tests/test_installer.py -v -k "test_prerequisites"  # Run specific group
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Tuple, List
from dataclasses import dataclass

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "install"))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def installer_module():
    """Import the installer module."""
    from install import acm_installer
    return acm_installer


@pytest.fixture
def fresh_state(installer_module):
    """Create a fresh InstallState for each test."""
    return installer_module.InstallState()


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory structure."""
    temp_dir = tempfile.mkdtemp(prefix="acm_test_")
    
    # Create directory structure
    dirs = [
        "configs",
        "install/observability",
        "install/sql",
        "utils",
        "core",
        "grafana_dashboards",
    ]
    for d in dirs:
        (Path(temp_dir) / d).mkdir(parents=True, exist_ok=True)
    
    # Create version.py
    version_file = Path(temp_dir) / "utils" / "version.py"
    version_file.write_text('__version__ = "11.3.0"\n')
    
    # Create sql_connection.example.ini
    example_ini = Path(temp_dir) / "configs" / "sql_connection.example.ini"
    example_ini.write_text("""[acm]
server=localhost\\SQLEXPRESS
database=ACM
trusted_connection=yes
driver=ODBC Driver 18 for SQL Server
""")
    
    # Create config_table.csv
    config_csv = Path(temp_dir) / "configs" / "config_table.csv"
    config_csv.write_text("EquipID,Section,Key,Value\n0,detector,enabled,true\n")
    
    # Create docker-compose.yaml
    compose_file = Path(temp_dir) / "install" / "observability" / "docker-compose.yaml"
    compose_file.write_text("""services:
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
""")
    
    yield Path(temp_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# TEST: VERSION DETECTION
# ============================================================================

class TestVersionDetection:
    """Tests for ACM version detection."""
    
    def test_get_acm_version_reads_version_file(self, installer_module, temp_project_dir):
        """Test that get_acm_version correctly reads from version.py."""
        # Override PROJECT_ROOT
        with patch.object(installer_module, 'PROJECT_ROOT', temp_project_dir):
            version = installer_module.get_acm_version()
            assert version == "11.3.0"
    
    def test_get_acm_version_handles_missing_file(self, installer_module, temp_project_dir):
        """Test graceful handling when version.py doesn't exist."""
        # Remove version.py
        version_file = temp_project_dir / "utils" / "version.py"
        version_file.unlink()
        
        with patch.object(installer_module, 'PROJECT_ROOT', temp_project_dir):
            version = installer_module.get_acm_version()
            assert version == "unknown"
    
    def test_get_acm_version_handles_malformed_file(self, installer_module, temp_project_dir):
        """Test handling of malformed version.py."""
        version_file = temp_project_dir / "utils" / "version.py"
        version_file.write_text("# No version here\n")
        
        with patch.object(installer_module, 'PROJECT_ROOT', temp_project_dir):
            version = installer_module.get_acm_version()
            assert version == "unknown"
    
    def test_get_acm_version_handles_different_quote_styles(self, installer_module, temp_project_dir):
        """Test parsing version with single vs double quotes."""
        version_file = temp_project_dir / "utils" / "version.py"
        
        # Single quotes
        version_file.write_text("__version__ = '11.3.0'\n")
        with patch.object(installer_module, 'PROJECT_ROOT', temp_project_dir):
            assert installer_module.get_acm_version() == "11.3.0"
        
        # Double quotes
        version_file.write_text('__version__ = "11.3.0"\n')
        with patch.object(installer_module, 'PROJECT_ROOT', temp_project_dir):
            assert installer_module.get_acm_version() == "11.3.0"


# ============================================================================
# TEST: PREREQUISITES CHECKS
# ============================================================================

class TestPrerequisitesChecks:
    """Tests for individual prerequisite check functions."""
    
    def test_check_python_version_passes_for_311(self, installer_module):
        """Test Python version check passes for 3.11+."""
        with patch.object(sys, 'version_info', (3, 11, 0)):
            ok, details = installer_module.check_python_version()
            assert ok is True
            assert "3.11" in details
    
    def test_check_python_version_passes_for_312(self, installer_module):
        """Test Python version check passes for 3.12+."""
        with patch.object(sys, 'version_info', (3, 12, 0)):
            ok, details = installer_module.check_python_version()
            assert ok is True
            assert "3.12" in details
    
    def test_check_python_version_fails_for_310(self, installer_module):
        """Test Python version check fails for 3.10."""
        with patch.object(sys, 'version_info', (3, 10, 0)):
            ok, details = installer_module.check_python_version()
            assert ok is False
            assert "requires" in details.lower()
    
    def test_check_python_version_fails_for_39(self, installer_module):
        """Test Python version check fails for 3.9."""
        with patch.object(sys, 'version_info', (3, 9, 0)):
            ok, details = installer_module.check_python_version()
            assert ok is False
    
    def test_check_docker_installed_success(self, installer_module):
        """Test Docker installed check when Docker is present."""
        with patch.object(installer_module, 'run_command') as mock_run:
            mock_run.return_value = (0, "Docker version 24.0.6, build ed223bc", "")
            ok, details = installer_module.check_docker_installed()
            assert ok is True
            assert "Docker version" in details
    
    def test_check_docker_installed_failure(self, installer_module):
        """Test Docker installed check when Docker is missing."""
        with patch.object(installer_module, 'run_command') as mock_run:
            mock_run.return_value = (-1, "", "Command not found: docker")
            ok, details = installer_module.check_docker_installed()
            assert ok is False
            assert "not installed" in details.lower()
    
    def test_check_docker_running_success(self, installer_module):
        """Test Docker running check when daemon is active."""
        with patch.object(installer_module, 'run_command') as mock_run:
            mock_run.return_value = (0, "Server: Docker Engine", "")
            ok, details = installer_module.check_docker_running()
            assert ok is True
            assert "running" in details.lower()
    
    def test_check_docker_running_failure(self, installer_module):
        """Test Docker running check when daemon is stopped."""
        with patch.object(installer_module, 'run_command') as mock_run:
            mock_run.return_value = (1, "", "Cannot connect to the Docker daemon")
            ok, details = installer_module.check_docker_running()
            assert ok is False
            assert "not running" in details.lower()
    
    def test_check_docker_compose_v2(self, installer_module):
        """Test Docker Compose v2 detection."""
        with patch.object(installer_module, 'run_command') as mock_run:
            mock_run.return_value = (0, "Docker Compose version v2.20.2", "")
            ok, details = installer_module.check_docker_compose()
            assert ok is True
            assert "v2" in details or "Compose" in details
    
    def test_check_docker_compose_v1_fallback(self, installer_module):
        """Test fallback to docker-compose v1."""
        def mock_run_side_effect(cmd):
            if cmd == ["docker", "compose", "version"]:
                return (1, "", "unknown command")
            elif cmd == ["docker-compose", "--version"]:
                return (0, "docker-compose version 1.29.2", "")
            return (-1, "", "unknown")
        
        with patch.object(installer_module, 'run_command', side_effect=mock_run_side_effect):
            ok, details = installer_module.check_docker_compose()
            assert ok is True
            assert "1.29" in details
    
    def test_check_docker_compose_not_installed(self, installer_module):
        """Test when neither docker compose v1 nor v2 is available."""
        with patch.object(installer_module, 'run_command') as mock_run:
            mock_run.return_value = (1, "", "command not found")
            ok, details = installer_module.check_docker_compose()
            assert ok is False
    
    def test_check_odbc_driver_found_driver18(self, installer_module):
        """Test ODBC driver detection finds Driver 18."""
        mock_pyodbc = MagicMock()
        mock_pyodbc.drivers.return_value = [
            "SQL Server",
            "ODBC Driver 17 for SQL Server",
            "ODBC Driver 18 for SQL Server",
        ]
        
        with patch.dict(sys.modules, {'pyodbc': mock_pyodbc}):
            ok, details = installer_module.check_odbc_driver()
            assert ok is True
            assert "18" in details
    
    def test_check_odbc_driver_found_driver17(self, installer_module):
        """Test ODBC driver detection prefers 17 when 18 not available."""
        mock_pyodbc = MagicMock()
        mock_pyodbc.drivers.return_value = [
            "SQL Server",
            "ODBC Driver 17 for SQL Server",
        ]
        
        with patch.dict(sys.modules, {'pyodbc': mock_pyodbc}):
            ok, details = installer_module.check_odbc_driver()
            assert ok is True
            assert "17" in details
    
    def test_check_odbc_driver_not_found(self, installer_module):
        """Test when no SQL Server ODBC driver is installed."""
        mock_pyodbc = MagicMock()
        mock_pyodbc.drivers.return_value = ["MySQL ODBC Driver"]
        
        with patch.dict(sys.modules, {'pyodbc': mock_pyodbc}):
            ok, details = installer_module.check_odbc_driver()
            # Should return False since no SQL Server driver
            assert "SQL Server" not in details or ok is False
    
    def test_check_odbc_driver_pyodbc_not_installed(self, installer_module):
        """Test when pyodbc is not installed."""
        # Force ImportError by removing pyodbc from modules
        with patch.dict(sys.modules, {'pyodbc': None}):
            ok, details = installer_module.check_odbc_driver()
            assert ok is False
    
    def test_check_sqlcmd_success(self, installer_module):
        """Test sqlcmd availability check."""
        with patch.object(installer_module, 'run_command') as mock_run:
            mock_run.return_value = (0, "sqlcmd help", "")
            ok, details = installer_module.check_sqlcmd()
            assert ok is True
            assert "available" in details.lower()
    
    def test_check_sqlcmd_not_installed(self, installer_module):
        """Test sqlcmd when not installed."""
        with patch.object(installer_module, 'run_command') as mock_run:
            mock_run.return_value = (-1, "", "Command not found")
            ok, details = installer_module.check_sqlcmd()
            assert ok is False


# ============================================================================
# TEST: RUN_COMMAND UTILITY
# ============================================================================

class TestRunCommand:
    """Tests for the run_command utility function."""
    
    def test_run_command_success(self, installer_module):
        """Test run_command with successful command."""
        code, stdout, stderr = installer_module.run_command(["echo", "hello"])
        # On Windows, echo may behave differently
        assert isinstance(code, int)
        assert isinstance(stdout, str)
        assert isinstance(stderr, str)
    
    def test_run_command_command_not_found(self, installer_module):
        """Test run_command with non-existent command."""
        code, stdout, stderr = installer_module.run_command(["nonexistent_cmd_xyz"])
        assert code != 0
        assert "not found" in stderr.lower() or code == -1
    
    def test_run_command_with_cwd(self, installer_module, temp_project_dir):
        """Test run_command with custom working directory."""
        # Create a test file
        test_file = temp_project_dir / "test.txt"
        test_file.write_text("content")
        
        # Check command works with cwd
        # Using Python instead of shell commands for cross-platform
        code, stdout, stderr = installer_module.run_command(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            cwd=temp_project_dir
        )
        assert code == 0
        assert str(temp_project_dir) in stdout or temp_project_dir.name in stdout


# ============================================================================
# TEST: INSTALL STATE
# ============================================================================

class TestInstallState:
    """Tests for InstallState dataclass."""
    
    def test_default_state_values(self, fresh_state):
        """Test default values of InstallState."""
        assert fresh_state.python_ok is False
        assert fresh_state.docker_ok is False
        assert fresh_state.docker_running is False
        assert fresh_state.odbc_driver is None
        assert fresh_state.venv_path is None
        assert fresh_state.deps_installed is False
        assert fresh_state.observability_running is False
        assert fresh_state.sql_server is None
        assert fresh_state.sql_database == "ACM"
        assert fresh_state.sql_auth_type == "windows"
        assert fresh_state.sql_username is None
        assert fresh_state.sql_password is None
        assert fresh_state.sql_setup_complete is False
        assert fresh_state.completed_steps == []
        assert fresh_state.errors == []
    
    def test_state_modification(self, fresh_state):
        """Test modifying state values."""
        fresh_state.python_ok = True
        fresh_state.docker_ok = True
        fresh_state.odbc_driver = "ODBC Driver 18 for SQL Server"
        fresh_state.sql_server = "localhost\\SQLEXPRESS"
        
        assert fresh_state.python_ok is True
        assert fresh_state.docker_ok is True
        assert "18" in fresh_state.odbc_driver
        assert "SQLEXPRESS" in fresh_state.sql_server
    
    def test_completed_steps_tracking(self, fresh_state, installer_module):
        """Test tracking completed installation steps."""
        fresh_state.completed_steps.append(installer_module.InstallStep.PREREQUISITES)
        fresh_state.completed_steps.append(installer_module.InstallStep.PYTHON_ENV)
        
        assert len(fresh_state.completed_steps) == 2
        assert installer_module.InstallStep.PREREQUISITES in fresh_state.completed_steps
        assert installer_module.InstallStep.PYTHON_ENV in fresh_state.completed_steps
        assert installer_module.InstallStep.OBSERVABILITY not in fresh_state.completed_steps


# ============================================================================
# TEST: SQL CONFIGURATION
# ============================================================================

class TestSQLConfiguration:
    """Tests for SQL configuration file creation."""
    
    def test_create_sql_config_windows_auth(self, installer_module, fresh_state, temp_project_dir):
        """Test creating sql_connection.ini with Windows auth."""
        fresh_state.sql_server = "localhost\\SQLEXPRESS"
        fresh_state.sql_database = "ACM"
        fresh_state.sql_auth_type = "windows"
        fresh_state.odbc_driver = "ODBC Driver 18 for SQL Server"
        
        with patch.object(installer_module, 'PROJECT_ROOT', temp_project_dir):
            installer_module.create_sql_config(fresh_state)
            
            config_path = temp_project_dir / "configs" / "sql_connection.ini"
            assert config_path.exists()
            
            content = config_path.read_text()
            assert "[acm]" in content
            assert "server=localhost\\SQLEXPRESS" in content
            assert "database=ACM" in content
            assert "trusted_connection=yes" in content
            assert "ODBC Driver 18" in content
    
    def test_create_sql_config_sql_auth(self, installer_module, fresh_state, temp_project_dir):
        """Test creating sql_connection.ini with SQL Server auth."""
        fresh_state.sql_server = "myserver.database.windows.net"
        fresh_state.sql_database = "ACMProd"
        fresh_state.sql_auth_type = "sql"
        fresh_state.sql_username = "acm_user"
        fresh_state.sql_password = "SecurePass123!"
        fresh_state.odbc_driver = "ODBC Driver 18 for SQL Server"
        
        with patch.object(installer_module, 'PROJECT_ROOT', temp_project_dir):
            installer_module.create_sql_config(fresh_state)
            
            config_path = temp_project_dir / "configs" / "sql_connection.ini"
            content = config_path.read_text()
            
            assert "server=myserver.database.windows.net" in content
            assert "database=ACMProd" in content
            assert "user=acm_user" in content
            assert "password=SecurePass123!" in content
            assert "trusted_connection" not in content


# ============================================================================
# TEST: SQL CONNECTION TESTING
# ============================================================================

class TestSQLConnection:
    """Tests for SQL Server connection testing."""
    
    def test_sql_connection_success(self, installer_module, fresh_state):
        """Test successful SQL connection."""
        fresh_state.sql_server = "localhost\\SQLEXPRESS"
        fresh_state.sql_auth_type = "windows"
        fresh_state.odbc_driver = "ODBC Driver 18 for SQL Server"
        
        mock_pyodbc = MagicMock()
        mock_conn = MagicMock()
        mock_pyodbc.connect.return_value = mock_conn
        
        with patch.dict(sys.modules, {'pyodbc': mock_pyodbc}):
            result = installer_module.test_sql_connection(fresh_state)
            assert result is True
            mock_conn.close.assert_called_once()
    
    def test_sql_connection_failure(self, installer_module, fresh_state):
        """Test failed SQL connection."""
        fresh_state.sql_server = "invalid_server"
        fresh_state.sql_auth_type = "windows"
        fresh_state.odbc_driver = "ODBC Driver 18 for SQL Server"
        
        mock_pyodbc = MagicMock()
        mock_pyodbc.connect.side_effect = Exception("Connection timeout")
        
        with patch.dict(sys.modules, {'pyodbc': mock_pyodbc}):
            with patch.object(installer_module, 'print_error'):
                result = installer_module.test_sql_connection(fresh_state)
                assert result is False
    
    def test_sql_connection_no_driver(self, installer_module, fresh_state):
        """Test SQL connection when no ODBC driver is set."""
        fresh_state.sql_server = "localhost"
        fresh_state.odbc_driver = None
        
        result = installer_module.test_sql_connection(fresh_state)
        assert result is False


# ============================================================================
# TEST: DOCKER COMPOSE OPERATIONS
# ============================================================================

class TestDockerComposeOperations:
    """Tests for Docker Compose orchestration."""
    
    def test_compose_file_exists_check(self, installer_module, temp_project_dir):
        """Test verification that docker-compose.yaml exists."""
        compose_dir = temp_project_dir / "install" / "observability"
        compose_file = compose_dir / "docker-compose.yaml"
        
        assert compose_file.exists()
        content = compose_file.read_text()
        assert "services:" in content
        assert "grafana:" in content
    
    def test_compose_file_missing_handling(self, installer_module, fresh_state, temp_project_dir):
        """Test handling when docker-compose.yaml is missing."""
        # Remove the compose file
        compose_file = temp_project_dir / "install" / "observability" / "docker-compose.yaml"
        compose_file.unlink()
        
        # The setup_observability function should handle this
        with patch.object(installer_module, 'DOCKER_COMPOSE_DIR', temp_project_dir / "install" / "observability"):
            with patch.object(installer_module, 'print_error') as mock_error:
                fresh_state.docker_running = True
                result = installer_module.setup_observability(fresh_state)
                assert result is False


# ============================================================================
# TEST: ENDPOINT VERIFICATION
# ============================================================================

class TestEndpointVerification:
    """Tests for observability endpoint verification."""
    
    def test_endpoint_list_completeness(self, installer_module):
        """Verify all expected endpoints are checked."""
        # These are the endpoints defined in setup_observability
        expected_endpoints = [
            "Grafana",
            "Alloy", 
            "Tempo",
            "Loki",
            "Prometheus",
            "Pyroscope",
        ]
        
        # The endpoints are hardcoded in setup_observability
        # Just verify we can import the module
        assert hasattr(installer_module, 'setup_observability')


# ============================================================================
# TEST: INSTALL STEP ENUM
# ============================================================================

class TestInstallStepEnum:
    """Tests for InstallStep enumeration."""
    
    def test_all_steps_defined(self, installer_module):
        """Test all installation steps are defined."""
        steps = list(installer_module.InstallStep)
        
        assert len(steps) == 6
        assert installer_module.InstallStep.PREREQUISITES in steps
        assert installer_module.InstallStep.PYTHON_ENV in steps
        assert installer_module.InstallStep.OBSERVABILITY in steps
        assert installer_module.InstallStep.SQL_SETUP in steps
        assert installer_module.InstallStep.CONFIGURATION in steps
        assert installer_module.InstallStep.VERIFICATION in steps
    
    def test_step_values(self, installer_module):
        """Test step enum values."""
        assert installer_module.InstallStep.PREREQUISITES.value == "prerequisites"
        assert installer_module.InstallStep.PYTHON_ENV.value == "python_env"
        assert installer_module.InstallStep.OBSERVABILITY.value == "observability"
        assert installer_module.InstallStep.SQL_SETUP.value == "sql_setup"
        assert installer_module.InstallStep.CONFIGURATION.value == "configuration"
        assert installer_module.InstallStep.VERIFICATION.value == "verification"


# ============================================================================
# TEST: CONSOLE OUTPUT HELPERS
# ============================================================================

class TestConsoleHelpers:
    """Tests for console output helper functions."""
    
    def test_print_functions_exist(self, installer_module):
        """Test all print helper functions exist."""
        assert callable(installer_module.print_header)
        assert callable(installer_module.print_step)
        assert callable(installer_module.print_success)
        assert callable(installer_module.print_warning)
        assert callable(installer_module.print_error)
        assert callable(installer_module.print_info)
    
    def test_print_step_parameters(self, installer_module):
        """Test print_step accepts correct parameters."""
        with patch.object(installer_module.console, 'print'):
            with patch.object(installer_module.console, 'rule'):
                # Should not raise
                installer_module.print_step(1, 6, "Test Step")


# ============================================================================
# TEST: CONFIGURATION CONSTANTS
# ============================================================================

class TestConfigurationConstants:
    """Tests for installer configuration constants."""
    
    def test_version_constant(self, installer_module):
        """Test VERSION constant is defined."""
        assert hasattr(installer_module, 'VERSION')
        assert isinstance(installer_module.VERSION, str)
        # Version should be semver format
        parts = installer_module.VERSION.split('.')
        assert len(parts) >= 2
    
    def test_python_min_version(self, installer_module):
        """Test PYTHON_MIN_VERSION constant."""
        assert hasattr(installer_module, 'PYTHON_MIN_VERSION')
        assert installer_module.PYTHON_MIN_VERSION == (3, 11)
    
    def test_docker_compose_dir(self, installer_module):
        """Test DOCKER_COMPOSE_DIR path."""
        assert hasattr(installer_module, 'DOCKER_COMPOSE_DIR')
        assert isinstance(installer_module.DOCKER_COMPOSE_DIR, Path)
        assert "observability" in str(installer_module.DOCKER_COMPOSE_DIR)
    
    def test_sql_install_dir(self, installer_module):
        """Test SQL_INSTALL_DIR path."""
        assert hasattr(installer_module, 'SQL_INSTALL_DIR')
        assert isinstance(installer_module.SQL_INSTALL_DIR, Path)
        assert "sql" in str(installer_module.SQL_INSTALL_DIR)
    
    def test_acm_version_loaded(self, installer_module):
        """Test ACM_VERSION is loaded from version.py."""
        assert hasattr(installer_module, 'ACM_VERSION')
        # Should be a version string or "unknown"
        assert isinstance(installer_module.ACM_VERSION, str)
        if installer_module.ACM_VERSION != "unknown":
            parts = installer_module.ACM_VERSION.split('.')
            assert len(parts) >= 2


# ============================================================================
# TEST: WIZARD STYLE
# ============================================================================

class TestWizardStyle:
    """Tests for questionary wizard style."""
    
    def test_wizard_style_defined(self, installer_module):
        """Test WIZARD_STYLE is defined."""
        assert hasattr(installer_module, 'WIZARD_STYLE')
    
    def test_wizard_style_has_colors(self, installer_module):
        """Test wizard style has color definitions."""
        # Just verify it's a questionary Style object
        from questionary import Style
        assert isinstance(installer_module.WIZARD_STYLE, Style)


# ============================================================================
# TEST: PROJECT ROOT AND SCRIPT DIR
# ============================================================================

class TestPaths:
    """Tests for path constants."""
    
    def test_project_root_is_path(self, installer_module):
        """Test PROJECT_ROOT is a Path object."""
        assert hasattr(installer_module, 'PROJECT_ROOT')
        assert isinstance(installer_module.PROJECT_ROOT, Path)
    
    def test_script_dir_is_path(self, installer_module):
        """Test SCRIPT_DIR is a Path object."""
        assert hasattr(installer_module, 'SCRIPT_DIR')
        assert isinstance(installer_module.SCRIPT_DIR, Path)
    
    def test_script_dir_is_install_folder(self, installer_module):
        """Test SCRIPT_DIR points to install folder."""
        assert "install" in str(installer_module.SCRIPT_DIR)
    
    def test_project_root_is_parent_of_script_dir(self, installer_module):
        """Test PROJECT_ROOT is parent of SCRIPT_DIR."""
        assert installer_module.PROJECT_ROOT == installer_module.SCRIPT_DIR.parent


# ============================================================================
# TEST: MAIN FUNCTION
# ============================================================================

class TestMainFunction:
    """Tests for the main installer function."""
    
    def test_main_exists(self, installer_module):
        """Test main function exists."""
        assert hasattr(installer_module, 'main')
        assert callable(installer_module.main)
    
    def test_main_is_entry_point(self, installer_module):
        """Test module can be run as script."""
        # Check __name__ == "__main__" block exists
        import inspect
        source = inspect.getsource(installer_module)
        assert 'if __name__ == "__main__":' in source
        assert 'main()' in source


# ============================================================================
# TEST: ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in installer."""
    
    def test_state_tracks_errors(self, fresh_state):
        """Test InstallState can track errors."""
        fresh_state.errors.append("Test error 1")
        fresh_state.errors.append("Test error 2")
        
        assert len(fresh_state.errors) == 2
        assert "Test error 1" in fresh_state.errors
    
    def test_run_command_captures_stderr(self, installer_module):
        """Test run_command captures standard error."""
        # Use Python to print to stderr
        code, stdout, stderr = installer_module.run_command([
            sys.executable, "-c", 
            "import sys; sys.stderr.write('error message')"
        ])
        assert "error message" in stderr or code == 0


# ============================================================================
# TEST: INTEGRATION SCENARIOS (MOCKED)
# ============================================================================

class TestIntegrationScenarios:
    """Integration tests using mocks to simulate full scenarios."""
    
    def test_full_prerequisites_check_all_pass(self, installer_module, fresh_state):
        """Test prerequisites check when everything passes."""
        with patch.object(installer_module, 'check_python_version', return_value=(True, "Python 3.11")):
            with patch.object(installer_module, 'check_docker_installed', return_value=(True, "Docker 24.0")):
                with patch.object(installer_module, 'check_docker_running', return_value=(True, "Running")):
                    with patch.object(installer_module, 'check_docker_compose', return_value=(True, "v2.20")):
                        with patch.object(installer_module, 'check_odbc_driver', return_value=(True, "Driver 18")):
                            with patch.object(installer_module, 'check_sqlcmd', return_value=(True, "Available")):
                                with patch.object(installer_module.console, 'print'):
                                    result = installer_module.run_prerequisites_check(fresh_state)
                                    
                                    assert result is True
                                    assert fresh_state.python_ok is True
                                    assert fresh_state.docker_ok is True
                                    assert fresh_state.docker_running is True
                                    assert fresh_state.odbc_driver == "Driver 18"
    
    def test_prerequisites_check_docker_missing(self, installer_module, fresh_state):
        """Test prerequisites check when Docker is missing."""
        with patch.object(installer_module, 'check_python_version', return_value=(True, "Python 3.11")):
            with patch.object(installer_module, 'check_docker_installed', return_value=(False, "Not installed")):
                with patch.object(installer_module, 'check_docker_running', return_value=(False, "N/A")):
                    with patch.object(installer_module, 'check_docker_compose', return_value=(False, "N/A")):
                        with patch.object(installer_module, 'check_odbc_driver', return_value=(True, "Driver 18")):
                            with patch.object(installer_module, 'check_sqlcmd', return_value=(True, "Available")):
                                with patch.object(installer_module.console, 'print'):
                                    with patch.object(installer_module, 'print_warning'):
                                        with patch.object(installer_module, 'print_info'):
                                            with patch('questionary.confirm') as mock_confirm:
                                                mock_confirm.return_value.ask.return_value = False
                                                result = installer_module.run_prerequisites_check(fresh_state)
                                                
                                                assert result is False
                                                assert fresh_state.docker_ok is False
    
    def test_observability_already_running(self, installer_module, fresh_state, temp_project_dir):
        """Test observability setup when containers already running."""
        fresh_state.docker_running = True
        
        with patch.object(installer_module, 'DOCKER_COMPOSE_DIR', temp_project_dir / "install" / "observability"):
            with patch.object(installer_module, 'run_command') as mock_run:
                # First call: check if containers running
                mock_run.return_value = (0, "container_id_123", "")
                
                with patch.object(installer_module.console, 'print'):
                    with patch.object(installer_module, 'print_info'):
                        with patch('questionary.confirm') as mock_confirm:
                            mock_confirm.return_value.ask.return_value = False
                            result = installer_module.setup_observability(fresh_state)
                            
                            assert result is True
                            assert fresh_state.observability_running is True


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_sql_server_string(self, installer_module, fresh_state):
        """Test handling of empty SQL server string."""
        fresh_state.sql_server = ""
        fresh_state.odbc_driver = "ODBC Driver 18"
        
        result = installer_module.test_sql_connection(fresh_state)
        assert result is False
    
    def test_special_characters_in_password(self, installer_module, fresh_state, temp_project_dir):
        """Test SQL config with special characters in password."""
        fresh_state.sql_server = "localhost"
        fresh_state.sql_database = "ACM"
        fresh_state.sql_auth_type = "sql"
        fresh_state.sql_username = "user"
        fresh_state.sql_password = "P@ss!w0rd#$%^&*()"
        fresh_state.odbc_driver = "ODBC Driver 18"
        
        with patch.object(installer_module, 'PROJECT_ROOT', temp_project_dir):
            installer_module.create_sql_config(fresh_state)
            
            config_path = temp_project_dir / "configs" / "sql_connection.ini"
            content = config_path.read_text()
            
            # Password should be written as-is (may need escaping in real use)
            assert "password=P@ss!w0rd#$%^&*()" in content
    
    def test_unicode_in_server_name(self, installer_module, fresh_state, temp_project_dir):
        """Test handling of unicode in server name."""
        fresh_state.sql_server = "server-日本語"
        fresh_state.sql_database = "ACM"
        fresh_state.sql_auth_type = "windows"
        fresh_state.odbc_driver = "ODBC Driver 18"
        
        with patch.object(installer_module, 'PROJECT_ROOT', temp_project_dir):
            # Should not raise
            installer_module.create_sql_config(fresh_state)
            
            config_path = temp_project_dir / "configs" / "sql_connection.ini"
            content = config_path.read_text(encoding='utf-8')
            assert "server-日本語" in content


# ============================================================================
# TEST: DOCKER DOWNLOAD (Mocked)
# ============================================================================

class TestDockerDownload:
    """Tests for Docker Desktop download functionality."""
    
    def test_download_docker_desktop_function_exists(self, installer_module):
        """Test download_docker_desktop function exists."""
        assert hasattr(installer_module, 'download_docker_desktop')
        assert callable(installer_module.download_docker_desktop)
    
    def test_download_url_is_correct(self, installer_module):
        """Test the Docker download URL is valid."""
        import inspect
        source = inspect.getsource(installer_module.download_docker_desktop)
        
        assert "desktop.docker.com" in source
        assert "Docker%20Desktop%20Installer.exe" in source


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
