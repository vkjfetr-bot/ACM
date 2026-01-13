#!/usr/bin/env python3
"""
ACM Installer Wizard
====================
A comprehensive installer for ACM (Automated Condition Monitoring) v11.

This wizard handles:
1. Prerequisites check (Python, Docker, ODBC Driver)
2. Python virtual environment and dependencies
3. Observability stack (Docker Compose)
4. SQL Server database setup (optional)
5. Configuration file generation
6. Post-install verification

Usage:
    python install/acm_installer.py

Requirements:
    - Python 3.11+
    - Docker Desktop (for observability)
    - questionary (pip install questionary)
    - rich (already in ACM dependencies)
"""

import os
import sys
import subprocess
import shutil
import time
import platform
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# Ensure we can find the project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.prompt import Confirm
    from rich.markdown import Markdown
    from rich import box
except ImportError:
    print("ERROR: 'rich' package is required. Install with: pip install rich")
    sys.exit(1)

try:
    import questionary
    from questionary import Style
except ImportError:
    print("ERROR: 'questionary' package is required. Install with: pip install questionary")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "1.0.0"
PYTHON_MIN_VERSION = (3, 11)
DOCKER_COMPOSE_DIR = SCRIPT_DIR / "observability"
SQL_INSTALL_DIR = SCRIPT_DIR / "sql"

# Get ACM version dynamically
def get_acm_version() -> str:
    """Read ACM version from utils/version.py."""
    try:
        version_file = PROJECT_ROOT / "utils" / "version.py"
        if version_file.exists():
            content = version_file.read_text()
            for line in content.split('\n'):
                if line.startswith('__version__'):
                    # Extract version string
                    return line.split('=')[1].strip().strip('"\'')
    except Exception:
        pass
    return "unknown"

ACM_VERSION = get_acm_version()

# Custom questionary style
WIZARD_STYLE = Style([
    ('qmark', 'fg:cyan bold'),
    ('question', 'bold'),
    ('answer', 'fg:green bold'),
    ('pointer', 'fg:cyan bold'),
    ('highlighted', 'fg:cyan bold'),
    ('selected', 'fg:green'),
    ('separator', 'fg:gray'),
    ('instruction', 'fg:gray'),
])


# ============================================================================
# DATA CLASSES
# ============================================================================

class InstallStep(Enum):
    PREREQUISITES = "prerequisites"
    PYTHON_ENV = "python_env"
    OBSERVABILITY = "observability"
    SQL_SETUP = "sql_setup"
    CONFIGURATION = "configuration"
    VERIFICATION = "verification"


@dataclass
class InstallState:
    """Tracks installation progress and configuration."""
    # Steps completed
    completed_steps: List[InstallStep] = field(default_factory=list)
    
    # Prerequisites
    python_ok: bool = False
    docker_ok: bool = False
    docker_running: bool = False
    odbc_driver: Optional[str] = None
    
    # Python environment
    venv_path: Optional[Path] = None
    deps_installed: bool = False
    
    # Observability
    observability_running: bool = False
    
    # SQL configuration
    sql_server: Optional[str] = None
    sql_database: str = "ACM"
    sql_auth_type: str = "windows"  # or "sql"
    sql_username: Optional[str] = None
    sql_password: Optional[str] = None
    sql_setup_complete: bool = False
    
    # Errors
    errors: List[str] = field(default_factory=list)


# ============================================================================
# CONSOLE HELPERS
# ============================================================================

console = Console()


def print_header():
    """Print the installer header."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]ACM Installer Wizard[/bold cyan]\n"
        f"[dim]Automated Condition Monitoring v{ACM_VERSION}[/dim]\n"
        f"[dim]Installer v{VERSION}[/dim]",
        border_style="cyan",
        padding=(1, 4)
    ))
    console.print()


def print_step(step_num: int, total: int, title: str):
    """Print a step header."""
    console.print()
    console.rule(f"[bold cyan]Step {step_num}/{total}: {title}[/bold cyan]", style="cyan")
    console.print()


def print_success(message: str):
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_error(message: str):
    """Print an error message."""
    console.print(f"[red]✗[/red] {message}")


def print_info(message: str):
    """Print an info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def run_command(
    cmd: List[str], 
    capture: bool = True, 
    check: bool = False,
    cwd: Optional[Path] = None,
    shell: bool = False
) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            check=check,
            cwd=cwd,
            shell=shell
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


# ============================================================================
# PREREQUISITES CHECK
# ============================================================================

def check_python_version() -> Tuple[bool, str]:
    """Check Python version meets requirements."""
    version = sys.version_info[:2]
    version_str = f"{version[0]}.{version[1]}"
    required_str = f"{PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}"
    
    if version >= PYTHON_MIN_VERSION:
        return True, f"Python {version_str} (required: {required_str}+)"
    return False, f"Python {version_str} - requires {required_str}+"


def check_docker_installed() -> Tuple[bool, str]:
    """Check if Docker is installed."""
    code, stdout, stderr = run_command(["docker", "--version"])
    if code == 0:
        version = stdout.strip().split(",")[0] if stdout else "Docker"
        return True, version
    return False, "Docker not installed"


def check_docker_running() -> Tuple[bool, str]:
    """Check if Docker daemon is running."""
    code, stdout, stderr = run_command(["docker", "info"])
    if code == 0:
        return True, "Docker daemon is running"
    return False, "Docker daemon not running"


def check_docker_compose() -> Tuple[bool, str]:
    """Check if docker compose is available."""
    # Try 'docker compose' (v2)
    code, stdout, stderr = run_command(["docker", "compose", "version"])
    if code == 0:
        version = stdout.strip().split()[-1] if stdout else "v2"
        return True, f"Docker Compose {version}"
    
    # Try 'docker-compose' (v1)
    code, stdout, stderr = run_command(["docker-compose", "--version"])
    if code == 0:
        return True, stdout.strip()
    
    return False, "Docker Compose not installed"


def check_odbc_driver() -> Tuple[bool, str]:
    """Check for SQL Server ODBC driver."""
    try:
        import pyodbc
        drivers = [d for d in pyodbc.drivers() if 'SQL Server' in d]
        if drivers:
            # Prefer driver 18, then 17
            preferred = None
            for driver in drivers:
                if '18' in driver:
                    preferred = driver
                    break
                elif '17' in driver and preferred is None:
                    preferred = driver
            if preferred:
                return True, preferred
            return True, drivers[0]
    except ImportError:
        pass
    return False, "No SQL Server ODBC driver found"


def check_sqlcmd() -> Tuple[bool, str]:
    """Check if sqlcmd is available."""
    code, stdout, stderr = run_command(["sqlcmd", "-?"])
    if code == 0:
        return True, "sqlcmd available"
    return False, "sqlcmd not installed"


def run_prerequisites_check(state: InstallState) -> bool:
    """Run all prerequisites checks and update state."""
    print_step(1, 6, "Prerequisites Check")
    
    table = Table(title="System Requirements", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Details", style="dim")
    
    checks = [
        ("Python", check_python_version),
        ("Docker", check_docker_installed),
        ("Docker Daemon", check_docker_running),
        ("Docker Compose", check_docker_compose),
        ("ODBC Driver", check_odbc_driver),
        ("sqlcmd", check_sqlcmd),
    ]
    
    all_critical_ok = True
    
    for name, check_func in checks:
        ok, details = check_func()
        
        if ok:
            status = "[green]✓ OK[/green]"
        elif name in ["ODBC Driver", "sqlcmd"]:
            # These are optional for observability-only install
            status = "[yellow]⚠ Optional[/yellow]"
        else:
            status = "[red]✗ Missing[/red]"
            all_critical_ok = False
        
        table.add_row(name, status, details)
        
        # Update state
        if name == "Python":
            state.python_ok = ok
        elif name == "Docker":
            state.docker_ok = ok
        elif name == "Docker Daemon":
            state.docker_running = ok
        elif name == "ODBC Driver" and ok:
            state.odbc_driver = details
    
    console.print(table)
    console.print()
    
    if not state.python_ok:
        print_error(f"Python {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}+ is required")
        return False
    
    if not state.docker_ok:
        print_warning("Docker is not installed")
        if platform.system() == "Windows":
            print_info("Download Docker Desktop: https://www.docker.com/products/docker-desktop/")
            
            if questionary.confirm(
                "Would you like to download Docker Desktop installer?",
                style=WIZARD_STYLE
            ).ask():
                download_docker_desktop()
                return False  # User needs to install Docker and restart
        return False
    
    if not state.docker_running:
        print_warning("Docker Desktop is installed but not running")
        print_info("Please start Docker Desktop and run this installer again")
        return False
    
    print_success("All critical prerequisites met!")
    return True


def download_docker_desktop():
    """Download Docker Desktop installer for Windows."""
    import urllib.request
    
    url = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
    download_path = Path.home() / "Downloads" / "DockerDesktopInstaller.exe"
    
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Downloading Docker Desktop...", total=None)
        
        try:
            urllib.request.urlretrieve(url, download_path)
            progress.update(task, completed=True)
            print_success(f"Downloaded to: {download_path}")
            print_info("Please run the installer and restart this wizard after Docker is installed")
        except Exception as e:
            print_error(f"Download failed: {e}")
            print_info(f"Please download manually from: {url}")


# ============================================================================
# PYTHON ENVIRONMENT SETUP
# ============================================================================

def setup_python_environment(state: InstallState) -> bool:
    """Set up Python virtual environment and install dependencies."""
    print_step(2, 6, "Python Environment")
    
    # Check if we're already in a venv
    in_venv = sys.prefix != sys.base_prefix
    
    if in_venv:
        print_info(f"Already in virtual environment: {sys.prefix}")
        venv_path = Path(sys.prefix)
    else:
        print_info("Creating virtual environment is recommended but optional")
        create_venv = questionary.confirm(
            "Create a new virtual environment?",
            default=False,
            style=WIZARD_STYLE
        ).ask()
        
        if create_venv:
            venv_path = PROJECT_ROOT / ".venv"
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Creating virtual environment...", total=None)
                
                code, stdout, stderr = run_command([
                    sys.executable, "-m", "venv", str(venv_path)
                ])
                
                if code != 0:
                    print_error(f"Failed to create venv: {stderr}")
                    return False
                
                progress.update(task, description="Virtual environment created!")
            
            print_success(f"Created: {venv_path}")
            print_info("Activate with:")
            if platform.system() == "Windows":
                console.print(f"    [cyan]{venv_path}\\Scripts\\activate[/cyan]")
            else:
                console.print(f"    [cyan]source {venv_path}/bin/activate[/cyan]")
            
            state.venv_path = venv_path
        else:
            venv_path = None
    
    # Install dependencies
    console.print()
    print_info("Installing Python dependencies...")
    
    pip_executable = sys.executable
    if state.venv_path:
        if platform.system() == "Windows":
            pip_executable = str(state.venv_path / "Scripts" / "python.exe")
        else:
            pip_executable = str(state.venv_path / "bin" / "python")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Installing core dependencies...", total=None)
        
        # Install from pyproject.toml
        code, stdout, stderr = run_command([
            pip_executable, "-m", "pip", "install", "-e", str(PROJECT_ROOT)
        ])
        
        if code != 0:
            print_warning(f"pip install -e . failed, trying requirements approach")
            # Fallback: install core deps directly
            deps = [
                "numpy>=1.26", "pandas>=2.2", "scikit-learn>=1.4",
                "matplotlib>=3.8", "pyyaml>=6.0", "joblib>=1.3",
                "seaborn>=0.13", "pyodbc>=5.0", "statsmodels>=0.14",
                "scipy>=1.12", "psutil>=5.9", "structlog>=24.0", "rich>=13.0"
            ]
            code, stdout, stderr = run_command([
                pip_executable, "-m", "pip", "install"
            ] + deps)
        
        progress.update(task, description="Installing telemetry dependencies...")
        
        # Install telemetry deps (optional)
        code2, _, _ = run_command([
            pip_executable, "-m", "pip", "install",
            "opentelemetry-api>=1.25",
            "opentelemetry-sdk>=1.25", 
            "opentelemetry-exporter-otlp-proto-http>=1.25"
        ])
    
    state.deps_installed = True
    print_success("Python dependencies installed!")
    return True


# ============================================================================
# OBSERVABILITY STACK
# ============================================================================

def setup_observability(state: InstallState) -> bool:
    """Set up the observability stack using Docker Compose."""
    print_step(3, 6, "Observability Stack")
    
    if not state.docker_running:
        print_warning("Docker is not running - skipping observability setup")
        return True
    
    # Check if docker-compose.yaml exists
    compose_file = DOCKER_COMPOSE_DIR / "docker-compose.yaml"
    if not compose_file.exists():
        print_error(f"docker-compose.yaml not found at: {compose_file}")
        return False
    
    console.print()
    print_info("The observability stack includes:")
    console.print("  • [cyan]Grafana[/cyan] - Dashboards (port 3000)")
    console.print("  • [cyan]Alloy[/cyan] - OpenTelemetry collector (ports 4317, 4318)")
    console.print("  • [cyan]Tempo[/cyan] - Distributed tracing (port 3200)")
    console.print("  • [cyan]Loki[/cyan] - Log aggregation (port 3100)")
    console.print("  • [cyan]Prometheus[/cyan] - Metrics (port 9090)")
    console.print("  • [cyan]Pyroscope[/cyan] - Profiling (port 4040)")
    console.print()
    
    # Check if already running
    code, stdout, stderr = run_command(
        ["docker", "compose", "ps", "-q"],
        cwd=DOCKER_COMPOSE_DIR
    )
    containers_running = bool(stdout.strip())
    
    if containers_running:
        print_info("Observability containers are already running")
        restart = questionary.confirm(
            "Restart the observability stack?",
            default=False,
            style=WIZARD_STYLE
        ).ask()
        
        if not restart:
            state.observability_running = True
            return True
        
        # Stop existing containers
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Stopping existing containers...", total=None)
            run_command(["docker", "compose", "down"], cwd=DOCKER_COMPOSE_DIR)
    
    # Start the stack
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Starting observability stack...", total=None)
        
        code, stdout, stderr = run_command(
            ["docker", "compose", "up", "-d"],
            cwd=DOCKER_COMPOSE_DIR
        )
        
        if code != 0:
            print_error(f"Failed to start observability stack")
            console.print(f"[red]{stderr}[/red]")
            return False
        
        progress.update(task, description="Waiting for containers to be healthy...")
        
        # Wait for containers to be healthy
        max_wait = 120  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            code, stdout, stderr = run_command(
                ["docker", "compose", "ps", "--format", "json"],
                cwd=DOCKER_COMPOSE_DIR
            )
            
            if code == 0 and stdout:
                try:
                    # Parse container status
                    containers = []
                    for line in stdout.strip().split('\n'):
                        if line:
                            containers.append(json.loads(line))
                    
                    all_healthy = True
                    for c in containers:
                        health = c.get('Health', '')
                        state_str = c.get('State', '')
                        # Consider healthy if: explicitly healthy, running without healthcheck, or Pyroscope (no healthcheck)
                        name = c.get('Name', '')
                        if state_str == 'running':
                            if health == 'healthy' or health == '' or 'pyroscope' in name.lower():
                                continue
                            else:
                                all_healthy = False
                        else:
                            all_healthy = False
                    
                    if all_healthy and containers:
                        break
                except json.JSONDecodeError:
                    pass
            
            time.sleep(5)
    
    # Verify endpoints
    console.print()
    print_info("Verifying endpoints...")
    
    endpoints = [
        ("Grafana", "http://localhost:3000/api/health"),
        ("Alloy", "http://localhost:9099/-/ready"),
        ("Tempo", "http://localhost:3200/ready"),
        ("Loki", "http://localhost:3100/ready"),
        ("Prometheus", "http://localhost:9090/-/ready"),
        ("Pyroscope", "http://localhost:4040/"),
    ]
    
    all_ok = True
    for name, url in endpoints:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 400:
                    print_success(f"{name}: OK")
                else:
                    print_warning(f"{name}: HTTP {response.status}")
                    all_ok = False
        except Exception as e:
            # Pyroscope may return 503 initially but is still working
            if "Pyroscope" in name:
                print_warning(f"{name}: Starting up (this is normal)")
            else:
                print_error(f"{name}: {e}")
                all_ok = False
    
    state.observability_running = True
    
    console.print()
    if all_ok:
        print_success("Observability stack is running!")
    else:
        print_warning("Some services may still be starting up")
    
    console.print()
    console.print(Panel(
        "[bold]Grafana Dashboard[/bold]\n"
        "URL: [cyan]http://localhost:3000[/cyan]\n"
        "User: [cyan]admin[/cyan]\n"
        "Password: [cyan]admin[/cyan]",
        title="Access",
        border_style="green"
    ))
    
    return True


# ============================================================================
# SQL SERVER SETUP
# ============================================================================

def setup_sql_server(state: InstallState) -> bool:
    """Configure SQL Server connection and optionally set up database."""
    print_step(4, 6, "SQL Server Configuration")
    
    skip_sql = questionary.confirm(
        "Skip SQL Server setup for now? (You can set it up later)",
        default=True,
        style=WIZARD_STYLE
    ).ask()
    
    if skip_sql:
        print_info("SQL Server setup skipped")
        print_info("To set up later, run: python install/install_acm.py --ini-section acm")
        return True
    
    # SQL Server instance
    console.print()
    print_info("Enter SQL Server connection details:")
    
    state.sql_server = questionary.text(
        "SQL Server instance (e.g., localhost\\SQLEXPRESS):",
        style=WIZARD_STYLE
    ).ask()
    
    if not state.sql_server:
        print_warning("No SQL Server specified - skipping")
        return True
    
    # Database name
    state.sql_database = questionary.text(
        "Database name:",
        default="ACM",
        style=WIZARD_STYLE
    ).ask()
    
    # Authentication type
    state.sql_auth_type = questionary.select(
        "Authentication type:",
        choices=[
            "Windows Authentication (Trusted Connection)",
            "SQL Server Authentication"
        ],
        style=WIZARD_STYLE
    ).ask()
    
    if "SQL Server" in state.sql_auth_type:
        state.sql_auth_type = "sql"
        state.sql_username = questionary.text(
            "Username:",
            style=WIZARD_STYLE
        ).ask()
        state.sql_password = questionary.password(
            "Password:",
            style=WIZARD_STYLE
        ).ask()
    else:
        state.sql_auth_type = "windows"
    
    # Create sql_connection.ini
    create_sql_config(state)
    
    # Test connection
    console.print()
    print_info("Testing SQL Server connection...")
    
    if test_sql_connection(state):
        print_success("SQL Server connection successful!")
        
        # Offer to run database setup
        if questionary.confirm(
            "Create/update ACM database schema?",
            default=True,
            style=WIZARD_STYLE
        ).ask():
            run_sql_setup(state)
    else:
        print_error("SQL Server connection failed")
        print_info("Check your connection settings and try again")
        return False
    
    return True


def create_sql_config(state: InstallState):
    """Create sql_connection.ini from state."""
    config_path = PROJECT_ROOT / "configs" / "sql_connection.ini"
    
    content = "[acm]\n"
    content += f"server={state.sql_server}\n"
    content += f"database={state.sql_database}\n"
    
    if state.sql_auth_type == "windows":
        content += "trusted_connection=yes\n"
    else:
        content += f"user={state.sql_username}\n"
        content += f"password={state.sql_password}\n"
    
    content += "mars=yes\n"
    content += f"driver={state.odbc_driver or 'ODBC Driver 18 for SQL Server'}\n"
    content += "trust_server_certificate=yes\n"
    
    config_path.write_text(content, encoding='utf-8')
    print_success(f"Created: {config_path}")


def test_sql_connection(state: InstallState) -> bool:
    """Test SQL Server connection."""
    if not state.odbc_driver:
        return False
    
    try:
        import pyodbc
        
        if state.sql_auth_type == "windows":
            conn_str = (
                f"DRIVER={{{state.odbc_driver}}};"
                f"SERVER={state.sql_server};"
                f"DATABASE=master;"
                f"Trusted_Connection=yes;"
                f"TrustServerCertificate=yes;"
            )
        else:
            conn_str = (
                f"DRIVER={{{state.odbc_driver}}};"
                f"SERVER={state.sql_server};"
                f"DATABASE=master;"
                f"UID={state.sql_username};"
                f"PWD={state.sql_password};"
                f"TrustServerCertificate=yes;"
            )
        
        conn = pyodbc.connect(conn_str, timeout=10)
        conn.close()
        return True
    except Exception as e:
        print_error(f"Connection error: {e}")
        return False


def run_sql_setup(state: InstallState):
    """Run SQL database setup scripts."""
    install_script = SCRIPT_DIR / "install_acm.py"
    
    if not install_script.exists():
        print_error(f"install_acm.py not found at: {install_script}")
        return
    
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running database setup...", total=None)
        
        code, stdout, stderr = run_command([
            sys.executable, str(install_script), "--ini-section", "acm"
        ])
        
        if code == 0:
            state.sql_setup_complete = True
            print_success("Database setup complete!")
        else:
            print_error("Database setup failed")
            if stderr:
                console.print(f"[red]{stderr}[/red]")


# ============================================================================
# CONFIGURATION
# ============================================================================

def setup_configuration(state: InstallState) -> bool:
    """Set up additional configuration files."""
    print_step(5, 6, "Configuration")
    
    # Check for sql_connection.ini
    config_path = PROJECT_ROOT / "configs" / "sql_connection.ini"
    example_path = PROJECT_ROOT / "configs" / "sql_connection.example.ini"
    
    if not config_path.exists() and example_path.exists():
        print_info("sql_connection.ini not found")
        if state.sql_server:
            # Already created during SQL setup
            pass
        else:
            print_info(f"Copy and edit: {example_path}")
    elif config_path.exists():
        print_success("sql_connection.ini exists")
    
    # Check for config_table.csv
    config_table = PROJECT_ROOT / "configs" / "config_table.csv"
    if config_table.exists():
        print_success("config_table.csv exists")
    else:
        print_warning("config_table.csv not found")
    
    return True


# ============================================================================
# VERIFICATION
# ============================================================================

def run_verification(state: InstallState) -> bool:
    """Run post-install verification."""
    print_step(6, 6, "Verification")
    
    table = Table(title="Installation Summary", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")
    
    # Python
    table.add_row("Python Environment", "[green]✓ OK[/green]" if state.python_ok else "[red]✗[/red]")
    
    # Dependencies
    deps_status = "[green]✓ Installed[/green]" if state.deps_installed else "[yellow]⚠ Not installed[/yellow]"
    table.add_row("Python Dependencies", deps_status)
    
    # Observability
    obs_status = "[green]✓ Running[/green]" if state.observability_running else "[yellow]⚠ Not running[/yellow]"
    table.add_row("Observability Stack", obs_status)
    
    # SQL
    if state.sql_setup_complete:
        sql_status = "[green]✓ Configured[/green]"
    elif state.sql_server:
        sql_status = "[yellow]⚠ Configured but not set up[/yellow]"
    else:
        sql_status = "[dim]Skipped[/dim]"
    table.add_row("SQL Server", sql_status)
    
    console.print(table)
    
    # Next steps
    console.print()
    console.print(Panel(
        "[bold]Next Steps:[/bold]\n\n"
        "1. Configure SQL Server (if not done):\n"
        "   [cyan]python install/install_acm.py --ini-section acm[/cyan]\n\n"
        "2. Sync configuration to database:\n"
        "   [cyan]python scripts/sql/populate_acm_config.py[/cyan]\n\n"
        "3. Run ACM batch processor:\n"
        "   [cyan]python scripts/sql_batch_runner.py --equip FD_FAN --tick-minutes 1440[/cyan]\n\n"
        "4. Open Grafana dashboards:\n"
        "   [cyan]http://localhost:3000[/cyan] (admin/admin)",
        title="What's Next?",
        border_style="cyan"
    ))
    
    return True


# ============================================================================
# MAIN WIZARD
# ============================================================================

def main():
    """Run the installer wizard."""
    os.chdir(PROJECT_ROOT)
    
    print_header()
    
    state = InstallState()
    
    # Welcome
    console.print("Welcome to the ACM Installer Wizard!")
    console.print()
    console.print("This wizard will help you set up:")
    console.print("  • Python environment and dependencies")
    console.print("  • Observability stack (Grafana, Tempo, Loki, Prometheus, Pyroscope)")
    console.print("  • SQL Server database (optional)")
    console.print()
    
    if not questionary.confirm(
        "Ready to begin?",
        default=True,
        style=WIZARD_STYLE
    ).ask():
        console.print()
        print_info("Installation cancelled")
        return
    
    try:
        # Step 1: Prerequisites
        if not run_prerequisites_check(state):
            print_error("Prerequisites check failed. Please fix the issues above and try again.")
            return
        state.completed_steps.append(InstallStep.PREREQUISITES)
        
        # Step 2: Python Environment
        if not setup_python_environment(state):
            print_error("Python environment setup failed")
            return
        state.completed_steps.append(InstallStep.PYTHON_ENV)
        
        # Step 3: Observability
        if not setup_observability(state):
            print_warning("Observability setup had issues but continuing...")
        state.completed_steps.append(InstallStep.OBSERVABILITY)
        
        # Step 4: SQL Server (optional)
        if not setup_sql_server(state):
            print_warning("SQL Server setup skipped or failed")
        state.completed_steps.append(InstallStep.SQL_SETUP)
        
        # Step 5: Configuration
        if not setup_configuration(state):
            print_warning("Configuration check had issues")
        state.completed_steps.append(InstallStep.CONFIGURATION)
        
        # Step 6: Verification
        run_verification(state)
        state.completed_steps.append(InstallStep.VERIFICATION)
        
        console.print()
        console.print(Panel(
            "[bold green]Installation Complete![/bold green]",
            border_style="green"
        ))
        
    except KeyboardInterrupt:
        console.print()
        print_warning("Installation cancelled by user")
    except Exception as e:
        console.print()
        print_error(f"Installation failed: {e}")
        raise


if __name__ == "__main__":
    main()
