<#
.SYNOPSIS
    Install pyroscope-io Python SDK with Rust toolchain
    
.DESCRIPTION
    pyroscope-io requires Rust to build from source on Windows.
    This script:
    1. Downloads and installs Rust via rustup
    2. Installs pyroscope-io Python package
    
.NOTES
    Run as Administrator
    Requires internet connection
    
.EXAMPLE
    .\scripts\install_pyroscope_sdk.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " pyroscope-io Installation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# 1. Check if Rust is already installed
# =============================================================================
Write-Host "[1/3] Checking for Rust toolchain..." -ForegroundColor Yellow

$rustVersion = rustc --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Rust already installed: $rustVersion" -ForegroundColor Green
} else {
    Write-Host "  Rust not found. Installing..." -ForegroundColor Yellow
    
    # Download rustup-init.exe
    $rustupUrl = "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe"
    $rustupPath = "$env:TEMP\rustup-init.exe"
    
    Write-Host "  Downloading rustup-init.exe..."
    try {
        Invoke-WebRequest -Uri $rustupUrl -OutFile $rustupPath -UseBasicParsing
    } catch {
        Write-Host "  ERROR: Failed to download rustup-init.exe" -ForegroundColor Red
        Write-Host "  Please download manually from: https://rustup.rs" -ForegroundColor Yellow
        exit 1
    }
    
    # Install Rust with default options
    Write-Host "  Running rustup installer (default profile)..."
    Start-Process -FilePath $rustupPath -ArgumentList "-y" -Wait -NoNewWindow
    
    # Add Rust to PATH for current session
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
    
    # Verify installation
    $rustVersion = rustc --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Rust installed successfully: $rustVersion" -ForegroundColor Green
    } else {
        Write-Host "  ERROR: Rust installation failed" -ForegroundColor Red
        Write-Host "  Please restart your terminal and try again" -ForegroundColor Yellow
        exit 1
    }
}

# =============================================================================
# 2. Install Visual C++ Build Tools (required for Rust on Windows)
# =============================================================================
Write-Host ""
Write-Host "[2/3] Checking for Visual C++ Build Tools..." -ForegroundColor Yellow

# Check if cl.exe is available
$clPath = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($clPath) {
    Write-Host "  Visual C++ Build Tools found" -ForegroundColor Green
} else {
    Write-Host "  Visual C++ Build Tools not found" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  To install, you have two options:" -ForegroundColor Cyan
    Write-Host "  Option 1: Run this command (requires winget):" -ForegroundColor Gray
    Write-Host "    winget install Microsoft.VisualStudio.2022.BuildTools" -ForegroundColor White
    Write-Host ""
    Write-Host "  Option 2: Download manually:" -ForegroundColor Gray
    Write-Host "    https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor White
    Write-Host "    Select 'Desktop development with C++' workload" -ForegroundColor White
    Write-Host ""
    Write-Host "  After installing, restart your terminal and run this script again." -ForegroundColor Yellow
    
    $response = Read-Host "  Do you want to try installing with winget now? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "  Installing Visual Studio Build Tools..."
        winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        Write-Host "  Please restart your terminal after installation completes" -ForegroundColor Yellow
    }
}

# =============================================================================
# 3. Install pyroscope-io
# =============================================================================
Write-Host ""
Write-Host "[3/3] Installing pyroscope-io..." -ForegroundColor Yellow

# Ensure PATH includes Rust
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

# Try to install pyroscope-io
pip install pyroscope-io 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " pyroscope-io installed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Verify with: python -c `"import pyroscope; print('OK')`"" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host " Installation failed" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  1. Visual C++ Build Tools not installed" -ForegroundColor Gray
    Write-Host "  2. Rust not in PATH (restart terminal)" -ForegroundColor Gray
    Write-Host "  3. Missing Windows SDK" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Try running 'Developer Command Prompt for VS 2022' and" -ForegroundColor Cyan
    Write-Host "then run: pip install pyroscope-io" -ForegroundColor Cyan
}
