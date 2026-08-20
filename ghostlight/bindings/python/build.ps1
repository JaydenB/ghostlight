#Requires -Version 5.1
<#
.SYNOPSIS
    Build the Ghostlight Python extension wheel on Windows.

.PARAMETER Install
    After building, install the wheel into the current Python environment.

.PARAMETER CudaArchitectures
    Semicolon-separated CUDA architecture list (e.g. "86;89").
    Defaults to the broad range defined in the root CMakeLists.txt.

.PARAMETER CudaToolkit
    CUDA toolkit version to build against, e.g. "11.8". Only meaningful with
    the Visual Studio generator, which takes CUDA from its MSBuild toolset and
    ignores CMAKE_CUDA_COMPILER/CUDACXX. Without this, CMake selects whatever
    CUDA_PATH points at, which is not necessarily the nvcc first on PATH when
    several toolkits are installed.

.PARAMETER Jobs
    Parallel build jobs passed to cmake --build (default: logical CPU count).

.EXAMPLE
    .\build.ps1
    .\build.ps1 -Install
    .\build.ps1 -CudaArchitectures "89" -Install
    .\build.ps1 -CudaToolkit "11.8" -Install
#>
param(
    [switch]$Install,
    [string]$CudaArchitectures = "",
    [string]$CudaToolkit = "",
    [int]$Jobs = [Environment]::ProcessorCount
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Step { param([string]$msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Abort      { param([string]$msg) Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------------------
# Check Python
# ---------------------------------------------------------------------------
Write-Step "Checking Python"
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) { Abort "python not found on PATH" }

$pyVer = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "  Python $pyVer at $($pythonCmd.Source)"
if ([version]$pyVer -lt [version]"3.9") { Abort "Python 3.9+ required (found $pyVer)" }

# ---------------------------------------------------------------------------
# Check CMake
# ---------------------------------------------------------------------------
Write-Step "Checking CMake"
$cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmakeCmd) { Abort "cmake not found on PATH. Install from https://cmake.org/download/" }
Write-Host "  $(cmake --version | Select-Object -First 1)"

# ---------------------------------------------------------------------------
# Install build dependencies
# ---------------------------------------------------------------------------
Write-Step "Installing build dependencies"
& python -m pip install --quiet --upgrade pip
& python -m pip install --quiet "scikit-build-core>=0.8" "pybind11>=2.12" "numpy>=1.23"
if ($LASTEXITCODE -ne 0) { Abort "pip install of build dependencies failed" }

# ---------------------------------------------------------------------------
# Set optional CUDA overrides
# ---------------------------------------------------------------------------
# Whatever the caller already placed in CMAKE_ARGS is preserved and extended,
# so arbitrary CMake options stay reachable through this script.
$cmakeArgs = @()
if ($CudaArchitectures -ne "") {
    $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
    Write-Host "  CUDA architectures: $CudaArchitectures"
}
if ($CudaToolkit -ne "") {
    # A toolset spec, not -D: the Visual Studio generator resolves CUDA through
    # its MSBuild integration and ignores CMAKE_CUDA_COMPILER entirely.
    $cmakeArgs += "-T cuda=$CudaToolkit"
    Write-Host "  CUDA toolkit: $CudaToolkit"
}

if ($cmakeArgs.Count -gt 0) {
    $joined = $cmakeArgs -join " "
    if ([string]::IsNullOrWhiteSpace($env:CMAKE_ARGS)) {
        $env:CMAKE_ARGS = $joined
    } else {
        $env:CMAKE_ARGS = "$($env:CMAKE_ARGS) $joined"
    }
}
if (-not [string]::IsNullOrWhiteSpace($env:CMAKE_ARGS)) {
    Write-Host "  CMAKE_ARGS: $($env:CMAKE_ARGS)"
}

# ---------------------------------------------------------------------------
# Build wheel
# ---------------------------------------------------------------------------
Write-Step "Building wheel"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

New-Item -ItemType Directory -Force -Path dist | Out-Null
& python -m pip wheel . --no-deps -w dist --no-build-isolation

if ($LASTEXITCODE -ne 0) { Abort "pip wheel failed" }

$wheel = Get-ChildItem dist\ghostlight*.whl | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $wheel) { Abort "No wheel found in dist\" }
Write-Host "`nBuilt: $($wheel.Name)" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Optionally install
# ---------------------------------------------------------------------------
if ($Install) {
    Write-Step "Installing into current environment"
    # --no-deps avoids re-uninstalling numpy etc., which can fail with
    # WinError 5 when pip can't replace files like Scripts\f2py.exe held
    # by a system-wide Python install. Build deps are already ensured above.
    & python -m pip install --force-reinstall --no-deps $wheel.FullName
    if ($LASTEXITCODE -ne 0) { Abort "pip install failed" }
    Write-Host "Installed $($wheel.Name)" -ForegroundColor Green
}

Write-Host "`nDone." -ForegroundColor Green
