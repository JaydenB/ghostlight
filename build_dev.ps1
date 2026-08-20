#Requires -Version 5.1
<#
.SYNOPSIS
    Local dev build: compile the Ghostlight Python extension and copy the .pyd
    into the source tree so you can run the tests and the validation harness
    without installing a wheel.

.PARAMETER Config
    MSBuild configuration: Release (default) or Debug.

.PARAMETER CudaArchitectures
    Semicolon-separated CUDA arch list (e.g. "86;89"). Only used on first
    configure; ignored if the build directory already exists.

.PARAMETER Jobs
    Parallel compile jobs (default: logical CPU count).

.EXAMPLE
    .\build_dev.ps1
    .\build_dev.ps1 -Config Debug
    .\build_dev.ps1 -CudaArchitectures "89"
#>
param(
    [string]$Config             = "Release",
    [string]$CudaArchitectures  = "86;89",
    [int]   $Jobs               = [Environment]::ProcessorCount
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root      = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir  = Join-Path $Root "build"
$SourceDir = Join-Path $Root "ghostlight\bindings\python"
$PkgDir    = Join-Path $Root "ghostlight\bindings\python\ghostlight"

function Write-Step { param([string]$msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Abort      { param([string]$msg) Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------------------
# Check tools
# ---------------------------------------------------------------------------
Write-Step "Checking tools"
if (-not (Get-Command cmake  -ErrorAction SilentlyContinue)) { Abort "cmake not found on PATH" }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Abort "python not found on PATH" }
Write-Host "  $(cmake --version | Select-Object -First 1)"
$pyVer = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "  Python $pyVer"

# ---------------------------------------------------------------------------
# Configure (only when build dir is missing or cache is gone)
# ---------------------------------------------------------------------------
$cacheFile = Join-Path $BuildDir "CMakeCache.txt"
if (-not (Test-Path $cacheFile)) {
    Write-Step "Configuring (first-time)"

    # find_package(pybind11 CONFIG) only resolves when pip's pybind11 CMake
    # package dir is on CMAKE_PREFIX_PATH. scikit-build-core adds it when
    # building a wheel, but a bare cmake configure like this one does not, so
    # ask the installed module where it lives. Without this, a first configure
    # (i.e. any run after deleting the build directory) fails on pybind11_DIR.
    $pybindDir = & python -m pybind11 --cmakedir 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $pybindDir) {
        Abort "pybind11 not importable - run: python -m pip install pybind11"
    }
    $pybindDir = $pybindDir.Trim()
    Write-Host "  pybind11: $pybindDir"

    $configArgs = @(
        "-S", $SourceDir,
        "-B", $BuildDir,
        "-DCMAKE_BUILD_TYPE=$Config",
        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
        "-Dpybind11_DIR=$pybindDir"
    )
    & cmake @configArgs
    if ($LASTEXITCODE -ne 0) { Abort "cmake configure failed" }
} else {
    Write-Host "`n  Build directory already configured - skipping cmake configure."
    Write-Host "  (Delete '$BuildDir' and re-run to reconfigure.)"
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
Write-Step "Building (_ghostlight extension)"
& cmake --build $BuildDir --config $Config --parallel $Jobs
if ($LASTEXITCODE -ne 0) { Abort "cmake build failed" }

# ---------------------------------------------------------------------------
# Find the .pyd (filename encodes the Python ABI tag, e.g. cp310-win_amd64)
# ---------------------------------------------------------------------------
Write-Step "Copying .pyd to package directory"
$pydFiles = Get-ChildItem (Join-Path $BuildDir $Config) -Filter "_ghostlight*.pyd" -ErrorAction SilentlyContinue
if (-not $pydFiles) {
    # Some generators drop directly into BuildDir/Release even with Debug config
    $pydFiles = Get-ChildItem $BuildDir -Filter "_ghostlight*.pyd" -ErrorAction SilentlyContinue
}
if (-not $pydFiles) { Abort "No _ghostlight*.pyd found under '$BuildDir'" }

$pyd = $pydFiles | Sort-Object LastWriteTime | Select-Object -Last 1
$dst = Join-Path $PkgDir $pyd.Name

Copy-Item $pyd.FullName $dst -Force
Write-Host "  $($pyd.FullName)"
Write-Host "  -> $dst" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
Write-Step "Smoke test"
$env:PYTHONPATH = $SourceDir
& python -c 'import ghostlight; from ghostlight._ghostlight import render_point_flare, render_source_flare'
$importOk = $LASTEXITCODE
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
if ($importOk -ne 0) { Abort "Module import failed after copy" }
Write-Host "  ghostlight imported successfully" -ForegroundColor Green

Write-Host "`nDone. Run the tests with:" -ForegroundColor Green
Write-Host "  cd $SourceDir"
Write-Host "  python -m pytest"
Write-Host "`nOr the render value-oracle gate:" -ForegroundColor Green
Write-Host "  cd $Root"
Write-Host "  python validation\aperture_baseline.py"
