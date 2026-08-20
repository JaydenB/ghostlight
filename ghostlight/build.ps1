param(
    [string]$BuildType = "Release",
    [string]$BuildDir  = "build"
)

$ErrorActionPreference = "Stop"

Write-Host "Configuring ($BuildType)..."
cmake -S . -B $BuildDir -DCMAKE_BUILD_TYPE=$BuildType
if (-not $?) { exit 1 }

Write-Host "Building..."
cmake --build $BuildDir --config $BuildType
if (-not $?) { exit 1 }

Write-Host "Done. Output in: $BuildDir"
