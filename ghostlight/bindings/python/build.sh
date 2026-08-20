#!/usr/bin/env bash
# Build the Ghostlight Python extension wheel on Linux/macOS.
#
# Usage:
#   bash build.sh [--install] [--cuda-architectures "86;89"]
#                 [--cuda-toolkit /usr/local/cuda-11.8] [--jobs N]
#
# Options:
#   --install                 Install the wheel after building.
#   --cuda-architectures STR  Semicolon-separated CUDA arch list (e.g. "86;89").
#   --cuda-toolkit PATH       Build against a specific CUDA toolkit, given as
#                             its root directory or its nvcc. Without this,
#                             CMake takes the first nvcc on PATH, which is not
#                             necessarily the toolkit CUDA_HOME/CUDA_PATH names
#                             when several are installed.
#   --jobs N                  Parallel build jobs (default: nproc).
#
# Anything already exported in CMAKE_ARGS is preserved and extended.

set -euo pipefail

INSTALL=0
CUDA_ARCHS=""
CUDA_TOOLKIT=""
JOBS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --install)            INSTALL=1 ;;
        --cuda-architectures) CUDA_ARCHS="$2"; shift ;;
        --cuda-toolkit)       CUDA_TOOLKIT="$2"; shift ;;
        --jobs)               JOBS="$2";       shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

step() { echo -e "\n==> $*"; }
abort() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Check Python
# ---------------------------------------------------------------------------
step "Checking Python"
python_cmd=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
[[ -z "$python_cmd" ]] && abort "python3 not found on PATH"

py_ver=$("$python_cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python $py_ver at $python_cmd"
python3 -c "import sys; assert sys.version_info >= (3,9), 'Python 3.9+ required'" \
    || abort "Python 3.9+ required (found $py_ver)"

# ---------------------------------------------------------------------------
# Check CMake
# ---------------------------------------------------------------------------
step "Checking CMake"
command -v cmake >/dev/null 2>&1 || abort "cmake not found on PATH"
cmake --version | head -1

# ---------------------------------------------------------------------------
# Install build dependencies
# ---------------------------------------------------------------------------
step "Installing build dependencies"
"$python_cmd" -m pip install --quiet --upgrade pip
"$python_cmd" -m pip install --quiet "scikit-build-core>=0.8" "pybind11>=2.12" "numpy>=1.23"

# ---------------------------------------------------------------------------
# Set optional CUDA overrides
# ---------------------------------------------------------------------------
# Extend CMAKE_ARGS rather than overwrite it, so a caller-supplied value and
# arbitrary CMake options stay reachable through this script.
extra_args=""
if [[ -n "$CUDA_ARCHS" ]]; then
    extra_args="${extra_args} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
    echo "  CUDA architectures: $CUDA_ARCHS"
fi

if [[ -n "$CUDA_TOOLKIT" ]]; then
    # Accept either a toolkit root or the nvcc binary itself.
    if [[ -d "$CUDA_TOOLKIT" ]]; then
        nvcc_path="${CUDA_TOOLKIT%/}/bin/nvcc"
    else
        nvcc_path="$CUDA_TOOLKIT"
    fi
    [[ -x "$nvcc_path" ]] || abort "no executable nvcc at '$nvcc_path'"
    # The Make and Ninja generators honour CUDACXX. (The Visual Studio
    # generator does not - see build.ps1 -CudaToolkit for that case.)
    export CUDACXX="$nvcc_path"
    # pipefail would abort the build if this cosmetic parse ever missed.
    cuda_ver=$("$nvcc_path" --version | grep -o 'release [0-9.]*' | cut -d' ' -f2 || echo "unknown")
    echo "  CUDA toolkit: ${cuda_ver} ($nvcc_path)"
fi

if [[ -n "$extra_args" ]]; then
    export CMAKE_ARGS="${CMAKE_ARGS:-}${extra_args}"
fi
if [[ -n "${CMAKE_ARGS:-}" ]]; then
    echo "  CMAKE_ARGS:${CMAKE_ARGS}"
fi

# ---------------------------------------------------------------------------
# Build wheel
# ---------------------------------------------------------------------------
step "Building wheel"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p dist
"$python_cmd" -m pip wheel . --no-deps -w dist --no-build-isolation

wheel=$(ls dist/ghostlight*.whl 2>/dev/null | sort -t- -k2 | tail -1)
[[ -z "$wheel" ]] && abort "No wheel found in dist/"
echo -e "\nBuilt: $(basename "$wheel")"

# ---------------------------------------------------------------------------
# Optionally install
# ---------------------------------------------------------------------------
if [[ "$INSTALL" -eq 1 ]]; then
    step "Installing into current environment"
    # --no-deps avoids re-uninstalling numpy etc., which can fail on
    # locked/system-managed Python installs. Build deps are already ensured above.
    "$python_cmd" -m pip install --force-reinstall --no-deps "$wheel"
    echo "Installed $(basename "$wheel")"
fi

echo -e "\nDone."
