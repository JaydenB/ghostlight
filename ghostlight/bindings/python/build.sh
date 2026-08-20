#!/usr/bin/env bash
# Build the Ghostlight Python extension wheel on Linux/macOS.
#
# Usage:
#   bash build.sh [--install] [--cuda-architectures "86;89"] [--jobs N]
#
# Options:
#   --install                 Install the wheel after building.
#   --cuda-architectures STR  Semicolon-separated CUDA arch list (e.g. "86;89").
#   --jobs N                  Parallel build jobs (default: nproc).

set -euo pipefail

INSTALL=0
CUDA_ARCHS=""
JOBS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --install)            INSTALL=1 ;;
        --cuda-architectures) CUDA_ARCHS="$2"; shift ;;
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
# Set optional CUDA architecture override
# ---------------------------------------------------------------------------
if [[ -n "$CUDA_ARCHS" ]]; then
    export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
    echo "  CUDA architectures: $CUDA_ARCHS"
fi

# ---------------------------------------------------------------------------
# Build wheel
# ---------------------------------------------------------------------------
step "Building wheel"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p dist
"$python_cmd" -m pip wheel . --no-deps -w dist --no-build-isolation

wheel=$(ls dist/ghostlight-*.whl 2>/dev/null | sort -t- -k2 | tail -1)
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
