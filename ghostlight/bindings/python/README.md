# Ghostlight Python Bindings

Python module exposing the Ghostlight C++/CUDA lens-flare renderer.  Built with
**pybind11** and packaged via **scikit-build-core** so the existing CMake +
CUDA toolchain is reused without any wrapper translation layer.

---

## Prerequisites

| Tool | Minimum version | Notes |
|---|---|---|
| Python | 3.9 | 3.10–3.12 recommended |
| CUDA Toolkit | 11.8 | Must match the GPU driver; 12.x preferred |
| MSVC (Windows) | VS 2019 (v16) | Or VS 2022 — must include C++ and CUDA workloads |
| GCC (Linux) | 9 | Tested with 11 and 12 |
| CMake | 3.18 | Must be on `PATH` |
| Ninja | any | Optional but faster than MSBuild/Make |

Python build dependencies are installed automatically by the build scripts
below — you do not need to pre-install pybind11 or scikit-build-core manually.

---

## Building

### Windows

```powershell
cd bindings\python
.\build.ps1
```

This installs build dependencies, compiles the extension, and places the wheel
in `dist\`.  To also install it into the current Python environment:

```powershell
.\build.ps1 -Install
```

### Linux

```bash
cd bindings/python
bash build.sh
```

Same flags apply:

```bash
bash build.sh --install
```

---

## Development (editable) install

An editable install lets you modify the pure-Python files (`ghostlight/*.py`)
without rebuilding.  The C++ extension is still compiled once and linked in
place.

```powershell
# Windows — from bindings/python/
python -m pip install -e . --no-build-isolation
```

```bash
# Linux — from bindings/python/
python -m pip install -e . --no-build-isolation
```

`--no-build-isolation` is required when you have already installed the build
dependencies (scikit-build-core, pybind11, numpy) into the same environment.

---

## Running tests

```powershell
# from bindings/python/
python -m pytest tests/ -v
```

Tests that require a CUDA GPU are marked `@pytest.mark.gpu` and are
automatically skipped when no compatible device is detected.

```powershell
# Run only non-GPU tests explicitly
python -m pytest tests/ -v -m "not gpu"

# Run only GPU tests
python -m pytest tests/ -v -m gpu
```

---

## Controlling the CUDA architecture

By default the build targets a broad range of GPU architectures (Maxwell
through Blackwell).  To speed up local builds, restrict to your GPU's
architecture:

```powershell
# RTX 3090 / 4090 — sm_86 / sm_89
$env:CMAKE_ARGS = "-DCMAKE_CUDA_ARCHITECTURES=86"
python -m pip wheel .
```

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" python -m pip wheel .
```

---

## Wheel layout

The produced wheel contains:

```
ghostlight/
  __init__.py          # re-exports + Lens wrapper
  lens.py              # Lens cache wrapper
  _arrays.py           # optional HWC reshape helpers
  _ghostlight.pyd        # compiled C++/CUDA extension (Windows)
  # or _ghostlight.so    # (Linux)
```

---

## Troubleshooting

**`cmake` not found**  
Add the CMake `bin/` directory to `PATH`, or install via `winget install Kitware.CMake`.

**CUDA version mismatch**  
Run `nvcc --version` and `nvidia-smi` to confirm the toolkit and driver versions
agree.  The driver must be at least as new as the toolkit.

**`No module named _ghostlight`**  
The extension compiled but is not on `sys.path`.  Run from the `bindings/python/`
directory or install with `pip install .` first.

**MSVC `C1083` / missing headers**  
Open the solution in Visual Studio and ensure the "Desktop development with C++"
and "CUDA" workloads are both installed via the Visual Studio Installer.
