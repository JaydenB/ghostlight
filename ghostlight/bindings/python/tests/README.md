# Ghostlight Test Suite

## Requirements

- Python ≥ 3.9
- `ghostlight` package installed (see `bindings/python/README.md` for build instructions)
- `pytest` and `numpy`: `pip install pytest numpy`
- CUDA GPU + drivers for GPU tests (optional — CPU-only tests skip GPU tests automatically)

---

## Running Tests

### CPU tests only (no GPU required)
```
cd bindings/python
pytest tests/ -m "not gpu" -v
```

### All tests (requires CUDA GPU)
```
cd bindings/python
pytest tests/ -v
```

### Single section
```
pytest tests/test_math.py -v
pytest tests/test_physics.py -v
```

### Verbose with full tracebacks
```
pytest tests/ --tb=long -v
```

### Stop on first failure
```
pytest tests/ -x -v
```

### Run only GPU tests
```
pytest tests/ -m gpu -v
```

---

## Test Sections

Test modules follow a `test_<area>.py` naming convention — each file covers
`<area>`, so the listing under `tests/` is self-documenting (e.g.
`test_trace.py` covers ray tracing, `test_calibration.py` covers calibration,
`test_render_psf.py` covers the PSF renderer). Browse `tests/test_*.py` for the
full, current set.

GPU-dependent modules mark their tests with `@pytest.mark.gpu` (see Markers
below) and are skipped automatically when no CUDA device is present.

---

## Markers

- `@pytest.mark.gpu` — requires a CUDA-capable GPU. Skipped automatically if none is detected via `ghostlight._cuda_available()`.

---

## Test Design Notes

### CPU tests (no `@pytest.mark.gpu`)
These exercise the Python bindings and C++ CPU code: math, optics, ray tracing, lens loading, spectral system, calibration, and ghost enumeration. They run anywhere and are fast (< 10s total).

### GPU tests
These call the CUDA render pipeline (`render_point_flare`, `render_source_flare`, `render_psf`). They are automatically skipped on machines without a compatible GPU.

### Physical invariants tested
- Normal dispersion: IOR(F-line) > IOR(d-line) > IOR(C-line) for glass
- Fresnel transmittance matches `1 - ((n-1)/(n+1))^2` at normal incidence
- Ghost weight < primary weight (double reflection loses energy)
- On-axis ghost lands near the optical axis
- More surfaces → lower cumulative transmittance
- Render output is deterministic (same inputs → identical pixels)

---

## Adding New Tests

1. Create a file in `tests/` following the naming pattern above, or extend an existing file.
2. Add `@pytest.mark.gpu` to any test that calls a render function.
3. Add shared fixtures to `conftest.py`.
4. Verify locally:
   ```
   pytest tests/your_new_file.py -v
   ```
