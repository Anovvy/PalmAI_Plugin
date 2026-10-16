# PalmAI Verification & Test Matrix

This document defines the automated and manual testing tiers for PalmAI, covering unit, integration, synthetic geospatial, container smoke, and hardware-specific validation.

---

## 1. Test Tier Breakdown

| Tier | Test Scope | Triggers | Environment Requirements | Typical Runtime |
|---|---|---|---|:---:|
| **Tier 1: Static** | Ruff linting, PEP 8 compliance, QGIS plugin `metadata.txt` syntax | Every PR & commit | Python 3.9+ | < 5s |
| **Tier 2: Core & Backend** | Path mapping, mocked Docker daemon, venv interpreter discovery, `@@PALMAI@@` worker IPC, SHA-256 model store | Every PR & commit | Python standard library + pytest | < 5s |
| **Tier 3: Synthetic Spatial** | Affine transformations, bounding box & centroid math, Fiona/GeoPandas Shapefile export, road skeleton topology | Every PR & commit | GDAL/Rasterio, Shapely, GeoPandas, scikit-image (No real weights) | < 15s |
| **Tier 4: Container Smoke** | Build CPU image, run `python -m palmai_core.worker --check` in container | PRs modifying docker, core, or workflows | Docker daemon | ~3–5 min |
| **Tier 5: Checkpoint E2E** | Full raster inference using official `models-v1` weights (~500 MB) on demo rasters | Pre-release / manual QA | Full environment + model weights | ~2–10 min |
| **Tier 6: GPU Hardware** | NVIDIA CUDA tensor acceleration, TensorRT/CUDA memory allocation | Pre-release / self-hosted | NVIDIA GPU + driver + Container Toolkit | Hardware dependent |

---

## 2. Host Platform Support Matrix

| Host Platform | Supported Backends | Acceleration Mode | Verification Status in CI |
|---|---|---|:---:|
| **Windows 10 / 11 (x64)** | Docker Desktop, Virtualenv | CPU, NVIDIA CUDA (WSL2) | **Automated** (Tiers 1–3 in GitHub Actions) |
| **macOS (Apple Silicon)** | Virtualenv, Docker Desktop | Apple Metal (MPS via venv), CPU (Docker) | **Mocked / Unit Tested** |
| **Linux (Ubuntu 22.04 / 24.04)** | Docker Engine, Virtualenv | CPU, NVIDIA CUDA (NVIDIA Container Toolkit) | **Automated** (Tiers 1–4 in GitHub Actions) |

---

## 3. Local Developer Testing Commands

### Run Full Pytest Suite:
```bash
python -m pytest -v tests/
```

### Run Ruff Code Quality Checks:
```bash
ruff check plugin_qgis/backends/ plugin_qgis/palmai_core/ scripts/ tests/
```

### Validate QGIS Plugin Packaging:
```bash
python scripts/package_plugin.py --dist-dir dist
```

### Test Container Environment (Docker):
```bash
# Build local CPU test image
docker build -f plugin_qgis/docker/Dockerfile.cpu -t palmai-core:test plugin_qgis

# Run internal dependency and worker probe
docker run --rm palmai-core:test python -m palmai_core.worker --check
```
