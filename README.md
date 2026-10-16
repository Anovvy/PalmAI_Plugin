# **PalmAI QGIS Plugin**
## A Deep Learning-powered QGIS plugin for automated feature extraction from oil palm plantation drone imagery.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-34%20Passed-brightgreen.svg)](tests/)
[![Inference: Docker](https://img.shields.io/badge/Inference-Docker%20Container-blue.svg)](https://www.docker.com/)
[![Python: 3.9--3.12](https://img.shields.io/badge/Python-3.9%20|%203.11%20|%203.12-green.svg)](https://www.python.org/)
[![QGIS: 3.22--3.99](https://img.shields.io/badge/QGIS-3.22%20to%203.99%20LTR-darkgreen.svg)](https://qgis.org/)

---

## **Description**
PalmAI automates the spatial analysis and data processing workflow for oil palm plantations, replacing manual digitization with state-of-the-art Deep Learning models. By combining **YOLOv8** object detection and **DeepLabV3+** semantic segmentation with specialized geospatial post-processing, PalmAI delivers rapid, reliable, and operationally accurate vector outputs directly into QGIS.

To eliminate dependency conflicts ("dependency hell") across different operating systems and QGIS environments, PalmAI uses an isolated **decoupled architecture**:
- **QGIS Plugin (Thin Client):** Manages user inputs, parameters, and renders vector/raster layers seamlessly in QGIS.
- **Inference Engine (Isolated Container / Virtualenv):** Runs PyTorch, CUDA, SAHI, and geospatial processing libraries in a dedicated, version-pinned environment with least-privilege volume mounts.

---

## **Main Features**

### 1. Tree Counting (YOLOv8)
Automated detection and enumeration of oil palm trees with Shapefile outputs (Bounding Boxes & Centroid Points).

https://github.com/user-attachments/assets/98feab8e-b895-4ee3-b47f-22ab097fc2bc

### 2. Center Tree (YOLOv8)
High-precision crown centroid extraction for precision agriculture navigation and targeted interventions (e.g., spraying *Oryctes rhinoceros* pests).

https://github.com/user-attachments/assets/34584588-0f9d-404b-8fed-267057f0ef7f

### 3. Tree Classification (YOLOv8)
Multi-class condition assessment categorizing trees into three health statuses (**Healthy**, **Yellowish**, and **Dead**) with distinct symbology.

https://github.com/user-attachments/assets/1938980e-e6e4-4759-9d31-44d6a70f6ded

### 4. Road Detection (DeepLabV3+)
Canopy-resistant semantic segmentation and topological vectorization that extracts plantation road networks into polyline Shapefiles using morphological skeletonization and A* pathfinding.

https://github.com/user-attachments/assets/643d65d7-d6e7-4b14-a943-de487a838b0d

---

## **Repository Structure**
```text
PalmAI_Plugin/
├── plugin_qgis/              # QGIS Plugin package (Thin UI Client)
│   ├── metadata.txt          # Validated QGIS plugin metadata (v1.0.0, QGIS 3.22-3.99)
│   ├── palmai_plugin.py      # Plugin entry point & menu setup
│   ├── icon.png              # Plugin manager icon
│   ├── backends/             # Execution backends (Docker, Virtualenv, PathMapper)
│   ├── palmai_core/          # Core pipelines (YOLOv8, DeepLabV3+, Worker, ModelStore, Provenance)
│   ├── docker/               # Dockerfiles (CPU/CUDA GPU) & pinned requirements.lock
│   ├── models/               # Model weights directory (SHA-256 verified & auto-downloaded)
│   ├── scripts/              # UI dialog handlers (Counting, Center Tree, Classification, Road)
│   ├── ui/                   # Qt Designer UI forms
│   └── icons/                # Interface assets
├── tests/                    # Deterministic automated test suite (34 passing tests)
│   ├── test_docker_backend.py      # Docker daemon diagnostics, pull, fallback, and GPU probe
│   ├── test_geospatial_core.py     # Affine transforms, bboxes, centroids, and Shapefiles
│   ├── test_geospatial_qa.py       # Road skeleton topology and run_manifest provenance
│   ├── test_model_store.py         # Cryptographic SHA-256 checksums, retries, and atomic .part
│   ├── test_path_mapping.py        # Least-privilege mounts (:ro inputs, :rw outputs), unicode paths
│   ├── test_plugin_metadata.py     # QGIS metadata INI syntax, URL validity, and LTR versions
│   ├── test_venv_backend.py        # Standalone Python detection and PyTorch wheel indexes
│   └── test_worker_protocol.py     # @@PALMAI@@ JSON-lines IPC protocol and error relay
├── scripts/
│   └── package_plugin.py     # Release packager creating PalmAI.zip (excluding large weights)
├── demo_data/                # Sample orthophoto raster guidance
├── notebooks/                # R&D training notebooks, tuning history, and model genesis
├── docs/                     # Technical architecture documentation
│   ├── ci-cd.md              # CI/CD quality gates, workflows, and GHCR delivery
│   ├── release-policy.md     # SemVer, compatibility matrix, and model hashes
│   └── test-matrix.md        # Test execution tiers and platform matrix
├── .github/workflows/        # Automated GitHub Actions pipelines
│   ├── ci.yml                # PR quality gate: Ruff, pytest matrix, Docker CPU smoke
│   ├── publish-images.yml    # Versioned container publisher (vX.Y.Z-cpu/gpu to GHCR)
│   └── release-plugin.yml    # Plugin ZIP packager & GitHub Release asset publisher
├── pyproject.toml            # Project configuration (pytest & ruff linter settings)
├── requirements-dev.txt      # Development and testing dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## **Tech Stack**
- **Deep Learning:** PyTorch (v2.5.1), YOLOv8 Nano, DeepLabV3+ ResNet50, SAHI (Slicing Aided Hyper Inference)
- **Containerization & Delivery:** Docker, GitHub Container Registry (GHCR), OCI Labels
- **Geospatial Processing:** PyQGIS, GeoPandas, Rasterio, Shapely, Fiona, scikit-image
- **GUI:** PyQt5 / Qt Designer (QGIS 3.x LTR)
- **Quality & Testing:** Pytest, Ruff, GitHub Actions CI/CD

---

## **User Guide**
For an in-depth operational guide covering drone data flight planning, radiometric calibration, and parameter optimization, refer to the [PalmAI User Guide](https://drive.google.com/file/d/1pgkmoNwnQgRRP3JbT9n_loTzG5tLFDIz/view?usp=sharing).

---

## **Installation & Quickstart Guide**

### **Prerequisites**
1. **[QGIS](https://qgis.org/) 3.x** (v3.22 up to v3.x LTR, e.g., v3.34, v3.38, v3.44):
   - **Important:** PalmAI is designed exclusively for **QGIS 3.x**. The upcoming **QGIS 4.x is currently NOT supported** due to major architectural changes (Qt6/PyQt6 migration, Python API redesign, and removal of deprecated QGIS 3 APIs).
2. **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (Recommended inference backend):
   - Needs to be installed and **running** (daemon active in the system tray) when executing PalmAI tasks.
   - *Windows Users:* Requires WSL 2 (Windows Subsystem for Linux). Automatically supports both per-user and system-wide Docker Desktop installations.
   - *macOS Users:* Docker runs in CPU mode. (Or use the Virtualenv backend for native Apple Silicon MPS).
   - *(Alternative)*: If you do not want to install Docker, you can use the built-in **Virtualenv backend** instead (requires standalone Python 3.9–3.12, see [Step 2](#step-2-configure-inference-backend)).

<details>
<summary><b>Docker Desktop Setup Guide for Windows (Click to expand)</b></summary>

Setting up Docker Desktop on Windows for the first time? Follow these quick steps to get started smoothly:

1. **Verify Virtualization:** Ensure Hardware Virtualization (VT-x / AMD-V) is enabled in your BIOS/UEFI. Check this in Windows Task Manager > **Performance** > **CPU** > **Virtualization: Enabled**.
2. **Install WSL 2 (Windows Subsystem for Linux):**
   - Open **PowerShell** as Administrator.
   - Run:
     ```powershell
     wsl --install
     ```
   - If WSL is already installed, update it to the latest version:
     ```powershell
     wsl --update
     ```
   - Restart your computer if prompted.
   - *Official Reference:* [Microsoft WSL Installation Guide](https://learn.microsoft.com/en-us/windows/wsl/install).
3. **Download & Install Docker Desktop:**
   - Download the installer from the [Official Docker Desktop for Windows Guide](https://docs.docker.com/desktop/setup/install/windows-install/).
   - Run `Docker Desktop Installer.exe` and make sure the option **"Use WSL 2 instead of Hyper-V"** is checked.
4. **Skip Account Sign-In / Survey (Optional):**
   - When Docker Desktop launches and prompts you to sign in or create an account, **you do NOT need to create an account** for personal/local usage.
   - Simply click **"Continue without signing in"** (or "Skip"), then skip the survey.
5. **Keep Docker Desktop Running:**
   - Docker Desktop must be running (showing a green whale icon in the Windows taskbar/system tray) whenever you execute AI inference in QGIS.
   - *Tip:* In Docker Desktop, go to **Settings (Gear Icon) > General** and check **"Start Docker Desktop when you log in"** so it automatically starts in the background without needing manual launch every time.

</details>

<details>
<summary><b>Docker Desktop Setup Guide for macOS (Click to expand)</b></summary>

Follow these steps to set up Docker Desktop on macOS:

1. **Identify Your Mac Processor:**
   - Click the Apple logo () in the top-left corner > **About This Mac**.
   - Check whether your Mac has an **Apple Chip** (M1, M2, M3, M4) or an **Intel Processor**.
2. **Download & Install Docker Desktop:**
   - Visit the [Official Docker Desktop for Mac Guide](https://docs.docker.com/desktop/setup/install/mac-install/).
   - Click **"Mac with Apple chip"** or **"Mac with Intel chip"** according to your hardware.
   - Double-click the downloaded `Docker.dmg`, then drag and drop the **Docker** icon into your **Applications** folder.
   - Open **Docker** from Applications (or Spotlight `Cmd + Space`).
   - When macOS prompts for system permissions, authorize with your Mac password or Touch ID to allow Docker to install its helper tools.
3. **Skip Account Sign-In / Survey (Optional):**
   - Click **"Continue without signing in"** (or "Skip") and dismiss the survey. A Docker account is not needed for local usage.
4. **Enable Rosetta 2 (Recommended for Apple Silicon):**
   - To ensure compatibility with x86/amd64 containers, install Apple's Rosetta 2 via Terminal:
     ```bash
     softwareupdate --install-rosetta
     ```
   - In Docker Desktop, go to **Settings (Gear Icon) > General**, and verify that **"Use Rosetta for x86/amd64 emulation on Apple Silicon"** is enabled.
5. **Keep Docker Desktop Running:**
   - Docker must be active (look for the whale icon in the top macOS menu bar) before running PalmAI tasks.
   - *Tip:* In Docker Desktop **Settings > General**, check **"Start Docker Desktop when you log in"**.
6. **💡 Tip for Apple Silicon (M1/M2/M3/M4) - Apple Metal (MPS) Acceleration:**
   - Docker on macOS runs Linux containers in CPU mode.
   - If you want native **Apple Silicon GPU acceleration (Metal Performance Shaders / MPS)** for much faster inference, you can use PalmAI's built-in **Virtualenv backend** instead of Docker! See [Step 2: Configure Inference Backend](#step-2-configure-inference-backend).

</details>

---

### **Step 1: Install the Plugin in QGIS**

#### **Method A: Install via Official Release ZIP (Recommended)**
1. Go to the [Releases](https://github.com/Anovvy/PalmAI_Plugin/releases) page and download `PalmAI.zip`.
2. In QGIS, navigate to **Plugins > Manage and Install Plugins... > Install from ZIP**.
3. Select the downloaded `PalmAI.zip` and click **Install Plugin**.

#### **Method B: Manual / Repository Installation**
1. Clone or download this repository.
2. Copy the `plugin_qgis` folder into your QGIS plugins directory and rename it to `PalmAI`:
   - **Windows:**
     `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\PalmAI`  
     *(Full path: `C:\Users\<Your-Username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\PalmAI`)*  
     > *(For developers)*: You can create a Directory Junction so repo changes reflect instantly:  
     > `mklink /J "%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\PalmAI" "<path-to-repo>\plugin_qgis"`
   - **macOS:**
     `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/PalmAI`
   - **Linux:**
     `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/PalmAI`
3. In QGIS, open **Plugins > Manage and Install Plugins... > Installed**, and enable **PalmAI**.

---

### **Step 2: Configure Inference Backend**
Open **Plugins > PalmAI > Settings (Inference Backend)**.

| Backend Option | Recommended Use | Details |
|---|---|---|
| **Docker** *(Recommended)* | Default choice for all platforms | Fully isolated. Automatically detects Docker Desktop across Windows, macOS, and Linux. Pulls verified container images (`ghcr.io/anovvy/palmai-core:v2.0.0-cpu` or `:v2.0.0-gpu`). Applies least-privilege volume isolation (`:ro` inputs and weights, `:rw` outputs). Zero host Python package installation required. |
| **Virtualenv** | macOS (Apple Silicon MPS) or systems without Docker | Creates an isolated virtual environment outside QGIS. Requires standalone **Python 3.9–3.12** from [python.org](https://www.python.org/downloads/) on your host machine. Click **"Set up Python environment"** in the Settings dialog to initialize. |
| **Automatic** | Fallback mode | Checks Docker first, then Virtualenv, then QGIS Python. |

Click **Save** once configured.

---

### **Step 3: Testing & Running Tools**
To test the plugin using the sample demo datasets:
1. Open any PalmAI tool from the menu, e.g., **Plugins > PalmAI > Palm Oil Tree Counting**.
2. Set the tool inputs:
   - **Input Raster:** Select `demo_data/Demo_Data_1.tif` (download via [demo_data/README.md](demo_data/README.md)) or your own GeoTIFF.
   - **Output Bounding Box & Centroid:** Choose the target `.shp` save locations.
   - **Device:** Select `GPU (CUDA)` if an NVIDIA GPU is available, or `CPU`.
3. Click **Run**.
4. The task will execute in the background via Docker or Virtualenv.
5. Upon completion, the resulting Bounding Box polygons and Centroid points are automatically added and styled on the QGIS map canvas, accompanied by a reproducible `run_manifest.json`!

---

## **Cryptographic Model Weight Integrity & Verification**

Trained model weights are hosted on GitHub Releases under tag `models-v1`:  
👉 **[Official PalmAI Model Releases (tag: models-v1)](https://github.com/Anovvy/PalmAI_Plugin/releases/tag/models-v1)**

### **Security & Cryptographic Checksum Registry**
To safeguard users against corrupted downloads, partial writes, or tampered binary assets, PalmAI cryptographically verifies both cached and newly downloaded weights against authoritative SHA-256 digests:

| Model Filename | Task / Model | Size | Official SHA-256 Checksum | Direct Download |
|---|---|---|---|:---:|
| `yolov8n_counting_1.0.pt` | Tree Counting (YOLOv8) | ~6.3 MB | `2c5a4bec86354e808cbd3324fe36656c1a3e1be342ddc1797087e9c48592cfd8` | [Download](https://github.com/Anovvy/PalmAI_Plugin/releases/download/models-v1/yolov8n_counting_1.0.pt) |
| `yolov8n_centertree_1.3.pt` | Palm Oil Centre Tree (YOLOv8) | ~6.4 MB | `6ed11bab0e8fd3b61a7ab5d751b76d6096d958ce02d0870919667b4048ceef35` | [Download](https://github.com/Anovvy/PalmAI_Plugin/releases/download/models-v1/yolov8n_centertree_1.3.pt) |
| `yolov8n_classification_2.1.pt` | Condition Classification (YOLOv8) | ~6.3 MB | `05162683b0842e759d17e3448331ece0b5c12678c8d3c66f5d296e1d9b02f795` | [Download](https://github.com/Anovvy/PalmAI_Plugin/releases/download/models-v1/yolov8n_classification_2.1.pt) |
| `deeplabv3+_roaddetection_1.pth.tar` | Road Network Detection (DeepLabV3+) | ~504 MB | `6f8ec47d0d5be7573f128bb6f1ec53b6dd185fd90bcec2975c5efc451cea9ee0` | [Download](https://github.com/Anovvy/PalmAI_Plugin/releases/download/models-v1/deeplabv3%2B_roaddetection_1.pth.tar) |

### **Automatic Download & Integrity Handling**
- **First Run Verification:** If missing, PalmAI downloads weights to a `.part` temporary file with bounded retries and exponential backoff.
- **Atomic Promotion:** The model is only renamed to its final `.pt` or `.pth.tar` path after successfully passing the SHA-256 verification.
- **Fail Closed:** If a cached or downloaded weight file does not match the checksum, PalmAI removes the corrupt file, raises a descriptive error, and prevents unverified binary execution.

---

## **Execution Provenance: `run_manifest.json`**

Every executed PalmAI job automatically outputs a structured `run_manifest.json` in the output folder documenting:
- Plugin version, protocol version, and container image digest.
- Runtime environment details (Python, PyTorch, GDAL, Rasterio, Shapely versions).
- Input raster metadata (CRS, resolution/GSD, bounding box dimensions).
- Processing parameters and total feature counts.
- SHA-256 digests of all generated output Shapefile components (`.shp`, `.shx`, `.dbf`).
- **Privacy Protection:** Paths are automatically sanitized to prevent leaking host usernames or local filesystem structures.

---

## **Developer & Contributor Guide**

### Setting Up Development Environment
```bash
git clone https://github.com/Anovvy/PalmAI_Plugin.git
cd PalmAI_Plugin

# Create virtual environment and install dev dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### Running Test Suite
PalmAI contains 34 automated unit, integration, and synthetic geospatial tests that run in seconds without requiring real weights or GPU hardware:
```bash
python -m pytest -v tests/
```

### Running Code Quality Checks
```bash
ruff check plugin_qgis/backends/ plugin_qgis/palmai_core/ scripts/ tests/
```

### Packaging QGIS Plugin ZIP
To build an installable ZIP archive excluding heavy model weights and development caches:
```bash
python scripts/package_plugin.py --dist-dir dist
```
This produces `dist/PalmAI.zip` (root directory `PalmAI/`, <1 MB) and outputs its SHA-256 checksum.

For in-depth architectural details, see:
- [CI/CD Architecture (`docs/ci-cd.md`)](docs/ci-cd.md)
- [Release Policy & Compatibility Matrix (`docs/release-policy.md`)](docs/release-policy.md)
- [Test Matrix (`docs/test-matrix.md`)](docs/test-matrix.md)

---

## **License**
This project is licensed under the [MIT License](LICENSE).
