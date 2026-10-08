# **PalmAI QGIS Plugin**
## A Deep Learning-powered QGIS plugin for automated feature extraction from oil palm plantation drone imagery.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Inference-Docker%20Container-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)

---

## **Description**
PalmAI automates the spatial analysis and data processing workflow for oil palm plantations, replacing manual digitization with state-of-the-art Deep Learning models. By combining **YOLOv8** object detection and **DeepLabV3+** semantic segmentation with specialized geospatial post-processing, PalmAI delivers rapid, reliable, and operationally accurate vector outputs directly into QGIS.

To eliminate dependency conflicts ("dependency hell") across different operating systems and QGIS environments, PalmAI uses a **decoupled architecture**:
- **QGIS Plugin (Thin Client):** Manages user inputs, parameters, and renders vector/raster layers seamlessly in QGIS.
- **Inference Engine (Isolated Docker Container):** Runs PyTorch, CUDA, SAHI, and geospatial processing libraries in a dedicated, version-pinned environment.

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

## **Installation & Quickstart Guide**

### **Prerequisites**
1. **[QGIS](https://qgis.org/)** (v3.22 or newer recommended).
2. **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (Recommended inference backend):
   - Needs to be installed and **running** (daemon active in the system tray) when executing PalmAI tasks.
   - *Windows Users:* Requires WSL 2 (Windows Subsystem for Linux).
   - *macOS Users:* Docker runs in CPU mode. (Or use the Virtualenv backend for Apple Silicon MPS).
   - *(Alternative)*: If you do not want to install Docker, you can use the built-in **Virtualenv backend** instead (see [Step 2](#step-2-configure-inference-backend)).

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
1. Clone or download this repository to your computer.
2. Copy the `plugin_qgis` folder into your QGIS plugins directory and rename it to `PalmAI`:
   - **Windows:**
     ```text
     C:\Program Files\QGIS 3.38.3\apps\qgis\python\plugins\PalmAI
     ```
     *(Requires administrator privileges. If your QGIS version differs from 3.38.3, adjust the folder path accordingly).*
   - **macOS:**
     ```text
     ~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/PalmAI
     ```
   - **Linux:**
     ```text
     ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/PalmAI
     ```
3. Open **QGIS**.
4. Navigate to **Plugins > Manage and Install Plugins...**
5. In the **Installed** tab, check the box next to **PalmAI** to enable it. The **PalmAI** menu will appear in the top toolbar.

---

### **Step 2: Configure Inference Backend**
Open **Plugins > PalmAI > Settings (Inference Backend)**.

| Backend Option | Recommended Use | Details |
|---|---|---|
| **Docker** *(Recommended)* | Default choice for all platforms | Fully isolated. Automatically pulls the pre-built container image (`ghcr.io/anovvy/palmai-core`) on first run. Zero Python package installation required on the host. |
| **Virtualenv** | macOS (Apple Silicon MPS) or systems without Docker | Creates an isolated virtual environment outside QGIS. Click **"Set up Python environment"** in the Settings dialog to initialize. |
| **Automatic** | Fallback mode | Checks Docker first, then Virtualenv, then QGIS Python. |

Click **Save** once configured.

---

### **Step 3: Testing & Running Tools**
To test the plugin using the bundled sample data:
1. Open any PalmAI tool from the menu, e.g., **Plugins > PalmAI > Palm Oil Tree Counting**.
2. Set the tool inputs:
   - **Input Raster:** Select the sample orthophoto at `demo_data/orthophoto_sample.tif` (or your own GeoTIFF).
   - **Output Bounding Box & Centroid:** Choose the target `.shp` save locations.
   - **Device:** Select `GPU (CUDA)` if an NVIDIA GPU is available, or `CPU`.
3. Click **Run**.
4. The task will execute in the background via Docker (ensure Docker Desktop is running).
5. Upon completion, the resulting Bounding Box polygons and Centroid points are automatically added and styled on the QGIS map canvas!

---

## **AI Models & Weights**
Trained weights are managed automatically:
- If model files are not present locally in `plugin_qgis/models/`, the plugin automatically downloads them on first use from the official GitHub Release:
  [PalmAI Releases (tag: models-v1)](https://github.com/Anovvy/PalmAI_Plugin/releases/tag/models-v1).
- Bundled models:
  - `yolov8n_counting_1.0.pt` (Tree Counting)
  - `yolov8n_centertree_1.3.pt` (Center Tree)
  - `yolov8n_classification_2.1.pt` (Condition Classification)
  - `deeplabv3+_roaddetection_1.pth.tar` (Road Network Segmentation)

---

## **Building Docker Images Locally (Optional)**
If you wish to build the container images locally instead of pulling from GHCR:
```bash
cd plugin_qgis

# CPU image
docker build -f docker/Dockerfile.cpu -t ghcr.io/anovvy/palmai-core:cpu .

# NVIDIA GPU (CUDA) image
docker build -f docker/Dockerfile.gpu -t ghcr.io/anovvy/palmai-core:gpu .
```

---

## **Repository Structure**
```text
PalmAI_Plugin/
├── plugin_qgis/              # QGIS Plugin package (Thin UI Client)
│   ├── metadata.txt          # QGIS plugin metadata
│   ├── palmai_plugin.py      # Plugin entry point & menu setup
│   ├── backends/             # Execution backends (Docker, Virtualenv, Process Runner)
│   ├── palmai_core/          # Pure AI inference pipelines (YOLOv8, DeepLabV3+, Worker)
│   ├── docker/               # Dockerfiles (CPU/GPU) & pinned requirements.lock
│   ├── models/               # Model weights directory (auto-downloaded if missing)
│   ├── scripts/              # UI dialog handlers (Counting, Center Tree, Classification, Road)
│   ├── ui/                   # Qt Designer UI forms
│   └── icons/                # Interface assets
├── demo_data/                # Sample orthophoto raster for testing
├── notebooks/                # Research notebooks & model training history
├── docs/                     # Documentation assets and screenshots
├── .github/workflows/        # Automated CI image build workflows
├── .gitignore
├── LICENSE
└── README.md
```

---

## **Tech Stack**
- **Deep Learning:** PyTorch, YOLOv8, DeepLabV3+, SAHI (Slicing Aided Hyper Inference)
- **Containerization:** Docker, GitHub Container Registry (GHCR)
- **Geospatial Processing:** PyQGIS, GeoPandas, Rasterio, Shapely, Fiona
- **GUI:** PyQt5 / Qt Designer

---

## **User Guide**
For an in-depth operational guide covering drone data flight planning, radiometric calibration, and parameter optimization, refer to the [PalmAI User Guide](https://drive.google.com/file/d/1pgkmoNwnQgRRP3JbT9n_loTzG5tLFDIz/view?usp=sharing).

---

## **License**
This project is licensed under the [MIT License](LICENSE).
