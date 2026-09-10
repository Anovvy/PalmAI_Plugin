# **PalmAI QGIS Plugin**
## A Deep Learning-powered QGIS plugin for the automated extraction of oil palm plantation features from drone imagery.

## **Description**
PalmAI was developed to automate the data processing workflow for oil palm plantations, which was previously carried out manually. By utilising the YOLOv8 and DeepLabV3+ architectures, this plugin drastically speeds up the digitisation process whilst maintaining spatial accuracy for operational purposes

## **Main Feature**
### 1. Tree Counting (YOLOv8)
Automatic detection and counting of oil palm trees with Shapefile output (Point & Bounding Box).

![Demo](docs\Tree_Counting_GIF.gif?raw=true)

### 2. Center Tree (YOLOv8)
High-precision extraction of the centre point of oil palm crowns for use as a guide in precision positioning (e.g. spraying for Oryctes pests).

![Demo](docs\Center_Tree_GIF.gif?raw=true)

### 3. Tree Classification (YOLOv8)
Classification of the main conditions affecting oil palm trees into three categories (Healthy, Yellowish, Dead) for the analysis of plantation health.

![Demo](docs\Tree_Classification_GIF.gif?raw=true)

### 4. Road Detection (DeepLabV3+)
Segmentation and extraction of plantation road networks into Shapefile line vectors (polylines) using hybrid pathfinding technology.

![Demo](docs\Road_Detection_GIF.gif?raw=true)

## **User Guide**
I have provided a step-by-step guide (from data preparation to the execution of each algorithm) [Read the PalmAI User Guide here (Coming Soon)](https://duckduckgo.com).

## **Repository Structure**
```text
PalmAI QGIS Plugin/
├── plugin_qgis/              # Main Plugins folder
│   ├── metadata.txt
│   ├── requirements.txt
│   ├── palmai_plugin.py
│   ├── icons/                # Visual assets
│   ├── models/               # Trained weights (YOLO & DeepLab)
│   ├── scripts/              # Inference algorithms & worker tasks
│   └── ui/                   # QtDesigner interface design
├── demo_data/                # Sample raster data for testing the plugin
├── notebooks/                # Documentation of AI research and training processes
├── docs/                     # Screenshots of results and guidance
├── .gitignore
├── LICENSE
└── README.md
```
## **Tech Stack**
- **Deep Learning:** PyTorch, YOLOv8, DeepLabV3+, Logika Slicing Iteratif
- **Geospatial Processing:** QGIS Python API (PyQGIS), GeoPandas, Rasterio, Shapely, Fiona
- **UI/UX:** PyQt5

## **Installation Guide (GPU/MPS Configuration)**
As this plugin runs a deep learning model, it is strongly recommended that you use a device with an NVIDIA GPU (CUDA-enabled). Installation must be carried out via the OSGeo4W Shell built into QGIS.

### Step 1: Install the plugin in QGIS
1. Clone or download the ZIP file for this repository to your computer.
2. Open the repository folder, then copy the `plugin_qgis` folder.
3. Paste that folder into your QGIS plugins directory, and rename it to `PalmAI`.
    - Windows:
    `C:\Program Files\QGIS 3.38.3\apps\qgis\python\plugins\PalmAI`
    (Requires administrator access. If your version of QGIS is different, adjust the number 3.38.3 accordingly)
    - Mac:
    `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/PalmAI`
### Step 2: Open Terminal and navigate to the folder
**For Windows users (OSGeo4W Shell):**
1. Click the Windows Start Menu, search for and open **OSGeo4W Shell** (Run as Administrator).
2. Change the terminal directory using the `cd` command (**Replace ‘Username’)**:
```text
cd "C:\Program Files\QGIS 3.38.3\apps\qgis\python\plugins\PalmAI"
```
**For MacOS users (Terminal):**
1. Open the Mac’s built-in **Terminal** app.
2. Change the terminal location using the command:
```text
cd ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins/PalmAI
```
### Step 3: Installing Basic Dependencies
Once the terminal is open in the `PalmAI` folder, run the following command:
- **Windows:** `pip install -r requrements.txt`
- **MacOS:** `pip3 install -r requirements.txt`*(Note: If an error occurs, use `/Applications/QGIS.app/Contents/MacOS/bin/python3 -m pip install -r requirements.txt`)*
### Step 4: PyTorch Instalation
**For Windows users (NVIDIA CUDA):**
1. Visit [PyTorch Get Started Page](https://pytorch.org/get-started/locally/).
2. Select the appropriate OS, package (pip), Python and CUDA version (check using `nvidia-smi`).
3. Copy the command; here is an example for CUDA 11.8:
```text
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
**For Windows users without a GPU/Integrated CPU only:**
Simply run this command in the **OSGeo4W Shell**
```text
pip install torch torchvision torchaudio
``` 
**For macOS (Apple Silicon / Intel) users:**
macOS uses Metal (MPS) acceleration, not CUDA. Simply run the standard PyTorch installation:
```text
pip3 install torch torchvision torchaudio
```
### Step 5: Activate Plugin
1. Open **QGIS** application
2. Go to the **Plugins > Manage and Install Plugins** menu.
3. Find **PalmAI** in the *Installed* tab, then tick the box to enable it.

## **LICENSE**
This project uses [MIT License](LICENSE)