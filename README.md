


# **PalmAI QGIS Plugin**
## A Deep Learning-powered QGIS plugin for the automated extraction of oil palm plantation features from drone imagery.

## **Description**
PalmAI was developed to automate the data processing workflow for oil palm plantations, which was previously carried out manually. By utilising the YOLOv8 and DeepLabV3+ architectures, this plugin drastically speeds up the digitisation process whilst maintaining spatial accuracy for operational purposes

## **Main Feature**
### 1. Tree Counting (YOLOv8)
Automatic detection and counting of oil palm trees with Shapefile output (Point & Bounding Box).

https://github.com/user-attachments/assets/98feab8e-b895-4ee3-b47f-22ab097fc2bc

### 2. Center Tree (YOLOv8)
High-precision extraction of the centre point of oil palm crowns for use as a guide in precision positioning (e.g. spraying for Oryctes pests).

https://github.com/user-attachments/assets/34584588-0f9d-404b-8fed-267057f0ef7f

### 3. Tree Classification (YOLOv8)
Classification of the main conditions affecting oil palm trees into three categories (Healthy, Yellowish, Dead) for the analysis of plantation health.

https://github.com/user-attachments/assets/1938980e-e6e4-4759-9d31-44d6a70f6ded

### 4. Road Detection (DeepLabV3+)
Segmentation and extraction of plantation road networks into Shapefile line vectors (polylines) using hybrid pathfinding technology.

https://github.com/user-attachments/assets/643d65d7-d6e7-4b14-a943-de487a838b0d

## **User Guide**
I have provided a step-by-step guide (from data preparation to the execution of each algorithm) [Read the PalmAI User Guide here](https://drive.google.com/file/d/1pgkmoNwnQgRRP3JbT9n_loTzG5tLFDIz/view?usp=sharing).

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
- **Deep Learning:** PyTorch, YOLOv8, DeepLabV3+, Slicing Iterative Logic
- **Geospatial Processing:** QGIS Python API (PyQGIS), GeoPandas, Rasterio, Shapely, Fiona
- **UI/UX:** PyQt5

## **Installation & Quickstart Guide**

Mulai versi ini, AI inference PalmAI diisolasi dari Python internal QGIS agar **bebas dari masalah bentrok dependensi (dependency hell)**. Plugin QGIS hanya bertindak sebagai antarmuka (thin UI), sementara proses inferensi dapat dijalankan lewat:
1. **Docker (Rekomendasi)**: Paling stabil, terisolasi 100%, otomatis.
2. **Virtualenv (Tanpa Docker)**: Cocok untuk macOS (support Apple Silicon MPS) atau pengguna tanpa Docker.
3. **QGIS Python (Legacy)**: Cara lama (OSGeo4W Shell) tetap didukung.

---

### **Langkah 1: Pasang Plugin ke QGIS**
1. Unduh atau clone repositori ini ke komputer Anda.
2. Salin folder `plugin_qgis`.
3. Tempel (Paste) folder tersebut ke direktori plugin QGIS Anda, lalu ubah namanya menjadi `PalmAI`:
   - **Windows (Rekomendasi - User Profile, tanpa perlu akses admin):**
     `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\PalmAI`
     *(Bisa buka Windows Run `Win+R` lalu paste path di atas)*
   - **Windows (Alternative - System-wide):**
     `C:\Program Files\QGIS <versi>\apps\qgis\python\plugins\PalmAI`
   - **macOS:**
     `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/PalmAI`
   - **Linux:**
     `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/PalmAI`
4. Buka aplikasi **QGIS**.
5. Masuk ke menu **Plugins > Manage and Install Plugins...**
6. Di tab **Installed**, centang **PalmAI** untuk mengaktifkan plugin. Menu **PalmAI** akan muncul di menu bar QGIS.

---

### **Langkah 2: Pilih Mode Inferensi (Backend)**

Buka menu **Plugins > PalmAI > Settings (Inference Backend)**.

Pilih opsi yang sesuai dengan kebutuhan Anda:

| Opsi Backend | Kapan Digunakan? | Persiapan yang Dibutuhkan |
|---|---|---|
| **Docker** *(Direkomendasikan)* | Bebas pusing instalasi Python/CUDA. Cocok untuk semua device. | Cukup install & jalankan [Docker Desktop](https://www.docker.com/products/docker-desktop/). Image AI akan diunduh/dibangun otomatis pada pemakaian pertama. |
| **Virtualenv** | Ingin GPU di Mac (Apple Silicon MPS) atau komputer tidak memiliki Docker. | Pastikan Python 3.9–3.12 terpasang di sistem, lalu cukup klik tombol **"Set up Python environment"** di menu Settings PalmAI. |
| **QGIS Python (Legacy)** | Sudah berhasil menginstall torch dkk di Python bawaan QGIS. | Ikuti panduan legacy instalasi via OSGeo4W Shell di bawah. |
| **Automatic** *(Default)* | Mencoba Docker terlebih dahulu, lalu Virtualenv, lalu QGIS Python. | - |

---

### **Langkah 3: Menjalankan & Mengetes Plugin (Testing)**

Untuk menguji apakah plugin sudah berfungsi dengan benar:
1. Buka salah satu fitur di menu **Plugins > PalmAI**, misalnya:
   - **Palm Oil Tree Counting**
2. Pada dialog yang muncul:
   - **Input Raster**: Pilih file raster contoh yang ada di repositori: `demo_data/orthophoto_sample.tif` (atau ortofoto Anda sendiri).
   - **Output Bounding Box & Centroid**: Tentukan lokasi dan nama file shapefile (`.shp`) keluaran.
   - **Device**: Pilih `GPU (CUDA)` jika ada NVIDIA GPU, atau `CPU`.
3. Klik tombol **Run**.
4. Proses akan berjalan di latar belakang (dapat dipantau di task manager QGIS).
5. Setelah selesai, layer hasil deteksi berupa poligon bounding box dan titik pusat pohon akan otomatis dimuat dan distilasi langsung di kanvas peta QGIS!

---

### **Opsional: Build Docker Image Sendiri**
Jika Anda ingin membangun image Docker secara lokal (dari folder `plugin_qgis`):
```bash
# Untuk CPU:
docker build -f docker/Dockerfile.cpu -t ghcr.io/anovvy/palmai-core:cpu .

# Untuk NVIDIA GPU (CUDA):
docker build -f docker/Dockerfile.gpu -t ghcr.io/anovvy/palmai-core:gpu .
```

---

## **Panduan Instalasi Legacy (Python Internal QGIS / OSGeo4W Shell)**
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
- **Windows:** `pip install -r requirements.txt`
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
