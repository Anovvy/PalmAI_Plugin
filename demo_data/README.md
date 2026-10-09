# **PalmAI Demo Data**

This directory is designed to provide sample geospatial raster datasets (drone orthophotos) so you can immediately test and evaluate PalmAI tools in QGIS without needing to prepare your own imagery.

---

## **Download Demo Data**

Due to the large file sizes of geospatial GeoTIFF rasters, these datasets are hosted externally on Google Drive. Please download the sample files via the links below:

| File Name | Description & Resolution | Size | Target PalmAI Tools | Download Link |
|---|---|---|---|:---:|
| `Demo_Data_1.tif` | Drone orthophoto imagery with **GSD ~7 cm**. Ideal for plantation-scale palm canopy detection, road extraction, and condition classification. | ~1.75 GB | • **Palm Oil Tree Counting**<br>• **Road Network Detection**<br>• **Condition Classification** | [📥 Download (Google Drive)](https://drive.google.com/file/d/1rh4UeR2fodad73ZMaVFY2zJhwtVNownZ/view?usp=sharing) |
| `Demo_Data_2.tif` | Ultra high-resolution drone orthophoto imagery with **GSD ~2 cm**. Provides fine canopy texture needed for precise tree apical center point localization. | ~598 MB | • **Palm Oil Centre Tree** | [📥 Download (Google Drive)](https://drive.google.com/file/d/1o7vhpVf4UCsogXZj2L1HBT_CiwqyAuqU/view?usp=sharing) |

---

## **How to Use Demo Data in QGIS**

1. **Download:** Click the download link for the dataset matching your test scenario.
2. **Save Location:** Save the downloaded `.tif` file into this `demo_data/` directory (or any local folder on your computer).
3. **Load in QGIS:**
   - Drag and drop the `.tif` file directly onto the QGIS map canvas, or
   - Go to **Layer > Add Layer > Add Raster Layer...** and select the `.tif` file.
4. **Execute PalmAI Tools:**
   - Open the desired tool from **Plugins > PalmAI** (e.g., *Palm Oil Tree Counting*).
   - In the **Input Raster** selector, choose `Demo_Data_1.tif` (or `Demo_Data_2.tif` for Centre Tree).
   - Specify output shapefile (`.shp`) paths and click **Run**.