# -*- coding: utf-8 -*-
"""
/***************************************************************************
 RoadDetectionDialog
                                 A QGIS plugin
 Tool for road segmentation using DeepLabV3+ and hybrid pathfinding.
 ***************************************************************************/
"""

import os

from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QDialog, QTextBrowser, QWidget, QHBoxLayout, QLineEdit
from qgis.PyQt.QtCore import QEvent
from qgis.gui import QgsFileWidget
from qgis.core import (
    QgsApplication,
    Qgis,
)

from ..backends.task import PalmAITask
from ..backends.layer_loader import load_road_layers, report_failure

PLUGIN_NAME = "Road Detection"
MODEL_FILENAME = 'deeplabv3+_roaddetection_1.pth.tar'

FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), '..', 'ui', 'road_detection_dialog_base.ui'))

# ============================================================================
# --- HELP TEXT DATA ---
# ============================================================================
HELP_TEXTS = {
    "default": """
    <h3>Road Detection Tool (DeepLabV3+)</h3>
    <p>This tool utilizes deep learning AI to automatically extract road networks from orthophoto imagery.</p>
    <p><b>How to use:</b></p>
    <ul>
    <li>Click on any parameter field to see a detailed explanation here.</li>
    <li>Ensure input raster has sufficient resolution (GSD 5-50 cm).</li>
    <li>GPU usage is highly recommended for best performance.</li>
    </ul>
    """,
    
    "mQgsFileWidget_Raster": """
    <h3>Input Raster</h3>
    <p>Select the orthophoto image (GeoTIFF) to be processed.</p>
    <p><b>Tips:</b></p>
    <ul>
    <li><b>.tif</b> format is recommended.</li>
    <li>Ensure the image has a valid Coordinate Reference System (CRS), e.g., UTM.</li>
    <li>Ideal pixel resolution is between 10cm and 50cm.</li>
    </ul>
    """,
    
    "mQgsFileWidget_Extent": """
    <h3>Area of Interest (AOI)</h3>
    <p><i>(Optional)</i> Vector file (Shapefile/GeoJSON) defining the processing boundary.</p>
    <p><b>Function:</b></p>
    <p>If provided, the tool will only process areas within this polygon. Useful for speeding up processing if you only need a specific part of a large orthophoto.</p>
    """,
    
    "mQgsFileWidget_Mask": """
    <h3>Output Raster Mask</h3>
    <p>Output path for the binary detection result.</p>
    <p><b>Format:</b> GeoTIFF (.tif)</p>
    <p><b>Pixel Values:</b></p>
    <ul>
    <li><b>1 (White):</b> Detected Road.</li>
    <li><b>0 (Black):</b> Background.</li>
    </ul>
    """,
    
    "mQgsFileWidget_Polyline": """
    <h3>Output Polyline</h3>
    <p>Output path for the final road centerline vectors.</p>
    <p><b>Format:</b> ESRI Shapefile (.shp)</p>
    <p>This result is derived from <i>Skeletonization</i> and <i>Tracing</i> processes performed on the raster mask.</p>
    """,
    
    "mQgsFileWidget_TempDir": """
    <h3>Temporary / Cache Folder</h3>
    <p>Directory to store large temporary files during processing.</p>
    <p><b>Disk Space Estimation:</b></p>
    <p>Use this formula to prepare sufficient free space:</p>
    <p style="background-color:#e1e1e1; padding:5px;"><b>Min. Space = Ortho File Size + (25% x Ortho File Size)</b></p>
    <p><i>Example: If your ortho is 100 GB, ensure the drive has at least 125 GB of free space.</i></p>
    """,
    
    "mComboBox_Device": """
    <h3>Inference Device</h3>
    <p>Hardware used to run the AI model.</p>
    <ul>
    <li><b>GPU (CUDA):</b> Highly Recommended. Uses NVIDIA graphics card. Processing is significantly faster (10x - 50x).</li>
    <li><b>CPU:</b> Slow. Use only if NVIDIA GPU is unavailable. May cause system lag during processing.</li>
    </ul>
    """,
    
    "le_probability_threshold": """
    <h3>Probability Threshold</h3>
    <p>Confidence threshold (0.0 - 1.0) for considering a pixel as 'road'.</p>
    <ul>
    <li><b>High Value (> 0.5):</b> Cleaner results, but faint/shadowed roads might be missed.</li>
    <li><b>Low Value (< 0.3):</b> Detects faint roads better, but increases noise (false positives).</li>
    </ul>
    <p><b>Recommendation:</b> 0.3 - 0.5</p>
    """,
    
    "le_batch_size": """
    <h3>Batch Size</h3>
    <p>Number of image tiles processed simultaneously by the GPU. Affects speed and VRAM usage.</p>
    <p><b>Setting Guide (Based on GPU VRAM):</b></p>
    <ul>
    <li><b>High-End GPU (24GB+ VRAM):</b><br><i>(e.g., RTX 3090/4090, RTX 5000/6000 Ada)</i><br>Set to <b>16 - 32</b>.</li>
    <li><b>Mid-Range GPU (12GB - 16GB VRAM):</b><br><i>(e.g., RTX 3060/4070/4080, RTX A4000)</i><br>Set to <b>4 - 8</b>.</li>
    <li><b>Entry-Level GPU (8GB VRAM):</b><br><i>(e.g., RTX 3050/4060)</i><br>Set to <b>1 - 2</b>.</li>
    <li><b>CPU:</b><br>Must set to <b>1</b>.</li>
    </ul>
    <p><i>If "Out of Memory (OOM)" error occurs, reduce this value.</i></p>
    """,
    
    "le_opening_kernel_size": """
    <h3>Opening Kernel Size</h3>
    <p>Filter size to remove small noise (speckles) from the initial detection.</p>
    <p><b>Effect:</b></p>
    <ul>
    <li>Larger values (e.g., 5 or 7) produce cleaner results but may remove narrow paths.</li>
    <li>Odd numbers (3, 5, 7) are recommended.</li>
    </ul>
    """,
    
    "le_min_object_size_pixels": """
    <h3>Minimum Object Size</h3>
    <p>Advanced filter to remove disconnected or too short road segments.</p>
    <p><b>Unit:</b> Pixels.</p>
    <p>The system will remove any road segment with a total area less than this value. Useful for removing false detections on rooftops or rocks.</p>
    """,
    
    "le_pathfinding_max_gap_pixels": """
    <h3>Maximum Gap (Pathfinding)</h3>
    <p>Maximum distance (in pixels) the algorithm will attempt to reconnect broken road ends.</p>
    <p><b>Usage:</b></p>
    <p>If roads are frequently broken by tree shadows, increase this value. </p>
    <p><i>Caution: Excessive values may cause incorrect connections (jumping to non-road objects).</i></p>
    """,
    
    "le_pathfinding_max_avg_cost": """
    <h3>Maximum Average Cost</h3>
    <p>Tolerance for obstacles when reconnecting roads.</p>
    <p>Higher values allow the algorithm to traverse non-road areas (like dense forest) to connect two points. Lower values force connections only where there is a trace of a road.</p>
    <p><b>Recommendation:</b> Keep default (0.6919).</p>
    """,
    
    "le_pathfinding_bbox_padding": """
    <h3>Bounding Box Padding</h3>
    <p>Additional buffer area around road endpoints analyzed for connections.</p>
    <p>Only change if using extremely large <i>Maximum Gap</i> values.</p>
    """
}

# NOTE: The model architecture and the inference/post-processing pipeline now live in
# palmai_core/road.py so they can run in QGIS, a virtualenv or a Docker container.

# ============================================================================
# --- MAIN DIALOG ---
# ============================================================================

class RoadDetectionDialog(QDialog, FORM_CLASS):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.setMinimumWidth(self.width() + 220) 
        self.resize(self.width() + 220, self.height())
        self.help_browser = QTextBrowser()
        self.help_browser.setHtml(HELP_TEXTS["default"])
        self.help_browser.setStyleSheet("background-color: #fcfcfc; padding: 5px; border: 1px solid #ccc;")
        self.help_browser.setFixedWidth(200)
        self.gridLayout_3.addWidget(self.help_browser, 0, 3, 8, 1)
        
        for w in [self.mQgsFileWidget_Raster, self.mQgsFileWidget_Extent, self.mQgsFileWidget_Mask, self.mQgsFileWidget_Polyline, self.mQgsFileWidget_TempDir, self.mComboBox_Device, self.le_probability_threshold, self.le_batch_size, self.le_opening_kernel_size, self.le_min_object_size_pixels, self.le_pathfinding_max_gap_pixels, self.le_pathfinding_max_avg_cost, self.le_pathfinding_bbox_padding]:
            w.installEventFilter(self)
            if hasattr(w, 'lineEdit') and w.lineEdit(): w.lineEdit().installEventFilter(self)

        self.mQgsFileWidget_Raster.setFilter("Raster Files (*.tif *.tiff)")
        self.mQgsFileWidget_Extent.setFilter("Vector Files (*.shp)")
        self.mQgsFileWidget_Mask.setStorageMode(QgsFileWidget.SaveFile); self.mQgsFileWidget_Mask.setFilter("GTiff Files (*.tif)")
        self.mQgsFileWidget_Polyline.setStorageMode(QgsFileWidget.SaveFile); self.mQgsFileWidget_Polyline.setFilter("ESRI Shapefile (*.shp)")
        self.mQgsFileWidget_TempDir.setStorageMode(QgsFileWidget.GetDirectory)
        self.btn_run.clicked.connect(self.start_task)
        self.mComboBox_Device.addItems(["GPU (CUDA)", "CPU"])
        
        for le, val in zip([self.le_batch_size, self.le_probability_threshold, self.le_min_object_size_pixels, self.le_opening_kernel_size, self.le_pathfinding_max_gap_pixels, self.le_pathfinding_max_avg_cost, self.le_pathfinding_bbox_padding], ["4", "0.4", "50", "3", "550", "0.6919", "35"]): le.setText(val)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.FocusIn:
            name = obj.parent().objectName() if obj.parent() in [self.mQgsFileWidget_Raster, self.mQgsFileWidget_Mask, self.mQgsFileWidget_Polyline, self.mQgsFileWidget_Extent, self.mQgsFileWidget_TempDir] else obj.objectName()
            self.help_browser.setHtml(HELP_TEXTS.get(name, HELP_TEXTS["default"]))
        return super().eventFilter(obj, event)

    def start_task(self):
        params = {
            'model_path': os.path.join(os.path.dirname(__file__), '..', 'models', MODEL_FILENAME),
            'raster_path': self.mQgsFileWidget_Raster.filePath(), 'extent_path': self.mQgsFileWidget_Extent.filePath(),
            'output_raster_path': self.mQgsFileWidget_Mask.filePath(), 'output_vector_path': self.mQgsFileWidget_Polyline.filePath(),
            'temp_dir': self.mQgsFileWidget_TempDir.filePath(),
            'device': 'gpu' if 'GPU' in self.mComboBox_Device.currentText() else 'cpu',
            'tile_size': 750, 'overlap': 150, 'batch_size': int(self.le_batch_size.text()),
            'probability_threshold': float(self.le_probability_threshold.text()), 'closing_iterations': 50,
            'opening_kernel_size': int(self.le_opening_kernel_size.text()), 'min_object_size_pixels': int(self.le_min_object_size_pixels.text()),
            'min_road_length_meters': 15.0, 'pathfinding_max_gap_pixels': int(self.le_pathfinding_max_gap_pixels.text()),
            'pathfinding_max_avg_cost': float(self.le_pathfinding_max_avg_cost.text()), 'pathfinding_bbox_padding': int(self.le_pathfinding_bbox_padding.text())
        }

        if not params['raster_path'] or not params['temp_dir']:
            return self.iface.messageBar().pushMessage(
                "Error", "Please fill in all required inputs (Raster and Temp Directory)!", level=Qgis.Critical)
        self.task = PalmAITask("Road Detection", 'road', params, PLUGIN_NAME, self._on_done,
                               models=[('model_path', MODEL_FILENAME)])
        QgsApplication.taskManager().addTask(self.task)
        self.close()

    def _on_done(self, result, error, canceled):
        if result is None:
            report_failure(self.iface, error, canceled)
            return
        self.iface.messageBar().pushMessage(
            "Success", "Road detection completed successfully!", level=Qgis.Success, duration=10)
        load_road_layers(self.iface, result.get('layers', []))
