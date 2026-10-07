# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Palm Oil Center Tree Dialog
                                 A QGIS plugin
                             -------------------
        begin                : 2026
        copyright            : (C) 2026 by Bayu Nabiil
        email                : bayunabiil1365@gmail.com
 ***************************************************************************/
"""
import os

from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QDialog
from qgis.core import QgsApplication, Qgis
from qgis.gui import QgsFileWidget

from ..backends.task import PalmAITask
from ..backends.layer_loader import load_detection_layers, report_failure

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), '..', 'ui', 'centertree_dialog_base.ui'))

PLUGIN_NAME = "Palm Oil Center Tree"
WEIGHT_FILENAME = 'yolov8n_centertree_1.3.pt'

# NOTE: The inference logic now lives in palmai_core/yolo.py.


class CenterTreeDialog(QDialog, FORM_CLASS):
    def __init__(self, iface, parent=None):
        super(CenterTreeDialog, self).__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.task = None
        self.mQgsFileWidget_Raster.setFilter("Raster Files (*.tif *.tiff)")
        self.mQgsFileWidget_Output.setFilter("ESRI Shapefile (*.shp)")
        self.mQgsFileWidget_Centroid.setFilter("ESRI Shapefile (*.shp)")
        self.mComboBox_Device.addItems(["GPU (CUDA)", "CPU"])
        self.btn_run.clicked.connect(self.start_inference_task)

        self.le_tile_size.setText("1280")
        self.le_overlap.setText("0.3")
        self.le_confidence.setText("0.3")

    def start_inference_task(self):
        params = {
            'raster_path': self.mQgsFileWidget_Raster.filePath(),
            'bbox_output_path': self.mQgsFileWidget_Output.filePath(),
            'centroid_output_path': self.mQgsFileWidget_Centroid.filePath(),
            'slice_size': int(self.le_tile_size.text()),
            'overlap_ratio': float(self.le_overlap.text()),
            'confidence': float(self.le_confidence.text()),
            'extent_path': self.mQgsFileWidget_Extent.filePath(),
            'device': 'gpu' if "GPU" in self.mComboBox_Device.currentText() else 'cpu',
        }

        if not all([params['raster_path'], params['bbox_output_path'], params['centroid_output_path']]):
            return

        self.task = PalmAITask(
            f"Center Tree {os.path.basename(params['raster_path'])}...", 'centertree', params, PLUGIN_NAME,
            self._on_done, models=[('weight_path', WEIGHT_FILENAME)])
        QgsApplication.taskManager().addTask(self.task)
        self.close()

    def _on_done(self, result, error, canceled):
        if result is None:
            report_failure(self.iface, error, canceled)
            return
        total = result['total_detections']
        if total == 0:
            self.iface.messageBar().pushMessage("Info", "No object.", level=Qgis.Warning)
            return
        self.iface.messageBar().pushMessage("Success", f"Found {total} object.", level=Qgis.Success, duration=10)
        load_detection_layers(self.iface, result['layers'], PLUGIN_NAME)