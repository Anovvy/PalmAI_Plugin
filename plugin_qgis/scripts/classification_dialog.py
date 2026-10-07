"""
/***************************************************************************
 Palm Oil Tree Classification Dialog
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
from ..backends.layer_loader import load_detection_layers, report_failure, CLASS_COLORS

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), '..', 'ui', 'classification_dialog_base.ui'))

PLUGIN_NAME = "Palm Oil Classification"
WEIGHT_FILENAME = 'yolov8n_classification_2.1.pt'

# NOTE: The inference logic now lives in palmai_core/yolo.py.


class ClassificationDialog(QDialog, FORM_CLASS):
    def __init__(self, iface, parent=None):
        super(ClassificationDialog, self).__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.task = None
        self.mQgsFileWidget_Raster.setFilter("Raster Files (*.tif *.tiff)")
        self.mQgsFileWidget_Output.setFilter("ESRI Shapefile (*.shp)")
        self.mQgsFileWidget_Centroid.setFilter("ESRI Shapefile (*.shp)")
        self.mComboBox_Device.addItems(["GPU (CUDA)", "CPU"])
        self.btn_run.clicked.connect(self.start_inference_task)

        self.le_tile_size.setText("640")
        self.le_overlap.setText("0.2")
        self.le_conf_healthy.setText("0.3")
        self.le_conf_yellowish.setText("0.3")
        self.le_conf_dead.setText("0.3")

    def start_inference_task(self):
        params = {
            'raster_path': self.mQgsFileWidget_Raster.filePath(),
            'bbox_output_path': self.mQgsFileWidget_Output.filePath(),
            'centroid_output_path': self.mQgsFileWidget_Centroid.filePath(),
            'slice_size': int(self.le_tile_size.text()),
            'overlap_ratio': float(self.le_overlap.text()),
            'class_confidences': {
                'Healthy': float(self.le_conf_healthy.text()),
                'Yellowish': float(self.le_conf_yellowish.text()),
                'Dead': float(self.le_conf_dead.text()),
            },
            'extent_path': self.mQgsFileWidget_Extent.filePath(),
            'device': 'gpu' if "GPU" in self.mComboBox_Device.currentText() else 'cpu',
        }

        if not all([params['raster_path'], params['bbox_output_path'], params['centroid_output_path']]):
            return

        self.task = PalmAITask(
            f"Klasifikasi {os.path.basename(params['raster_path'])}...", 'classification', params, PLUGIN_NAME,
            self._on_done, models=[('weight_path', WEIGHT_FILENAME)])
        QgsApplication.taskManager().addTask(self.task)
        self.close()

    def _on_done(self, result, error, canceled):
        if result is None:
            if not report_failure(self.iface, error, canceled):
                self.iface.messageBar().pushMessage("Info", "Cancelled / Failed", level=Qgis.Warning)
            return
        self.iface.messageBar().pushMessage(
            "Success", f"Classification complete! Found {result['total_detections']} object.",
            level=Qgis.Success, duration=10)
        load_detection_layers(self.iface, result['layers'], PLUGIN_NAME, class_colors=CLASS_COLORS)