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
import torch
import rasterio
import fiona
import traceback
import numpy as np
import geopandas
from pathlib import Path
from shapely.geometry import Polygon, mapping
from rasterio.windows import Window
from PIL import Image

from sahi import AutoDetectionModel
from sahi.prediction import ObjectPrediction

from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QDialog
from qgis.core import (
    QgsApplication, QgsTask, QgsMessageLog, Qgis, 
    QgsVectorLayer, QgsFillSymbol, QgsMarkerSymbol
)
from qgis.gui import QgsFileWidget

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), '..', 'ui', 'centertree_dialog_base.ui'))

def merge_object_prediction_pair(pred1: ObjectPrediction, pred2: ObjectPrediction):
    merged_bbox = [
        min(pred1.bbox.minx, pred2.bbox.minx),
        min(pred1.bbox.miny, pred2.bbox.miny),
        max(pred1.bbox.maxx, pred2.bbox.maxx),
        max(pred1.bbox.maxy, pred2.bbox.maxy),
    ]
    if pred1.score.value > pred2.score.value:
        base_pred = pred1
    else:
        base_pred = pred2
    return ObjectPrediction(
        bbox=merged_bbox, category_id=base_pred.category.id,
        category_name=base_pred.category.name, score=max(pred1.score.value, pred2.score.value)
    )

def manual_greedy_nmm(task, object_predictions, match_metric="IOS", match_threshold=0.5):
    num_predictions = len(object_predictions)
    if num_predictions == 0: return []
        
    tensor_data = []
    for pred in object_predictions:
        tensor_data.append([
            pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy,
            pred.score.value, pred.category.id
        ])
    predictions_tensor = torch.tensor(tensor_data, dtype=torch.float32)

    keep_to_merge_list = {}
    x1, y1, x2, y2 = predictions_tensor[:, 0], predictions_tensor[:, 1], predictions_tensor[:, 2], predictions_tensor[:, 3]
    scores = predictions_tensor[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()
    
    total_to_process = len(order)

    while len(order) > 0:
        processed_count = total_to_process - len(order)
        progress = int(82 + (processed_count / total_to_process) * 12)
        task.setProgress(progress)
        task.setDescription(f"Clearing duplicate detections ({processed_count}/{total_to_process})...")

        if task.isCanceled(): return None

        idx = order[-1]
        order = order[:-1]

        if len(order) == 0:
            keep_to_merge_list[idx.tolist()] = []
            break

        xx1 = torch.index_select(x1, dim=0, index=order).clamp(min=x1[idx].item())
        yy1 = torch.index_select(y1, dim=0, index=order).clamp(min=y1[idx].item())
        xx2 = torch.index_select(x2, dim=0, index=order).clamp(max=x2[idx].item())
        yy2 = torch.index_select(y2, dim=0, index=order).clamp(max=y2[idx].item())

        w = (xx2 - xx1).clamp(min=0.0)
        h = (yy2 - yy1).clamp(min=0.0)
        inter = w * h
        rem_areas = torch.index_select(areas, dim=0, index=order)
        
        if match_metric == "IOS":
            smaller = torch.min(rem_areas, areas[idx])
            match_metric_value = inter / (smaller + 1e-7)
        else:
            union = (rem_areas - inter) + areas[idx]
            match_metric_value = inter / (union + 1e-7)

        mask = match_metric_value < match_threshold
        matched_box_indices = order[~mask]
        unmatched_indices = order[mask]
        order = unmatched_indices[scores[unmatched_indices].argsort()]

        keep_to_merge_list[idx.tolist()] = matched_box_indices.tolist()

    final_predictions = []
    for keep_ind, merge_ind_list in keep_to_merge_list.items():
        merged_pred = object_predictions[keep_ind]
        for merge_ind in merge_ind_list:
            merged_pred = merge_object_prediction_pair(merged_pred, object_predictions[merge_ind])
        final_predictions.append(merged_pred)

    return final_predictions

class InferenceTask(QgsTask):
    def __init__(self, description, parameters, iface):
        super().__init__(description, QgsTask.CanCancel)
        self.parameters = parameters
        self.iface = iface
        self.total_detections = 0
        self.exception = None
        self.error_traceback = ""

    def run(self):
        try:
            p = self.parameters
            plugin_name = "Palm Oil Center Tree"

            self.setProgress(5)
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8', model_path=p['weight_path'],
                confidence_threshold=p['confidence'], device=p['device'],
            )

            Path(p['bbox_output_path']).parent.mkdir(parents=True, exist_ok=True)
            Path(p['centroid_output_path']).parent.mkdir(parents=True, exist_ok=True)
            
            aoi_polygon = None
            source_crs = None
            with rasterio.open(p['raster_path']) as src:
                source_crs = src.crs

            if p['extent_path']:
                extent_gdf = geopandas.read_file(p['extent_path'])
                extent_gdf = extent_gdf.to_crs(source_crs.to_wkt())
                aoi_polygon = extent_gdf.unary_union
            
            all_predictions = []
            
            with rasterio.open(p['raster_path']) as src:
                transform = src.transform
                img_h, img_w = src.height, src.width
                slice_h, slice_w = p['slice_size'], p['slice_size']
                overlap_h, overlap_w = int(p['overlap_ratio'] * slice_h), int(p['overlap_ratio'] * slice_w)
                step_h, step_w = slice_h - overlap_h, slice_w - overlap_w
                y_steps, x_steps = range(0, img_h, step_h), range(0, img_w, step_w)
                total_tiles = len(y_steps) * len(x_steps)
                tile_count = 0

                for y in y_steps:
                    for x in x_steps:
                        if self.isCanceled(): return False
                        tile_count += 1
                        self.setProgress(int(5 + 77 * (tile_count / total_tiles)))
                        self.setDescription(f"Tile inference {tile_count}/{total_tiles}...")

                        window = Window(x, y, min(slice_w, img_w - x), min(slice_h, img_h - y))
                        tile_data = src.read(window=window)
                        
                        if tile_data.shape[0] > 3: tile_data = tile_data[:3, :, :]
                        tile_np = np.transpose(tile_data, (1, 2, 0))
                        
                        pil_image = Image.fromarray(tile_np).convert("RGB")
                        tile_for_model = np.array(pil_image)

                        if not tile_for_model.any(): continue

                        detection_model.perform_inference(tile_for_model)
                        raw_predictions_tensor = detection_model._original_predictions[0]
                        
                        for pred_tensor in raw_predictions_tensor:
                            score = pred_tensor[4].item()
                            if score < p['confidence']: continue
                            
                            bbox = pred_tensor[0:4].tolist()
                            shifted_bbox = [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y]
                            category_id = int(pred_tensor[5].item())
                            
                            all_predictions.append(
                                ObjectPrediction(
                                    bbox=shifted_bbox, category_id=category_id,
                                    score=score, category_name=detection_model.model.names[category_id]
                                )
                            )

            final_predictions = manual_greedy_nmm(self, all_predictions, match_metric="IOS", match_threshold=0.5)
            if final_predictions is None: return False

            self.setProgress(95)
            self.setDescription("Saving the final result...")
            
            schema_poly = {'geometry': 'Polygon', 'properties': {'score': 'float:9.5', 'category': 'str:80'}}
            schema_point = {'geometry': 'Point', 'properties': {'score': 'float:9.5', 'category': 'str:80'}}
            
            detections_in_aoi = 0
            with fiona.open(p['bbox_output_path'], 'w', crs=source_crs.to_dict(), driver='ESRI Shapefile', schema=schema_poly) as poly_collection, \
                 fiona.open(p['centroid_output_path'], 'w', crs=source_crs.to_dict(), driver='ESRI Shapefile', schema=schema_point) as point_collection:
                
                for pred in final_predictions:
                    top_left = transform * (pred.bbox.minx, pred.bbox.miny)
                    bottom_right = transform * (pred.bbox.maxx, pred.bbox.maxy)
                    poly = Polygon.from_bounds(top_left[0], bottom_right[1], bottom_right[0], top_left[1])
                    point = poly.centroid

                    if aoi_polygon and not point.within(aoi_polygon): continue
                    
                    detections_in_aoi += 1
                    poly_collection.write({'geometry': mapping(poly), 'properties': {'score': pred.score.value, 'category': pred.category.name}})
                    point_collection.write({'geometry': mapping(point), 'properties': {'score': pred.score.value, 'category': pred.category.name}})
            
            self.total_detections = detections_in_aoi
            self.setProgress(100)
            return True

        except Exception as e:
            self.exception = e
            self.error_traceback = traceback.format_exc()
            QgsMessageLog.logMessage(f"FATAL ERROR: {e}\nTraceback:\n{self.error_traceback}", "Palm Oil Center Tree", Qgis.Critical)
            return False

    def finished(self, result):
        if not result:
            if self.isCanceled():
                self.iface.messageBar().pushMessage("Info", "Cancelled by the user.", level=Qgis.Info)
            elif self.exception:
                self.iface.messageBar().pushMessage("Error", f"Failed: {self.exception}", level=Qgis.Critical)
            return

        if self.total_detections == 0:
            self.iface.messageBar().pushMessage("Info", "No object.", level=Qgis.Warning)
            return
            
        self.iface.messageBar().pushMessage("Success", f"Found {self.total_detections} object.", level=Qgis.Success, duration=10)
        
        try:
            bbox_path = self.parameters['bbox_output_path']
            centroid_path = self.parameters['centroid_output_path']
            
            bbox_layer = self.iface.addVectorLayer(bbox_path, Path(bbox_path).stem, "ogr")
            if bbox_layer:
                symbol = QgsFillSymbol.createSimple({'color': 'transparent', 'outline_color': 'red', 'outline_width': '1'})
                bbox_layer.renderer().setSymbol(symbol)
                bbox_layer.triggerRepaint()
                self.iface.layerTreeView().refreshLayerSymbology(bbox_layer.id())

            centroid_layer = self.iface.addVectorLayer(centroid_path, Path(centroid_path).stem, "ogr")
            if centroid_layer:
                symbol_pt = QgsMarkerSymbol.createSimple({'color': 'red', 'size': '2'})
                centroid_layer.renderer().setSymbol(symbol_pt)
                centroid_layer.triggerRepaint()
                self.iface.layerTreeView().refreshLayerSymbology(centroid_layer.id())
                
        except Exception as e:
            QgsMessageLog.logMessage(f"Failed to load layer: {e}", "Palm Oil Center Tree", Qgis.Warning)

class CenterTreeDialog(QDialog, FORM_CLASS):
    def __init__(self, iface, parent=None):
        super(CenterTreeDialog, self).__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.mQgsFileWidget_Raster.setFilter("Raster Files (*.tif *.tiff)")
        self.mQgsFileWidget_Output.setFilter("ESRI Shapefile (*.shp)")
        self.mQgsFileWidget_Centroid.setFilter("ESRI Shapefile (*.shp)")
        self.mComboBox_Device.addItems(["GPU (CUDA)", "CPU"])
        self.btn_run.clicked.connect(self.start_inference_task)
        
        self.le_tile_size.setText("1280")
        self.le_overlap.setText("0.3")
        self.le_confidence.setText("0.3")

    def start_inference_task(self):
        weight_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n_centertree_1.3.pt')

        params = {
            'raster_path': self.mQgsFileWidget_Raster.filePath(),
            'weight_path': weight_path,
            'bbox_output_path': self.mQgsFileWidget_Output.filePath(),
            'centroid_output_path': self.mQgsFileWidget_Centroid.filePath(),
            'slice_size': int(self.le_tile_size.text()),
            'overlap_ratio': float(self.le_overlap.text()),
            'confidence': float(self.le_confidence.text()),
            'extent_path': self.mQgsFileWidget_Extent.filePath(),
        }

        if not all([params['raster_path'], params['bbox_output_path'], params['centroid_output_path']]):
            return
            
        params['device'] = "cuda:0" if "GPU" in self.mComboBox_Device.currentText() and torch.cuda.is_available() else "cpu"

        self.task = InferenceTask(f"Center Tree {os.path.basename(params['raster_path'])}...", params, self.iface)
        QgsApplication.taskManager().addTask(self.task)
        self.close()