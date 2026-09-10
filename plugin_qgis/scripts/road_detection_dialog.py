# -*- coding: utf-8 -*-
"""
/***************************************************************************
 RoadDetectionDialog
                                 A QGIS plugin
 Tool for road segmentation using DeepLabV3+ and hybrid pathfinding.
 ***************************************************************************/
"""

import os
import traceback
import tempfile
import gc

from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QDialog, QTextBrowser, QWidget, QHBoxLayout, QLineEdit
from qgis.PyQt.QtCore import QEvent
from qgis.gui import QgsFileWidget
from qgis.core import (
    QgsApplication,
    QgsTask,
    QgsMessageLog,
    Qgis,
    QgsLineSymbol
)

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

# ============================================================================
# --- MODEL ARCHITECTURE ---R
# ============================================================================

def define_model_architecture():
    import torch.nn as nn
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return model

def load_model(model_path, device):
    import torch
    try:
        model = define_model_architecture().to(device)
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        model.eval()
        return model
    except Exception as e:
        QgsMessageLog.logMessage(f"Error loading model: {e}", "Road Detection", Qgis.Critical)
        return None

# ============================================================================
# --- WORKER TASK ---
# ============================================================================

class RoadInferenceTask(QgsTask):
    def __init__(self, description, parameters, iface):
        super().__init__(description, QgsTask.CanCancel)
        self.p = parameters
        self.iface = iface
        self.exception, self.output_layers = None, []
        
        self.temp_dir = self.p['temp_dir']
        self.memmap_sum_path = os.path.join(self.temp_dir, 'temp_pred_sum.mmap')
        self.memmap_count_path = os.path.join(self.temp_dir, 'temp_overlap_count.mmap')
        self.memmap_binary_path = os.path.join(self.temp_dir, 'temp_binary_mask.mmap') 
        self.memmap_skeleton_path = os.path.join(self.temp_dir, 'temp_skeleton.mmap')
        
    def _skeletonize_blocked(self, input_memmap, output_memmap, shape, overlap=100, block_size=4096):
        import numpy as np
        from skimage.morphology import skeletonize
        h, w = shape
        count, total = 0, len(range(0, h, block_size)) * len(range(0, w, block_size))
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if self.isCanceled(): return False
                count += 1
                self.setProgress(int(92 + 2 * (count / total)))
                
                block_in = input_memmap[max(0, y-overlap):min(h, y+block_size+overlap), max(0, x-overlap):min(w, x+block_size+overlap)]
                if not np.any(block_in): continue
                block_skel = skeletonize(block_in).astype(np.uint8)
                
                oy, ox = y - max(0, y-overlap), x - max(0, x-overlap)
                th, tw = min(y+block_size, h) - y, min(x+block_size, w) - x
                output_memmap[y:y+th, x:x+tw] = block_skel[oy:oy+th, ox:ox+tw]
        return True
        
    def _trace_simple_path(self, skeleton_simple, start_rc, visited):
        path_pixels = [start_rc]
        curr_rc = start_rc
        visited[curr_rc] = True
        while True:
            r, c = curr_rc
            next_pixel = None
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < skeleton_simple.shape[0] and 0 <= nc < skeleton_simple.shape[1]:
                        if skeleton_simple[nr, nc] == 1 and not visited[nr, nc]:
                            next_pixel = (nr, nc)
                            break 
                if next_pixel: break
            if next_pixel:
                curr_rc = next_pixel
                visited[curr_rc] = True
                path_pixels.append(curr_rc)
            else: return path_pixels

    def run(self):
        import geopandas, rasterio, torch, numpy as np
        try:
            model = load_model(self.p['model_path'], self.p['device'])
            if not model: return False
                
            aoi_polygon, aoi_gdf = None, None
            with rasterio.open(self.p['raster_path']) as src:
                self.p['transform'], self.p['meta'], source_crs = src.transform, src.meta.copy(), src.crs

            if self.p['extent_path']:
                try:
                    extent_gdf = geopandas.read_file(self.p['extent_path']).to_crs(source_crs.to_wkt())
                    aoi_polygon, aoi_gdf = extent_gdf.unary_union, extent_gdf
                except: self.p['extent_path'] = None

            self.prediction_prob_map = self._inference_tiling(model)
            if self.prediction_prob_map is None or self.isCanceled(): return False

            return self._post_process_and_vectorize(aoi_gdf, aoi_polygon)

        except Exception as e:
            self.exception = e
            return False
        finally:
            self.prediction_prob_map = self.memmap_sum = self.memmap_count = self.memmap_binary = self.memmap_skeleton = None
            gc.collect()
            for fpath in [self.memmap_sum_path, self.memmap_count_path, self.memmap_binary_path, self.memmap_skeleton_path]:
                if os.path.exists(fpath):
                    try: os.remove(fpath)
                    except: pass

    def _inference_tiling(self, model):
        import torch, numpy as np, rasterio
        from rasterio.windows import Window
        try:
            with rasterio.open(self.p['raster_path']) as src:
                w, h = src.width, src.height
                pred_mmap = np.memmap(self.memmap_sum_path, dtype='float16', mode='w+', shape=(h, w))
                count_mmap = np.memmap(self.memmap_count_path, dtype='uint8', mode='w+', shape=(h, w))
                
                step = self.p['tile_size'] - self.p['overlap']
                tiles = [Window(min(x, w-self.p['tile_size']) if x+self.p['tile_size']>w else x, min(y, h-self.p['tile_size']) if y+self.p['tile_size']>h else y, self.p['tile_size'], self.p['tile_size']) for y in range(0, h, step) for x in range(0, w, step)]
                
                total, batch = len(tiles), self.p['batch_size']
                for i in range(0, total, batch):
                    if self.isCanceled(): return None
                    windows = tiles[i:i+batch]
                    tensors = [torch.from_numpy(src.read(window=win)[:3, :, :].astype(np.float32) / 255.0) for win in windows]
                    
                    with torch.no_grad():
                        res = torch.sigmoid(model(torch.stack(tensors).to(self.p['device']))['out']).cpu().numpy().squeeze(1)
                    
                    for idx, win in enumerate(windows):
                        pred_mmap[win.row_off:win.row_off+win.height, win.col_off:win.col_off+win.width] += res[idx].astype(np.float16)
                        count_mmap[win.row_off:win.row_off+win.height, win.col_off:win.col_off+win.width] += 1
                        
                    if i % (max(1, total//50)*batch) == 0: self.setProgress(int(5 + 50 * (i/total)))

                chunk_h = int(np.ceil(h / 100))
                for i in range(100):
                    y1, y2 = i * chunk_h, min((i+1) * chunk_h, h)
                    if y1 >= y2: continue
                    c_chunk = count_mmap[y1:y2, :]
                    c_chunk[c_chunk==0] = 1
                    pred_mmap[y1:y2, :] /= c_chunk
                
                pred_mmap.flush()
                return pred_mmap
        except Exception as e:
            self.exception = e
            return None

    def _post_process_and_vectorize(self, aoi_gdf, aoi_polygon):
        import numpy as np, rasterio, geopandas as gpd
        from shapely.geometry import LineString
        from scipy.ndimage import convolve
        try: import cv2; HAS_CV2 = True
        except: HAS_CV2 = False

        h, w = self.prediction_prob_map.shape
        binary_mask = np.memmap(self.memmap_binary_path, dtype='uint8', mode='w+', shape=(h, w))

        chunk_h = int(np.ceil(h / 50))
        for i in range(50):
            if self.isCanceled(): return False
            y1, y2 = i * chunk_h, min((i+1) * chunk_h, h)
            if y1 < y2: binary_mask[y1:y2, :] = (self.prediction_prob_map[y1:y2, :] > self.p['probability_threshold']).astype(np.uint8)

        self.prediction_prob_map = None
        
        CHUNK_SIZE, OVERLAP = 2048, self.p['closing_iterations'] + 20
        kernel_open = np.ones((self.p['opening_kernel_size'], self.p['opening_kernel_size']), np.uint8) if HAS_CV2 else None
        kernel_close = np.ones((3,3), np.uint8) if HAS_CV2 else None
        min_size = self.p['min_object_size_pixels']

        for y in range(0, h, CHUNK_SIZE):
            for x in range(0, w, CHUNK_SIZE):
                if self.isCanceled(): return False
                chunk = binary_mask[max(0, y-OVERLAP):min(h, y+CHUNK_SIZE+OVERLAP), max(0, x-OVERLAP):min(w, x+CHUNK_SIZE+OVERLAP)].copy()
                if HAS_CV2:
                    chunk = cv2.morphologyEx(cv2.morphologyEx(chunk, cv2.MORPH_OPEN, kernel_open), cv2.MORPH_CLOSE, kernel_close, iterations=self.p['closing_iterations'])
                    if min_size > 0:
                        n, l, s, c = cv2.connectedComponentsWithStats(chunk, connectivity=8)
                        for lab in range(1, n):
                            if s[lab, cv2.CC_STAT_AREA] < min_size: chunk[l == lab] = 0
                
                y_tgt, x_tgt = y - max(0, y-OVERLAP), x - max(0, x-OVERLAP)
                y_w, x_w = min(y+CHUNK_SIZE, h)-y, min(x+CHUNK_SIZE, w)-x
                binary_mask[y:y+y_w, x:x+x_w] = chunk[y_tgt:y_tgt+y_w, x_tgt:x_tgt+x_w]

        skeleton_memmap = np.memmap(self.memmap_skeleton_path, dtype='uint8', mode='w+', shape=(h, w))
        if not self._skeletonize_blocked(binary_mask, skeleton_memmap, (h, w)): return False
        
        meta = self.p['meta']
        meta.update(dtype='uint8', count=1, compress='lzw', driver='GTiff', nodata=0)
        with rasterio.open(self.p['output_raster_path'], 'w', **meta) as dst:
            for y in range(0, h, 4096):
                dst.write(skeleton_memmap[y:y+min(4096, h-y), :], 1, window=rasterio.windows.Window(0, y, w, min(4096, h-y)))
        self.output_layers.append({'path': self.p['output_raster_path'], 'type': 'raster'})

        geometries = []
        for y in range(0, h, 8192):
            for x in range(0, w, 8192):
                sub_skel = skeleton_memmap[max(0, y-1):min(y+8192, h), max(0, x-1):min(x+8192, w)].copy()
                if not np.any(sub_skel): continue
                
                kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype='uint8')
                neighbors = convolve(sub_skel, kernel, mode='constant')
                sub_skel[neighbors > 2] = 0
                eps = np.argwhere((sub_skel==1) & (convolve(sub_skel, kernel, mode='constant')<=1))
                visited = np.zeros_like(sub_skel, dtype=bool)
                
                for ep in eps:
                    if not visited[tuple(ep)]:
                        path = self._trace_simple_path(sub_skel, tuple(ep), visited)
                        if len(path) > 5:
                            geometries.append(LineString([rasterio.transform.xy(self.p['transform'], r+max(0, y-1), c+max(0, x-1)) for r, c in path]))

        self.setProgress(99)
        if geometries:
            gdf = gpd.GeoDataFrame(geometry=geometries, crs=self.p['meta']['crs'])
            gdf = gdf[gdf.length > self.p['min_road_length_meters']]
            if aoi_gdf is not None: gdf = gdf[gdf.geometry.intersects(aoi_polygon)]
            gdf.to_file(self.p['output_vector_path'])
            self.output_layers.append({'path': self.p['output_vector_path'], 'type': 'vector'})
        return True

    def finished(self, result):
        if result:
            self.iface.messageBar().pushMessage("Sukses", "Road Detection Selesai!", level=Qgis.Success, duration=10)
            for l in self.output_layers:
                if l['type'] == 'raster': 
                    self.iface.addRasterLayer(l['path'], os.path.basename(l['path']))
                else: 
                    # Set Line (Jalan) menjadi warna merah tebal 1.5mm
                    layer = self.iface.addVectorLayer(l['path'], os.path.basename(l['path']), "ogr")
                    if layer:
                        symbol = QgsLineSymbol.createSimple({'color': 'red', 'line_width': '1.5'})
                        layer.renderer().setSymbol(symbol)
                        layer.triggerRepaint()
                        self.iface.layerTreeView().refreshLayerSymbology(layer.id())
        else:
            if self.exception:
                 self.iface.messageBar().pushMessage("Gagal", str(self.exception), level=Qgis.Critical)

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
        try: import torch, rasterio, geopandas, skimage, cv2
        except ImportError as e: return self.iface.messageBar().pushMessage("Error", f"Missing Lib: {e}", level=Qgis.Critical)

        params = {
            'model_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'deeplabv3+_roaddetection_1.pth.tar'),
            'raster_path': self.mQgsFileWidget_Raster.filePath(), 'extent_path': self.mQgsFileWidget_Extent.filePath(),
            'output_raster_path': self.mQgsFileWidget_Mask.filePath(), 'output_vector_path': self.mQgsFileWidget_Polyline.filePath(),
            'temp_dir': self.mQgsFileWidget_TempDir.filePath(),
            'device': 'cuda' if 'GPU' in self.mComboBox_Device.currentText() and torch.cuda.is_available() else 'cpu',
            'tile_size': 750, 'overlap': 150, 'batch_size': int(self.le_batch_size.text()),
            'probability_threshold': float(self.le_probability_threshold.text()), 'closing_iterations': 50, 
            'opening_kernel_size': int(self.le_opening_kernel_size.text()), 'min_object_size_pixels': int(self.le_min_object_size_pixels.text()),
            'min_road_length_meters': 15.0, 'pathfinding_max_gap_pixels': int(self.le_pathfinding_max_gap_pixels.text()),
            'pathfinding_max_avg_cost': float(self.le_pathfinding_max_avg_cost.text()), 'pathfinding_bbox_padding': int(self.le_pathfinding_bbox_padding.text())
        }
        
        if not params['raster_path'] or not params['temp_dir']: return self.iface.messageBar().pushMessage("Error", "Lengkapi Input!", level=Qgis.Critical)
        QgsApplication.taskManager().addTask(RoadInferenceTask("Road Detection", params, self.iface))
        self.close()
