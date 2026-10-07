# -*- coding: utf-8 -*-
"""DeepLabV3+ road segmentation + skeleton/vectorisation (ported from RoadInferenceTask)."""

import os
import gc

from .context import JobContext
from .device import resolve_device


def define_model_architecture():
    import torch.nn as nn
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return model


def load_model(model_path, device):
    import torch
    model = define_model_architecture().to(device)
    # Trusted bundled checkpoint; weights_only=False keeps behaviour on torch>=2.6
    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()
    return model


class RoadPipeline:
    def __init__(self, params, ctx: JobContext):
        self.p = dict(params)
        self.ctx = ctx
        self.output_layers = []
        self.temp_dir = self.p["temp_dir"]
        os.makedirs(self.temp_dir, exist_ok=True)
        self.memmap_sum_path = os.path.join(self.temp_dir, "temp_pred_sum.mmap")
        self.memmap_count_path = os.path.join(self.temp_dir, "temp_overlap_count.mmap")
        self.memmap_binary_path = os.path.join(self.temp_dir, "temp_binary_mask.mmap")
        self.memmap_skeleton_path = os.path.join(self.temp_dir, "temp_skeleton.mmap")
        self.prediction_prob_map = None

    def _skeletonize_blocked(self, input_memmap, output_memmap, shape, overlap=100, block_size=4096):
        import numpy as np
        from skimage.morphology import skeletonize
        h, w = shape
        count, total = 0, len(range(0, h, block_size)) * len(range(0, w, block_size))
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                self.ctx.check_cancel()
                count += 1
                self.ctx.progress(int(92 + 2 * (count / total)))

                block_in = input_memmap[max(0, y - overlap):min(h, y + block_size + overlap),
                                        max(0, x - overlap):min(w, x + block_size + overlap)]
                if not np.any(block_in):
                    continue
                block_skel = skeletonize(block_in).astype(np.uint8)

                oy, ox = y - max(0, y - overlap), x - max(0, x - overlap)
                th, tw = min(y + block_size, h) - y, min(x + block_size, w) - x
                output_memmap[y:y + th, x:x + tw] = block_skel[oy:oy + th, ox:ox + tw]

    def _trace_simple_path(self, skeleton_simple, start_rc, visited):
        path_pixels = [start_rc]
        curr_rc = start_rc
        visited[curr_rc] = True
        while True:
            r, c = curr_rc
            next_pixel = None
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < skeleton_simple.shape[0] and 0 <= nc < skeleton_simple.shape[1]:
                        if skeleton_simple[nr, nc] == 1 and not visited[nr, nc]:
                            next_pixel = (nr, nc)
                            break
                if next_pixel:
                    break
            if next_pixel:
                curr_rc = next_pixel
                visited[curr_rc] = True
                path_pixels.append(curr_rc)
            else:
                return path_pixels

    def run(self):
        import geopandas
        import rasterio
        try:
            device = resolve_device(self.p.get("device", "cpu"))
            self.p["device"] = device
            self.ctx.describe("Loading the road model...")
            model = load_model(self.p["model_path"], device)

            aoi_polygon, aoi_gdf = None, None
            with rasterio.open(self.p["raster_path"]) as src:
                self.p["transform"], self.p["meta"], source_crs = src.transform, src.meta.copy(), src.crs

            if self.p.get("extent_path"):
                try:
                    extent_gdf = geopandas.read_file(self.p["extent_path"]).to_crs(source_crs.to_wkt())
                    aoi_polygon, aoi_gdf = extent_gdf.unary_union, extent_gdf
                except Exception:
                    self.p["extent_path"] = None

            self.prediction_prob_map = self._inference_tiling(model)
            self.ctx.check_cancel()
            self._post_process_and_vectorize(aoi_gdf, aoi_polygon)
            return {"layers": self.output_layers}
        finally:
            self.prediction_prob_map = None
            gc.collect()
            for fpath in [self.memmap_sum_path, self.memmap_count_path,
                          self.memmap_binary_path, self.memmap_skeleton_path]:
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass

    def _inference_tiling(self, model):
        import torch
        import numpy as np
        import rasterio
        from rasterio.windows import Window
        p = self.p
        with rasterio.open(p["raster_path"]) as src:
            w, h = src.width, src.height
            pred_mmap = np.memmap(self.memmap_sum_path, dtype="float16", mode="w+", shape=(h, w))
            count_mmap = np.memmap(self.memmap_count_path, dtype="uint8", mode="w+", shape=(h, w))

            step = p["tile_size"] - p["overlap"]
            tiles = [Window(min(x, w - p["tile_size"]) if x + p["tile_size"] > w else x,
                            min(y, h - p["tile_size"]) if y + p["tile_size"] > h else y,
                            p["tile_size"], p["tile_size"])
                     for y in range(0, h, step) for x in range(0, w, step)]

            total, batch = len(tiles), p["batch_size"]
            for i in range(0, total, batch):
                self.ctx.check_cancel()
                windows = tiles[i:i + batch]
                tensors = [torch.from_numpy(src.read(window=win)[:3, :, :].astype(np.float32) / 255.0)
                           for win in windows]

                with torch.no_grad():
                    res = torch.sigmoid(model(torch.stack(tensors).to(p["device"]))["out"]).cpu().numpy().squeeze(1)

                for idx, win in enumerate(windows):
                    pred_mmap[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width] += res[idx].astype(np.float16)
                    count_mmap[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width] += 1

                if i % (max(1, total // 50) * batch) == 0:
                    self.ctx.progress(int(5 + 50 * (i / total)))
                    self.ctx.describe(f"Road inference {min(i + batch, total)}/{total} tiles...")

            chunk_h = int(np.ceil(h / 100))
            for i in range(100):
                y1, y2 = i * chunk_h, min((i + 1) * chunk_h, h)
                if y1 >= y2:
                    continue
                c_chunk = count_mmap[y1:y2, :]
                c_chunk[c_chunk == 0] = 1
                pred_mmap[y1:y2, :] /= c_chunk

            pred_mmap.flush()
            return pred_mmap

    def _post_process_and_vectorize(self, aoi_gdf, aoi_polygon):
        import numpy as np
        import rasterio
        import geopandas as gpd
        from shapely.geometry import LineString
        from scipy.ndimage import convolve
        try:
            import cv2
            HAS_CV2 = True
        except Exception:
            HAS_CV2 = False

        p = self.p
        ctx = self.ctx
        h, w = self.prediction_prob_map.shape
        binary_mask = np.memmap(self.memmap_binary_path, dtype="uint8", mode="w+", shape=(h, w))

        chunk_h = int(np.ceil(h / 50))
        for i in range(50):
            ctx.check_cancel()
            y1, y2 = i * chunk_h, min((i + 1) * chunk_h, h)
            if y1 < y2:
                binary_mask[y1:y2, :] = (self.prediction_prob_map[y1:y2, :] > p["probability_threshold"]).astype(np.uint8)

        self.prediction_prob_map = None

        CHUNK_SIZE, OVERLAP = 2048, p["closing_iterations"] + 20
        kernel_open = np.ones((p["opening_kernel_size"], p["opening_kernel_size"]), np.uint8) if HAS_CV2 else None
        kernel_close = np.ones((3, 3), np.uint8) if HAS_CV2 else None
        min_size = p["min_object_size_pixels"]

        for y in range(0, h, CHUNK_SIZE):
            for x in range(0, w, CHUNK_SIZE):
                ctx.check_cancel()
                chunk = binary_mask[max(0, y - OVERLAP):min(h, y + CHUNK_SIZE + OVERLAP),
                                    max(0, x - OVERLAP):min(w, x + CHUNK_SIZE + OVERLAP)].copy()
                if HAS_CV2:
                    chunk = cv2.morphologyEx(cv2.morphologyEx(chunk, cv2.MORPH_OPEN, kernel_open),
                                             cv2.MORPH_CLOSE, kernel_close, iterations=p["closing_iterations"])
                    if min_size > 0:
                        n, l, s, c = cv2.connectedComponentsWithStats(chunk, connectivity=8)
                        for lab in range(1, n):
                            if s[lab, cv2.CC_STAT_AREA] < min_size:
                                chunk[l == lab] = 0

                y_tgt, x_tgt = y - max(0, y - OVERLAP), x - max(0, x - OVERLAP)
                y_w, x_w = min(y + CHUNK_SIZE, h) - y, min(x + CHUNK_SIZE, w) - x
                binary_mask[y:y + y_w, x:x + x_w] = chunk[y_tgt:y_tgt + y_w, x_tgt:x_tgt + x_w]

        skeleton_memmap = np.memmap(self.memmap_skeleton_path, dtype="uint8", mode="w+", shape=(h, w))
        self._skeletonize_blocked(binary_mask, skeleton_memmap, (h, w))

        meta = p["meta"]
        meta.update(dtype="uint8", count=1, compress="lzw", driver="GTiff", nodata=0)
        with rasterio.open(p["output_raster_path"], "w", **meta) as dst:
            for y in range(0, h, 4096):
                dst.write(skeleton_memmap[y:y + min(4096, h - y), :], 1,
                          window=rasterio.windows.Window(0, y, w, min(4096, h - y)))
        self.output_layers.append({"path": p["output_raster_path"], "type": "raster"})

        geometries = []
        for y in range(0, h, 8192):
            for x in range(0, w, 8192):
                sub_skel = skeleton_memmap[max(0, y - 1):min(y + 8192, h), max(0, x - 1):min(x + 8192, w)].copy()
                if not np.any(sub_skel):
                    continue

                kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype="uint8")
                neighbors = convolve(sub_skel, kernel, mode="constant")
                sub_skel[neighbors > 2] = 0
                eps = np.argwhere((sub_skel == 1) & (convolve(sub_skel, kernel, mode="constant") <= 1))
                visited = np.zeros_like(sub_skel, dtype=bool)

                for ep in eps:
                    if not visited[tuple(ep)]:
                        path = self._trace_simple_path(sub_skel, tuple(ep), visited)
                        if len(path) > 5:
                            geometries.append(LineString(
                                [rasterio.transform.xy(p["transform"], r + max(0, y - 1), c + max(0, x - 1))
                                 for r, c in path]))

        ctx.progress(99)
        if geometries:
            gdf = gpd.GeoDataFrame(geometry=geometries, crs=p["meta"]["crs"])
            gdf = gdf[gdf.length > p["min_road_length_meters"]]
            if aoi_gdf is not None:
                gdf = gdf[gdf.geometry.intersects(aoi_polygon)]
            gdf.to_file(p["output_vector_path"])
            self.output_layers.append({"path": p["output_vector_path"], "type": "vector"})


def run_road_job(p, ctx: JobContext):
    return RoadPipeline(p, ctx).run()
