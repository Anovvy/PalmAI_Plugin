# -*- coding: utf-8 -*-
"""
YOLOv8 + SAHI-style tiled inference for: counting, centertree, classification.

Logic is ported 1:1 from the former *_dialog.py InferenceTask classes; only the
QgsTask hooks were replaced by a JobContext.
"""

import os
from pathlib import Path

from .context import JobContext
from .device import resolve_device

PLUGIN_NAMES = {
    "counting": "Palm Oil Counter",
    "centertree": "Palm Oil Center Tree",
    "classification": "Palm Oil Classification",
}


def merge_object_prediction_pair(pred1, pred2):
    from sahi.prediction import ObjectPrediction
    merged_bbox = [
        min(pred1.bbox.minx, pred2.bbox.minx),
        min(pred1.bbox.miny, pred2.bbox.miny),
        max(pred1.bbox.maxx, pred2.bbox.maxx),
        max(pred1.bbox.maxy, pred2.bbox.maxy),
    ]
    base_pred = pred1 if pred1.score.value > pred2.score.value else pred2
    return ObjectPrediction(
        bbox=merged_bbox,
        category_id=base_pred.category.id,
        category_name=base_pred.category.name,
        score=max(pred1.score.value, pred2.score.value),
    )


def manual_greedy_nmm(ctx, object_predictions, match_metric="IOS", match_threshold=0.5):
    import torch
    if len(object_predictions) == 0:
        return []

    tensor_data = [
        [p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy, p.score.value, p.category.id]
        for p in object_predictions
    ]
    predictions_tensor = torch.tensor(tensor_data, dtype=torch.float32)

    keep_to_merge_list = {}
    x1, y1, x2, y2 = (predictions_tensor[:, 0], predictions_tensor[:, 1],
                      predictions_tensor[:, 2], predictions_tensor[:, 3])
    scores = predictions_tensor[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()
    total_to_process = len(order)

    while len(order) > 0:
        processed_count = total_to_process - len(order)
        ctx.progress(int(82 + (processed_count / total_to_process) * 12))
        ctx.describe(f"Clearing duplicate detections ({processed_count}/{total_to_process})...")
        ctx.check_cancel()

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


def run_yolo_job(job, p, ctx: JobContext):
    """
    p keys: raster_path, weight_path, bbox_output_path, centroid_output_path,
            slice_size, overlap_ratio, extent_path, device ('gpu'|'cpu'),
            confidence (counting/centertree) or class_confidences (classification)
    Returns: {'total_detections': int, 'layers': [ {path, name, type, class_name?} ]}
    """
    import numpy as np
    import rasterio
    import geopandas
    import fiona
    from PIL import Image
    from rasterio.windows import Window
    from shapely.geometry import Polygon, mapping
    from sahi import AutoDetectionModel
    from sahi.prediction import ObjectPrediction

    classification = job == "classification"
    plugin_name = PLUGIN_NAMES[job]
    device = resolve_device(p.get("device", "cpu"))
    if p.get("device", "cpu") != "cpu" and device == "cpu":
        ctx.log("GPU requested but not available. Running on CPU.", "warning")

    ctx.progress(5)
    ctx.describe("Loading the detection model...")
    model_conf = 0.01 if classification else p["confidence"]
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=p["weight_path"],
        confidence_threshold=model_conf,
        device=device,
    )
    ctx.log(f"The model has been successfully loaded onto the device '{device}'.", "success")

    Path(p["bbox_output_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(p["centroid_output_path"]).parent.mkdir(parents=True, exist_ok=True)

    aoi_polygon = None
    with rasterio.open(p["raster_path"]) as src:
        source_crs = src.crs

    if p.get("extent_path"):
        extent_gdf = geopandas.read_file(p["extent_path"]).to_crs(source_crs.to_wkt())
        aoi_polygon = extent_gdf.unary_union

    ctx.log("Starting the slicing and inference process on a per-tile basis...")
    all_predictions = []

    with rasterio.open(p["raster_path"]) as src:
        transform = src.transform
        img_h, img_w = src.height, src.width

        slice_h = slice_w = p["slice_size"]
        overlap_h = int(p["overlap_ratio"] * slice_h)
        overlap_w = int(p["overlap_ratio"] * slice_w)
        step_h, step_w = slice_h - overlap_h, slice_w - overlap_w

        y_steps = range(0, img_h, step_h)
        x_steps = range(0, img_w, step_w)
        total_tiles = len(y_steps) * len(x_steps)
        tile_count = 0

        for y in y_steps:
            for x in x_steps:
                ctx.check_cancel()
                tile_count += 1
                ctx.progress(int(5 + 77 * (tile_count / total_tiles)))
                ctx.describe(f"Inference tile {tile_count}/{total_tiles}...")

                window = Window(x, y, min(slice_w, img_w - x), min(slice_h, img_h - y))
                tile_data = src.read(window=window)
                if tile_data.shape[0] > 3:
                    tile_data = tile_data[:3, :, :]
                tile_np = np.transpose(tile_data, (1, 2, 0))

                tile_for_model = np.array(Image.fromarray(tile_np).convert("RGB"))
                if not tile_for_model.any():
                    continue

                detection_model.perform_inference(tile_for_model)
                raw_predictions_tensor = detection_model._original_predictions[0]

                for pred_tensor in raw_predictions_tensor:
                    score = pred_tensor[4].item()
                    category_id = int(pred_tensor[5].item())
                    category_name = detection_model.model.names[category_id]

                    if classification:
                        if score < p["class_confidences"].get(category_name, 0.01):
                            continue
                    elif score < p["confidence"]:
                        continue

                    bbox = pred_tensor[0:4].tolist()
                    shifted_bbox = [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y]
                    all_predictions.append(
                        ObjectPrediction(
                            bbox=shifted_bbox, category_id=category_id,
                            score=score, category_name=category_name,
                        )
                    )

    ctx.log(f"Total detections collected BEFORE post-processing: {len(all_predictions)}", "warning")
    ctx.log("Running a manual post-process: GreedyNMM with the IOS metric...")
    final_predictions = manual_greedy_nmm(ctx, all_predictions, match_metric="IOS", match_threshold=0.5)

    ctx.progress(95)
    ctx.describe("Saving the final result to a shapefile...")
    crs_dict = source_crs.to_dict()
    detections_in_aoi = 0
    layers = []

    if not classification:
        schema_poly = {"geometry": "Polygon", "properties": {"score": "float:9.5", "category": "str:80"}}
        schema_point = {"geometry": "Point", "properties": {"score": "float:9.5", "category": "str:80"}}
        with fiona.open(p["bbox_output_path"], "w", crs=crs_dict, driver="ESRI Shapefile", schema=schema_poly) as poly_c, \
             fiona.open(p["centroid_output_path"], "w", crs=crs_dict, driver="ESRI Shapefile", schema=schema_point) as point_c:
            for pred in final_predictions:
                top_left = transform * (pred.bbox.minx, pred.bbox.miny)
                bottom_right = transform * (pred.bbox.maxx, pred.bbox.maxy)
                poly = Polygon.from_bounds(top_left[0], bottom_right[1], bottom_right[0], top_left[1])
                point = poly.centroid
                if aoi_polygon and not point.within(aoi_polygon):
                    continue
                detections_in_aoi += 1
                props = {"score": pred.score.value, "category": pred.category.name}
                poly_c.write({"geometry": mapping(poly), "properties": props})
                point_c.write({"geometry": mapping(point), "properties": props})
        layers = [
            {"path": p["bbox_output_path"], "name": Path(p["bbox_output_path"]).stem, "type": "bbox"},
            {"path": p["centroid_output_path"], "name": Path(p["centroid_output_path"]).stem, "type": "centroid"},
        ]
    else:
        writers_poly, writers_point = {}, {}
        try:
            for pred in final_predictions:
                top_left = transform * (pred.bbox.minx, pred.bbox.miny)
                bottom_right = transform * (pred.bbox.maxx, pred.bbox.maxy)
                poly = Polygon.from_bounds(top_left[0], bottom_right[1], bottom_right[0], top_left[1])
                point = poly.centroid
                if aoi_polygon and not point.within(aoi_polygon):
                    continue
                detections_in_aoi += 1
                class_name = pred.category.name

                if class_name not in writers_poly:
                    bbox_dir, bbox_filename = os.path.split(p["bbox_output_path"])
                    bbox_name, bbox_ext = os.path.splitext(bbox_filename)
                    class_bbox_path = os.path.join(bbox_dir, f"{bbox_name}_{class_name}{bbox_ext}")

                    c_dir, c_filename = os.path.split(p["centroid_output_path"])
                    c_name, c_ext = os.path.splitext(c_filename)
                    class_centroid_path = os.path.join(c_dir, f"{c_name}_{class_name}{c_ext}")

                    writers_poly[class_name] = fiona.open(
                        class_bbox_path, "w", crs=crs_dict, driver="ESRI Shapefile",
                        schema={"geometry": "Polygon", "properties": {"score": "float:9.5"}})
                    writers_point[class_name] = fiona.open(
                        class_centroid_path, "w", crs=crs_dict, driver="ESRI Shapefile",
                        schema={"geometry": "Point", "properties": {"score": "float:9.5"}})

                    layers.append({"path": class_bbox_path, "name": f"{bbox_name}_{class_name}",
                                   "type": "bbox", "class_name": class_name})
                    layers.append({"path": class_centroid_path, "name": f"{c_name}_{class_name}",
                                   "type": "centroid", "class_name": class_name})

                writers_poly[class_name].write({"geometry": mapping(poly), "properties": {"score": pred.score.value}})
                writers_point[class_name].write({"geometry": mapping(point), "properties": {"score": pred.score.value}})
        finally:
            for w in writers_poly.values():
                w.close()
            for w in writers_point.values():
                w.close()

    ctx.progress(100)
    return {"total_detections": detections_in_aoi, "layers": layers}
