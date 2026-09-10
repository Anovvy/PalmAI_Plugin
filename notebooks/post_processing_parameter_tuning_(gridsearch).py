# -*- coding: utf-8 -*-
"""4. Post-Processing Parameter Tuning (GridSearch).ipynb
"""

# =============================================================================
# STEP 1: INSTALLING THE LIBRARY AND IMPORTING
# =============================================================================
print("Installing the library...")
# !pip install rasterio geopandas scikit-image tqdm skan pandas networkx scipy

import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from skimage.morphology import skeletonize, remove_small_objects, binary_opening
from skimage.graph import route_through_array
from scipy.ndimage import binary_closing
import skan
from tqdm import tqdm
import os
import math
import itertools
from scipy.spatial import KDTree
from google.colab import drive

# =============================================================================
# STEP 1B: MOUNT GOOGLE DRIVE
# =============================================================================
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")

# =============================================================================
# STEP 2: CONFIGURING THE MAIN PATH
# =============================================================================
print("Path configuration for tuning...")

# --- Path Input ---
# 1. Path to the Probability Map (output from segformer_detection_hybrid.py)
PROB_MAP_PATH = '/content/drive/MyDrive/Road Detection_Data_/Hasil Prediksi/SegFormer/probmap_1.2.npy'

# 2. Path to the raster ground truth (manually masked, 0 for background, 1 or 255 for roads)
GROUND_TRUTH_RASTER_PATH = '/content/drive/MyDrive/Road Detection_Data_/Data Test/Test_Masking.tif'

# 3. Path to the Ground Truth Vector (manual SHP)
GROUND_TRUTH_SHP_PATH = '/content/drive/MyDrive/Road Detection_Data_/Data Test/Test_skeleton.shp'

# 4. Path to the original TIF file (ONLY for retrieving CRS and transformation metadata)
IMAGE_PATH_FOR_META = '/content/drive/MyDrive/Road Detection_Data_/Data Test/Test.tif'

# The best path for saving SHP files
BEST_SHP_PATH = '/content/drive/MyDrive/Road Detection_Data_/Hasil Prediksi/SegFormer/Tuning_1.2.shp'
BEST_RASTER_PATH = '/content/drive/MyDrive/Road Detection_Data_/Hasil Prediksi/SegFormer/Tuning_1.2.tif'

# --- Tuning Configuration ---
VECTOR_BUFFER_METERS = 3.0

# =============================================================================
# STEP 3: SEARCH SPACE PARAMETERS
# =============================================================================
print("Defining the parameter search space...")

# --- Group 1: Cleaner (To be fine-tuned using Raster IoU) ---
SEARCH_PROBABILITY_THRESHOLD = [0.4, 0.5, 0.6]
SEARCH_OPENING_KERNEL_SIZE = [3, 5, 7]
SEARCH_MIN_OBJECT_SIZE_PIXELS = [250, 450, 750]

# --- Group 2: Gap Filler (To be fine-tuned using Raster IoU) ---
SEARCH_CLOSING_ITERATIONS = [50, 100, 150]
SEARCH_PATHFINDING_MAX_GAP_PIXELS = [200, 350, 500]
SEARCH_PATHFINDING_MAX_AVG_COST = [0.6, 0.8, 1.0]

# --- Group 3: Vector Filter (To be tuned using Vector IoU) ---
SEARCH_MIN_ROAD_LENGTH_METERS = [2.0, 5.0, 10.0, 20.0]

# --- Fixed Parameters ---
FIXED_PATHFINDING_COST_FACTOR = 15.0
FIXED_PATHFINDING_BBOX_PADDING = 20

# =============================================================================
# STEP 4: EVALUATION FUNCTION
# =============================================================================

def calculate_raster_iou(pred_mask, true_mask):
    """Calculating IoU (Intersection over Union) for two NumPy masks."""
    try:
        pred_bool = pred_mask.astype(bool)
        true_bool = true_mask.astype(bool)

        intersection = np.logical_and(pred_bool, true_bool).sum()
        union = np.logical_or(pred_bool, true_bool).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return intersection / union
    except Exception as e:
        print(f"Error in calculate_raster_iou: {e}")
        print(f"  Shape Pred: {pred_mask.shape}, Shape True: {true_mask.shape}")
        return 0.0

def calculate_vector_iou(pred_shp_path, true_shp_path, buffer_m=3.0):
    """Calculating the IoU (Intersection over Union) for two SHP files with buffers."""
    try:
        pred_gdf = gpd.read_file(pred_shp_path)
        true_gdf = gpd.read_file(true_shp_path)

        if pred_gdf.crs != true_gdf.crs:
            print(f"Warning: CRS mismatch. Reprojecting prediction to match ground truth CRS: {true_gdf.crs}")
            pred_gdf = pred_gdf.to_crs(true_gdf.crs)

        pred_buffer = pred_gdf.buffer(buffer_m)
        true_buffer = true_gdf.buffer(buffer_m)

        pred_geom = pred_buffer.unary_union
        true_geom = true_buffer.unary_union

        intersection_area = pred_geom.intersection(true_geom).area
        union_area = pred_geom.union(true_geom).area

        if union_area == 0:
            return 1.0 if intersection_area == 0 else 0.0

        return intersection_area / union_area
    except Exception as e:
        print(f"Error in calculate_vector_iou: {e}")
        return 0.0

# =============================================================================
# STEP 5: PIPELINE FUNCTION (CHOPPED UP)
# =============================================================================
# These are the functions from ‘post_process_and_vectorize’ that have been split up
# so that we can call them one by one.

def run_cleaner_logic(prob_map, threshold, kernel_size, min_obj):
    """Stage 1: Converting the probability map into a binary mask."""
    binary_mask = (prob_map > threshold).astype(np.uint8)
    cleaned_mask = remove_small_objects(binary_mask.astype(bool), min_size=min_obj).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size))
    cleaned_mask = binary_opening(cleaned_mask, footprint=kernel).astype(np.uint8)
    return cleaned_mask

def run_gap_filler_logic(cleaned_mask, prob_map, iterations, max_gap, max_cost_norm):
    """Stage 2: Filling in the gaps in the cleaned binary mask."""

    global_height, global_width = cleaned_mask.shape

    # 1. Binary Closing
    closed_mask = binary_closing(cleaned_mask, iterations=iterations).astype(np.uint8)

    # 2. Pathfinding
    skeleton = skeletonize(closed_mask).astype(np.uint8)
    skeleton_obj = skan.Skeleton(skeleton)

    try:
        degrees = np.array(skeleton_obj.graph.sum(axis=1)).flatten()
        degree1_node_ids = np.where(degrees == 1)[0]
        endpoints = [skeleton_obj.coordinates[node_id] for node_id in degree1_node_ids]
    except Exception:
        endpoints = []

    if len(endpoints) < 2:
        return closed_mask

    cost_map = (1.0 - prob_map) * FIXED_PATHFINDING_COST_FACTOR
    cost_map = np.maximum(cost_map, 1e-6)
    path_mask = np.zeros_like(closed_mask)

    endpoint_array = np.array(endpoints)
    tree = KDTree(endpoint_array)
    pairs_to_check = tree.query_pairs(r=max_gap)

    for (i, j) in pairs_to_check:
        start_point = endpoints[i]
        end_point = endpoints[j]
        try:
            r_min = int(min(start_point[0], end_point[0]))
            c_min = int(min(start_point[1], end_point[1]))
            r_max = int(max(start_point[0], end_point[0]))
            c_max = int(max(start_point[1], end_point[1]))
            r_start, c_start = max(0, r_min - FIXED_PATHFINDING_BBOX_PADDING), max(0, c_min - FIXED_PATHFINDING_BBOX_PADDING)
            r_end, c_end = min(global_height, r_max + FIXED_PATHFINDING_BBOX_PADDING), min(global_width, c_max + FIXED_PATHFINDING_BBOX_PADDING)
            sub_cost_map = cost_map[r_start:r_end, c_start:c_end]
            local_start_rc = (start_point[0] - r_start, start_point[1] - c_start)
            local_end_rc = (end_point[0] - r_start, end_point[1] - c_start)
            path_indices_local, cost = route_through_array(sub_cost_map, local_start_rc, local_end_rc, fully_connected=True, geometric=True)
            if 0 < len(path_indices_local) < (max_gap * 1.5):
                path_len = len(path_indices_local)
                path_cost_sum = sum(sub_cost_map[r, c] for r, c in path_indices_local)
                avg_cost = path_cost_sum / path_len
                if avg_cost < (max_cost_norm * FIXED_PATHFINDING_COST_FACTOR):
                    for local_r, local_c in path_indices_local:
                        path_mask[local_r + r_start, local_c + c_start] = 1
        except Exception:
            pass

    return np.maximum(closed_mask, path_mask).astype(np.uint8)

def run_vectorizer_logic(connected_mask, meta, min_length_m, temp_shp_path="temp_tuning_vector.shp"):
    """Stage 3: Converting the final mask to SHP format and filtering it."""

    final_skeleton = skeletonize(connected_mask).astype(np.uint8)

    try:
        final_skeleton_obj = skan.Skeleton(final_skeleton)
        final_branch_data = skan.summarize(final_skeleton_obj, separator='_')
    except Exception:
        final_branch_data = pd.DataFrame()

    transform = meta['transform']
    geometries = []

    for index, row in final_branch_data.iterrows():
        try:
            path_coords_px = final_skeleton_obj.path_coordinates(index)
            path_coords_geo = [rasterio.transform.xy(transform, c[0], c[1]) for c in path_coords_px]
            if len(path_coords_geo) >= 2:
                geometries.append(LineString(path_coords_geo))
        except Exception:
            pass

    if not geometries:
        return None

    gdf = gpd.GeoDataFrame(geometry=geometries, crs=meta['crs'])
    gdf['length_m'] = gdf.geometry.length

    gdf_filtered = gdf[gdf['length_m'] >= min_length_m].copy()
    gdf_filtered = gdf_filtered[~gdf_filtered.is_empty].reset_index(drop=True)

    if gdf_filtered.empty:
        return None

    gdf_filtered.to_file(temp_shp_path, driver='ESRI Shapefile')
    return temp_shp_path

# =============================================================================
# STEP 6: MAIN FUNCTION (MAIN SEQUENTIAL TUNING)
# =============================================================================

def main_tuning():
    print("Starting Sequential Parameter Tuning...")

    print(f"Loading probability map from: {PROB_MAP_PATH}")
    prob_map_full = np.load(PROB_MAP_PATH)

    print(f"Loading raster ground truth from: {GROUND_TRUTH_RASTER_PATH}")
    with rasterio.open(GROUND_TRUTH_RASTER_PATH) as src:
        true_mask_raster_full = (src.read(1) > 0).astype(bool)

    print(f"Loading vector ground truth from: {GROUND_TRUTH_SHP_PATH}")

    if not os.path.exists(GROUND_TRUTH_SHP_PATH):
        print("Warning: Vector ground truth (SHP) not found. Skipping vector tuning.")
        run_vector_tuning = False
    else:
        run_vector_tuning = True

    print(f"Loading metadata from: {IMAGE_PATH_FOR_META}")
    with rasterio.open(IMAGE_PATH_FOR_META) as src:
        geo_meta = src.meta.copy()

    h_prob, w_prob = prob_map_full.shape
    h_true, w_true = true_mask_raster_full.shape

    if (h_prob != h_true) or (w_prob != w_true):
        print(f"Warning: Shape mismatch detected! Cropping to smallest common shape.")
        print(f"  Probability Map Shape: ({h_prob}, {w_prob})")
        print(f"  Shape Ground Truth: ({h_true}, {w_true})")

        h_final = min(h_prob, h_true)
        w_final = min(w_prob, w_true)

        prob_map = prob_map_full[:h_final, :w_final]
        true_mask_raster = true_mask_raster_full[:h_final, :w_final]

        print(f"New shape after cropping: ({h_final}, {w_final})")
    else:
        prob_map = prob_map_full
        true_mask_raster = true_mask_raster_full
        print("Shapes match")

    # =========================================================
    # --- STAGE 1: TUNE CLEANER (Raster IoU) ---
    # =========================================================
    print("\n--- STAGE 1: Finding the Best Cleaner Parameters (Raster IoU) ---")
    best_cleaner_score = -1.0
    best_cleaner_params = {}

    cleaner_combinations = list(itertools.product(
        SEARCH_PROBABILITY_THRESHOLD,
        SEARCH_OPENING_KERNEL_SIZE,
        SEARCH_MIN_OBJECT_SIZE_PIXELS
    ))

    for threshold, kernel_size, min_obj in tqdm(cleaner_combinations, desc="Tuning Cleaner"):
        cleaned_mask = run_cleaner_logic(prob_map, threshold, kernel_size, min_obj)
        score = calculate_raster_iou(cleaned_mask, true_mask_raster)

        if score > best_cleaner_score:
            best_cleaner_score = score
            best_cleaner_params = {
                "threshold": threshold,
                "kernel_size": kernel_size,
                "min_obj": min_obj
            }

    print(f"Stage 1 Results -> Best Cleaner Parameters: {best_cleaner_params} (IoU: {best_cleaner_score:.4f})")

    best_cleaned_mask = run_cleaner_logic(
        prob_map,
        best_cleaner_params['threshold'],
        best_cleaner_params['kernel_size'],
        best_cleaner_params['min_obj']
    )

    # =========================================================
    # --- STAGE 2: TUNE GAP FILLER (Raster IoU) ---
    # =========================================================
    print("\n--- STAGE 2: Identifying the Optimal Gap Filler Parameters (Raster IoU) ---")
    best_gap_score = -1.0
    best_gap_params = {}

    gap_filler_combinations = list(itertools.product(
        SEARCH_CLOSING_ITERATIONS,
        SEARCH_PATHFINDING_MAX_GAP_PIXELS,
        SEARCH_PATHFINDING_MAX_AVG_COST
    ))

    for iterations, max_gap, max_cost_norm in tqdm(gap_filler_combinations, desc="Tuning Gap Filler"):
        connected_mask = run_gap_filler_logic(
            best_cleaned_mask, 
            prob_map,         
            iterations,
            max_gap,
            max_cost_norm
        )
        score = calculate_raster_iou(connected_mask, true_mask_raster)

        if score > best_gap_score:
            best_gap_score = score
            best_gap_params = {
                "iterations": iterations,
                "max_gap": max_gap,
                "max_cost_norm": max_cost_norm
            }

    print(f"Stage 2 Results -> Best Gap Filler Parameters: {best_gap_params} (IoU: {best_gap_score:.4f})")

    best_connected_mask = run_gap_filler_logic(
        best_cleaned_mask,
        prob_map,
        best_gap_params['iterations'],
        best_gap_params['max_gap'],
        best_gap_params['max_cost_norm']
    )

    print(f"Save the best raster mask to {BEST_RASTER_PATH}...")
    geo_meta.update(dtype='uint8', count=1, compress='lzw', driver='GTiff')
    with rasterio.open(BEST_RASTER_PATH, 'w', **geo_meta) as dst:
        dst.write(best_connected_mask, 1)


    # =========================================================
    # --- STAGE 3: TUNE VECTOR FILTER (Vector IoU) ---
    # =========================================================
    if not run_vector_tuning:
        print("\n--- STAGE 3: Skipped (SHP ground truth not found) ---")
        best_min_length = SEARCH_MIN_ROAD_LENGTH_METERS[0] 
    else:
        print(f"\n--- STAGE 3: Finding the Best Long Filter (Vector IoU, Buffer {VECTOR_BUFFER_METERS}m) ---")
        best_vector_score = -1.0
        best_min_length = None

        temp_shp_for_eval = "/content/drive/MyDrive/temp_eval.shp"

        for min_length in tqdm(SEARCH_MIN_ROAD_LENGTH_METERS, desc="Tuning Vector Filter"):
            temp_shp_path = run_vectorizer_logic(
                best_connected_mask,
                geo_meta,
                min_length,
                temp_shp_for_eval
            )

            if temp_shp_path is None:
                print(f"Warning: No geometry has been generated for min_length={min_length}m. Score = 0.")
                score = 0.0
            else:
                score = calculate_vector_iou(
                    temp_shp_path,
                    GROUND_TRUTH_SHP_PATH,
                    buffer_m=VECTOR_BUFFER_METERS
                )

            print(f"Testing min_length = {min_length}m ... Vector IoU = {score:.4f}")

            if score > best_vector_score:
                best_vector_score = score
                best_min_length = min_length

        print(f"Stage 3 Results -> Best Long Filters: {best_min_length}m (Vector IoU: {best_vector_score:.4f})")

        print(f"Saving the best SHP files (with filters {best_min_length}m) to {BEST_SHP_PATH}...")
        run_vectorizer_logic(best_connected_mask, geo_meta, best_min_length, BEST_SHP_PATH)


    # =========================================================
    # --- FINAL REPORT ---
    # =========================================================
    print("\n--- === TUNING PARAMETERS REPORT COMPLETE === ---")
    print("Optimal parameters found:")
    print("-------------------------------------------------")
    print(f"PROBABILITY_THRESHOLD:     {best_cleaner_params.get('threshold')}")
    print(f"OPENING_KERNEL_SIZE:       {best_cleaner_params.get('kernel_size')}")
    print(f"MIN_OBJECT_SIZE_PIXELS:  {best_cleaner_params.get('min_obj')}")
    print("-------------------------------------------------")
    print(f"CLOSING_ITERATIONS:        {best_gap_params.get('iterations')}")
    print(f"PATHFINDING_MAX_GAP_PIXELS:  {best_gap_params.get('max_gap')}")
    print(f"PATHFINDING_MAX_AVG_COST:  {best_gap_params.get('max_cost_norm')}")
    print("-------------------------------------------------")
    print(f"MIN_ROAD_LENGTH_METERS:    {best_min_length}")
    print("-------------------------------------------------")
    print(f"Parameter Tetap (Fixed):")
    print(f"PATHFINDING_COST_FACTOR:   {FIXED_PATHFINDING_COST_FACTOR}")
    print(f"PATHFINDING_BBOX_PADDING:    {FIXED_PATHFINDING_BBOX_PADDING}")
    print("\nDone.")


if __name__ == "__main__":
    main_tuning()

try:
    from google.colab import runtime
    print("\nTuning complete. Disconnecting from the runtime.")
    runtime.unassign()
except ImportError:
    print("\nTuning complete.")

