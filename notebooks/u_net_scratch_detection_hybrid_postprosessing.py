# -*- coding: utf-8 -*-
"""U-Net_Scratch_Detection_Hybrid-PostProsessing.ipynb
"""

# =============================================================================
# STEP 1: INSTALLING THE LIBRARY AND IMPORTING
# =============================================================================
print("Installing the library...")
# !pip install rasterio geopandas scikit-image tqdm skan torch torchvision pandas networkx scipy

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import features
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from skimage.morphology import skeletonize, remove_small_objects, binary_opening
from skimage.graph import route_through_array
import skan
from tqdm.notebook import tqdm
import os
import math
import itertools
from scipy.spatial import KDTree
from scipy.ndimage import binary_closing
from google.colab import drive

# Mount Google Drive to access files
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    print("Make sure you grant permission in the authentication pop-up.")

# =============================================================================
# STEP 2: CONFIGURING PATH AND PARAMETERS
# =============================================================================
print("Parameter configuration...")
# The path to your model file on Google Drive
MODEL_PATH = '/content/drive/MyDrive/Road Detection_Data_/Model/u-net_roaddetection_3.pth.tar'
# Path to the .tif image to be used for inference
IMAGE_PATH = '/content/drive/MyDrive/Road Detection_Data_/Data Test/Test.tif'
# Path for saving the shapefile output
OUTPUT_SHAPEFILE_PATH = '/content/drive/MyDrive/Road Detection_Data_/Hasil Prediksi/U-Net/UNet_Detection_4.1.shp'
# Path for saving raster results
OUTPUT_RASTER_PATH = '/content/drive/MyDrive/Road Detection_Data_/Hasil Prediksi/U-Net/UNet_Detection_4.1.tif'
# Path for storing the probability map (for tuning)
OUTPUT_PROB_MAP_PATH = '/content/drive/MyDrive/Road Detection_Data_/Hasil Prediksi/U-Net/prob_map_4.0.npy'

# Parameters for Tiling/Sliding Window
TILE_SIZE = 750
OVERLAP = 150
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- PARAMETER POST-PROCESSING ---
OPENING_KERNEL_SIZE = 7
MIN_OBJECT_SIZE_PIXELS = 758
CLOSING_ITERATIONS = 50
PROBABILITY_THRESHOLD = 0.3075

# --- PARAMETER PATHFINDING ---
PATHFINDING_MAX_GAP_PIXELS = 500
PATHFINDING_COST_FACTOR = 15.0
PATHFINDING_MAX_AVG_COST = 1.0552
PATHFINDING_BBOX_PADDING = 20

# --- PARAMETER FILTER AKHIR ---
MIN_ROAD_LENGTH_METERS = 14.0

# =============================================================================
# STEP 3: DEFINITION OF THE U-NET MODEL ARCHITECTURE (FROM SCRATCH)
# =============================================================================
print("Defining the U-Net architecture...")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)

# =============================================================================
# STEP 4: CREATING A HELPER FUNCTION
# =============================================================================
print("Defining helper functions...")

def load_model(model_path):
    """Loading the U-Net model (from scratch) and its weights."""
    print(f"Loading model from: {model_path}")
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    try:
        checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return model

def inference_tiling(model, image_path):
    """Performing inference on large images using the tiling method."""
    try:
        with rasterio.open(image_path) as src:
            width, height = src.width, src.height
            meta = src.meta.copy()
            print(f"Image opened: {width}x{height} pixels, {src.count} bands.")

            predictions_sum = np.zeros((height, width), dtype=np.float32)
            overlap_count = np.zeros((height, width), dtype=np.uint8)
            step = TILE_SIZE - OVERLAP

            tiles_to_process = []
            for y in range(0, height, step):
                for x in range(0, width, step):
                    y_win = min(y, height - TILE_SIZE)
                    x_win = min(x, width - TILE_SIZE)
                    window = Window(x_win, y_win, TILE_SIZE, TILE_SIZE)
                    tiles_to_process.append(window)

            for i in tqdm(range(0, len(tiles_to_process), BATCH_SIZE), desc="Processing Batches"):
                batch_windows = tiles_to_process[i:i+BATCH_SIZE]
                batch_tiles = []
                for window in batch_windows:
                    tile = src.read(window=window)[:3, :, :].astype(np.float32) / 255.0
                    batch_tiles.append(torch.from_numpy(tile))

                batch_tensor = torch.stack(batch_tiles).to(DEVICE)
                with torch.no_grad():
                    preds = model(batch_tensor)
                    preds = torch.sigmoid(preds)
                    preds = preds.cpu().numpy().squeeze(1)

                for j, window in enumerate(batch_windows):
                    pred_tile = preds[j]
                    predictions_sum[window.row_off:window.row_off+TILE_SIZE, window.col_off:window.col_off+TILE_SIZE] += pred_tile
                    overlap_count[window.row_off:window.row_off+TILE_SIZE, window.col_off:window.col_off+TILE_SIZE] += 1

            print("Averaging predictions in overlapping areas...")
            overlap_count[overlap_count == 0] = 1
            final_prediction = predictions_sum / overlap_count

            return final_prediction, meta
    except Exception as e:
        print(f"Error during tiling inference: {e}")
        return None, None

def post_process_and_vectorize(prediction_mask, meta, output_vector_path, output_raster_path=None):
    """
    POST-PROCESSING (Smart A*)
    """
    global_height, global_width = prediction_mask.shape

    # --- PART 1: INITIAL CLEANING ---
    print("Applying threshold...")
    binary_mask = (prediction_mask > PROBABILITY_THRESHOLD).astype(np.uint8)
    print(f"Removing small objects < {MIN_OBJECT_SIZE_PIXELS} px...")
    cleaned_mask = remove_small_objects(binary_mask.astype(bool), min_size=MIN_OBJECT_SIZE_PIXELS).astype(np.uint8)
    print(f"Cleaning mask with {OPENING_KERNEL_SIZE}x{OPENING_KERNEL_SIZE} kernel...")
    kernel = np.ones((OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE))
    cleaned_mask = binary_opening(cleaned_mask, footprint=kernel).astype(np.uint8)

    # --- PART 2: BINARY CLOSING ---
    print(f"Connecting segments with {CLOSING_ITERATIONS} iterations of binary closing...")
    cleaned_mask = binary_closing(cleaned_mask, iterations=CLOSING_ITERATIONS).astype(np.uint8)

    # --- PART 3: PATHFINDING (Smart A*) ---
    print("Skeletonizing (pass 1) to find gaps...")
    skeleton = skeletonize(cleaned_mask).astype(np.uint8)
    skeleton_obj = skan.Skeleton(skeleton)

    try:
        degrees = np.array(skeleton_obj.graph.sum(axis=1)).flatten()
        degree1_node_ids = np.where(degrees == 1)[0]
        endpoints = [skeleton_obj.coordinates[node_id] for node_id in degree1_node_ids]
        print(f"Found {len(endpoints)} segment endpoints to analyze.")
    except Exception as e:
        print(f"Error analyzing skeleton graph: {e}")
        endpoints = []

    if len(endpoints) < 2:
        print("No endpoints found, skipping pathfinding.")
        connected_mask = cleaned_mask
    else:
        print("Creating cost map from probabilities...")
        cost_map = (1.0 - prediction_mask) * PATHFINDING_COST_FACTOR
        cost_map = np.maximum(cost_map, 1e-6)
        path_mask = np.zeros_like(binary_mask)
        connect_count = 0

        print("Building KDTree for fast neighbor search...")
        endpoint_array = np.array(endpoints)
        tree = KDTree(endpoint_array)
        pairs_to_check = tree.query_pairs(r=PATHFINDING_MAX_GAP_PIXELS)

        available_endpoints = set(range(len(endpoints)))
        print(f"Analyzing {len(pairs_to_check)} realistic gap combinations...")

        for (i, j) in tqdm(pairs_to_check, desc="Pathfinding Gaps", total=len(pairs_to_check)):

            if i not in available_endpoints or j not in available_endpoints:
                continue

            start_point = endpoints[i] # (row, col)
            end_point = endpoints[j]   # (row, col)

            try:
                r_min = int(min(start_point[0], end_point[0]))
                c_min = int(min(start_point[1], end_point[1]))
                r_max = int(max(start_point[0], end_point[0]))
                c_max = int(max(start_point[1], end_point[1]))

                r_start = max(0, r_min - PATHFINDING_BBOX_PADDING)
                c_start = max(0, c_min - PATHFINDING_BBOX_PADDING)
                r_end = min(global_height, r_max + PATHFINDING_BBOX_PADDING)
                c_end = min(global_width, c_max + PATHFINDING_BBOX_PADDING)

                sub_cost_map = cost_map[r_start:r_end, c_start:c_end]
                local_start_rc = (start_point[0] - r_start, start_point[1] - c_start)
                local_end_rc = (end_point[0] - r_start, end_point[1] - c_start)

                path_indices_local, cost = route_through_array(
                    sub_cost_map, local_start_rc, local_end_rc, fully_connected=True, geometric=True
                )

                if 0 < len(path_indices_local) < (PATHFINDING_MAX_GAP_PIXELS * 1.5):
                    path_len = len(path_indices_local)
                    path_cost_sum = sum(sub_cost_map[r, c] for r, c in path_indices_local)
                    avg_cost = path_cost_sum / path_len

                    if avg_cost < (PATHFINDING_MAX_AVG_COST * PATHFINDING_COST_FACTOR):
                        for local_r, local_c in path_indices_local:
                            path_mask[local_r + r_start, local_c + c_start] = 1
                        connect_count += 1

                        available_endpoints.remove(i)
                        available_endpoints.remove(j)
            except Exception as e:
                pass

        print(f"Successfully connected {connect_count} gaps.")
        print("Combining masks...")
        connected_mask = np.maximum(cleaned_mask, path_mask).astype(np.uint8)

    # --- SECTION 4: FINAL VECTORISATION ---
    if output_raster_path:
        print(f"Saving final hybrid raster mask to {output_raster_path}")
        meta.update(dtype='uint8', count=1, compress='lzw', driver='GTiff')
        os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
        with rasterio.open(output_raster_path, 'w', **meta) as dst:
            dst.write(connected_mask, 1)

    print("Skeletonizing (pass 2) for final vectorization...")
    final_skeleton = skeletonize(connected_mask).astype(np.uint8)

    print("Vectorizing final skeleton...")
    try:
        final_skeleton_obj = skan.Skeleton(final_skeleton)
        final_branch_data = skan.summarize(final_skeleton_obj, separator='_')
    except Exception as e:
        print(f"Error summarizing final skeleton: {e}")
        final_branch_data = pd.DataFrame()

    transform = meta['transform']
    geometries = []
    initial_skeleton_count = len(final_branch_data)

    for index, row in tqdm(final_branch_data.iterrows(), total=initial_skeleton_count, desc="Creating Geometries"):
        try:
            path_coords_px = final_skeleton_obj.path_coordinates(index)
            path_coords_geo = [rasterio.transform.xy(transform, c[0], c[1]) for c in path_coords_px]
            if len(path_coords_geo) >= 2:
                line = LineString(path_coords_geo)
                geometries.append(line)
        except Exception as e:
            pass

    if not geometries:
        print("Warning: No line features were found to vectorize.")
        return

    gdf = gpd.GeoDataFrame(geometry=geometries, crs=meta['crs'])

    print(f"Original CRS: {gdf.crs}")
    original_crs = gdf.crs # Simpan CRS asli

    print("Reprojecting to EPSG:3857 (meters) for accurate calculation...")
    gdf_meter = gdf.to_crs("EPSG:3857")
    gdf_meter['length_m'] = gdf_meter.geometry.length
    print(f"Filtering polylines shorter than {MIN_ROAD_LENGTH_METERS} meters...")
    gdf_filtered = gdf_meter[gdf_meter['length_m'] >= MIN_ROAD_LENGTH_METERS].copy()

    print("Sanitizing geometries and resetting index...")
    final_gdf_meter = gdf_filtered.reset_index(drop=True)
    final_count = len(final_gdf_meter)

    print(f"Kept {final_count} road segments from {initial_skeleton_count} initial skeleton branches.")

    if final_gdf_meter.empty:
        print("Warning: No features remained after all filtering steps.")
        return

    print(f"Reprojecting back to original CRS ({original_crs}) before saving...")
    final_gdf_original_crs = final_gdf_meter.to_crs(original_crs)

    print(f"Saving final {final_count} valid road segments to {output_vector_path}...")
    os.makedirs(os.path.dirname(output_vector_path), exist_ok=True)
    final_gdf_original_crs.to_file(output_vector_path, driver='ESRI Shapefile')

    print("Processing complete!")

# =============================================================================
# STEP 5: RUN THE MAIN PIPELINE
# =============================================================================
def main_pipeline():
    print("Starting the main pipeline...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}")
        return

    try:
        print("--- Loading the U-Net Model (from scratch) ---")
        model = load_model(MODEL_PATH)
        if not model:
            print("Model loading failed, stopping pipeline.")
            return

        print("\n--- Getting Started with Tilin Inferenceg ---")
        prediction_result, geo_meta = inference_tiling(model, IMAGE_PATH)
        if prediction_result is None:
            print("Inference tiling failed, stopping pipeline.")
            return

        print(f"\n--- Saving the Probability Map (prob_map) ---")
        os.makedirs(os.path.dirname(OUTPUT_PROB_MAP_PATH), exist_ok=True)
        np.save(OUTPUT_PROB_MAP_PATH, prediction_result)
        print(f"Probability maps are stored at: {OUTPUT_PROB_MAP_PATH}")

        print("\n--- Getting Started with Post-Processing & Vectorisation ---")
        post_process_and_vectorize(
            prediction_result,
            geo_meta,
            OUTPUT_SHAPEFILE_PATH,
            OUTPUT_RASTER_PATH
        )
        print("Pipeline finished successfully!")

    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}")
        import traceback
        traceback.print_exc()

# Running the pipeline
if __name__ == "__main__":
    main_pipeline()

# =============================================================================
# STEP 6: DISCONNECT
# =============================================================================
try:
    from google.colab import runtime
    print("\nDisconnecting runtime.")
    runtime.unassign()
except ImportError:
    print("\nNot in the Google Colab environment. Done.")