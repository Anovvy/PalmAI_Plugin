# -*- coding: utf-8 -*-
"""Spesific Slicing to PNG.ipynb
"""

# Install necessary libraries if not already installed
# !pip install rasterio geopandas

# Import libraries
import rasterio
import geopandas
import os

# Define input paths
# Replace with your actual file paths
input_raster_path = "/content/drive/MyDrive/YOLO-Tree Counting/Data Anotasi/Data Anotasi Oryctes/ORTHO_SKPE_A.tif"
input_shapefile_path = "/content/drive/MyDrive/YOLO-Tree Counting/Data Anotasi/Data Anotasi Oryctes/Spesific Point/SKPE/SKPE_A_False.shp"
output_directory = "/content/drive/MyDrive/YOLO-Tree Counting/Data Anotasi/Data Anotasi Oryctes/Spesific SKPE Slices"

# Define clipping dimensions (in pixels)
clip_width = 1280 
clip_height = 1280 

# Define the buffer distance in pixels
buffer_distance_pixels = 80


# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

print(f"Input Raster Path: {input_raster_path}")
print(f"Input Shapefile Path: {input_shapefile_path}")
print(f"Output Directory: {output_directory}")

# Load the shapefile
try:
    gdf_points = geopandas.read_file(input_shapefile_path)
    if gdf_points.empty:
        print(f"Error: The shapefile at '{input_shapefile_path}' could not be loaded or is empty.")
    else:
        print(f"Shapefile loaded successfully from '{input_shapefile_path}'.")
        display(gdf_points.head())
        print(f"Number of points in the shapefile: {len(gdf_points)}")
except Exception as e:
    print(f"An error occurred while loading the shapefile: {e}")
    gdf_points = None # Set gdf_points to None to indicate loading failed

from rasterio.windows import Window
from rasterio.transform import from_bounds

def clip_raster_at_point(raster_path, output_dir, center_x, center_y, clip_width, clip_height, point_id):
    """
    Clips a raster image based on a center point and specified dimensions.

    Args:
        raster_path (str): Path to the input raster file.
        output_dir (str): Directory to save the clipped image.
        center_x (float): X coordinate of the center point.
        center_y (float): Y coordinate of the center point.
        clip_width (int): Width of the clipping window in pixels.
        clip_height (int): Height of the clipping window in pixels.
        point_id (str or int): Identifier for the point, used in the output filename.
    """
    try:
        with rasterio.open(raster_path) as src:
            # Ensure the point coordinates are in the raster's CRS
            row, col = src.index(center_x, center_y)

            # Calculate the top-left corner of the window
            col_start = int(col - clip_width / 2)
            row_start = int(row - clip_height / 2)

            # Define the window
            window = Window(col_start, row_start, clip_width, clip_height)

            # Read the data from the window
            clipped_data = src.read(window=window)

            # Get the transform for the window
            # Define the output path with .png extension
            output_path = os.path.join(output_dir, f"clipped_point_{point_id}.png")

            # Write the clipped data to a new PNG file
            with rasterio.open(
                output_path,
                'w',
                driver='PNG', 
                width=clip_width,
                height=clip_height,
                count=src.count,
                dtype=src.dtypes[0],
            ) as dst:
                dst.write(clipped_data)

            print(f"Successfully clipped raster for point {point_id} and saved to {output_path}")

    except Exception as e:
        print(f"Error clipping raster for point {point_id}: {e}")

from shapely.geometry import Point

# Process points
if gdf_points is not None and not gdf_points.empty:

    # Create a temporary column for sorting by y-coordinate
    gdf_points['y_coord'] = gdf_points.geometry.y
    # Sort by y-coordinate
    gdf_points_sorted = gdf_points.sort_values(by='y_coord', ascending=False)
    # Drop the temporary column
    gdf_points_sorted = gdf_points_sorted.drop(columns=['y_coord'])


    valid_points_data = []
    invalidated_points_indices = set()

    for index, row in gdf_points_sorted.iterrows():
        if index in invalidated_points_indices:
            print(f"Point {row.get('id', index)} at coordinates ({row.geometry.x}, {row.geometry.y}) was already invalidated. Skipping.")
            continue 

        point_geometry = row.geometry
        point_id = row.get('id', index)

        # Access y-coordinate using row.geometry.y
        print(f"Processing point {point_id} at coordinates ({point_geometry.x}, {point_geometry.y})")

        # Create a buffer around the current point
        point_buffer = point_geometry.buffer(buffer_distance_crs)

        # Add the current point to the list of valid points
        valid_points_data.append(row)

        # Find other points that fall within this buffer
        for other_index, other_row in gdf_points.iterrows():
            if other_index != index and other_row.geometry.within(point_buffer):
                 invalidated_points_indices.add(other_index)
                 print(f"Point {other_row.get('id', other_index)} at coordinates ({other_row.geometry.x}, {other_row.geometry.y}) is within the buffer of point {point_id}. Invalidating.")


    # Create a new GeoDataFrame with the valid points
    gdf_valid_points = geopandas.GeoDataFrame(valid_points_data, crs=gdf_points.crs)

    print(f"Found {len(gdf_valid_points)} valid points after processing with buffers.")

    # Iterate through each valid point and clip the raster
    for index, row in gdf_valid_points.iterrows():
        point_geometry = row.geometry
        point_id = row.get('id', index)

        # Get the center coordinates
        center_x = point_geometry.x
        center_y = point_geometry.y

        print(f"Clipping raster for valid point {point_id} at coordinates ({center_x}, {center_y})")


        # Clip the raster at the current point
        clip_raster_at_point(
            input_raster_path,
            output_directory,
            center_x,
            center_y,
            clip_width,
            clip_height,
            point_id
        )

else:
    print("No points to process. Shapefile was not loaded or is empty.")

