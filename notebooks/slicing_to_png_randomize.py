# -*- coding: utf-8 -*-
"""Slicing to PNG Randomize.ipynb
"""

# !pip install rasterio

import os
import rasterio
from rasterio.windows import Window
from itertools import product
import numpy as np
from PIL import Image

# 1. imagery path configuration
input_raster_path = '/content/drive/MyDrive/YOLO-Tree Counting/Data Estate Mapping/LAR.tif'

# 2. output path configuration
output_directory = '/content/drive/MyDrive/YOLO-Tree Counting/Data Estate Mapping/LAR Slices'

import random

def chip_raster_to_png(input_file, output_dir, chip_width=640, chip_height=640, max_chips=300):
    """
    Split a raster file (GeoTIFF) into several sections (chips)
    and save them as PNG files. Only chips of the correct size are saved
    and those containing fewer than 100 NaN pixels. The sections are selected at random.
    WARNING: The PNG format will remove all georeference data.

    Args:
        input_file (str): The full path to the .tif file to be cropped.
        output_dir (str): Directory for storing PNG cropped images.
        chip_width (int): The width of each slice in pixels.
        chip_height (int): The height of each slice in pixels.
        max_chips (int): The maximum number of chips to be stored.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' has been created.")

    with rasterio.open(input_file) as src:
        total_width = src.width
        total_height = src.height

        print(f"Original Image Size: {total_width}x{total_height} pixel")
        print(f"Target Cut Size: {chip_width}x{chip_height} pixel")
        print(f"Maximum Number of Saved Discounts: {max_chips}")
        print("Save as PNG (geospatial data will be lost).")
        print("Only slices of the correct size and with fewer than 100 NaN pixels will be saved.")
        print("-" * 30)

        col_offsets = range(0, total_width, chip_width)
        row_offsets = range(0, total_height, chip_height)

        possible_windows = []
        for col_off, row_off in product(col_offsets, row_offsets):
            width = min(chip_width, total_width - col_off)
            height = min(chip_height, total_height - row_off)

            if width == chip_width and height == chip_height:
                possible_windows.append((col_off, row_off, width, height))

        random.shuffle(possible_windows)

        saved_chips_count = 0
        processed_chips_count = 0

        for col_off, row_off, width, height in possible_windows:
            if saved_chips_count >= max_chips:
                print(f"The maximum number of slices ({max_chips}) has been reached. Stopping processing.")
                break

            processed_chips_count += 1
            window = Window(col_off, row_off, width, height)

            # Reading data from the window (bands)
            try:
                chip_data = src.read([1, 2, 3], window=window)
            except IndexError:
                print("  -> Warning: The image does not have three bands. Attempting to read the first band.")
                chip_data = src.read(1, window=window)

            # Check the number of NaN pixels
            nan_count = np.sum(np.isnan(chip_data))

            if nan_count < 100:
                print(f"Processing the {processed_chips_count}th chip: Size {width}x{height}, NaN: {nan_count}. SAVING....")

                # Normalising the data to the range 0–255
                if chip_data.ndim == 3: 
                    scaled_data = np.empty_like(chip_data, dtype=np.uint8)
                    for i in range(chip_data.shape[0]):
                        band = chip_data[i, :, :]
                        min_val, max_val = np.nanmin(band), np.nanmax(band)
                        if max_val > min_val:
                            # Scale to 0–255
                            scaled_data[i, :, :] = ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else:
                            scaled_data[i, :, :] = np.zeros_like(band, dtype=np.uint8)

                    # Change the dimension order from (band, height, width) to (height, width, band)
                    # to match Pillow/PIL
                    img_data = np.transpose(scaled_data, (1, 2, 0))
                    img = Image.fromarray(img_data, 'RGB')

                else: # If the data is single-band (greyscale)
                    min_val, max_val = np.nanmin(chip_data), np.nanmax(chip_data)
                    if max_val > min_val:
                        scaled_data = ((chip_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        scaled_data = np.zeros_like(chip_data, dtype=np.uint8)
                    img = Image.fromarray(scaled_data, 'L')

                base_filename = os.path.splitext(os.path.basename(input_file))[0]
                output_filename = f"{base_filename}_chip_{row_off//chip_height}_{col_off//chip_width}.png"
                output_path = os.path.join(output_dir, output_filename)

                img.save(output_path)

                saved_chips_count += 1
            else:
                print(f"Processing the {processed_chips_count}th slice: Size {width}x{height}, NaN: {nan_count}. SKIPPED (too many NaNs).")


    print("-" * 30)
    print(f"The cropping process is complete. {saved_chips_count} PNG cropped images have been successfully saved to the directory '{output_dir}'.")

chip_raster_to_png(input_raster_path, output_directory, chip_width=640, chip_height=640)

