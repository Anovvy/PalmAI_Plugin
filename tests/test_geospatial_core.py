# -*- coding: utf-8 -*-
"""Synthetic geospatial core tests: CRS, affine transform, bboxes, centroids, and attributes."""

import os
import shutil
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon


class TestGeospatialCore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="palmai_test_gis_")
        self.raster_path = os.path.join(self.tmp, "synthetic_ortho.tif")

        # Create tiny synthetic GeoTIFF (100x100 px, 3 bands, UTM Zone 48N - EPSG:32648)
        self.width = 100
        self.height = 100
        self.crs = "EPSG:32648"
        self.transform = from_origin(500000.0, 100000.0, 0.05, 0.05)  # 5cm GSD

        data = (np.random.rand(3, self.height, self.width) * 255).astype(np.uint8)
        with rasterio.open(
            self.raster_path,
            "w",
            driver="GTiff",
            height=self.height,
            width=self.width,
            count=3,
            dtype=np.uint8,
            crs=self.crs,
            transform=self.transform,
        ) as dst:
            dst.write(data)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_raster_affine_metadata(self):
        with rasterio.open(self.raster_path) as src:
            self.assertEqual(src.width, 100)
            self.assertEqual(src.height, 100)
            self.assertEqual(src.count, 3)
            self.assertEqual(src.crs.to_string(), "EPSG:32648")
            self.assertAlmostEqual(src.res[0], 0.05)
            self.assertAlmostEqual(src.res[1], 0.05)

    def test_pixel_to_geospatial_bounding_box_and_centroid(self):
        with rasterio.open(self.raster_path) as src:
            tf = src.transform

            # Simulate object detection pixel coordinates (x1, y1, x2, y2)
            px_minx, px_miny, px_maxx, px_maxy = 20, 20, 40, 40

            top_left = tf @ (px_minx, px_miny) if hasattr(tf, "__matmul__") else (tf * (px_minx, px_miny))
            bottom_right = tf @ (px_maxx, px_maxy) if hasattr(tf, "__matmul__") else (tf * (px_maxx, px_maxy))

            # Construct spatial polygon
            poly = Polygon.from_bounds(top_left[0], bottom_right[1], bottom_right[0], top_left[1])
            centroid = poly.centroid

            self.assertTrue(poly.is_valid)
            self.assertFalse(poly.is_empty)
            self.assertTrue(centroid.within(poly))

            # Mathematical check of expected coordinates
            expected_minx = 500000.0 + (20 * 0.05)
            expected_maxy = 100000.0 - (20 * 0.05)
            self.assertAlmostEqual(top_left[0], expected_minx)
            self.assertAlmostEqual(top_left[1], expected_maxy)

    def test_vector_shapefile_export(self):
        import geopandas as gpd

        with rasterio.open(self.raster_path) as src:
            tf = src.transform
            boxes = []
            centroids = []
            categories = ["Healthy", "Yellowish", "Dead"]

            for i in range(3):
                p1 = tf @ (10 + i * 20, 10 + i * 20) if hasattr(tf, "__matmul__") else tf * (10 + i * 20, 10 + i * 20)
                p2 = tf @ (25 + i * 20, 25 + i * 20) if hasattr(tf, "__matmul__") else tf * (25 + i * 20, 25 + i * 20)
                poly = Polygon.from_bounds(p1[0], p2[1], p2[0], p1[1])
                boxes.append(poly)
                centroids.append(poly.centroid)

            gdf_boxes = gpd.GeoDataFrame(
                {"score": [0.95, 0.88, 0.72], "category": categories},
                geometry=boxes,
                crs=self.crs,
            )
            gdf_points = gpd.GeoDataFrame(
                {"score": [0.95, 0.88, 0.72], "category": categories},
                geometry=centroids,
                crs=self.crs,
            )

            bbox_shp = os.path.join(self.tmp, "test_boxes.shp")
            pts_shp = os.path.join(self.tmp, "test_points.shp")

            gdf_boxes.to_file(bbox_shp)
            gdf_points.to_file(pts_shp)

            # Re-read and verify schema & CRS persistence
            read_boxes = gpd.read_file(bbox_shp)
            read_points = gpd.read_file(pts_shp)

            self.assertEqual(len(read_boxes), 3)
            self.assertEqual(len(read_points), 3)
            self.assertEqual(read_boxes.crs.to_string(), "EPSG:32648")
            self.assertListEqual(list(read_boxes["category"]), categories)


if __name__ == "__main__":
    unittest.main()
