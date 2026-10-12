# -*- coding: utf-8 -*-
"""Domain QA & Benchmarks: road skeleton topology, line connectivity, and provenance manifests."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from shapely.geometry import LineString
from skimage.morphology import skeletonize

from plugin_qgis.palmai_core.provenance import (
    build_run_manifest,
    write_run_manifest,
)


class TestGeospatialDomainQA(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="palmai_test_qa_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_road_skeletonization_and_line_topology(self):
        # Create a synthetic binary mask with a 6-pixel wide road strip
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:51, 10:90] = 1  # horizontal road bar

        skel = skeletonize(mask).astype(np.uint8)
        self.assertTrue(np.any(skel))

        # Skeleton width must be 1 pixel at any column along the line
        active_rows = np.sum(skel[:, 20:80], axis=0)
        self.assertTrue(np.all(active_rows == 1))

        # Trace points along the skeleton to form a LineString
        coords = np.argwhere(skel == 1)
        # Sort along column coordinate
        coords = coords[coords[:, 1].argsort()]
        line_pts = [(float(c[1]), float(c[0])) for c in coords]
        line = LineString(line_pts)

        self.assertTrue(line.is_valid)
        self.assertFalse(line.is_empty)
        self.assertGreater(line.length, 50.0)

    def test_run_manifest_provenance_and_privacy(self):
        fake_input_tif = os.path.join(self.tmp, "user_secret_data", "ortho.tif")
        fake_out_shp = os.path.join(self.tmp, "user_secret_data", "trees.shp")
        os.makedirs(os.path.dirname(fake_input_tif), exist_ok=True)
        with open(fake_input_tif, "w") as f:
            f.write("raster data")
        with open(fake_out_shp, "w") as f:
            f.write("vector data")

        raw_params = {
            "raster_path": fake_input_tif,
            "bbox_output_path": fake_out_shp,
            "device": "cpu",
            "tile_size": 640,
        }
        results = {
            "total_detections": 88,
            "layers": [
                {"name": "trees", "type": "bbox", "path": fake_out_shp}
            ],
        }
        input_meta = {"crs": "EPSG:32648", "width": 1000, "height": 1000}

        manifest = build_run_manifest(
            job_type="counting",
            params=raw_params,
            results=results,
            input_meta=input_meta,
            elapsed_seconds=3.45,
        )

        # Verify manifest structure
        self.assertEqual(manifest["job_type"], "counting")
        self.assertEqual(manifest["summary"]["total_detections"], 88)
        self.assertEqual(manifest["elapsed_seconds"], 3.45)

        # Verify privacy: parameters MUST NOT expose user absolute host paths
        self.assertEqual(manifest["parameters"]["raster_path"], "ortho.tif")
        self.assertEqual(manifest["parameters"]["bbox_output_path"], "trees.shp")

        # Verify outputs recorded with checksum and size
        self.assertEqual(len(manifest["outputs"]), 1)
        self.assertEqual(manifest["outputs"][0]["filename"], "trees.shp")
        self.assertIsNotNone(manifest["outputs"][0]["sha256"])

        # Write and re-read manifest file
        manifest_path = write_run_manifest(self.tmp, manifest)
        self.assertTrue(os.path.isfile(manifest_path))

        with open(manifest_path, encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["job_type"], "counting")


if __name__ == "__main__":
    unittest.main()
