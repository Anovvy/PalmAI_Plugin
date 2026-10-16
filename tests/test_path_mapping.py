# -*- coding: utf-8 -*-
"""Tests for PathMapper (backends/path_mapping.py)."""

import os
import shutil
import tempfile
import unittest

from plugin_qgis.backends.path_mapping import PathMapper


class TestPathMapper(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="palmai_test_pm_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_mount_deduplication_and_modes(self):
        mapper = PathMapper()
        input_dir = os.path.join(self.test_dir, "inputs")
        output_dir = os.path.join(self.test_dir, "outputs")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        img1 = os.path.join(input_dir, "ortho1.tif")
        img2 = os.path.join(input_dir, "ortho2.tif")
        out1 = os.path.join(output_dir, "trees.shp")
        out2 = os.path.join(output_dir, "centroids.shp")

        # Two inputs in same dir -> single mount point
        c_img1 = mapper.file_to_container(img1, writable=False)
        c_img2 = mapper.file_to_container(img2, writable=False)

        # Output in separate dir -> second mount point
        c_out1 = mapper.file_to_container(out1, writable=True)
        c_out2 = mapper.file_to_container(out2, writable=True)

        self.assertTrue(c_img1.startswith("/mnt/0/"))
        self.assertTrue(c_img2.startswith("/mnt/0/"))
        self.assertTrue(c_out1.startswith("/mnt/1/"))
        self.assertTrue(c_out2.startswith("/mnt/1/"))

        specs = mapper.mount_specs()
        self.assertEqual(len(specs), 2)
        # First mount is input-only -> ro
        self.assertEqual(specs[0][2], "ro")
        # Second mount is output -> rw
        self.assertEqual(specs[1][2], "rw")

    def test_shared_input_output_folder_elevates_to_rw(self):
        mapper = PathMapper()
        shared_dir = os.path.join(self.test_dir, "shared")
        os.makedirs(shared_dir, exist_ok=True)

        raster = os.path.join(shared_dir, "input.tif")
        output = os.path.join(shared_dir, "output.shp")

        # Register input first (ro)
        mapper.file_to_container(raster, writable=False)
        self.assertEqual(mapper.mount_specs()[0][2], "ro")

        # Register output in the same folder -> elevates to rw
        mapper.file_to_container(output, writable=True)
        self.assertEqual(len(mapper.mount_specs()), 1)
        self.assertEqual(mapper.mount_specs()[0][2], "rw")

    def test_spaces_and_unicode_in_paths(self):
        mapper = PathMapper()
        special_dir = os.path.join(self.test_dir, "kebun sawit blok A 🌴")
        os.makedirs(special_dir, exist_ok=True)
        file_path = os.path.join(special_dir, "foto udara.tif")

        cpath = mapper.file_to_container(file_path, writable=False)
        self.assertTrue(cpath.startswith("/mnt/0/"))
        self.assertEqual(os.path.basename(cpath), "foto udara.tif")

        # Reconstruct host path
        reconstructed = mapper.to_host(cpath)
        self.assertEqual(os.path.abspath(reconstructed), os.path.abspath(file_path))

    def test_nested_directories_reconstruction(self):
        mapper = PathMapper()
        base_dir = os.path.join(self.test_dir, "dataset")
        nested_file = os.path.join(base_dir, "subfolder", "deep", "test.tif")
        os.makedirs(os.path.dirname(nested_file), exist_ok=True)

        cpath = mapper.file_to_container(nested_file)
        host_rebuilt = mapper.to_host(cpath)
        self.assertEqual(os.path.abspath(host_rebuilt), os.path.abspath(nested_file))

    def test_translate_params_and_result(self):
        mapper = PathMapper()
        in_raster = os.path.join(self.test_dir, "raster.tif")
        out_shp = os.path.join(self.test_dir, "sub", "output.shp")

        params = {
            "raster_path": in_raster,
            "bbox_output_path": out_shp,
            "device": "cpu",
            "tile_size": 640,
        }

        cparams = mapper.translate_params(params)
        self.assertTrue(cparams["raster_path"].startswith("/mnt/"))
        self.assertTrue(cparams["bbox_output_path"].startswith("/mnt/"))
        self.assertEqual(cparams["device"], "cpu")

        # Subdirectory for output was created
        self.assertTrue(os.path.isdir(os.path.dirname(out_shp)))

        # Translate result layers
        result_from_container = {
            "total_detections": 42,
            "layers": [
                {"path": cparams["bbox_output_path"], "name": "output", "type": "bbox"}
            ],
            "manifest_path": cparams["bbox_output_path"] + ".manifest.json",
        }
        host_result = mapper.translate_result(result_from_container)
        self.assertEqual(os.path.abspath(host_result["layers"][0]["path"]), os.path.abspath(out_shp))
        self.assertEqual(
            os.path.abspath(host_result["manifest_path"]),
            os.path.abspath(out_shp + ".manifest.json"),
        )


if __name__ == "__main__":
    unittest.main()
