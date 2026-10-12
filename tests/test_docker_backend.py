# -*- coding: utf-8 -*-
"""Tests for DockerBackend (backends/docker_backend.py)."""

import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from plugin_qgis.backends.docker_backend import DockerBackend


class TestDockerBackend(unittest.TestCase):
    def setUp(self):
        self.plugin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plugin_qgis"))
        self.backend = DockerBackend(self.plugin_dir)

    def test_diagnose_when_docker_binary_missing(self):
        self.backend.docker = None
        diag = self.backend.diagnose()
        self.assertIsNotNone(diag)
        self.assertIn("not installed", diag)
        self.assertFalse(self.backend.is_available())

    @patch("plugin_qgis.backends.docker_backend._run")
    def test_diagnose_when_daemon_stopped(self, mock_run):
        self.backend.docker = "/usr/bin/docker"
        mock_run.return_value = subprocess.CompletedProcess(
            args=["docker", "info"], returncode=1, stdout="", stderr="daemon down"
        )
        diag = self.backend.diagnose()
        self.assertIsNotNone(diag)
        self.assertIn("daemon is not running", diag)

    @patch("plugin_qgis.backends.docker_backend._run")
    def test_has_nvidia_runtime_detection(self, mock_run):
        self.backend.docker = "/usr/bin/docker"
        # Test daemon without nvidia
        mock_run.return_value = subprocess.CompletedProcess(
            args=["docker", "info"], returncode=0, stdout='{"runc": {}}', stderr=""
        )
        self.assertFalse(self.backend.has_nvidia_runtime())

        # Test daemon with nvidia runtime and mocked smi
        mock_run.return_value = subprocess.CompletedProcess(
            args=["docker", "info"], returncode=0, stdout='{"nvidia": {}}', stderr=""
        )
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("sys.platform", "linux"):
                self.assertTrue(self.backend.has_nvidia_runtime())

    @patch.object(DockerBackend, "image_present", return_value=True)
    def test_ensure_image_when_locally_present(self, mock_present):
        self.backend.docker = "/usr/bin/docker"
        ctx = MagicMock()
        img = self.backend.ensure_image(gpu=False, ctx=ctx)
        self.assertIn("cpu", img)
        ctx.describe.assert_not_called()

    @patch.object(DockerBackend, "image_present", return_value=False)
    @patch("plugin_qgis.backends.docker_backend.run_command_streaming", return_value=0)
    def test_ensure_image_pulls_when_missing(self, mock_streaming, mock_present):
        self.backend.docker = "/usr/bin/docker"
        ctx = MagicMock()
        img = self.backend.ensure_image(gpu=False, ctx=ctx)
        self.assertIn("cpu", img)
        mock_streaming.assert_called_once()
        ctx.log.assert_called()

    @patch.object(DockerBackend, "image_present", return_value=False)
    @patch("plugin_qgis.backends.docker_backend.run_command_streaming", side_effect=[1, 0])
    @patch("os.path.exists", return_value=True)
    def test_ensure_image_fallback_to_local_build(self, mock_exists, mock_streaming, mock_present):
        self.backend.docker = "/usr/bin/docker"
        ctx = MagicMock()
        img = self.backend.ensure_image(gpu=False, ctx=ctx)
        self.assertEqual(mock_streaming.call_count, 2)
        self.assertIn("cpu", img)

    @patch.object(DockerBackend, "diagnose", return_value=None)
    @patch.object(DockerBackend, "ensure_image", return_value="palmai-core:v2.0.0-cpu")
    @patch.object(DockerBackend, "has_nvidia_runtime", return_value=False)
    @patch("plugin_qgis.backends.docker_backend.run_worker")
    def test_run_mounts_and_cpu_fallback(self, mock_run_worker, mock_gpu, mock_ensure, mock_diag):
        self.backend.docker = "/usr/bin/docker"
        ctx = MagicMock()
        mock_run_worker.return_value = {"layers": [], "total_detections": 10}

        with tempfile.TemporaryDirectory() as tmpdir:
            raster = os.path.join(tmpdir, "in.tif")
            bbox = os.path.join(tmpdir, "out.shp")
            with open(raster, "w") as f:
                f.write("dummy")

            params = {
                "raster_path": raster,
                "bbox_output_path": bbox,
                "centroid_output_path": bbox,
                "weight_path": "model.pt",
                "device": "cuda",  # Should fall back to CPU because mock_gpu is False
            }

            result = self.backend.run("counting", params, ctx)
            self.assertEqual(result["total_detections"], 10)

            # Check executed docker command passed to run_worker
            cmd_executed = mock_run_worker.call_args[0][0]
            self.assertEqual(cmd_executed[0], "/usr/bin/docker")
            self.assertIn("run", cmd_executed)
            self.assertIn("--rm", cmd_executed)
            # Verify /models:ro and /job:ro are present
            self.assertTrue(any("/models:ro" in arg for arg in cmd_executed))
            self.assertTrue(any("/job:ro" in arg for arg in cmd_executed))
            # Verify no --gpus flag due to CPU fallback
            self.assertNotIn("--gpus", cmd_executed)


if __name__ == "__main__":
    unittest.main()
