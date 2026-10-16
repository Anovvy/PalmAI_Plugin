# -*- coding: utf-8 -*-
"""Tests for VenvBackend (backends/venv_backend.py)."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from plugin_qgis.backends.venv_backend import (
    READY_MARKER,
    VenvBackend,
    _is_qgis_python,
)


class TestVenvBackend(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="palmai_test_venv_")
        self.plugin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plugin_qgis"))
        self.env_dir = os.path.join(self.tmp, "env")
        self.backend = VenvBackend(self.plugin_dir, self.env_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_qgis_python_exclusion(self):
        self.assertTrue(_is_qgis_python("C:\\OSGeo4W\\bin\\python3.exe"))
        self.assertTrue(_is_qgis_python("/usr/lib/qgis/python3"))
        self.assertFalse(_is_qgis_python("C:\\Python311\\python.exe"))
        self.assertFalse(_is_qgis_python("/usr/bin/python3.11"))

    def test_torch_index_args_by_platform(self):
        with patch("sys.platform", "darwin"):
            args = self.backend._torch_index_args()
            self.assertEqual(args, [])

        with patch("sys.platform", "linux"):
            with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
                args = self.backend._torch_index_args()
                self.assertIn("cu121", args[1])

            with patch("shutil.which", return_value=None):
                args = self.backend._torch_index_args()
                self.assertIn("cpu", args[1])

    def test_is_ready_marker_verification(self):
        # When environment directory is empty
        self.assertFalse(self.backend.is_ready())

        # Simulate presence of python_exe and ready marker
        os.makedirs(os.path.dirname(self.backend.python_exe), exist_ok=True)
        with open(self.backend.python_exe, "w") as f:
            f.write("#!dummy")

        marker_file = os.path.join(self.env_dir, READY_MARKER)
        with open(marker_file, "w", encoding="utf-8") as f:
            f.write("invalid_signature")

        # Wrong signature -> False
        self.assertFalse(self.backend.is_ready())

        # Correct signature -> True
        with patch.object(self.backend, "_lock_signature", return_value="valid_sig|torch==2.5.1"):
            with open(marker_file, "w", encoding="utf-8") as f:
                f.write("valid_sig|torch==2.5.1")
            self.assertTrue(self.backend.is_ready())


if __name__ == "__main__":
    unittest.main()
