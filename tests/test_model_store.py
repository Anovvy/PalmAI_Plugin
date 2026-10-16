# -*- coding: utf-8 -*-
"""Tests for model_store.py (cryptographic integrity, retries, atomic download)."""

import hashlib
import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from plugin_qgis.palmai_core.model_store import (
    MODEL_HASHES,
    calculate_file_hash,
    ensure_model,
    verify_model_integrity,
)


class FakeHTTPResponse:
    def __init__(self, data, status=200):
        self.data = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data))}
        self.status = status

    def read(self, amt=None):
        return self.data.read(amt)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestModelStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="palmai_test_store_")
        self.test_filename = "yolov8n_counting_1.0.pt"
        self.valid_hash = MODEL_HASHES[self.test_filename]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_calculate_file_hash(self):
        dummy_file = os.path.join(self.tmp, "dummy.bin")
        content = b"PalmAI weights verification test"
        expected = hashlib.sha256(content).hexdigest()

        with open(dummy_file, "wb") as f:
            f.write(content)

        actual = calculate_file_hash(dummy_file)
        self.assertEqual(actual, expected)

    def test_verify_model_integrity(self):
        dummy_file = os.path.join(self.tmp, self.test_filename)
        # Content that doesn't match official hash
        with open(dummy_file, "wb") as f:
            f.write(b"corrupt model weights")

        self.assertFalse(verify_model_integrity(dummy_file, self.test_filename))

        # Test non-existent file
        self.assertFalse(verify_model_integrity(os.path.join(self.tmp, "missing.pt"), self.test_filename))

    @patch("plugin_qgis.palmai_core.model_store.verify_model_integrity", return_value=True)
    def test_ensure_model_reuses_valid_cache(self, mock_verify):
        path = os.path.join(self.tmp, self.test_filename)
        with open(path, "wb") as f:
            f.write(b"valid cached data")

        ctx = MagicMock()
        with patch("urllib.request.urlopen") as mock_urlopen:
            res_path = ensure_model(self.test_filename, self.tmp, ctx)
            self.assertEqual(res_path, path)
            mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_ensure_model_downloads_and_verifies_successfully(self, mock_urlopen):
        fake_payload = b"official valid test payload"
        fake_hash = hashlib.sha256(fake_payload).hexdigest()

        mock_urlopen.return_value = FakeHTTPResponse(fake_payload)
        ctx = MagicMock()

        # Temporarily register custom hash for testing
        with patch.dict(MODEL_HASHES, {self.test_filename: fake_hash}):
            res_path = ensure_model(self.test_filename, self.tmp, ctx)
            self.assertTrue(os.path.isfile(res_path))
            with open(res_path, "rb") as f:
                self.assertEqual(f.read(), fake_payload)

            # Ensure .part file was removed upon atomic replacement
            self.assertFalse(os.path.exists(res_path + ".part"))

    @patch("urllib.request.urlopen")
    def test_ensure_model_fails_closed_on_checksum_mismatch(self, mock_urlopen):
        fake_corrupted_payload = b"compromised or damaged payload"
        mock_urlopen.return_value = FakeHTTPResponse(fake_corrupted_payload)
        ctx = MagicMock()

        with self.assertRaises(RuntimeError) as cm:
            ensure_model(self.test_filename, self.tmp, ctx)

        self.assertIn("Failed to download verified model weights", str(cm.exception))
        # Ensure neither target file nor .part file remains
        target_path = os.path.join(self.tmp, self.test_filename)
        self.assertFalse(os.path.exists(target_path))
        self.assertFalse(os.path.exists(target_path + ".part"))


if __name__ == "__main__":
    unittest.main()
