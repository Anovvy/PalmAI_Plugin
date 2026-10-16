# -*- coding: utf-8 -*-
"""Tests for QGIS plugin metadata compliance (plugin_qgis/metadata.txt)."""

import configparser
import os
import unittest


class TestPluginMetadata(unittest.TestCase):
    def setUp(self):
        self.plugin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plugin_qgis"))
        self.meta_path = os.path.join(self.plugin_dir, "metadata.txt")

    def test_metadata_file_exists(self):
        self.assertTrue(os.path.isfile(self.meta_path), f"metadata.txt missing at {self.meta_path}")

    def test_metadata_general_section_and_mandatory_keys(self):
        config = configparser.ConfigParser()
        config.read(self.meta_path, encoding="utf-8")
        self.assertIn("general", config, "metadata.txt must contain a [general] section")

        general = config["general"]
        mandatory = [
            "name",
            "qgisMinimumVersion",
            "description",
            "version",
            "author",
            "email",
            "about",
            "tracker",
            "repository",
        ]
        for key in mandatory:
            self.assertIn(key, general, f"Mandatory metadata key '{key}' missing")
            self.assertTrue(len(general[key].strip()) > 0, f"Metadata key '{key}' cannot be empty")

    def test_no_placeholder_urls(self):
        config = configparser.ConfigParser()
        config.read(self.meta_path, encoding="utf-8")
        general = config["general"]

        for key in ["tracker", "repository", "homepage"]:
            if key in general:
                val = general[key]
                self.assertNotIn("http://bugs", val, f"Placeholder detected in {key}")
                self.assertNotIn("http://repo", val, f"Placeholder detected in {key}")
                self.assertNotIn("http://homepage", val, f"Placeholder detected in {key}")
                self.assertTrue(val.startswith("https://") or val.startswith("http://"))

    def test_qgis_versions(self):
        config = configparser.ConfigParser()
        config.read(self.meta_path, encoding="utf-8")
        general = config["general"]

        min_ver = float(general["qgisMinimumVersion"])
        max_ver = float(general["qgisMaximumVersion"])

        self.assertGreaterEqual(min_ver, 3.0, "Plugin must target QGIS 3.x minimum")
        self.assertLessEqual(max_ver, 3.99, "Plugin must cap at QGIS 3.99 (QGIS 4 not yet supported)")

    def test_icon_exists(self):
        config = configparser.ConfigParser()
        config.read(self.meta_path, encoding="utf-8")
        general = config["general"]

        if "icon" in general:
            icon_rel = general["icon"]
            icon_path = os.path.join(self.plugin_dir, icon_rel)
            self.assertTrue(os.path.isfile(icon_path), f"Icon '{icon_rel}' does not exist at {icon_path}")


if __name__ == "__main__":
    unittest.main()
