#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package the PalmAI QGIS plugin into a release-ready, installable ZIP archive.

Ensures:
1. Root directory in the ZIP is 'PalmAI/'
2. Mandatory metadata.txt and entry points are validated.
3. Heavyweight model checkpoints (~500 MB) and development caches are excluded.
4. SHA-256 checksum is computed and displayed for release provenance.
"""

import argparse
import configparser
import hashlib
import os
import sys
import zipfile

REQUIRED_ENTRIES = [
    "metadata.txt",
    "__init__.py",
    "palmai_plugin.py",
]

EXCLUDED_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".part", ".pt", ".pth.tar", ".tar", ".gz"
}

EXCLUDED_DIR_NAMES = {
    "__pycache__", ".git", ".github", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "env", "tests", "dist", "build"
}


def validate_metadata(plugin_dir):
    meta_path = os.path.join(plugin_dir, "metadata.txt")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing metadata.txt at {meta_path}")

    config = configparser.ConfigParser()
    config.read(meta_path, encoding="utf-8")
    if "general" not in config:
        raise ValueError("metadata.txt missing [general] section")

    general = config["general"]
    required_keys = ["name", "qgisMinimumVersion", "description", "version", "author", "email", "about"]
    for k in required_keys:
        if k not in general or not general[k].strip():
            raise ValueError(f"metadata.txt missing or empty key: {k}")

    # Check for placeholder URLs
    for url_key in ["tracker", "repository", "homepage"]:
        val = general.get(url_key, "")
        if "http://bugs" in val or "http://repo" in val or "http://homepage" in val:
            raise ValueError(f"Placeholder URL detected in metadata.txt: {url_key}={val}")

    return general["name"], general["version"]


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def package_plugin(source_dir, dist_dir, zip_name="PalmAI.zip"):
    source_dir = os.path.abspath(source_dir)
    dist_dir = os.path.abspath(dist_dir)
    os.makedirs(dist_dir, exist_ok=True)

    name, version = validate_metadata(source_dir)
    print(f"Validating plugin package: {name} v{version}")

    for entry in REQUIRED_ENTRIES:
        target = os.path.join(source_dir, entry)
        if not os.path.exists(target):
            raise FileNotFoundError(f"Required plugin entry missing: {entry}")

    zip_path = os.path.join(dist_dir, zip_name)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    total_files = 0
    total_uncompressed_bytes = 0

    print(f"Packaging files into {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                # Skip excluded extensions
                if ext in EXCLUDED_EXTENSIONS:
                    continue
                if file.startswith(".") or file.endswith("~"):
                    continue

                abs_file = os.path.join(root, file)
                rel_file = os.path.relpath(abs_file, source_dir)

                # Store under top-level PalmAI/ directory inside zip
                arcname = os.path.join("PalmAI", rel_file).replace("\\", "/")
                zf.write(abs_file, arcname)
                total_files += 1
                total_uncompressed_bytes += os.path.getsize(abs_file)

    # Verify archive integrity
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if "PalmAI/metadata.txt" not in names:
            raise RuntimeError("Verification failed: PalmAI/metadata.txt missing in archive")
        if "PalmAI/__init__.py" not in names:
            raise RuntimeError("Verification failed: PalmAI/__init__.py missing in archive")
        # Ensure no heavy model files leaked into zip
        for n in names:
            for bad_ext in [".pt", ".pth.tar"]:
                if n.endswith(bad_ext):
                    raise RuntimeError(f"Heavyweight model weight leaked into archive: {n}")

    zip_size = os.path.getsize(zip_path)
    checksum = compute_sha256(zip_path)

    print("\n--- Packaging Successful ---")
    print(f"Archive:       {zip_path}")
    print(f"Files:         {total_files}")
    print(f"Uncompressed:  {total_uncompressed_bytes / (1024 * 1024):.2f} MB")
    print(f"Archive Size:  {zip_size / (1024 * 1024):.2f} MB")
    print(f"SHA-256:       {checksum}")
    print("----------------------------\n")

    return zip_path, checksum


def main():
    parser = argparse.ArgumentParser(description="Package PalmAI QGIS Plugin into installable ZIP")
    parser.add_argument("--src-dir", default="plugin_qgis", help="Path to plugin_qgis directory")
    parser.add_argument("--dist-dir", default="dist", help="Target distribution directory")
    parser.add_argument("--zip-name", default="PalmAI.zip", help="Output ZIP file name")
    args = parser.parse_args()

    try:
        package_plugin(args.src_dir, args.dist_dir, args.zip_name)
    except Exception as e:
        print(f"Error packaging plugin: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
