# -*- coding: utf-8 -*-
"""
Host <-> container path translation and mount mapping for the Docker backend.

Security & Isolation:
- Least-privilege volume mounts: input-only directories are mounted read-only (:ro).
- Output and temporary working directories are mounted read-write (:rw).
- Directories containing both inputs and outputs are automatically elevated to (:rw).
- Preserves full cross-platform compatibility (Windows drives/UNC, macOS, Linux).
"""

import os

INPUT_FILE_KEYS = ("raster_path", "extent_path")
OUTPUT_FILE_KEYS = (
    "bbox_output_path",
    "centroid_output_path",
    "output_raster_path",
    "output_vector_path",
)
PATH_FILE_KEYS = INPUT_FILE_KEYS + OUTPUT_FILE_KEYS
PATH_DIR_KEYS = ("temp_dir",)


class PathMapper:
    """Maps host directories to numbered container mount points (/mnt/0, /mnt/1, etc.)."""

    def __init__(self):
        # host_norm_key -> {"host_dir": str, "container_dir": str, "writable": bool}
        self._mount_map = {}

    def _normalize_key(self, path_str):
        # Normalize slashes and case (important for Windows drive letter case-insensitivity)
        norm = os.path.normcase(os.path.abspath(path_str))
        return norm.rstrip("\\/")

    def _register_dir(self, host_dir, writable=False):
        host_dir = os.path.abspath(host_dir)
        key = self._normalize_key(host_dir)
        if key not in self._mount_map:
            cdir = f"/mnt/{len(self._mount_map)}"
            self._mount_map[key] = {
                "host_dir": host_dir,
                "container_dir": cdir,
                "writable": writable,
            }
        else:
            if writable:
                self._mount_map[key]["writable"] = True
        return self._mount_map[key]["container_dir"]

    def mounts(self):
        """Returns list of (host_dir, container_dir) tuples for backward compatibility."""
        return [(m["host_dir"], m["container_dir"]) for m in self._mount_map.values()]

    def mount_specs(self):
        """Returns list of (host_dir, container_dir, mode) with least-privilege ro/rw."""
        return [
            (m["host_dir"], m["container_dir"], "rw" if m["writable"] else "ro")
            for m in self._mount_map.values()
        ]

    def file_to_container(self, host_path, writable=False):
        host_path = os.path.abspath(host_path)
        parent_dir = os.path.dirname(host_path)
        cdir = self._register_dir(parent_dir, writable=writable)
        return f"{cdir}/{os.path.basename(host_path)}"

    def dir_to_container(self, host_dir, writable=True):
        return self._register_dir(host_dir, writable=writable)

    def to_host(self, container_path):
        """Reconstruct the original host path from a container path."""
        norm_cpath = container_path.replace("\\", "/")
        for m in self._mount_map.values():
            cdir = m["container_dir"]
            if norm_cpath == cdir:
                return m["host_dir"]
            if norm_cpath.startswith(cdir + "/"):
                rel_parts = norm_cpath[len(cdir) + 1:].split("/")
                return os.path.join(m["host_dir"], *rel_parts)
        return container_path

    def translate_params(self, params):
        """
        Translate all file and directory paths in params to container paths.
        Creates output and temp parent directories on host before execution.
        """
        out = dict(params)
        for k in PATH_FILE_KEYS:
            if out.get(k):
                is_output = k in OUTPUT_FILE_KEYS
                abs_val = os.path.abspath(str(out[k]))
                if is_output:
                    os.makedirs(os.path.dirname(abs_val), exist_ok=True)
                out[k] = self.file_to_container(abs_val, writable=is_output)

        for k in PATH_DIR_KEYS:
            if out.get(k):
                abs_dir = os.path.abspath(str(out[k]))
                os.makedirs(abs_dir, exist_ok=True)
                out[k] = self.dir_to_container(abs_dir, writable=True)

        return out

    def translate_result(self, result):
        """Reconstruct host paths for all output layer artifacts in worker result."""
        res = dict(result)
        if "layers" in res and isinstance(res["layers"], list):
            translated_layers = []
            for l in res["layers"]:
                layer_copy = dict(l)
                if "path" in layer_copy:
                    layer_copy["path"] = self.to_host(layer_copy["path"])
                translated_layers.append(layer_copy)
            res["layers"] = translated_layers
        if "manifest_path" in res and res["manifest_path"]:
            res["manifest_path"] = self.to_host(res["manifest_path"])
        return res
