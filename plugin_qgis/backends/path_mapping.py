# -*- coding: utf-8 -*-
"""Host <-> container path translation for the Docker backend."""

import os

PATH_FILE_KEYS = ("raster_path", "extent_path", "bbox_output_path", "centroid_output_path",
                  "output_raster_path", "output_vector_path")
OUTPUT_FILE_KEYS = ("bbox_output_path", "centroid_output_path",
                    "output_raster_path", "output_vector_path")
PATH_DIR_KEYS = ("temp_dir",)


class PathMapper:
    """Mounts the *parent directory* of every file (or the directory itself) under /mnt/N."""

    def __init__(self):
        self._mounts = {}  # normalised host dir -> (host dir, container dir)

    def _mount(self, host_dir):
        host_dir = os.path.abspath(host_dir)
        key = os.path.normcase(host_dir)
        if key not in self._mounts:
            self._mounts[key] = (host_dir, f"/mnt/{len(self._mounts)}")
        return self._mounts[key][1]

    def mounts(self):
        return list(self._mounts.values())

    def file_to_container(self, host_path):
        host_path = os.path.abspath(host_path)
        return self._mount(os.path.dirname(host_path)) + "/" + os.path.basename(host_path)

    def dir_to_container(self, host_dir):
        return self._mount(host_dir)

    def to_host(self, container_path):
        for host_dir, cdir in self._mounts.values():
            if container_path == cdir:
                return host_dir
            if container_path.startswith(cdir + "/"):
                rest = container_path[len(cdir) + 1:].split("/")
                return os.path.join(host_dir, *rest)
        return container_path

    def translate_params(self, params):
        out = dict(params)
        for k in PATH_FILE_KEYS:
            if out.get(k):
                if k in OUTPUT_FILE_KEYS:
                    os.makedirs(os.path.dirname(os.path.abspath(out[k])), exist_ok=True)
                out[k] = self.file_to_container(out[k])
        for k in PATH_DIR_KEYS:
            if out.get(k):
                os.makedirs(out[k], exist_ok=True)
                out[k] = self.dir_to_container(out[k])
        return out

    def translate_result(self, result):
        res = dict(result)
        if "layers" in res:
            res["layers"] = [dict(l, path=self.to_host(l["path"])) for l in res["layers"]]
        return res
