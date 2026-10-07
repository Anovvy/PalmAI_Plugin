# -*- coding: utf-8 -*-
"""Legacy backend: run inside QGIS' own Python (requires manual pip install)."""

import importlib.util

from ..palmai_core.runner import run_job
from .base import InferenceBackend, BackendUnavailable

REQUIRED_MODULES = ("torch", "ultralytics", "sahi", "rasterio", "geopandas", "fiona", "shapely", "skimage")


class LocalBackend(InferenceBackend):
    name = "local"

    def missing_modules(self):
        return [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]

    def is_available(self):
        return not self.missing_modules()

    def run(self, job, params, ctx):
        missing = self.missing_modules()
        if missing:
            raise BackendUnavailable(
                "Missing Python libraries in QGIS: " + ", ".join(missing) +
                ". Use the Docker or Virtualenv backend (PalmAI > Settings) or install them manually.")
        return run_job(job, params, ctx)
