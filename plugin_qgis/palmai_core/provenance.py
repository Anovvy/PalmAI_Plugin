# -*- coding: utf-8 -*-
"""
Execution provenance and run manifest generator.

Emits structured run_manifest.json records per job to ensure:
- Model and runtime traceability (PyTorch/GDAL versions, device used).
- Input spatial context recording (CRS, GSD/resolution, dimensions).
- Output artifact integrity verification (hashes, feature counts).
- Privacy safety: Sanitizes paths to prevent leaking host usernames and local directory trees.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

PLUGIN_VERSION = "2.0.0"
PROTOCOL_VERSION = "1.0"


def sanitize_path(path_str):
    """Return only the basename of a path to protect user filesystem privacy."""
    if not path_str:
        return ""
    return os.path.basename(str(path_str))


def compute_file_sha256(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def collect_runtime_metadata():
    info = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    for mod_name in ("torch", "torchvision", "rasterio", "geopandas", "shapely", "ultralytics"):
        try:
            mod = __import__(mod_name)
            info[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[mod_name] = "not_loaded"
    return info


def build_run_manifest(job_type, params, results, input_meta=None, elapsed_seconds=None):
    """
    Construct a structured run manifest dictionary.
    """
    runtime = collect_runtime_metadata()

    # Sanitize parameter dictionary for privacy
    safe_params = {}
    for k, v in params.items():
        if "path" in k.lower() or "dir" in k.lower():
            safe_params[k] = sanitize_path(v)
        else:
            safe_params[k] = v

    # Collect output summaries and file checksums
    outputs = []
    if "layers" in results and isinstance(results["layers"], list):
        for layer in results["layers"]:
            lp = layer.get("path")
            outputs.append({
                "name": layer.get("name", sanitize_path(lp)),
                "type": layer.get("type", "unknown"),
                "filename": sanitize_path(lp),
                "sha256": compute_file_sha256(lp) if lp else None,
                "size_bytes": os.path.getsize(lp) if (lp and os.path.isfile(lp)) else 0,
            })

    manifest = {
        "palmai_plugin_version": PLUGIN_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "job_type": job_type,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 2) if elapsed_seconds is not None else None,
        "runtime_environment": runtime,
        "device": params.get("device", "cpu"),
        "input_metadata": input_meta or {},
        "parameters": safe_params,
        "outputs": outputs,
        "summary": {
            "total_detections": results.get("total_detections"),
            "output_layers_count": len(outputs),
        },
    }
    return manifest


def write_run_manifest(target_dir, manifest_dict):
    """Write run_manifest.json to the given target directory and return full path."""
    os.makedirs(target_dir, exist_ok=True)
    manifest_path = os.path.join(target_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)
    return manifest_path
