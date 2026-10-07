# -*- coding: utf-8 -*-
"""Download model weights on demand (stdlib only, runs on the host side)."""

import os
import urllib.request

# Upload the files from plugin_qgis/models/ to this GitHub Release (tag: models-v1).
MODEL_BASE_URL = "https://github.com/Anovvy/PalmAI_Plugin/releases/tag/models-v1"


def ensure_model(filename, models_dir, ctx=None):
    """Return the local path of a model, downloading it if it is missing."""
    path = os.path.join(models_dir, filename)
    if os.path.exists(path):
        return path

    os.makedirs(models_dir, exist_ok=True)
    url = MODEL_BASE_URL + filename
    tmp = path + ".part"
    if ctx:
        ctx.describe(f"Downloading model {filename}...")
        ctx.log(f"Model '{filename}' not found locally. Downloading from {url}", "warning")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while True:
                ctx and ctx.check_cancel()
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if ctx and total:
                    ctx.progress(int(5 * done / total))
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path
