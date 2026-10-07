# -*- coding: utf-8 -*-
"""Download model weights on demand from GitHub Releases (stdlib only, runs on host)."""

import os
import urllib.parse
import urllib.request

MODEL_BASE_URL = "https://github.com/Anovvy/PalmAI_Plugin/releases/download/models-v1/"


def ensure_model(filename, models_dir, ctx=None):
    """Return the local path of a model, downloading it if it is missing."""
    path = os.path.join(models_dir, filename)
    if os.path.exists(path):
        return path

    os.makedirs(models_dir, exist_ok=True)
    url = MODEL_BASE_URL + urllib.parse.quote(filename)
    tmp = path + ".part"
    if ctx:
        ctx.describe(f"Downloading model {filename}...")
        ctx.log(f"Model '{filename}' not found locally. Downloading from {url}", "warning")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PalmAI-QGIS-Plugin/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while True:
                if ctx:
                    ctx.check_cancel()
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
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    return path
