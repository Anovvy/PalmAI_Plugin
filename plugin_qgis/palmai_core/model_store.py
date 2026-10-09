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
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PalmAI-QGIS-Plugin/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            total_mb = total / (1024 * 1024) if total else 0
            if ctx:
                if total_mb > 0:
                    ctx.log(f"Model '{filename}' not found locally. Downloading from GitHub Releases (~{total_mb:.1f} MB)...", "info")
                    ctx.describe(f"Downloading model {filename} (~{total_mb:.1f} MB)...")
                else:
                    ctx.log(f"Model '{filename}' not found locally. Downloading from {url}", "warning")
                    ctx.describe(f"Downloading model {filename}...")

            while True:
                if ctx:
                    ctx.check_cancel()
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if ctx and total:
                    pct = int(100 * done / total)
                    done_mb = done / (1024 * 1024)
                    ctx.progress(pct)
                    ctx.describe(f"Downloading model {filename} ({done_mb:.1f}/{total_mb:.1f} MB - {pct}%)...")

        os.replace(tmp, path)
        if ctx:
            ctx.progress(100)
            ctx.log(f"Model '{filename}' downloaded successfully.", "info")
    except PermissionError as e:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise RuntimeError(
            f"Permission denied when saving model to '{models_dir}'. "
            f"If PalmAI is installed in Program Files, please run QGIS as Administrator for the first download, "
            f"or download the model weights manually from GitHub Releases into the 'models' folder.") from e
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    return path

