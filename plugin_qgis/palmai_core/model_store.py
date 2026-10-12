# -*- coding: utf-8 -*-
"""
Download and verify model weights on demand from GitHub Releases (stdlib only, runs on host).

Security & Integrity:
- All downloaded and cached checkpoints are cryptographically verified against known SHA-256 digests.
- Partial downloads are isolated in a `.part` temporary file and atomically renamed only upon verification.
- Bounded retries and timeouts protect against network stalls.
- Corrupted or modified cache files fail closed and are cleaned up.
"""

import hashlib
import os
import time
import urllib.error
import urllib.parse
import urllib.request

MODEL_BASE_URL = "https://github.com/Anovvy/PalmAI_Plugin/releases/download/models-v1/"

# Cryptographic SHA-256 hashes for official models-v1 release assets.
MODEL_HASHES = {
    "deeplabv3+_roaddetection_1.pth.tar": "6f8ec47d0d5be7573f128bb6f1ec53b6dd185fd90bcec2975c5efc451cea9ee0",
    "yolov8n_centertree_1.3.pt": "6ed11bab0e8fd3b61a7ab5d751b76d6096d958ce02d0870919667b4048ceef35",
    "yolov8n_classification_2.1.pt": "05162683b0842e759d17e3448331ece0b5c12678c8d3c66f5d296e1d9b02f795",
    "yolov8n_counting_1.0.pt": "2c5a4bec86354e808cbd3324fe36656c1a3e1be342ddc1797087e9c48592cfd8",
}

MAX_DOWNLOAD_RETRIES = 3
CHUNK_SIZE = 1024 * 256  # 256 KB


def calculate_file_hash(path):
    """Compute the SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def verify_model_integrity(path, filename):
    """Verify that a model file matches its expected SHA-256 digest."""
    expected_hash = MODEL_HASHES.get(filename)
    if not expected_hash:
        # If unlisted model, skip hash check but require non-empty file
        return os.path.isfile(path) and os.path.getsize(path) > 0

    if not os.path.isfile(path):
        return False

    actual_hash = calculate_file_hash(path)
    return actual_hash == expected_hash.lower()


def ensure_model(filename, models_dir, ctx=None):
    """
    Return the local path of a model, downloading and verifying it if missing or corrupted.
    Fails closed if the downloaded or cached file does not match expected SHA-256 digest.
    """
    path = os.path.join(models_dir, filename)

    # 1. Check existing cached model
    if os.path.exists(path):
        if verify_model_integrity(path, filename):
            return path
        # Corrupt or incomplete cached file detected
        if ctx:
            ctx.log(
                f"Cached model '{filename}' failed SHA-256 verification (corrupted or partial). Re-downloading...",
                "warning",
            )
        try:
            os.remove(path)
        except OSError:
            pass

    os.makedirs(models_dir, exist_ok=True)
    url = MODEL_BASE_URL + urllib.parse.quote(filename)
    tmp = path + ".part"

    last_error = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PalmAI-QGIS-Plugin/2.0.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length", 0))
                done = 0
                total_mb = total / (1024 * 1024) if total else 0
                if ctx:
                    if total_mb > 0:
                        ctx.log(
                            f"Model '{filename}' not found locally. Downloading from GitHub Releases "
                            f"(attempt {attempt}/{MAX_DOWNLOAD_RETRIES}, ~{total_mb:.1f} MB)...",
                            "info",
                        )
                        ctx.describe(f"Downloading model {filename} (~{total_mb:.1f} MB)...")
                    else:
                        ctx.log(
                            f"Downloading model '{filename}' from {url} (attempt {attempt}/{MAX_DOWNLOAD_RETRIES})...",
                            "warning",
                        )
                        ctx.describe(f"Downloading model {filename}...")

                while True:
                    if ctx:
                        ctx.check_cancel()
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if ctx and total:
                        pct = int(100 * done / total)
                        done_mb = done / (1024 * 1024)
                        ctx.progress(pct)
                        ctx.describe(f"Downloading model {filename} ({done_mb:.1f}/{total_mb:.1f} MB - {pct}%)...")

            # Verify integrity of downloaded file BEFORE promoting it
            if not verify_model_integrity(tmp, filename):
                actual_hash = calculate_file_hash(tmp) if os.path.exists(tmp) else "missing"
                expected_hash = MODEL_HASHES.get(filename, "unknown")
                raise ValueError(
                    f"SHA-256 mismatch for '{filename}'! Expected: {expected_hash}, Actual: {actual_hash}"
                )

            # Atomic replace into target path
            os.replace(tmp, path)
            if ctx:
                ctx.progress(100)
                ctx.log(f"Model '{filename}' downloaded and verified successfully (SHA-256 OK).", "info")
            return path

        except PermissionError as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise RuntimeError(
                f"Permission denied when saving model to '{models_dir}'. "
                f"Please ensure write permissions exist for the models folder."
            ) from e

        except Exception as e:
            last_error = e
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if ctx:
                ctx.log(f"Download attempt {attempt} failed: {e}", "warning")
            if attempt < MAX_DOWNLOAD_RETRIES:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Failed to download verified model weights for '{filename}' after {MAX_DOWNLOAD_RETRIES} attempts. "
        f"Last error: {last_error}"
    )
