# -*- coding: utf-8 -*-
"""Virtualenv backend: isolated Python env outside QGIS (works with CUDA and Apple MPS)."""

import json
import os
import shutil
import subprocess
import sys
import tempfile

from .base import InferenceBackend, BackendError, BackendUnavailable
from .process_runner import run_worker, run_command_streaming, _popen_flags

TORCH_PINS = ["torch==2.5.1", "torchvision==0.20.1"]
SUPPORTED = ((3, 9), (3, 12))  # inclusive range of base interpreter versions
READY_MARKER = ".palmai_ready"


def _py_version(exe):
    try:
        r = subprocess.run([exe, "-c", "import sys;print('%d.%d'%sys.version_info[:2])"],
                           capture_output=True, text=True, timeout=20, **_popen_flags())
        if r.returncode == 0:
            major, minor = r.stdout.strip().split(".")
            return int(major), int(minor)
    except Exception:
        pass
    return None


def find_base_python():
    """Find a real Python interpreter (NOT the QGIS executable) in the supported range."""
    candidates = []
    if sys.platform == "win32":
        candidates += [os.path.join(sys.exec_prefix, "python.exe"),
                       os.path.join(sys.exec_prefix, "python3.exe")]
    else:
        candidates += [os.path.join(sys.exec_prefix, "bin", "python3")]
    for name in ("python3.12", "python3.11", "python3.10", "python3.9", "python3", "python"):
        p = shutil.which(name)
        if p:
            candidates.append(p)
    if sys.platform == "win32":
        py = shutil.which("py")
        if py:
            for v in ("3.12", "3.11", "3.10", "3.9"):
                try:
                    r = subprocess.run([py, f"-{v}", "-c", "import sys;print(sys.executable)"],
                                       capture_output=True, text=True, timeout=20, **_popen_flags())
                    if r.returncode == 0 and r.stdout.strip():
                        candidates.append(r.stdout.strip())
                except Exception:
                    pass
    for exe in candidates:
        if exe and os.path.exists(exe):
            ver = _py_version(exe)
            if ver and SUPPORTED[0] <= ver <= SUPPORTED[1]:
                return exe
    return None


class VenvBackend(InferenceBackend):
    name = "venv"

    def __init__(self, plugin_dir, env_dir):
        self.plugin_dir = plugin_dir
        self.env_dir = env_dir

    @property
    def python_exe(self):
        sub = "Scripts" if sys.platform == "win32" else "bin"
        return os.path.join(self.env_dir, sub, "python.exe" if sys.platform == "win32" else "python")

    def lock_file(self):
        return os.path.join(self.plugin_dir, "docker", "requirements.lock")

    def _lock_signature(self):
        with open(self.lock_file(), "rb") as f:
            return f.read().decode("utf-8", "ignore") + "|" + ",".join(TORCH_PINS)

    def is_ready(self):
        marker = os.path.join(self.env_dir, READY_MARKER)
        if not (os.path.exists(self.python_exe) and os.path.exists(marker)):
            return False
        try:
            with open(marker, "r", encoding="utf-8") as f:
                return f.read() == self._lock_signature()
        except Exception:
            return False

    def is_available(self):
        return self.is_ready()

    # --- one-time setup -----------------------------------------------------
    def _torch_index_args(self):
        if sys.platform == "darwin":
            return []  # default PyPI wheels include MPS support
        if shutil.which("nvidia-smi"):
            return ["--index-url", "https://download.pytorch.org/whl/cu121"]
        return ["--index-url", "https://download.pytorch.org/whl/cpu"]

    def setup(self, ctx):
        base = find_base_python()
        if not base:
            raise BackendUnavailable(
                "No suitable Python (3.9-3.12) found. Install Python 3.11 from python.org, "
                "or use the Docker backend.")

        def relay(line):
            if line.strip():
                ctx.describe(line.strip()[:120])
                ctx.log(line.strip(), "info")

        def step(cmd, msg, progress):
            ctx.progress(progress)
            ctx.describe(msg)
            ctx.log(msg)
            if run_command_streaming(cmd, ctx, on_line=relay) != 0:
                raise BackendError(f"Setup failed during: {msg}. See the QGIS log for details.")

        if os.path.isdir(self.env_dir):
            shutil.rmtree(self.env_dir, ignore_errors=True)
        os.makedirs(os.path.dirname(self.env_dir), exist_ok=True)

        step([base, "-m", "venv", self.env_dir], "Creating virtual environment...", 3)
        py = self.python_exe
        step([py, "-m", "pip", "install", "--upgrade", "pip", "wheel"], "Upgrading pip...", 8)
        step([py, "-m", "pip", "install", *TORCH_PINS, *self._torch_index_args()],
             "Installing PyTorch (large download)...", 15)
        step([py, "-m", "pip", "install", "-r", self.lock_file()], "Installing geospatial and AI libraries...", 70)

        ctx.describe("Verifying installation...")
        env = dict(os.environ, PYTHONPATH=self.plugin_dir)
        if run_command_streaming([py, "-m", "palmai_core.worker", "--check"], ctx,
                                 cwd=self.plugin_dir, env=env, on_line=relay) != 0:
            raise BackendError("Environment verification failed. See the QGIS log for details.")

        with open(os.path.join(self.env_dir, READY_MARKER), "w", encoding="utf-8") as f:
            f.write(self._lock_signature())
        ctx.progress(100)

    # --- job execution ------------------------------------------------------
    def run(self, job, params, ctx):
        if not self.is_ready():
            raise BackendUnavailable(
                "The Python environment is not set up yet. Open PalmAI > Settings and click "
                "'Set up Python environment'.")
        fd, job_file = tempfile.mkstemp(prefix="palmai_job_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"job": job, "params": params}, f)
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=self.plugin_dir)
            return run_worker([self.python_exe, "-m", "palmai_core.worker", job_file], ctx,
                              cwd=self.plugin_dir, env=env)
        finally:
            try:
                os.remove(job_file)
            except OSError:
                pass
