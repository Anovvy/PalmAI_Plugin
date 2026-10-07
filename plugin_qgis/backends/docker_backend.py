# -*- coding: utf-8 -*-
"""Docker backend: one ephemeral `docker run --rm` container per job (no server, no ports)."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

from .base import InferenceBackend, BackendError, BackendUnavailable
from .path_mapping import PathMapper
from .process_runner import run_worker, run_command_streaming, _popen_flags

IMAGE_REPO = "ghcr.io/anovvy/palmai-core"
WINDOWS_DOCKER_FALLBACK = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"


def find_docker():
    exe = shutil.which("docker")
    if exe:
        return exe
    if sys.platform == "win32" and os.path.exists(WINDOWS_DOCKER_FALLBACK):
        return WINDOWS_DOCKER_FALLBACK
    return None


def _run(cmd, timeout=20):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, **_popen_flags())


class DockerBackend(InferenceBackend):
    name = "docker"

    def __init__(self, plugin_dir):
        self.plugin_dir = plugin_dir
        self.docker = find_docker()

    # --- environment checks -------------------------------------------------
    def is_available(self):
        return self.diagnose() is None

    def diagnose(self):
        """Return None if Docker is usable, else a human-readable problem."""
        if not self.docker:
            return "Docker is not installed (or not on PATH). Install Docker Desktop first."
        try:
            r = _run([self.docker, "info", "--format", "{{.ServerVersion}}"], timeout=15)
        except Exception as e:  # noqa
            return f"Cannot talk to Docker: {e}"
        if r.returncode != 0:
            return "Docker is installed but the daemon is not running. Start Docker Desktop and try again."
        return None

    def has_nvidia_runtime(self):
        try:
            r = _run([self.docker, "info", "--format", "{{json .Runtimes}}"], timeout=15)
            return r.returncode == 0 and "nvidia" in r.stdout.lower()
        except Exception:
            return False

    def image_name(self, gpu):
        return f"{IMAGE_REPO}:{'gpu' if gpu else 'cpu'}"

    def image_present(self, image):
        return _run([self.docker, "image", "inspect", image]).returncode == 0

    def ensure_image(self, gpu, ctx):
        image = self.image_name(gpu)
        if self.image_present(image):
            return image

        def relay(line):
            if line.strip():
                ctx.describe(line.strip()[:120])

        ctx.log(f"Docker image {image} not found locally. Pulling (first run only, may be several GB)...", "warning")
        ctx.describe(f"Pulling Docker image {image}...")
        if run_command_streaming([self.docker, "pull", image], ctx, on_line=relay) == 0:
            return image

        dockerfile = os.path.join(self.plugin_dir, "docker", f"Dockerfile.{'gpu' if gpu else 'cpu'}")
        if os.path.exists(dockerfile):
            ctx.log("Pull failed. Building the image locally instead...", "warning")
            ctx.describe("Building Docker image locally...")
            rc = run_command_streaming(
                [self.docker, "build", "-f", dockerfile, "-t", image, self.plugin_dir], ctx, on_line=relay)
            if rc == 0:
                return image
        raise BackendError(f"Could not pull or build the Docker image '{image}'.")

    # --- job execution ------------------------------------------------------
    def run(self, job, params, ctx):
        problem = self.diagnose()
        if problem:
            raise BackendUnavailable(problem)

        want_gpu = params.get("device", "cpu") != "cpu"
        use_gpu = want_gpu and self.has_nvidia_runtime()
        if want_gpu and not use_gpu:
            ctx.log("GPU requested, but Docker has no NVIDIA runtime. Running the container on CPU.", "warning")
            params = dict(params, device="cpu")

        image = self.ensure_image(use_gpu, ctx)

        models_dir = os.path.abspath(os.path.join(self.plugin_dir, "models"))
        mapper = PathMapper()
        cparams = mapper.translate_params(params)
        for key in ("weight_path", "model_path"):
            if cparams.get(key):
                cparams[key] = "/models/" + os.path.basename(cparams[key])

        job_dir = tempfile.mkdtemp(prefix="palmai_job_")
        try:
            with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as f:
                json.dump({"job": job, "params": cparams}, f)

            name = f"palmai-{uuid.uuid4().hex[:12]}"
            cmd = [self.docker, "run", "--rm", "--name", name, "-e", "PYTHONUNBUFFERED=1"]
            if use_gpu:
                cmd += ["--gpus", "all"]
            for host_dir, cdir in mapper.mounts():
                cmd += ["-v", f"{host_dir}:{cdir}"]
            cmd += ["-v", f"{models_dir}:/models:ro", "-v", f"{job_dir}:/job:ro", image,
                    "python", "-m", "palmai_core.worker", "/job/job.json"]

            ctx.describe("Starting container...")

            def kill_container():
                try:
                    _run([self.docker, "kill", name], timeout=30)
                except Exception:
                    pass

            return run_worker(cmd, ctx, map_result=mapper.translate_result, on_cancel=kill_container)
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
