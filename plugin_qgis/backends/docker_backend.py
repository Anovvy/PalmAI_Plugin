# -*- coding: utf-8 -*-
"""
Docker backend: one ephemeral `docker run --rm` container per job (no server, no ports).

Security & Stability:
- Least-privilege container volume mounts (inputs :ro, models :ro, job json :ro, outputs :rw).
- Versioned container image resolution with backward-compatible fallback tags.
- Robust GPU probing avoiding false positives on systems with missing drivers.
- Complete container lifecycle cancellation and temporary working file cleanup.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

from .base import BackendError, BackendUnavailable, InferenceBackend
from .path_mapping import PathMapper
from .process_runner import _popen_flags, run_command_streaming, run_worker

IMAGE_REPO = "ghcr.io/anovvy/palmai-core"
DEFAULT_VERSION_TAG = "v2.0.0"
LEGACY_TAG_COMPAT = True


def _windows_docker_candidates():
    candidates = []
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")

    if local_app_data:
        candidates.extend([
            os.path.join(local_app_data, "Programs", "DockerDesktop", "resources", "bin", "docker.exe"),
            os.path.join(local_app_data, "Programs", "Docker Desktop", "resources", "bin", "docker.exe"),
            os.path.join(local_app_data, "Docker Desktop", "resources", "bin", "docker.exe"),
            os.path.join(local_app_data, "Docker", "Docker", "resources", "bin", "docker.exe"),
        ])

    candidates.extend([
        os.path.join(program_files, "Docker", "Docker", "resources", "bin", "docker.exe"),
        os.path.join(program_files, "Docker", "Docker", "bin", "docker.exe"),
        r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
        r"C:\Program Files\Docker\Docker\bin\docker.exe",
    ])
    return candidates


def _macos_docker_candidates():
    home = os.path.expanduser("~")
    return [
        "/usr/local/bin/docker",
        "/opt/homebrew/bin/docker",
        os.path.join(home, ".docker", "bin", "docker"),
        "/Applications/Docker.app/Contents/Resources/bin/docker",
        "/usr/bin/docker",
    ]


def _linux_docker_candidates():
    home = os.path.expanduser("~")
    return [
        "/usr/bin/docker",
        "/usr/local/bin/docker",
        "/snap/bin/docker",
        os.path.join(home, ".docker", "bin", "docker"),
    ]


def find_docker():
    exe = shutil.which("docker")
    if exe:
        return exe

    candidates = []
    if sys.platform == "win32":
        candidates = _windows_docker_candidates() or []
    elif sys.platform == "darwin":
        candidates = _macos_docker_candidates() or []
    else:
        candidates = _linux_docker_candidates() or []

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            docker_dir = os.path.dirname(candidate)
            current_path = os.environ.get("PATH", "")
            if docker_dir.lower() not in current_path.lower():
                os.environ["PATH"] = docker_dir + os.pathsep + current_path
            return candidate
    return None


def _run(cmd, timeout=20):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        **_popen_flags(),
    )


def get_remote_image_info(docker_bin, image):
    """Fetch total compressed download size in bytes and layer count from image manifest without pulling."""
    try:
        r = _run([docker_bin, "manifest", "inspect", "-v", image], timeout=15)
        if r.returncode == 0 and r.stdout:
            data = json.loads(r.stdout)
            manifests = data if isinstance(data, list) else [data]
            for m in manifests:
                oci = m.get("OCIManifest") or m
                layers = oci.get("layers", [])
                if len(layers) > 1:
                    total_bytes = sum(l.get("size", 0) for l in layers)
                    return total_bytes, len(layers)
    except Exception:
        pass
    return None, 0


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
        except Exception as e:
            return f"Cannot talk to Docker: {e}"
        if r.returncode != 0:
            return "Docker is installed but the daemon is not running. Start Docker Desktop and try again."
        return None

    def has_nvidia_runtime(self):
        """
        Check whether Docker can actually run with NVIDIA GPU acceleration.
        Verifies that Docker daemon supports nvidia runtime AND that an NVIDIA GPU
        adapter/driver is physically present on the host (prevents WSL2 false positives).
        """
        try:
            r = _run([self.docker, "info", "--format", "{{json .Runtimes}}"], timeout=15)
            if r.returncode != 0 or "nvidia" not in r.stdout.lower():
                return False
        except Exception:
            return False

        # macOS Docker does not support NVIDIA GPU pass-through
        if sys.platform == "darwin":
            return False

        # On Windows (WSL2), Docker pre-registers nvidia runtime by default even without
        # an NVIDIA GPU installed. Verify that an NVIDIA GPU driver/smi actually exists.
        if sys.platform == "win32":
            has_smi = (
                shutil.which("nvidia-smi")
                or os.path.exists(r"C:\Windows\System32\nvidia-smi.exe")
                or os.path.exists(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
            )
            return bool(has_smi)

        # On Linux, verify nvidia-smi is available
        if sys.platform.startswith("linux"):
            return bool(shutil.which("nvidia-smi"))

        return True

    def get_configured_version(self):
        """Read configured version or return default pinned version."""
        try:
            from qgis.PyQt.QtCore import QSettings
            tag = QSettings().value("PalmAI/docker_version", DEFAULT_VERSION_TAG)
            return tag if tag else DEFAULT_VERSION_TAG
        except Exception:
            return DEFAULT_VERSION_TAG

    def image_name(self, gpu, version=None):
        ver = version or self.get_configured_version()
        variant = "gpu" if gpu else "cpu"
        return f"{IMAGE_REPO}:{ver}-{variant}"

    def legacy_image_name(self, gpu):
        return f"{IMAGE_REPO}:{'gpu' if gpu else 'cpu'}"

    def image_present(self, image):
        if not self.docker:
            return False
        return _run([self.docker, "image", "inspect", image]).returncode == 0

    def get_image_digest(self, image):
        """Return image repo digest (sha256:...) if present locally."""
        try:
            r = _run([self.docker, "image", "inspect", "--format", "{{index .RepoDigests 0}}", image], timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
        return None

    def ensure_image(self, gpu, ctx):
        primary_image = self.image_name(gpu)
        legacy_image = self.legacy_image_name(gpu)

        # 1. Use primary versioned image if already present
        if self.image_present(primary_image):
            return primary_image

        # 2. Check legacy tag if already present locally
        if LEGACY_TAG_COMPAT and self.image_present(legacy_image):
            ctx.log(f"Using existing local compatibility image '{legacy_image}'.", "info")
            return legacy_image

        # 3. Pull primary versioned image, falling back to legacy tag on GHCR
        target_images = [primary_image]
        if LEGACY_TAG_COMPAT and legacy_image != primary_image:
            target_images.append(legacy_image)

        for img in target_images:
            total_bytes, total_layers = get_remote_image_info(self.docker, img)
            if total_bytes and total_bytes >= 1024 * 1024 * 1024:
                size_str = f"{total_bytes / (1024 * 1024 * 1024):.2f} GB"
            elif total_bytes:
                size_str = f"{total_bytes / (1024 * 1024):.1f} MB"
            else:
                size_str = "~656 MB" if not gpu else "~3.14 GB"

            ctx.log(
                f"Docker image '{img}' not found locally. Downloading from GitHub Container Registry (GHCR)... "
                f"Estimated download size: {size_str} (first run only).",
                "info",
            )
            ctx.describe(f"Pulling Docker image {img} ({size_str})...")

            completed_downloads = set()
            completed_pulls = set()

            def relay(line):
                sline = line.strip()
                if not sline:
                    return

                parts = sline.split(":")
                if len(parts) >= 2:
                    layer_id = parts[0].strip()
                    status = parts[1].strip()
                    if "Download complete" in status:
                        completed_downloads.add(layer_id)
                    elif "Pull complete" in status:
                        completed_pulls.add(layer_id)

                if total_layers > 0:
                    pct = int(
                        (len(completed_downloads) * 50 / total_layers)
                        + (len(completed_pulls) * 50 / total_layers)
                    )
                    pct = min(99, max(1, pct))
                    ctx.progress(pct)
                    ctx.describe(f"Downloading Docker image ({size_str}) [{pct}%]: {sline[:70]}")
                else:
                    ctx.describe(f"Pulling Docker image: {sline[:100]}")

            if run_command_streaming([self.docker, "pull", img], ctx, on_line=relay) == 0:
                ctx.progress(100)
                ctx.log(f"Docker image '{img}' successfully pulled and verified.", "info")
                return img

        # 4. Fallback to building locally
        dockerfile = os.path.join(self.plugin_dir, "docker", f"Dockerfile.{'gpu' if gpu else 'cpu'}")
        if os.path.exists(dockerfile):
            ctx.log("Remote pull failed. Building the image locally instead...", "warning")
            ctx.describe("Building Docker image locally...")
            rc = run_command_streaming(
                [self.docker, "build", "-f", dockerfile, "-t", primary_image, self.plugin_dir],
                ctx,
                on_line=relay,
            )
            if rc == 0:
                ctx.progress(100)
                return primary_image

        raise BackendError(f"Could not pull or build the Docker image '{primary_image}'.")

    # --- job execution ------------------------------------------------------
    def _execute_container(self, job, params, use_gpu, image, models_dir, ctx):
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

            # Least privilege volume mounts: input paths :ro, output paths :rw
            for host_dir, cdir, mode in mapper.mount_specs():
                cmd += ["-v", f"{host_dir}:{cdir}:{mode}"]

            # Models mounted strictly read-only; job payload mounted read-only
            cmd += [
                "-v", f"{models_dir}:/models:ro",
                "-v", f"{job_dir}:/job:ro",
                image,
                "python", "-m", "palmai_core.worker", "/job/job.json",
            ]

            ctx.describe(f"Starting {'GPU' if use_gpu else 'CPU'} container...")

            def kill_container():
                try:
                    _run([self.docker, "kill", name], timeout=30)
                except Exception:
                    pass

            return run_worker(cmd, ctx, map_result=mapper.translate_result, on_cancel=kill_container)
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    def run(self, job, params, ctx):
        problem = self.diagnose()
        if problem:
            raise BackendUnavailable(problem)

        want_gpu = params.get("device", "cpu") != "cpu"
        use_gpu = want_gpu and self.has_nvidia_runtime()
        if want_gpu and not use_gpu:
            ctx.log("GPU requested, but Docker has no NVIDIA runtime. Running container on CPU.", "warning")
            params = dict(params, device="cpu")

        image = self.ensure_image(use_gpu, ctx)
        digest = self.get_image_digest(image)
        if digest:
            ctx.log(f"Execution runtime image digest: {digest}", "info")

        models_dir = os.path.abspath(os.path.join(self.plugin_dir, "models"))

        if use_gpu:
            try:
                return self._execute_container(job, params, True, image, models_dir, ctx)
            except BackendError as err:
                err_str = str(err).lower()
                is_gpu_runtime_error = any(sig in err_str for sig in (
                    "no adapters were found",
                    "wsl environment detected",
                    "nvidia-container-cli",
                    "could not select device driver",
                    "unknown or invalid runtime name",
                    "code 125",
                ))
                if not is_gpu_runtime_error:
                    raise

                ctx.log(
                    "GPU container execution failed: Incompatible CUDA driver or missing WSL 2 GPU adapter.\n"
                    "Error details: nvidia-container-cli initialization error (WSL environment detected but no adapters were found).\n"
                    "Notice: Your host NVIDIA display driver or CUDA version is incompatible with the GPU container runtime. "
                    "A modern NVIDIA display driver (version >= 528.xx supporting CUDA 12+) is required for WSL 2 GPU acceleration.\n"
                    "Automatically falling back to CPU execution so your analysis can proceed...",
                    "warning",
                )
                cpu_params = dict(params, device="cpu")
                cpu_image = self.ensure_image(gpu=False, ctx=ctx)
                cpu_digest = self.get_image_digest(cpu_image)
                if cpu_digest:
                    ctx.log(f"Fallback CPU image digest: {cpu_digest}", "info")
                return self._execute_container(job, cpu_params, False, cpu_image, models_dir, ctx)

        return self._execute_container(job, params, False, image, models_dir, ctx)
