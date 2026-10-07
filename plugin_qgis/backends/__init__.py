# -*- coding: utf-8 -*-
import os

from .base import BackendError, BackendUnavailable

BACKEND_CHOICES = (
    ("auto", "Automatic (Docker > Virtualenv > QGIS Python)"),
    ("docker", "Docker (recommended, most reproducible)"),
    ("venv", "Virtualenv (no Docker; supports CUDA and Apple MPS)"),
    ("local", "QGIS Python (legacy, manual pip install)"),
)

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_env_dir():
    from qgis.core import QgsApplication
    return os.path.join(QgsApplication.qgisSettingsDirPath(), "palmai_env")


def get_choice():
    from qgis.PyQt.QtCore import QSettings
    return QSettings().value("PalmAI/backend", "auto")


def set_choice(choice):
    from qgis.PyQt.QtCore import QSettings
    QSettings().setValue("PalmAI/backend", choice)


def create_backend(choice=None, plugin_dir=PLUGIN_DIR, env_dir=None):
    """Instantiate the configured backend. Raises BackendUnavailable with guidance."""
    from .docker_backend import DockerBackend
    from .venv_backend import VenvBackend
    from .local_backend import LocalBackend

    choice = choice or get_choice()
    env_dir = env_dir or default_env_dir()

    if choice == "docker":
        b = DockerBackend(plugin_dir)
        problem = b.diagnose()
        if problem:
            raise BackendUnavailable(problem)
        return b
    if choice == "venv":
        return VenvBackend(plugin_dir, env_dir)
    if choice == "local":
        return LocalBackend()

    # auto
    docker = DockerBackend(plugin_dir)
    if docker.is_available():
        return docker
    venv = VenvBackend(plugin_dir, env_dir)
    if venv.is_ready():
        return venv
    local = LocalBackend()
    if local.is_available():
        return local
    raise BackendUnavailable(
        "No inference backend is ready. Install Docker Desktop and start it, or open "
        "PalmAI > Settings and set up the Python environment.")
