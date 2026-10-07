# -*- coding: utf-8 -*-
"""Backend settings dialog (built in code, no .ui file)."""

from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton,
                                 QHBoxLayout, QPlainTextEdit)
from qgis.core import QgsApplication, QgsTask, QgsMessageLog, Qgis

from ..palmai_core.context import JobContext, JobCanceled
from . import (BACKEND_CHOICES, PLUGIN_DIR, default_env_dir, get_choice, set_choice)
from .docker_backend import DockerBackend
from .venv_backend import VenvBackend
from .local_backend import LocalBackend
from .base import BackendError


class _VenvSetupTask(QgsTask):
    def __init__(self, backend, on_done):
        super().__init__("PalmAI: setting up Python environment", QgsTask.CanCancel)
        self.backend, self.on_done, self.error = backend, on_done, None

    def run(self):
        ctx = JobContext(
            on_progress=self.setProgress, on_describe=self.setDescription,
            on_log=lambda m, l: QgsMessageLog.logMessage(m, "PalmAI Setup", Qgis.Info),
            is_canceled=self.isCanceled)
        try:
            self.backend.setup(ctx)
            return True
        except JobCanceled:
            return False
        except BackendError as e:
            self.error = str(e)
            return False
        except Exception as e:  # noqa
            self.error = str(e)
            return False

    def finished(self, result):
        self.on_done(result, self.error)


class SettingsDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("PalmAI - Inference Backend")
        self.setMinimumWidth(520)
        self._task = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Where should PalmAI run the AI models?</b>"))

        self.combo = QComboBox()
        for key, label in BACKEND_CHOICES:
            self.combo.addItem(label, key)
        idx = self.combo.findData(get_choice())
        self.combo.setCurrentIndex(max(0, idx))
        layout.addWidget(self.combo)

        self.status = QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMinimumHeight(140)
        layout.addWidget(self.status)

        row = QHBoxLayout()
        self.btn_check = QPushButton("Check status")
        self.btn_setup = QPushButton("Set up Python environment")
        self.btn_save = QPushButton("Save")
        row.addWidget(self.btn_check)
        row.addWidget(self.btn_setup)
        row.addStretch(1)
        row.addWidget(self.btn_save)
        layout.addLayout(row)

        self.btn_check.clicked.connect(self.check_status)
        self.btn_setup.clicked.connect(self.setup_venv)
        self.btn_save.clicked.connect(self.save)
        self.check_status()

    def check_status(self):
        lines = []
        docker = DockerBackend(PLUGIN_DIR)
        problem = docker.diagnose()
        if problem:
            lines.append(f"Docker:      NOT READY - {problem}")
        else:
            gpu = "yes" if docker.has_nvidia_runtime() else "no"
            cpu_img = "ready" if docker.image_present(docker.image_name(False)) else "will be pulled on first run"
            lines.append(f"Docker:      READY (NVIDIA GPU runtime: {gpu}; CPU image: {cpu_img})")

        venv = VenvBackend(PLUGIN_DIR, default_env_dir())
        lines.append("Virtualenv:  " + ("READY" if venv.is_ready() else "not set up yet"))

        missing = LocalBackend().missing_modules()
        lines.append("QGIS Python: " + ("READY" if not missing else "missing " + ", ".join(missing)))
        self.status.setPlainText("\n".join(lines))

    def save(self):
        set_choice(self.combo.currentData())
        self.iface.messageBar().pushMessage("PalmAI", "Backend saved.", level=Qgis.Success, duration=4)
        self.accept()

    def setup_venv(self):
        backend = VenvBackend(PLUGIN_DIR, default_env_dir())
        self.btn_setup.setEnabled(False)

        def done(ok, error):
            self.btn_setup.setEnabled(True)
            if ok:
                self.iface.messageBar().pushMessage("PalmAI", "Python environment is ready.", level=Qgis.Success, duration=8)
            else:
                self.iface.messageBar().pushMessage(
                    "PalmAI", f"Environment setup failed: {error or 'cancelled'}", level=Qgis.Critical, duration=15)
            self.check_status()

        self._task = _VenvSetupTask(backend, done)
        QgsApplication.taskManager().addTask(self._task)
        self.iface.messageBar().pushMessage(
            "PalmAI", "Setting up the Python environment in the background (can take several minutes).",
            level=Qgis.Info, duration=8)
