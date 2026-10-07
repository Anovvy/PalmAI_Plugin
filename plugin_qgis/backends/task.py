# -*- coding: utf-8 -*-
"""Generic QgsTask that runs a palmai_core job through the configured backend."""

import traceback

from qgis.core import QgsTask, QgsMessageLog, Qgis

from ..palmai_core.context import JobContext, JobCanceled
from .base import BackendError
from . import create_backend

_LEVELS = {"info": Qgis.Info, "success": Qgis.Success, "warning": Qgis.Warning, "critical": Qgis.Critical}


class PalmAITask(QgsTask):
    """
    on_done(result: dict | None, error: str | None, canceled: bool) is called in the
    main thread once the job is finished.
    """

    def __init__(self, description, job, params, plugin_name, on_done, models=None):
        super().__init__(description, QgsTask.CanCancel)
        self.job = job
        self.params = params
        self.plugin_name = plugin_name
        self.on_done = on_done
        self.models = models or []  # [(param_key, filename)] resolved/downloaded before the run
        self.result = None
        self.error = None
        self.was_canceled = False

    def _ctx(self):
        return JobContext(
            on_progress=self.setProgress,
            on_describe=self.setDescription,
            on_log=lambda m, l: QgsMessageLog.logMessage(m, self.plugin_name, _LEVELS.get(l, Qgis.Info)),
            is_canceled=self.isCanceled,
        )

    def run(self):
        ctx = self._ctx()
        try:
            import os
            from ..palmai_core.model_store import ensure_model
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            for key, filename in self.models:
                self.params[key] = ensure_model(filename, models_dir, ctx)

            backend = create_backend()
            ctx.log(f"Using backend: {backend.name}")
            self.result = backend.run(self.job, self.params, ctx)
            return True
        except JobCanceled:
            self.was_canceled = True
            return False
        except BackendError as e:
            self.error = str(e)
            QgsMessageLog.logMessage(self.error, self.plugin_name, Qgis.Critical)
            return False
        except Exception as e:  # noqa
            self.error = f"{e}"
            QgsMessageLog.logMessage(f"FATAL ERROR: {e}\n{traceback.format_exc()}", self.plugin_name, Qgis.Critical)
            return False

    def finished(self, result):
        self.on_done(self.result if result else None, self.error, self.was_canceled or self.isCanceled())
