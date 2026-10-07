# -*- coding: utf-8 -*-
"""Progress / cancel / log abstraction so the pipelines do not depend on QgsTask."""


class JobCanceled(Exception):
    """Raised when the user cancels a running job."""


class JobContext:
    def __init__(self, on_progress=None, on_describe=None, on_log=None, is_canceled=None):
        self._on_progress = on_progress
        self._on_describe = on_describe
        self._on_log = on_log
        self._is_canceled = is_canceled

    def progress(self, value):
        if self._on_progress:
            self._on_progress(int(value))

    def describe(self, text):
        if self._on_describe:
            self._on_describe(str(text))

    def log(self, message, level="info"):
        """level: info | success | warning | critical"""
        if self._on_log:
            self._on_log(str(message), level)

    def canceled(self):
        return bool(self._is_canceled and self._is_canceled())

    def check_cancel(self):
        if self.canceled():
            raise JobCanceled()
