# -*- coding: utf-8 -*-
"""Inference backends. This package does not import qgis (except task.py)."""


class BackendError(Exception):
    """Inference failed (message is user-presentable)."""


class BackendUnavailable(BackendError):
    """The selected backend cannot be used on this machine."""


class InferenceBackend:
    name = "base"

    def is_available(self):
        raise NotImplementedError

    def run(self, job, params, ctx):
        """Blocking. Returns result dict; raises JobCanceled / BackendError."""
        raise NotImplementedError
