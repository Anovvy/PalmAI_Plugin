# -*- coding: utf-8 -*-
from .context import JobContext


def run_job(job, params, ctx: JobContext):
    """Dispatch a job by name. Returns a JSON-serialisable dict."""
    if job in ("counting", "centertree", "classification"):
        from .yolo import run_yolo_job
        return run_yolo_job(job, params, ctx)
    if job == "road":
        from .road import run_road_job
        return run_road_job(params, ctx)
    raise ValueError(f"Unknown job type: {job}")
