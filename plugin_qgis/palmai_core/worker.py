# -*- coding: utf-8 -*-
"""
Subprocess worker used by the venv and Docker backends.

    python -m palmai_core.worker JOB.json     # run a job
    python -m palmai_core.worker --check      # verify the environment

Protocol: every message is ONE stdout line `@@PALMAI@@ {json}`. Any other stdout
line (library chatter) is forwarded as a log message by the parent.
Message types: progress, describe, log, result, error, canceled, check.
"""

import json
import sys
import traceback

PREFIX = "@@PALMAI@@ "


def emit(**msg):
    sys.stdout.write(PREFIX + json.dumps(msg) + "\n")
    sys.stdout.flush()


def check_environment():
    info = {}
    for mod in ("numpy", "torch", "torchvision", "ultralytics", "sahi", "rasterio",
                "geopandas", "fiona", "shapely", "skimage", "scipy", "cv2", "PIL"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "ok")
        except Exception as e:  # noqa
            info[mod] = f"MISSING: {e}"
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps is not None and mps.is_available())
    except Exception:
        pass
    emit(type="check", data=info)
    return 0 if not any(str(v).startswith("MISSING") for v in info.values()) else 1


def main(argv):
    if len(argv) >= 2 and argv[1] == "--check":
        return check_environment()
    if len(argv) < 2:
        print("usage: worker JOB.json | --check", file=sys.stderr)
        return 2

    from .context import JobContext, JobCanceled
    from .runner import run_job

    with open(argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)

    ctx = JobContext(
        on_progress=lambda v: emit(type="progress", value=v),
        on_describe=lambda t: emit(type="describe", text=t),
        on_log=lambda m, l: emit(type="log", message=m, level=l),
        is_canceled=None,  # cancellation = parent terminates the process
    )
    try:
        result = run_job(payload["job"], payload["params"], ctx)
        emit(type="result", data=result)
        return 0
    except JobCanceled:
        emit(type="canceled")
        return 0
    except Exception as e:  # noqa
        emit(type="error", message=str(e), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
