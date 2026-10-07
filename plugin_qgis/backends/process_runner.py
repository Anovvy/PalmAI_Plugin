# -*- coding: utf-8 -*-
"""Run the palmai_core worker as a child process and relay its protocol messages."""

import json
import queue
import subprocess
import sys
import threading

from ..palmai_core.context import JobCanceled
from ..palmai_core.worker import PREFIX
from .base import BackendError


def _popen_flags():
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def run_command_streaming(cmd, ctx, cwd=None, env=None, on_line=None, timeout_poll=0.2):
    """Run a generic command, stream merged output lines to on_line. Returns exit code."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", cwd=cwd, env=env, **_popen_flags())
    q = queue.Queue()

    def reader():
        for line in proc.stdout:
            q.put(line.rstrip("\r\n"))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    try:
        while True:
            if ctx and ctx.canceled():
                proc.kill()
                raise JobCanceled()
            try:
                line = q.get(timeout=timeout_poll)
            except queue.Empty:
                continue
            if line is None:
                break
            if on_line:
                on_line(line)
    finally:
        if proc.poll() is None:
            proc.kill()
    return proc.wait()


def run_worker(cmd, ctx, map_result=None, cwd=None, env=None, on_cancel=None):
    """Execute a worker command line; returns the job result dict."""
    state = {"result": None, "error": None, "canceled": False, "tail": []}

    def handle(line):
        if line.startswith(PREFIX):
            try:
                msg = json.loads(line[len(PREFIX):])
            except ValueError:
                return
            t = msg.get("type")
            if t == "progress":
                ctx.progress(msg["value"])
            elif t == "describe":
                ctx.describe(msg["text"])
            elif t == "log":
                ctx.log(msg["message"], msg.get("level", "info"))
            elif t == "result":
                state["result"] = msg["data"]
            elif t == "error":
                state["error"] = msg
            elif t == "canceled":
                state["canceled"] = True
        elif line.strip():
            state["tail"] = (state["tail"] + [line])[-25:]

    try:
        rc = run_command_streaming(cmd, ctx, cwd=cwd, env=env, on_line=handle)
    except JobCanceled:
        if on_cancel:
            on_cancel()
        raise

    if state["canceled"]:
        raise JobCanceled()
    if state["error"]:
        err = state["error"]
        ctx.log(err.get("traceback", ""), "critical")
        raise BackendError(err.get("message", "Unknown worker error"))
    if state["result"] is None:
        raise BackendError("The worker exited without a result (code {}).\n{}".format(
            rc, "\n".join(state["tail"])))
    return map_result(state["result"]) if map_result else state["result"]
