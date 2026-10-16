# -*- coding: utf-8 -*-
"""Tests for the @@PALMAI@@ worker protocol and process runner."""

import json
import unittest
from unittest.mock import MagicMock, patch

from plugin_qgis.backends.base import BackendError
from plugin_qgis.backends.process_runner import run_worker
from plugin_qgis.palmai_core.context import JobCanceled
from plugin_qgis.palmai_core.worker import PREFIX, emit


class TestWorkerProtocol(unittest.TestCase):
    def test_emit_formatting(self):
        with patch("sys.stdout.write") as mock_write, patch("sys.stdout.flush"):
            emit(type="progress", value=45)
            mock_write.assert_called_once()
            line = mock_write.call_args[0][0]
            self.assertTrue(line.startswith(PREFIX))
            data = json.loads(line[len(PREFIX):])
            self.assertEqual(data["type"], "progress")
            self.assertEqual(data["value"], 45)

    @patch("plugin_qgis.backends.process_runner.run_command_streaming")
    def test_run_worker_dispatches_messages(self, mock_streaming):
        ctx = MagicMock()

        # Simulate stream of lines
        def fake_streaming(cmd, ctx, cwd=None, env=None, on_line=None):
            on_line("Library initializing...")  # chatter line
            on_line(PREFIX + json.dumps({"type": "describe", "text": "Detecting trees..."}))
            on_line(PREFIX + json.dumps({"type": "progress", "value": 75}))
            on_line(PREFIX + json.dumps({"type": "log", "message": "Batch 1 done", "level": "info"}))
            on_line(PREFIX + json.dumps({"type": "result", "data": {"total_detections": 120, "layers": []}}))
            return 0

        mock_streaming.side_effect = fake_streaming

        res = run_worker(["python", "worker.py"], ctx)
        self.assertEqual(res["total_detections"], 120)
        ctx.describe.assert_called_with("Detecting trees...")
        ctx.progress.assert_called_with(75)
        ctx.log.assert_called_with("Batch 1 done", "info")

    @patch("plugin_qgis.backends.process_runner.run_command_streaming")
    def test_run_worker_relays_errors(self, mock_streaming):
        ctx = MagicMock()

        def fake_streaming(cmd, ctx, cwd=None, env=None, on_line=None):
            on_line(PREFIX + json.dumps({
                "type": "error",
                "message": "CUDA out of memory",
                "traceback": "Traceback (most recent call last)..."
            }))
            return 1

        mock_streaming.side_effect = fake_streaming

        with self.assertRaises(BackendError) as cm:
            run_worker(["python", "worker.py"], ctx)
        self.assertIn("CUDA out of memory", str(cm.exception))
        ctx.log.assert_called()

    @patch("plugin_qgis.backends.process_runner.run_command_streaming")
    def test_run_worker_handles_canceled(self, mock_streaming):
        ctx = MagicMock()

        def fake_streaming(cmd, ctx, cwd=None, env=None, on_line=None):
            on_line(PREFIX + json.dumps({"type": "canceled"}))
            return 0

        mock_streaming.side_effect = fake_streaming

        with self.assertRaises(JobCanceled):
            run_worker(["python", "worker.py"], ctx)


if __name__ == "__main__":
    unittest.main()
