import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest

import dill
import ffmpeg

from data_juicer.utils.ffmpeg_utils import STDERR_TAIL_CHARS, run_ffmpeg
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

REPO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")


class FakeStream:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.kwargs = None

    def run(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.result


class RunFFmpegTest(DataJuicerTestCaseBase):

    def test_success_returns_run_result(self):
        stream = FakeStream(result=(b"out", b"err"))
        self.assertEqual(run_ffmpeg(stream, capture_stderr=True), (b"out", b"err"))
        self.assertEqual(stream.kwargs, {"capture_stderr": True})

    def test_error_becomes_picklable_runtime_error(self):
        stderr = b"/tmp/corrupt.mp4: moov atom not found\n"
        original = ffmpeg.Error("ffmpeg", b"", stderr)
        # the root cause: the original error cannot be rebuilt after dill
        with self.assertRaises(TypeError):
            dill.loads(dill.dumps(original))

        with self.assertRaises(RuntimeError) as ctx:
            run_ffmpeg(FakeStream(error=original))
        err = ctx.exception
        self.assertIn("moov atom not found", str(err))
        self.assertIs(err.__cause__, original)
        for dumps, loads in ((pickle.dumps, pickle.loads), (dill.dumps, dill.loads)):
            restored = loads(dumps(err))
            self.assertIsInstance(restored, RuntimeError)
            self.assertEqual(str(restored), str(err))

    def test_error_without_captured_stderr(self):
        with self.assertRaises(RuntimeError) as ctx:
            run_ffmpeg(FakeStream(error=ffmpeg.Error("ffmpeg", None, None)))
        self.assertEqual(str(ctx.exception), "ffmpeg error")

    def test_long_stderr_keeps_only_the_tail(self):
        stderr = b"x" * (STDERR_TAIL_CHARS * 3) + b"the real cause"
        with self.assertRaises(RuntimeError) as ctx:
            run_ffmpeg(FakeStream(error=ffmpeg.Error("ffmpeg", b"", stderr)))
        message = str(ctx.exception)
        self.assertTrue(message.endswith("the real cause"))
        self.assertLessEqual(len(message), STDERR_TAIL_CHARS + len("ffmpeg error: "))

    def test_strict_mode_on_corrupt_video_fails_instead_of_hanging(self):
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = os.path.join(tmp, "corrupt.mp4")
            with open(corrupt, "wb") as f:
                f.write(b"\x00" * 2048 + b"not a real mp4 payload" * 16)
            script = textwrap.dedent(
                f"""
                from data_juicer.core.data import NestedDataset as Dataset
                from data_juicer.ops.mapper.video_ffmpeg_wrapped_mapper import VideoFFmpegWrappedMapper

                if __name__ == "__main__":
                    op = VideoFFmpegWrappedMapper("scale", filter_kwargs={{"w": 96, "h": 54}}, save_dir={tmp!r})
                    Dataset.from_list([{{"videos": [{corrupt!r}]}}]).map(op.process, num_proc=2)
                """
            )
            # a file, not a pipe: on a hang, orphaned pool workers would keep a
            # pipe open and the timeout below would never return
            log_path = os.path.join(tmp, "stderr.log")
            with open(log_path, "wb") as log:
                proc = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=REPO_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                    timeout=300,
                )
            with open(log_path, encoding="utf-8", errors="replace") as log:
                stderr = log.read()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ffmpeg error", stderr)
        self.assertNotIn("missing 2 required positional arguments", stderr)


if __name__ == "__main__":
    unittest.main()
