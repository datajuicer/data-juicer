import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from data_juicer.utils import model_utils


class VideoPerceptionModelsTest(unittest.TestCase):
    def test_metric3d_recovers_after_an_interrupted_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metric3d" / "placeholder").mkdir(parents=True)
            checkpoint = root / "downloaded.onnx"
            checkpoint.touch()
            download = Mock(return_value=str(checkpoint))
            runtime = Mock()
            hub = SimpleNamespace(hf_hub_download=download)
            with (
                patch.object(model_utils, "DJMC", directory),
                patch.object(model_utils, "LazyLoader", return_value=hub),
                patch.dict(sys.modules, {"onnxruntime": runtime}),
                patch("time.sleep", side_effect=AssertionError("Download must not poll a placeholder")),
            ):
                model_utils.prepare_normal_map_metric3d("missing.onnx", device="cpu")
            download.assert_called_once()
            self.assertEqual(runtime.InferenceSession.call_args.args[0], str(checkpoint))
            self.assertEqual(runtime.InferenceSession.call_args.kwargs["providers"], ["CPUExecutionProvider"])

    def test_metric3d_download_errors_propagate(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = SimpleNamespace(hf_hub_download=Mock(side_effect=OSError("offline")))
            with (
                patch.object(model_utils, "DJMC", directory),
                patch.object(model_utils, "LazyLoader", return_value=hub),
            ):
                with self.assertRaisesRegex(OSError, "offline"):
                    model_utils.prepare_normal_map_metric3d("missing.onnx", device="cpu")


if __name__ == "__main__":
    unittest.main()
