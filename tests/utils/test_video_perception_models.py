import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from data_juicer.utils import model_utils


class VideoPerceptionModelsTest(unittest.TestCase):
    def test_metric3d_rejects_session_fallback_to_cpu(self):
        runtime = Mock()
        runtime.InferenceSession.return_value.get_providers.return_value = ["CPUExecutionProvider"]
        with tempfile.NamedTemporaryFile(suffix=".onnx") as checkpoint:
            with patch.dict(sys.modules, {"onnxruntime": runtime}):
                with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"):
                    model_utils.prepare_normal_map_metric3d(checkpoint.name, device="cuda:0")

    def test_metric3d_uses_the_requested_cuda_provider(self):
        runtime = Mock()
        session = runtime.InferenceSession.return_value
        session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        with tempfile.NamedTemporaryFile(suffix=".onnx") as checkpoint:
            with patch.dict(sys.modules, {"onnxruntime": runtime}):
                result = model_utils.prepare_normal_map_metric3d(checkpoint.name, device="cuda:2")
        self.assertIs(result, session)
        providers = runtime.InferenceSession.call_args.kwargs["providers"]
        self.assertEqual(providers[0][0], "CUDAExecutionProvider")
        self.assertEqual(providers[0][1]["device_id"], "2")

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
