import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from data_juicer.utils import model_utils


class Human3RModelsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "Human3R"
        self.repo.mkdir()
        cache = patch.object(model_utils, "DATA_JUICER_ASSETS_CACHE", str(self.root))
        cache.start()
        self.addCleanup(cache.stop)
        for name in [
            "smplx/SMPLX_NEUTRAL.npz",
            "smplx/SMPLX_MALE.npz",
            "smplx/SMPLX_FEMALE.npz",
            "smplx/smplx2smpl.pkl",
            "smpl/SMPL_NEUTRAL.pkl",
            "smpl/SMPL_MALE.pkl",
            "smpl/SMPL_FEMALE.pkl",
            "smpl/J_regressor_h36m.npy",
        ]:
            path = self.repo / "src/models" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.checkpoint = self.root / "human3r_896L.pth"
        self.checkpoint.touch()
        self.model_class = Mock()
        self.model = self.model_class.from_pretrained.return_value.to.return_value
        modules = {name: ModuleType(name) for name in ["src", "src.dust3r", "src.dust3r.model", "src.dust3r.inference"]}
        modules["src.dust3r.model"].ARCroco3DStereo = self.model_class
        modules["src.dust3r.inference"].inference_recurrent_lighter = Mock()
        self.modules = modules

    def load(self, **kwargs):
        return model_utils.prepare_human_3d_pose_human3r(str(self.checkpoint), device="cpu", **kwargs)

    def test_prepared_checkout_loads_and_restores_import_path(self):
        original = sys.path.copy()
        with patch.dict(sys.modules, self.modules):
            model, _ = self.load(human3r_repo_path=str(self.repo))
        self.assertIs(model, self.model)
        self.assertEqual(sys.path, original)
        self.model_class.from_pretrained.assert_called_once_with(str(self.checkpoint))

    def test_missing_licensed_assets_are_reported_before_import(self):
        (self.repo / "src/models/smplx/SMPLX_NEUTRAL.npz").unlink()
        with patch.dict(sys.modules, self.modules):
            with self.assertRaisesRegex(FileNotFoundError, "SMPLX_NEUTRAL"):
                self.load(human3r_repo_path=str(self.repo))
        self.model_class.from_pretrained.assert_not_called()

    def test_new_checkout_with_assets_does_not_force_a_restart(self):
        assets = self.repo / "src/models"
        with (
            patch.object(model_utils, "DATA_JUICER_ASSETS_CACHE", str(self.root)),
            patch.object(model_utils.os.path, "exists", wraps=model_utils.os.path.exists) as exists,
            patch.object(model_utils.subprocess, "run") as clone,
            patch.dict(sys.modules, self.modules),
        ):
            exists.side_effect = lambda path: False if str(path) == str(self.repo) else Path(path).exists()
            model, _ = self.load()
        clone.assert_called_once()
        self.assertIs(model, self.model)
        self.assertTrue(assets.is_dir())

    def test_import_failure_restores_import_path(self):
        import builtins

        original_path = sys.path.copy()
        original_import = builtins.__import__

        def fail(name, *args, **kwargs):
            if name == "src.dust3r.inference":
                raise ImportError("missing Human3R dependency")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fail):
            with self.assertRaisesRegex(ImportError, "Human3R dependency"):
                self.load(human3r_repo_path=str(self.repo))
        self.assertEqual(sys.path, original_path)

    def test_stale_placeholder_does_not_block_checkpoint_download(self):
        (self.root / "human3r/placeholder").mkdir(parents=True)
        hub = SimpleNamespace(hf_hub_download=Mock(return_value=str(self.checkpoint)))
        with (
            patch.object(model_utils, "DJMC", str(self.root)),
            patch.object(model_utils, "LazyLoader", return_value=hub),
            patch.dict(sys.modules, self.modules),
        ):
            model_utils.prepare_human_3d_pose_human3r("missing.pth", human3r_repo_path=str(self.repo), device="cpu")
        hub.hf_hub_download.assert_called_once()
        self.model_class.from_pretrained.assert_called_once_with(str(self.checkpoint))


if __name__ == "__main__":
    unittest.main()
