"""Load private source instances so optional-dependency stubs cannot escape tests."""

import importlib.util
import sys
import types
import unittest
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import einops
import einops.layers.torch
import torch
import torch.utils.checkpoint
import yacs.config

from data_juicer.ops import common
from data_juicer.ops.common import hawor_constants  # noqa: F401
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


def _load_source(name):
    source = Path(hawor_constants.__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_dj_test_{name}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_test_modules():
    """Keep mocked class bases out of the canonical production module namespace."""
    timm = types.ModuleType("timm")
    timm.layers = types.SimpleNamespace(
        drop_path=lambda x, drop_prob=0.0, training=False: (
            torch.zeros_like(x) if training and drop_prob == 1.0 else x
        ),
        to_2tuple=lambda x: x if isinstance(x, tuple) else (x, x),
        trunc_normal_=torch.nn.init.trunc_normal_,
    )
    smplx = types.ModuleType("smplx")
    smplx.MANOLayer = torch.nn.Module
    smplx.utils = types.SimpleNamespace(MANOOutput=object, to_tensor=torch.as_tensor)
    smplx.vertex_ids = types.SimpleNamespace(vertex_ids={"mano": {}})
    smplx.lbs = types.SimpleNamespace(vertices2joints=lambda regressor, vertices: vertices)
    lightning = types.ModuleType("pytorch_lightning")
    lightning.LightningModule = torch.nn.Module
    lightning.utilities = types.SimpleNamespace(
        rank_zero_only=lambda fn: fn,
        rank_zero=types.SimpleNamespace(rank_zero_only=lambda fn: fn),
    )
    runtime = types.ModuleType("onnxruntime")
    runtime.InferenceSession = object
    hv = _load_source("hawor_func_vit")
    stubs = {
        "smplx": smplx,
        "pytorch_lightning": lightning,
        "onnxruntime": runtime,
        "data_juicer.ops.common.hawor_func_vit": hv,
    }
    with patch.dict(sys.modules, stubs):
        hw = _load_source("hawor_func")
        dw = _load_source("dwpose_func")
    # These are private modules; changing their globals cannot affect real operators.
    hw.einops = einops
    hw.yacs = yacs
    hv.timm = timm
    dw.onnxruntime = runtime
    return hw, hv, dw


class HaworImportIsolationTest(DataJuicerTestCaseBase):
    def test_private_imports_preserve_production_modules(self):
        missing = object()
        names = ("hawor_func", "hawor_func_vit", "dwpose_func")
        modules = {name: sys.modules.get(f"data_juicer.ops.common.{name}", missing) for name in names}
        attributes = {name: vars(common).get(name, missing) for name in names}
        dependencies = {name: sys.modules.get(name, missing) for name in ("smplx", "pytorch_lightning", "onnxruntime")}
        with patch.object(LazyLoader, "_install_package", side_effect=AssertionError("unexpected installation")):
            hw, hv, dw = load_test_modules.__wrapped__()
        for name in names:
            self.assertIs(sys.modules.get(f"data_juicer.ops.common.{name}", missing), modules[name])
            self.assertIs(vars(common).get(name, missing), attributes[name])
        for name, original in dependencies.items():
            self.assertIs(sys.modules.get(name, missing), original)
        for module in (hw, hv, dw):
            self.assertTrue(module.__name__.startswith("_dj_test_"))


if __name__ == "__main__":
    unittest.main()
