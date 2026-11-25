import unittest
from unittest import mock

from data_juicer.ops import (
    OP_PLUGIN_ENTRY_POINT_GROUP,
    load_op_plugins,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


def _make_ep(name, load_side_effect=None):
    """Build a fake importlib.metadata EntryPoint-like object."""
    ep = mock.Mock()
    ep.name = name
    if load_side_effect is not None:
        ep.load.side_effect = load_side_effect
    else:
        ep.load.return_value = mock.Mock()
    return ep


class LoadOpPluginsTest(DataJuicerTestCaseBase):
    """Test load_op_plugins: entry-point based external operator discovery."""

    def test_default_group_name(self):
        # the entry point group external plugin packages must declare
        self.assertEqual(OP_PLUGIN_ENTRY_POINT_GROUP, "data_juicer.ops")

    def test_discovers_and_loads_plugins(self):
        eps = [_make_ep("textclean"), _make_ep("llmgen")]
        with mock.patch("data_juicer.ops.entry_points", return_value=eps):
            loaded = load_op_plugins(group="data_juicer.ops")
        # both plugins loaded (ep.load() called -> triggers registration)
        self.assertEqual(sorted(loaded), ["llmgen", "textclean"])
        for ep in eps:
            ep.load.assert_called_once()

    def test_no_plugins_is_noop(self):
        with mock.patch("data_juicer.ops.entry_points", return_value=[]):
            loaded = load_op_plugins(group="data_juicer.ops")
        self.assertEqual(loaded, [])

    def test_broken_plugin_is_skipped_not_crash(self):
        # a broken plugin (load raises) must not crash the whole pipeline;
        # good plugins alongside it must still be loaded.
        good = _make_ep("good_plugin")
        bad = _make_ep("bad_plugin", load_side_effect=ImportError("boom"))
        with mock.patch("data_juicer.ops.entry_points", return_value=[bad, good]):
            loaded = load_op_plugins(group="data_juicer.ops")
        # bad one skipped, good one still loaded
        self.assertEqual(loaded, ["good_plugin"])
        bad.load.assert_called_once()
        good.load.assert_called_once()

    def test_legacy_entry_points_dict_api(self):
        # older importlib.metadata returns a dict keyed by group and does not
        # accept the group= kwarg; load_op_plugins must fall back gracefully.
        eps = [_make_ep("textclean")]

        def _dict_api(*args, **kwargs):
            if kwargs.get("group"):
                raise TypeError("entry_points() got an unexpected keyword")
            return {"data_juicer.ops": eps}

        with mock.patch("data_juicer.ops.entry_points", side_effect=_dict_api):
            loaded = load_op_plugins(group="data_juicer.ops")
        self.assertEqual(loaded, ["textclean"])

    def test_real_call_does_not_crash(self):
        # regression: calling against the real environment (no external
        # plugins installed in CI) must return a list and never raise.
        loaded = load_op_plugins()
        self.assertIsInstance(loaded, list)


if __name__ == "__main__":
    unittest.main()
