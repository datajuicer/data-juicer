"""
Tests for the preflight check system.

These tests verify that configuration errors are caught early (at preflight)
rather than silently passing through to runtime.
"""

import unittest


class TestPreflightPreInstantiation(unittest.TestCase):
    """
    Tests for pre-instantiation checks (before load_ops).
    These test the DESIRED behavior after preflight is implemented.
    """

    def test_unknown_op_name_with_suggestion(self):
        """Should report the error with fuzzy-match suggestions."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check([{"remove_long_wordss_mapper": {"min_len": 1}}])

        error = ctx.exception
        # Should contain the bad op name and suggestions
        self.assertIn("remove_long_wordss_mapper", str(error))
        self.assertIn("remove_long_words_mapper", str(error))  # suggestion

    def test_completely_unknown_op_name(self):
        """Should report error even without close matches."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check([{"totally_fake_xyz_op": None}])
        self.assertIn("totally_fake_xyz_op", str(ctx.exception))

    def test_unknown_param_name_detected(self):
        """Should catch typos in parameter names."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check([{"remove_long_words_mapper": {"min_length": 1}}])

        error = ctx.exception
        self.assertIn("min_length", str(error))
        self.assertIn("min_len", str(error))  # suggestion

    def test_wrong_param_type_detected(self):
        """Should catch type mismatches based on annotations."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check([{"remove_long_words_mapper": {"min_len": "hello"}}])

        error = ctx.exception
        self.assertIn("min_len", str(error))
        self.assertIn("int", str(error))

    def test_numeric_type_coercion_accepted(self):
        """int annotation should accept int-compatible float (e.g., 10.0)."""
        from data_juicer.core.preflight import pre_instantiation_check

        # Should NOT raise - 10.0 is int-compatible (YAML may parse 10 as 10.0)
        pre_instantiation_check([{"remove_long_words_mapper": {"min_len": 10}}])
        pre_instantiation_check([{"remove_long_words_mapper": {"min_len": 10.0}}])

    def test_none_args_accepted(self):
        """OP with null config (all defaults) should pass."""
        from data_juicer.core.preflight import pre_instantiation_check

        # Should NOT raise
        pre_instantiation_check([{"remove_long_words_mapper": None}])

    def test_valid_config_passes(self):
        """A fully valid config should pass without errors."""
        from data_juicer.core.preflight import pre_instantiation_check

        pre_instantiation_check([
            {"remove_long_words_mapper": {"min_len": 1, "max_len": 100}},
            {"language_id_score_filter": {"lang": "en", "min_score": 0.8}},
        ])

    def test_multiple_errors_collected(self):
        """All errors in the pipeline should be reported at once."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check([
                {"remove_long_words_mapper": {"min_length": 1}},  # typo param
                {"fake_op_xyz": None},  # bad op name
            ])

        error = ctx.exception
        # Both errors should be reported
        self.assertTrue(len(error.errors) >= 2)

    def test_base_op_params_accepted(self):
        """Parameters defined in OP base class should be valid for any op."""
        from data_juicer.core.preflight import pre_instantiation_check

        # These are base class params, should be valid for any OP
        pre_instantiation_check([
            {"remove_long_words_mapper": {
                "min_len": 1,
                "text_key": "content",
                "batch_size": 500,
                "num_proc": 4,
            }},
        ])


class TestPreflightPostInstantiation(unittest.TestCase):
    """
    Tests for post-instantiation checks (after load_ops, before process).
    """

    def test_missing_custom_text_key_in_schema(self):
        """Should detect when a user-specified (non-default) text_key doesn't exist."""
        from data_juicer.core.data.schema import Schema
        from data_juicer.core.preflight import (
            PipelineRuntimeError,
            post_instantiation_check,
        )
        from data_juicer.ops.load import load_ops

        ops = load_ops([{"remove_long_words_mapper": {"text_key": "content"}}])
        schema = Schema(column_types={"text": str, "meta": dict}, columns=["text", "meta"])

        with self.assertRaises(PipelineRuntimeError) as ctx:
            post_instantiation_check(ops, schema)

        self.assertIn("content", str(ctx.exception))

    def test_default_text_key_missing_not_blocked(self):
        """Default text_key missing should NOT block — multimedia ops may not need it."""
        from data_juicer.core.data.schema import Schema
        from data_juicer.core.preflight import post_instantiation_check
        from data_juicer.ops.load import load_ops

        ops = load_ops([{"remove_long_words_mapper": {"min_len": 1}}])
        # Dataset without 'text' column — but since text_key is default, don't block
        schema = Schema(column_types={"images": str, "meta": dict}, columns=["images", "meta"])

        # Should NOT raise
        post_instantiation_check(ops, schema)

    def test_valid_schema_passes(self):
        """When text_key exists in schema, should pass."""
        from data_juicer.core.data.schema import Schema
        from data_juicer.core.preflight import post_instantiation_check
        from data_juicer.ops.load import load_ops

        ops = load_ops([{"remove_long_words_mapper": {"min_len": 1}}])
        schema = Schema(column_types={"text": str}, columns=["text"])

        # Should not raise
        post_instantiation_check(ops, schema)

    def test_unsupported_op_type_in_ray_mode(self):
        """Selector/Grouper/Aggregator should be rejected in Ray mode."""
        from types import SimpleNamespace

        from data_juicer.core.data.schema import Schema
        from data_juicer.core.preflight import (
            PipelineRuntimeError,
            post_instantiation_check,
        )
        from data_juicer.ops.load import load_ops

        ops = load_ops([{"topk_specified_field_selector": {
            "field_key": "stats.lang_score",
            "top_ratio": 0.5,
        }}])
        schema = Schema(column_types={"text": str}, columns=["text"])
        cfg = SimpleNamespace(executor_type="ray", export_path="/tmp/out")

        with self.assertRaises(PipelineRuntimeError) as ctx:
            post_instantiation_check(ops, schema, cfg)

        self.assertIn("does not support executor mode", str(ctx.exception))

    def test_mapper_in_ray_mode_passes(self):
        """Mapper should be fine in Ray mode."""
        from types import SimpleNamespace

        from data_juicer.core.data.schema import Schema
        from data_juicer.core.preflight import post_instantiation_check
        from data_juicer.ops.load import load_ops

        ops = load_ops([{"remove_long_words_mapper": {"min_len": 1}}])
        schema = Schema(column_types={"text": str}, columns=["text"])
        cfg = SimpleNamespace(executor_type="ray", export_path="/tmp/out")

        # Should not raise
        post_instantiation_check(ops, schema, cfg)


if __name__ == "__main__":
    unittest.main()
