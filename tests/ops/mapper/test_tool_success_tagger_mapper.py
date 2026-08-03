import unittest

from data_juicer.ops.mapper.tool_success_tagger_mapper import (
    ToolSuccessTaggerMapper,
    _classify_tool_content,
    _content_to_str,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ToolSuccessTaggerMapperTest(DataJuicerTestCaseBase):

    def test_content_to_str_normalizes_runtime_payload_shapes(self):
        self.assertEqual(_content_to_str(None), "")
        self.assertEqual(_content_to_str(" ok "), "ok")
        self.assertEqual(_content_to_str({"a": 1}), '{"a": 1}')
        self.assertEqual(
            _content_to_str(
                [
                    {"type": "text", "text": " first "},
                    {"type": "text", "text": {"nested": "value"}},
                    " second ",
                    {"other": "field"},
                ]
            ),
            'first\n{"nested": "value"}\nsecond\n{"other": "field"}',
        )

    def test_classify_tool_content_prefers_error_over_success(self):
        op = ToolSuccessTaggerMapper()

        self.assertEqual(_classify_tool_content("", op._success_pats, op._error_pats), "unknown")
        self.assertEqual(_classify_tool_content("OK", op._success_pats, op._error_pats), "success")
        self.assertEqual(_classify_tool_content("Wrote 12 bytes", op._success_pats, op._error_pats), "success")
        self.assertEqual(
            _classify_tool_content("successfully wrote file but Error: denied", op._success_pats, op._error_pats),
            "error",
        )
        self.assertEqual(_classify_tool_content("plain output rows", op._success_pats, op._error_pats), "success")

    def test_process_single_counts_success_error_unknown_and_stores_results(self):
        op = ToolSuccessTaggerMapper()
        sample = {
            "messages": [
                {"role": "user", "content": "run"},
                {"role": "tool", "content": "OK"},
                {"role": "tool", "content": {"error": "not found"}},
                {"role": "tool", "content": ""},
                {"role": "tool_use", "content": [{"type": "text", "text": "data rows"}]},
            ]
        }

        out = op.process_single(sample)
        meta = out[Fields.meta]

        self.assertEqual(meta[MetaKeys.tool_success_count], 2)
        self.assertEqual(meta[MetaKeys.tool_fail_count], 1)
        self.assertEqual(meta[MetaKeys.tool_unknown_count], 1)
        self.assertEqual(meta[MetaKeys.tool_success_ratio], 2 / 3)
        self.assertEqual([r["result"] for r in meta[MetaKeys.tool_results]], ["success", "error", "unknown", "success"])
        self.assertEqual(meta[MetaKeys.tool_results][1]["content_preview"], '{"error": "not found"}')

    def test_process_single_can_disable_per_tool_results_and_custom_roles(self):
        op = ToolSuccessTaggerMapper(
            tool_role_names=["function"],
            success_patterns=[r"^done$"],
            error_patterns=[r"^bad$"],
            store_per_tool_results=False,
        )
        sample = {
            Fields.meta: {"kept": True},
            "messages": [
                {"role": "tool", "content": "bad"},
                {"role": "function", "content": "done"},
                {"role": "function", "content": "bad"},
            ],
        }

        out = op.process_single(sample)
        meta = out[Fields.meta]

        self.assertTrue(meta["kept"])
        self.assertEqual(meta[MetaKeys.tool_success_count], 1)
        self.assertEqual(meta[MetaKeys.tool_fail_count], 1)
        self.assertEqual(meta[MetaKeys.tool_unknown_count], 0)
        self.assertEqual(meta[MetaKeys.tool_success_ratio], 0.5)
        self.assertNotIn(MetaKeys.tool_results, meta)

    def test_process_single_non_list_messages_yields_zero_counts(self):
        out = ToolSuccessTaggerMapper().process_single({"messages": "bad"})
        meta = out[Fields.meta]

        self.assertEqual(meta[MetaKeys.tool_success_count], 0)
        self.assertEqual(meta[MetaKeys.tool_fail_count], 0)
        self.assertEqual(meta[MetaKeys.tool_unknown_count], 0)
        self.assertIsNone(meta[MetaKeys.tool_success_ratio])
        self.assertEqual(meta[MetaKeys.tool_results], [])


if __name__ == "__main__":
    unittest.main()
