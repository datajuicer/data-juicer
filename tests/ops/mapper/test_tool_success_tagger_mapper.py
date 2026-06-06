# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.ops.mapper.tool_success_tagger_mapper import ToolSuccessTaggerMapper
from data_juicer.utils.constant import Fields, MetaKeys


class TestToolSuccessTaggerMapper(unittest.TestCase):
    def test_default_label_unknown_for_ambiguous_tool_body(self):
        op = ToolSuccessTaggerMapper(default_label="unknown")
        sample = {
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "y"},
                {"role": "tool", "content": '{"status": "ok", "payload": 1}'},
            ],
            Fields.meta: {},
        }
        out = op.process_single(sample)
        meta = out[Fields.meta]
        self.assertEqual(meta[MetaKeys.tool_unknown_count], 1)
        self.assertEqual(meta[MetaKeys.tool_success_count], 0)
        self.assertEqual(meta[MetaKeys.tool_fail_count], 0)
        self.assertEqual(meta[MetaKeys.tool_success_ratio], -1.0)


if __name__ == "__main__":
    unittest.main()
